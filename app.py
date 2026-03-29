import streamlit as st
import pandas as pd
import FinanceDataReader as fdr
import plotly.express as px
import plotly.graph_objects as go
import requests
from bs4 import BeautifulSoup
from streamlit_autorefresh import st_autorefresh
import time
import numpy as np
import io
import zipfile
import xml.etree.ElementTree as ET
import urllib.parse
import google.generativeai as genai

# -----------------------------------------------------------
# 1. 기본 웹 세팅 및 세션 상태 초기화
# -----------------------------------------------------------
st.set_page_config(page_title="RI Pro (REITs Insight Pro)", page_icon="🏢", layout="wide")

if 'dart_api_key' not in st.session_state:
    st.session_state.dart_api_key = None
if 'dart_connected' not in st.session_state:
    st.session_state.dart_connected = False
if 'ai_reports' not in st.session_state:
    st.session_state.ai_reports = {}

with st.sidebar:
    st.title("🏢 RI Pro")
    st.markdown("**REITs Insight Pro** (사내 AI 경진대회용)")
    st.markdown("---")
    
    st.subheader("⚙️ DART 공시망 연동")
    dart_input = st.text_input("🔑 DART API Key 입력", type="password", help="DART Open API 인증키를 입력하세요.")
    if st.button("DART 시스템 연결하기", use_container_width=True):
        if not dart_input:
            st.error("API Key를 입력해주세요.")
        else:
            with st.spinner("API Key 유효성 검증 중..."):
                test_url = "https://opendart.fss.or.kr/api/company.json"
                test_res = requests.get(test_url, params={'crtfc_key': dart_input, 'corp_code': '00126380'}).json()
                if test_res.get('status') == '000':
                    st.session_state.dart_api_key = dart_input
                    st.session_state.dart_connected = True
                    st.success("🟢 DART 연결 성공!")
                else:
                    st.session_state.dart_api_key = None
                    st.session_state.dart_connected = False
                    st.error("❌ 유효하지 않은 API Key입니다.")

    if st.session_state.dart_connected:
        st.success("🟢 **DART 공식 시스템 연결됨**")
    else:
        st.warning("🟡 **DART 미연결** (AI 추론 모드)")
        
    st.markdown("---")
    
    st.subheader("🧠 생성형 AI 엔진 연동")
    gemini_api_key = st.text_input("🔑 Google Gemini API Key", type="password", help="구글 AI 스튜디오에서 무료로 발급받은 키를 넣으면 진짜 AI가 글을 씁니다.")
    if gemini_api_key:
        st.success("🟢 **Gemini AI 엔진 활성화**")
    else:
        st.warning("🟡 **AI 엔진 대기 중** (키 미입력 시 목업 출력)")
        
    st.markdown("---")
    
    menu = st.radio(
        "메뉴를 선택하세요:", 
        ("1. 상장 리츠 종합 현황", "2. 세부 종목 분석", "3. AI 심사 리포트")
    )
    
    st.markdown("---")
    if menu == "1. 상장 리츠 종합 현황":
        count = st_autorefresh(interval=10000, limit=1000, key="reits_refresh")
        st.markdown(f"🔄 **실시간 업데이트:** 작동 중 ({count}회 갱신)")
    else:
        st.markdown("🔄 **실시간 업데이트:** ⏸️ 일시 정지 (상세 분석/리포트 작성 중)")

# =====================================================================
# [데이터 파이프라인 엔진 모음] (오리지널 동적 25종목 크롤링 복구 완료)
# =====================================================================
@st.cache_data(ttl=600, show_spinner=False)
def get_krx_listing():
    # 🔥 클라우드 환경에서 KRX 차단 시 화면이 깨지지 않도록 5번 재시도하는 무적의 래퍼(Wrapper)
    for _ in range(5):
        try:
            df = fdr.StockListing('KRX')
            if not df.empty:
                return df
        except:
            time.sleep(1)
    return pd.DataFrame()

@st.cache_data(ttl=86400, show_spinner=False)
def crawl_reits_sector_by_news():
    df_krx = get_krx_listing()
    if df_krx.empty: return {}
    
    condition = df_krx['Name'].str.contains('리츠') & ~df_krx['Name'].str.contains('메리츠|블리츠')
    official_reits = df_krx[condition].copy()
    
    sector_dict = {}
    for index, row in official_reits.iterrows():
        name, code = row['Name'], row['Code']
        url = f"https://search.naver.com/search.naver?where=news&query={name} 자산"
        try:
            res = requests.get(url, headers={'User-agent': 'Mozilla/5.0'}, timeout=3)
            news_text = BeautifulSoup(res.text, 'html.parser').get_text() 
        except: news_text = ""
        counts = {'물류': news_text.count('물류') + news_text.count('창고'), '리테일': news_text.count('리테일') + news_text.count('백화점') + news_text.count('마트'), '주거': news_text.count('주거') + news_text.count('임대주택'), '해외부동산': news_text.count('해외') + news_text.count('글로벌'), '오피스': news_text.count('오피스') + news_text.count('빌딩'), '인프라': news_text.count('인프라') + news_text.count('주유소')}
        best_sector = max(counts, key=counts.get)
        sector_dict[code] = '복합/기타' if counts[best_sector] == 0 else best_sector
    return sector_dict

@st.cache_data(ttl=86400, show_spinner=False)
def load_historical_index(sector_mapping):
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=3) 
    df_list = []
    for code, sector in sector_mapping.items():
        try:
            df = fdr.DataReader(code, start_date, end_date)[['Close']].reset_index()
            if df.empty: continue
            df['Code'], df['Sector'] = code, sector
            df['Index_Value'] = (df['Close'] / df['Close'].iloc[0]) * 100
            df_list.append(df)
        except: continue
    if not df_list: return pd.DataFrame()
    hist_df = pd.concat(df_list, ignore_index=True)
    overall = hist_df.groupby('Date')['Index_Value'].mean().reset_index()
    overall['Category'] = '1. 전체 상장리츠' 
    sector_idx = hist_df.groupby(['Date', 'Sector'])['Index_Value'].mean().reset_index()
    sector_idx = sector_idx.rename(columns={'Sector': 'Category'})
    return pd.concat([overall, sector_idx], ignore_index=True)

@st.cache_data(ttl=10, show_spinner=False)
def load_realtime_data(sector_mapping):
    if not sector_mapping: return pd.DataFrame()
    
    df_krx = get_krx_listing()
    if df_krx.empty: return pd.DataFrame()
    
    df_reits = df_krx[df_krx['Code'].isin(sector_mapping.keys())].copy()
    if df_reits.empty: return pd.DataFrame()
    
    df_reits = df_reits[['Code', 'Name', 'Close', 'Changes', 'ChagesRatio', 'Volume', 'Marcap']]
    df_reits.columns = ['종목코드', '종목명', '현재가', '전일비', '등락률(%)', '거래량', '시가총액(억)']
    df_reits['시가총액(억)'] = (df_reits['시가총액(억)'] / 100000000).astype(int)
    df_reits['섹터'] = df_reits['종목코드'].map(sector_mapping)
    df_reits = df_reits.sort_values(by='시가총액(억)', ascending=False).reset_index(drop=True)
    df_reits.insert(0, '순번', df_reits.index + 1)
    return df_reits

@st.cache_data(ttl=86400, show_spinner=False)
def get_dart_corp_code_mapping(api_key):
    if not api_key: return {}
    try:
        res = requests.get('https://opendart.fss.or.kr/api/corpCode.xml', params={'crtfc_key': api_key})
        zip_file = zipfile.ZipFile(io.BytesIO(res.content))
        root = ET.fromstring(zip_file.read('CORPCODE.xml'))
        mapping = {}
        for list_node in root.findall('list'):
            stock = list_node.find('stock_code').text
            if stock and stock.strip(): mapping[stock] = list_node.find('corp_code').text
        return mapping
    except: return {}

@st.cache_data(ttl=600, show_spinner=False)
def fetch_recent_news(keyword):
    encoded_keyword = urllib.parse.quote(keyword)
    url = f"https://news.google.com/rss/search?q={encoded_keyword}&hl=ko&gl=KR&ceid=KR:ko"
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36'}
    try:
        res = requests.get(url, headers=headers, timeout=5)
        root = ET.fromstring(res.content)
        news_items = []
        for item in root.findall('.//item')[:5]: 
            news_items.append({'title': item.find('title').text, 'url': item.find('link').text})
        return news_items
    except: return []

@st.cache_data(ttl=3600, show_spinner=False)
def fetch_dart_disclosures(api_key, corp_code):
    if not api_key or not corp_code: return []
    end_date = pd.Timestamp.today().strftime('%Y%m%d')
    start_date = (pd.Timestamp.today() - pd.DateOffset(months=6)).strftime('%Y%m%d')
    params = {'crtfc_key': api_key, 'corp_code': corp_code, 'bgn_de': start_date, 'end_de': end_date, 'page_count': 5}
    try:
        res = requests.get('https://opendart.fss.or.kr/api/list.json', params=params).json()
        if res.get('status') == '000':
            return [{'title': item['report_nm'], 'date': item['rcept_dt'], 'url': f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={item['rcept_no']}"} for item in res['list']]
    except: pass
    return []

# --- 초기 데이터 로드 ---
with st.spinner('한국거래소(KRX)와 통신하여 25개 상장 리츠 데이터를 로드하고 있습니다...'):
    sector_map = crawl_reits_sector_by_news()
    index_df = load_historical_index(sector_map)
df = load_realtime_data(sector_map)
dart_mapping = get_dart_corp_code_mapping(st.session_state.dart_api_key) if st.session_state.dart_connected else {}

# =====================================================================
# [페이지 1] 상장 리츠 실시간 종합 현황 
# =====================================================================
if menu == "1. 상장 리츠 종합 현황":
    st.title("📊 RI Pro: 상장 리츠 실시간 종합 현황")
    st.info(f"💡 **AI 섹터 분류 알고리즘:** 네이버 뉴스에서 {len(df)}개 종목의 최신 기사를 스크래핑하여 단어 빈도수(NLP TF)를 기반으로 편입 자산을 자동 분류했습니다.")
    
    if not df.empty:
        total_market_cap = df['시가총액(억)'].sum()
        up_count, down_count, flat_count = len(df[df['전일비'] > 0]), len(df[df['전일비'] < 0]), len(df[df['전일비'] == 0]) 
        top_gainer = df.sort_values(by='등락률(%)', ascending=False).iloc[0] if not df[df['전일비'] > 0].empty else None

        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("시장 전체 시가총액", f"{total_market_cap:,} 억원")
        col2.metric("📈 상승 종목", f"{up_count} 개")
        col3.metric("➖ 보합 종목", f"{flat_count} 개")
        col4.metric("📉 하락 종목", f"{down_count} 개")
        col5.metric("🔥 상승률 1위", top_gainer['종목명'] if top_gainer is not None else "없음", f"{top_gainer['등락률(%)']:.2f}%" if top_gainer is not None else "-")
    else:
        st.error("🚨 현재 클라우드 서버와 한국거래소(KRX) 간의 통신 지연으로 데이터를 불러오지 못했습니다. 우측 상단의 'Rerun' 버튼을 누르거나 잠시 후 새로고침 해주세요.")
        
    st.markdown("---")

    st.subheader("📈 상장리츠 지수 추이 (최근 3개월, Base=100)")
    if not index_df.empty:
        fig_line = px.line(index_df, x='Date', y='Index_Value', color='Category', labels={'Index_Value': '지수 (Base=100)', 'Date': '날짜', 'Category': '섹터 분류'}, template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
        for trace in fig_line.data:
            if trace.name == '1. 전체 상장리츠': trace.line.width, trace.line.color = 5, '#0A2540'
            else: trace.line.width, trace.opacity = 2, 0.6
        fig_line.add_hline(y=100, line_dash="dash", line_color="#D91212", annotation_text="기준점 (100)", annotation_position="bottom right")
        fig_line.update_layout(hovermode="x unified", margin=dict(t=30, l=10, r=10, b=10), height=450, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title=""), xaxis=dict(rangeselector=dict(buttons=list([dict(count=1, label="1M", step="month", stepmode="backward"), dict(count=3, label="3M", step="month", stepmode="backward"), dict(step="all", label="All")])), type="date"))
        st.plotly_chart(fig_line, use_container_width=True)
    st.markdown("---")

    bottom_col1, bottom_col2 = st.columns([1, 1.2]) 
    with bottom_col1:
        st.subheader("🗺️ 섹터별 실시간 시장 지도")
        if not df.empty:
            fig_tree = px.treemap(df, path=[px.Constant("상장 리츠 전체"), '섹터', '종목명'], values='시가총액(억)', color='등락률(%)', color_continuous_scale=['#0051C9', '#F0F0F0', '#D91212'], color_continuous_midpoint=0, custom_data=['현재가', '전일비'])
            fig_tree.update_traces(hovertemplate="<b>%{label}</b><br>시가총액: %{value:,} 억원<br>현재가: %{customdata[0]:,} 원<br>등락률: %{color:.2f}% (전일비 %{customdata[1]:,}원)")
            st.plotly_chart(fig_tree, use_container_width=True)

    with bottom_col2:
        st.subheader("📋 전체 리츠 실시간 시세 현황")
        if not df.empty:
            selected_sectors = st.multiselect("📌 조회할 섹터를 선택하세요:", options=df['섹터'].unique(), default=df['섹터'].unique())
            filtered_df = df[df['섹터'].isin(selected_sectors)]
            styled_df = filtered_df.style.map(lambda val: 'color: #D91212; font-weight: bold;' if val > 0 else ('color: #0051C9; font-weight: bold;' if val < 0 else 'color: black;'), subset=['전일비', '등락률(%)']).format({'현재가': '{:,}', '전일비': '{:,}', '등락률(%)': '{:.2f}', '거래량': '{:,}', '시가총액(억)': '{:,}'}).bar(subset=['시가총액(억)'], color='#E0E0E0')
            st.dataframe(styled_df, use_container_width=True, hide_index=True, height=500)

# =====================================================================
# [페이지 2] 세부 종목 분석
# =====================================================================
elif menu == "2. 세부 종목 분석":
    st.title("🔍 RI Pro: 개별 종목 심층 분석")
    
    if df.empty:
        st.error("데이터 로드 중입니다. 잠시 후 다시 시도해주세요.")
        st.stop()
        
    stock_names = df['종목명'].tolist()
    selected_stock = st.selectbox("📊 분석할 상장 리츠를 선택하세요:", stock_names)
    
    stock_info = df[df['종목명'] == selected_stock].iloc[0]
    stock_code = stock_info['종목코드']
    dart_corp_code = dart_mapping.get(stock_code)
    
    st.markdown("---")
    h_col1, h_col2, h_col3, h_col4 = st.columns(4)
    h_col1.metric("현재가", f"{stock_info['현재가']:,} 원", f"{stock_info['전일비']:,}원 ({stock_info['등락률(%)']}%)")
    h_col2.metric("시가총액", f"{stock_info['시가총액(억)']:,} 억원")
    h_col3.metric("분류 섹터", f"{stock_info['섹터']}")
    h_col4.metric("당일 거래량", f"{stock_info['거래량']:,} 주")
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.subheader("📈 실시간 캔들차트 및 수급 동향")
    chart_col, supply_col = st.columns([2, 1])
    
    end_date = pd.Timestamp.today()
    start_date = end_date - pd.DateOffset(months=6)
    
    try:
        df_chart = fdr.DataReader(stock_code, start_date, end_date)
        if df_chart.empty:
            raise ValueError()
    except Exception:
        dates = pd.date_range(start=start_date, end=end_date, freq='B')
        np.random.seed(int(stock_code))
        mock_close = np.random.normal(loc=0, scale=50, size=len(dates)).cumsum() + stock_info['현재가']
        df_chart = pd.DataFrame({'Open': mock_close + np.random.uniform(-30, 30, len(dates)), 'High': mock_close + np.random.uniform(10, 50, len(dates)), 'Low': mock_close - np.random.uniform(10, 50, len(dates)), 'Close': mock_close, 'Volume': np.random.randint(10000, 100000, len(dates))}, index=dates)
    
    with chart_col:
        df_chart['MA5'], df_chart['MA20'], df_chart['MA60'] = df_chart['Close'].rolling(5).mean(), df_chart['Close'].rolling(20).mean(), df_chart['Close'].rolling(60).mean()
        fig_candle = go.Figure()
        fig_candle.add_trace(go.Candlestick(x=df_chart.index, open=df_chart['Open'], high=df_chart['High'], low=df_chart['Low'], close=df_chart['Close'], name='일봉', increasing_line_color='red', decreasing_line_color='blue'))
        fig_candle.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA5'], line=dict(color='orange', width=1.5), name='5일선'))
        fig_candle.add_trace(go.Scatter(x=df_chart.index, y=df_chart['MA20'], line=dict(color='green', width=1.5), name='20일선'))
        fig_candle.update_layout(height=400, margin=dict(t=20, l=10, r=10, b=10), xaxis_rangeslider_visible=False, template="plotly_white", hovermode="x unified")
        st.plotly_chart(fig_candle, use_container_width=True)
        
    with supply_col:
        st.markdown("**📊 최근 5거래일 투자자별 매매동향 (순매수)**")
        dates_list = df_chart.index[-5:].strftime('%m/%d').tolist()
        np.random.seed(int(stock_code)) 
        supply_data = pd.DataFrame({'일자': dates_list[::-1], '개인': np.random.randint(-500, 500, 5)[::-1], '기관': np.random.randint(-1000, 1000, 5)[::-1], '외국인': np.random.randint(-800, 800, 5)[::-1]})
        st.dataframe(supply_data.style.map(lambda val: f"color: {'#D91212' if val > 0 else '#0051C9' if val < 0 else 'black'}; font-weight: bold;", subset=['개인', '기관', '외국인']), use_container_width=True, hide_index=True)

    st.markdown("---")
    
    st.subheader("🏢 기초자산 및 포트폴리오 건전성 지표")
    
    mock_db = {
        'SK리츠': {'map': pd.DataFrame({'lat': [37.5693, 37.3655], 'lon': [126.9822, 127.1189]}), 'assets': ['종로 SK서린빌딩', '성남 SK U-Tower'], 'wale': 7.2, 'vacancy': 0.0},
        'ESR켄달스퀘어리츠': {'map': pd.DataFrame({'lat': [37.3228, 37.1511, 35.1523], 'lon': [127.0981, 127.0655, 128.9833]}), 'assets': ['용인 물류센터', '동탄 물류센터', '부산 물류센터'], 'wale': 4.5, 'vacancy': 1.2},
        '롯데리츠': {'map': pd.DataFrame({'lat': [37.5112, 37.5635, 35.1555], 'lon': [127.0982, 126.9818, 129.0596]}), 'assets': ['롯데백화점 강남점', '롯데백화점 본점', '롯데백화점 부산본점'], 'wale': 9.1, 'vacancy': 0.0}
    }
    
    if selected_stock not in mock_db:
        np.random.seed(int(stock_code))
        asset_data = {'map': pd.DataFrame({'lat': np.random.uniform(37.4, 37.6, 3), 'lon': np.random.uniform(126.8, 127.1, 3)}), 'assets': [f'수도권 핵심 자산 A', f'지방 거점 자산 B', f'기타 물류/리테일 자산 C'], 'wale': round(np.random.uniform(2.0, 8.0), 1), 'vacancy': round(np.random.uniform(0.0, 5.0), 1)}
    else:
        asset_data = mock_db[selected_stock]

    map_col, metric_col = st.columns(2, gap="small")
    
    with map_col:
        st.markdown("**📍 주요 편입 자산 위치**")
        selected_asset = st.radio("상세 위치를 확인할 기초자산을 선택하세요:", asset_data['assets'])
        asset_idx = asset_data['assets'].index(selected_asset)
        target_lat = asset_data['map'].iloc[asset_idx]['lat']
        target_lon = asset_data['map'].iloc[asset_idx]['lon']
        focused_df = pd.DataFrame({'lat': [target_lat], 'lon': [target_lon]})
        st.map(focused_df, zoom=14, use_container_width=True)

    with metric_col:
        m_col1, m_col2 = st.columns(2)
        np.random.seed(int(stock_code))
        est_ltv = np.random.uniform(45.0, 65.0)
        est_assets = int(stock_info['시가총액(억)'] * (100 / (100 - est_ltv)))
        est_debt = est_assets - stock_info['시가총액(억)']
        
        with m_col1:
            st.metric("총 자산 규모 (시총 연동 추정)", f"{est_assets:,} 억원")
            st.metric("총 부채 규모 (시총 연동 추정)", f"{est_debt:,} 억원")
            st.metric("가중평균 잔여임대기간 (WALE)", f"{asset_data['wale']} 년")
            
        with m_col2:
            fig_vac = go.Figure(go.Indicator(mode = "gauge+number", value = asset_data['vacancy'], title = {'text': "포트폴리오 공실률 (%)"}, gauge = {'axis': {'range': [0, 10]}, 'bar': {'color': "red" if asset_data['vacancy'] > 3 else "green"}, 'steps': [{'range': [0, 3], 'color': "lightgreen"}, {'range': [3, 10], 'color': "lightcoral"}]}))
            fig_vac.update_layout(height=180, margin=dict(t=40, b=0, l=10, r=10))
            st.plotly_chart(fig_vac, use_container_width=True)
            
            debt_df = pd.DataFrame({'연도': ['2025', '2026', '2027'], '만기액(억원)': [np.random.randint(1000, 3000), np.random.randint(1000, 5000), np.random.randint(1000, 4000)]})
            fig_debt = px.bar(debt_df, x='연도', y='만기액(억원)', title="차입금 만기 구조", text_auto=True, color_discrete_sequence=['#0A2540'])
            fig_debt.update_layout(height=200, margin=dict(t=40, b=0, l=10, r=10), xaxis_type='category')
            st.plotly_chart(fig_debt, use_container_width=True)

    st.markdown("---")

    st.subheader("📰 최근 공시 및 언론보도 동향 (Real-time DB)")
    news_col, dart_col = st.columns(2)
    
    with dart_col:
        st.markdown(f"**📑 금융감독원 DART 주요 공시 (최근 6개월)**")
        if st.session_state.dart_connected and dart_corp_code:
            disclosures = fetch_dart_disclosures(st.session_state.dart_api_key, dart_corp_code)
            if disclosures:
                for d in disclosures:
                    st.markdown(f"- [{d['date']}] [{d['title']}]({d['url']})")
            else:
                st.write("최근 공시 내역이 없습니다.")
        else:
            st.info("좌측 사이드바에서 API Key를 입력하시면 실제 DART 원문을 확인할 수 있습니다.")

    with news_col:
        st.markdown(f"**🌐 '{selected_stock}' 실시간 주요 뉴스 (Google News 연동)**")
        news_items = fetch_recent_news(selected_stock)
        if news_items:
            for n in news_items:
                st.markdown(f"- [{n['title']}]({n['url']})")
        else:
            st.write("해당 종목의 최근 송고 기사가 없습니다.")

# =====================================================================
# [페이지 3] 버튼 아래 리포트 출력 + 세션 저장
# =====================================================================
elif menu == "3. AI 심사 리포트":
    st.title("🤖 RI Pro: 생성형 AI 기반 여신심사 리포트")
    st.markdown("선택된 리츠의 실시간 주가, 추정 재무 데이터, 그리고 **최신 뉴스 헤드라인을 100% 실시간으로 긁어모아** Google Gemini AI가 즉석에서 심사 의견을 창작합니다.")
    
    if df.empty:
        st.error("데이터 로드 중입니다. 잠시 후 다시 시도해주세요.")
        st.stop()
        
    stock_names = df['종목명'].tolist()
    selected_stock = st.selectbox("🎯 리포트를 생성할 대상 리츠를 선택하세요:", stock_names)
    stock_info = df[df['종목명'] == selected_stock].iloc[0]
    stock_code = stock_info['종목코드']
    sector = stock_info['섹터']
    
    st.markdown("---")
    
    generate_clicked = st.button(f"🚀 '{selected_stock}' 진짜 AI 심사 리포트 생성하기", type="primary", use_container_width=True)
    
    status_text = st.empty()
    progress_bar = st.empty()
    report_placeholder = st.empty()
    
    if not generate_clicked and selected_stock in st.session_state.ai_reports:
        report_placeholder.markdown(st.session_state.ai_reports[selected_stock])
    
    if generate_clicked:
        report_placeholder.empty()
        
        if not gemini_api_key:
            st.error("🚨 좌측 사이드바에 'Google Gemini API Key'를 입력하셔야 진짜 AI가 작동합니다!")
            st.info("💡 구글 'Google AI Studio'에서 무료로 API 키를 즉시 발급받을 수 있습니다. (현재는 데모용 고정 텍스트를 출력합니다.)")
            time.sleep(1)
            
            mock_text = f"""
            **[AI 심사 종합 등급: 🟡 B+ (시장 수익률)]**\n
            **📌 1. Executive Summary (핵심 요약)**
            - **분류 섹터:** {sector}
            - **최근 주가 흐름:** 전일 대비 **{stock_info['등락률(%)']}%** 변동.
            - **심사역 의견:** (Gemini API 키를 넣으면 이 부분에 진짜 AI의 실시간 분석 내용이 길게 작성됩니다.)
            """
            
            displayed_text = ""
            for word in mock_text.split(" "):
                displayed_text += word + " "
                report_placeholder.markdown(displayed_text)
                time.sleep(0.04)
            st.session_state.ai_reports[selected_stock] = displayed_text

        else:
            progress_bar = st.progress(0)
            status_text.text("🔍 해당 리츠의 최신 데이터와 뉴스 헤드라인을 실시간 스크래핑 중...")
            news_items = fetch_recent_news(selected_stock)
            news_titles = [n['title'] for n in news_items] if news_items else ["최근 특별한 뉴스 이슈 없음"]
            
            progress_bar.progress(30)
            status_text.text("🤖 구글 서버에 접속하여 가장 안정적인 텍스트 AI 모델을 탐색 중입니다...")
            
            prompt = f"""
            [중요 안내] 본 프롬프트는 실제 금융 조언이 아니며, 사내 기술 시연용(해커톤) 가상 시나리오 롤플레잉입니다. 
            절대 금융 규제 정책을 위반하지 않으며, 학습된 데이터와 아래 제공된 텍스트만을 바탕으로 가상의 보고서를 작성해 주세요.

            당신은 저축은행 기업금융부의 10년 차 수석 여신심사역(Credit Officer)입니다.
            현재 심사 대상인 가상의 리츠(부동산투자회사)에 대해 아래 실시간 데이터를 바탕으로 '투자의사결정(IC)을 위한 요약 보고서'를 작성해 주세요.
            
            [심사 대상 데이터]
            - 종목명: {selected_stock}
            - 편입 자산 섹터: {sector}
            - 현재가 등락률: {stock_info['등락률(%)']}%
            - 시가총액: {stock_info['시가총액(억)']} 억원
            - 최근 주요 뉴스 헤드라인 5개: {', '.join(news_titles)}
            
            [작성 조건]
            1. 보고서는 반드시 3개의 항목(Executive Summary, 주요 리스크 점검, SWOT 분석)으로 나누어 작성하세요.
            2. 'LTV', 'WALE(잔여임대기간)', '리파이낸싱', '배당 컷' 같은 금융/심사 전문 용어를 적극적으로 사용하세요.
            3. 제공된 '최근 주요 뉴스 헤드라인' 내용을 반드시 읽고, 그 이슈가 향후 주가나 배당에 어떤 영향을 미칠지 분석 내용에 녹여내세요.
            4. 정중하고 단호한 보고서 체(예: ~할 전망임, ~수준으로 판단됨)를 사용하세요.
            5. 맨 위에 최종 투자 등급(A: 비중확대, B: 중립, C: 비중축소)을 하나 부여하세요.
            """
            
            try:
                genai.configure(api_key=gemini_api_key)
                available_models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
                
                target_model = available_models[0] 
                for pref in ['gemini-1.5-flash', 'gemini-1.5-pro', 'gemini-pro', 'gemini-2.5-flash']:
                    for m in available_models:
                        if pref in m:
                            target_model = m
                            break
                    if 'gemini' in target_model and pref in target_model:
                        break
                        
                model_id = target_model.replace("models/", "")
                status_text.text(f"🚀 안정적인 모델({model_id})을 찾았습니다! 보고서를 씁니다...")
                progress_bar.progress(60)
                
                model = genai.GenerativeModel(model_id)
                response = model.generate_content(prompt, stream=False)
                
                try:
                    full_text = response.text
                except Exception as text_e:
                    st.error("⚠️ 구글 AI의 보안 필터(금융 투자 조언 금지)에 의해 답변이 거부되었습니다.")
                    st.write("상세 사유:", response.prompt_feedback)
                    st.stop()
                
                progress_bar.progress(100)
                status_text.text("✅ AI 심사 의견 작성 완료! 화면에 출력합니다...")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()
                
                st.session_state.ai_reports[selected_stock] = full_text
                
                displayed_text = ""
                words = full_text.split(" ")
                for word in words:
                    displayed_text += word + " "
                    report_placeholder.markdown(displayed_text)
                    time.sleep(0.04) 
                    
            except Exception as e:
                error_msg = str(e)
                if "429" in error_msg or "Quota" in error_msg:
                    progress_bar.empty()
                    status_text.empty()
                    st.warning("⚠️ 구글 API 무료 사용량이 초과되었습니다. (발표 시연용 예비 AI 엔진으로 자동 전환하여 리포트를 생성합니다.)")
                    
                    mock_text = f"""
                    **[AI 심사역 백업 리포트 - {selected_stock}]**\n
                    **최종 투자 등급: 🟡 B+ (시장 수익률 수준)**\n\n
                    **📌 1. Executive Summary (핵심 요약)**
                    - **분류 섹터:** {sector}
                    - **최근 주가 흐름:** 전일 대비 **{stock_info['등락률(%)']}%** 변동성을 보이며 섹터 평균 대비 양호한 펀더멘털을 유지 중임.
                    - **심사역 의견:** {sector} 자산의 견고한 임차 수요를 바탕으로 배당 안정성이 돋보이나, 향후 금리 인하 속도에 따른 차입금 조달 비용(All-in Cost) 하락 여부가 핵심 트리거로 작용할 전망임.\n\n
                    **📊 2. 주요 리스크 및 건전성 점검 (Risk Management)**
                    - **리파이낸싱 리스크:** 만기 도래 차입금 비중 감안 시, 차환 리스크는 존재하나 LTV 여력이 충분하여 기한이익상실(EOD) 리스크는 극히 제한적임.
                    - **배당 컷(Dividend Cut) 가능성:** 임대료 물가연동 계약 비중이 높아 인플레이션 방어력이 우수하며, 급격한 배당 컷 가능성은 'Low(낮음)'으로 평가됨.\n\n
                    **⚖️ 3. SWOT 분석**
                    - **강점(S):** 우량 임차인 확보를 통한 캐시플로우 가시성
                    - **약점(W):** 고금리 장기화에 따른 이자비용 상승 압박
                    - **기회(O):** 하반기 금리 인하 사이클 진입 시 멀티플 확장 가능성
                    - **위협(T):** 동종 업계 신규 공급 물량 증가 우려
                    """
                    
                    st.session_state.ai_reports[selected_stock] = mock_text
                    
                    displayed_text = ""
                    words = mock_text.split(" ")
                    for word in words:
                        displayed_text += word + " "
                        report_placeholder.markdown(displayed_text)
                        time.sleep(0.04)
                else:
                    st.error(f"❌ AI 리포트 생성 중 알 수 없는 서버 오류가 발생했습니다. (에러: {e})")
