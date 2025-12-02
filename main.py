import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re

# Matplotlib에서 한글 폰트 설정 (Mac/Linux/Windows에 따라 적절히 선택)
# 시스템에 나눔고딕이 설치되어 있지 않다면, 주석을 해제하고 다른 폰트를 사용하거나 설치해야 합니다.
plt.rcParams['font.family'] = 'Malgun Gothic' # Windows 기본 폰트
# plt.rcParams['font.family'] = 'AppleGothic' # Mac 기본 폰트
plt.rcParams['axes.unicode_minus'] = False # 마이너스 기호 깨짐 방지

# -----------------------------------------------------
# 1. 데이터 로드 및 전처리
# -----------------------------------------------------

# 파일 이름은 업로드된 파일 이름을 사용합니다.
FILE_PATH = "bicycle accidents_20201231.csv"

@st.cache_data
def load_and_preprocess_data(file_path):
    """CSV 파일을 로드하고 분석에 필요한 전처리를 수행합니다."""
    try:
        # 파일 인코딩 문제 해결을 위해 'cp949' 또는 'euc-kr' 시도
        df = pd.read_csv(file_path, encoding='cp949') 
    except UnicodeDecodeError:
        try:
            df = pd.read_csv(file_path, encoding='euc-kr')
        except:
             df = pd.read_csv(file_path, encoding='utf-8')
    
    # 1. 중상 사고 필터링
    df_severe = df[df['피해자신체상해정도'] == '중상'].copy()
    
    if df_severe.empty:
        st.error("⚠️ 데이터에서 '피해자신체상해정도'가 '중상'인 사고가 발견되지 않았습니다. 데이터를 확인해 주세요.")
        return pd.DataFrame()

    # 2. 연도 추출 및 데이터 타입 변환
    df_severe['발생일'] = pd.to_datetime(df_severe['발생일'], errors='coerce')
    df_severe.dropna(subset=['발생일'], inplace=True)
    df_severe['연도'] = df_severe['발생일'].dt.year.astype('Int64')
    
    # 3. 발생시간대 정제 (예: '07시' -> 7)
    df_severe['발생시간'] = df_severe['발생시간대'].str.replace('시', '').str.strip().astype('Int64', errors='ignore')
    
    # 4. 연령대 분류 (고등학생도 쉽게 이해할 수 있는 10세 단위)
    def categorize_age(age):
        if pd.isna(age):
            return '미상'
        age = int(re.sub(r'세', '', str(age).strip()))
        if age < 10: return '0~9세'
        elif age < 20: return '10대'
        elif age < 30: return '20대'
        elif age < 40: return '30대'
        elif age < 50: return '40대'
        elif age < 60: return '50대'
        elif age < 70: return '60대'
        else: return '70대 이상'
        
    df_severe['피해자_연령대'] = df_severe['피해자연령'].apply(categorize_age)
    
    # 분석에 필요한 열만 선택
    df_severe = df_severe[['연도', '발생시간', '피해자_연령대', '사고유형', '법규위반사항']].copy()

    return df_severe

# 데이터 로드
df = load_and_preprocess_data(FILE_PATH)

if not df.empty:
    
    # -----------------------------------------------------
    # 2. Streamlit UI 및 시각화 함수
    # -----------------------------------------------------

    st.set_page_config(layout="wide", page_title="자전거 중상 사고 분석")
    st.title("🚲 자전거 사고 (중상 피해자) 심층 분석")
    st.markdown("---")
    st.subheader("🔎 분석 대상: 피해자 신체상해정도가 **'중상'**인 사고")
    
    # 사이드바 메뉴
    st.sidebar.header("📊 시각화 메뉴")
    chart_type = st.sidebar.radio(
        "보고 싶은 그래프 유형을 선택하세요:",
        ('막대 그래프', '선 그래프', '원 그래프', '히스토그램', '히트맵', '산점도')
    )
    
    # 연도 선택 필터 (사이드바)
    available_years = sorted(df['연도'].unique())
    selected_year = st.sidebar.selectbox(
        "분석할 연도를 선택하세요:",
        ['전체 연도'] + available_years
    )
    
    if selected_year != '전체 연도':
        df_filtered = df[df['연도'] == selected_year]
    else:
        df_filtered = df.copy()

    
    def plot_bar_chart(df):
        """막대 그래프: 연령대별 중상 사고 분포 시각화"""
        st.subheader("📈 피해자 연령대별 중상 사고 분포 (막대 그래프)")
        
        # '10대' 연령대를 강조
        age_order = ['10대', '20대', '30대', '40대', '50대', '60대', '70대 이상', '0~9세', '미상']
        df_count = df['피해자_연령대'].value_counts().reindex(age_order).fillna(0).astype(int).reset_index()
        df_count.columns = ['연령대', '사고 건수']
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x='연령대', y='사고 건수', data=df_count, ax=ax, palette="RdYlBu")
        
        # 10대 막대에 강조 색상 적용
        if '10대' in df_count['연령대'].values:
            teen_index = df_count[df_count['연령대'] == '10대'].index[0]
            ax.patches[teen_index].set_color('red')

        ax.set_title("피해자 연령대별 중상 사고 건수", fontsize=16)
        ax.set_xlabel("피해자 연령대")
        ax.set_ylabel("사고 건수")
        st.pyplot(fig)

    def plot_pie_chart(df):
        """원 그래프: 사고 유형별 분포 시각화"""
        st.subheader("🍕 사고 유형별 중상 사고 분포 (원 그래프)")
        
        df_count = df['사고유형'].value_counts().nlargest(7) # 상위 7개 유형만 표시
        
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # 기타 항목으로 묶기
        other_sum = df['사고유형'].value_counts().sum() - df_count.sum()
        if other_sum > 0:
            df_pie = pd.concat([df_count, pd.Series([other_sum], index=['기타'])])
        else:
            df_pie = df_count
            
        wedges, texts, autotexts = ax.pie(
            df_pie.values, 
            labels=df_pie.index, 
            autopct='%1.1f%%', 
            startangle=90, 
            textprops={'fontsize': 10}
        )
        ax.set_title("주요 사고 유형별 비율", fontsize=16)
        st.pyplot(fig)
        
    def plot_line_chart(df):
        """선 그래프: 발생 시간대별 사고 건수 추이 시각화"""
        st.subheader("📉 발생 시간대별 중상 사고 건수 추이 (선 그래프)")
        
        # 시간대별 집계
        df_time = df['발생시간'].value_counts().sort_index().reset_index()
        df_time.columns = ['시간', '사고 건수']
        
        fig, ax = plt.subplots(figsize=(12, 6))
        sns.lineplot(x='시간', y='사고 건수', data=df_time, marker='o', ax=ax, color='darkorange')
        
        ax.set_xticks(range(0, 24, 2)) # 2시간 단위로 표시
        ax.set_title("시간대별 중상 사고 발생 분포", fontsize=16)
        ax.set_xlabel("발생 시간 (시)")
        ax.set_ylabel("사고 건수")
        ax.grid(True, alpha=0.5)
        st.pyplot(fig)
        
    def plot_histogram(df):
        """히스토그램: 법규 위반 사항 분포 시각화"""
        st.subheader("📊 법규 위반 사항 분포 (히스토그램)")
        st.caption("주요 법규 위반 사항의 건수 분포를 보여줍니다.")
        
        # 상위 10개 법규 위반 사항만 추출
        top_violations = df['법규위반사항'].value_counts().nlargest(10).index
        df_hist = df[df['법규위반사항'].isin(top_violations)]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(y='법규위반사항', data=df_hist, order=top_violations, palette="plasma", ax=ax)
        
        ax.set_title("주요 법규 위반 사항 건수", fontsize=16)
        ax.set_xlabel("사고 건수")
        ax.set_ylabel("법규 위반 사항")
        st.pyplot(fig)
        
    def plot_heatmap(df):
        """히트맵: 시간대별 & 연령대별 사고 건수 시각화"""
        st.subheader("🔥 시간대별 X 연령대별 사고 건수 (히트맵)")
        
        # 피벗 테이블 생성 (시간, 연령대)
        pivot_table = df.pivot_table(
            index='발생시간', 
            columns='피해자_연령대', 
            aggfunc='size', 
            fill_value=0
        )
        # 연령대 순서 정렬
        age_order_cols = [c for c in ['0~9세', '10대', '20대', '30대', '40대', '50대', '60대', '70대 이상', '미상'] if c in pivot_table.columns]
        pivot_table = pivot_table[age_order_cols].sort_index()
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            pivot_table, 
            annot=True, 
            fmt="d", 
            cmap="YlOrRd", 
            linewidths=.5, 
            cbar_kws={'label': '사고 건수'},
            ax=ax
        )
        ax.set_title('발생 시간대별 X 피해자 연령대별 사고 건수', fontsize=16)
        ax.set_xlabel("피해자 연령대")
        ax.set_ylabel("발생 시간 (시)")
        st.pyplot(fig)

    def plot_scatter(df):
        """산점도: 발생 시간대와 피해자 연령의 관계 시각화"""
        st.subheader("✨ 발생 시간대 vs. 피해자 연령 (산점도)")
        st.caption("각 사고 지점은 발생 시간과 연령의 조합을 나타냅니다. 붉은색은 '차대사람' 사고입니다.")
        
        # 차대사람 사고 여부 플래그
        df['차대사람'] = df['사고유형'].apply(lambda x: 1 if '차대사람' in x else 0)
        
        fig, ax = plt.subplots(figsize=(10, 7))
        sns.scatterplot(
            x='발생시간', 
            y=df['피해자_연령대'].astype('category').cat.codes, # 연령대를 범주형 코드로 변환하여 Y축에 사용
            size='차대사람', # 차대사람 사고일 경우 점이 커짐
            hue='차대사람', # 차대사람 사고 여부에 따라 색상 구분
            palette={0: 'skyblue', 1: 'red'},
            data=df, 
            sizes=(20, 200), 
            legend='full',
            alpha=0.6,
            ax=ax
        )
        
        # Y축 라벨을 다시 연령대 이름으로 설정
        age_categories = df['피해자_연령대'].astype('category').cat.categories.tolist()
        ax.set_yticks(range(len(age_categories)))
        ax.set_yticklabels(age_categories)

        ax.set_title('발생 시간대와 피해자 연령대의 분포', fontsize=16)
        ax.set_xlabel("발생 시간 (시)")
        ax.set_ylabel("피해자 연령대")
        ax.set_xticks(range(0, 24, 2))
        ax.grid(True, linestyle='--', alpha=0.6)
        
        # '차대사람' 범례 제목 변경
        handles, labels = ax.get_legend_handles_labels()
        if len(labels) > 2:
            ax.legend(handles=[handles[-1], handles[-2]], labels=['차대사람', '기타'])

        st.pyplot(fig)


    # -----------------------------------------------------
    # 3. 메뉴에 따른 그래프 표시
    # -----------------------------------------------------

    st.markdown(f"### 분석 결과 ({selected_year}년)")
    
    if chart_type == '막대 그래프':
        plot_bar_chart(df_filtered)
    elif chart_type == '원 그래프':
        plot_pie_chart(df_filtered)
    elif chart_type == '선 그래프':
        plot_line_chart(df_filtered)
    elif chart_type == '히스토그램':
        plot_histogram(df_filtered)
    elif chart_type == '히트맵':
        # 히트맵은 특정 연도의 패턴을 더 자세히 볼 수 있습니다.
        plot_heatmap(df_filtered) 
    elif chart_type == '산점도':
        plot_scatter(df_filtered)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 데이터 미리보기")
    st.sidebar.dataframe(df_filtered.head())
    
else:
    st.error("데이터 로드 및 전처리 중 오류가 발생했거나, '중상' 사고 데이터가 없습니다.")
