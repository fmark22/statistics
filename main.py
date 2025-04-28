import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.datasets import make_classification
from scipy import stats
import os
import urllib.request
import matplotlib.font_manager as fm

# 🔥 로컬 폰트 파일 경로로 설정
font_path = './fonts/NanumGothic-Regular.ttf'  # 여기로 바꿔줘
fm.fontManager.addfont(font_path)

# 폰트 설정
plt.rcParams['font.family'] = 'NanumGothic'
plt.rcParams['axes.unicode_minus'] = False

# 페이지 설정
st.set_page_config(page_title="통계", page_icon=":sparkles:", layout="wide")

# 탭 메뉴
tabs = st.tabs(["평균과 중앙값", "정규분포", "상관관계", "회귀분석", "혼동행렬 (Confusion Matrix)"])

# --- 평균과 중앙값 ---
with tabs[0]:
    # ▶️ 예시 데이터 세트
    examples = {
        "대칭 분포 (예: 정규분포 비슷)": "1, 2, 3, 4, 5, 6, 7",
        "오른쪽 치우친 데이터 (Positive Skew)": "2, 2, 2, 3, 3, 4, 10",
        "왼쪽 치우친 데이터 (Negative Skew)": "1, 7, 8, 8, 8, 9, 10",
        "최빈값이 여러 개인 데이터 (다봉)": "2, 2, 3, 3, 7, 7, 8, 8",
    }

    # ▶️ Session State 초기화
    if "example_input" not in st.session_state:
        st.session_state.example_input = ""

    # ▶️ 버튼 클릭 시 자동 입력
    st.subheader("좋은 예시를 골라보세요!")
    col1, col2 = st.columns(2)
    with col1:
        for label, example in list(examples.items())[:2]:
            if st.button(label):
                st.session_state.example_input = example
    with col2:
        for label, example in list(examples.items())[2:]:
            if st.button(label):
                st.session_state.example_input = example

    # ▶️ 직접 입력 창
    st.subheader("또는 직접 입력해보세요 (정수만, 쉼표로 구분)")
    data = st.text_input("숫자 입력", value=st.session_state.example_input)

    # ▶️ 입력 데이터 처리 및 시각화
    if data:
        try:
            numbers = [int(x.strip()) for x in data.split(',') if x.strip() != '']

            mean_val = np.mean(numbers)
            median_val = np.median(numbers)
            mode_val, _ = stats.mode(numbers, keepdims=True)

            st.success(f"평균: {mean_val:.2f}, 중앙값: {median_val:.2f}, 최빈값: {mode_val[0]:.2f}")

            plt.style.use('seaborn-v0_8-bright')
            fig, ax = plt.subplots()
            sns.histplot(numbers, bins=10, kde=True, ax=ax)
            plt.axvline(mean_val, color='red', linestyle='--', label=f"평균 {mean_val:.2f}")
            plt.axvline(median_val, color='green', linestyle='-.', label=f"중앙값 {median_val:.2f}")
            plt.axvline(mode_val[0], color='blue', linestyle=':', label=f"최빈값 {mode_val[0]:.2f}")
            plt.legend(fontsize=12)
            ax.set_title('숫자들의 분포와 평균/중앙값/최빈값', fontsize=14)
            st.pyplot(fig)

        except ValueError:
            st.error("⚠️ 숫자는 반드시 '정수'로만 입력해주세요! (예: 1, 2, 3)")

# --- 정규분포 ---
with tabs[1]:
    st.header("2. 정규분포")
    st.write("""
    - **정규분포**: 가운데가 높고 양쪽으로 점점 낮아지는 '종' 모양 그래프예요.
    - 평균(가운데 위치)과 표준편차(퍼지는 정도)를 조절할 수 있어요.
    """)

    mean = st.slider("평균 (Mean)", -10.0, 10.0, 0.0)
    std_dev = st.slider("표준편차 (Standard Deviation)", 0.5, 5.0, 1.0)
    size = st.slider("표본 수", 100, 5000, 1000)

    data = np.random.normal(loc=mean, scale=std_dev, size=size)

    plt.style.use('seaborn-v0_8-bright')
    fig, ax = plt.subplots()
    sns.histplot(data, kde=True, ax=ax)
    ax.set_title(f"정규분포 (평균={mean}, 표준편차={std_dev})", fontsize=14)
    st.pyplot(fig)

# --- 상관관계 ---
with tabs[2]:
    st.header("3. 상관관계")
    st.write("""
    - **상관관계**는 두 변수의 관계를 보여줘요.
    - 예를 들어, 날씨가 더워질수록 아이스크림 판매가 늘어나는 것처럼요!
    """)

    n = st.slider("표본 수", 50, 1000, 300)
    correlation = st.slider("상관계수", -1.0, 1.0, 0.8)

    x = np.random.normal(0, 1, n)
    noise = np.random.normal(0, (1 - correlation**2)**0.5, n)
    y = correlation * x + noise

    plt.style.use('seaborn-v0_8-bright')
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, ax=ax)
    ax.set_title(f"상관관계 시각화 (r ≈ {np.corrcoef(x, y)[0,1]:.2f})", fontsize=14)
    st.pyplot(fig)

# --- 회귀분석 ---
with tabs[3]:
    st.header("4. 회귀분석")
    st.write("""
    - **회귀분석**은 두 변수 사이의 관계를 직선으로 나타내는 방법이에요.
    """)

    n = st.slider("표본 수", 50, 500, 200)
    slope = st.slider("선의 기울기 (slope)", -5.0, 5.0, 1.0)
    intercept = st.slider("선의 절편 (intercept)", -10.0, 10.0, 0.0)

    x = np.linspace(0, 10, n)
    noise = np.random.normal(0, 2, n)
    y = slope * x + intercept + noise

    plt.style.use('seaborn-v0_8-bright')
    fig, ax = plt.subplots()
    sns.scatterplot(x=x, y=y, ax=ax)
    sns.regplot(x=x, y=y, scatter=False, color='red', ax=ax)
    ax.set_title(f"회귀선 (y = {slope:.1f}x + {intercept:.1f})", fontsize=14)
    st.pyplot(fig)

# --- 혼동행렬 (Confusion Matrix) ---
with tabs[4]:
    st.header("5. 혼동행렬 (Confusion Matrix)")
    st.write("""
    - **혼동행렬**은 분류 결과가 얼마나 정확한지 보여주는 표예요.
    - 정답과 예측을 비교해 '맞음/틀림'을 알 수 있어요.
    """)

    n_samples = st.slider("데이터 수", 100, 1000, 500)
    classes = st.slider("클래스 수", 2, 5, 2)

    X, y_true = make_classification(n_samples=n_samples, n_classes=classes, n_informative=2, n_clusters_per_class=1)
    y_pred = np.random.choice(range(classes), size=n_samples)

    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax)
    ax.set_title("혼동행렬", fontsize=14)
    st.pyplot(fig)

    st.info("대각선이 많으면 잘 분류한 거예요!")
