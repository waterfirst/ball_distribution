import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
from matplotlib import font_manager, rc

# 한글 폰트 설정
font_path = "NanumGothic-Bold.ttf"  # 나눔고딕 폰트 파일의 경로를 지정하세요
font = font_manager.FontProperties(fname=font_path).get_name()
rc("font", family=font)


def create_initial_distribution(case, total_marbles):
    if case == "균등 분포":
        return np.full((10, 10), total_marbles // 100)
    else:  # 정규 분포
        mean = total_marbles / 100
        std = total_marbles / 300
        dist = np.random.normal(loc=mean, scale=std, size=(10, 10))
        dist = np.clip(dist, 0, None)  # 음수 값 제거
        dist = dist / dist.sum() * total_marbles  # 총 구슬 수 맞추기
        return np.round(dist).astype(int)


def redistribute_marbles(y, distribution_method, fairness_index):
    for i in range(10):
        for j in range(10):
            if y[i, j] > 0:
                if fairness_index == "부자 패널티" and y[i, j] == np.max(y):
                    t = min(2, y[i, j])
                else:
                    t = 1

                y[i, j] -= t

                if distribution_method == "랜덤분배":
                    new_i, new_j = np.random.randint(0, 10), np.random.randint(0, 10)
                else:  # 이웃분배
                    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
                    di, dj = directions[np.random.randint(0, 4)]
                    new_i, new_j = (i + di) % 10, (j + dj) % 10

                y[new_i, new_j] += t
    return y


def create_histogram(data, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(data.flatten(), bins=50, color="#80fd3d")
    ax.set_title(title)
    ax.set_xlabel("구슬 갯수")
    ax.set_ylabel("빈도수")
    return fig


def create_marble_plot(y, title):
    fig, ax = plt.subplots(figsize=(8, 6))
    max_marbles = np.max(y)
    for i in range(10):
        for j in range(10):
            size = (y[i, j] / max_marbles) ** 0.5 * 500  # 비선형적 크기 조정
            ax.scatter(i + 1, j + 1, s=size, alpha=0.5)
    ax.set_xlim(0.5, 10.5)
    ax.set_ylim(0.5, 10.5)
    ax.set_title(title)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    return fig


# Streamlit 앱 구성
st.title("구슬 재분배 시뮬레이션")

st.write(
    """
이 시뮬레이션은 10x10 격자에 구슬을 분포시키고, 특정 규칙에 따라 재분배하는 과정을 보여줍니다.
각 반복에서 구슬은 선택된 분배 방법과 공정 지수에 따라 이동합니다.
시뮬레이션을 통해 초기 분포에서 최종 분포로의 변화를 관찰할 수 있습니다.
"""
)

initial_distribution = st.selectbox("초기 분배:", ["균등 분포", "정규 분포"])
distribution_method = st.selectbox("거래 방법:", ["이웃분배", "랜덤분배"])
fairness_index = st.selectbox("공정 지수:", ["공정분배", "부자 패널티"])

marbles = st.number_input("구슬의 수:", min_value=100, max_value=10000, value=1000)
iterations = st.number_input("반복 횟수:", min_value=10, max_value=1000, value=100)


# Streamlit 앱 구성 부분

if (
    "initial_dist" not in st.session_state
    or st.session_state.initial_distribution != initial_distribution
):
    st.session_state.initial_dist = create_initial_distribution(
        initial_distribution, marbles
    )
    st.session_state.current_dist = st.session_state.initial_dist.copy()
    st.session_state.iteration = 0
    st.session_state.initial_distribution = initial_distribution


# 초기 분포 표시
col1, col2 = st.columns(2)
with col1:
    st.pyplot(create_histogram(st.session_state.initial_dist, "초기 구슬 분포"))
with col2:
    st.pyplot(create_marble_plot(st.session_state.current_dist, "현재 구슬 분포"))

# 시뮬레이션 실행 및 중단 버튼
col1, col2 = st.columns(2)
start_button = col1.button("시뮬레이션 실행")
stop_button = col2.button("시뮬레이션 중단")

# 히스토그램
hist_plot = st.empty()

# 진행 상황 표시
progress_bar = st.progress(0)

if "running" not in st.session_state:
    st.session_state.running = False

if start_button:
    st.session_state.running = True
    y = st.session_state.current_dist.copy()

    for l in range(iterations):
        if not st.session_state.running:
            break

        y = redistribute_marbles(y, distribution_method, fairness_index)

        st.session_state.current_dist = y
        st.session_state.iteration = l + 1

        # 히스토그램 업데이트
        fig, ax = plt.subplots()
        ax.hist(y.flatten(), bins=50, color="#80fd3d")
        ax.set_title(f"구슬 분포 (반복: {st.session_state.iteration})")
        ax.set_xlabel("구슬 갯수")
        ax.set_ylabel("빈도수")
        hist_plot.pyplot(fig)

        # 진행 상황 업데이트
        progress_bar.progress((l + 1) / iterations)

        # 매우 짧은 지연 (UI 업데이트를 위해)
        time.sleep(0.001)

    st.session_state.running = False

if stop_button:
    st.session_state.running = False

# 최종 분포 업데이트
col1, col2 = st.columns(2)
with col1:
    st.pyplot(create_marble_plot(st.session_state.initial_dist, "초기 구슬 분포"))
with col2:
    st.pyplot(create_marble_plot(st.session_state.current_dist, "현재 구슬 분포"))
