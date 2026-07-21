import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(
    page_title="전국 인구구조 유사 지역 찾기",
    page_icon="📊",
    layout="wide",
)

DEFAULT_CSV_FILE = "202606_202606.csv"


# =========================================================
# 화면 디자인
# =========================================================
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1450px;
            padding-top: 2rem;
            padding-bottom: 4rem;
        }

        .main-title {
            font-size: 2.2rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .sub-title {
            color: #666666;
            margin-bottom: 1.5rem;
        }

        [data-testid="stMetric"] {
            background-color: rgba(128, 128, 128, 0.08);
            border: 1px solid rgba(128, 128, 128, 0.18);
            border-radius: 12px;
            padding: 14px;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# 데이터 불러오기
# =========================================================
@st.cache_data
def load_csv_from_bytes(file_bytes):
    """
    여러 인코딩을 순서대로 시도하여 CSV를 불러옵니다.
    """
    encodings = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]
    last_error = None

    for encoding in encodings:
        try:
            dataframe = pd.read_csv(
                io.BytesIO(file_bytes),
                encoding=encoding,
                dtype=str,
                low_memory=False,
            )

            dataframe.columns = (
                dataframe.columns
                .astype(str)
                .str.replace("\u3000", " ", regex=False)
                .str.strip()
            )

            return dataframe, encoding

        except Exception as error:
            last_error = error

    raise ValueError(
        f"CSV 파일을 읽을 수 없습니다. 마지막 오류: {last_error}"
    )


def remove_region_code(region_name):
    """
    지역명 끝에 붙은 행정구역 코드를 제거합니다.

    예:
    서울특별시 종로구 (1111000000)
    → 서울특별시 종로구
    """
    region_name = str(region_name).strip()

    return re.sub(
        r"\s*\(\d+\)\s*$",
        "",
        region_name,
    ).strip()


def extract_region_code(region_name):
    """
    지역명에서 행정구역 코드를 추출합니다.
    """
    match = re.search(r"\((\d+)\)\s*$", str(region_name))

    if match:
        return match.group(1)

    return ""


def find_column(dataframe, gender, item):
    """
    연월이 포함된 실제 열 이름을 자동으로 찾습니다.

    예:
    2026년06월_계_0세
    2026년06월_계_총인구수
    """
    target_suffix = f"_{gender}_{item}"

    matching_columns = [
        column
        for column in dataframe.columns
        if str(column).strip().endswith(target_suffix)
    ]

    if matching_columns:
        return matching_columns[0]

    return None


def extract_reference_month(dataframe):
    """
    열 이름에서 기준 연월을 추출합니다.
    """
    for column in dataframe.columns:
        match = re.search(
            r"(\d{4})년\s*(\d{1,2})월",
            str(column),
        )

        if match:
            year = match.group(1)
            month = int(match.group(2))

            return f"{year}년 {month:02d}월"

    return "기준 연월 미상"


def convert_numeric_series(series):
    """
    쉼표가 포함된 인구 문자열을 숫자로 변환합니다.
    """
    return pd.to_numeric(
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace(" ", "", regex=False)
        .str.strip(),
        errors="coerce",
    ).fillna(0)


# =========================================================
# 인구구조 분석용 데이터 생성
# =========================================================
@st.cache_data
def prepare_population_data(raw_dataframe):
    dataframe = raw_dataframe.copy()

    if "행정구역" in dataframe.columns:
        region_column = "행정구역"
    else:
        region_column = dataframe.columns[0]

    dataframe[region_column] = (
        dataframe[region_column]
        .astype(str)
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
    )

    dataframe["지역명"] = dataframe[region_column].apply(
        remove_region_code
    )

    dataframe["행정구역코드"] = dataframe[region_column].apply(
        extract_region_code
    )

    age_labels = [
        f"{age}세"
        for age in range(100)
    ] + ["100세 이상"]

    age_columns = []

    for age_label in age_labels:
        column = find_column(
            dataframe,
            gender="계",
            item=age_label,
        )

        if column is None:
            raise ValueError(
                f"'{age_label}'에 해당하는 전체 인구 열을 찾지 못했습니다."
            )

        age_columns.append(column)

    # 연령별 인구수를 숫자로 변환
    age_population = pd.DataFrame(
        {
            age_label: convert_numeric_series(dataframe[column])
            for age_label, column in zip(age_labels, age_columns)
        }
    )

    # 총인구 열
    total_population_column = find_column(
        dataframe,
        gender="계",
        item="총인구수",
    )

    if total_population_column is not None:
        total_population = convert_numeric_series(
            dataframe[total_population_column]
        )
    else:
        total_population = age_population.sum(axis=1)

    # 연령별 인구 합계
    age_population_sum = age_population.sum(axis=1)

    # 연령별 구성비
    age_distribution = age_population.div(
        age_population_sum.replace(0, np.nan),
        axis=0,
    ).fillna(0)

    # 코사인 유사도 계산을 위한 L2 정규화
    distribution_array = age_distribution.to_numpy(
        dtype=np.float64
    )

    vector_norms = np.linalg.norm(
        distribution_array,
        axis=1,
        keepdims=True,
    )

    normalized_vectors = np.divide(
        distribution_array,
        vector_norms,
        out=np.zeros_like(distribution_array),
        where=vector_norms != 0,
    )

    metadata = pd.DataFrame(
        {
            "지역명": dataframe["지역명"],
            "원본지역명": dataframe[region_column],
            "행정구역코드": dataframe["행정구역코드"],
            "총인구수": total_population.astype(int),
        }
    )

    return (
        metadata,
        age_population,
        age_distribution,
        normalized_vectors,
        age_labels,
    )


def is_eup_myeon_dong(region_name):
    """
    지역명이 읍·면·동에 해당하는지 대략적으로 판별합니다.
    """
    region_name = str(region_name).strip()

    patterns = [
        r"읍$",
        r"면$",
        r"동$",
        r"\d가동$",
        r"출장소$",
    ]

    return any(
        re.search(pattern, region_name)
        for pattern in patterns
    )


def calculate_similar_regions(
    selected_index,
    metadata,
    normalized_vectors,
    comparison_scope,
    minimum_population,
    top_n=5,
):
    """
    선택 지역과 전국 지역 간 코사인 유사도를 계산합니다.
    """
    selected_vector = normalized_vectors[selected_index]

    similarities = normalized_vectors @ selected_vector

    result = metadata.copy()
    result["원본인덱스"] = np.arange(len(metadata))
    result["유사도"] = similarities
    result["유사도점수"] = similarities * 100

    # 자기 자신 제외
    result = result[
        result["원본인덱스"] != selected_index
    ]

    # 인구가 없는 지역 제외
    result = result[
        result["총인구수"] >= minimum_population
    ]

    if comparison_scope == "읍면동만 비교":
        result = result[
            result["지역명"].apply(is_eup_myeon_dong)
        ]

    result = result.sort_values(
        by=["유사도", "총인구수"],
        ascending=[False, False],
    )

    return result.head(top_n).reset_index(drop=True)


# =========================================================
# Plotly 그래프
# =========================================================
def create_similarity_bar_chart(
    similar_regions,
    selected_region,
):
    chart_data = similar_regions.copy()

    chart_data = chart_data.sort_values(
        "유사도점수",
        ascending=True,
    )

    figure = go.Figure()

    figure.add_trace(
        go.Bar(
            x=chart_data["유사도점수"],
            y=chart_data["지역명"],
            orientation="h",
            text=chart_data["유사도점수"].map(
                lambda value: f"{value:.2f}%"
            ),
            textposition="outside",
            customdata=np.column_stack(
                [
                    chart_data["총인구수"],
                    chart_data["유사도점수"],
                ]
            ),
            hovertemplate=(
                "<b>%{y}</b><br>"
                "유사도: %{customdata[1]:.2f}%<br>"
                "총인구: %{customdata[0]:,.0f}명"
                "<extra></extra>"
            ),
            marker=dict(
                color=chart_data["유사도점수"],
                colorscale="Blues",
                showscale=False,
            ),
        )
    )

    minimum_score = chart_data["유사도점수"].min()

    x_axis_minimum = max(
        0,
        minimum_score - 1,
    )

    figure.update_layout(
        title=dict(
            text=f"{selected_region}과 인구구조가 비슷한 지역 Top 5",
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(
            title="인구구조 유사도 점수",
            range=[x_axis_minimum, 100.3],
            ticksuffix="%",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        yaxis=dict(
            title="",
            automargin=True,
        ),
        height=440,
        margin=dict(
            l=30,
            r=70,
            t=70,
            b=40,
        ),
        template="plotly_white",
    )

    return figure


def create_population_structure_chart(
    selected_index,
    selected_region,
    similar_regions,
    age_distribution,
):
    figure = go.Figure()

    age_values = list(range(101))

    selected_distribution = (
        age_distribution
        .iloc[selected_index]
        .to_numpy(dtype=float)
        * 100
    )

    # 선택 지역
    figure.add_trace(
        go.Scatter(
            x=age_values,
            y=selected_distribution,
            mode="lines",
            name=f"선택 지역: {selected_region}",
            line=dict(
                width=5,
                color="#111827",
            ),
            hovertemplate=(
                "<b>선택 지역</b><br>"
                "나이: %{x}세<br>"
                "구성비: %{y:.3f}%"
                "<extra></extra>"
            ),
        )
    )

    comparison_colors = [
        "#2563EB",
        "#DC2626",
        "#059669",
        "#D97706",
        "#7C3AED",
    ]

    for rank, row in similar_regions.iterrows():
        comparison_index = int(row["원본인덱스"])

        comparison_distribution = (
            age_distribution
            .iloc[comparison_index]
            .to_numpy(dtype=float)
            * 100
        )

        figure.add_trace(
            go.Scatter(
                x=age_values,
                y=comparison_distribution,
                mode="lines",
                name=(
                    f"{rank + 1}위 {row['지역명']} "
                    f"({row['유사도점수']:.2f}%)"
                ),
                line=dict(
                    width=2.5,
                    color=comparison_colors[
                        rank % len(comparison_colors)
                    ],
                ),
                opacity=0.85,
                hovertemplate=(
                    f"<b>{rank + 1}위 "
                    f"{row['지역명']}</b><br>"
                    "나이: %{x}세<br>"
                    "구성비: %{y:.3f}%"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=dict(
            text="연령별 인구 구성비 비교",
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(
            title="나이",
            tickmode="linear",
            tick0=0,
            dtick=10,
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        yaxis=dict(
            title="해당 지역 전체 인구 중 비율",
            ticksuffix="%",
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        height=650,
        margin=dict(
            l=30,
            r=30,
            t=130,
            b=40,
        ),
        template="plotly_white",
    )

    return figure


def create_population_count_chart(
    selected_index,
    selected_region,
    similar_regions,
    age_population,
):
    """
    실제 인구수 비교 그래프입니다.
    지역 규모 차이를 함께 확인할 때 사용합니다.
    """
    figure = go.Figure()

    age_values = list(range(101))

    selected_population = (
        age_population
        .iloc[selected_index]
        .to_numpy(dtype=float)
    )

    figure.add_trace(
        go.Scatter(
            x=age_values,
            y=selected_population,
            mode="lines",
            name=f"선택 지역: {selected_region}",
            line=dict(
                width=5,
                color="#111827",
            ),
            hovertemplate=(
                "<b>선택 지역</b><br>"
                "나이: %{x}세<br>"
                "인구: %{y:,.0f}명"
                "<extra></extra>"
            ),
        )
    )

    comparison_colors = [
        "#2563EB",
        "#DC2626",
        "#059669",
        "#D97706",
        "#7C3AED",
    ]

    for rank, row in similar_regions.iterrows():
        comparison_index = int(row["원본인덱스"])

        comparison_population = (
            age_population
            .iloc[comparison_index]
            .to_numpy(dtype=float)
        )

        figure.add_trace(
            go.Scatter(
                x=age_values,
                y=comparison_population,
                mode="lines",
                name=f"{rank + 1}위 {row['지역명']}",
                line=dict(
                    width=2.5,
                    color=comparison_colors[
                        rank % len(comparison_colors)
                    ],
                ),
                opacity=0.85,
                hovertemplate=(
                    f"<b>{rank + 1}위 "
                    f"{row['지역명']}</b><br>"
                    "나이: %{x}세<br>"
                    "인구: %{y:,.0f}명"
                    "<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        title=dict(
            text="연령별 실제 인구수 비교",
            x=0.01,
            xanchor="left",
        ),
        xaxis=dict(
            title="나이",
            tickmode="linear",
            tick0=0,
            dtick=10,
            range=[0, 100],
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        yaxis=dict(
            title="인구수",
            tickformat=",",
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0,
        ),
        height=650,
        margin=dict(
            l=30,
            r=30,
            t=130,
            b=40,
        ),
        template="plotly_white",
    )

    return figure


# =========================================================
# 제목
# =========================================================
st.markdown(
    '<div class="main-title">'
    "📊 전국 인구구조 유사 지역 찾기"
    "</div>",
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    "선택한 지역과 연령별 인구구조가 가장 비슷한 "
    "전국 지역 Top 5를 찾아 비교합니다."
    "</div>",
    unsafe_allow_html=True,
)


# =========================================================
# 파일 선택
# =========================================================
with st.sidebar:
    st.header("⚙️ 데이터 설정")

    uploaded_file = st.file_uploader(
        "다른 주민등록 인구 CSV 사용",
        type=["csv"],
        help=(
            "파일을 업로드하지 않으면 저장소의 "
            f"{DEFAULT_CSV_FILE} 파일을 사용합니다."
        ),
    )


try:
    if uploaded_file is not None:
        file_bytes = uploaded_file.getvalue()
        data_source_name = uploaded_file.name

    else:
        default_path = Path(DEFAULT_CSV_FILE)

        if not default_path.exists():
            st.error(
                f"`{DEFAULT_CSV_FILE}` 파일을 찾을 수 없습니다."
            )

            st.info(
                "CSV 파일을 main.py와 같은 위치에 올리거나 "
                "왼쪽 메뉴에서 직접 업로드하세요."
            )

            st.stop()

        file_bytes = default_path.read_bytes()
        data_source_name = DEFAULT_CSV_FILE

    raw_df, detected_encoding = load_csv_from_bytes(
        file_bytes
    )

    (
        metadata,
        age_population,
        age_distribution,
        normalized_vectors,
        age_labels,
    ) = prepare_population_data(raw_df)

except Exception as error:
    st.error("데이터를 처리하는 중 오류가 발생했습니다.")
    st.exception(error)
    st.stop()


reference_month = extract_reference_month(raw_df)


# =========================================================
# 사이드바 분석 설정
# =========================================================
with st.sidebar:
    st.divider()
    st.subheader("🔍 유사 지역 설정")

    comparison_scope = st.radio(
        "비교 대상",
        options=[
            "모든 행정구역",
            "읍면동만 비교",
        ],
        index=0,
        help=(
            "모든 행정구역을 선택하면 시도, 시군구, "
            "읍면동이 모두 비교 대상에 포함됩니다."
        ),
    )

    minimum_population = st.number_input(
        "비교 지역 최소 인구",
        min_value=0,
        max_value=1_000_000,
        value=1000,
        step=1000,
        help=(
            "인구가 매우 적은 지역은 연령별 구성비가 "
            "불안정할 수 있습니다."
        ),
    )

    chart_mode = st.radio(
        "비교 그래프 단위",
        options=[
            "인구 구성비",
            "실제 인구수",
        ],
        index=0,
        help=(
            "유사도는 그래프 선택과 관계없이 "
            "연령별 인구 구성비로 계산됩니다."
        ),
    )

    st.divider()
    st.caption(f"파일: {data_source_name}")
    st.caption(f"기준: {reference_month}")
    st.caption(f"인코딩: {detected_encoding}")
    st.caption(f"전체 행정구역: {len(metadata):,}개")


# =========================================================
# 지역 검색 및 선택
# =========================================================
st.subheader("1. 분석할 지역 선택")

search_keyword = st.text_input(
    "지역명 검색",
    placeholder="예: 서울특별시, 수원시, 청운효자동",
    help="지역명의 일부를 입력할 수 있습니다.",
)

if search_keyword.strip():
    search_mask = metadata["지역명"].str.contains(
        search_keyword.strip(),
        case=False,
        na=False,
        regex=False,
    )

    filtered_metadata = metadata[search_mask].copy()

else:
    filtered_metadata = metadata.copy()


if filtered_metadata.empty:
    st.warning(
        f"'{search_keyword}'에 해당하는 지역을 찾지 못했습니다."
    )
    st.stop()


region_options = filtered_metadata.index.tolist()

selected_index = st.selectbox(
    "검색 결과에서 정확한 지역 선택",
    options=region_options,
    format_func=lambda index: (
        f"{metadata.loc[index, '지역명']} "
        f"— {metadata.loc[index, '총인구수']:,}명"
    ),
)

selected_index = int(selected_index)
selected_region = metadata.loc[selected_index, "지역명"]
selected_population = int(
    metadata.loc[selected_index, "총인구수"]
)


# =========================================================
# 유사 지역 계산
# =========================================================
similar_regions = calculate_similar_regions(
    selected_index=selected_index,
    metadata=metadata,
    normalized_vectors=normalized_vectors,
    comparison_scope=comparison_scope,
    minimum_population=int(minimum_population),
    top_n=5,
)


if similar_regions.empty:
    st.warning(
        "현재 조건에 해당하는 비교 지역이 없습니다. "
        "최소 인구 기준을 낮추거나 비교 대상을 변경하세요."
    )
    st.stop()


# =========================================================
# 선택 지역 핵심 정보
# =========================================================
selected_distribution = age_distribution.iloc[
    selected_index
]

youth_ratio = (
    selected_distribution.iloc[0:15].sum() * 100
)

working_age_ratio = (
    selected_distribution.iloc[15:65].sum() * 100
)

senior_ratio = (
    selected_distribution.iloc[65:101].sum() * 100
)

largest_age_index = int(
    age_population
    .iloc[selected_index]
    .to_numpy()
    .argmax()
)

largest_age_label = (
    f"{largest_age_index}세"
    if largest_age_index < 100
    else "100세 이상"
)

metric_columns = st.columns(5)

with metric_columns[0]:
    st.metric(
        "선택 지역",
        selected_region,
    )

with metric_columns[1]:
    st.metric(
        "총인구",
        f"{selected_population:,}명",
    )

with metric_columns[2]:
    st.metric(
        "0~14세",
        f"{youth_ratio:.1f}%",
    )

with metric_columns[3]:
    st.metric(
        "65세 이상",
        f"{senior_ratio:.1f}%",
    )

with metric_columns[4]:
    st.metric(
        "인구 최다 연령",
        largest_age_label,
    )


# =========================================================
# Top 5 결과 표
# =========================================================
st.subheader("2. 인구구조 유사 지역 Top 5")

result_table = similar_regions[
    [
        "지역명",
        "총인구수",
        "유사도점수",
    ]
].copy()

result_table.insert(
    0,
    "순위",
    range(1, len(result_table) + 1),
)

result_table["유사도점수"] = (
    result_table["유사도점수"].round(4)
)

st.dataframe(
    result_table,
    use_container_width=True,
    hide_index=True,
    column_config={
        "순위": st.column_config.NumberColumn(
            "순위",
            format="%d위",
        ),
        "지역명": st.column_config.TextColumn(
            "지역",
        ),
        "총인구수": st.column_config.NumberColumn(
            "총인구",
            format="%d명",
        ),
        "유사도점수": st.column_config.ProgressColumn(
            "인구구조 유사도",
            min_value=0,
            max_value=100,
            format="%.2f%%",
        ),
    },
)


# =========================================================
# 유사도 막대그래프
# =========================================================
similarity_bar_chart = create_similarity_bar_chart(
    similar_regions=similar_regions,
    selected_region=selected_region,
)

st.plotly_chart(
    similarity_bar_chart,
    use_container_width=True,
    config={
        "displaylogo": False,
    },
)


# =========================================================
# 연령별 구조 비교 그래프
# =========================================================
st.subheader("3. 선택 지역과 Top 5 연령별 비교")

if chart_mode == "인구 구성비":
    comparison_chart = create_population_structure_chart(
        selected_index=selected_index,
        selected_region=selected_region,
        similar_regions=similar_regions,
        age_distribution=age_distribution,
    )

else:
    comparison_chart = create_population_count_chart(
        selected_index=selected_index,
        selected_region=selected_region,
        similar_regions=similar_regions,
        age_population=age_population,
    )


st.plotly_chart(
    comparison_chart,
    use_container_width=True,
    config={
        "displaylogo": False,
        "scrollZoom": True,
        "modeBarButtonsToRemove": [
            "lasso2d",
            "select2d",
        ],
    },
)


# =========================================================
# 분석 설명
# =========================================================
with st.expander("유사도 계산 방법 보기"):
    st.markdown(
        """
        **유사도 계산 과정**

        1. 각 지역의 0세부터 100세 이상까지 인구를 가져옵니다.
        2. 지역마다 연령별 인구를 해당 지역의 전체 연령 인구로 나눕니다.
        3. 지역별 연령 구성비를 101개의 숫자로 이루어진 벡터로 만듭니다.
        4. 선택 지역과 전국 지역의 코사인 유사도를 계산합니다.
        5. 선택 지역 자체를 제외하고 점수가 높은 5개 지역을 표시합니다.

        유사도 점수가 100%에 가까울수록 연령별 인구구조의 모양이
        비슷하다는 뜻입니다. 총인구 규모가 비슷하다는 뜻은 아닙니다.
        """
    )


# =========================================================
# 결과 다운로드
# =========================================================
download_table = result_table.copy()

download_table["기준지역"] = selected_region
download_table["기준지역_총인구"] = selected_population
download_table["기준연월"] = reference_month

download_table = download_table[
    [
        "기준연월",
        "기준지역",
        "기준지역_총인구",
        "순위",
        "지역명",
        "총인구수",
        "유사도점수",
    ]
]

download_csv = download_table.to_csv(
    index=False,
    encoding="utf-8-sig",
).encode("utf-8-sig")

safe_region_name = re.sub(
    r'[\\/:*?"<>|]',
    "_",
    selected_region,
)

st.download_button(
    label="Top 5 분석 결과 CSV 다운로드",
    data=download_csv,
    file_name=f"{safe_region_name}_유사인구구조_Top5.csv",
    mime="text/csv",
)
