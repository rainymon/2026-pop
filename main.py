import re
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ---------------------------------------------------------
# 페이지 기본 설정
# ---------------------------------------------------------
st.set_page_config(
    page_title="지역별 인구 구조 분석",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------
# 디자인
# ---------------------------------------------------------
st.markdown(
    """
    <style>
        .block-container {
            max-width: 1400px;
            padding-top: 2rem;
            padding-bottom: 3rem;
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


# ---------------------------------------------------------
# 상수
# ---------------------------------------------------------
DEFAULT_DATA_FILE = "202606_202606.csv"


# ---------------------------------------------------------
# 데이터 관련 함수
# ---------------------------------------------------------
@st.cache_data
def load_csv(file_source):
    """
    CSV 파일의 인코딩을 자동으로 확인해 불러옵니다.
    """
    encodings = ["cp949", "euc-kr", "utf-8-sig", "utf-8"]

    last_error = None

    for encoding in encodings:
        try:
            df = pd.read_csv(
                file_source,
                encoding=encoding,
                dtype=str,
                low_memory=False,
            )

            df.columns = df.columns.astype(str).str.strip()

            return df, encoding

        except Exception as error:
            last_error = error

            # 업로드 파일은 인코딩 재시도 전에 처음 위치로 되돌립니다.
            if hasattr(file_source, "seek"):
                file_source.seek(0)

    raise ValueError(
        f"CSV 파일을 읽을 수 없습니다. 마지막 오류: {last_error}"
    )


def clean_number(value):
    """
    '9,289,813'처럼 쉼표가 포함된 문자열을 숫자로 변환합니다.
    """
    if pd.isna(value):
        return 0

    text = str(value).strip()
    text = text.replace(",", "")
    text = text.replace(" ", "")

    return pd.to_numeric(text, errors="coerce")


def remove_region_code(region_name):
    """
    지역명 뒤의 행정구역 코드를 제거합니다.

    예:
    서울특별시 종로구 (1111000000)
    -> 서울특별시 종로구
    """
    return re.sub(r"\s*\(\d+\)\s*$", "", str(region_name)).strip()


def find_column(df, gender, item):
    """
    데이터의 연월이 달라져도 열을 자동으로 찾습니다.

    gender:
        계, 남, 여

    item:
        총인구수, 0세, 1세, ..., 100세 이상
    """
    ending = f"_{gender}_{item}"

    matching_columns = [
        column
        for column in df.columns
        if str(column).strip().endswith(ending)
    ]

    if not matching_columns:
        return None

    return matching_columns[0]


def extract_reference_month(df):
    """
    열 이름에서 기준 연월을 추출합니다.

    예:
    2026년06월_계_총인구수
    -> 2026년 06월
    """
    for column in df.columns:
        match = re.search(r"(\d{4})년(\d{2})월", str(column))

        if match:
            year = match.group(1)
            month = match.group(2)
            return f"{year}년 {month}월"

    return "기준 연월 미상"


def prepare_data(df):
    """
    지역명과 숫자형 인구 열을 정리합니다.
    """
    region_column = "행정구역"

    if region_column not in df.columns:
        region_column = df.columns[0]

    df = df.copy()

    df[region_column] = (
        df[region_column]
        .astype(str)
        .str.replace("\u3000", " ", regex=False)
        .str.strip()
    )

    df["지역명"] = df[region_column].apply(remove_region_code)

    return df, region_column


def get_age_population(row, df, gender):
    """
    선택한 지역의 0세~100세 이상 인구를 가져옵니다.
    """
    records = []

    for age in range(101):
        age_label = f"{age}세" if age < 100 else "100세 이상"
        column = find_column(df, gender, age_label)

        if column is None:
            population = 0
        else:
            population = clean_number(row[column])

        if pd.isna(population):
            population = 0

        records.append(
            {
                "나이": age_label,
                "나이값": age,
                "인구수": int(population),
                "구분": gender,
            }
        )

    return pd.DataFrame(records)


def calculate_age_group(age_df, start_age, end_age):
    """
    특정 연령 구간의 인구 합계를 계산합니다.
    """
    mask = age_df["나이값"].between(start_age, end_age)
    return int(age_df.loc[mask, "인구수"].sum())


def make_line_chart(chart_df, selected_region, display_mode):
    """
    Plotly 꺾은선 그래프를 생성합니다.
    """
    figure = go.Figure()

    line_settings = {
        "계": {
            "name": "전체",
            "color": "#3B82F6",
            "width": 4,
        },
        "남": {
            "name": "남성",
            "color": "#22A6F2",
            "width": 3,
        },
        "여": {
            "name": "여성",
            "color": "#F06292",
            "width": 3,
        },
    }

    for gender in chart_df["구분"].unique():
        gender_df = chart_df[chart_df["구분"] == gender]

        settings = line_settings.get(
            gender,
            {
                "name": gender,
                "color": "#6366F1",
                "width": 3,
            },
        )

        figure.add_trace(
            go.Scatter(
                x=gender_df["나이값"],
                y=gender_df["표시값"],
                mode="lines",
                name=settings["name"],
                line=dict(
                    color=settings["color"],
                    width=settings["width"],
                ),
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "나이: %{x}세<br>"
                    + (
                        "비율: %{y:.2f}%"
                        if display_mode == "비율"
                        else "인구: %{y:,.0f}명"
                    )
                    + "<extra></extra>"
                ),
            )
        )

    y_axis_title = (
        "전체 인구 대비 비율 (%)"
        if display_mode == "비율"
        else "인구수 (명)"
    )

    figure.update_layout(
        title=dict(
            text=f"{selected_region} 연령별 인구 구조",
            x=0.02,
            xanchor="left",
            font=dict(size=22),
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
            title=y_axis_title,
            tickformat=".2f" if display_mode == "비율" else ",",
            rangemode="tozero",
            showgrid=True,
            gridcolor="rgba(128, 128, 128, 0.15)",
        ),
        hovermode="x unified",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
        ),
        margin=dict(l=30, r=30, t=100, b=30),
        height=590,
        template="plotly_white",
    )

    return figure


# ---------------------------------------------------------
# 제목
# ---------------------------------------------------------
st.markdown(
    '<div class="main-title">📊 지역별 인구 구조 분석</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="sub-title">'
    "지역을 검색하거나 목록에서 선택해 연령별 인구 구조를 확인하세요."
    "</div>",
    unsafe_allow_html=True,
)


# ---------------------------------------------------------
# 데이터 불러오기
# ---------------------------------------------------------
default_file_path = Path(DEFAULT_DATA_FILE)

with st.sidebar:
    st.header("⚙️ 데이터 설정")

    uploaded_file = st.file_uploader(
        "다른 CSV 파일 사용",
        type=["csv"],
        help=(
            "파일을 업로드하지 않으면 GitHub 저장소의 "
            f"{DEFAULT_DATA_FILE} 파일을 사용합니다."
        ),
    )

    st.caption(
        "업로드한 파일은 현재 실행 중인 세션에서만 사용됩니다."
    )


try:
    if uploaded_file is not None:
        raw_df, detected_encoding = load_csv(uploaded_file)
        data_source_name = uploaded_file.name

    elif default_file_path.exists():
        raw_df, detected_encoding = load_csv(default_file_path)
        data_source_name = DEFAULT_DATA_FILE

    else:
        st.error(
            f"`{DEFAULT_DATA_FILE}` 파일을 찾을 수 없습니다."
        )

        st.info(
            "GitHub 저장소에 CSV 파일을 추가하거나 "
            "왼쪽 메뉴에서 CSV 파일을 업로드하세요."
        )

        st.stop()

except Exception as error:
    st.error("데이터를 불러오는 중 오류가 발생했습니다.")
    st.exception(error)
    st.stop()


df, region_column = prepare_data(raw_df)
reference_month = extract_reference_month(df)


# ---------------------------------------------------------
# 지역 검색 및 선택
# ---------------------------------------------------------
st.subheader("1. 지역 선택")

search_keyword = st.text_input(
    "지역 검색",
    placeholder="예: 서울특별시, 종로구, 청운효자동",
    help="지역명의 일부만 입력해도 검색됩니다.",
)

all_regions = (
    df["지역명"]
    .dropna()
    .astype(str)
    .drop_duplicates()
    .tolist()
)

if search_keyword.strip():
    keyword = search_keyword.strip()

    filtered_regions = [
        region
        for region in all_regions
        if keyword.lower() in region.lower()
    ]

else:
    filtered_regions = all_regions


if not filtered_regions:
    st.warning(
        f"'{search_keyword}'와 일치하는 지역이 없습니다. "
        "검색어를 더 짧게 입력해 보세요."
    )
    st.stop()


selected_region = st.selectbox(
    "검색 결과에서 지역 선택",
    options=filtered_regions,
    index=0,
    help=(
        "위 검색창에 지역을 입력한 뒤 "
        "이 목록에서 정확한 지역을 선택하세요."
    ),
)


selected_rows = df[df["지역명"] == selected_region]

if selected_rows.empty:
    st.error("선택한 지역의 데이터를 찾지 못했습니다.")
    st.stop()

selected_row = selected_rows.iloc[0]


# ---------------------------------------------------------
# 표시 옵션
# ---------------------------------------------------------
with st.sidebar:
    st.divider()
    st.subheader("📈 그래프 설정")

    gender_option = st.radio(
        "표시할 인구",
        options=[
            "전체",
            "남성",
            "여성",
            "남녀 비교",
            "전체·남성·여성",
        ],
        index=4,
    )

    display_mode = st.radio(
        "표시 단위",
        options=["인구수", "비율"],
        horizontal=True,
    )

    smoothing_window = st.slider(
        "이동평균",
        min_value=1,
        max_value=10,
        value=1,
        help=(
            "1은 원본 데이터입니다. 값을 높이면 "
            "연령별 변화가 부드럽게 표시됩니다."
        ),
    )

    st.divider()
    st.caption(f"데이터 파일: {data_source_name}")
    st.caption(f"기준: {reference_month}")
    st.caption(f"인코딩: {detected_encoding}")
    st.caption(f"지역 수: {len(df):,}개")


# ---------------------------------------------------------
# 선택 지역 데이터 만들기
# ---------------------------------------------------------
total_df = get_age_population(selected_row, df, "계")
male_df = get_age_population(selected_row, df, "남")
female_df = get_age_population(selected_row, df, "여")

gender_map = {
    "전체": ["계"],
    "남성": ["남"],
    "여성": ["여"],
    "남녀 비교": ["남", "여"],
    "전체·남성·여성": ["계", "남", "여"],
}

selected_genders = gender_map[gender_option]

population_frames = {
    "계": total_df,
    "남": male_df,
    "여": female_df,
}

chart_frames = []

for gender in selected_genders:
    current_df = population_frames[gender].copy()

    if smoothing_window > 1:
        current_df["인구수"] = (
            current_df["인구수"]
            .rolling(
                window=smoothing_window,
                center=True,
                min_periods=1,
            )
            .mean()
        )

    if display_mode == "비율":
        gender_total = current_df["인구수"].sum()

        if gender_total > 0:
            current_df["표시값"] = (
                current_df["인구수"] / gender_total * 100
            )
        else:
            current_df["표시값"] = 0

    else:
        current_df["표시값"] = current_df["인구수"]

    chart_frames.append(current_df)


chart_df = pd.concat(chart_frames, ignore_index=True)


# ---------------------------------------------------------
# 핵심 지표
# ---------------------------------------------------------
total_population_column = find_column(df, "계", "총인구수")
male_population_column = find_column(df, "남", "총인구수")
female_population_column = find_column(df, "여", "총인구수")

total_population = (
    int(clean_number(selected_row[total_population_column]))
    if total_population_column
    else int(total_df["인구수"].sum())
)

male_population = (
    int(clean_number(selected_row[male_population_column]))
    if male_population_column
    else int(male_df["인구수"].sum())
)

female_population = (
    int(clean_number(selected_row[female_population_column]))
    if female_population_column
    else int(female_df["인구수"].sum())
)

youth_population = calculate_age_group(total_df, 0, 14)
working_population = calculate_age_group(total_df, 15, 64)
senior_population = calculate_age_group(total_df, 65, 100)

largest_age_row = total_df.loc[total_df["인구수"].idxmax()]
largest_age = largest_age_row["나이"]
largest_age_population = int(largest_age_row["인구수"])


st.subheader("2. 주요 인구 지표")

metric_columns = st.columns(5)

with metric_columns[0]:
    st.metric(
        "총인구",
        f"{total_population:,}명",
    )

with metric_columns[1]:
    st.metric(
        "남성",
        f"{male_population:,}명",
        f"{male_population / total_population * 100:.1f}%"
        if total_population > 0
        else None,
    )

with metric_columns[2]:
    st.metric(
        "여성",
        f"{female_population:,}명",
        f"{female_population / total_population * 100:.1f}%"
        if total_population > 0
        else None,
    )

with metric_columns[3]:
    st.metric(
        "65세 이상",
        f"{senior_population:,}명",
        f"{senior_population / total_population * 100:.1f}%"
        if total_population > 0
        else None,
    )

with metric_columns[4]:
    st.metric(
        "인구가 가장 많은 나이",
        largest_age,
        f"{largest_age_population:,}명",
    )


# ---------------------------------------------------------
# Plotly 꺾은선 그래프
# ---------------------------------------------------------
st.subheader("3. 연령별 인구 구조")

figure = make_line_chart(
    chart_df=chart_df,
    selected_region=selected_region,
    display_mode=display_mode,
)

st.plotly_chart(
    figure,
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


# ---------------------------------------------------------
# 연령대별 요약
# ---------------------------------------------------------
st.subheader("4. 연령대별 요약")

summary_columns = st.columns(3)

with summary_columns[0]:
    st.metric(
        "유소년 인구",
        f"{youth_population:,}명",
        f"0~14세 · {youth_population / total_population * 100:.1f}%"
        if total_population > 0
        else "0~14세",
    )

with summary_columns[1]:
    st.metric(
        "생산연령 인구",
        f"{working_population:,}명",
        f"15~64세 · {working_population / total_population * 100:.1f}%"
        if total_population > 0
        else "15~64세",
    )

with summary_columns[2]:
    st.metric(
        "고령 인구",
        f"{senior_population:,}명",
        f"65세 이상 · {senior_population / total_population * 100:.1f}%"
        if total_population > 0
        else "65세 이상",
    )


# ---------------------------------------------------------
# 상세 데이터
# ---------------------------------------------------------
with st.expander("연령별 상세 데이터 보기"):
    detail_df = pd.DataFrame(
        {
            "나이": total_df["나이"],
            "전체": total_df["인구수"],
            "남성": male_df["인구수"],
            "여성": female_df["인구수"],
        }
    )

    detail_df["남성 비율(%)"] = (
        detail_df["남성"] / detail_df["전체"] * 100
    ).fillna(0).round(2)

    detail_df["여성 비율(%)"] = (
        detail_df["여성"] / detail_df["전체"] * 100
    ).fillna(0).round(2)

    st.dataframe(
        detail_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "전체": st.column_config.NumberColumn(format="%d명"),
            "남성": st.column_config.NumberColumn(format="%d명"),
            "여성": st.column_config.NumberColumn(format="%d명"),
            "남성 비율(%)": st.column_config.NumberColumn(format="%.2f%%"),
            "여성 비율(%)": st.column_config.NumberColumn(format="%.2f%%"),
        },
    )

    csv_data = detail_df.to_csv(
        index=False,
        encoding="utf-8-sig",
    ).encode("utf-8-sig")

    safe_region_name = re.sub(
        r'[\\/:*?"<>|]',
        "_",
        selected_region,
    )

    st.download_button(
        label="선택 지역 데이터 CSV 다운로드",
        data=csv_data,
        file_name=f"{safe_region_name}_인구구조.csv",
        mime="text/csv",
    )
