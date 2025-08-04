import streamlit as st
import pandas as pd
import os
import glob
import re
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import platform
import altair as alt
from collections import Counter
import io
import random
import base64
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# -----------------------
# 설정
# -----------------------
data_folder = r"C:\Users\Owner\Documents\python\data"
logo_path = os.path.join(data_folder, "Kepco_logo.png")

# -----------------------
# 사이드바 사용자 전환 UI
# -----------------------

# ----------------------------
# 사이드바 사용자 프로필 및 역할별 시각 구성
# ----------------------------
import streamlit as st
import os
import base64
import streamlit as st
import os
import base64

# 📁 사용자 이미지 경로
data_folder = r"C:\Users\Owner\Documents\python\data"

# 사용자 정보 정의
users = [
    {"name": "장지우", "role": "담당자", "image": "kim.jpg", "menu": ["데이터 확인", "점검표 보기"]},
    {"name": "나금영", "role": "책임자", "image": "lee.jpg", "menu": ["데이터 업로드", "통계 설정", "시스템 설정"]},
]

if "selected_user_idx" not in st.session_state:
    st.session_state.selected_user_idx = 0

if st.sidebar.button("사용자 전환"):
    st.session_state.selected_user_idx = (st.session_state.selected_user_idx + 1) % len(users)

current_user = users[st.session_state.selected_user_idx]
st.sidebar.markdown(f"⚙️ **현재 선택된 사용자:**<br>{current_user['name']} ({current_user['role']})", unsafe_allow_html=True)


# ✅ 스타일 삽입 (버튼처럼 보이게)
st.markdown("""
<style>
/* 🔷 사이드바 배경색 변경 */
section[data-testid="stSidebar"] {
    background-color: #e6f0fa !important;
}
.user-pic {
    width: 48px;
    height: 48px;
    border-radius: 50%;
    background-size: cover;
    background-color: #ccc;
    margin-right: 10px;
    border: 2px solid #4a4a4a;
}
.menu-box {
    background-color: #eeeeee;
    padding: 0.5em 1em;
    border-radius: 8px;
    margin-bottom: 6px;
    font-size: 0.95rem;
    font-weight: 500;
    cursor: default;
}
.role-label {
    font-size: 13px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ✅ 두 사용자 모두 표시
st.sidebar.markdown("## 👤 사용자 메뉴")

for user in users:
    name = user["name"]
    role = user["role"]
    menu = user["menu"]
    image_path = os.path.join(data_folder, user["image"])

    # base64 이미지 인코딩
    try:
        with open(image_path, "rb") as f:
            encoded_image = base64.b64encode(f.read()).decode()
    except FileNotFoundError:
        encoded_image = ""

    st.sidebar.markdown(f"""
    <div class='user-block'>
        <div class='user-pic' style="background-image:url('data:image/png;base64,{encoded_image}')"></div>
        <div>
            <strong>{name}</strong><br>
            <span class='role-label'>{role}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    for item in menu:
        st.sidebar.markdown(f"<div class='menu-box'>{item}</div>", unsafe_allow_html=True)

    st.sidebar.markdown("---")



# -----------------------
# 한글 폰트 설정
# -----------------------
if platform.system() == 'Windows':
    font_path = 'C:/Windows/Fonts/malgun.ttf'
elif platform.system() == 'Darwin':
    font_path = '/System/Library/Fonts/Supplemental/AppleGothic.ttf'
else:
    font_path = '/usr/share/fonts/truetype/nanum/NanumGothic.ttf'

font_name = fm.FontProperties(fname=font_path).get_name()
plt.rc('font', family=font_name)
plt.rcParams['axes.unicode_minus'] = False

# -----------------------
# 자동 분류 기준 설정
# -----------------------
chapter_keywords = {
    "1장 조직": ["조직", "책임", "권한", "협조", "독립성"],
    "2장 품질보증계획": ["품질보증계획", "qa 계획", "qap", "계획 수립"],
    "3장 설계관리": ["설계", "도면 검토", "설계 변경"],
    "4장 구매문서 관리": ["구매문서", "구매서류", "요구사항 전달"],
    "5장 지시서,절차서 및 도면": ["지시서", "절차서", "도면", "작업지시"],
    "6장 문서관리": ["문서", "개정", "통제"],
    "7장 구매품목 및 용역의 관리": ["구매품목", "수령검사", "용역", "입고"],
    "8장 품목의 식별 및 관리": ["식별", "추적성", "고유번호"],
    "9장 공정관리": ["공정", "작업공정", "공정변경"],
    "10장 검사": ["검사", "수입검사", "공정검사", "최종검사"],
    "11장 시험관리": ["시험", "성능시험", "검증시험"],
    "12장 측정 및 시험장비의 관리": ["시험장비", "계측기", "교정", "측정기"],
    "13장 취급, 저장 및 운송": ["취급", "보관", "운송", "저장"],
    "14장 검사, 시험 및 운전상태": ["운전상태", "운전 중", "시운전", "시운전 중 시험"],
    "15장 불일치품목의 관리": ["불일치", "부적합", "결함", "이탈"],
    "16장 시정조치": ["시정조치", "재발 방지", "개선조치"],
    "17장 품질보증기록": ["기록", "품질기록", "보존", "기록관리"],
    "18장 품질보증감사": ["감사", "품질감사", "내부감사"]
}

def classify_check_item(text):
    for chapter, keywords in chapter_keywords.items():
        for keyword in keywords:
            if re.search(keyword, text):
                return chapter
    return "기타"

def convert_to_check_item(text):
    text = text.strip()
    if len(text) < 5:
        return f"{text} 여부 확인."
    if text.endswith("하다") or text.endswith("합니다") or text.endswith("되어야 한다") or text.endswith("되어야 합니다"):
        base = re.sub(r'(하다|합니다|되어야 한다|되어야 합니다)$', '', text)
        return f"{base} 여부 확인."
    elif text.endswith("할 것") or text.endswith("할 것임"):
        base = re.sub(r'(할 것|할 것임)$', '', text)
        return f"{base} 점검."
    else:
        return f"{text} 여부 확인."


# -----------------------
# CSS 스타일
# -----------------------
st.markdown("""
<style>
/* 전체 배경 및 폰트 */
body {
    background-color: #f0f2f5;
    font-family: 'Nanum Gothic', sans-serif;
    color: #2f2f2f;
}

/* 일반 버튼 스타일 - 크기 소폭 증가 */
.stButton > button {
    background-color: #4a4a4a;
    color: white;
    font-weight: 600;
    border-radius: 32px;
    padding: 0.9em 2em;
    margin-bottom: 0.7em;
    font-size: 1.1rem;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.08);
    transition: background-color 0.3s ease, box-shadow 0.3s ease, transform 0.15s ease;
}

.stButton > button:hover {
    background-color: #2f2f2f;
    box-shadow: 0 4px 10px rgba(0, 0, 0, 0.15);
    transform: translateY(-1px);
    cursor: pointer;
}

/* 다운로드 버튼 - 크기 소폭 증가 */
.stDownloadButton > button {
    background-color: #6c757d !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 32px !important;
    padding: 0.85em 1.8em !important;
    font-size: 1.1rem !important;
    box-shadow: 0 2px 6px rgba(108, 117, 125, 0.12) !important;
    transition: background-color 0.3s ease, box-shadow 0.3s ease, transform 0.15s ease !important;
}

.stDownloadButton > button:hover {
    background-color: #5a6268 !important;
    box-shadow: 0 4px 10px rgba(90, 98, 104, 0.18) !important;
    transform: translateY(-1px);
    cursor: pointer;
}

/* 테이블 헤더 */
thead tr th {
    background-color: #4a4a4a !important;
    color: white !important;
}

/* 수평선 스타일 */
hr {
    border: 1.5px solid #4a4a4a;
    margin: 1em 0;
}
</style>
""", unsafe_allow_html=True)
# -----------------------
# 전역 함수 정의
# -----------------------
def normalize_project_name(name):
    if pd.isna(name):
        return ""
    name = str(name).strip()
    name = name.replace("-", ",").replace("–", ",").replace("—", ",")
    name = re.sub(r'([가-힣a-zA-Z])\s*([0-9])', r'\1 \2', name)
    name = name.replace("  ", " ")
    name = re.sub(r'\s*,\s*', ",", name)
    if re.search("새울.*(3.?4|3,4|3-4)", name):
        return "새울 3,4"
    return name

def extract_ngrams(text, n=2):
    words = text.split()
    return [' '.join(words[i:i+n]) for i in range(len(words)-n+1)]

def clean_text(text):
    text = str(text).strip()
    text = re.sub(r'\b(및|관련|의|에|에서|으로)\b', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def normalize_keyword(phrase):
    phrase = clean_text(phrase)
    replacements = {
        r'(일치 여부|여부 확인|확인 여부)': '여부 확인',
        r'(관련 요건.*|요건.*일치)': '요건 관련',
        r'(방지 교육|교육 및|교육 관련)': '교육 관련',
        r'(대상으로.*|.*대상으로)': '대상 지정',
        r'(재발 방지|재발.*)': '재발 방지',
        r'(담당자.*)': '담당자 관련',
    }
    for pattern, replacement in replacements.items():
        if re.search(pattern, phrase):
            return replacement
    return phrase

def convert_to_check_item(text):
    text = str(text).strip()
    if len(text) < 5:
        return f"{text} 여부 확인."
    if text.endswith("하다") or text.endswith("합니다") or text.endswith("되어야 한다") or text.endswith("되어야 합니다"):
        base = re.sub(r'(하다|합니다|되어야 한다|되어야 합니다)$', '', text)
        return f"{base} 여부 확인."
    elif text.endswith("할 것") or text.endswith("할 것임"):
        base = re.sub(r'(할 것|할 것임)$', '', text)
        return f"{base} 점검."
    else:
        return f"{text} 여부 확인."

def preprocess_excel(file):
    temp_df = pd.read_excel(file, nrows=1, header=None)
    if temp_df.iloc[0].isnull().all():
        df = pd.read_excel(file, header=1)
    else:
        df = pd.read_excel(file)

    year_col = [c for c in df.columns if "발행연도" in str(c)]
    if year_col:
        year_col = year_col[0]
        df[year_col] = pd.to_numeric(df[year_col], errors="coerce")
        df = df[(df[year_col] >= 2020) & (df[year_col] <= 2024)]
        df["연도"] = df[year_col]
    else:
        df["연도"] = np.nan

    if "납품사업소" not in df.columns:
        df["납품사업소"] = "새울 3,4"

    내용_col = [c for c in df.columns if "내용" in str(c)]
    if 내용_col:
        내용_col = 내용_col[0]
        df = df[~df[내용_col].astype(str).str.contains("보안 항목", na=False)]
        df["권고사항"] = df[내용_col]
    else:
        df["권고사항"] = ""

    df["지적유형"] = "감사/검사"
    codes = [f"A{str(i).zfill(2)}" for i in range(1, 71)] + [f"C{str(i).zfill(2)}" for i in range(1, 16)]
    df["위배 내용"] = [random.choice(codes) for _ in range(len(df))]
    df["출처파일"] = file.name
    df["정규화_사업명"] = df["납품사업소"].apply(normalize_project_name)

    if "등록일" in df.columns:
        df.loc[df["연도"].isna(), "연도"] = pd.to_datetime(df["등록일"], errors="coerce").dt.year
    if "검토일" in df.columns:
        df.loc[df["연도"].isna(), "연도"] = pd.to_datetime(df["검토일"], errors="coerce").dt.year

    return df

# ⚠️ 경고: 파일 미업로드 시 경고 메시지 추가
uploaded_csvs = st.file_uploader("CSV 파일 업로드", type="csv", accept_multiple_files=True)
uploaded_excels = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"], accept_multiple_files=True)

df = None
if uploaded_csvs:
    csv_dfs = []
    for csv in uploaded_csvs:
        for enc in ["cp949", "utf-8", "euc-kr"]:
            try:
                temp_df = pd.read_csv(csv, encoding=enc)
                temp_df["출처파일"] = csv.name
                csv_dfs.append(temp_df)
                break
            except:
                continue
    if csv_dfs:
        df = pd.concat(csv_dfs, ignore_index=True)

if uploaded_excels:
    excel_dfs = [preprocess_excel(excel) for excel in uploaded_excels]
    excel_df = pd.concat(excel_dfs, ignore_index=True)
    if df is not None:
        df = pd.concat([df, excel_df], ignore_index=True)
    else:
        df = excel_df

if not uploaded_csvs and not uploaded_excels:
    st.warning("📂 CSV 또는 엑셀 파일을 먼저 업로드해주세요.")

# -----------------------
# 파일 업로드 및 병합
# -----------------------
st.title("감사 통계 및 점검표 시스템")

uploaded_csvs = st.file_uploader("CSV 파일 업로드", type="csv", accept_multiple_files=True)
uploaded_excels = st.file_uploader("엑셀 파일 업로드", type=["xlsx", "xls"], accept_multiple_files=True)

df = None
if uploaded_csvs:
    csv_dfs = []
    for csv in uploaded_csvs:
        for enc in ["cp949", "utf-8", "euc-kr"]:
            try:
                temp_df = pd.read_csv(csv, encoding=enc)
                temp_df["출처파일"] = csv.name
                csv_dfs.append(temp_df)
                break
            except:
                continue
    if csv_dfs:
        df = pd.concat(csv_dfs, ignore_index=True)

if uploaded_excels:
    excel_dfs = [preprocess_excel(excel) for excel in uploaded_excels]
    excel_df = pd.concat(excel_dfs, ignore_index=True)
    if df is not None:
        df = pd.concat([df, excel_df], ignore_index=True)
    else:
        df = excel_df

if df is not None:
    st.success(f"📂 총 {len(df)}건 데이터 처리됨.")
    st.dataframe(df.head())


# -----------------------
# 타이틀
# -----------------------
# 중앙 타이틀
st.markdown("""
<h1 style='
    text-align: center;
    color: #2f2f2f;
    font-family: "Nanum Gothic", sans-serif;
    font-size: 36px;
    margin-bottom: 0.2em;
'>
사업별 감사 통계 및 점검표 생성 시스템
</h1>
""", unsafe_allow_html=True)

st.markdown("<hr>", unsafe_allow_html=True)  # 스타일에 맞는 구분선


# -----------------------
# 세션 초기화
# -----------------------
if 'selected_project' not in st.session_state:
    st.session_state.selected_project = None
if 'show_stats' not in st.session_state:
    st.session_state.show_stats = False
if 'show_checklist' not in st.session_state:
    st.session_state.show_checklist = False

# -----------------------
# 사업 선택
# -----------------------
st.markdown("## 사업 선택")
project_list = ["새울 3,4", "신한울 3,4", "신고리 5,6", "가동원전", "CTRF", "UAE"]
cols = st.columns(4)
for i, proj in enumerate(project_list):
    if cols[i % 4].button(proj):
        st.session_state.selected_project = proj
        st.session_state.show_stats = False
        st.session_state.show_checklist = False

# -----------------------
# 기능 버튼
# -----------------------
if st.session_state.selected_project:
    st.markdown(f"### 선택한 사업: {st.session_state.selected_project}")
    col_btn1, col_btn2 = st.columns(2)
    with col_btn1:
        if st.button("통계 보기"):
            st.session_state.show_stats = True
            st.session_state.show_checklist = False
    with col_btn2:
        if st.button("감사 점검표 생성하기"):
            st.session_state.show_stats = False
            st.session_state.show_checklist = True

# -----------------------
# 통계 시각화
# -----------------------

if st.session_state.show_stats:
    sub_df = df[df['정규화_사업명'] == st.session_state.selected_project]
    words = Counter()

    if '권고사항' in sub_df.columns:
        for text in sub_df['권고사항'].dropna().astype(str):
            text = re.sub(r'[^가-힣a-zA-Z0-9\s]', ' ', text)
            text = re.sub(r'\s+', ' ', text.strip())
            for n in [2, 3]:
                ngrams = extract_ngrams(text, n)
                for gram in ngrams:
                    normalized = normalize_keyword(gram)
                    if len(normalized) >= 4:
                        words[normalized] += 1

    if '지적유형' in sub_df.columns:
        st.markdown("#### 감사 유형 분포")
        counts = sub_df['지적유형'].value_counts().reset_index(name='횟수')
        counts.columns = ['지적유형', '횟수']
        chart = alt.Chart(counts).mark_bar().encode(
            x=alt.X('지적유형:N', sort='-y'),
            y='횟수:Q'
        ).properties(width=500, height=300)
        st.altair_chart(chart, use_container_width=False)

    if '연도' in sub_df.columns and '지적유형' in sub_df.columns:
        st.markdown("#### 연도별 감사유형 분포")
        yearly = sub_df.groupby(['연도', '지적유형']).size().reset_index(name='횟수')
        if not yearly.empty:
            chart = alt.Chart(yearly).mark_line(point=True).encode(
                x='연도:O',
                y='횟수:Q',
                color='지적유형:N'
            ).properties(width=500, height=300)
            st.altair_chart(chart, use_container_width=False)

    # 📌 위배 내용 2열 구성
    if '위배 내용' in sub_df.columns:
        st.markdown("### 위배 내용 분석")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### 위배사항 분포")
            vc = sub_df['위배 내용'].value_counts().nlargest(5)
            fig, ax = plt.subplots(figsize=(4, 4))
            ax.pie(vc, labels=vc.index, autopct='%1.1f%%', startangle=140, textprops={'fontsize': 8})
            ax.axis('equal')
            st.pyplot(fig)

        with col2:
            st.markdown("#### 상위 5개 항목")
            for i, (w, c) in enumerate(vc.items(), 1):
                st.markdown(f"{i}. {w} <span style='color:gray;'>({c}회)</span>", unsafe_allow_html=True)

    # 📌 권고사항 2열 구성
    if words:
        st.markdown("### 권고사항 분석")
        col3, col4 = st.columns(2)

        with col3:
            st.markdown("#### 키워드 분포")
            common = dict(words.most_common(10))
            fig, ax = plt.subplots(figsize=(5, 3))
            ax.bar(common.keys(), common.values(), color='#779ecb')
            ax.set_xticklabels(common.keys(), rotation=45, ha='right', fontsize=8)
            ax.tick_params(axis='y', labelsize=8)
            st.pyplot(fig)

        with col4:
            st.markdown("#### 상위 3개 키워드")
            for i, (w, c) in enumerate(words.most_common(3), 1):
                st.markdown(f"{i}. {w} <span style='color:gray;'>({c}회)</span>", unsafe_allow_html=True)

# -----------------------
# 감사 점검표 생성 (연도별 TF-IDF 매칭, 상위 20개 권고사항만 사용)
# -----------------------
if st.session_state.show_checklist:
    st.markdown("### 연도별 감사 점검표 보기")

    # ✅ 감사점검표 DB 파일 업로드
    uploaded_checklist_file = st.file_uploader("감사점검표 DB 업로드", type=["xlsx"], key="checklist_db")
    checklist_db = None
    if uploaded_checklist_file:
        checklist_db = pd.read_excel(uploaded_checklist_file)[["장", "점검항목"]].dropna()
        st.success("✅ 감사점검표 DB 파일이 업로드되었습니다.")

    # ✅ 연도별 버튼 생성
    year_cols = st.columns(4)
    for i, year in enumerate(range(2021, 2025)):
        if year_cols[i % 4].button(f"{year}년 감사점검표"):

            if checklist_db is None:
                st.warning("⚠️ 먼저 감사점검표 DB 파일을 업로드하세요.")
                continue

            # ✅ 기준 연도 이전(2020년 이상 ~ year-1년 이하)의 권고사항 사용
            check_df = df[(df["연도"] >= 2020) & (df["연도"] < year)]

            if check_df.empty:
                st.warning(f"{year}년 점검표를 생성할 권고사항 데이터가 없습니다.")
                continue

            # ✅ 권고사항 상위 20개 추출
            from collections import Counter
            counter = Counter(check_df["권고사항"].dropna().astype(str))
            top_titles = [t for t, c in counter.most_common(20)]

            if not top_titles:
                st.warning(f"{year}년 점검표를 생성할 권고사항 데이터가 없습니다.")
                continue

            # ✅ TF-IDF 매칭
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity

            vectorizer = TfidfVectorizer()
            tfidf = vectorizer.fit_transform(checklist_db["점검항목"].astype(str).tolist() + top_titles)

            checklist_vecs = tfidf[:len(checklist_db)]
            title_vecs = tfidf[len(checklist_db):]

            results = []
            for i_kw, title in enumerate(top_titles):
                sims = cosine_similarity(title_vecs[i_kw], checklist_vecs)[0]
                best_idx = sims.argmax()
                results.append({
                    "장": checklist_db.iloc[best_idx]["장"],
                    "점검항목 및 요건": checklist_db.iloc[best_idx]["점검항목"],
                    "근거": title,      # Streamlit에서만 표시
                    "점검 결과": "",    # 엑셀용
                    "비고": ""         # 엑셀용
                })

            checklist_df = pd.DataFrame(results)

            # ✅ 장 번호 기준으로 정렬
            checklist_df["장번호"] = checklist_df["장"].str.extract(r"(\d+)").astype(float)
            checklist_df = checklist_df.sort_values(by=["장번호"]).drop(columns=["장번호"])

            # ✅ Streamlit 화면 표시 (근거 강조, 인덱스 제거)
            display_df = checklist_df[["장", "점검항목 및 요건", "근거"]].reset_index(drop=True)
            styled_df = display_df.style.applymap(
                lambda _: "color: white; background-color: #1f77b4; font-weight: bold;",
                subset=["근거"]
            )

            st.markdown(
                f"""
                <h3 style="margin-bottom:0;">{year}년 감사점검표</h3>
                <p style="margin-top:0; font-size:14px; color:gray;">
                    기준: 2020년 ~ {year-1}년 권고사항 상위 20개
                </p>
                """,
                unsafe_allow_html=True
            )

            st.dataframe(styled_df, hide_index=True)

            # ✅ 엑셀 다운로드
            excel_df = checklist_df[["장", "점검항목 및 요건", "점검 결과", "비고"]]

            buffer = io.BytesIO()
            with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
                excel_df.to_excel(writer, index=False, sheet_name="감사점검표")

            st.download_button(
                label="엑셀로 다운로드",
                data=buffer.getvalue(),
                file_name=f"{st.session_state.selected_project}_{year}년_감사점검표.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                key=f"download_excel_{year}",
            )

# -----------------------
# 하단 고지
# -----------------------
st.markdown("""
---
<p style='text-align:center; font-size:13px; color:gray;'>
⚠️ 본 시스템은 개발 중이며, 한국수력원자력과 한국원자력안전기술원이 제공한 공공데이터를 기반으로 구현되었습니다.
</p>
""", unsafe_allow_html=True)

# -----------------------
# 하단 고정 로고 (항상 화면 하단 우측 고정)
# -----------------------
import base64

if os.path.exists(logo_path):
    with open(logo_path, "rb") as image_file:
        encoded_logo = base64.b64encode(image_file.read()).decode()

    st.markdown(f"""
        <div style='position: fixed; bottom: 20px; right: 30px; z-index: 999;'>
            <img src='data:image/png;base64,{encoded_logo}' width='100'/>
        </div>
    """, unsafe_allow_html=True)