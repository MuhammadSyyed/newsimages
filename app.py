"""Streamlit viewer for per-article retrieval images across FAST-DS submission runs."""

from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent
ARTICLES_CSV = ROOT / "dataset/newsimages_test_and_evaluation_26_v1.0/news_articles_test.csv"
SUBMISSION_DIR = ROOT / "output/FAST-DS_Submission"
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}


@st.cache_data
def load_articles() -> pd.DataFrame:
    df = pd.read_csv(ARTICLES_CSV, encoding="latin1")
    df = df[["article_id", "article_title"]].drop_duplicates(subset=["article_id"])
    df = df.sort_values("article_id")
    df["article_title"] = df["article_title"].astype(str).str.strip()
    return df.reset_index(drop=True)


@st.cache_data
def discover_model_dirs(submission_dir: str) -> tuple[str, ...]:
    root = Path(submission_dir)
    if not root.is_dir():
        return ()
    return tuple(
        sorted(
            p.name
            for p in root.iterdir()
            if p.is_dir() and not p.name.startswith(".")
        )
    )


def image_for_article(model_dir: Path, article_id: int) -> Path | None:
    prefix = f"{article_id}_"
    matches = sorted(
        p
        for p in model_dir.iterdir()
        if p.is_file()
        and p.suffix.lower() in IMAGE_EXTENSIONS
        and p.name.startswith(prefix)
    )
    return matches[0] if matches else None


def format_article_label(row: pd.Series) -> str:
    title = row["article_title"]
    if len(title) > 80:
        title = title[:77] + "..."
    return f"{row['article_id']} — {title}"


st.set_page_config(
    page_title="FAST-DS retrieval viewer",
    page_icon="📰",
    layout="wide",
)

st.title("FAST-DS retrieval viewer")
st.caption(
    "Compare images each run retrieved for a test article. "
    f"Source: `{ARTICLES_CSV.relative_to(ROOT)}`"
)

if not SUBMISSION_DIR.is_dir():
    st.error(f"Submission folder not found: `{SUBMISSION_DIR}`")
    st.stop()

articles = load_articles()
model_names = discover_model_dirs(str(SUBMISSION_DIR))

if not model_names:
    st.warning(f"No model subfolders found under `{SUBMISSION_DIR}`.")
    st.stop()

labels = [format_article_label(row) for _, row in articles.iterrows()]
id_by_label = dict(zip(labels, articles["article_id"], strict=True))

with st.sidebar:
    st.header("Articles")
    search = st.text_input("Filter by title or ID", placeholder="e.g. 8501 or Apollo")
    filtered_labels = labels
    if search.strip():
        q = search.strip().lower()
        filtered_labels = [
            label
            for label in labels
            if q in label.lower()
        ]
    if not filtered_labels:
        st.info("No articles match your filter.")
        st.stop()

    selected_label = st.selectbox(
        "Select article",
        filtered_labels,
        index=0,
    )
    st.divider()
    st.subheader("Article list")
    list_df = articles.copy()
    list_df["label"] = labels
    if search.strip():
        q = search.strip().lower()
        list_df = list_df[
            list_df["label"].str.lower().str.contains(q, regex=False)
            | list_df["article_id"].astype(str).str.contains(q, regex=False)
        ]
    st.dataframe(
        list_df[["article_id", "article_title"]],
        use_container_width=True,
        height=400,
        hide_index=True,
    )

article_id = int(id_by_label[selected_label])
row = articles.loc[articles["article_id"] == article_id].iloc[0]

st.subheader(f"Article {article_id}")
st.write(row["article_title"])

cols = st.columns(len(model_names))
found_any = False

for col, model_name in zip(cols, model_names, strict=True):
    model_dir = SUBMISSION_DIR / model_name
    image_path = image_for_article(model_dir, article_id)
    with col:
        st.markdown(f"**{model_name}**")
        if image_path is None:
            st.warning("No image")
        else:
            found_any = True
            st.image(str(image_path), use_container_width=True)
            st.caption(image_path.name)

if not found_any:
    st.info(
        f"No images with prefix `{article_id}_` were found in any model folder."
    )
