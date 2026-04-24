import html
import json
from pathlib import Path

import ftfy
import streamlit as st
from PIL import Image

DATA_PATH = (Path(__file__).resolve().parent / "." /
             "dataset" / "eval_dataset.json").resolve()
DATASET_DIR = DATA_PATH.parent
PROJECT_ROOT = DATA_PATH.parent.parent
APP_DIR = Path(__file__).resolve().parent


def resolve_image_path(raw):
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None
    p = Path(s).expanduser()
    try:
        if p.is_file():
            return p.resolve()
    except OSError:
        pass
    for base in (Path.cwd(), DATASET_DIR, PROJECT_ROOT, APP_DIR):
        try:
            c = (base / p).resolve()
        except (OSError, ValueError):
            continue
        try:
            if c.is_file():
                return c
        except OSError:
            continue
    return None


def normalize_candidate_labels(candidate):
    lab = candidate.get("labels")
    if lab is None or not isinstance(lab, dict):
        candidate["labels"] = {}


def fix_query_text(text):
    """Repair mojibake (e.g. UTF-8 read as Latin-1) in stored query strings."""
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    return ftfy.fix_text(text)


def _inject_sticky_query_css_once() -> None:
    if st.session_state.get("_dlp_sticky_query_css_injected"):
        return
    st.session_state._dlp_sticky_query_css_injected = True
    st.markdown(
        """
        <style>
        .main .block-container {
            max-width: 90rem;
            padding-left: 2rem;
            padding-right: 2rem;
        }
        .dlp-query-sticky {
            position: sticky;
            top: 0;
            z-index: 1000;
            margin: 0 0 1.25rem 0;
            padding: 0.85rem 1rem 0.95rem 1rem;
            background: var(--background-color, #ffffff);
            color: var(--text-color, #0f0f0f);
            border-bottom: 1px solid rgba(128, 128, 128, 0.32);
            box-shadow: 0 1px 4px rgba(0, 0, 0, 0.08);
        }
        .dlp-query-sticky-inner {
            max-width: 62rem;
            margin: 0 auto;
        }
        .dlp-cards-scroll {
            max-height: 68vh;
            overflow-y: auto;
            padding-right: 0.35rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sticky_query_header(query_text: str) -> None:
    _inject_sticky_query_css_once()
    safe = html.escape(query_text)
    st.markdown(
        f"""
        <div class="dlp-query-sticky">
            <div class="dlp-query-sticky-inner">
                <h3 style="margin:0;font-size:1.25rem;line-height:1.35;font-weight:600;">
                    Title: {safe}
                </h3>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def load_data():
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_data(data):
    with DATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def annotator_completed_query(query, annotator_id):
    for c in query.get("candidates", []):
        normalize_candidate_labels(c)
        v = c["labels"].get(annotator_id)
        if v not in (0, 1, 2):
            return False
    return True


def count_annotators_per_query(query):
    keys = set()
    for c in query.get("candidates", []):
        normalize_candidate_labels(c)
        keys.update(c["labels"].keys())
    n = 0
    for aid in keys:
        if annotator_completed_query(query, aid):
            n += 1
    return n


def get_next_query(queries, annotator_id):
    best_i = None
    best_count = None
    for i, q in enumerate(queries):
        if count_annotators_per_query(q) >= 3:
            continue
        if annotator_completed_query(q, annotator_id):
            continue
        c = count_annotators_per_query(q)
        if best_count is None or c < best_count:
            best_count = c
            best_i = i
    if best_i is None:
        return None, None
    return best_i, queries[best_i]


def render_progress(queries):
    total = len(queries)
    full = sum(1 for q in queries if count_annotators_per_query(q) >= 3)
    remaining = total - full
    st.write(f"Total queries: {total}")
    st.write(f"Queries fully annotated (3 annotators): {full}")
    st.write(f"Queries remaining: {remaining}")


st.title("Image Retrieval Evaluation")

if not DATA_PATH.exists():
    st.error(f"Dataset not found at {DATA_PATH}")
    st.stop()

data = load_data()
queries = data.get("queries", [])
if not queries:
    st.error("No queries found in the dataset.")
    st.stop()

annotator_id = st.text_input("annotator_id").strip()
if not annotator_id:
    st.warning("Enter annotator_id to continue.")
    st.stop()

if st.session_state.get("_annotator_id") != annotator_id:
    st.session_state["_annotator_id"] = annotator_id
    st.session_state["_assigned_idx"] = None

assigned_idx = st.session_state.get("_assigned_idx")
if assigned_idx is not None:
    if assigned_idx < 0 or assigned_idx >= len(queries):
        assigned_idx = None
        st.session_state["_assigned_idx"] = None
    else:
        q_chk = queries[assigned_idx]
        if count_annotators_per_query(q_chk) >= 3 or annotator_completed_query(
            q_chk, annotator_id
        ):
            assigned_idx = None
            st.session_state["_assigned_idx"] = None

if assigned_idx is None:
    nxt_i, _ = get_next_query(queries, annotator_id)
    if nxt_i is None:
        if all(count_annotators_per_query(q) >= 3 for q in queries):
            st.success("All queries are fully annotated")
        else:
            st.info("No queries are available for you to annotate.")
        render_progress(queries)
        st.stop()
    st.session_state["_assigned_idx"] = nxt_i
    assigned_idx = nxt_i

query = queries[assigned_idx]
qid = query.get("query_id", str(assigned_idx))

candidates = list(query.get("candidates", []))
if len(candidates) != 10:
    st.error(f"This query has {len(candidates)} candidates; expected 10.")
    st.stop()

n_cand = len(candidates)
with st.form(f"annotate_{annotator_id}_{qid}"):
    picks = []
    st.markdown('<div class="dlp-cards-scroll">', unsafe_allow_html=True)
    render_sticky_query_header(fix_query_text(query.get("query_text", "")))
    for i, candidate in enumerate(candidates):
        with st.container(border=True):
            col_img, col_ctrl = st.columns(
                [0.78, 0.22],
                vertical_alignment="top",
                gap="medium",
            )
            with col_img:
                img_path = resolve_image_path(candidate.get("image_path", ""))
                if img_path is not None:
                    try:
                        st.image(Image.open(img_path), width='stretch')
                    except (OSError, ValueError):
                        st.write("Image could not be opened")
                else:
                    st.write("Image not found")
                    st.caption(str(candidate.get("image_path", "")))
                cap = fix_query_text(candidate.get("caption", "") or "")
                if cap:
                    st.caption(cap)
            with col_ctrl:
                st.markdown(
                    f"**Image {i + 1} of {n_cand}** — rate relevance for this result"
                )
                picks.append(
                    st.radio(
                        "Relevance",
                        options=[0, 1, 2],
                        index=None,
                        label_visibility="collapsed",
                        horizontal=True,
                        format_func=lambda x: {
                            0: "0 (Low)",
                            1: "1 (Medium)",
                            2: "2 (High)",
                        }[x],
                        key=f"lbl_{annotator_id}_{qid}_{i}",
                    )
                )
    st.markdown("</div>", unsafe_allow_html=True)
    submitted = st.form_submit_button("Submit query", width='stretch', type="primary")

if submitted:
    if any(p is None for p in picks):
        st.error("Label all 10 images before submitting.")
    elif count_annotators_per_query(query) >= 3:
        st.error("This query already has 3 annotators.")
        st.session_state["_assigned_idx"] = None
        st.rerun()
    elif annotator_completed_query(query, annotator_id):
        st.error("You have already completed this query.")
        st.session_state["_assigned_idx"] = None
        st.rerun()
    else:
        for candidate, label in zip(candidates, picks):
            normalize_candidate_labels(candidate)
            candidate["labels"][annotator_id] = int(label)
        save_data(data)
        st.session_state["_assigned_idx"] = None
        st.success("Saved.")
        st.rerun()

render_progress(queries)