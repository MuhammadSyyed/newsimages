import os
import clip
import json
import torch
import re
import html
import unicodedata
import ftfy
import math
import numpy as np
from tqdm import tqdm
from pathlib import Path
from PIL import Image
from IPython.display import display


def normalize_text(
    text,
    *,
    max_len=None,
    fix_mojibake=True,
    normalize_unicode=True,
    collapse_whitespace=True,

):

    if text is None:
        return ""
    if isinstance(text, float) and math.isnan(text):
        return ""
    s = str(text)
    if not s.strip():
        return ""
    if fix_mojibake:
        s = ftfy.fix_text(s)
    s = html.unescape(s)
    if normalize_unicode:
        s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u00a0", " ")
    if collapse_whitespace:
        s = re.sub(r"\s+", " ", s).strip()
    if max_len is not None and len(s) > max_len:
        s = s[: max_len - 1].rstrip() + "…"
    return s


def load_json(path):
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def get_device():
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def make_path(images_dir, image_id):
    return os.path.normpath(os.path.join(images_dir, image_id + ".jpg"))


def load_model(model_name, device):
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    return model, preprocess


def load_saved_model(path, device):
    try:
        checkpoint = torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(path, device=device)
    name = checkpoint["model_name"]
    model, preprocess = clip.load(name, device=device, jit=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model, preprocess


def encode_images(image_paths, model, preprocess, batch_size, device):
    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc="Encoding Images"):
            batch_paths = image_paths[i:i + batch_size]

            images = torch.stack([
                preprocess(Image.open(p).convert("RGB"))
                for p in batch_paths
            ]).to(device)

            emb = model.encode_image(images)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            embeddings.append(emb.cpu().numpy())

    embeddings = np.vstack(embeddings).astype("float32")

    embeddings = embeddings / np.linalg.norm(
        embeddings, axis=1, keepdims=True
    )

    return embeddings


def encode_texts(texts, model, device, batch_size=32):
    all_embeddings = []

    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]

            tokens = clip.tokenize(batch, truncate=True).to(device)

            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)

            all_embeddings.append(emb.cpu().numpy())

    return np.vstack(all_embeddings).astype("float32")


def retrieve_candidates(query, model, device, image_embeddings, records, caption_embeddings=None, top_k=10):

    query_emb = encode_texts([query], model, device)[0]

    query_emb = query_emb / np.linalg.norm(query_emb)

    scores = image_embeddings @ query_emb

    indices = np.argsort(-scores)[:top_k]

    candidates = []
    for idx in indices:
        idx = int(idx)

        candidates.append({
            "idx": idx,
            "image_path": records[idx]["image_path"],
            "title": records[idx]["title"],
            "clip_score": float(scores[idx]),
            "image_emb": image_embeddings[idx],
            "caption_emb": caption_embeddings[idx] if caption_embeddings else []
        })

    return candidates


def retrieve_candidate(query, model, device, image_embeddings, image_paths, top_k=1):

    query_emb = encode_texts([query], model, device)[0]

    query_emb = query_emb / np.linalg.norm(query_emb)

    scores = image_embeddings @ query_emb

    indices = np.argsort(-scores)[:top_k]

    candidates = []
    for idx in indices:
        idx = int(idx)

        candidates.append({
            "idx": idx,
            "image_path": image_paths[idx],
            "clip_score": float(scores[idx]),
            "image_emb": image_embeddings[idx]
        })

    return candidates


def precision_at_k(candidates, relevant_set, k=10):
    top_k = candidates[:k]

    hits = 0
    for c in top_k:
        if c["idx"] in relevant_set:
            hits += 1

    return hits / k


def evaluate_clip_retrieval(
    queries,
    model,
    device,
    image_embeddings,
    ground_truth,
    k_values=[1, 5, 10]
):
    results = {f"Recall@{k}": [] for k in k_values}

    for query in queries:
        try:
            query_emb = encode_texts([query], model, device)[0]
            query_emb = query_emb / np.linalg.norm(query_emb)

            scores = image_embeddings @ query_emb
            ranked_indices = np.argsort(-scores)

            gt = set(ground_truth.get(query, []))
            if len(gt) == 0:
                continue

            for k in k_values:
                top_k = ranked_indices[:k]
                hit = any(idx in gt for idx in top_k)
                results[f"Recall@{k}"].append(int(hit))

        except Exception as e:
            print(f"Skipping query: {query}", e)

    final = {
        metric: np.mean(vals) if len(vals) > 0 else 0
        for metric, vals in results.items()
    }

    return final


def build_id_to_index(image_paths):
    return {
        path: idx
        for idx, path in enumerate(image_paths)
    }


def build_ground_truth(dataset, images_dir, id_to_index):
    ground_truth = {}
    article_titles = []

    for article_id, group in dataset.groupby(level=0):

        query = group["article_title"].iloc[0]
        article_titles.append(query)

        relevant_indices = []

        for _, row in group.iterrows():
            img_path = make_path(images_dir, row["image_id"])

            if img_path in id_to_index:
                relevant_indices.append(id_to_index[img_path])

        ground_truth[query] = list(set(relevant_indices))

    return article_titles, ground_truth


def show_top_k_images(records, indices, k=10):
    for i in indices[:k]:
        candidate = records[i]
        img = Image.open(candidate['image_path'])
        display(img)


def process_candidate(c):
    labels = list(c["labels"].values())

    if len(labels) < 2:
        return None, None

    spread = max(labels) - min(labels)
    mean_rel = sum(labels) / len(labels)

    if spread == 0:
        weight = 1.0
    elif spread == 1:
        weight = 0.7
    else:
        weight = 0.3

    enriched = {
        **c,
        "rel": mean_rel,
        "spread": spread,
        "weight": weight
    }

    return enriched, spread


def build_eval_sets(input_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    clean_queries = []
    full_queries = []

    for q in data["queries"]:
        clean_candidates = []
        full_candidates = []

        for c in q["candidates"]:
            enriched, spread = process_candidate(c)

            if enriched is None:
                continue

            full_candidates.append(enriched)

            if spread <= 1:
                clean_candidates.append(enriched)

        if len(clean_candidates) >= 5:
            clean_queries.append({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "candidates": clean_candidates
            })

        if len(full_candidates) >= 3:
            full_queries.append({
                "query_id": q["query_id"],
                "query_text": q["query_text"],
                "candidates": full_candidates
            })

    return {"queries": clean_queries}, {"queries": full_queries}


def dcg_at_k(rels, k=5):
    dcg = 0.0
    for i in range(min(k, len(rels))):
        gain = (2**rels[i] - 1)
        dcg += gain / np.log2(i + 2)
    return dcg


def dcg_at_k_weighted(rels, weights, k=5):
    dcg = 0.0
    for i in range(min(k, len(rels))):
        gain = (2**rels[i] - 1) * weights[i]
        dcg += gain / np.log2(i + 2)
    return dcg


def ndcg_at_k(rels, weights, k=5):
    dcg = dcg_at_k_weighted(rels, weights, k)

    ideal_pairs = sorted(
        zip(rels, weights),
        key=lambda x: (x[0], x[1]),
        reverse=True
    )
    ideal_rels = [r for r, _ in ideal_pairs]
    ideal_weights = [w for _, w in ideal_pairs]

    idcg = dcg_at_k_weighted(ideal_rels, ideal_weights, k)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def mrr(rels):
    for i, rel in enumerate(rels):
        if rel >= 1:
            return 1.0 / (i + 1)
    return 0.0


def precision_at_k(rels, k=5):
    k = min(k, len(rels))
    relevant = sum(1 for r in rels[:k] if r >= 1)
    return relevant / k


def evaluate_model(model, eval_data, k=5):
    ndcgs, mrrs, p5s = [], [], []

    for q in eval_data["queries"]:
        query = q["query_text"]
        candidates = q["candidates"]

        image_paths = [c["image_path"] for c in candidates]

        scores = model.score(query, image_paths)

        ranked = sorted(zip(scores, candidates),
                        key=lambda x: x[0], reverse=True)

        rels = [c["rel"] for _, c in ranked]
        weights = [c["weight"] for _, c in ranked]

        ndcgs.append(ndcg_at_k(rels, weights, k))
        mrrs.append(mrr(rels))
        p5s.append(precision_at_k(rels, k))

    return {
        "nDCG@5": np.mean(ndcgs),
        "MRR": np.mean(mrrs),
        "P@5": np.mean(p5s)
    }


def score_clip(query, model, image_embeddings, image_paths, id_to_index):
    q_emb = encode_texts([query], model, get_device())
    q_emb = q_emb.squeeze()
    q_emb = q_emb / np.linalg.norm(q_emb)
    scores = []
    for path in image_paths:
        if path not in id_to_index:
            raise ValueError(f"Missing mapping for: {path}")
        idx = id_to_index[path]
        img_emb = image_embeddings[idx]
        img_emb = img_emb.squeeze()
        img_emb = img_emb / np.linalg.norm(img_emb)
        score = float(np.dot(q_emb, img_emb))
        scores.append(score)
    return scores


def extract_queries(df, column="article_title"):
    if column not in df.columns:
        raise ValueError(f"Column '{column}' not found in DataFrame")

    queries = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )

    queries = queries[queries.str.len() > 5]

    return queries.tolist()


def get_recall_scores(dataset, model, image_embeddings, images_dir, id_to_index, device):
    article_titles, ground_truth = build_ground_truth(
        dataset,
        images_dir,
        id_to_index
    )

    return evaluate_clip_retrieval(
        article_titles,
        model,
        device,
        image_embeddings,
        ground_truth
    )


def clean_prompt(text):
    if not isinstance(text, str):
        return text
    text = re.sub(r'---', '', text)
    text = re.sub(r'Prompt:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Setting:?', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Image Prompt:?', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\n+', ' ', text)

    return text.strip()


def generate_expansion(generator, title):
    prompt = f"""
You are a query rewriting system for image retrieval.

Rewrite the query by expanding it ONLY with closely related terms.

STRICT RULES:
- Do NOT introduce new events, facts, or assumptions
- Do NOT add causes, explanations, or background stories
- Preserve original meaning exactly
- Keep all original entities (countries, people, places)
- You may add synonyms or clarify ambiguous words
- Output must stay short (max 2 sentence)

Original query:
{title}

Rewritten query:
"""

    result = generator(
        prompt
    )

    return result

def generate_expansion_by_openai(generator, title):
    prompt = f"""
You are a query expansion system for vision-language image retrieval.

Your task is to conservatively expand a query using ONLY visually grounded and directly implied concepts.

GOAL:
Improve image retrieval recall while preserving the original query intent exactly.

STRICT RULES:
- The rewritten query must remain suitable for dense image retrieval embeddings.
- Preserve the original meaning exactly
- Keep all original named entities unchanged
- Do NOT introduce new events, facts, causes, or assumptions
- Do NOT generate stories, explanations, or inferred context
- Do NOT invent specific objects, equipment, people, poses, emotions, lighting, weather, or actions unless explicitly implied
- Avoid cinematic or descriptive language
- Avoid image-caption style outputs
- Prefer generic scene descriptors over detailed scene generation
- Add only short visually relevant terms such as:
  - environment
  - scene type
  - visible setting
  - generic activity
  - appearance synonyms
  - broad contextual nouns
- The output must remain suitable as a retrieval query
- Keep output concise
- Output only the rewritten query
- Do not use quotes
- Maximum length: 25 words

GOOD EXAMPLES:

Query:
Former President Jimmy Carter hospitalized

Good Output:
Former President Jimmy Carter hospitalized, hospital setting, medical care, healthcare environment

Bad Output:
elderly man in hospital bed with IV, doctors, nurses, oxygen tank, worried family members

---

Query:
Police officers during street protest

Good Output:
Police officers during street protest, crowd, urban street scene, public demonstration

Bad Output:
angry protesters throwing objects at police under smoky night conditions

---

Query:
Lion resting in grass

Good Output:
Lion resting in grass, wildlife, savanna, outdoor nature scene

Bad Output:
hungry lion preparing to attack prey at sunset

---

Original query:
{title}

Rewritten query:
"""

    result = generator(
        prompt,
        max_new_tokens=100,
        temperature=0.2
    )

    return result


def evaluate(model, queries, gt, k_list=[1, 5, 10]):
    results = {k: [] for k in k_list}

    for query in queries:
        ranked_ids = model.retrieve(query, max(k_list))
        gt_ids = gt[query]

        for k in k_list:
            top_k = ranked_ids[:k]
            hit = int(any(g in top_k for g in gt_ids))
            results[k].append(hit)

    return {f'Recall@{k}': round(float(np.mean(results[k])), 3) for k in k_list}


def compute_mrr(model, data, max_k=10):

    reciprocal_ranks = []

    for query, gt_id in data:
        ranked_ids = model.retrieve(query, max_k)

        rr = 0.0
        for rank, pred_id in enumerate(ranked_ids, start=1):
            if pred_id == gt_id:
                rr = 1.0 / rank
                break

        reciprocal_ranks.append(rr)

    return round(float(np.mean(reciprocal_ranks)), 3)


def resize_with_transparent_padding(
    img: Image.Image,
    target_w: int = 460,
    target_h: int = 260
):

    # Convert to RGBA for transparency support
    img = img.convert("RGBA")

    original_w, original_h = img.size

    # Preserve aspect ratio
    ratio = min(target_w / original_w, target_h / original_h)

    new_w = round(original_w * ratio)
    new_h = round(original_h * ratio)

    resized_img = img.resize(
        (new_w, new_h),
        Image.Resampling.LANCZOS
    )

    # Fully transparent canvas
    canvas = Image.new(
        "RGBA",
        (target_w, target_h),
        (0, 0, 0, 0)  # transparent
    )

    # Center placement
    paste_x = (target_w - new_w) // 2
    paste_y = (target_h - new_h) // 2

    canvas.paste(resized_img, (paste_x, paste_y))

    return canvas
