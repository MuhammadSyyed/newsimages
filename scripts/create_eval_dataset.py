import random
import pandas as pd
import numpy as np
from utils import utils
import os
import sys
root = os.path.abspath(os.path.join(os.getcwd(), '.'))
sys.path.append(root)


def deduplicate_queries(queries):
    seen = set()
    unique = []

    for q in queries:
        q_norm = q.lower().strip()
        if q_norm not in seen:
            seen.add(q_norm)
            unique.append(q)

    return unique

def stratified_sample(queries, n=100, seed=42):
    random.seed(seed)

    short = [q for q in queries if len(q.split()) <= 5]
    medium = [q for q in queries if 5 < len(q.split()) <= 10]
    long = [q for q in queries if len(q.split()) > 10]

    def safe_sample(lst, k):
        return random.sample(lst, min(len(lst), k))

    sampled = []
    sampled += safe_sample(short, n // 3)
    sampled += safe_sample(medium, n // 3)
    sampled += safe_sample(long, n - len(sampled))

    remaining = list(set(queries) - set(sampled))
    if len(sampled) < n:
        sampled += random.sample(remaining, n - len(sampled))

    return sampled

def retrieve_top_k(query, k, records, image_embeddings, text_encoder):
    q_emb = text_encoder(query)
    q_emb = q_emb / np.linalg.norm(q_emb)
    img_embs = image_embeddings / \
        np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    scores = img_embs @ q_emb
    top_indices = np.argsort(scores)[::-1][:k]
    results = []
    for rank, idx in enumerate(top_indices, start=1):
        results.append({
            "rank": rank,
            "image_id": records[idx].get("image_id", idx),
            "image_path": records[idx]["image_path"],
            "title": records[idx].get("title", ""),
            "score": float(scores[idx])
        })
    return results

if __name__ == "__main__":
    batch_size = 64
    model_name = 'ViT-B/16'
    device = utils.get_device()
    dataset_dir = '../dataset/newsimages_train_26_v1.1'
    model, preprocess = utils.load_model(model_name, device)
    images_dir = f'{dataset_dir}/news_images'
    dataset = pd.read_csv(f"{dataset_dir}/news_articles.csv")
    dataset.set_index('article_id', inplace=True)
    records = [
        {
            "image_path": utils.make_path(images_dir, row['image_id']),
            "title": row['article_title']
        }
        for _, row in dataset.iterrows()
    ]
    queries = utils.extract_queries(dataset)
    queries = deduplicate_queries(queries)
    queries = stratified_sample(queries)

    image_paths = [r["image_path"] for r in records]
    model_path = f"../artifacts/{model_name.replace('/','_').lower()}_image_embeddings.npy"
    if os.path.exists(model_path):
        image_embeddings = np.load(model_path)
    else:
        image_embeddings = utils.encode_images(
            image_paths, model, preprocess, batch_size, device)
        np.save(model_path, image_embeddings)

    eval_data = []

    def text_encoder(q): return utils.encode_texts([q], model, device)[0]
    
    for i, query in enumerate(queries):
        results = retrieve_top_k(
            query, 10, records, image_embeddings, text_encoder)

        candidates = []
        for rank, r in enumerate(results, start=1):
            candidates.append({
                "rank": rank,
                "image_id": int(r["image_id"]),
                "image_path": r["image_path"],
                "title": r["title"],
                "score": float(r["score"]),
                "label": None
            })

        eval_data.append({
            "query_id": f"q_{i:03d}",
            "query_text": query,
            "candidates": candidates
        })

    final_json = {
        "dataset_name": "news_image_eval_v1",
        "created_from": "clip_baseline_vitb16",
        "queries": eval_data
    }

    utils.save_json(final_json, "../dataset/unlabelled_eval_dataset.json")
