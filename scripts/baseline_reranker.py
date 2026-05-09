import os
import sys
import numpy as np
import pandas as pd
import torch
import time
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)
from utils import utils, models, logger

def baseline_reranker(baseline_model, reranker_model_name, device, batch_size, k_reranking):
    start = time.time()

    model, preprocess = utils.load_model(baseline_model, device)

    dataset_dir = '../dataset/newsimages_test_and_evaluation_26_v1.0'

    images_dir = f'{dataset_dir}/news_images_evaluation'

    dataset = pd.read_csv(
        f"{dataset_dir}/news_articles_evaluation.csv", encoding="latin-1")

    dataset.set_index('article_id', inplace=True)

    dataset['article_title'] = dataset['article_title'].apply(
        utils.normalize_text)

    records = [
        {
            "image_path": utils.make_path(images_dir, row['image_id']),
            "title": row['article_title']
        }
        for _, row in dataset.iterrows()
    ]

    image_paths = [r["image_path"] for r in records]

    id_to_index = utils.build_id_to_index(image_paths)

    article_titles, ground_truth = utils.build_ground_truth(
        dataset, images_dir, id_to_index)

    data = [(q, img_id[0]) for q, img_id in ground_truth.items()]

    image_embeddings_path = f"../artifacts/{baseline_model.replace('/','-')}_embeddings_evaluation_images.npy"

    if os.path.exists(image_embeddings_path):
        print("Using cached embeddings")
        image_embeddings = np.load(image_embeddings_path)
    else:
        image_embeddings = utils.encode_images(
            image_paths, model, preprocess, batch_size, device)
        np.save(image_embeddings_path, image_embeddings)

    clip_reranker, reranker_preprocess = utils.load_model(
        reranker_model_name, device)

    reranker_embeddings_path = f"../artifacts/ViT-L-14@336px_embeddings_evaluation_images.npy"

    if os.path.exists(reranker_embeddings_path):
        reranker_image_embeddings = np.load(reranker_embeddings_path)
    else:
        print("Making reranker embeddings")
        reranker_image_embeddings = utils.encode_images(
            image_paths, clip_reranker, reranker_preprocess, batch_size, device)
        np.save(reranker_embeddings_path, reranker_image_embeddings)

    image_embeddings = image_embeddings / \
        np.linalg.norm(image_embeddings, axis=1, keepdims=True)

    reranker_image_embeddings = reranker_image_embeddings / \
        np.linalg.norm(reranker_image_embeddings, axis=1, keepdims=True)

    clip_retriever = models.CLIPRetrieval(model, image_embeddings, device)

    reranker = models.RerankRetrieval(
        retriever=clip_retriever,
        reranker=clip_reranker,
        reranker_image_embeddings=reranker_image_embeddings,
        device=device,
        k=k_reranking
    )

    queries = list(ground_truth.keys())
    q = queries[0]

    print("CLIP:", clip_retriever.retrieve(q, 5))
    print("Reranked:", reranker.retrieve(q, 5))

    print("Evaluating model")
    
    mrr_rerank = utils.compute_mrr(reranker, data)
    metrics = utils.evaluate(reranker, queries, ground_truth)
    metrics["MRR"] = mrr_rerank

    debug_ranks(clip_retriever, reranker, queries, ground_truth)

    end = round((time.time() - start)/60, 2)

    logger.log_experiment(
        "../logs/logs.json",
        "Baseline with reranker",
        baseline_model+" CLIP and "+ reranker_model_name + " Reranker",
        "Evaluation",
        config={
            "reranking": True},
        hyperparameters={
            "top_k_reranking": k_reranking
        },
        metrics=metrics,
        execution_time_mins=end
    )

def debug_ranks(model_before, model_after, queries, ground_truth):
    improved = 0
    worsened = 0
    missing_before = 0

    for q in queries:
        gt = ground_truth[q][0]

        before = model_before.retrieve(q, 100)
        after = model_after.retrieve(q, 100)

        rank_before = np.where(before == gt)[0]
        rank_after = np.where(after == gt)[0]

        if len(rank_before) == 0:
            missing_before += 1

        if len(rank_before) == 0 or len(rank_after) == 0:
            continue

        if rank_after[0] < rank_before[0]:
            improved += 1

        elif rank_after[0] > rank_before[0]:
            worsened += 1

    print(f"Improved: {improved}, Worsened: {worsened}")


if __name__ == "__main__":

    baseline_model = 'ViT-L/14'
    reranker_model_name = "ViT-L/14@336px"
    device = utils.get_device()
    batch_size = 64
    k_reranking = 100
    baseline_reranker(baseline_model, reranker_model_name, device, batch_size, k_reranking)

    
