from utils import utils, models, logger
import os
import sys
import time
import torch
import numpy as np
import pandas as pd
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)

def baseline_llm_qe_reranker(model_name, reranker_model_name, device, batch_size, k_reranking):
    start = time.time()
    model, preprocess = utils.load_model(model_name, device)
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
    image_embeddings_path = f"../artifacts/{model_name.replace('/','-')}_embeddings_evaluation_images.npy"
    if os.path.exists(image_embeddings_path):
        print("Using cached embeddings")
        image_embeddings = np.load(image_embeddings_path)
    else:
        image_embeddings = utils.encode_images(
            image_paths, model, preprocess, batch_size, device)
        np.save(image_embeddings_path, image_embeddings)

    id_to_index = utils.build_id_to_index(image_paths)
    article_titles, ground_truth = utils.build_ground_truth(
        dataset, images_dir, id_to_index)
    data = [(q, img_id[0]) for q, img_id in ground_truth.items()]

    generator = models.OllamaPipeline("phi3")

    generation_clip = models.QERetrieval(
        model, image_embeddings, generator, device, "../dataset/q-title-to-expanded-mapping.json")

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

    reranker = models.RerankRetrieval(
        retriever=generation_clip,
        reranker=clip_reranker,
        reranker_image_embeddings=reranker_image_embeddings,
        device=device,
        k=k_reranking
    )

    mrr = utils.compute_mrr(reranker, data)
    queries = list(ground_truth.keys())
    metrics = utils.evaluate(reranker, queries, ground_truth)
    metrics[f"MRR"] = mrr
    end = round((time.time() - start)/60, 2)

    logger.log_experiment(
        "../logs/logs.json",
        "Baseline with LLM-QE Reranking",
        model_name + " " + "Phi3 LLM-QE" + "CLIP Retriever",
        "Evaluation",
        config={"query_expansion": True},
        hyperparameters={"beta": 0.2},
        metrics=metrics,
        execution_time_mins=end
    )

if __name__ == "__main__":
    device = utils.get_device()
    print(f"Using device: {device}")

    model_name = 'ViT-L/14'
    reranker_model_name = "ViT-L/14@336px"
    batch_size = 64
    k_reranking = 50

    baseline_llm_qe_reranker(model_name, reranker_model_name, device, batch_size, k_reranking)
