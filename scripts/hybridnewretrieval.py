
import os
import sys
import time
import numpy as np
import pandas as pd
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)
from utils import utils, models, logger

def hybridnewretrieval(model_name, reranker_model_name, batch_size, k_reranking, device):

    start = time.time()

    model, preprocess = utils.load_model(model_name, device)

    dataset_dir = '../dataset/newsimages_test_and_evaluation_26_v1.0'

    images_dir = f'{dataset_dir}/news_images_evaluation'

    dataset = pd.read_csv(
        f"../dataset/evaluation_with_captions.csv", encoding="latin-1")

    dataset.set_index('article_id', inplace=True)

    dataset['article_title'] = dataset['article_title'].apply(
        utils.normalize_text)

    records = [
        {
            "image_path": utils.make_path(images_dir, row['image_id']),
            "title": row['article_title'],
            "caption": str(row['caption']) if pd.notna(row['caption']) else ""
        }
        for _, row in dataset.iterrows()

    ]

    captions = [r["caption"].strip() for r in records]

    image_paths = [r["image_path"] for r in records]

    model_path = f"../artifacts/{model_name.replace('/','-')}_embeddings_evaluation_images.npy"

    if os.path.exists(model_path):
        print("Using cached image embeddings")
        image_embeddings = np.load(model_path)

    else:
        image_embeddings = utils.encode_images(
            image_paths, model, preprocess, batch_size, device)
        np.save(model_path, image_embeddings)

    id_to_index = utils.build_id_to_index(image_paths)

    article_titles, ground_truth = utils.build_ground_truth(
        dataset, images_dir, id_to_index)
    data = [(q, img_id[0]) for q, img_id in ground_truth.items()]

    captions_embeddings_path = f"../artifacts/{model_name.replace('/','-')}_embeddings_evaluation_captions.npy"

    if os.path.exists(captions_embeddings_path):
        print("Using cached captions embeddings")
        caption_embeddings = np.load(captions_embeddings_path)
    else:
        caption_embeddings = utils.encode_texts(captions, model, device)
        np.save(captions_embeddings_path, caption_embeddings)

    generator = models.OllamaPipeline("phi3")

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

    allinone = models.HybridNewsRetrieval(model, clip_reranker, image_embeddings, caption_embeddings, reranker_image_embeddings,
                                          generator, device, "../dataset/q-title-to-expanded-mapping.json", 0.2, 0.25, k_reranking, 0.2, 0.15)

    print("Evaluating model")

    mrr = utils.compute_mrr(allinone, data)
    queries = list(ground_truth.keys())
    metrics = utils.evaluate(allinone, queries, ground_truth)
    metrics[f"MRR"] = mrr
    end = round((time.time() - start)/60, 2)

    logger.log_experiment(
        "../logs/logs.json",
        "Hybrid Model",
        "LLM-QE + Captions Fusion + CLIP Reranker",
        "Evaluation",
        config={
            "query_expansion": True,
            "fusion": True,
            "reranking": True
        },
        hyperparameters={"fusion_beta": 0.2,
                         "qe_beta":0.25,
                         "cap_sim_threshold":0.2,
                         "qe_sim_threshold": 0.15,
                         "top_k": k_reranking},
        metrics=metrics,
        execution_time_mins=end
    )

if __name__ == "__main__":
    model_name = 'ViT-L/14'
    reranker_model_name = "ViT-L/14@336px"
    batch_size = 64
    k_reranking = 100
    device = utils.get_device()
    hybridnewretrieval(model_name, reranker_model_name, batch_size, k_reranking, device)

    
