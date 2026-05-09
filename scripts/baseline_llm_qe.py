import os
import sys
import time
import numpy as np
import pandas as pd
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)
from utils import utils, models,logger

def baseline_llm_qe(model_name, device, batch_size):

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

    geberation_clip = models.QERetrieval(model,image_embeddings,generator,device,"../dataset/q-title-to-expanded-mapping.json")

    mrr = utils.compute_mrr(geberation_clip, data)
    queries = list(ground_truth.keys())
    metrics = utils.evaluate(geberation_clip, queries, ground_truth)
    metrics[f"MRR"] = mrr
    end = round((time.time() - start)/60, 2)

    logger.log_experiment(
        "../logs/logs.json",
        "Baseline with LLM-QE",
        model_name +"CLIP "+ "Phi3 LLM-QE" ,
        "Evaluation",
        config={"query_expansion": True},
        hyperparameters={"beta": 0.2},
        metrics=metrics,
        execution_time_mins=end
    )



if __name__ == "__main__":
    device = utils.get_device()
    print(f"Using device: {device}")
    batch_size = 64
    model_name = 'ViT-L/14'
    baseline_llm_qe(model_name, device, batch_size)