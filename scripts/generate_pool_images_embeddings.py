
import os
import sys
import numpy as np
from glob import glob
import pandas as pd
import time
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)
from utils import utils, models, logger

def generate_embeddings(model_name, device, batch_size):
    model, preprocess = utils.load_model(model_name, device)
    dataset_dir = '../dataset'
    images_dir = f'{dataset_dir}/yfcc100m_50k'
    image_paths = glob(os.path.join(images_dir, '*'))
    image_embeddings_path = f"../artifacts/{model_name.replace('/','-')}_embeddings_pool_images.npy"
    if os.path.exists(image_embeddings_path):
        print("Using cached embeddings")
        image_embeddings = np.load(image_embeddings_path)
    else:
        image_embeddings = utils.encode_images(
            image_paths, model, preprocess, batch_size, device)
        np.save(image_embeddings_path, image_embeddings)


if __name__ == "__main__":

    model_name = 'ViT-L/14'
    device = utils.get_device()
    batch_size = 32
    generate_embeddings(model_name, device, batch_size)
