import ollama
import torch
import os,sys
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)
from utils import utils
import numpy as np

class CLIPRetrieval:
    def __init__(self, model, image_embeddings, preprocess, device):
        self.model = model
        self.image_embeddings = image_embeddings
        self.device = device

    def retrieve(self, query, k):
        with torch.no_grad():
            text_emb = utils.encode_texts([query], self.model, self.device)[0]

        # cosine similarity
        scores = self.image_embeddings @ text_emb

        # rank
        ranked_ids = np.argsort(-scores)

        return ranked_ids[:k]

class QERetrieval(CLIPRetrieval):
    def __init__(self, model, image_embeddings, generator, device, cache_path):
        super().__init__(model, image_embeddings, None, device)
        self.generator = generator
        self.cache_path = cache_path
        self.cached_titles = utils.load_json(cache_path)

    def retrieve(self, query, k):
        q = utils.encode_texts([query], self.model, self.device)[0]

        if query in self.cached_titles:
            expanded = self.cached_titles[query]
            print('expanded version cached')
        else:
            print('expanded version not found',query)
            expanded = utils.clean_prompt(
                utils.generate_expansion(self.generator, query)
            )

        q_exp = utils.encode_texts([expanded], self.model, self.device)[0]
        q_final = 0.5 * (q + q_exp)

        scores = self.image_embeddings @ q_final
        ranked_ids = np.argsort(-scores)

        return ranked_ids[:k]

class CaptionRetrieval(CLIPRetrieval):
    def __init__(self, model, image_embeddings, caption_embeddings, device, w=0.3):
        super().__init__(model, image_embeddings, None, device)
        self.caption_embeddings = caption_embeddings
        self.w = w

    def retrieve(self, query, k):
        q = utils.encode_texts([query], self.model, self.device)[0]

        clip_scores = self.image_embeddings @ q
        caption_scores = self.caption_embeddings @ q

        scores = clip_scores + self.w * caption_scores

        ranked_ids = np.argsort(-scores)

        return ranked_ids[:k]

class OllamaPipeline:
    def __init__(self, model="phi3"):
        self.client = ollama
        self.model = model

    def __call__(self, prompt, max_new_tokens=50, temperature=0.7):
        response = self.client.chat(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            options={
                "num_predict": max_new_tokens,
                "temperature": temperature,
            }
        )
        return response["message"]["content"].strip()