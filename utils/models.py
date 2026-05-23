import numpy as np
from utils import utils
import ollama
from openai import OpenAI
import torch
import os
import sys
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)

from openai import OpenAI


class OpenAIPipeline:

    def __init__(self, model="gpt-5-mini"):

        self.client = OpenAI()

        self.model = model

    def __call__(

        self,

        prompt,

        max_new_tokens=120,

        temperature=0.2,

        reasoning_effort="minimal"

    ):

        response = self.client.chat.completions.create(

            model=self.model,

            messages=[

                {

                    "role": "user",

                    "content": prompt

                }

            ],

            max_completion_tokens=max_new_tokens,
            reasoning_effort=reasoning_effort

        )

        return response.choices[0].message.content.strip()

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

class CLIPRetrieval:
    def __init__(self, model, image_embeddings, device):
        self.model = model
        self.image_embeddings = image_embeddings
        self.device = device

    def retrieve(self, query, k):
        with torch.no_grad():
            text_emb = utils.encode_texts([query], self.model, self.device)[0]
        scores = self.image_embeddings @ text_emb
        ranked_ids = np.argsort(-scores)
        return ranked_ids[:k]

class RerankRetrieval:
    def __init__(self, retriever, reranker, reranker_image_embeddings, device, k=50):
        self.retriever = retriever
        self.reranker = reranker
        self.reranker_image_embeddings = reranker_image_embeddings
        self.device = device
        self.k = k

    def retrieve(self, query, k_final):
        # retrieval
        candidates = self.retriever.retrieve(query, self.k)

        with torch.no_grad():
            q_emb = utils.encode_texts(
                [query], self.reranker, self.device)[0]

        # normalization
        norm = np.linalg.norm(q_emb)
        if norm == 0:
            return candidates[:k_final]
        q_emb = q_emb / norm

        candidate_embs = self.reranker_image_embeddings[candidates]

        assert len(self.reranker_image_embeddings) >= np.max(candidates)

        scores = candidate_embs @ q_emb

        reranked_idx = np.argsort(-scores)
        reranked_candidates = candidates[reranked_idx]

        return reranked_candidates[:k_final]

class QERetrieval(CLIPRetrieval):
    def __init__(self, model, image_embeddings, generator, device, cache_path):
        super().__init__(model, image_embeddings, device)

        self.generator = generator
        self.cache_path = cache_path
        self.cached_titles = utils.load_json(cache_path)
        self.image_embeddings = self.image_embeddings / np.linalg.norm(
            self.image_embeddings, axis=1, keepdims=True
        )

    def retrieve(self, query, k):

        q = utils.encode_texts([query], self.model, self.device)[0]
        q = q / np.linalg.norm(q)

        if query in self.cached_titles:
            # print("cached")
            expanded = self.cached_titles[query]
        else:
            expanded = utils.clean_prompt(
                utils.generate_expansion(self.generator, query)
            )

            self.cached_titles[query] = expanded
            utils.save_json(self.cached_titles, self.cache_path)

        q_exp = utils.encode_texts([expanded], self.model, self.device)[0]
        q_exp = q_exp / np.linalg.norm(q_exp)

        sim = np.dot(q, q_exp)
        if sim < 0.2:
            q_final = q
        else:
            beta = 0.2
            q_final = (1 - beta) * q + beta * q_exp
            q_final = q_final / np.linalg.norm(q_final)

        scores = self.image_embeddings @ q_final
        ranked_ids = np.argsort(-scores)

        return ranked_ids[:k]