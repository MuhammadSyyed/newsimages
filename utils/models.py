import numpy as np
from utils import utils
import ollama
import torch
import os
import sys
root = os.path.abspath(os.path.join(os.getcwd(), '..'))
sys.path.append(root)


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

class FusionRetrieval(CLIPRetrieval):
    def __init__(self, model, image_embeddings, caption_embeddings, device, beta=0.2, sim_threshold=0.2):
        super().__init__(model, image_embeddings, device)

        # Normalize inputs
        image_embeddings = image_embeddings / \
            np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        caption_embeddings = caption_embeddings / \
            np.linalg.norm(caption_embeddings, axis=1, keepdims=True)

        fused_embeddings = []

        for i in range(len(image_embeddings)):
            img = image_embeddings[i]
            cap = caption_embeddings[i]

            sim = np.dot(img, cap)

            if sim > sim_threshold:
                emb = (1 - beta) * img + beta * cap
            else:
                emb = img

            emb = emb / (np.linalg.norm(emb) + 1e-8)
            fused_embeddings.append(emb)

        self.fused_embeddings = np.array(fused_embeddings)

    def retrieve(self, query, k):
        q = utils.encode_texts([query], self.model, self.device)[0]
        q = q / (np.linalg.norm(q) + 1e-8)

        scores = self.fused_embeddings @ q
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

class HybridNewsRetrieval:

    def __init__(
        self,

        # retrieval model
        retrieval_model,

        # reranker model
        reranker_model,

        # embeddings
        image_embeddings,
        caption_embeddings,
        reranker_image_embeddings,

        # llm qe
        generator,

        device,

        cache_path,

        # fusion params
        fusion_beta=0.2,
        qe_beta=0.25,

        # candidate retrieval
        candidate_k=100,

        # thresholds
        caption_sim_threshold=0.2,
        qe_sim_threshold=0.15,
    ):

        self.retrieval_model = retrieval_model
        self.reranker_model = reranker_model

        self.generator = generator

        self.device = device

        self.cache_path = cache_path

        self.cached_expansions = utils.load_json(
            cache_path
        )

        self.candidate_k = candidate_k

        self.qe_beta = qe_beta

        self.qe_sim_threshold = qe_sim_threshold


        self.image_embeddings = (
            image_embeddings /
            np.linalg.norm(
                image_embeddings,
                axis=1,
                keepdims=True
            )
        )

        self.caption_embeddings = (
            caption_embeddings /
            np.linalg.norm(
                caption_embeddings,
                axis=1,
                keepdims=True
            )
        )

        self.reranker_image_embeddings = (
            reranker_image_embeddings /
            np.linalg.norm(
                reranker_image_embeddings,
                axis=1,
                keepdims=True
            )
        )

        fused = []

        for img, cap in zip(
            self.image_embeddings,
            self.caption_embeddings
        ):

            sim = np.dot(img, cap)

            # adaptive fusion
            if sim > caption_sim_threshold:

                emb = (
                    (1 - fusion_beta) * img +
                    fusion_beta * cap
                )

            else:
                emb = img

            emb = emb / (
                np.linalg.norm(emb) + 1e-8
            )

            fused.append(emb)

        self.fused_embeddings = np.array(fused)

    def expand_query(self, query):

        if query in self.cached_expansions:

            expanded = self.cached_expansions[query]

        else:

            expanded = utils.clean_prompt(
                utils.generate_expansion(
                    self.generator,
                    query
                )
            )

            self.cached_expansions[query] = expanded

            utils.save_json(
                self.cached_expansions,
                self.cache_path
            )

        return expanded

    def build_query_embedding(self, query):

        # original query embedding
        q = utils.encode_texts(
            [query],
            self.retrieval_model,
            self.device
        )[0]

        q = q / (
            np.linalg.norm(q) + 1e-8
        )

        # expanded query
        expanded = self.expand_query(query)

        q_exp = utils.encode_texts(
            [expanded],
            self.retrieval_model,
            self.device
        )[0]

        q_exp = q_exp / (
            np.linalg.norm(q_exp) + 1e-8
        )

        # similarity
        sim = np.dot(q, q_exp)

        # adaptive blend
        if sim > self.qe_sim_threshold:

            alpha = min(
                self.qe_beta * sim * 2,
                0.5
            )

            q_final = (
                (1 - alpha) * q +
                alpha * q_exp
            )

            q_final = q_final / (
                np.linalg.norm(q_final) + 1e-8
            )

        else:
            q_final = q

        return q_final

    
    def retrieve(
        self,
        query,
        k_final=10
    ):
        # build query
        q_final = self.build_query_embedding(
            query
        )

        # retrieve candidates

        retrieval_scores = (
            self.fused_embeddings @ q_final
        )

        candidate_ids = np.argsort(
            -retrieval_scores
        )[:self.candidate_k]


        # reranking
        with torch.no_grad():

            q_rerank = utils.encode_texts(
                [query],
                self.reranker_model,
                self.device
            )[0]

        q_rerank = (
            q_rerank /
            (np.linalg.norm(q_rerank) + 1e-8)
        )

        candidate_embs = (
            self.reranker_image_embeddings[
                candidate_ids
            ]
        )

        rerank_scores = (
            candidate_embs @ q_rerank
        )

        reranked_idx = np.argsort(
            -rerank_scores
        )

        final_ids = candidate_ids[
            reranked_idx
        ]

        return final_ids[:k_final]