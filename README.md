NewsImages - MediaEval 2026
===========================

This repository contains the workflow material for the NewsImages task at
MediaEval 2026. The task is to recommend or generate one fitting thumbnail
image for each news article, using the article text as the main input.

The current workflow focuses on CLIP-based retrieval, caption-aware retrieval
experiments, reranking, and LLM-based query expansion. The code is organized
around notebooks and small utility modules rather than a single end-to-end
command-line pipeline.

Repository Layout
-----------------

- `dataset/` - MediaEval CSV metadata, query-expansion caches, caption-enriched
  CSVs, and local image folders.
  - `newsimages_train_26_v1.1/news_articles.csv`
  - `newsimages_train_26_v1.1/news_articles_2025.csv`
  - `newsimages_train_26_v1.1/news_images/`
  - `newsimages_test_and_evaluation_26_v1.0/news_articles_test.csv`
  - `newsimages_test_and_evaluation_26_v1.0/news_articles_evaluation.csv`
  - `newsimages_test_and_evaluation_26_v1.0/news_articles_combined.csv`
  - `newsimages_test_and_evaluation_26_v1.0/news_images_evaluation/`
  - `yfcc100m_50k/` if the local retrieval pool has been downloaded or
    generated.
- `notebooks/` - runnable experiment notebooks.
  - `generate_pool_images_embeddings.ipynb` builds image embeddings for a local
    image pool.
  - `retrieval_from_pool.ipynb` runs retrieval experiments from saved
    embeddings.
  - `phi3_llm_qe_pipeline.ipynb` performs LLM query expansion and retrieval.
- `utils/` - shared Python helpers.
  - `utils/utils.py` includes text normalization, CLIP model loading, image and
    text encoding, retrieval helpers, and evaluation metrics.
  - `utils/models.py` includes retrieval wrappers for plain CLIP retrieval,
    reranking, and query expansion.
- `guidelines.md` - local copy of the MediaEval 2026 NewsImages task and
  submission guidance.
- `papers/` - background papers used while developing the approach.
- `requirements.txt` - Python dependencies used for the notebooks and
  experiments.

Environment Setup
-----------------

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The dependency list includes PyTorch, OpenAI CLIP, FAISS, transformers,
sentence-transformers, Pillow, pandas, and notebook tooling. Install a
platform-specific PyTorch build first if your CUDA or MPS setup requires it.

For query expansion through `OllamaPipeline`, install Ollama separately and pull
the model used by the notebook, for example:

```bash
ollama pull phi3
```

Typical Workflow
----------------

1. Confirm the required CSVs and image folders are present under `dataset/`.
2. Use `notebooks/generate_pool_images_embeddings.ipynb` to encode the image
   pool.
3. Use `notebooks/retrieval_from_pool.ipynb` to run CLIP retrieval and inspect
   top candidates.
4. Use `notebooks/phi3_llm_qe_pipeline.ipynb` to generate or reuse cached query
   expansions, then rerun retrieval.
5. Save the final recommendation mapping from `article_id` to selected image.
6. Convert each selected image to a valid MediaEval PNG submission image.

The utility code normalizes CLIP embeddings before similarity scoring. If you
combine caption embeddings, expanded queries, or reranking scores, keep the
normalization step explicit so scores remain comparable.

Implemented Components
----------------------

- `CLIPRetrieval` retrieves image indices by text-image cosine similarity over a
  precomputed image embedding matrix.
- `RerankRetrieval` first retrieves a candidate set, then reranks those
  candidates with a second model and embedding matrix.
- `QERetrieval` expands an article title with an Ollama-backed LLM, caches the
  expanded query in JSON, and blends original and expanded query embeddings when
  their similarity is high enough.
- `utils.retrieve_candidates` and `utils.retrieve_candidate` provide lower-level
  retrieval helpers for notebooks.
- `utils.evaluate_clip_retrieval`, `precision_at_k`, and
  `build_ground_truth` support local sanity checks on train/evaluation data.

Reproducibility Notes
---------------------

- The notebooks assume local data paths under `dataset/`; update paths in the
  notebook if your image pools are stored elsewhere.
- `utils.get_device()` selects CUDA, then MPS, then CPU. CPU execution works but
  embedding generation may be slow.
- The local image folders are large and may be ignored by Git. Document where
  they came from and how to recreate or download them.
- Keep generated embeddings, final mappings, and submission ZIPs out of version
  control unless they are intentionally small artifacts.
- When reporting results in the Working Notes Paper, describe the exact model,
  retrieval pool, query expansion prompt/model, reranker, and any manual
  filtering used for each submitted run.

License
-------

See `LICENSE`.
