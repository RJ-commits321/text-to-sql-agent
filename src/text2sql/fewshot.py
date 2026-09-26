"""Dynamic few-shot retrieval.

For a given question, find the most similar solved (question, SQL) pairs from the
Spider training split, using local nomic-embed-text embeddings via Ollama. The
retrieved pairs are injected into the prompt as few-shot examples.

The index is built once (eval/build_fewshot_index.py) and cached on load.
"""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

EMBED_MODEL = "nomic-embed-text"


def _embed(texts: list[str], batch: int = 128) -> np.ndarray:
    """Embed texts with nomic-embed-text and L2-normalize (for cosine via dot product)."""
    import ollama

    vecs: list[list[float]] = []
    for i in range(0, len(texts), batch):
        resp = ollama.embed(model=EMBED_MODEL, input=texts[i : i + batch])
        vecs.extend(resp["embeddings"])
    arr = np.asarray(vecs, dtype=np.float32)
    arr /= np.linalg.norm(arr, axis=1, keepdims=True) + 1e-8
    return arr


def build_index(train_json: str, out_path: str) -> int:
    items = json.loads(Path(train_json).read_text(encoding="utf-8"))
    questions = [i["question"] for i in items]
    queries = [i["query"] for i in items]
    vectors = _embed(questions)
    np.savez(
        out_path,
        vectors=vectors,
        questions=np.array(questions, dtype=object),
        queries=np.array(queries, dtype=object),
    )
    return len(items)


@lru_cache(maxsize=2)
def _load(path: str):
    data = np.load(path, allow_pickle=True)
    return data["vectors"], data["questions"], data["queries"]


def retrieve(question: str, index_path: str, k: int = 3) -> list[tuple[str, str]]:
    """Return the top-k most similar (question, SQL) pairs for the question."""
    return retrieve_batch([question], index_path, k)[0]


def retrieve_batch(questions: list[str], index_path: str, k: int = 3) -> list[list[tuple[str, str]]]:
    """Retrieve top-k examples for many questions in one embedding pass.

    Used by the eval harness so all embedding happens up front (embed model only),
    before the SQL model runs — avoids two models thrashing in tight RAM.
    """
    vectors, all_q, all_sql = _load(index_path)
    qvs = _embed(questions)  # (M, D), normalized
    sims = qvs @ vectors.T  # (M, N)
    out = []
    for row in sims:
        top = np.argsort(-row)[:k]
        out.append([(str(all_q[i]), str(all_sql[i])) for i in top])
    return out
