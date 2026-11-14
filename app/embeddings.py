from __future__ import annotations

import os
from threading import Lock
from typing import Iterable

MODEL_NAME = os.getenv("DOCUMIND_EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")
_model = None
_model_lock = Lock()


def get_embedding_model():
    """Load the local ONNX embedding model once, on the first indexing request."""
    global _model
    if _model is None:
        with _model_lock:
            if _model is None:
                try:
                    from fastembed import TextEmbedding
                except ImportError as exc:
                    raise RuntimeError(
                        "FastEmbed is not installed. Install the project requirements."
                    ) from exc
                try:
                    _model = TextEmbedding(model_name=MODEL_NAME)
                except Exception as exc:
                    raise RuntimeError(
                        "The local embedding model could not be loaded. Check model download access and cache storage."
                    ) from exc
    return _model


def _to_vectors(values: Iterable) -> list[list[float]]:
    return [[float(value) for value in vector] for vector in values]


def embed_passages(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    try:
        return _to_vectors(get_embedding_model().passage_embed(texts))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Passage embedding generation failed.") from exc


def embed_query(query: str) -> list[float]:
    try:
        vectors = _to_vectors(get_embedding_model().query_embed([query]))
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError("Query embedding generation failed.") from exc
    if len(vectors) != 1:
        raise RuntimeError("The embedding model returned an unexpected query result.")
    return vectors[0]
