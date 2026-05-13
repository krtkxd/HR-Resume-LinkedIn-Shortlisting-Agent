"""
core/embedding.py
─────────────────
Embedding engine that supports multiple backends:
  - SentenceTransformers (local, free, default)
  - OpenAI text-embedding-ada-002
  - Gemini embedding-001

Provides LRU-cached embeddings to avoid redundant API calls.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from typing import List, Literal, Optional

import numpy as np


# ──────────────────────────────────────────────
# Backend type alias
# ──────────────────────────────────────────────
EmbeddingBackend = Literal["sentence_transformers", "openai", "gemini"]

# Global singleton model (lazy-loaded)
_st_model = None


def _get_st_model(model_name: str = "all-MiniLM-L6-v2"):
    """Lazy-load the SentenceTransformer model (singleton)."""
    global _st_model
    if _st_model is None:
        try:
            from sentence_transformers import SentenceTransformer
            _st_model = SentenceTransformer(model_name)
        except ImportError:
            raise ImportError(
                "sentence-transformers is not installed. "
                "Run: pip install sentence-transformers"
            )
    return _st_model


# ──────────────────────────────────────────────
# Public API
# ──────────────────────────────────────────────

class EmbeddingEngine:
    """
    Unified embedding interface.

    Usage:
        engine = EmbeddingEngine(backend="sentence_transformers")
        vec = engine.embed("Machine Learning Engineer with 5 years experience")
        vecs = engine.embed_batch(["text1", "text2"])
    """

    def __init__(
        self,
        backend: EmbeddingBackend = "sentence_transformers",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
    ):
        self.backend = backend
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")

        # Default model names per backend
        self._model_name = model_name or {
            "sentence_transformers": "all-MiniLM-L6-v2",
            "openai": "text-embedding-ada-002",
            "gemini": "models/embedding-001",
        }.get(backend, "all-MiniLM-L6-v2")

        self._cache: dict[str, np.ndarray] = {}

    # ── Cache helpers ──────────────────────────

    def _cache_key(self, text: str) -> str:
        return hashlib.md5(f"{self.backend}::{self._model_name}::{text}".encode()).hexdigest()

    # ── Core embedding methods ─────────────────

    def embed(self, text: str) -> np.ndarray:
        """Return a 1-D numpy embedding vector for the given text."""
        key = self._cache_key(text)
        if key in self._cache:
            return self._cache[key]

        vec = self._embed_single(text)
        self._cache[key] = vec
        return vec

    def embed_batch(self, texts: List[str]) -> List[np.ndarray]:
        """Return embeddings for multiple texts (batched for efficiency)."""
        uncached = [(i, t) for i, t in enumerate(texts) if self._cache_key(t) not in self._cache]

        if uncached:
            indices, raw_texts = zip(*uncached)
            vecs = self._embed_batch_raw(list(raw_texts))
            for idx, vec in zip(indices, vecs):
                key = self._cache_key(texts[idx])
                self._cache[key] = vec

        return [self._cache[self._cache_key(t)] for t in texts]

    # ── Backend dispatch ───────────────────────

    def _embed_single(self, text: str) -> np.ndarray:
        if self.backend == "sentence_transformers":
            model = _get_st_model(self._model_name)
            return model.encode(text, normalize_embeddings=True)

        elif self.backend == "openai":
            return self._openai_embed([text])[0]

        elif self.backend == "gemini":
            return self._gemini_embed([text])[0]

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    def _embed_batch_raw(self, texts: List[str]) -> List[np.ndarray]:
        if self.backend == "sentence_transformers":
            model = _get_st_model(self._model_name)
            return list(model.encode(texts, normalize_embeddings=True))

        elif self.backend == "openai":
            return self._openai_embed(texts)

        elif self.backend == "gemini":
            return self._gemini_embed(texts)

        else:
            raise ValueError(f"Unknown backend: {self.backend}")

    # ── OpenAI ────────────────────────────────

    def _openai_embed(self, texts: List[str]) -> List[np.ndarray]:
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(model=self._model_name, input=texts)
            return [np.array(d.embedding, dtype=np.float32) for d in response.data]
        except ImportError:
            raise ImportError("openai package not installed. Run: pip install openai")

    # ── Gemini ────────────────────────────────

    def _gemini_embed(self, texts: List[str]) -> List[np.ndarray]:
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            results = []
            for text in texts:
                result = genai.embed_content(
                    model=self._model_name,
                    content=text,
                    task_type="retrieval_document",
                )
                results.append(np.array(result["embedding"], dtype=np.float32))
            return results
        except ImportError:
            raise ImportError(
                "google-generativeai package not installed. "
                "Run: pip install google-generativeai"
            )
