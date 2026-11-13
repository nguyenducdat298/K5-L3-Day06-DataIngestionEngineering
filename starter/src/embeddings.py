"""OpenRouter and sentence-transformers Embedder implementations."""
import os
import time
import requests
from typing import Protocol


class Embedder(Protocol):
    model_name: str
    dimensions: int

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        ...

    def embed_query(self, question: str) -> list[float]:
        ...


class SentenceTransformerEmbedder:
    """Local/Kaggle GPU embedder powered by sentence-transformers (e.g. embeddinggemma-2)."""

    def __init__(
        self,
        model_name: str = "/kaggle/input/models/google/embeddinggemma-2/transformers/embeddinggemma-2/1",
        device: str | None = None,
        batch_size: int = 32,
    ) -> None:
        from sentence_transformers import SentenceTransformer

        self.model_name = model_name
        self.batch_size = batch_size
        self.model = SentenceTransformer(model_name, device=device)
        dim = self.model.get_sentence_embedding_dimension()
        self.dimensions = int(dim) if dim is not None else len(self.embed_query("probe"))

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if hasattr(self.model, "encode_document"):
            vecs = self.model.encode_document(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vecs = self.model.encode(
                texts,
                batch_size=self.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return [v.tolist() for v in vecs]

    def embed_query(self, question: str) -> list[float]:
        if hasattr(self.model, "encode_query"):
            vec = self.model.encode_query(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        else:
            vec = self.model.encode(
                question,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        return vec.tolist()


class OpenRouterEmbedder:
    """Embedder using OpenRouter embeddings API."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided via argument or environment variable.")
        self.api_url = "https://openrouter.ai/api/v1/embeddings"

        probe_res = self.embed_query("probe")
        self.dimensions = len(probe_res)

    def _call_api(self, input_texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {"model": self.model_name, "input": input_texts}
        for attempt in range(4):
            try:
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    return [item["embedding"] for item in sorted_data]
                if resp.status_code == 429:
                    time.sleep(2 ** attempt + 1)
                else:
                    raise RuntimeError(f"OpenRouter API error {resp.status_code}: {resp.text}")
            except requests.RequestException as e:
                if attempt == 3:
                    raise e
                time.sleep(2 ** attempt + 1)
        raise RuntimeError("Failed to fetch embeddings from OpenRouter after retries.")

    def embed_documents(self, texts: list[str], batch_size: int = 32) -> list[list[float]]:
        results: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            results.extend(self._call_api(batch))
        return results

    def embed_query(self, question: str) -> list[float]:
        return self._call_api([question])[0]


class MyEmbedder(SentenceTransformerEmbedder):
    """Default implementation aliasing SentenceTransformerEmbedder."""
    pass
