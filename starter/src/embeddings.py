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


class OpenRouterEmbedder:
    """Embedder using OpenRouter embeddings API."""

    def __init__(self, model_name: str = "text-embedding-3-small", api_key: str | None = None) -> None:
        self.model_name = model_name
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise ValueError("OPENROUTER_API_KEY must be provided via argument or environment variable.")
        self.api_url = "https://openrouter.ai/api/v1/embeddings"
        
        # Determine dimensions via probe
        probe_res = self.embed_query("probe")
        self.dimensions = len(probe_res)

    def _call_api(self, input_texts: list[str]) -> list[list[float]]:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": self.model_name,
            "input": input_texts
        }
        for attempt in range(4):
            try:
                resp = requests.post(self.api_url, json=payload, headers=headers, timeout=60)
                if resp.status_code == 200:
                    data = resp.json()
                    # Return embeddings ordered by index
                    sorted_data = sorted(data["data"], key=lambda x: x["index"])
                    return [item["embedding"] for item in sorted_data]
                elif resp.status_code == 429:
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
            batch = texts[i:i + batch_size]
            embeddings = self._call_api(batch)
            results.extend(embeddings)
        return results

    def embed_query(self, question: str) -> list[float]:
        res = self._call_api([question])
        return res[0]


class MyEmbedder(OpenRouterEmbedder):
    """Default implementation aliasing OpenRouterEmbedder."""
    pass
