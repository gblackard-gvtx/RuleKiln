"""Fake embedding provider for offline testing."""

import hashlib
import math

from rulekiln.providers.contracts import EmbeddingClient, ProviderConfig

_FAKE_DIM = 384


class FakeEmbeddingClient(EmbeddingClient):
    """Returns deterministic pseudo-embeddings derived from text hashes."""

    async def embed_texts(
        self,
        *,
        texts: list[str],
        config: ProviderConfig,
    ) -> list[list[float]]:
        results: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode()).digest()
            # Expand 32 bytes into _FAKE_DIM floats using sine for spread
            vec: list[float] = []
            for i in range(_FAKE_DIM):
                byte_val = digest[i % 32]
                vec.append(math.sin(byte_val + i) * 0.5)
            results.append(vec)
        return results
