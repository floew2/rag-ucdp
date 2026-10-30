"""BGE-large embedding generation via sentence-transformers.

Wraps :class:`sentence_transformers.SentenceTransformer` so the rest of the
codebase can produce embeddings without depending on the library's API
surface directly. The model is lazy-loaded on first use to keep import time
fast and to make the module mockable in tests.

The default model is ``BAAI/bge-large-en-v1.5`` (1024-dim, 512-token window),
selected for retrieval quality and English coverage. UCDP source text is
predominantly English; if multilingual content becomes significant, switch to
``BAAI/bge-m3`` via the ``model_name`` parameter.
"""

from __future__ import annotations

import logging
from typing import Final

import numpy as np
from numpy.typing import NDArray
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

#: Default Sentence-Transformers model identifier.
DEFAULT_MODEL_NAME: Final[str] = "BAAI/bge-large-en-v1.5"

#: Default batch size for encoding. Tuned for a MacBook-class GPU/CPU.
DEFAULT_BATCH_SIZE: Final[int] = 32


class EmbeddingModel:
    """Lazy-loaded embedding model with a small, stable interface.

    The underlying ``SentenceTransformer`` is created on first call to
    :meth:`embed` or :attr:`dimension`, not at construction time. This keeps
    module import cheap and allows tests to substitute a mock model without
    triggering the ~1.3 GB BGE-large download.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        device: str | None = None,
    ) -> None:
        """Create the wrapper without loading the model.

        Args:
            model_name: Hugging Face model identifier.
            device: Optional device string ("cpu", "mps", "cuda"). If None,
                sentence-transformers picks automatically. Set "cpu" explicitly
                to avoid MPS out-of-memory errors on macOS.
        """
        self.model_name = model_name
        self.device = device
        self._model: SentenceTransformer | None = None

    @property
    def model(self) -> SentenceTransformer:
        """Return the underlying ``SentenceTransformer``, loading if needed."""
        if self._model is None:
            logger.info(
                "Loading embedding model %s on device=%s",
                self.model_name,
                self.device or "auto",
            )
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    @property
    def dimension(self) -> int:
        """Output embedding dimensionality (1024 for BGE-large-en-v1.5)."""
        # sentence-transformers >= 5 renamed the method; Intel Macs run an older version.
        get_dim = getattr(self.model, "get_embedding_dimension", None)
        dim = get_dim() if get_dim is not None else self.model.get_sentence_embedding_dimension()
        if dim is None:
            raise RuntimeError(f"Model {self.model_name} reports no embedding dimension.")
        return dim

    def embed(
        self,
        texts: list[str],
        batch_size: int = DEFAULT_BATCH_SIZE,
        show_progress: bool = False,
    ) -> NDArray[np.float32]:
        """Encode a list of texts into L2-normalized embeddings.

        L2 normalization means cosine similarity reduces to dot product, which
        is what ChromaDB expects when using its default ``cosine`` distance.

        Args:
            texts: Input strings. May be empty.
            batch_size: Number of texts processed per forward pass.
            show_progress: If True, sentence-transformers prints a tqdm bar.

        Returns:
            A ``(len(texts), dimension)`` float32 array. For an empty input,
            returns ``(0, dimension)``.
        """
        if not texts:
            return np.zeros((0, self.dimension), dtype=np.float32)

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=show_progress,
        )
        return embeddings.astype(np.float32)
