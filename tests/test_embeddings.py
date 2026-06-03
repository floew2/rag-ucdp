"""Tests for the embedding model wrapper.

Uses a mock model to avoid downloading 1.3 GB of BGE-large weights in CI.
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from rag_ucdp.embeddings import DEFAULT_MODEL_NAME, EmbeddingModel


@pytest.fixture
def mock_st() -> MagicMock:
    """Build a mock SentenceTransformer with deterministic outputs."""
    mock = MagicMock()
    mock.get_sentence_embedding_dimension.return_value = 1024
    mock.encode.side_effect = lambda texts, **kwargs: np.ones((len(texts), 1024), dtype=np.float32)
    return mock


def test_lazy_loading_does_not_load_on_init() -> None:
    """Constructor must not trigger the model download."""
    model = EmbeddingModel()
    assert model._model is None
    assert model.model_name == DEFAULT_MODEL_NAME


def test_dimension_loads_model(mock_st: MagicMock) -> None:
    """Accessing .dimension triggers lazy load."""
    with patch("rag_ucdp.embeddings.SentenceTransformer", return_value=mock_st):
        model = EmbeddingModel()
        assert model.dimension == 1024
        assert model._model is mock_st


def test_embed_returns_correct_shape(mock_st: MagicMock) -> None:
    """Embedding N texts yields an (N, dim) array."""
    with patch("rag_ucdp.embeddings.SentenceTransformer", return_value=mock_st):
        model = EmbeddingModel()
        out = model.embed(["hello", "world", "foo"])
        assert out.shape == (3, 1024)
        assert out.dtype == np.float32


def test_embed_empty_input(mock_st: MagicMock) -> None:
    """Empty input returns a (0, dim) array, not an error."""
    with patch("rag_ucdp.embeddings.SentenceTransformer", return_value=mock_st):
        model = EmbeddingModel()
        out = model.embed([])
        assert out.shape == (0, 1024)
