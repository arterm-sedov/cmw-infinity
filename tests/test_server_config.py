"""Tests for cmw-infinity."""

from __future__ import annotations

import pytest

from cmw_infinity.server_config import (
    ModelRegistry,
    get_model_config,
    list_available_models,
)


def test_get_model_config_frida():
    """Test getting FRIDA config (case-insensitive slug)."""
    config = get_model_config("ai-forever/FRIDA")
    assert config.model_id == "ai-forever/FRIDA"
    assert config.model_type == "embedding"
    assert config.port == 7997
    config_lower = get_model_config("ai-forever/frida")
    assert config_lower.model_id == "ai-forever/FRIDA"


def test_get_model_config_dity():
    """Test getting DiTy reranker config (case-insensitive slug)."""
    config = get_model_config("DiTy/cross-encoder-russian-msmarco")
    assert config.model_id == "DiTy/cross-encoder-russian-msmarco"
    assert config.model_type == "reranker"
    assert config.port == 7998
    config_lower = get_model_config("dity/cross-encoder-russian-msmarco")
    assert config_lower.model_id == "DiTy/cross-encoder-russian-msmarco"


def test_get_model_config_unknown():
    """Test getting unknown model raises error."""
    with pytest.raises(ValueError, match="Unknown model"):
        get_model_config("unknown-model")


def test_list_available_models():
    """Test listing available models."""
    models = list_available_models()
    assert "embedding" in models
    assert "reranker" in models
    assert "ai-forever/FRIDA" in models["embedding"]
    assert "DiTy/cross-encoder-russian-msmarco" in models["reranker"]


def test_to_infinity_args():
    """Test converting config to CLI args."""
    config = get_model_config("ai-forever/FRIDA")
    args = config.to_infinity_args()

    assert "v2" in args
    assert "--model-name-or-path" in args
    assert "ai-forever/FRIDA" in args
    assert "--port" in args
    assert "7997" in args


def test_port_validation():
    """Test port range validation."""
    from pydantic import ValidationError
    from cmw_infinity.server_config import InfinityModelConfig

    # Valid port
    config = InfinityModelConfig(
        model_id="test/model",
        model_type="embedding",
        port=8000,
        memory_gb=4.0,
    )
    assert config.port == 8000

    # Invalid port (too low)
    with pytest.raises(ValidationError):
        InfinityModelConfig(
            model_id="test/model",
            model_type="embedding",
            port=1000,  # Too low
            memory_gb=4.0,
        )


def test_all_embedding_models():
    """Test all embedding models have required fields."""
    registry = ModelRegistry()
    for slug in registry.list_embeddings():
        config = registry.get_embedding_config(slug)
        assert config.model_id
        assert config.model_type == "embedding"
        assert config.port > 0
        assert config.memory_gb > 0


def test_all_reranker_models():
    """Test all reranker models have required fields."""
    registry = ModelRegistry()
    for slug in registry.list_rerankers():
        config = registry.get_reranker_config(slug)
        assert config.model_id
        assert config.model_type == "reranker"
        assert config.port > 0
        assert config.memory_gb > 0
