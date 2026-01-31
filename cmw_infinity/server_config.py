"""Server configuration management for Infinity.

Supports case-insensitive model slug lookup with canonical normalization.
All model slugs are stored in HuggingFace format (e.g., "Qwen/Qwen3-Embedding-8B")
but can be looked up with any case variation (e.g., "qwen/qwen3-embedding-8b").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class InfinityModelConfig(BaseModel):
    """Server-side configuration for Infinity models."""

    model_id: str = Field(description="HuggingFace model ID")
    model_type: Literal["embedding", "reranker"] = Field(description="Model type")
    port: int = Field(description="Server port (must be unique per model)")
    device: str = Field(default="auto", description="Device (auto/cpu/cuda)")
    dtype: Literal["float16", "float32", "int8"] = Field(default="float16")
    batch_size: int = Field(default=32, description="Dynamic batching size")
    memory_gb: float = Field(description="Estimated VRAM usage in GB")

    @field_validator("port")
    def validate_port_range(cls, v: int) -> int:
        if not 7000 <= v <= 65535:
            raise ValueError("Port must be between 7000-65535")
        return v

    def to_infinity_args(self) -> list[str]:
        """Convert to infinity_emb CLI arguments (v2 API)."""
        args = [
            "v2",
            "--model-id",
            self.model_id,
            "--port",
            str(self.port),
            "--dtype",
            self.dtype,
            "--batch-size",
            str(self.batch_size),
        ]
        if self.device != "auto":
            args.extend(["--device", self.device])
        return args


class ServerStatus(BaseModel):
    """Status of a running Infinity server."""

    model_key: str = Field(description="Model identifier")
    model_id: str = Field(description="HuggingFace model ID")
    port: int = Field(description="Server port")
    pid: int | None = Field(None, description="Process ID")
    is_running: bool = Field(False, description="Whether server is responding")
    uptime_seconds: float | None = Field(None, description="Server uptime")


class ModelRegistry:
    """Registry for model metadata loaded from YAML.

    Supports case-insensitive model slug lookup with canonical normalization.
    """

    _instance = None
    _embeddings: dict[str, dict[str, Any]] = {}
    _rerankers: dict[str, dict[str, Any]] = {}

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._load_registry()
        return cls._instance

    def _load_registry(self) -> None:
        """Load model registry from YAML file."""
        # Look in config/ subdirectory (same level as cmw_infinity package)
        config_path = Path(__file__).parent.parent / "config" / "models.yaml"

        if not config_path.exists():
            raise FileNotFoundError(f"Model registry not found: {config_path}")

        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        # Build case-insensitive lookup tables
        for model_slug, model_data in data.get("embedding_models", {}).items():
            normalized = model_slug.lower()
            self._embeddings[normalized] = {
                "canonical_slug": model_slug,
                "model_type": "embedding",
                **model_data,
            }

        for model_slug, model_data in data.get("reranker_models", {}).items():
            normalized = model_slug.lower()
            self._rerankers[normalized] = {
                "canonical_slug": model_slug,
                "model_type": "reranker",
                **model_data,
            }

        logger.info(
            f"Loaded {len(self._embeddings)} embedding models and {len(self._rerankers)} reranker models"
        )

    def _normalize_slug(self, model_slug: str) -> str:
        """Normalize model slug to lowercase for case-insensitive lookup."""
        return model_slug.lower().strip()

    def _to_config(self, data: dict[str, Any]) -> InfinityModelConfig:
        """Build InfinityModelConfig from registry dict (exclude canonical_slug)."""
        return InfinityModelConfig(
            **{k: v for k, v in data.items() if k != "canonical_slug"}
        )

    def get_embedding_config(self, model_slug: str) -> InfinityModelConfig:
        """Get configuration for an embedding model (case-insensitive).

        Args:
            model_slug: Model identifier (e.g., "Qwen/Qwen3-Embedding-8B")

        Returns:
            Model configuration

        Raises:
            ValueError: If model not found
        """
        normalized = self._normalize_slug(model_slug)
        if normalized not in self._embeddings:
            available = [m["canonical_slug"] for m in self._embeddings.values()]
            raise ValueError(f"Unknown embedding model: {model_slug}. Available: {available}")
        return self._to_config(self._embeddings[normalized])

    def get_reranker_config(self, model_slug: str) -> InfinityModelConfig:
        """Get configuration for a reranker model (case-insensitive).

        Args:
            model_slug: Model identifier (e.g., "DiTy/cross-encoder-russian-msmarco")

        Returns:
            Model configuration

        Raises:
            ValueError: If model not found
        """
        normalized = self._normalize_slug(model_slug)
        if normalized not in self._rerankers:
            available = [m["canonical_slug"] for m in self._rerankers.values()]
            raise ValueError(f"Unknown reranker model: {model_slug}. Available: {available}")
        return self._to_config(self._rerankers[normalized])

    def get_config(self, model_slug: str) -> InfinityModelConfig:
        """Get configuration for any model (case-insensitive).

        Args:
            model_slug: Model identifier

        Returns:
            Model configuration

        Raises:
            ValueError: If model not found
        """
        normalized = self._normalize_slug(model_slug)
        if normalized in self._embeddings:
            return self._to_config(self._embeddings[normalized])
        if normalized in self._rerankers:
            return self._to_config(self._rerankers[normalized])
        available = [m["canonical_slug"] for m in self._embeddings.values()] + [
            m["canonical_slug"] for m in self._rerankers.values()
        ]
        raise ValueError(f"Unknown model: {model_slug}. Available: {available}")

    def get_model_type(self, model_slug: str) -> Literal["embedding", "reranker"]:
        """Get model type for a model slug."""
        normalized = self._normalize_slug(model_slug)
        if normalized in self._embeddings:
            return "embedding"
        if normalized in self._rerankers:
            return "reranker"
        raise ValueError(f"Unknown model: {model_slug}")

    def list_embeddings(self) -> list[str]:
        """List all available embedding models."""
        return [m["canonical_slug"] for m in self._embeddings.values()]

    def list_rerankers(self) -> list[str]:
        """List all available reranker models."""
        return [m["canonical_slug"] for m in self._rerankers.values()]

    def list_all(self) -> dict[str, list[str]]:
        """List all available models by type."""
        return {
            "embedding": self.list_embeddings(),
            "reranker": self.list_rerankers(),
        }


def get_model_config(model_slug: str) -> InfinityModelConfig:
    """Get configuration for a model (case-insensitive).

    Args:
        model_slug: Model identifier (e.g., "frida", "Qwen/Qwen3-Embedding-8B")

    Returns:
        Model configuration

    Raises:
        ValueError: If model not found
    """
    return ModelRegistry().get_config(model_slug)


def list_available_models() -> dict[str, list[str]]:
    """List all available models by type."""
    return ModelRegistry().list_all()
