"""Server configuration management for Infinity."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


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
        """Convert to infinity_emb CLI arguments."""
        args = [
            "v2",
            "--model-name-or-path",
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


# Memory estimates include model weights + activation overhead + batch buffer
EMBEDDING_MODELS: dict[str, InfinityModelConfig] = {
    "frida": InfinityModelConfig(
        model_id="ai-forever/FRIDA",
        model_type="embedding",
        port=7997,
        memory_gb=4.0,
    ),
    "qwen3-embedding-0.6b": InfinityModelConfig(
        model_id="Qwen/Qwen3-Embedding-0.6B",
        model_type="embedding",
        port=7997,  # Same port - won't run simultaneously
        memory_gb=2.0,
    ),
    "qwen3-embedding-4b": InfinityModelConfig(
        model_id="Qwen/Qwen3-Embedding-4B",
        model_type="embedding",
        port=7997,
        memory_gb=12.0,
    ),
    "qwen3-embedding-8b": InfinityModelConfig(
        model_id="Qwen/Qwen3-Embedding-8B",
        model_type="embedding",
        port=7997,
        memory_gb=22.0,
    ),
}

RERANKER_MODELS: dict[str, InfinityModelConfig] = {
    "bge-reranker": InfinityModelConfig(
        model_id="BAAI/bge-reranker-v2-m3",
        model_type="reranker",
        port=7998,
        memory_gb=2.0,
    ),
    "dity-reranker": InfinityModelConfig(
        model_id="DiTy/cross-encoder-russian-msmarco",
        model_type="reranker",
        port=7998,  # Same port - won't run simultaneously
        memory_gb=2.0,
    ),
    "qwen3-reranker-0.6b": InfinityModelConfig(
        model_id="Qwen/Qwen3-Reranker-0.6B",
        model_type="reranker",
        port=7998,
        memory_gb=2.0,
    ),
    "qwen3-reranker-4b": InfinityModelConfig(
        model_id="Qwen/Qwen3-Reranker-4B",
        model_type="reranker",
        port=7998,
        memory_gb=12.0,
    ),
    "qwen3-reranker-8b": InfinityModelConfig(
        model_id="Qwen/Qwen3-Reranker-8B",
        model_type="reranker",
        port=7998,
        memory_gb=22.0,
    ),
}


def get_model_config(model_key: str) -> InfinityModelConfig:
    """Get configuration for a model.

    Args:
        model_key: Model identifier (e.g., "frida", "dity-reranker")

    Returns:
        Model configuration

    Raises:
        ValueError: If model not found
    """
    all_models = {**EMBEDDING_MODELS, **RERANKER_MODELS}
    if model_key not in all_models:
        available = list(all_models.keys())
        raise ValueError(f"Unknown model: {model_key}. Available: {available}")
    return all_models[model_key]


def list_available_models() -> dict[str, list[str]]:
    """List all available models by type."""
    return {
        "embedding": list(EMBEDDING_MODELS.keys()),
        "reranker": list(RERANKER_MODELS.keys()),
    }
