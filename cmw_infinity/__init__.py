"""CMW Infinity - Server management for embedding/reranker inference."""

from __future__ import annotations

from .server_config import (
    InfinityModelConfig,
    ModelRegistry,
    ServerStatus,
    get_model_config,
    list_available_models,
)
from .server_manager import InfinityServerManager

__version__ = "0.1.0"
__all__ = [
    "InfinityModelConfig",
    "ServerStatus",
    "InfinityServerManager",
    "get_model_config",
    "list_available_models",
    "ModelRegistry",
]
