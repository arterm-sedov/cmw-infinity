"""Integration tests for cmw-infinity with real inference.

These tests start actual Infinity servers and test real embedding/reranking.
They use small models and device='auto' to work on both CPU and GPU.
"""

from __future__ import annotations

import json
import time

import pytest
import requests

from cmw_infinity.server_config import (
    InfinityModelConfig,
    ServerStatus,
)
from cmw_infinity.server_manager import (
    InfinityServerManager,
    _check_server_health,
    _get_pid_file,
    _remove_pid_file,
)


# Small models suitable for testing (works on CPU or GPU)
TEST_EMBEDDING_CONFIG = InfinityModelConfig(
    model_id="sentence-transformers/all-MiniLM-L6-v2",
    model_type="embedding",
    port=7001,
    device="auto",
    dtype="float32",
    batch_size=8,
    memory_gb=0.5,
)

TEST_RERANKER_CONFIG = InfinityModelConfig(
    model_id="cross-encoder/ms-marco-MiniLM-L-2-v2",
    model_type="reranker",
    port=7002,
    device="auto",
    dtype="float32",
    batch_size=8,
    memory_gb=0.5,
)


def cleanup_test_servers():
    """Stop any running test servers and clean up PID files."""
    manager = InfinityServerManager()

    # Stop test servers
    for model_key in ["test-embedding", "test-reranker"]:
        try:
            manager.stop(model_key)
        except Exception:
            pass
        _remove_pid_file(model_key)


@pytest.fixture(scope="module", autouse=True)
def setup_test_module():
    """Clean up before and after test module."""
    cleanup_test_servers()
    yield
    cleanup_test_servers()


@pytest.fixture
def manager():
    """Create a fresh server manager."""
    return InfinityServerManager()


@pytest.fixture
def embedding_server(manager):
    """Start an embedding server for testing."""
    model_key = "test-embedding"

    # Clean up any existing server
    manager.stop(model_key)
    _remove_pid_file(model_key)

    # Start server
    success = manager.start(model_key, TEST_EMBEDDING_CONFIG, background=True)
    if not success:
        pytest.skip(
            "Failed to start embedding server (infinity-emb may not be installed or dependencies missing)"
        )

    # Wait for server to be ready
    max_retries = 60
    for i in range(max_retries):
        if _check_server_health(TEST_EMBEDDING_CONFIG.port, timeout=2.0):
            break
        time.sleep(1)
    else:
        manager.stop(model_key)
        pytest.skip("Embedding server failed to become healthy within timeout")

    yield TEST_EMBEDDING_CONFIG.port

    # Cleanup
    manager.stop(model_key)


@pytest.fixture
def reranker_server(manager):
    """Start a reranker server for testing."""
    model_key = "test-reranker"

    # Clean up any existing server
    manager.stop(model_key)
    _remove_pid_file(model_key)

    # Start server
    success = manager.start(model_key, TEST_RERANKER_CONFIG, background=True)
    if not success:
        pytest.skip(
            "Failed to start reranker server (infinity-emb may not be installed or dependencies missing)"
        )

    # Wait for server to be ready
    max_retries = 60
    for i in range(max_retries):
        if _check_server_health(TEST_RERANKER_CONFIG.port, timeout=2.0):
            break
        time.sleep(1)
    else:
        manager.stop(model_key)
        pytest.skip("Reranker server failed to become healthy within timeout")

    yield TEST_RERANKER_CONFIG.port

    # Cleanup
    manager.stop(model_key)


class TestServerLifecycle:
    """Test server start/stop lifecycle."""

    def test_start_embedding_server(self, manager):
        """Test starting an embedding server."""
        model_key = "test-lifecycle-embedding"
        config = InfinityModelConfig(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            model_type="embedding",
            port=7003,
            device="auto",
            dtype="float32",
            batch_size=8,
            memory_gb=0.5,
        )

        # Clean up first
        manager.stop(model_key)
        _remove_pid_file(model_key)

        try:
            success = manager.start(model_key, config, background=True)
            if not success:
                pytest.skip("infinity-emb not installed or dependencies missing")

            # Wait for health
            for i in range(60):
                if _check_server_health(config.port, timeout=2.0):
                    break
                time.sleep(1)
            else:
                pytest.fail("Server failed to become healthy")

            # Check status
            status = manager.get_status(model_key, config)
            assert status.is_running is True
            assert status.port == config.port
            assert status.pid is not None

        finally:
            manager.stop(model_key)
            _remove_pid_file(model_key)

    def test_auto_device_argument_in_cli_args(self):
        """Test that auto device doesn't pass --device arg."""
        config = InfinityModelConfig(
            model_id="test/model",
            model_type="embedding",
            port=7004,
            device="auto",
            dtype="float32",
            batch_size=8,
            memory_gb=1.0,
        )

        args = config.to_infinity_args()
        # auto device should not add --device argument
        assert "--device" not in args

    def test_server_health_check(self, embedding_server):
        """Test health check endpoint."""
        port = embedding_server

        # Test health endpoint
        response = requests.get(f"http://localhost:{port}/health", timeout=5.0)
        assert response.status_code == 200

        # Verify response contains expected fields
        data = response.json()
        assert "status" in data


class TestEmbeddingAPI:
    """Test real embedding API calls."""

    def test_single_text_embedding(self, embedding_server):
        """Test embedding a single text."""
        port = embedding_server

        response = requests.post(
            f"http://localhost:{port}/embeddings",
            json={
                "input": "This is a test sentence for embedding.",
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            timeout=10.0,
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "data" in data
        assert len(data["data"]) == 1
        assert "embedding" in data["data"][0]

        # Check embedding is a non-empty list of floats
        embedding = data["data"][0]["embedding"]
        assert isinstance(embedding, list)
        assert len(embedding) > 0
        assert all(isinstance(x, (int, float)) for x in embedding)

    def test_batch_text_embedding(self, embedding_server):
        """Test embedding multiple texts at once."""
        port = embedding_server

        texts = [
            "First test sentence.",
            "Second test sentence.",
            "Third test sentence.",
        ]

        response = requests.post(
            f"http://localhost:{port}/embeddings",
            json={
                "input": texts,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        # Check we got embeddings for all inputs
        assert len(data["data"]) == len(texts)

        # Check all embeddings have same dimension
        dimensions = [len(item["embedding"]) for item in data["data"]]
        assert all(d == dimensions[0] for d in dimensions)

    def test_embedding_similarity(self, embedding_server):
        """Test that similar texts have higher similarity."""
        port = embedding_server

        texts = [
            "The quick brown fox jumps over the lazy dog.",
            "A fast brown fox leaps over a sleepy dog.",
            "Machine learning is a subset of artificial intelligence.",
        ]

        response = requests.post(
            f"http://localhost:{port}/embeddings",
            json={
                "input": texts,
                "model": "sentence-transformers/all-MiniLM-L6-v2",
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        embeddings = [item["embedding"] for item in data["data"]]

        # Compute cosine similarities
        def cosine_similarity(a, b):
            import math

            dot = sum(x * y for x, y in zip(a, b))
            norm_a = math.sqrt(sum(x * x for x in a))
            norm_b = math.sqrt(sum(x * x for x in b))
            return dot / (norm_a * norm_b)

        # Similar texts (0 and 1) should have higher similarity than dissimilar (0 and 2)
        sim_01 = cosine_similarity(embeddings[0], embeddings[1])
        sim_02 = cosine_similarity(embeddings[0], embeddings[2])

        assert sim_01 > sim_02, f"Similar texts should have higher similarity: {sim_01} vs {sim_02}"


class TestRerankingAPI:
    """Test real reranking API calls."""

    def test_basic_reranking(self, reranker_server):
        """Test basic reranking functionality."""
        port = reranker_server

        query = "What is machine learning?"
        documents = [
            "Machine learning is a method of data analysis.",
            "The weather is sunny today.",
            "Deep learning is a subset of machine learning.",
        ]

        response = requests.post(
            f"http://localhost:{port}/rerank",
            json={
                "query": query,
                "documents": documents,
                "model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        # Check response structure
        assert "results" in data
        assert len(data["results"]) == len(documents)

        # Check results have scores
        for result in data["results"]:
            assert "score" in result
            assert isinstance(result["score"], (int, float))

    def test_reranking_relevance_ordering(self, reranker_server):
        """Test that reranking orders documents by relevance."""
        port = reranker_server

        query = "artificial intelligence"
        documents = [
            "The capital of France is Paris.",
            "AI and machine learning are transforming technology.",
            "Python is a programming language.",
            "Artificial intelligence enables machines to learn.",
        ]

        response = requests.post(
            f"http://localhost:{port}/rerank",
            json={
                "query": query,
                "documents": documents,
                "model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        results = data["results"]

        # Sort by score descending
        sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)

        # Top results should be AI-related documents
        top_indices = [r["index"] for r in sorted_results[:2]]
        ai_related_indices = [1, 3]  # Indices of AI-related documents

        # At least one of the top 2 should be AI-related
        assert any(i in ai_related_indices for i in top_indices), (
            "Top reranked results should include AI-related documents"
        )

    def test_reranking_with_top_k(self, reranker_server):
        """Test reranking with top_k parameter."""
        port = reranker_server

        query = "test query"
        documents = [f"Document {i} content here." for i in range(10)]

        response = requests.post(
            f"http://localhost:{port}/rerank",
            json={
                "query": query,
                "documents": documents,
                "model": "cross-encoder/ms-marco-MiniLM-L-2-v2",
                "top_k": 3,
            },
            timeout=15.0,
        )

        assert response.status_code == 200
        data = response.json()

        # Should return only top_k results
        assert len(data["results"]) == 3


class TestServerManagement:
    """Test server management functionality."""

    def test_list_running_servers(self, manager, embedding_server):
        """Test listing running servers."""
        running = manager.list_running()

        # Should find our test embedding server
        test_servers = [s for s in running if s.model_key == "test-embedding"]
        assert len(test_servers) > 0

        status = test_servers[0]
        assert status.is_running is True
        assert status.port == TEST_EMBEDDING_CONFIG.port

    def test_server_status_reporting(self, manager, embedding_server):
        """Test that server status is correctly reported."""
        status = manager.get_status("test-embedding", TEST_EMBEDDING_CONFIG)

        assert status.model_key == "test-embedding"
        assert status.model_id == TEST_EMBEDDING_CONFIG.model_id
        assert status.port == TEST_EMBEDDING_CONFIG.port
        assert status.is_running is True
        assert status.pid is not None
        assert status.uptime_seconds is not None
        assert status.uptime_seconds >= 0

    def test_stop_server(self, manager):
        """Test stopping a server."""
        model_key = "test-stop-server"
        config = InfinityModelConfig(
            model_id="sentence-transformers/all-MiniLM-L6-v2",
            model_type="embedding",
            port=7005,
            device="auto",
            dtype="float32",
            batch_size=8,
            memory_gb=0.5,
        )

        # Clean up first
        manager.stop(model_key)
        _remove_pid_file(model_key)

        try:
            # Start server
            success = manager.start(model_key, config, background=True)
            if not success:
                pytest.skip(f"Failed to start server on port {config.port}")

            # Wait for it to be ready
            for i in range(60):
                if _check_server_health(config.port, timeout=2.0):
                    break
                time.sleep(1)
            else:
                pytest.fail("Server failed to start")

            # Verify it's running
            assert _check_server_health(config.port) is True

            # Stop server
            assert manager.stop(model_key) is True

            # Give it time to shut down
            time.sleep(2)

            # Verify it's stopped
            assert _check_server_health(config.port) is False

        finally:
            # Ensure cleanup
            manager.stop(model_key)
            _remove_pid_file(model_key)


class TestInfinityCLIArgs:
    """Test CLI argument generation for different configurations."""

    def test_auto_float32_config(self):
        """Test CLI args for auto device float32 configuration."""
        config = InfinityModelConfig(
            model_id="test/model",
            model_type="embedding",
            port=7006,
            device="auto",
            dtype="float32",
            batch_size=16,
            memory_gb=1.0,
        )

        args = config.to_infinity_args()

        assert "v2" in args
        assert "--model-name-or-path" in args
        assert "test/model" in args
        assert "--port" in args
        assert "7006" in args
        # auto device should not add --device argument
        assert "--device" not in args
        assert "--dtype" in args
        assert "float32" in args
        assert "--batch-size" in args
        assert "16" in args
