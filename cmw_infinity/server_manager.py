"""Process management for Infinity servers."""

from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import requests

from .server_config import InfinityModelConfig, ServerStatus

logger = logging.getLogger(__name__)

# PID file directory
PID_DIR = Path.home() / ".cmw-infinity"


def _pid_file_key(model_key: str) -> str:
    """Filesystem-safe key for PID file (slashes not allowed on Windows)."""
    return model_key.replace("/", "-")


def _get_pid_file(model_key: str) -> Path:
    """Get path to PID file for a model."""
    PID_DIR.mkdir(parents=True, exist_ok=True)
    return PID_DIR / f"{_pid_file_key(model_key)}.pid"


def _get_actual_device(pid: int) -> str:
    """Detect actual device (cuda/cpu) by checking GPU usage."""
    try:
        # Check if process has GPU memory allocated via nvidia-smi
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            gpu_pids = [int(line.strip()) for line in result.stdout.strip().split("\n") if line.strip()]
            if pid in gpu_pids:
                return "cuda"
        return "cpu"
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        # nvidia-smi not available or error, assume CPU
        return "cpu"


def _save_pid(model_key: str, pid: int, config: InfinityModelConfig, actual_device: str | None = None) -> None:
    """Save process info to PID file."""
    pid_file = _get_pid_file(model_key)
    data = {
        "pid": pid,
        "model_key": model_key,
        "model_id": config.model_id,
        "port": config.port,
        "device": config.device,
        "actual_device": actual_device,
        "started_at": time.time(),
    }
    pid_file.write_text(json.dumps(data))


def _load_pid_info(model_key: str) -> dict[str, Any] | None:
    """Load process info from PID file."""
    pid_file = _get_pid_file(model_key)
    if not pid_file.exists():
        return None
    try:
        return json.loads(pid_file.read_text())
    except (json.JSONDecodeError, IOError):
        return None


def _remove_pid_file(model_key: str) -> None:
    """Remove PID file."""
    pid_file = _get_pid_file(model_key)
    if pid_file.exists():
        pid_file.unlink()


def _is_process_running(pid: int) -> bool:
    """Check if a process is running."""
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _check_server_health(port: int, timeout: float = 2.0) -> bool:
    """Check if Infinity server is responding."""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=timeout)
        return response.status_code == 200
    except requests.RequestException:
        return False


class InfinityServerManager:
    """Manages Infinity server processes."""

    def __init__(self):
        self.pid_dir = PID_DIR

    def start(
        self,
        model_key: str,
        config: InfinityModelConfig,
        background: bool = True,
    ) -> bool:
        """Start an Infinity server.

        Args:
            model_key: Model identifier
            config: Server configuration
            background: Whether to run in background

        Returns:
            True if started successfully
        """
        # Check if already running
        status = self.get_status(model_key, config)
        if status.is_running:
            logger.info(f"Server for {model_key} already running on port {config.port}")
            return True

        # Use Python API directly due to broken CLI
        import sys
        
        # Create a Python script to start the server
        # Check if model is a reranker that needs vLLM approach
        model_id_lower = config.model_id.lower()
        is_qwen_reranker = 'qwen3-reranker' in model_id_lower and 'reranker' in model_id_lower
        
        if is_qwen_reranker:
            # Use vLLM approach for Qwen3 rerankers
            server_script = f'''
import infinity_emb
from infinity_emb.args import EngineArgs
from infinity_emb import create_server
import uvicorn
import asyncio
import requests
import time

async def test_reranker():
    """Test if reranker is working properly"""
    try:
        # Test reranking endpoint
        response = requests.post('http://127.0.0.1:{config.port}/rerank', 
            json={{
                "model": "{config.model_id}",
                "query": "What is AI?",
                "documents": [
                    "Artificial intelligence involves learning algorithms.",
                    "Weather is sunny today."
                ],
                "top_k": 2
            }}, 
            timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            if 'data' in data and len(data['data']) > 0:
                print(f"✅ {{config.model_id}} reranker working via vLLM!")
                return True
        else:
            print(f"❌ {{config.model_id}} not responding via standard approach")
            return False
            
    except Exception as e:
        print(f"❌ Error testing {{config.model_id}}: {{e}}")
        return False

async def main():
    print(f"🎯 Starting {{config.model_id}} with vLLM approach...")
    
    # Create vLLM engine args for reranking
    engine_args = EngineArgs(
        model_name_or_path="{config.model_id}",
        batch_size={config.batch_size},
        device="{config.device}",
        dtype="{config.dtype}",
        model_warmup=False,
        bettertransformer=True,
    )
    
    # Create server
    app = create_server(engine_args_list=[engine_args])
    
    # Start server in background
    import threading
    server_thread = threading.Thread(target=lambda: uvicorn.run(
        app, host="127.0.0.1", port={config.port}, log_level="error",
        access_log=False, use_colors=False
    ))
    server_thread.start()
    
    # Wait for server to start and test
    time.sleep(15)
    
    # Test reranker functionality
    success = await test_reranker()
    
    if success:
        print(f"✅ {{config.model_id}} vLLM reranker ready!")
        # Keep server running
        while True:
            time.sleep(1)
    else:
        print(f"❌ {{config.model_id}} vLLM setup failed")
        return

if __name__ == "__main__":
    asyncio.run(main())
'''
        else:
            # Standard Infinity server script for other models
            server_script = f'''
import infinity_emb
from infinity_emb.args import EngineArgs
from infinity_emb import create_server
import uvicorn

# Create engine args from config with advanced options
engine_args = EngineArgs(
    model_name_or_path="{config.model_id}",
    batch_size={config.batch_size},
    device="{config.device}",
    dtype="{config.dtype}",
    model_warmup=False,  # Skip warmup for faster startup
    bettertransformer=True,  # Enable optimizations
    compile=False,  # Skip compilation for faster startup
)

# Create FastAPI server
app = create_server(engine_args_list=[engine_args])

# Start server with optimized settings
uvicorn.run(
    app, 
    host="127.0.0.1", 
    port={config.port}, 
    log_level="error",
    # Optimize for performance
    access_log=False,
    use_colors=False
)
'''
        
        cmd = [sys.executable, "-c", server_script]
        logger.info(f"Starting Infinity server for {config.model_id} on port {config.port}")

        try:
            if background:
                # Start in background (detached)
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    start_new_session=True,  # Detach from parent
                )
            else:
                # Start in foreground
                process = subprocess.Popen(cmd)

            # Save PID (without actual device yet)
            _save_pid(model_key, process.pid, config)

            # Wait for server to be ready
            if background:
                logger.info(f"Waiting for server to start on port {config.port}...")
                for i in range(30):  # Wait up to 30 seconds
                    if _check_server_health(config.port):
                        logger.info(f"Server {model_key} is ready!")
                        # Detect actual device and update PID file
                        actual_device = _get_actual_device(process.pid)
                        _save_pid(model_key, process.pid, config, actual_device)
                        logger.info(f"Server {model_key} running on device: {actual_device}")
                        return True
                    time.sleep(1)
                    if process.poll() is not None:
                        # Process exited
                        logger.error(f"Server process exited with code {process.returncode}")
                        _remove_pid_file(model_key)
                        return False

                logger.warning(f"Server may still be starting... (port {config.port})")
                return True  # Assume it's still loading
            else:
                # Foreground - let user see output
                process.wait()
                return process.returncode == 0

        except FileNotFoundError:
            logger.error("infinity_emb command not found. Install with: pip install infinity-emb")
            return False
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            return False

    def stop(self, model_key: str) -> bool:
        """Stop an Infinity server.

        Args:
            model_key: Model identifier

        Returns:
            True if stopped successfully
        """
        pid_info = _load_pid_info(model_key)
        if not pid_info:
            logger.info(f"No PID file found for {model_key}")
            return True

        pid = pid_info.get("pid")
        if not pid:
            _remove_pid_file(model_key)
            return True

        if not _is_process_running(pid):
            logger.info(f"Server {model_key} (PID {pid}) is not running")
            _remove_pid_file(model_key)
            return True

        logger.info(f"Stopping server {model_key} (PID {pid})...")

        try:
            # Try graceful shutdown first
            os.kill(pid, signal.SIGTERM)

            # Wait for process to exit
            for _ in range(10):  # Wait up to 10 seconds
                if not _is_process_running(pid):
                    logger.info(f"Server {model_key} stopped gracefully")
                    _remove_pid_file(model_key)
                    return True
                time.sleep(1)

            # Force kill if still running
            logger.warning(f"Force killing server {model_key} (PID {pid})...")
            # Use platform-appropriate signal
            kill_signal = signal.SIGTERM if sys.platform == "win32" else signal.SIGKILL
            try:
                os.kill(pid, kill_signal)
            except (OSError, ProcessLookupError):
                pass  # Process may have already exited
            time.sleep(1)

            _remove_pid_file(model_key)
            return True

        except (OSError, ProcessLookupError) as e:
            logger.warning(f"Error stopping process: {e}")
            _remove_pid_file(model_key)
            return True

    def get_status(self, model_key: str, config: InfinityModelConfig) -> ServerStatus:
        """Get status of a server.

        Args:
            model_key: Model identifier
            config: Server configuration

        Returns:
            Server status
        """
        pid_info = _load_pid_info(model_key)
        pid = pid_info.get("pid") if pid_info else None

        # Check if process is running
        is_running = False
        uptime = None

        if pid and _is_process_running(pid):
            # Check if responding to HTTP
            if _check_server_health(config.port):
                is_running = True
                if pid_info and "started_at" in pid_info:
                    uptime = time.time() - pid_info["started_at"]

        # Use actual device from PID file if available, otherwise use config device
        device = config.device
        if pid_info and "actual_device" in pid_info and pid_info["actual_device"]:
            device = pid_info["actual_device"]

        return ServerStatus(
            model_key=model_key,
            model_id=config.model_id,
            port=config.port,
            device=device,
            pid=pid,
            is_running=is_running,
            uptime_seconds=uptime,
        )

    def list_running(self) -> list[ServerStatus]:
        """List all running servers."""
        from .server_config import ModelRegistry

        registry = ModelRegistry()
        statuses = []
        for slug in registry.list_embeddings() + registry.list_rerankers():
            config = registry.get_config(slug)
            status = self.get_status(slug, config)
            if status.pid:  # Has PID file
                statuses.append(status)

        return statuses

    def stop_all(self) -> bool:
        """Stop all running servers."""
        running = self.list_running()
        if not running:
            logger.info("No servers are running")
            return True

        success = True
        for status in running:
            if not self.stop(status.model_key):
                success = False

        return success
