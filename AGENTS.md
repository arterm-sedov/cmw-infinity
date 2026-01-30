# Agent Guide for cmw-infinity

This document provides guidance for AI agents working on the cmw-infinity project.

## Project Overview

cmw-infinity is a CLI tool for managing Infinity embedding and reranker servers. It provides:
- Server lifecycle management (start, stop, status)
- Pre-configured model definitions
- Process management with PID files
- Health checking

## Architecture

```
cmw_infinity/
├── __init__.py          # Package exports
├── cli.py              # Click CLI commands
├── server_config.py    # Pydantic schemas and model definitions
└── server_manager.py   # Process management
```

## Key Components

### ServerConfig (Pydantic)
Defines model configurations including:
- model_id: HuggingFace model identifier
- port: Server port (unique per model)
- memory_gb: Estimated VRAM usage
- dtype: Data type (float16, float32, int8)
- batch_size: Dynamic batching size

### ServerManager
Manages Infinity server processes:
- start(): Launch server in background/foreground
- stop(): Graceful shutdown with fallback to force kill
- get_status(): Check if server is running and responding
- list_running(): List all servers with PID files

### CLI Commands
- setup: Verify dependencies
- start <model>: Start server for model
- stop <model>: Stop server
- status: Show running servers
- list: Show available models

## Dependencies

Core:
- click: CLI framework
- pydantic: Data validation
- requests: HTTP health checks

External (user-installed):
- infinity-emb: The actual server binary
- torch: For GPU detection

## Error Handling

- Use try/except around process operations
- Log errors with logger, not print
- Return True/False from manager methods
- CLI catches exceptions and exits with code 1

## Platform Notes

- Windows: SIGKILL not available, use SIGTERM
- Linux/macOS: Full signal support
- PID files stored in ~/.cmw-infinity/

## Testing

Test scenarios:
1. Start/stop FRIDA server
2. Health check via HTTP
3. Multiple start calls (idempotent)
4. Stop non-running server
5. List running servers

## Code Style

- Follow Google docstring convention
- Type hints required
- Line length: 100
- Use ruff for linting
