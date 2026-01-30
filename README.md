# CMW Infinity

Infinity server management tool for CMW projects. Provides easy setup and server management for Infinity embedding and reranking inference servers.

## Features

- **Easy Setup**: One-command installation and verification
- **Model Management**: Download and serve embedding/reranker models from HuggingFace
- **Server Management**: Start, stop, and monitor Infinity servers
- **Configuration**: YAML-based configuration with sensible defaults

## Installation

```bash
# Clone repository
git clone https://github.com/your-org/cmw-infinity.git
cd cmw-infinity

# Install
pip install -e .

# Or install from git
pip install git+https://github.com/your-org/cmw-infinity.git
```

## Quick Start

### 1. Setup

```bash
cmw-infinity setup
```

This verifies:
- Infinity installation
- GPU availability
- Required dependencies

### 2. Start Server

```bash
# Start FRIDA embedding server (default)
cmw-infinity start frida

# Start DiTy reranker server
cmw-infinity start dity-reranker

# Start Qwen3-8B embedding (when vLLM is not running)
cmw-infinity start qwen3-embedding-8b
```

### 3. Check Status

```bash
# Check if server is running
cmw-infinity status

# Get detailed information
cmw-infinity info
```

### 4. Stop Server

```bash
cmw-infinity stop frida
cmw-infinity stop --all
```

## Configuration

Configuration is done via YAML file in `config/models.yaml`. Models are predefined with optimal settings.

## Available Models

### Embedding Models
- `frida` (ai-forever/FRIDA) - 1024 dim, ~4GB
- `qwen3-embedding-0.6b` - 1024 dim, ~2GB
- `qwen3-embedding-4b` - 2560 dim, ~12GB
- `qwen3-embedding-8b` - 4096 dim, ~22GB

### Reranker Models
- `dity-reranker` (DiTy/cross-encoder-russian-msmarco) - Russian optimized, ~2GB
- `bge-reranker` (BAAI/bge-reranker-v2-m3) - ~2GB
- `qwen3-reranker-0.6b` - ~2GB
- `qwen3-reranker-4b` - ~12GB
- `qwen3-reranker-8b` - ~22GB

## License

MIT
