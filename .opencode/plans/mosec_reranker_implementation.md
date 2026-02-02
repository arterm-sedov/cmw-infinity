# MOSEC Reranker Implementation Plan

## Overview
Production-ready MOSEC-based reranker server for DiTy Russian cross-encoder with Intel CPU optimizations and OpenVINO acceleration.

## Project Structure
```
cmw-mosec-reranker/
├── pyproject.toml          # Dependencies and project metadata
├── README.md              # Comprehensive documentation
├── LICENSE                # MIT license
├── Dockerfile             # Production deployment with OpenVINO
├── .gitignore            # Exclude model files
├── cmw_mosec_reranker/   # Python package
│   ├── __init__.py
│   ├── worker.py          # MOSEC Worker with ONNX/OpenVINO
│   ├── server.py          # Server configuration
│   ├── config.py          # Settings and validation
│   └── cli.py            # Click CLI interface
├── tests/                 # Unit and integration tests
├── models/               # ONNX model storage
└── docker/               # Docker compose files
```

## Implementation Phases

### Phase 1: Core Implementation (Priority: High)
**Timeline: 2-3 hours**

#### 1.1 MOSEC Worker Implementation
```python
# cmw_mosec_reranker/worker.py
import os
from typing import List
from mosec import Server, Worker, Runtime
from mosec.mixin import MsgpackMixin
from optimum.intel import OVModelForSequenceClassification  # OpenVINO for Intel
from transformers import AutoTokenizer

# Intel CPU optimizations
os.environ["OMP_NUM_THREADS"] = "1" 
os.environ["MKL_NUM_THREADS"] = "1"

class DiTyRerankerWorker(MsgpackMixin, Worker):
    def __init__(self, model_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        # Use OpenVINO for optimal Intel performance
        self.model = OVModelForSequenceClassification.from_pretrained(
            model_path, 
            export=True,  # Compiles ONNX to OpenVINO IR
            device="CPU"
        )

    def forward(self, data: List[dict]) -> List[float]:
        # MOSEC aggregates requests into 'data'
        # Expecting: [{"query": "...", "passage": "..."}, ...]
        pairs = [(item["query"], item["passage"]) for item in data]
        
        inputs = self.tokenizer(
            pairs, 
            padding=True, 
            truncation=True, 
            max_length=512, 
            return_tensors="pt"
        )
        
        outputs = self.model(**inputs)
        # Apply sigmoid to normalize scores to [0, 1]
        probabilities = torch.sigmoid(outputs.logits).flatten().tolist()
        
        return probabilities
```

#### 1.2 Server Configuration
```python
# cmw_mosec_reranker/server.py
from mosec import Server, Runtime
from .worker import DiTyRerankerWorker
from .config import ServerConfig

def create_server(config: ServerConfig) -> Server:
    """Create MOSEC server with optimal settings for Intel CPUs."""
    server = Server(
        # Rust-based high-performance controller
        max_batch_size=config.max_batch_size,
        # Optimize for request patterns
        max_wait_time=config.max_wait_time,
        # Performance monitoring
        enable_metrics=True,
    )
    
    # Append workers with Intel-optimized threading
    server.append_worker(
        DiTyRerankerWorker, 
        num=config.workers,
        max_batch_size=config.max_batch_size
    )
    
    return server
```

#### 1.3 CLI Interface (cmw-infinity style)
```python
# cmw_mosec_reranker/cli.py
import click
from pathlib import Path
from .server import create_server
from .config import ServerConfig

@click.group()
def cli():
    """CMW MOSEC Reranker - High-performance reranker server."""
    pass

@cli.command()
@click.option('--model', '-m', required=True, help='Path to ONNX model directory')
@click.option('--port', '-p', default=8080, help='Server port (default: 8080)')
@click.option('--workers', '-w', default=4, help='Number of worker processes')
@click.option('--batch-size', '-b', default=8, help='Max batch size')
@click.option('--max-wait-time', default=50, help='Max wait time for batching (ms)')
def serve(model: str, port: int, workers: int, batch_size: int, max_wait_time: int):
    """Start MOSEC reranker server."""
    config = ServerConfig(
        model_path=Path(model),
        port=port,
        workers=workers,
        max_batch_size=batch_size,
        max_wait_time=max_wait_time,
    )
    
    server = create_server(config)
    server.run()

@cli.command()
def convert():
    """Convert DiTy model to ONNX format."""
    # Check if optimum is available
    try:
        from optimum.cli.export import main as export_main
    except ImportError:
        click.echo("Error: optimum[onnxruntime] not installed", err=True)
        return
    
    # Run conversion
    click.echo("Converting DiTy model to ONNX...")
    export_main([
        "--model", "DiTy/cross-encoder-russian-msmarco",
        "--task", "text-classification", 
        "--output", "models/dity_onnx_model/"
    ])
```

#### 1.4 Configuration Management
```python
# cmw_mosec_reranker/config.py
from pydantic import BaseModel
from typing import Optional
from pathlib import Path

class ServerConfig(BaseModel):
    """Server configuration for MOSEC reranker."""
    model_path: Path
    port: int = 8080
    workers: int = 4
    max_batch_size: int = 8
    max_wait_time: int = 50  # ms
    timeout: float = 30.0
    log_level: str = "info"
    health_port: int = 8081
    
    class Config:
        """Pydantic configuration."""
        env_prefix = "CMW_MOSEC_"
        env_file = ".env"
```

### Phase 2: Testing Protocol (Priority: High)
**Timeline: 1 hour**

#### 2.1 cURL Testing Commands
```bash
# FastAPI-style testing (single request object)
curl -X POST http://localhost:8080/rerank \
     -H "Content-Type: application/json" \
     -d '{
       "query": "Как приготовить борщ?",
       "documents": [
         "Борщ — это традиционный свекольный суп.",
         "Вчера была хорошая погода для прогулки.",
         "Машинное обучение использует нейронные сети."
       ]
     }' \
     --max-time 10

# MOSEC-style testing (array of pairs)  
curl -X POST http://localhost:8000/inference \
     -H "Content-Type: application/json" \
     -d '[
       {"query": "Как приготовить борщ?", "passage": "Борщ — это традиционный свекольный суп."},
       {"query": "Как приготовить борщ?", "passage": "Вчера была хорошая погода для прогулки."},
       {"query": "Как приготовить борщ?", "passage": "Машинное обучение использует нейронные сети."}
     ]' \
     --max-time 10
```

#### 2.2 Integration Testing with RAG System
```python
# Test script: tests/test_integration.py
import requests
import json

def test_reranker_endpoint():
    """Test reranker integration with existing RAG system."""
    
    # Test basic functionality
    response = requests.post("http://localhost:8080/rerank", 
        json={
            "query": "машинное обучение",
            "documents": [
                "Машинное обучение и искусственный интеллект",
                "Приготовление кофе и рецепты выпечки",
                "Нейронные сети и глубокое обучение"
            ],
            "top_k": 2
        },
        timeout=30
    )
    
    assert response.status_code == 200, f"Expected 200, got {response.status_code}"
    
    data = response.json()
    assert "scores" in data, "Missing scores in response"
    assert len(data["scores"]) == 3, "Expected 3 scores"
    
    # Validate score normalization
    scores = data["scores"]
    assert all(0 <= s <= 1 for s in scores), f"Scores not normalized: {scores}"
    
    print(f"✓ Integration test passed. Scores: {scores}")
    return True
```

### Phase 3: Production Deployment (Priority: Medium)
**Timeline: 1-2 hours**

#### 3.1 Docker with OpenVINO Optimization
```dockerfile
# Multi-stage build for optimized production image
FROM python:3.10-slim as base

# Install system dependencies for OpenVINO
RUN apt-get update && apt-get install -y \
    build-essential \
    libgomp1 \
    libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml /tmp/
RUN pip install --no-cache-dir /tmp[openvino] mosec

# Production stage
FROM python:3.10-slim as production

# Copy only necessary files
WORKDIR /app
COPY --from=base /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=base /tmp/pyproject.toml ./

# Create model directory
RUN mkdir -p /app/models

# Copy application code
COPY cmw_mosec_reranker/ ./cmw_mosec_reranker/

# Set OpenVINO backend for Intel optimization
ENV OPTIMUM_INTEL_BACKEND=openvino
ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1

# Expose ports
EXPOSE 8080 8081

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8081/health || exit 1

# Start server with production settings
CMD ["cmw-mosec-reranker", "serve", \
     "--model", "/app/models/dity_onnx_model/", \
     "--port", "8080", \
     "--workers", "4", \
     "--batch-size", "8", \
     "--health-port", "8081"]
```

#### 3.2 Production Commands
```bash
# Build optimized Docker image
docker build -t cmw-mosec-reranker:latest .

# Run with production settings
docker run -d \
  --name cmw-reranker \
  -p 8080:8080 \
  -p 8081:8081 \
  -v /path/to/models:/app/models \
  --env OMP_NUM_THREADS=1 \
  --env MKL_NUM_THREADS=1 \
  cmw-mosec-reranker:latest

# Or use Docker Compose
version: '3.8'
services:
  cmw-reranker:
    build: .
    ports:
      - "8080:8080"
      - "8081:8081"
    volumes:
      - ./models:/app/models
    environment:
      - OMP_NUM_THREADS=1
      - MKL_NUM_THREADS=1
      - OPTIMUM_INTEL_BACKEND=openvino
    restart: unless-stopped
```

### Phase 4: RAG Integration (Priority: Medium)
**Timeline: 1 hour**

#### 4.1 RAG System Integration
```python
# Update rag_engine/retrieval/reranker.py
class CMWMosecReranker(HTTPClientMixin):
    """CMW MOSEC-based reranker server integration."""
    
    def __init__(self, config: ServerRerankerConfig):
        super().__init__(
            endpoint=config.endpoint,
            timeout=60.0,
            max_retries=3,
        )
        # Model name for logging
        self.model_name = getattr(config, 'model_name', 'DiTy/cross-encoder-russian-msmarco')

    def rerank(self, query: str, candidates, top_k: int, **kwargs):
        """Rerank using MOSEC server with optimized batching."""
        # Format request for MOSEC (array of objects)
        documents = [
            doc.page_content if hasattr(doc, "page_content") else str(doc)
            for doc, _ in candidates
        ]
        
        # Build MOSEC-style request body
        request_data = [
            {"query": query, "passage": doc} 
            for doc in documents
        ]
        
        response = self._post("/rerank", request_data)
        
        scores = response["scores"]
        
        # Apply metadata boosts (reuse existing logic)
        scored: list[tuple[Any, float]] = []
        for (doc, _), score in zip(candidates, scores):
            # ... existing metadata boost logic ...
            scored.append((doc, float(score)))
        
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]
```

#### 4.2 Environment Configuration
```bash
# Update .env for RAG system
RERANKER_PROVIDER_TYPE=cmw_mosec
CMW_MOSEC_RERANKER_ENDPOINT=http://localhost:8080
CMW_MOSEC_RERANKER_MODEL=DiTy/cross-encoder-russian-msmarco

# Intel optimizations
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
OPTIMUM_INTEL_BACKEND=openvino

# Performance tuning
CMW_MOSEC_MAX_BATCH_SIZE=8
CMW_MOSEC_WORKERS=4
CMW_MOSEC_MAX_WAIT_TIME=50
```

## Performance Benchmarks & Monitoring

### Expected Performance (Intel Xeon, 4 cores)
| Metric | MOSEC | FastAPI | infinity_emb (broken) |
|---------|--------|----------|---------------------|
| **Throughput** | **15 req/s** | 4 req/s | 0 req/s |
| **Latency (p95)** | **85ms** | 120ms | ∞ |
| **Memory Usage** | **800MB** | 1.2GB | N/A |
| **CPU Utilization** | **75%** | 95% | 100% |
| **Development Time** | 4-6 hours | 1-2 hours | ∞ |

### Monitoring Endpoints
```bash
# Health check
curl http://localhost:8081/health
# Response: {"status": "healthy", "model": "DiTy/cross-encoder-russian-msmarco"}

# Prometheus metrics
curl http://localhost:8080/metrics
# Response: reranker_requests_total, reranker_request_duration_seconds, etc.

# Load testing
hey -c 10 -z 30s -m POST -H "Content-Type: application/json" \
  -d '{"query":"test","documents":["test document"],"top_k":1}' \
  http://localhost:8080/rerank
```

## Testing Protocol Checklist

### Pre-deployment Testing
- [ ] Model conversion to ONNX successful
- [ ] ONNX model loads with OpenVINO
- [ ] Basic rerank functionality works
- [ ] Score normalization (sigmoid) working
- [ ] Health endpoints responding
- [ ] Memory usage within limits
- [ ] Error handling for invalid requests

### Load Testing
- [ ] Concurrent request handling (10+ req/s)
- [ ] Graceful degradation under load
- [ ] Memory stability over time
- [ ] Error rate monitoring
- [ ] Performance regression testing

### Integration Testing
- [ ] RAG system integration works
- [ ] Backward compatibility maintained
- [ ] Error handling in RAG context
- [ ] Performance improvements measurable

## Migration Strategy

### From infinity_emb to MOSEC
1. **Parallel Deployment**: Run both services temporarily
2. **A/B Testing**: Compare results on identical queries
3. **Performance Validation**: Ensure MOSEC >= infinity_emb performance
4. **Cutover**: Update RAG config to use MOSEC
5. **Decommission**: Remove infinity_emb reranker

### Rollback Plan
1. **Config Reversion**: Revert to infinity_emb settings
2. **Service Restart**: Restart infinity_emb reranker
3. **Validation**: Confirm RAG system working
4. **Post-mortem**: Analyze MOSEC failure reasons

## Troubleshooting Guide

### Common Issues
1. **OpenVINO Import Errors**
   ```bash
   # Ensure optimum[openvino] is installed
   pip install "optimum[openvino]"
   
   # Verify OpenVINO availability
   python -c "from optimum.intel import OVModel; print('OpenVINO available')"
   ```

2. **Threading Performance Issues**
   ```bash
   # For 4-core CPU, use 1 thread per process
   export OMP_NUM_THREADS=1
   export MKL_NUM_THREADS=1
   
   # Monitor CPU usage
   htop -p $(pgrep -f cmw-mosec-reranker)
   ```

3. **Memory Overruns**
   ```bash
   # Reduce batch size if memory constrained
   export CMW_MOSEC_MAX_BATCH_SIZE=4
   
   # Monitor memory usage
   ps -o pid,vsz,rss,comm -p $(pgrep -f cmw-mosec-reranker)
   ```

4. **Performance Degradation**
   ```bash
   # Check if OpenVINO is being used
   curl http://localhost:8081/health | grep -i openvino
   
   # Verify ONNX model conversion
   ls -la models/dity_onnx_model/
   ```

## Implementation Timeline

| Day | Tasks | Deliverables |
|------|--------|-------------|
| **Day 1** | Phase 1: Core Implementation | Working MOSEC server with basic functionality |
| **Day 2** | Phase 2: Testing & Validation | Comprehensive test suite, performance benchmarks |
| **Day 3** | Phase 3: Production Deployment | Docker image, production deployment |
| **Day 4** | Phase 4: RAG Integration | RAG system integration, cutover plan |

## Success Criteria

### Technical Success
- [ ] MOSEC server runs without errors
- [ ] Performance meets benchmarks (15+ req/s)
- [ ] Intel optimizations applied (OpenVINO, threading)
- [ ] Health and metrics endpoints working
- [ ] Docker deployment successful

### Business Success
- [ ] RAG system integration complete
- [ ] Performance improvement over infinity_emb (measurable)
- [ ] Production stability validated
- [ ] Team trained on operations and troubleshooting

## Risk Mitigation

### Technical Risks
- **OpenVINO Compatibility**: Test on target hardware early
- **Performance Regression**: Benchmark against baseline
- **Memory Constraints**: Implement configurable batch sizes
- **Docker Issues**: Multi-stage builds, size optimization

### Operational Risks
- **Service Downtime**: Parallel deployment strategy
- **Team Training**: Documentation, runbooks
- **Integration Complexity**: Incremental migration approach
- **Performance Bottlenecks**: Monitoring and alerting

---

**Next Steps: Proceed with Phase 1 Implementation**