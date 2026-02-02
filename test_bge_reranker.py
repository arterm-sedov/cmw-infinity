#!/usr/bin/env python3
"""Manually start the BGE reranker server on port 7999."""

import subprocess
import sys
import time
import json
from pathlib import Path

server_script = '''
import infinity_emb
from infinity_emb.args import EngineArgs
from infinity_emb import create_server
import uvicorn

engine_args = EngineArgs(
    model_name_or_path="BAAI/bge-reranker-v2-m3",
    batch_size=16,
    device="cpu",
    dtype="float32",
    model_warmup=False,
    bettertransformer=True,
    compile=False,
)

app = create_server(engine_args_list=[engine_args])

uvicorn.run(
    app, 
    host="127.0.0.1", 
    port=7999, 
    log_level="error",
    access_log=False, 
    use_colors=False
)
'''

cmd = [sys.executable, "-c", server_script]

print(f"Starting BGE server with command: {' '.join(cmd)}")

# Start in background
process = subprocess.Popen(
    cmd,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    start_new_session=True,
)

print(f"Server started with PID: {process.pid}")

# Wait a moment
time.sleep(5)

# Test if server is responsive
try:
    import requests
    response = requests.get("http://127.0.0.1:7999/health", timeout=5)
    if response.status_code == 200:
        print("✓ BGE Server is ready!")
        
        # Test reranking
        rerank_response = requests.post("http://127.0.0.1:7999/rerank", 
                                      json={"query":"test","documents":["hello world"],"top_k":1}, 
                                      timeout=10)
        print(f"Rerank test: {rerank_response.status_code}")
        if rerank_response.status_code == 200:
            print("✓ BGE Reranker is working!")
            print(f"Response: {rerank_response.text}")
        else:
            print("✗ BGE Reranker failed")
            
    else:
        print("✗ BGE Server health check failed")
except Exception as e:
    print(f"Error testing BGE server: {e}")
    
# Kill the test server
process.terminate()
print("BGE server stopped for testing")