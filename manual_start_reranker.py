#!/usr/bin/env python3
"""Manually start the DiTy reranker server."""

import subprocess
import sys
import time
import json
from pathlib import Path

PID_DIR = Path.home() / ".cmw-infinity"
PID_DIR.mkdir(parents=True, exist_ok=True)

server_script = '''
import infinity_emb
from infinity_emb.args import EngineArgs
from infinity_emb import create_server
import uvicorn

engine_args = EngineArgs(
    model_name_or_path="DiTy/cross-encoder-russian-msmarco",
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
    port=7998, 
    log_level="error",
    access_log=False, 
    use_colors=False
)
'''

cmd = [sys.executable, "-c", server_script]

print(f"Starting server with command: {' '.join(cmd)}")

# Start in background
process = subprocess.Popen(
    cmd,
    stdout=subprocess.DEVNULL,
    stderr=subprocess.DEVNULL,
    start_new_session=True,
)

print(f"Server started with PID: {process.pid}")

# Save PID
pid_file = PID_DIR / "DiTy-cross-encoder-russian-msmarco.pid"
data = {
    "pid": process.pid,
    "model_key": "DiTy/cross-encoder-russian-msmarco",
    "model_id": "DiTy/cross-encoder-russian-msmarco",
    "port": 7998,
    "started_at": time.time(),
}
pid_file.write_text(json.dumps(data))

print("Waiting for server to start...")
for i in range(30):
    try:
        import requests
        response = requests.get("http://127.0.0.1:7998/health", timeout=1)
        if response.status_code == 200:
            print("✓ Server is ready!")
            break
    except:
        pass
    time.sleep(1)
    
    if process.poll() is not None:
        print(f"✗ Server process exited with code {process.returncode}")
        pid_file.unlink(missing_ok=True)
        sys.exit(1)

print("Server started successfully!")