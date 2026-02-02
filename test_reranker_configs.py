#!/usr/bin/env python3
"""Test reranker with different configurations."""

import subprocess
import sys
import time
import requests

def test_reranker_config(config_name, engine_args_dict):
    print(f"\n--- Testing {config_name} ---")
    
    server_script = f'''
import infinity_emb
from infinity_emb.args import EngineArgs
from infinity_emb import create_server
import uvicorn

engine_args = EngineArgs(**{engine_args_dict})
app = create_server(engine_args_list=[engine_args])
uvicorn.run(app, host="127.0.0.1", port=8000, log_level="error", access_log=False, use_colors=False)
'''
    
    cmd = [sys.executable, "-c", server_script]
    process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    
    try:
        time.sleep(3)  # Wait for startup
        
        # Test health
        health = requests.get("http://127.0.0.1:8000/health", timeout=5)
        if health.status_code != 200:
            print("✗ Health check failed")
            return False
            
        print("✓ Health check passed")
        
        # Test reranking
        rerank_response = requests.post("http://127.0.0.1:8000/rerank", 
                                      json={"query":"test","documents":["hello world","test document"],"top_k":2}, 
                                      timeout=10)
        
        if rerank_response.status_code == 200:
            print("✓ Reranking works!")
            print(f"Response: {rerank_response.text}")
            return True
        else:
            print(f"✗ Reranking failed with status {rerank_response.status_code}")
            return False
            
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    finally:
        process.terminate()
        time.sleep(1)
        process.kill()
        time.sleep(1)

# Test different configurations
configs = {
    "minimal": '''
{
    "model_name_or_path": "BAAI/bge-reranker-v2-m3",
    "batch_size": 1,
    "device": "cpu",
    "dtype": "float32"
}
''',
    "without_bettertransformer": '''
{
    "model_name_or_path": "BAAI/bge-reranker-v2-m3", 
    "batch_size": 1,
    "device": "cpu",
    "dtype": "float32",
    "model_warmup": False,
    "bettertransformer": False,
    "compile": False
}
''',
    "with_warmup": '''
{
    "model_name_or_path": "BAAI/bge-reranker-v2-m3",
    "batch_size": 1, 
    "device": "cpu",
    "dtype": "float32",
    "model_warmup": True,
    "bettertransformer": False,
    "compile": False
}
'''
}

for name, config in configs.items():
    test_reranker_config(name, config)