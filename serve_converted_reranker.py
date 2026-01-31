#!/usr/bin/env python3
"""Serve Qwen3-Reranker-0.6B with in-memory conversion to classifier."""

import asyncio
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification
from infinity_emb.args import EngineArgs
from infinity_emb import create_server
import uvicorn

print("🚀 Starting Qwen3-Reranker-0.6B with in-memory conversion...")
print("=" * 60)

# Load original model (generative)
model_name = "Qwen/Qwen3-Reranker-0.6B"
print(f"📦 Loading generative model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)

print(f"✅ Model loaded as generative")
print(f"📊 Original model type: AutoModelForCausalLM")

# Create classifier head
vocab_size = original_model.config.vocab_size
hidden_size = original_model.config.hidden_size

print(f"📏 Model configuration: vocab={vocab_size}, hidden={hidden_size}")

# Replace LM head with classifier head
print("🔧 Converting to classifier in-memory...")
classifier_head = nn.Linear(hidden_size, 1, bias=False)
original_model.lm_head = classifier_head

# Create proper config for sequence classifier
original_model.config.num_labels = 1
original_model.config.problem_type = "single_label_classification"

# Save converted model temporarily
output_path = "/tmp/qwen-reranker-converted"
torch.save(original_model.state_dict(), f"{output_path}/pytorch_model.bin")

# Save config
import json
config_dict = {
    "architectures": ["AutoModelForSequenceClassification"],
    "model_type": "qwen3",  # Qwen3 base model type
    "vocab_size": vocab_size,
    "hidden_size": hidden_size,
    "num_labels": 1,
    "problem_type": "single_label_classification",
    "torch_dtype": "bfloat16"
}

with open(f"{output_path}/config.json", "w") as f:
    json.dump(config_dict, f)

print(f"💾 Saved converted model to: {output_path}")

# Load as proper classifier
print("🔄 Loading as sequence classifier...")
classifier_model = AutoModelForSequenceClassification.from_pretrained(
    output_path, 
    torch_dtype=torch.float16
)

print(f"✅ Model loaded as: AutoModelForSequenceClassification")
print(f"🎯 Capabilities: rerank (should be available)")

# Start server
print("\n🚀 Starting Infinity server with converted model...")
print("=" * 60)

engine_args = EngineArgs(
    model_name_or_path=output_path,
    batch_size=16,
    device="cpu",
    dtype="float16",
    model_warmup=False,
    bettertransformer=True
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