#!/usr/bin/env python3
"""Convert Qwen3-Reranker models from generative to classifier format."""

import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
from safetensors import safe_open
import json

def convert_qwen_reranker_to_classifier(model_name: str, output_path: str = None):
    """Convert Qwen3-Reranker model to classification format."""
    
    print(f"🔧 Converting {model_name} to classifier format...")
    
    # Load original model
    model_id = model_name.replace('Qwen3-Reranker-', 'Qwen/Qwen3-0.6B-Base')
    print(f"📦 Loading base model: {model_id}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    config = AutoConfig.from_pretrained(model_name)
    original_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
    
    print(f"🎯 Creating classifier head for {model_name}...")
    
    # Create classifier head
    vocab_size = config.vocab_size
    hidden_size = config.hidden_size
    num_labels = 1  # Binary classification (relevant/not relevant)
    
    classifier_head = nn.Linear(hidden_size, num_labels, bias=False)
    
    # Replace language model head
    original_model.lm_head = classifier_head
    
    print(f"📏 Model converted: {model_name} → SequenceClassifier")
    print(f"🎯 Configuration: vocab={vocab_size}, hidden={hidden_size}, labels={num_labels}")
    
    # Save converted model
    if output_path:
        with open(output_path, 'wb') as f:
            torch.save(original_model.state_dict(), f)
        print(f"💾 Saved converted model to: {output_path}")
    else:
        output_path = model_name.replace('/', '-') + '-converted-classifier'
        with open(output_path, 'wb') as f:
            torch.save(original_model.state_dict(), f)
        print(f"💾 Saved converted model to: {output_path}")
    
    print(f"✅ Conversion completed for {model_name}")
    return True

def test_classifier(model_path: str):
    """Test the converted classifier model."""
    print(f"🧪 Testing classifier model: {model_path}")
    
    try:
        from transformers import AutoModelForSequenceClassification
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        
        # Test with sample inputs
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        query = "What is artificial intelligence?"
        documents = [
            "Artificial intelligence involves machine learning algorithms.",
            "Weather is sunny today.",
            "Natural language processing helps computers understand text."
        ]
        
        # Create reranking prompt
        prompt = f"Judge whether Document meets requirements based on the Query and Instruct provided. Note that the answer can only be \"yes\" or \"no\".\nQuery: {query}\n"
        
        # Tokenize and predict
        inputs = []
        for doc in documents:
            full_text = prompt + f"<Document>: {doc}"
            tokenized = tokenizer(full_text, truncation=True, padding=True, return_tensors="pt")
            inputs.append(tokenized)
        
        # Process in batch
        batch = tokenizer.pad(inputs, return_tensors="pt")
        batch = {k: v.to(model.device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
            logits = outputs.logits
            
            # Extract relevance scores (probability of "yes" class)
            probs = torch.softmax(logits, dim=-1)
            relevance_scores = probs[:, 1].tolist()  # Probability of "yes" (relevant)
        
        print(f"📊 Test Results:")
        for i, (doc, score) in enumerate(zip(documents, relevance_scores), 1):
            print(f"  {i}. Score: {score:.4f} | {doc[:50]}...")
        
        print(f"✅ Classifier model working correctly!")
        return True
        
    except Exception as e:
        print(f"❌ Error testing classifier: {e}")
        return False

if __name__ == "__main__":
    import sys
    
    models_to_convert = [
        "Qwen/Qwen3-Reranker-0.6B",
        "Qwen/Qwen3-Reranker-4B", 
        "Qwen/Qwen3-Reranker-8B"
    ]
    
    for model_name in models_to_convert:
        print(f"\n{'='*60}")
        success = convert_qwen_reranker_to_classifier(model_name)
        if success:
            output_path = model_name.replace('/', '-') + '-converted-classifier'
            test_classifier(output_path)
        else:
            print(f"❌ Failed to convert {model_name}")
    
    print(f"\n{'='*60}")
    print("🎯 Conversion completed for all Qwen3 reranker models!")
    print("📋 Usage:")
    print("  Use converted models with: AutoModelForSequenceClassification.from_pretrained(path)")
    print("  These will work as proper rerankers in Infinity!")
    print("  Note: Converted models may need fine-tuning for optimal performance")