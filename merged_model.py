"""
Module for merging a fine-tuned model with the base model and saving the merged model.
"""

import os
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_NAME = "openai-community/gpt2"
ADAPTER_MODEL_NAME = "./runs/gpt2-fine-tuned"  # Path to the checkpoint
OUTPUT_DIR = "./docker/app/gpt2-fine-tuned_merged"

# Load the base model and adapter model, merge them, and unload
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="cuda")
model = PeftModel.from_pretrained(model, ADAPTER_MODEL_NAME)
model = model.merge_and_unload()

# Create output directory and save the merged model
os.makedirs(OUTPUT_DIR, exist_ok=True)
model.save_pretrained(OUTPUT_DIR, safe_serialization=False)

# Save tokenizer for easy inference
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(OUTPUT_DIR, safe_serialization=False)
