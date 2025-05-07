import json
import re
import os
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

INPUT_FILE = "./val_rollout/0.json"  # or input.json if it's a list
OUTPUT_FILE = "./predictions/converted_patches.json"
MODEL_NAME = "Qwen2.5-7B-Base"

test_path = "/nlp/data/sikaili/Ret_Sweagent/data/swe_eval/test.parquet"
test_dataset = load_dataset("parquet", data_files=test_path, split="train")

results = []

with open(INPUT_FILE, 'r') as f:
    # Load as list of dicts (jsonl or pure list)
    data = [json.loads(line) for line in f] if INPUT_FILE.endswith('.jsonl') else json.load(f)
    no_patch = 0
    for i, entry in enumerate(data):
        output = entry.get("output", "")
        
        # Extract patch content between ```diff and ```
        match = re.search(r'<patch>(.*?)</patch>', output, re.DOTALL)
        if not match:
            no_patch += 1
            patch = ""  # Skip if no patch found
        
        patch = match.group(1).strip() if match else ""
        instance = {
            "instance_id": test_dataset[i]['instance_id'],
            "model_patch": patch,
            "model_name_or_path": MODEL_NAME
        }
        results.append(instance)
        
print(f"Total number of invalid data: {no_patch}")

# Save result
with open(OUTPUT_FILE, 'w') as f:
    json.dump(results, f, indent=2)