from datasets import load_dataset
from transformers import AutoTokenizer
import time
import os
import json

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# train_path = "/nlp/data/sikaili/Ret_Sweagent/data/swe_eval/train.parquet"
eval_path = "/nlp/data/sikaili/Ret_Sweagent/data/swe_eval/test.parquet"

# train_dataset = load_dataset("parquet", data_files=train_path, split="train")
eval_dataset = load_dataset("parquet", data_files=eval_path, split="train")

test_path = "/nlp/data/sikaili/Ret_Sweagent/data/auto_sweagent/test.parquet"
test_dataset = load_dataset("parquet", split="train", data_files=test_path)
test_instance_ids = set(test_dataset['instance_id'])
# print("Length of train dataset: ", len(train_dataset))
print("Length of eval dataset: ", len(eval_dataset))
print("Length of test dataset: ", len(test_dataset))

i = 0
for example in eval_dataset:
    if not example["instance_id"] in test_instance_ids:
        # print("example not in test dataset")
        i += 1
        continue
    
print("Number of examples not in test dataset: ", i)
