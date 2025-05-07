from datasets import load_dataset
from transformers import AutoTokenizer
import time
import os
import json

# tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

# train_path = "/nlp/data/sikaili/Ret_Sweagent/data/swe_eval/train.parquet"
test_path = "/nlp/data/sikaili/Ret_Sweagent/data/swe_eval/test.parquet"

# train_dataset = load_dataset("parquet", data_files=train_path, split="train")
test_dataset = load_dataset("parquet", data_files=test_path, split="train")

# print("Length of train dataset: ", len(train_dataset))
print("Length of test dataset: ", len(test_dataset))

i = 0
unembedded_file = "/nlp/data/sikaili/Ret_Sweagent/unembedded_test.json"
for data in test_dataset:
    repo = data['repo']
    commit = data['base_commit']
    print(data['instance_id'])
    done_flag = f'/nlp/data/sikaili/SWE-bench-former/repos_test/{repo}/embedding_flags/{commit}.done'
    if not os.path.exists(done_flag):
        print(f"Embedding flag not found for {repo} {commit}")
        with open(unembedded_file, "a") as f:
            repo_commit = {
                "repo": repo,
                "commit": commit
            }
            json.dump(repo_commit, f)
            f.write("\n")
        i += 1
        continue
print(f"Total number of invalid data: {i}")
