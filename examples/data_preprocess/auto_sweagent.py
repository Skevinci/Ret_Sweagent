import re
import os
import json
from datasets import Dataset, load_dataset
from random import randint, seed, choice
from typing import List, Tuple
from tqdm import tqdm
from verl.utils.hdfs_io import copy, makedirs
from transformers import AutoTokenizer
import argparse


SYSTEM_PROMPT = "<|im_start|>system\nYou will be provided with a partial code base and an issue statement explaining a problem to resolve. Please reason step by step, and respond with a single patch file that I can apply directly to this repository using git apply. Here is a patch file example:\n<patch>\ndiff --git a/example.py b/example.py\n--- a/example.py\n+++ b/example.py\n@@ -1,5 +1,5 @@\n def add(a, b):\n-    return a + b\n+    return a + b + 1\n \n def greet():\n-    print(\"Hello, World!\")\n+    print(\"Hello, OpenAI!\")\n</patch>\nPlease generate your final patch file inside <patch></patch> tags.\n<|im_end|>\n<|im_start|>user\n"

SYSTEM_PROMPT_ACR = """<|im_start|>system\nYou are a helpful assistant. You are a software developer maintaining a large project.
You are working on an issue submitted by the user to your project.
The issue contains a description marked between <issue> and </issue>.
Your task is to first generate a concise query to retrieve relevant and sufficient code context for resolving the issue.
The collected context will later be sent to you for writing a patch.
Do not worry about test files or writing test; you are only interested in crafting a patch.\n<|im_end|>\n<|im_start|>user\n
"""

SYSTEM_PROMPT_BASE = """A conversation between User and Assistant. The assistant is a software developer maintaining a large project. The assistant is working on an issue submitted by the user. The issue contains a description marked between <issue> and </issue>. The assistant first generates a concise query. The user retrieves relevant and sufficient code context based on the query. The assistant then generates a patch to resolve the issue.

User: 
"""

QUERY_PROMPT = "Based on the files, classes, methods, and code statements from the issue related to the bug, you should now generate a query, which will be used to retrieve more context of the project.\n\nThe query should be concise and no more than 100 words.<|im_end|>\n<|im_start|>assistant\nQuery: "



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='/nlp/data/sikaili/Ret_Sweagent/data/auto_sweagent')
    parser.add_argument('--hdfs_dir', default=None)
    parser.add_argument('--num_samples', type=int, default=100000)
    parser.add_argument('--num_operands', type=int, default=6)
    parser.add_argument('--max_target', type=int, default=1000)
    parser.add_argument('--min_number', type=int, default=1)
    parser.add_argument('--max_number', type=int, default=100)
    parser.add_argument('--train_size', type=int, default=327680)
    parser.add_argument('--test_size', type=int, default=1024)
    parser.add_argument('--template_type', type=str, default='base')

    args = parser.parse_args()

    data_source = 'auto_sweagent'
    
    train_dataset = load_dataset("princeton-nlp/SWE-bench", split="train")
    test_dataset = load_dataset("princeton-nlp/SWE-bench", split="test")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B")

    def make_map_fn(split):
        def process_fn(example, idx):
            ground_truth_patch = example['patch']
            problem_stmt = example['problem_statement']
            commit = example['base_commit']
            repo = example['repo'].replace('/', '__')
            
            # whether the embedding is valid
            done_flag = f'/nlp/data/sikaili/SWE-bench-former/repos_{split}/{repo}/embedding_flags/{commit}.done'
            if not os.path.exists(done_flag):
                return {"data_source": "", "prompt": [{"role": "", "content": ""}], "ability": "", "reward_model": {"style": "", "ground_truth": ""}, "extra_info": {"split": "", "index": None}, "base_commit": "", "repo": "", "is_valid": False}
            
            # prepare_issue_prompt following ACR
            # remove markdown comments
            problem_wo_comments = re.sub(r"<!--.*?-->", "", problem_stmt, flags=re.DOTALL)
            content_lines = problem_wo_comments.split("\n")
            # remove spaces and empty lines
            content_lines = [x.strip() for x in content_lines]
            content_lines = [x for x in content_lines if x != ""]
            problem_stripped = "\n".join(content_lines)
            # add tags
            issue_prompt = "<issue>" + problem_stripped + "\n</issue>\n"
            
            init_prompt = SYSTEM_PROMPT_ACR + issue_prompt + QUERY_PROMPT
            
            token_length = len(tokenizer.encode(init_prompt))
            if token_length > 8000:
                return {"data_source": "", "prompt": [{"role": "", "content": ""}], "ability": "", "reward_model": {"style": "", "ground_truth": ""}, "extra_info": {"split": "", "index": None}, "base_commit": "", "repo": "", "is_valid": False}
            
            data = {
                "data_source": data_source,
                "prompt": [{
                    "role": "user",
                    "content": init_prompt,
                }],
                "ability": "reason",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": ground_truth_patch
                },
                "extra_info": {
                    'split': split,
                    'index': idx,
                },
                "base_commit": commit,
                "repo": repo,
                "is_valid": True
            }
            return data
        return process_fn
    
    train_dataset = train_dataset.map(function=make_map_fn('train'), with_indices=True)
    train_dataset = train_dataset.filter(lambda x: x['is_valid'])
    test_dataset = test_dataset.map(function=make_map_fn('test'), with_indices=True)
    test_dataset = test_dataset.filter(lambda x: x['is_valid'])
    print("Length of test dataset: ", len(test_dataset))

    local_dir = args.local_dir
    hdfs_dir = args.hdfs_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    if hdfs_dir is not None:
        makedirs(hdfs_dir)
        copy(src=local_dir, dst=hdfs_dir) 