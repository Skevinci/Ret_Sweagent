import re
import random
import ast
import operator
import difflib
from zss import simple_distance, Node
from collections import Counter
import time
from multiprocessing import Lock
log_lock = Lock()


def extract_solution(solution_str):
    parts = re.split(r'<\|im_start\|>assistant', solution_str)
    if len(parts) < 3:
        return None

    # Reconstruct the second assistant's message with the marker
    model_output = "<|im_start|>assistant" + parts[-1]
    stop_words = ["</s>", "<|im_end|>", "<|endoftext|>"] 
    for stop_word in stop_words:
        if stop_word in model_output:
            model_output = model_output.split(stop_word)[0].strip()
    
    return model_output

def extract_patch(model_output):
    match = re.search(r'<patch>.*?</patch>', model_output, re.DOTALL)
    return match.group(0).strip() if match else None

def extract_modified_lines(patch):
    """Extracts the modified lines from the patch."""
    lines = patch.splitlines()
    return [line for line in lines if line.startswith("+ ") or line.startswith("- ")]

def validate_patch_format(patch):
    """Validate the format of the patch."""
    lines = patch.splitlines()
    return len(lines) > 1 and lines[0].startswith("<patch>") and lines[1].startswith("diff --git") and lines[-1].startswith("</patch>")

def parse_patch_to_tree(patch):
    """Converts a patch file to a tree structure for comparison."""
    root = Node("patch")
    file_nodes = {}

    for line in patch.split("\n"):
        if line.startswith("diff --git"):
            filename = line.split()[-1]
            file_nodes[filename] = Node(filename)
            root.addkid(file_nodes[filename])
        elif line.startswith("@@"):
            hunk_node = Node(line)
            last_file = list(file_nodes.keys())[-1]
            file_nodes[last_file].addkid(hunk_node)
        elif line.startswith("+") or line.startswith("-"):
            if "hunk_node" in locals():
                hunk_node.addkid(Node(line))

    return root

def compute_tree_similarity(tree1, tree2):
    """Computes edit distance between two trees. (0 to 1)"""
    return 1 / (1 + simple_distance(tree1, tree2))

def compute_line_similarity(gt_patch, gen_patch):
    """Computes line-based similarity using difflib. (0 to 1)"""
    gt_lines = gt_patch.splitlines()
    gen_lines = gen_patch.splitlines()
    
    matcher = difflib.SequenceMatcher(None, gt_lines, gen_lines)
    return matcher.ratio()

def compute_exact_match(gt_patch, gen_patch):
    """Computes exact match between two patches."""
    gt_modified_lines = extract_modified_lines(gt_patch)
    gen_modified_lines = extract_modified_lines(gen_patch)
    
    if gt_modified_lines == gen_modified_lines:
        return 2
    else:
        return 0

def compute_patch_reward(gt_patch, gen_patch, do_print, alpha=0.4, beta=0.1):
    """Computes the final reward score."""
    S_exact = compute_exact_match(gt_patch, gen_patch)
    if S_exact > 0:
        if do_print:
            print(f"Exact match found: {S_exact}")
        return S_exact
    
    # tree1 = parse_patch_to_tree(gt_patch)
    # tree2 = parse_patch_to_tree(gen_patch)
    # S_tree = compute_tree_similarity(tree1, tree2)
    
    reward = compute_line_similarity(gt_patch, gen_patch)

    if do_print:
        print("================================")
        print(f"Reward: {reward}")
    return reward

def compute_score(solution_str, ground_truth, method='strict', format_score=0.1, score=1.):
    """The scoring function for countdown task.
    
    Args:
        solution_str: the solution text
        ground_truth: dictionary containing target number and available numbers
        method: the method to extract the solution
        format_score: the score for correct format but wrong answer
        score: the score for the correct answer
    """
    do_print = random.randint(1, 64) == 1
    # do_print = 1
    model_output = solution_str

    if model_output is None:
        if do_print:
            print(f"No output found")
        return 0
    
    model_output_patch = extract_patch(model_output)
    if model_output_patch is None:
        if do_print:
            print(f"No patch found")
        return 0
    
    if not validate_patch_format(model_output_patch):
        if do_print:
            print(f"Invalid patch format")
        return format_score
    
    ground_truth = f"<patch>\n{ground_truth.strip()}\n</patch>"
    
    return compute_patch_reward(model_output_patch, ground_truth, do_print)
        