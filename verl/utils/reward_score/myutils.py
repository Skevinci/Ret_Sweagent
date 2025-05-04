from verl.utils.reward_score import auto_sweagent

def compute_score_wrapper(args):
    sequences_str, ground_truth, do_print = args
    return auto_sweagent.compute_score(solution_str=sequences_str, ground_truth=ground_truth, do_print=do_print)