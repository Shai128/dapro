import json
import os
from typing import Optional, List, Dict, Any


# #save_dir = os.path.join("results", f"dataset_{dataset_name}", f"attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify")
# save_dir = os.path.join("results", f"dataset_{dataset_name}", "attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify")


def get_log_path(idx: int, save_dir: str):
    return os.path.join(save_dir, f"attack_log_{idx}.jsonl")

def get_records(idx: int, save_dir: str) -> Optional[List[Dict[str, Any]]]:
    log_path = get_log_path(idx, save_dir)
    if not os.path.exists(log_path):
        return None
    try:
        with open(log_path, 'r', encoding='utf-8') as f:
            # content = f.read()
            if os.path.getsize(log_path) == 0:
                return None
            # f.seek(0)
            records = []
            for i, line in enumerate(f):
                try:
                    records.append(json.loads(line))
                except Exception as e:
                    print(f"got error in line {i}, line is: {line}")
                    raise
    except Exception as e:
        print(f"got error loading log {log_path} because {e}")
        raise
    return records

def is_sample_finished(idx: int, save_dir: str, n_iterations:int):
    log_path = get_log_path(idx, save_dir)
    if not os.path.exists(log_path):
        return False
    records = get_records(idx, save_dir)
    if records is None:
        return False
    return records[-1]['score'] == 10 or len(records) >= n_iterations


def main():
    i = 0
    n_iterations = 200
    dataset_name = 'toxicity'
    #save_dir = os.path.join("results", f"dataset_{dataset_name}", f"attack_toxic_attack_qwen25_14b_instruct_lm_target_mini_phi_4_instruct_judge_detoxify")
    save_dir = os.path.join("results", f"dataset_{dataset_name}", "attack_toxic_attack_qwen25_14b_instruct_lm_target_qwen25_14b_instruct_judge_detoxify")

    while is_sample_finished(idx=i, save_dir=save_dir, n_iterations=n_iterations):
        i += 1

    last_existing = i-1

    for j in range(last_existing, 5000):
        is_sample_finished(idx=j, save_dir=save_dir, n_iterations=n_iterations) # checking for errors

    print(f"last existing: {last_existing}")

if __name__ == '__main__':
    main()
