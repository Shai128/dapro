from pathlib import Path
from typing import List, Union, Tuple
from sentence_transformers import SentenceTransformer
import json
import os
import torch
from tqdm import tqdm
import argparse

def load_from_jsonl(path: Union[str, Path]) -> Tuple[List[str], List[str]]:
    """
    Load the 'prompt' field from each line in a .jsonl file.
    """
    prompts = []
    responses = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            record = json.loads(line)
            if 'prompt' in record:
                prompts.append(record['prompt'])
            else:
                prompts.append('')
                print(f"Warning: no prompt in path: {path} line: {i}")
            if 'response' in record:
                responses.append(record['response'])
            else:
                responses.append('')
                print(f"Warning: no response in path: {path} line: {i}")
    return prompts, responses


def embed_data(model, prompts, responses):
    formatted = [f"query: {p}" for p in prompts] + [f"passage: {r}" for r in responses]
    embedded = model.encode(formatted, convert_to_tensor=True, batch_size=512)
    embedded_prompts = embedded[:len(prompts)]
    embedded_responses = embedded[len(prompts):]
    return embedded_prompts, embedded_responses





# ... [keep your existing load_from_jsonl, embed_data, and _process_buffer functions here] ...

def main():
    parser = argparse.ArgumentParser(description="Embed prompts from a .jsonl file.")
    parser.add_argument('--model', default='intfloat/multilingual-e5-large')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default=None)
    parser.add_argument('--idx-start', type=int, default=0)
    parser.add_argument('--idx-end', type=int, default=10000)
    args = parser.parse_args()

    model = SentenceTransformer(args.model, device=args.device)

    # Configuration mappings
    datasets = {
        'toxicity': ['detoxify'],
        'red_team': ['llm-judge', 'llama_guard'],
        'hallucination': ['llm-judge'],
        'autoif': ['llm-judge']
    }
    attacker_name = {
        'toxicity': 'toxic',
        'red_team': 'default',
        'hallucination': 'hallucination',
        'autoif': 'autoif_helper'
    }

    target_models = ['qwen25_14b_instruct', 'llama_31_8B_instruct', 'mini_phi_4_instruct', 'gemma3_4b_it']
    attacker_model = 'qwen25_14b_instruct'

    # Build the list of method strings dynamically
    methods = []
    for ds, judges in datasets.items():
        attacker_name = attacker_name[ds]
        for target in target_models:
            for judge in judges:
                method_name = f"attack_{attacker_name}_attack_{attacker_model}_lm_target_{target}_judge_{judge}"
                methods.append({'ds': ds, 'name': method_name})

    for entry in methods:
        dataset_name = f"dataset_{entry['ds']}"
        method = entry['name']

        print(f"--- Starting processing: {dataset_name} | {method} ---")

        source_dir_path = os.path.join('results', dataset_name, method)
        if not os.path.exists(source_dir_path):
            print(f"Skipping: {source_dir_path} not found.")
            continue

        dest_dir_path = os.path.join('results', "embedding", dataset_name, method)
        os.makedirs(dest_dir_path, exist_ok=True)

        file_buffer = []
        chunk_size = 20

        for idx in tqdm(range(args.idx_start, args.idx_end)):
            file = f"attack_log_{idx}.jsonl"
            input_jsonl = os.path.join(source_dir_path, file)
            dest_file_path = os.path.join(dest_dir_path, f"{os.path.splitext(file)[0]}_embeddings.pt")

            if os.path.exists(dest_file_path) or not os.path.exists(input_jsonl):
                continue

            prompts, responses = load_from_jsonl(input_jsonl)
            if not prompts: continue

            file_buffer.append(
                {'dest_path': dest_file_path, 'prompts': prompts, 'responses': responses, 'num_items': len(prompts)})

            if len(file_buffer) >= chunk_size:
                _process_buffer(model, file_buffer)
                file_buffer = []

        if file_buffer:
            _process_buffer(model, file_buffer)


def _process_buffer(model, file_buffer):
    """Helper function to embed a chunk of files and save them individually."""
    all_prompts = []
    all_responses = []

    for item in file_buffer:
        all_prompts.extend(item['prompts'])
        all_responses.extend(item['responses'])

    # GPU encodes everything continuously
    emb_prompts, emb_responses = embed_data(model, all_prompts, all_responses)

    current_idx = 0
    for item in file_buffer:
        n = item['num_items']

        file_emb_prompts = emb_prompts[current_idx: current_idx + n]
        file_emb_responses = emb_responses[current_idx: current_idx + n]

        torch.save({
            'prompts': item['prompts'],
            'responses': item['responses'],
            'responses_embeddings': file_emb_responses.cpu().detach(),
            'embeddings': file_emb_prompts.cpu().detach()
        }, item['dest_path'])

        current_idx += n

if __name__ == '__main__':
    main()
