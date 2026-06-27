import os

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from tqdm import tqdm

import torch.nn.functional as F
import glob

from src.multi_turn_data_generation.utils.utils import get_records, is_sample_finished

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def generate_real_data(dataset_name, dataset_setup, load_x=True):
    data_store_dir = os.path.join(BASE_DIR, 'datasets/real_data', dataset_name, dataset_setup)
    if os.path.exists(f"{data_store_dir}/n_samples_test.npy"):
        p_train = np.load(f"{data_store_dir}/p_train.npy", allow_pickle=True)
        p_cal = np.load(f"{data_store_dir}/p_cal.npy", allow_pickle=True)
        p_test = np.load(f"{data_store_dir}/p_test.npy", allow_pickle=True)

        if load_x:
            x_train = np.load(f"{data_store_dir}/x_train.npy", allow_pickle=True)
            x_cal = np.load(f"{data_store_dir}/x_cal.npy", allow_pickle=True)
            x_test = np.load(f"{data_store_dir}/x_test.npy", allow_pickle=True)
        else:
            x_train = x_cal = x_test = None

        y_train = np.load(f"{data_store_dir}/y_train.npy", allow_pickle=True)
        y_cal = np.load(f"{data_store_dir}/y_cal.npy", allow_pickle=True)
        y_test = np.load(f"{data_store_dir}/y_test.npy", allow_pickle=True)

        t_tilde_train = np.load(f"{data_store_dir}/t_tilde_train.npy", allow_pickle=True)
        t_tilde_cal = np.load(f"{data_store_dir}/t_tilde_cal.npy", allow_pickle=True)
        t_tilde_test = np.load(f"{data_store_dir}/t_tilde_test.npy", allow_pickle=True)

        e_train = np.load(f"{data_store_dir}/e_train.npy", allow_pickle=True)
        e_cal = np.load(f"{data_store_dir}/e_cal.npy", allow_pickle=True)
        e_test = np.load(f"{data_store_dir}/e_test.npy", allow_pickle=True)

        b_train = np.load(f"{data_store_dir}/b_train.npy", allow_pickle=True)
        b_cal = np.load(f"{data_store_dir}/b_cal.npy", allow_pickle=True)
        b_test = np.load(f"{data_store_dir}/b_test.npy", allow_pickle=True)

        n_samples_train = np.load(f"{data_store_dir}/n_samples_train.npy", allow_pickle=True)
        n_samples_cal = np.load(f"{data_store_dir}/n_samples_cal.npy", allow_pickle=True)
        n_samples_test = np.load(f"{data_store_dir}/n_samples_test.npy", allow_pickle=True)
        return (
            p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal,
            t_tilde_test,
            e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test)
    print(f"data does not exist in {os.path.abspath(f'{data_store_dir}/n_samples_test.npy')}, so creating it")
    embedding_data_dir = f'./results/embedding/{dataset_name}/{dataset_setup}'
    all_prompt_embeddings = []
    all_response_embeddings = []
    search_pattern = os.path.join(embedding_data_dir, "attack_log_*_embeddings.pt")
    valid_files = glob.glob(search_pattern)
    n_files = len(valid_files)
    error_files = []
    for i in tqdm(range(n_files), desc="loading embeddings"):
        file_name = f'attack_log_{i}_embeddings.pt'
        file_path = os.path.join(embedding_data_dir, file_name)
        if not os.path.exists(file_path):
            raise Exception(f"File {file_name} does not exist in {embedding_data_dir}")
        try:
            file = torch.load(file_path, map_location="cpu")
            all_prompt_embeddings.append(file['embeddings'])
            all_response_embeddings.append(file['responses_embeddings'])
        except Exception as e:
            error_files.append(file_path)
    if len(error_files) > 0:
        print(f"error loading files:\n{error_files}")
        raise Exception(f"error loading files")

    save_dir = attack_data_dir = os.path.join("results", dataset_name,dataset_setup) # f'./results/{dataset_name}/{dataset_setup}'
    # save_dir = os.path.join("results", dataset_name,dataset_setup)

    scores = []
    for i in tqdm(range(n_files), desc="loading records and scores"):
        file_name = f'attack_log_{i}.jsonl'
        if not os.path.exists(os.path.join(embedding_data_dir, f'attack_log_{i}_embeddings.pt')) or \
                not os.path.exists(os.path.join(attack_data_dir, file_name)):
            raise Exception(f"File {file_name} does not exist in {attack_data_dir} or in {embedding_data_dir}")
        records = get_records(i, save_dir)
        record_i = 0
        while record_i < len(records):
            # removing duplicating rows, should not really happen
            if records[record_i]['iteration'] != record_i:
                records.pop(record_i)
                all_prompt_embeddings[i] = torch.cat(
                    (all_prompt_embeddings[i][:record_i], all_prompt_embeddings[i][record_i + 1:]), dim=0)
                all_response_embeddings[i] = torch.cat(
                    (all_response_embeddings[i][:record_i], all_response_embeddings[i][record_i + 1:]), dim=0)
                record_i -= 1
            record_i += 1
        assert is_sample_finished(i, save_dir, n_iterations=len(all_response_embeddings[i])), f"print {i} is not finished yet"
        none_idxs = [j for j in range(len(records)) if records[j]['score'] is None]
        for idx in none_idxs:
            print(f"warning, in index {i}, {idx} is None")
            records[idx]['score'] = 0
        scores.append(torch.Tensor(list(map(lambda x: x['score'], records))))

    max_time = max(e.shape[0] for e in all_prompt_embeddings)
    for i in range(len(all_prompt_embeddings)):
        all_prompt_embeddings[i] = F.pad(all_prompt_embeddings[i],
                                         (0, 0, 0, max_time - all_prompt_embeddings[i].shape[0]), "constant", 0)
        all_response_embeddings[i] = F.pad(all_response_embeddings[i],
                                           (0, 0, 0, max_time - all_response_embeddings[i].shape[0]), "constant", 0)

    all_prompt_embeddings = torch.stack(all_prompt_embeddings)
    all_response_embeddings = torch.stack(all_response_embeddings)

    all_x = torch.zeros(all_prompt_embeddings.shape[0], max_time,
                        all_prompt_embeddings.shape[2] * 2)
    all_x[:, :, :all_prompt_embeddings.shape[2]] = all_prompt_embeddings
    all_x[:, 1:, all_prompt_embeddings.shape[2]:] = all_response_embeddings[:, :-1]

    for i in range(len(scores)):
        scores[i] = F.pad(scores[i], (0, max_time - scores[i].shape[0],), "constant", 0)

    all_scores = torch.stack(scores)
    all_y = all_scores == 10
    b = torch.Tensor([yi.sum() for yi in all_y])

    t_tilde = torch.Tensor([yi.float().argmax().item() + 1 if yi.any().item() else yi.shape[0] + 1 for yi in all_y])

    x = torch.Tensor(all_x).numpy()
    e = torch.Tensor([1 if yi.any() else 0 for yi in all_y])
    p = np.zeros(len(all_x))
    y = all_y.numpy()
    t_tilde = t_tilde.numpy()
    e = e.numpy()
    b = b.numpy()
    n_samples = max_time * np.ones(len(x), dtype=int)
    assert x.shape[0] == y.shape[0] and x.shape[1] == y.shape[1]

    (
        p_train,
        p_test,
        x_train,
        x_test,
        y_train,
        y_test,
        t_tilde_train,
        t_tilde_test,
        e_train,
        e_test,
        b_train,
        b_test,
        n_samples_train,
        n_samples_test,
    ) = train_test_split(p, x, y, t_tilde, e, b, n_samples, test_size=0.2, random_state=42)

    (
        p_train,
        p_cal,
        x_train,
        x_cal,
        y_train,
        y_cal,
        t_tilde_train,
        t_tilde_cal,
        e_train,
        e_cal,
        b_train,
        b_cal,
        n_samples_train,
        n_samples_cal,
    ) = train_test_split(p_train, x_train, y_train, t_tilde_train, e_train, b_train, n_samples_train, test_size=0.5,
                         random_state=42)

    os.makedirs(data_store_dir, exist_ok=True)
    np.save(f"{data_store_dir}/p_train", p_train)
    np.save(f"{data_store_dir}/p_cal", p_cal)
    np.save(f"{data_store_dir}/p_test", p_test)
    np.save(f"{data_store_dir}/x_train", x_train)
    np.save(f"{data_store_dir}/x_cal", x_cal)
    np.save(f"{data_store_dir}/x_test", x_test)
    np.save(f"{data_store_dir}/y_train", y_train)
    np.save(f"{data_store_dir}/y_cal", y_cal)
    np.save(f"{data_store_dir}/y_test", y_test)
    np.save(f"{data_store_dir}/t_tilde_train", t_tilde_train)
    np.save(f"{data_store_dir}/t_tilde_cal", t_tilde_cal)
    np.save(f"{data_store_dir}/t_tilde_test", t_tilde_test)
    np.save(f"{data_store_dir}/e_train", e_train)
    np.save(f"{data_store_dir}/e_cal", e_cal)
    np.save(f"{data_store_dir}/e_test", e_test)
    np.save(f"{data_store_dir}/b_train", b_train)
    np.save(f"{data_store_dir}/b_cal", b_cal)
    np.save(f"{data_store_dir}/b_test", b_test)
    np.save(f"{data_store_dir}/n_samples_train", n_samples_train)
    np.save(f"{data_store_dir}/n_samples_cal", n_samples_cal)
    np.save(f"{data_store_dir}/n_samples_test", n_samples_test)

    return (
        p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal,
        t_tilde_test,
        e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test)

