from src.dataset_utils.real_data import generate_real_data
from src.dataset_utils.synthetic_data import generate_syn_data

import numpy as np
import torch
import gc  # Garbage Collection


def get_data(is_real, device, dataset_name, data_setup, load_x=True, seed=0):
    print(f"loading data real: {is_real}, dataset_name: {dataset_name}, data_setup: {data_setup}")

    # 1. Load initial Data
    if is_real:
        p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal, t_tilde_test, \
            e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test = generate_real_data(
            dataset_name, data_setup, load_x)
    else:
        p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal, t_tilde_test, \
            e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test = generate_syn_data()
        train_size = 500
    #     p_train, x_train, y_train, t_tilde_train, e_train, b_train, n_samples_train = p_train[:train_size], \
    #         x_train[:train_size], \
    #         y_train[:train_size], \
    #         t_tilde_train[:train_size], \
    #         e_train[:train_size], \
    #         b_train[:train_size], \
    #         n_samples_train[:train_size]
    # # -------------------------------------------------------------------------
    # 2. Shuffle and Re-split Logic
    # -------------------------------------------------------------------------
    # Calculate original split sizes to restore them later
    n_train = len(p_train)
    n_cal = len(p_cal)
    # n_test is implicitly the remainder

    # Concatenate all data arrays
    p_all = np.concatenate([p_train, p_cal, p_test], axis=0)
    y_all = np.concatenate([y_train, y_cal, y_test], axis=0)
    t_tilde_all = np.concatenate([t_tilde_train, t_tilde_cal, t_tilde_test], axis=0)
    e_all = np.concatenate([e_train, e_cal, e_test], axis=0)
    b_all = np.concatenate([b_train, b_cal, b_test], axis=0)
    # Generate shuffled indices based on the seed
    total_samples = len(p_all)

    def shuffle_in_place(arr, s):
        rng = np.random.RandomState(s)
        rng.shuffle(arr)  # In-place operation on the first axis

    if load_x:
        # MEMORY OPTIMIZATION 1: Concatenate into a float32 array immediately
        # float32 = 4 bytes (vs 8 bytes for double). This cuts memory usage by 50%.
        # We pre-allocate the buffer to avoid memory fragmentation.
        # print("Concatenating X (converting to float64)...")

        # Determine shape
        feat_dim = x_train.shape[2]  # 2048
        # Note: We do NOT add the time dimension here. It adds 32GB overhead for a simple index.
        # Add the time feature in your Model's forward() method instead.

        x_all = np.empty((total_samples, x_train.shape[1], feat_dim ), dtype=np.float64)

        # Fill buffer
        x_all[:n_train, :,] = x_train
        del x_train
        gc.collect()

        x_all[n_train:n_train + n_cal, :,] = x_cal
        del x_cal
        gc.collect()
        x_all[n_train + n_cal:, :,] = x_test
        del x_test
        gc.collect()
        # x_all[:, :, -1] = torch.arange(0, x_all.shape[1], 1).unsqueeze(0).repeat(len(x_all), 1).numpy()
        # Delete originals to free ~16-32GB RAM immediately

        shuffle_in_place(x_all, seed)
        x_train = torch.from_numpy(x_all[:n_train])
        x_cal = torch.from_numpy(x_all[n_train:n_train + n_cal])
        x_test = torch.from_numpy(x_all[n_train + n_cal:])

        del x_all
        gc.collect()
    else:
        del x_train
        del x_cal
        del x_test
        gc.collect()
        x_train, x_cal, x_test = None, None, None

    shuffle_in_place(p_all, seed)
    shuffle_in_place(y_all, seed)
    shuffle_in_place(t_tilde_all, seed)
    shuffle_in_place(e_all, seed)
    shuffle_in_place(b_all, seed)

    # Split back into train, cal, test
    p_train, p_cal, p_test = p_all[:n_train], p_all[n_train:n_train + n_cal], p_all[n_train + n_cal:]
    y_train, y_cal, y_test = y_all[:n_train], y_all[n_train:n_train + n_cal], y_all[n_train + n_cal:]
    t_tilde_train, t_tilde_cal, t_tilde_test = t_tilde_all[:n_train], t_tilde_all[n_train:n_train + n_cal], t_tilde_all[
                                                                                                            n_train + n_cal:]
    e_train, e_cal, e_test = e_all[:n_train], e_all[n_train:n_train + n_cal], e_all[n_train + n_cal:]
    b_train, b_cal, b_test = b_all[:n_train], b_all[n_train:n_train + n_cal], b_all[n_train + n_cal:]

    # -------------------------------------------------------------------------

    # 3. Convert to Torch Tensors (Existing Logic)
    p_train, p_cal, p_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal, t_tilde_test, e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test = [
        torch.from_numpy(arr).to(device)
        for arr in (p_train, p_cal, p_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal,
                    t_tilde_test, e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal,
                    n_samples_test)
    ]
    # print("finished data generation")

    return p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal, t_tilde_test, \
        e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test
