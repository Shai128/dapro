import gc
import os

import numpy as np
import torch

from src.dataset_utils.data_utils import get_data
from src.dataset_utils.datasets import PartialSequenceDataset
from src.safety_evaluation.utils.utils import multi_time_probabilities_compute_and_save_metrics
from src.train_model.acquisition_strategies.dummy_acquisition import DummyAcquisition
from src.train_model.active_learning import AcquisitionStrategy, ActiveLearner
from src.train_model.models.transformer_survival_model import TransformerSurvivalModel, DiscreteSurvivalLoss
from src.utils.utils import set_seeds


def parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description=""
    )
    parser.add_argument(
        '--seed-start',
        type=int,
        default=0,
    )
    parser.add_argument(
        '--data-type',
        type=str,
        default='real',
    )
    parser.add_argument(
        '--seed-end',
        type=int,
        default=1,
    )
    parser.add_argument(
        '--allocations',
        type=str,
        default='all',
    )
    parser.add_argument(
        '--total-budget',
        type=int,
        default=1000,
    )
    parser.add_argument(
        '--n-seed',
        type=int,
        default=100,
    )
    parser.add_argument(
        '--target-model',
        type=str,
        default='qwen25_14b',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        default='',
    )
    parser.add_argument(
        '--dataset-setup',
        type=str,
        default='',
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda:0',
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
    )
    parser.add_argument(
        '--last-round-epochs',
        type=int,
        default=200,
    )
    parser.add_argument(
        '--acquisition-strategy',
        type=str,
    )
    parser.add_argument(
        '--patience',
        type=int,
        default=50
    )
    parser.add_argument(
        '--acquire-full-time',
        type=int,
        default=0
    )
    args = parser.parse_args()
    if not torch.cuda.is_available() and 'cuda' in args.device or 'gpu' in args.device:
        print("cuda is unavailable, using cpu instead")
        args.device = 'cpu'
    args.device = torch.device(args.device)
    print("device: ", args.device)
    args.is_real = True if args.data_type.lower() == 'real' else False
    if args.acquire_full_time not in [0, 1]:
        raise Exception(f"error, acquire-full-time must be either 0 or 1, got {args.acquire_full_time}")
    args.acquire_full_time = True if args.acquire_full_time == 1 else False
    return args


def get_acquisition_strategy(acquisition_strategy: str, device) -> AcquisitionStrategy:
    if acquisition_strategy == 'naive':
        acq = DummyAcquisition()
    else:
        raise ValueError(f'Acquisition strategy {acquisition_strategy} not recognized')

    return acq


def transformer_run_one_exp(x_train, y_train, t_tilde_train, e_train, x_test, y_test, t_tilde_test, e_test,
                           data_setup, total_budget, n_seed, acquisition_strategy: str,
                           epochs: int, last_round_epochs: int, patience: int,
                           device, seed: int, acquire_full_time: bool):
    dropout = 0.2
    base_experiment_name = 'transformer'
    model_class = lambda: TransformerSurvivalModel(x_train.shape[-1], x_train.shape[1], dropout)
    loss_class = DiscreteSurvivalLoss()
    loss_function = loss_class.forward
    model, pred, results_save_dir, figures_save_dir = run_one_exp(x_train, y_train, t_tilde_train, e_train, x_test, y_test, t_tilde_test, e_test,
                data_setup, total_budget, n_seed, acquisition_strategy, epochs, last_round_epochs, patience,
                device, seed, acquire_full_time, model_class, loss_function, base_experiment_name)
    multi_time_probabilities_compute_and_save_metrics(pred, t_tilde_test, e_test, results_save_dir)


def get_samples_acquire_batch_size(acquire_full_time: bool, total_budget: int, T: int):
    samples_acquire_batch_size = 32
    n_repeated_samples_acquire = 10
    acquire_per_round = samples_acquire_batch_size * n_repeated_samples_acquire
    target_rounds = 24
    if acquire_full_time:
        rounds = int(np.ceil((total_budget / T) / acquire_per_round))
        if rounds < target_rounds:
            rounds = int(np.ceil(min((total_budget / T), target_rounds)))
        samples_acquire_batch_size = int(np.ceil(total_budget / rounds / T))
    else:
        rounds = min(total_budget, min(target_rounds, int(total_budget / acquire_per_round)))
        if rounds < target_rounds:
            rounds = int(np.ceil(min(total_budget, target_rounds)))
        samples_acquire_batch_size = int(np.ceil(total_budget / rounds))
    return samples_acquire_batch_size, rounds, n_repeated_samples_acquire

def observe_samples_for_naive_sampling(acquire_full_time, training_data, total_budget, T, pool_idx):
    if acquire_full_time:
        perm = np.random.permutation(len(pool_idx))
        to_acquire = min(total_budget // T, len(pool_idx))
        training_data.set_fully_observed(pool_idx[perm[:to_acquire]])
        budget_left = total_budget - to_acquire
        if budget_left > 0 and to_acquire < len(pool_idx):
            for _ in range(0, budget_left):
                training_data.observe_next_step([pool_idx[perm[to_acquire]]])
    else:
        budget_left = total_budget
        while budget_left > 0 and len(pool_idx) > 0:
            idx = np.random.randint(0, len(pool_idx))
            training_data.observe_next_step([pool_idx[idx]])
            budget_left -= 1

def run_one_exp(x_train, y_train, t_tilde_train, e_train, x_test, y_test, t_tilde_test, e_test,
                data_setup, total_budget, n_seed, acquisition_strategy: str, epochs: int, last_round_epochs: int,
                patience: int, device, seed: int, acquire_full_time: bool, model_class, loss_function,
                base_experiment_name: str):
    set_seeds(seed)
    N, T = x_train.shape[:2]
    samples_acquire_batch_size, rounds, n_repeated_samples_acquire = get_samples_acquire_batch_size(acquire_full_time,
                                                                                                    total_budget, T)
    if acquire_full_time:
        experiment_name = f'{base_experiment_name}_ft'
    else:
        experiment_name = base_experiment_name
    val_split = 0.1
    n_val = int(val_split * N)
    n_train = N - n_val

    if n_seed > n_train:
        raise Exception(f"n_seed is too high, it should be at most: {n_train}")

    perm = np.random.permutation(N)
    seed_idx = perm[:n_seed]
    val_idx = perm[n_seed:n_seed + n_val]
    pool_idx = perm[n_seed + n_val:]
    training_data = PartialSequenceDataset(x_train, y_train, t_tilde_train, dataset_name=data_setup, initial_obs_len=1)
    training_data.initialize_seed(seed_idx)
    training_data.set_fully_observed(val_idx)
    initial_total_budget = total_budget
    if acquisition_strategy == 'naive':
        observe_samples_for_naive_sampling(acquire_full_time, training_data, total_budget, T, pool_idx)
        total_budget = 0
        rounds = 1

    def loss_fn(model_output, batch) -> torch.Tensor:
        (x_prefix, label_objs, idxs) = batch
        delta = (label_objs['is_event']).long()

        return loss_function(model_output, label_objs['censor_time'].to(device),
                             delta.to(device))

    acq = get_acquisition_strategy(acquisition_strategy, device=device)
    learner = ActiveLearner(model_class=model_class, loss_fn=loss_fn, dataset=training_data, seed_indices=seed_idx,
                            val_indices=val_idx, pool_indices=pool_idx, acquisition=acq, device=device,
                            acquire_full_time=acquire_full_time,
                            retrain_from_scratch=True, verbose=False)
    save_path_suffix = os.path.join(experiment_name, data_setup, acq.name,
                                    f'seed_{n_seed}_budget_{initial_total_budget}', f"seed={seed}")
    saved_models_dir = os.path.join('./saved_models/al', save_path_suffix)
    tmp_save_dir = os.path.join('./tmp_save/al', save_path_suffix)
    os.makedirs(saved_models_dir, exist_ok=True)
    os.makedirs(tmp_save_dir, exist_ok=True)
    model = learner.run(rounds=rounds, samples_acquire_batch_size=samples_acquire_batch_size,
                        n_repeated_samples_acquire=n_repeated_samples_acquire, max_epochs=epochs, patience=patience,
                        batch_size=64, acquire_batch_size=20000, lr=1e-4, weight_decay=0.001, tmp_save_dir=tmp_save_dir,
                        last_round_epochs=last_round_epochs, saved_models_dir=saved_models_dir, total_budget=total_budget)

    model.eval()
    model = model.to(device)
    model.calibrate(training_data.x[val_idx], training_data.t[val_idx], training_data.delta[val_idx], device=device)
    learner.store_state(model, saved_models_dir, rounds)
    del x_train
    gc.collect()
    x_test = x_test.float()
    x_test = x_test.to(device)
    with torch.no_grad():
        pred = model.predict_proba(x_test)
        # pred = model(x_test)

    results_save_dir = os.path.join('./results/al', save_path_suffix)
    figures_save_dir = os.path.join('./figures/al', save_path_suffix)
    os.makedirs(results_save_dir, exist_ok=True)
    os.makedirs(figures_save_dir, exist_ok=True)

    print('Finished dry-run, outputs in', results_save_dir)
    return model, pred, results_save_dir, figures_save_dir


def main():
    args = parse_args()
    is_real = args.is_real
    device = args.device
    dataset_name = args.dataset_name
    data_setup = args.dataset_setup
    acquisition_strategy = args.acquisition_strategy
    acquire_full_time = args.acquire_full_time
    # data_setup = f"attack_toxic_attack_qwen25_14b_instruct_lm_target_{target_model}_instruct_judge_detoxify"
    if not is_real:
        data_setup = 'synthetic'
        dataset_name = 'synthetic'
    else:
        data_setup = data_setup
        dataset_name = dataset_name

    n_seed = args.n_seed
    epochs = args.epochs
    last_round_epochs = args.last_round_epochs
    total_budget = args.total_budget
    patience = args.patience
    for seed in range(args.seed_start, args.seed_end):
        p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal, t_tilde_test, \
            e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test = get_data(
            is_real, device, dataset_name, data_setup, load_x=True, seed=seed)
        del x_cal
        gc.collect()
        print("loaded data, starting experiment #{}".format(seed))

        transformer_run_one_exp(x_train, y_train, t_tilde_train, e_train, x_test, y_test, t_tilde_test,
                                                 e_test,
                                                 data_setup, total_budget, n_seed, acquisition_strategy,
                                                 epochs, last_round_epochs, patience,
                                                 device, seed, acquire_full_time)


if __name__ == '__main__':
    print("starting main")
    main()
