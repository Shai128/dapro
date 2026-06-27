import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from torch import nn

from src.utils.utils import set_seeds


def simulate_converging_probabilities(
        n=100,
        T=100,
        dt=0.1,
        alpha=2.0,  # drift strength
        sigma=0.3,  # noise level
        target_ratio_to_one=0.3,  # fraction of paths expected to converge to 1
        seed=None
):
    if seed is not None:
        np.random.seed(seed)

    P = np.zeros((n, T))
    P[:, 0] = np.random.uniform(0, 0.2, size=n)

    # Determine which processes are biased toward 1
    target_to_one = np.zeros(n)
    num_to_one = int(n * target_ratio_to_one)
    target_to_one[:num_to_one] = 1
    np.random.shuffle(target_to_one)

    for t in range(1, T):
        p_prev = P[:, t - 1]

        # Nonlinear symmetric drift
        drift_core = p_prev * (1 - p_prev) * (2 * p_prev - 1)

        # Directional bias: flip sign for target-to-one
        drift = alpha * drift_core * (-1 + 2 * target_to_one)

        # Add stochastic noise
        noise = sigma * np.sqrt(dt) * np.random.randn(n)

        # Euler update
        p_new = p_prev + drift * dt + noise
        p_new = np.clip(p_new, 0, 1)

        # Freeze absorbed states
        absorbed = (p_prev == 0) | (p_prev == 1)
        p_new[absorbed] = p_prev[absorbed]

        P[:, t] = p_new

    return P


def visualize_data(P=None):
    if P is None:
        P = simulate_converging_probabilities(n=10, T=500, alpha=0.01, sigma=0.1, dt=0.05,
                                              target_ratio_to_one=0.3, seed=42)
    P = P[:10]
    # Plot results
    plt.figure(figsize=(10, 6))
    for i in range(min(50, P.shape[0])):
        plt.plot(P[i], alpha=0.7)
    plt.title("Event probabilities")
    plt.xlabel("Time Step")
    plt.ylabel("Probability")
    plt.ylim(0, 1)
    plt.grid(True)
    plt.show()


class TCNEncoder(nn.Module):
    def __init__(self, in_ch, num_channels, kernel_size=3, dropout=0.1):
        super().__init__()
        layers = []
        for i, out_ch in enumerate(num_channels):
            dilation = 2 ** i
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size, padding=(kernel_size - 1) * dilation, dilation=dilation),
                       nn.ReLU(), nn.Dropout(dropout)]
            in_ch = out_ch
        self.network = nn.Sequential(*layers)

    def forward(self, x):  # x: (B, T, C)
        x = x  # (B, C, T)
        return self.network(x)


def generate_syn_data(train_size=4000, cal_size = 4000, test_size = 4000):
    set_seeds(0)
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    print(f"BASE_DIR: {BASE_DIR}")

    data_store_dir = os.path.join(BASE_DIR, 'datasets/synthetic_data')
    if os.path.exists(f"{data_store_dir}/p_train.npy"):
        p_train = np.load(f"{data_store_dir}/p_train.npy", allow_pickle=True)[:train_size]
        p_cal = np.load(f"{data_store_dir}/p_cal.npy", allow_pickle=True)[:cal_size]
        p_test = np.load(f"{data_store_dir}/p_test.npy", allow_pickle=True)[:test_size]

        x_train = np.load(f"{data_store_dir}/x_train.npy", allow_pickle=True)[:train_size]
        x_cal = np.load(f"{data_store_dir}/x_cal.npy", allow_pickle=True)[:cal_size]
        x_test = np.load(f"{data_store_dir}/x_test.npy", allow_pickle=True)[:test_size]

        y_train = np.load(f"{data_store_dir}/y_train.npy", allow_pickle=True)[:train_size]
        y_cal = np.load(f"{data_store_dir}/y_cal.npy", allow_pickle=True)[:cal_size]
        y_test = np.load(f"{data_store_dir}/y_test.npy", allow_pickle=True)[:test_size]

        t_tilde_train = np.load(f"{data_store_dir}/t_tilde_train.npy", allow_pickle=True)[:train_size]
        t_tilde_cal = np.load(f"{data_store_dir}/t_tilde_cal.npy", allow_pickle=True)[:cal_size]
        t_tilde_test = np.load(f"{data_store_dir}/t_tilde_test.npy", allow_pickle=True)[:test_size]

        e_train = np.load(f"{data_store_dir}/e_train.npy", allow_pickle=True)[:train_size]
        e_cal = np.load(f"{data_store_dir}/e_cal.npy", allow_pickle=True)[:cal_size]
        e_test = np.load(f"{data_store_dir}/e_test.npy", allow_pickle=True)[:test_size]

        b_train = np.load(f"{data_store_dir}/b_train.npy", allow_pickle=True)[:train_size]
        b_cal = np.load(f"{data_store_dir}/b_cal.npy", allow_pickle=True)[:cal_size]
        b_test = np.load(f"{data_store_dir}/b_test.npy", allow_pickle=True)[:test_size]

        n_samples_train = np.load(f"{data_store_dir}/n_samples_train.npy", allow_pickle=True)[:train_size]
        n_samples_cal = np.load(f"{data_store_dir}/n_samples_cal.npy", allow_pickle=True)
        n_samples_test = np.load(f"{data_store_dir}/n_samples_test.npy", allow_pickle=True)
        return (
            p_train, p_cal, p_test, x_train, x_cal, x_test, y_train, y_cal, y_test, t_tilde_train, t_tilde_cal,
            t_tilde_test,
            e_train, e_cal, e_test, b_train, b_cal, b_test, n_samples_train, n_samples_cal, n_samples_test)
    n_x = 10000

    # starting_p = np.random.uniform(low=0.0, high=1.0, size=n_x)
    # n_1 = int(0.5 * n_x)
    # n_0 = n_x - n_1
    # is_1_idx = np.random.choice(np.arange(len(starting_p)), size=n_1, replace=False)
    # is_1 = np.zeros(len(starting_p))
    # is_1[is_1_idx] = 1.0
    n_time_steps = 20
    p = simulate_converging_probabilities(n=n_x, T=n_time_steps, alpha=0.01, sigma=0.1, dt=0.05,
                                          target_ratio_to_one=0.2, seed=42)
    # Determine probabilities p:
    # Create two log-spaced arrays and concatenate them
    # p1 = np.logspace(-3, -2, (n_x * 9) // 10)
    # p2 = np.logspace(-5, -4, n_x - len(p1))
    # p = np.concatenate([p1, p2])

    # Set covariates x:
    input_size = 10
    taus = np.linspace(0.1, 0.9, input_size)
    # For each quantile level, compute the geometric quantile (raised to the 0.25 power) and stack as columns.
    # x = p[:, np.newaxis, :].repeat(input_size, 1) #np.stack([stats.geom.ppf(tau, p*0.95+1e-5) ** 0.25 for tau in taus], axis=1)
    x = np.stack(
        [(stats.geom.ppf(tau, p * 0.95 + 1e-5) + 5 * np.random.gamma(shape=2.0, scale=0.5, size=p.shape)) ** 0.25 for
         tau in taus], axis=1)
    # normalization = x.mean(axis=1)[:, np.newaxis]
    # normalization[normalization == 0] = 1
    # x /= normalization
    sigma_squared = 0.2
    # x += np.random.normal(0, sigma_squared, x.shape) + 2*np.random.beta(a=2, b=2, size=x.shape)  # uncalibrated overcovers -- good
    x -= x.mean(axis=0)
    x += np.random.normal(0, sigma_squared, x.shape)
    # n_x, input_size, _ = x.shape  # Update n_x if needed

    # For each observation, sample a random number of trials from a uniform distribution between 1 and 1000.
    n_samples = n_time_steps * np.ones(n_x, dtype=int)

    # Generate binary outcomes for each observation:
    # For each observation, perform binomial trials with probability p and sample size (n_samples)
    y = np.random.binomial(1, p)

    # Sum the successes for each observation.
    b = np.array([yi.sum() for yi in y])

    # Compute the "first success" (t_tilde) for each observation using a geometric random variable;
    # then cap it at the number of trials.
    # t_tilde = np.random.geometric(p, size=len(p))
    # t_tilde = np.minimum(t_tilde, n_samples)

    t_tilde = np.array([yi.argmax().item() + 1 if any(yi) else yi.shape[0] for yi in y])

    # Event indicator: 1 if there is at least one success, else 0.
    e = np.array([1 if yi.any() else 0 for yi in y])
    x = torch.Tensor(x).permute(0, 2, 1).numpy()
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

    # For use in later plots, sort p_test and adjust corresponding predictions accordingly.
    # sort_idx = np.argsort(p_test)
    # p_test = p_test[sort_idx]
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
