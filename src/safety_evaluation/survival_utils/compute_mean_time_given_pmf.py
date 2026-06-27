import torch
import torch.nn.functional as F


def mask_and_renormalize_probs(probs, epsilon=1e-10):
    """
    Mask invalid probabilities (t_future < t_curr) and renormalize.

    Args:
        probs: (B, T_curr, T_future) - raw probabilities (may not sum to 1)
        epsilon: Small constant for numerical stability

    Returns:
        probs_normalized: (B, T_curr, T_future) - masked and renormalized probabilities
    """
    B, T_curr, T_future = probs.shape
    device = probs.device

    # Create future mask: valid only if t_future >= t_curr
    t_curr_grid = torch.arange(T_curr, device=device).unsqueeze(1)  # (T_curr, 1)
    t_future_grid = torch.arange(T_future, device=device).unsqueeze(0)  # (1, T_future)
    future_mask = t_future_grid >= t_curr_grid  # (T_curr, T_future)
    future_mask = future_mask.unsqueeze(0).expand(B, -1, -1)  # (B, T_curr, T_future)

    # Mask invalid probabilities to 0
    probs_masked = probs * future_mask.float()

    # Renormalize using softmax on logits
    # Convert back to logits (inverse of softmax)
    logits = torch.log(probs_masked + epsilon)
    logits_masked = logits.masked_fill(~future_mask, float('-inf'))

    # Softmax to renormalize
    probs_normalized = F.softmax(logits_masked, dim=-1)

    return probs_normalized


def estimate_tail_decay_rate(probs, method='exponential', lookback=3):
    """
    Estimate tail decay rate from the last few observed probabilities.

    Args:
        probs: (B, T_curr, T_future) - normalized probabilities
        method: 'exponential' or 'geometric' - which decay model to fit
        lookback: Number of last time points to use for estimation

    Returns:
        decay_rates: (B, T_curr) - estimated decay parameter for each (batch, t_curr)
    """
    B, T_curr, T_future = probs.shape
    device = probs.device
    T_max = T_future - 1

    # Get probabilities for last 'lookback' observable times
    # probs[:, :, -lookback-1:-1] are the last lookback probabilities before survival class
    if lookback > T_max:
        lookback = T_max

    last_probs = probs[:, :, T_max - lookback:T_max]  # (B, T_curr, lookback)

    if method == 'exponential':
        # Fit: p(t) = p(T_max - lookback) * exp(-λ * (t - (T_max - lookback)))
        # => log(p(t)) = log(p(T_max - lookback)) - λ * (t - (T_max - lookback))
        # Estimate λ from slope of log probabilities

        log_probs = torch.log(last_probs + 1e-10)  # (B, T_curr, lookback)

        # Time indices: 0, 1, 2, ..., lookback-1
        time_idx = torch.arange(lookback, device=device, dtype=torch.float)

        # Linear regression: log_prob = a + b * time
        # We want -b as our decay rate
        mean_time = time_idx.mean()
        mean_log_prob = log_probs.mean(dim=-1, keepdim=True)  # (B, T_curr, 1)

        numerator = ((time_idx - mean_time) * (log_probs - mean_log_prob)).sum(dim=-1)
        denominator = ((time_idx - mean_time) ** 2).sum()

        slope = numerator / (denominator + 1e-10)  # (B, T_curr)
        decay_rate = -slope  # Positive decay rate
        decay_rate = decay_rate.clamp(min=0.01, max=5.0)  # Reasonable bounds

    elif method == 'geometric':
        # Fit: p(t) = p(T_max - lookback) * (1-θ)^(t - (T_max - lookback))
        # => log(p(t)) = log(p(T_max - lookback)) + (t - (T_max - lookback)) * log(1-θ)
        # Similar to exponential, but we estimate θ

        log_probs = torch.log(last_probs + 1e-10)
        time_idx = torch.arange(lookback, device=device, dtype=torch.float)

        mean_time = time_idx.mean()
        mean_log_prob = log_probs.mean(dim=-1, keepdim=True)

        numerator = ((time_idx - mean_time) * (log_probs - mean_log_prob)).sum(dim=-1)
        denominator = ((time_idx - mean_time) ** 2).sum()

        slope = numerator / (denominator + 1e-10)  # log(1-θ)
        # θ = 1 - exp(slope), but we'll return this as the "failure rate" parameter
        theta = 1.0 - torch.exp(slope.clamp(max=0))
        decay_rate = theta.clamp(min=0.01, max=0.99)

    else:
        raise ValueError(f"Unknown method: {method}")

    return decay_rate  # (B, T_curr)


def compute_mean_survival_time(probs, tail_distribution='geometric', tail_param=None):
    """
    Compute E[T | T >= t_curr] with extrapolation for t > T_max.

    Tail distributions for P(T = T_max + k | T > T_max):
    - 'constant': All mass at T_max + 1
    - 'geometric': p(1-p)^(k-1) for k=1,2,3,... (discrete memoryless)
    - 'exponential': p * exp(-λ*k) / Z for k=1,2,3,... (exponential decay)
    - 'power': p * k^(-α) / Z for k=1,2,3,... (heavy tail)
    - 'linear': p * max(0, 1-λk) / Z for k=1,2,3,... (finite support)

    Args:
        probs: (B, T_curr, T_future) where T_future = T_max + 1
        tail_distribution: Type of tail distribution
        tail_param: Decay parameter. If None, estimated from last few probabilities

    Returns:
        mean_times: (B, T_curr)
    """
    B, T_curr, T_future = probs.shape
    device = probs.device
    T_max = T_future - 1

    probs = mask_and_renormalize_probs(probs)

    probs_observed = probs[:, :, :-1]  # (B, T_curr, T_max)
    probs_survival = probs[:, :, -1]  # (B, T_curr)

    time_values = torch.arange(T_max, device=device, dtype=torch.float)
    time_values = time_values.unsqueeze(0).unsqueeze(0)

    mean_observed = (probs_observed * time_values).sum(dim=-1)

    # Estimate tail parameter if not provided
    if tail_param is None:
        if tail_distribution in ['exponential', 'geometric']:
            tail_param = estimate_tail_decay_rate(probs, method=tail_distribution, lookback=3)
        elif tail_distribution == 'power':
            tail_param = 2.0  # Default power law exponent
        elif tail_distribution == 'linear':
            tail_param = 0.1  # Default linear decay rate
        else:
            tail_param = 1.0

    # Make tail_param broadcastable
    if isinstance(tail_param, (int, float)):
        tail_param = torch.tensor(tail_param, device=device, dtype=torch.float)
    if tail_param.dim() == 0:
        tail_param = tail_param.unsqueeze(0).unsqueeze(0).expand(B, T_curr)

    # Compute E[T | T > T_max] based on tail distribution
    if tail_distribution == 'constant':
        mean_tail = T_max + 1.0

    elif tail_distribution == 'geometric':
        # Geometric: P(T = T_max + k) ∝ (1-p)^(k-1) for k=1,2,3,...
        # E[T | T > T_max] = T_max + E[Geometric(p)] = T_max + 1/p
        p = tail_param
        mean_tail = T_max + 1.0 / (p + 1e-10)

    elif tail_distribution == 'exponential':
        # Exponential decay: P(T = T_max + k) ∝ exp(-λk) for k=1,2,3,...
        # Need to normalize: Z = sum_{k=1}^∞ exp(-λk) = exp(-λ)/(1 - exp(-λ))
        # E[k] = sum_{k=1}^∞ k * exp(-λk) / Z
        # Using sum_{k=1}^∞ k*x^k = x/(1-x)^2, we get:
        # E[k] = [exp(-λ)/(1-exp(-λ))] * [1/(1-exp(-λ))] = 1/(exp(λ) - 1)
        lambda_rate = tail_param
        mean_excess = 1.0 / (torch.exp(lambda_rate) - 1.0 + 1e-10)
        mean_tail = T_max + mean_excess

    elif tail_distribution == 'power':
        # Power law: P(T = T_max + k) ∝ k^(-α) for k=1,2,3,...
        # For α > 2: E[k] ≈ ζ(α-1) / ζ(α) where ζ is Riemann zeta
        # Approximation: E[k] ≈ α/(α-2) for α > 2
        alpha = tail_param
        if (alpha > 2).all():
            mean_excess = alpha / (alpha - 2.0 + 1e-10)
        else:
            # For α <= 2, mean is infinite; use large value
            mean_excess = torch.where(alpha > 2, alpha / (alpha - 2.0),
                                      torch.tensor(10.0, device=device))
        mean_tail = T_max + mean_excess

    elif tail_distribution == 'linear':
        # Linear decay: P(T = T_max + k) ∝ max(0, 1 - λk)
        # Support: k ∈ {1, 2, ..., floor(1/λ)}
        # E[k] for uniform-ish distribution
        lambda_rate = tail_param
        max_k = (1.0 / (lambda_rate + 1e-10)).floor()
        # Approximate mean for triangular distribution
        mean_excess = max_k / 3.0  # Rough approximation
        mean_tail = T_max + mean_excess

    else:
        raise ValueError(f"Unknown tail_distribution: {tail_distribution}")

    # Ensure mean_tail is broadcastable to (B, T_curr)
    if isinstance(mean_tail, float):
        mean_tail = torch.tensor(mean_tail, device=device).expand(B, T_curr)
    elif mean_tail.dim() == 0:
        mean_tail = mean_tail.expand(B, T_curr)

    mean_times = mean_observed + probs_survival * mean_tail

    return mean_times


def compute_std_survival_time(probs, tail_distribution='exponential', tail_param=None):
    """
    Compute Std[T | T >= t_curr] with extrapolation for t > T_max.

    Args:
        probs: (B, T_curr, T_future)
        tail_distribution: Type of tail distribution
        tail_param: Decay parameter. If None, estimated from probabilities

    Returns:
        std_times: (B, T_curr)
    """
    B, T_curr, T_future = probs.shape
    device = probs.device
    T_max = T_future - 1

    probs = mask_and_renormalize_probs(probs)

    probs_observed = probs[:, :, :-1]
    probs_survival = probs[:, :, -1]

    time_values = torch.arange(T_max, device=device, dtype=torch.float)
    time_values = time_values.unsqueeze(0).unsqueeze(0)

    # E[T]
    mean_times = compute_mean_survival_time(probs, tail_distribution, tail_param)

    # E[T^2 | t_curr <= T <= T_max-1]
    squared_times = time_values ** 2
    mean_squared_observed = (probs_observed * squared_times).sum(dim=-1)

    # Estimate tail parameter if not provided
    if tail_param is None:
        if tail_distribution in ['exponential', 'geometric']:
            tail_param = estimate_tail_decay_rate(probs, method=tail_distribution, lookback=3)
        elif tail_distribution == 'power':
            tail_param = 2.0
        elif tail_distribution == 'linear':
            tail_param = 0.1
        else:
            tail_param = 1.0

    if isinstance(tail_param, (int, float)):
        tail_param = torch.tensor(tail_param, device=device, dtype=torch.float)
    if tail_param.dim() == 0:
        tail_param = tail_param.unsqueeze(0).unsqueeze(0).expand(B, T_curr)

    # Compute E[T^2 | T > T_max]
    if tail_distribution == 'constant':
        mean_squared_tail = (T_max + 1.0) ** 2

    elif tail_distribution == 'geometric':
        # E[(T_max + k)^2] where k ~ Geometric(p)
        # = T_max^2 + 2*T_max*E[k] + E[k^2]
        # E[k] = 1/p, E[k^2] = (2-p)/p^2
        p = tail_param
        E_k = 1.0 / (p + 1e-10)
        E_k2 = (2.0 - p) / (p ** 2 + 1e-10)
        mean_squared_tail = T_max ** 2 + 2 * T_max * E_k + E_k2

    elif tail_distribution == 'exponential':
        # E[(T_max + k)^2] where P(k) ∝ exp(-λk)
        # E[k] = 1/(exp(λ) - 1)
        # E[k^2] = [exp(λ) + 1] / [exp(λ) - 1]^2
        lambda_rate = tail_param
        exp_lambda = torch.exp(lambda_rate)
        E_k = 1.0 / (exp_lambda - 1.0 + 1e-10)
        E_k2 = (exp_lambda + 1.0) / ((exp_lambda - 1.0) ** 2 + 1e-10)
        mean_squared_tail = T_max ** 2 + 2 * T_max * E_k + E_k2

    elif tail_distribution == 'power':
        # For power law, variance can be large or infinite
        # Rough approximation
        alpha = tail_param
        E_k = alpha / (alpha - 2.0 + 1e-10)
        # Var[k] large for α close to 2, use approximation
        E_k2 = E_k ** 2 * 2  # Rough approximation
        mean_squared_tail = T_max ** 2 + 2 * T_max * E_k + E_k2

    elif tail_distribution == 'linear':
        # Triangular-ish distribution
        lambda_rate = tail_param
        max_k = (1.0 / (lambda_rate + 1e-10)).floor()
        E_k = max_k / 3.0
        E_k2 = (max_k ** 2) / 5.0  # Rough approximation
        mean_squared_tail = T_max ** 2 + 2 * T_max * E_k + E_k2

    else:
        raise ValueError(f"Unknown tail_distribution: {tail_distribution}")

    if isinstance(mean_squared_tail, float):
        mean_squared_tail = torch.tensor(mean_squared_tail, device=device).expand(B, T_curr)
    elif mean_squared_tail.dim() == 0:
        mean_squared_tail = mean_squared_tail.expand(B, T_curr)

    mean_squared_total = mean_squared_observed + probs_survival * mean_squared_tail
    variance = mean_squared_total - mean_times ** 2
    std_times = torch.sqrt(variance.clamp(min=0))

    return std_times


def compute_quantile_survival_time(probs, quantile=0.5, tail_distribution='exponential',
                                   tail_param=None):
    """
    Compute quantile of T | T >= t_curr with extrapolation for t > T_max.

    Args:
        probs: (B, T_curr, T_future)
        quantile: Quantile level (0 to 1)
        tail_distribution: Type of tail distribution
        tail_param: Decay parameter. If None, estimated from probabilities

    Returns:
        quantile_times: (B, T_curr)
    """
    B, T_curr, T_future = probs.shape
    device = probs.device
    T_max = T_future - 1

    probs = mask_and_renormalize_probs(probs)

    probs_observed = probs[:, :, :-1]
    probs_survival = probs[:, :, -1]

    # CDF for observed times
    cdf_observed = torch.cumsum(probs_observed, dim=-1)
    cdf_at_tmax = cdf_observed[:, :, -1]  # CDF just before survival class

    # Find empirical quantile
    quantile_mask = cdf_observed >= quantile
    quantile_indices = quantile_mask.float().argmax(dim=-1)
    time_values = torch.arange(T_max, device=device, dtype=torch.float)
    empirical_quantile = time_values[quantile_indices]

    # Check where we need tail extrapolation
    tail_needed = cdf_at_tmax < quantile

    # Estimate tail parameter if not provided
    if tail_param is None:
        if tail_distribution in ['exponential', 'geometric']:
            tail_param = estimate_tail_decay_rate(probs, method=tail_distribution, lookback=3)
        elif tail_distribution == 'power':
            tail_param = 2.0
        elif tail_distribution == 'linear':
            tail_param = 0.1
        else:
            tail_param = 1.0

    if isinstance(tail_param, (int, float)):
        tail_param = torch.tensor(tail_param, device=device, dtype=torch.float)
    if tail_param.dim() == 0:
        tail_param = tail_param.unsqueeze(0).unsqueeze(0).expand(B, T_curr)

    # Compute tail quantile
    # Need: P(T_max < T <= T_max + k | T > T_max) = (quantile - cdf_at_tmax) / probs_survival
    target_prob = (quantile - cdf_at_tmax) / (probs_survival + 1e-10)
    target_prob = target_prob.clamp(0, 0.9999)

    if tail_distribution == 'constant':
        tail_quantile = torch.full_like(empirical_quantile, T_max + 1.0)

    elif tail_distribution == 'geometric':
        # Geometric CDF: P(k <= K) = 1 - (1-p)^K
        # Solve: 1 - (1-p)^K = target_prob
        # => K = log(1 - target_prob) / log(1-p)
        p = tail_param
        K = torch.log(1 - target_prob + 1e-10) / torch.log(1 - p + 1e-10)
        tail_quantile = T_max + K.clamp(min=1)

    elif tail_distribution == 'exponential':
        # CDF: P(k <= K) = sum_{j=1}^K exp(-λj) / Z
        # For continuous approx: 1 - exp(-λK) ≈ target_prob * (1 - exp(-λ))
        # => K ≈ -log(1 - target_prob * (1 - exp(-λ))) / λ
        lambda_rate = tail_param
        # Discrete version: solve numerically or use approximation
        # Approximation: K ≈ -log(1 - target_prob) / lambda_rate
        K = -torch.log(1 - target_prob + 1e-10) / (lambda_rate + 1e-10)
        tail_quantile = T_max + K.clamp(min=1)

    elif tail_distribution == 'power':
        # Power law CDF is complex; use approximation
        # For large k: F(k) ≈ 1 - k^(1-α)
        alpha = tail_param
        # Rough inverse: k ≈ (1 - target_prob)^(1/(1-α))
        K = (1 - target_prob + 1e-10) ** (1.0 / (1.0 - alpha + 1e-10))
        tail_quantile = T_max + K.clamp(min=1)

    elif tail_distribution == 'linear':
        # Linear decay stops at 1/λ
        # CDF is quadratic; use approximation
        lambda_rate = tail_param
        max_k = (1.0 / (lambda_rate + 1e-10)).floor()
        K = target_prob * max_k  # Linear approximation
        tail_quantile = T_max + K.clamp(min=1)

    else:
        raise ValueError(f"Unknown tail_distribution: {tail_distribution}")

    quantile_times = torch.where(tail_needed, tail_quantile, empirical_quantile)

    return quantile_times


def compute_median_survival_time(probs, tail_distribution='exponential', tail_param=None):
    """Compute median (50th percentile) of T | T >= t_curr."""
    return compute_quantile_survival_time(probs, quantile=0.5,
                                          tail_distribution=tail_distribution,
                                          tail_param=tail_param)