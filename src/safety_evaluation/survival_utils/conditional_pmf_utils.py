import torch


def get_conditional_pmf(hazards: torch.Tensor, validate_logic: bool = False) -> torch.Tensor:
    """
    Computes the conditional probability mass function (PMF) of the event occurring
    at future times, given survival up to the start of the current time.

    Mathematically, for each pair of (current_time 'c', future_time 'f'), computes:
        P(T = f | T >= c)

    Args:
        hazards (torch.Tensor): A tensor of shape (Batch, Time) containing the
            discrete hazard rates h(t) = P(T = t | T >= t).
            Values should be in [0, 1].
        validate_logic (bool, optional): If True, scripts a slow, iterative implementation
            to verify the correctness of the vectorized computation. Defaults to False.

    Returns:
        torch.Tensor: A tensor of shape (Batch, Time, Time).
            The element at [b, c, f] is the probability that the event occurs at
            time index 'f', given that it had not occurred before time 'c'.
            - For f < c: Value is 0.
            - For f == c: Value is h(c).
            - For f > c: Value is P(T=f | T>=c).
    """
    B, T = hazards.shape
    device = hazards.device

    # --- 1. Precompute Survival and PDF ---
    # S(t) = Product_{i=0}^t (1 - h_i)
    # Probability of surviving *through* time t
    surv_through_t = (1 - hazards).cumprod(dim=1)

    # S(t-1): Probability of surviving up to the *start* of time t
    # We shift surv_through_t to the right and pad with 1.0 (prob of surviving time -1 is 1)
    surv_prev = torch.cat([torch.ones(B, 1, device=device), surv_through_t[:, :-1]], dim=1)

    # Unconditional PDF: P(T = t) = h(t) * S(t-1)
    pdf_uncond = hazards * surv_prev  # Shape: (B, T)

    # --- 2. Vectorized Construction of Conditional Grid ---

    # We want: P(T=f | T>c) = P(T=f) / S(c)
    # Numerator: PDF at future time 'f' -> shape (B, 1, T)
    numerator = pdf_uncond.unsqueeze(1)

    # Denominator: Survival through current time 'c' -> shape (B, T, 1)
    # Used for normalizing the future probabilities.
    denominator = surv_through_t.unsqueeze(2)
    denominator = torch.clamp(denominator, min=1e-8)  # Avoid division by zero

    # Initial Grid: P(T=f | T>c)
    conditional_pmf = numerator / denominator

    # Mask out the past (f <= c for now) to keep upper triangle clean
    # We strictly want f > c for the calculation above
    future_mask = torch.triu(torch.ones(T, T, device=device), diagonal=1).unsqueeze(0)
    conditional_pmf = conditional_pmf * future_mask

    # --- 3. Correction & Diagonal ---

    # Correction: The grid currently contains P(T=f | T > c).
    # We need P(T=f | T >= c).
    # Relation: P(T=f | T >= c) = P(T=f | T > c) * P(T > c | T >= c)
    # And P(T > c | T >= c) is simply (1 - h(c)).
    survival_current_step = (1 - hazards).unsqueeze(2)  # Shape: (B, T, 1)
    conditional_pmf = conditional_pmf * survival_current_step

    # Fill Diagonal: P(T=c | T>=c) is just the hazard h(c)
    diagonal_mask = torch.eye(T, device=device).bool().unsqueeze(0).expand(B, -1, -1)
    conditional_pmf[diagonal_mask] = hazards.flatten()

    # --- 4. Logic Validation (Optional) ---
    if validate_logic:
        _validate_implementation(hazards, conditional_pmf)

    return conditional_pmf


def _validate_implementation(hazards, vectorized_result):
    """
    Internal helper to validate the vectorized implementation against
    an explicit iterative approach.
    """
    B, max_time = hazards.shape
    device = hazards.device
    iterative_result = torch.zeros(B, max_time, max_time, device=device)

    for curr_time in range(max_time - 1):
        # Diagonal
        iterative_result[:, curr_time, curr_time] = hazards[:, curr_time]

        # Scaling factor: P(T > curr | T >= curr)
        p_survive_curr = (1 - hazards[:, curr_time])

        # Denominator: P(T > curr)
        # Note: logic requires product up to curr_time inclusive
        survival_through_curr = (1 - hazards[:, :curr_time + 1]).prod(dim=-1)

        for future_time in range(curr_time + 1, max_time):
            # Numerator: P(T = future)
            # S(future-1) * h(future)
            p_t_future = (1 - hazards[:, :future_time]).prod(dim=-1) * hazards[:, future_time]

            # P(T=f | T>c)
            prob_cond_on_strictly_survived = p_t_future / (survival_through_curr + 1e-8)

            # P(T=f | T>=c)
            iterative_result[:, curr_time, future_time] = prob_cond_on_strictly_survived * p_survive_curr

    # Last element edge case
    iterative_result[:, max_time - 1, max_time - 1] = hazards[:, max_time - 1]

    # Check
    diff = (iterative_result - vectorized_result).abs()
    assert (diff < 1e-4).all(), f"Validation failed! Max diff: {diff.max().item()}"