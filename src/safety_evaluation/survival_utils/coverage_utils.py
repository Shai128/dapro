import torch

def compute_coverage(Q, conditional_grid, true_times, tau):
    # Q = get_quantiles_from_grid(conditional_grid, tau)

    # 2. Gather P(T > Q) and P(T = Q) specific to each sample
    # We need to grab values from the grid at the specific indices Q

    # Create batch indices
    batch_indices = torch.arange(len(true_times), device=true_times.device)

    # Get Cumulative Probability up to Q (CDF[Q])
    cdf = conditional_grid.cumsum(dim=-1)

    # P(T <= Q) is cdf[Q].
    # Therefore P(T > Q) = 1 - cdf[Q]
    prob_le_Q = cdf[batch_indices, 0, Q[:, 0].long()]  # Assuming evaluating at t=0
    prob_gt_Q = 1.0 - prob_le_Q

    # P(T = Q) is just the PDF value at Q
    prob_eq_Q = conditional_grid[batch_indices, 0, Q[:, 0].long()]

    target_prob = 1.0 - tau
    beta = (target_prob - prob_gt_Q) / (prob_eq_Q + 1e-8)

    beta = beta.clamp(0.0, 1.0)

    score_strictly_greater = (true_times > Q[:, 0]).float()

    # beta if exactly equal
    score_equal = abs(true_times - Q[:, 0] < 1e-5).float() * beta
    smoothed_coverage = (score_strictly_greater + score_equal).mean()