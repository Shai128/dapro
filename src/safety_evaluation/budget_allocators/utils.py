import torch


def sample_c(C_probs, prior_quantile_est):
    C = torch.where(torch.rand(len(prior_quantile_est), device=C_probs.device) < C_probs, prior_quantile_est,
                    0).int().reshape(
        -1, 1)
    return C
