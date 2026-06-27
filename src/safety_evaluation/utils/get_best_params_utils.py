import numpy as np


def get_best_rexp3_params():
    # global global_calibration_set_size
    # global global_budget_per_sample
    # budget_per_sample = global_budget_per_sample
    # calibration_set_size = global_calibration_set_size
    # V_t = 0.2
    # first_step_budget_ratio = 0.2
    # T = calibration_set_size * budget_per_sample * first_step_budget_ratio
    # paper_delta_t = ((calibration_set_size * np.log(calibration_set_size)) ** (1/3) * ((T / V_t) ** (2/3))).item()
    # delta = min(paper_delta_t /T, 1)
    #
    # gamma = min(1., np.sqrt((calibration_set_size * np.log(calibration_set_size)) / ((np.e -1)* paper_delta_t) ).item())
    #
    # gamma_values = np.array([0., 0.001, 0.01, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.6, 0.7, 0.75, 0.8, 0.95, 1.])
    # delta_values = np.array([0., 0.01, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    #
    # gamma = gamma_values[np.argmin(gamma_values - gamma)].item()
    # delta = delta_values[np.argmin(delta_values - delta)].item()

    # return gamma, delta, first_step_budget_ratio
    return 0.1, 0.1, 0.1

def get_best_discounted_ucb_params():
    return 0.1, 0.1, 0.1


def new_alg_best_params():
    # global global_calibration_set_size
    # global global_budget_per_sample
    # budget_per_sample = global_budget_per_sample
    # calibration_set_size = global_calibration_set_size
    # total_budget = calibration_set_size * budget_per_sample
    # first_step_budget_values = np.array([0.01, 0.05, 0.075, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5, 0.75, 0.95])
    #
    # target_first_step_budget = 200
    # first_budget_ratio_idx = np.argmin(abs(first_step_budget_values * total_budget- target_first_step_budget))
    # first_budget_ratio = first_step_budget_values[first_budget_ratio_idx]
    # k = 1

    return 5, 0.1