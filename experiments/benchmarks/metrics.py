import torch


def get_mse_sum(y_actual, y_predicted):
    return ((y_actual - y_predicted) ** 2).sum().item()


def get_nll_sum(y_actual, y_predicted, predictive_variance):
    first_term = 0.5 * torch.log(2 * torch.pi * predictive_variance)
    second_term = ((y_actual - y_predicted) ** 2) / (2 * predictive_variance)
    return (first_term + second_term).sum().item()


def get_calibration_sum(y_actual, y_predicted, predictive_variance):
    return (((y_actual - y_predicted) ** 2) / predictive_variance).sum().item()
