import torch


def aggregate_poe(means, precisions, noise_var):
    pred_var = 1 / precisions.sum(axis=0)
    pred_mean = pred_var * (means * precisions).sum(axis=0)
    pred_var += noise_var
    return pred_mean, pred_var


def aggregate_gpoe(means, precisions, noise_var):
    beta = 1 / len(means)
    pred_var = 1 / (beta * precisions.sum(axis=0))
    pred_mean = pred_var * (beta * means * precisions).sum(axis=0)
    pred_var += noise_var
    return pred_mean, pred_var


def aggregate_bcm(means, precisions, noise_var, kss):
    pred_var = 1 / (precisions.sum(axis=0) + (1 - means.shape[0] / kss))
    pred_mean = pred_var * (means * precisions).sum(axis=0)
    pred_var += noise_var
    return pred_mean, pred_var


def aggregate_rbcm(means, precisions, noise_var, kss):
    beta = 0.5 * (torch.log(kss).reshape((1, -1)) + torch.log(precisions))
    pred_var = 1 / ((beta * precisions).sum(axis=0) + (1 - beta.sum(axis=0)) / kss)
    pred_mean = pred_var * (beta * means * precisions).sum(axis=0)
    pred_var += noise_var
    return pred_mean, pred_var
