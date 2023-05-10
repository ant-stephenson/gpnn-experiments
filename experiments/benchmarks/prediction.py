import time

import torch

from distributed_aggregations import aggregate_bcm, aggregate_poe, aggregate_gpoe, aggregate_rbcm
from gpmodel import GPModel
from metrics import get_mse_sum, get_calibration_sum, get_nll_sum


def set_to_eval_mode(likelihood, models):
    likelihood.eval()
    for model in models:
        model.eval()


def get_batch_predictions(args, batch_id, x_test, y_test, models):
    batch_left = batch_id * args.batch_size
    batch_right = (batch_id + 1) * args.batch_size
    x_batch = x_test[batch_left:batch_right]
    y_batch = y_test[batch_left:batch_right]
    dists = [model(x_batch) for model in models]
    means = torch.vstack([dist.mean for dist in dists])
    variances = torch.vstack([dist.variance for dist in dists])
    precisions = 1 / variances
    return means, precisions, y_batch


def get_grbcm_prediction_models(experts_x, experts_y, likelihood, covar, mean):
    enhanced_models = []
    for i in range(len(experts_x)):
        if i == 0:  # communications expert
            enh_mod = GPModel(experts_x[0], experts_y[0], likelihood, covar, mean)
            enh_mod.eval()
            enhanced_models = [enh_mod]
        else:  # local experts enhanced with communications expert data
            enhanced_x = torch.vstack([experts_x[0], experts_x[i]])
            enhanced_y = torch.concat([experts_y[0], experts_y[i]])
            enh_mod = GPModel(enhanced_x, enhanced_y, likelihood, covar, mean)
            enh_mod.eval()
            enhanced_models.append(enh_mod)
    return enhanced_models


def run_grbcm_predictions(experts_x, experts_y, likelihood, covar, mean, args, x_test, y_test, n_batches, noise_var):
    start_time = time.time()
    models = get_grbcm_prediction_models(experts_x, experts_y, likelihood, covar, mean)
    mse, nll, cal = 0, 0, 0
    for batch_id in range(n_batches):
        means, variances, betas = [], [], []
        batch_left = batch_id * args.batch_size
        batch_right = (batch_id + 1) * args.batch_size
        x_batch = x_test[batch_left:batch_right]
        y_batch = y_test[batch_left:batch_right]
        for i, model in enumerate(models):
            dist = model(x_batch)
            expert_mean = dist.mean
            expert_var = dist.variance
            means.append(expert_mean)
            variances.append(expert_var)
            if i == 0:
                comm_var = expert_var
            if i < 2:
                betas.append(torch.ones_like(expert_mean))
            else:
                betas.append(0.5 * (torch.log(comm_var) - torch.log(expert_var)))
        means = torch.vstack(means)
        variances = torch.vstack(variances)
        precisions = 1 / variances
        betas = torch.vstack(betas)
        betas[0] = 1 - betas[1:].sum(axis=0)  # simplify the communication expert adjustment
        pred_var = 1 / (precisions * betas).sum(axis=0)
        pred_mean = pred_var * (precisions * betas * means).sum(axis=0)
        pred_var = pred_var + noise_var

        mse += get_mse_sum(y_batch, pred_mean)
        nll += get_nll_sum(y_batch, pred_mean, pred_var)
        cal += get_calibration_sum(y_batch, pred_mean, pred_var)
    n_test = len(y_test)
    return mse / n_test, nll / n_test, cal / n_test, time.time() - start_time


def run_predictions(args, x_test, y_test, models, n_batches, noise_var, model_type):
    start_time = time.time()
    mse, nll, cal = 0, 0, 0
    for batch_id in range(n_batches):
        means, precisions, y_batch = get_batch_predictions(args, batch_id, x_test, y_test, models)

        if model_type == 'poe':
            pred_mean, pred_var = aggregate_poe(means, precisions, noise_var)
        elif model_type == 'gpoe':
            pred_mean, pred_var = aggregate_gpoe(means, precisions, noise_var)
        elif model_type == 'bcm':
            kss = models[0].covar_module(x_test[batch_id*args.batch_size:(batch_id+1)*args.batch_size]).diagonal() + noise_var
            pred_mean, pred_var = aggregate_bcm(means, precisions, noise_var, kss)
        elif model_type == 'rbcm':
            kss = models[0].covar_module(x_test[batch_id*args.batch_size:(batch_id+1)*args.batch_size]).diagonal() + noise_var
            pred_mean, pred_var = aggregate_rbcm(means, precisions, noise_var, kss)
        else:
            raise ValueError("run_predictions only designed for poe, gpoe, bcm, rbcm")

        mse += get_mse_sum(y_batch, pred_mean)
        nll += get_nll_sum(y_batch, pred_mean, pred_var)
        cal += get_calibration_sum(y_batch, pred_mean, pred_var)
    n_test = len(y_test)
    return mse / n_test, nll / n_test, cal / n_test, time.time() - start_time
