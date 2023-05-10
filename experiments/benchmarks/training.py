import time

import gpytorch
import torch


def prepare_shared_model_objects():
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    covar = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())
    mean = gpytorch.means.ZeroMean()
    return likelihood, covar, mean


def run_distributed_training(args, likelihood, covar, models, experts_x, experts_y):
    optimiser = torch.optim.Adam(list(covar.parameters()) + list(likelihood.parameters()), lr=args.lr)
    train_start = time.time()
    for _ in range(args.iters):
        optimiser.zero_grad()
        loss = torch.zeros(1)
        for i, model in enumerate(models):
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(model.likelihood, model)
            output = model(experts_x[i])
            loss -= mll(output, experts_y[i]) * len(experts_y[i])
        loss.backward()
        optimiser.step()
    train_time = time.time() - train_start
    return train_time


def get_trained_hyperparams(likelihood, covar):
    return covar.outputscale.item(), covar.base_kernel.lengthscale.item(), likelihood.noise_covar.noise.item()
