# Broadly following https://docs.gpytorch.ai/en/latest/examples/04_Variational_and_Approximate_GPs/SVGP_Regression_CUDA.html

from collections import OrderedDict
import os
import time

import gpytorch
import torch
from torch.utils.data import TensorDataset, DataLoader

from arguments import parse_variational_args
from data_processing import prepare_data
from metrics import get_mse_sum, get_calibration_sum, get_nll_sum
from training import get_trained_hyperparams
from variational_gp import VariationalGPModel


def main_variational():
    args = parse_variational_args()

    # prepare data
    x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std, tt_split_hsh = prepare_data(args)
    dim, n_all, n_train, n_test = x_train.shape[1], len(y_train) + len(y_test), len(y_train), len(y_test)
    train_dataset = TensorDataset(x_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=args.minibatch_training_size, shuffle=True)
    test_dataset = TensorDataset(x_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)

    # prepare model
    inducing_points = x_train[:args.inducing_points, :]
    model = VariationalGPModel(inducing_points=inducing_points)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()
    if torch.cuda.is_available():
        model = model.cuda()
        likelihood = likelihood.cuda()
    model.train()
    likelihood.train()

    # run training
    optimiser = torch.optim.Adam(list(model.parameters()) + list(likelihood.parameters()), lr=args.lr)
    mll = gpytorch.mlls.VariationalELBO(likelihood, model, num_data=y_train.size(0))
    train_start = time.time()
    for epoch in range(args.iters):
        for x_batch, y_batch in train_loader:
            optimiser.zero_grad()
            output = model(x_batch)
            loss = -mll(output, y_batch)
            loss.backward()
            optimiser.step()
    train_time = time.time() - train_start
    output_scale, length_scale, noise_var = get_trained_hyperparams(likelihood, model.covar_module)

    # run prediction
    model.eval()
    likelihood.eval()
    means = torch.tensor([0.])
    variances = torch.tensor([0.])
    n_pred_batches = 0
    pred_start = time.time()
    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            n_pred_batches += 1
            preds = likelihood(model(x_batch))
            means = torch.cat([means, preds.mean.cpu()])
            variances = torch.cat([variances, preds.variance.cpu()])
    pred_time = time.time() - pred_start
    means = means[1:]
    variances = variances[1:]
    y_test = y_test.cpu()
    mse = get_mse_sum(y_test, means) / n_test
    nll = get_nll_sum(y_test, means, variances) / n_test
    cal = get_calibration_sum(y_test, means, variances) / n_test

    # print results
    results = OrderedDict({
        'dataset': args.datafile,
        'dimension': dim,
        'num_points': n_all,
        'num_train': n_train,
        'num_test': n_test,
        'traintest_seed': args.tt_seed,
        'training_iters': args.iters,
        'optimiser': 'Adam',
        'kernel_type': 'RBF',
        'learning_rate': args.lr,
        'pre_processing': args.preprocess_method,
        'num_inducing_points': args.inducing_points,
        'minibatch_size': args.minibatch_training_size,
        'train_time': train_time,
        'output_scale': output_scale,
        'length_scale': length_scale,
        'noise_var': noise_var,
        'pred_batch_size': args.batch_size,
        'num_pred_batches': n_pred_batches,
        'y_raw_mean': y_train_r_mean.item(),
        'y_raw_std': y_train_r_std.item(),
        'tt_split_hash': tt_split_hsh,
        'model_type': 'variational',
        'pred_time': pred_time,
        'mse': mse,
        'rmse': mse ** 0.5,
        'nll': nll,
        'calibration': cal
    })

    if not os.path.exists(args.output):
        headers = results.keys()
        with open(args.output, 'w') as f:
            f.write(','.join(headers) + '\n')

    train_info = results.values()
    with open(args.output, 'a') as f:
        f.write(','.join([str(v) for v in train_info]) + '\n')


if __name__ == '__main__':
    main_variational()
