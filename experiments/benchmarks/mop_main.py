import gpytorch
import numpy as np
from sklearn.neighbors import NearestNeighbors
import torch

from arguments import parse_mop_args
from data_processing import prepare_data
from gpmodels import ExactGP
from metrics import get_all_metrics


def get_predictions(x_test, x_train, y_train, nn_model, assum_kern, likelihood, ls, ks, nv):
    nn_pred_mean, nn_pred_sd = np.zeros(len(x_test)), np.zeros(len(x_test))
    for i_trial in range(len(x_test)):
        test_point = x_test[np.newaxis, i_trial, :]
        neigh_list = nn_model.kneighbors(test_point, return_distance=False)
        nearest_x = x_train[neigh_list.flatten(), :]
        nearest_y = y_train[neigh_list.flatten()]
        gp = ExactGP(nearest_x, nearest_y, likelihood, assum_kern)
        gp.covar_module.base_kernel.lengthscale, gp.likelihood.noise, gp.covar_module.outputscale = ls, nv, ks
        gp.eval()
        output_nn_model = gp(test_point)
        nn_pred_mean[i_trial], nn_pred_sd[i_trial] = output_nn_model.mean, (output_nn_model.variance + nv) ** 0.5
    return torch.tensor(nn_pred_mean), torch.tensor(nn_pred_sd)


def estimate_params(all_subsets_x, all_subsets_y, subset_size, num_subsets, assum_kern, likelihood, optim_iters, lr):
    model = ExactGP(all_subsets_x, all_subsets_y, likelihood, assum_kern)
    model.train(), likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
    for i in range(optim_iters):
        optimizer.zero_grad()
        loss = 0
        with gpytorch.settings.debug(False):
            for si in range(num_subsets):
                output = model(all_subsets_x[si * subset_size: (si + 1) * subset_size])
                loss -= mll(output, all_subsets_y[si * subset_size: (si + 1) * subset_size])
        loss.backward(retain_graph=(i == optim_iters - 1 or num_subsets == 1))
        optimizer.step()
    return model.covar_module.base_kernel.lengthscale.item(), model.likelihood.noise.item(), model.covar_module.outputscale.item()


def main():
    # prep
    args = parse_mop_args()
    x_train, y_train, x_other, y_other, y_train_r_mean, y_train_r_std, tt_split_hsh = prepare_data(args, method_of_paper=True)
    n_recal = args.recal_data_size
    x_recal, y_recal, x_test, y_test = x_other[:n_recal], y_other[:n_recal], x_other[n_recal:], y_other[n_recal:]
    dim, n_all, n_train, n_test = x_train.shape[1], len(y_train) + len(y_recal) + len(y_test), len(y_train), len(y_test)

    # training
    nn_model = NearestNeighbors(n_neighbors=args.number_nn)
    nn_model.fit(x_train)

    num_subsets = min(n_train // args.subset_size, args.max_nsubsets)
    likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.GreaterThan(1.000E-06))
    x_phase1, y_phase1 = x_train[: num_subsets * args.subset_size], y_train[: num_subsets * args.subset_size]
    est_ls, est_nv, est_ks = estimate_params(x_phase1, y_phase1, args.subset_size, num_subsets, args.assum_kernel_type, likelihood, args.iters, args.lr)

    recal_mean, recal_sd = get_predictions(x_recal, x_train, y_train, nn_model, args.assum_kernel_type, likelihood, est_ls, est_ks, est_nv)
    recal_mse, recal_nll, recal_mscal = get_all_metrics(y_recal, recal_mean, recal_sd ** 2)

    # prediction
    pred_ls, pred_ks, pred_nv = est_ls, recal_mscal * est_ks, recal_mscal * est_nv
    nn_pred_mean_test, nn_pred_sd_test = get_predictions(x_test, x_train, y_train, nn_model, args.assum_kernel_type, likelihood, pred_ls, pred_ks, pred_nv)
    mse, nll, mscal = get_all_metrics(y_test, nn_pred_mean_test, nn_pred_sd_test ** 2)

    # results
    results = {'kernel': args.assum_kernel_type, 'seed': args.tt_seed, 'max_num_subsets': args.max_nsubsets, 'subset_size': args.subset_size,
               'num_neighbors': args.number_nn, 'n_test': n_test, 'n_recal': n_recal, 'preprocessing': args.preprocess_method, 'datafile': args.datafile,
               'iters': args.iters, 'n_all': n_all, 'dim': dim, 'n_train': n_train, 'num_subsets': num_subsets, 'mse': mse, 'rmse': mse ** 0.5, 'nll': nll, 'mscal': mscal}
    for k, v in results.items():
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()
