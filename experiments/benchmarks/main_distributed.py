from collections import OrderedDict
import os

import torch

from arguments import parse_distributed_args
from data_processing import prepare_data
from gpmodel import GPModel
from partitioning import prepare_experts
from prediction import run_predictions, set_to_eval_mode, run_grbcm_predictions
from training import prepare_shared_model_objects, run_distributed_training, get_trained_hyperparams


def main():
    args = parse_distributed_args()

    # prepare data
    x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std, tt_split_hsh = prepare_data(args)
    dim, n_all, n_train, n_test = x_train.shape[1], len(y_train) + len(y_test), len(y_train), len(y_test)

    # prepare experts
    experts_x, experts_y, expert_hsh = prepare_experts(args, x_train, y_train)
    num_experts, min_expert_size, max_expert_size = \
        len(experts_x), min([len(v) for v in experts_x]), max([len(v) for v in experts_x])

    # prepare models
    likelihood, covar, mean = prepare_shared_model_objects()
    models = [GPModel(experts_x[i], experts_y[i], likelihood, covar, mean) for i in range(num_experts)]

    # run training
    train_time = run_distributed_training(args, likelihood, covar, models, experts_x, experts_y)
    output_scale, length_scale, noise_var = get_trained_hyperparams(likelihood, covar)

    # run prediction
    n_batches = 1 + n_test // args.batch_size
    set_to_eval_mode(likelihood, models)
    info = []
    with torch.no_grad():
        if args.model_type in {'all', 'poe'}:
            poe_mse, poe_nll, poe_cal, poe_time = run_predictions(args, x_test, y_test, models, n_batches, noise_var, 'poe')
            info.append(['poe', poe_time, poe_mse, poe_mse ** 0.5, poe_nll, poe_cal])
        if args.model_type in {'all', 'gpoe'}:
            gpoe_mse, gpoe_nll, gpoe_cal, gpoe_time = run_predictions(args, x_test, y_test, models, n_batches, noise_var, 'gpoe')
            info.append(['gpoe', gpoe_time, gpoe_mse, gpoe_mse ** 0.5, gpoe_nll, gpoe_cal])
        if args.model_type in {'all', 'bcm'}:
            bcm_mse, bcm_nll, bcm_cal, bcm_time = run_predictions(args, x_test, y_test, models, n_batches, noise_var, 'bcm')
            info.append(['bcm', bcm_time, bcm_mse, bcm_mse ** 0.5, bcm_nll, bcm_cal])
        if args.model_type in {'all', 'rbcm'}:
            rbcm_mse, rbcm_nll, rbcm_cal, rbcm_time = run_predictions(args, x_test, y_test, models, n_batches, noise_var, 'rbcm')
            info.append(['rbcm', rbcm_time, rbcm_mse, rbcm_mse ** 0.5, rbcm_nll, rbcm_cal])
        if args.model_type in {'all', 'grbcm'}:
            # special treatment for GRBCM due to communications expert
            grbcm_mse, grbcm_nll, grbcm_cal, grbcm_time = run_grbcm_predictions(experts_x, experts_y, likelihood, covar, mean, args, x_test, y_test, n_batches, noise_var)
            info.append(['grbcm', grbcm_time, grbcm_mse, grbcm_mse ** 0.5, grbcm_nll, grbcm_cal])

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
        'num_experts': num_experts,
        'min_expert_size': min_expert_size,
        'max_expert_size': max_expert_size,
        'expert_partition_method': 'random',
        'partition_seed': args.partition_seed,
        'train_time': train_time,
        'output_scale': output_scale,
        'length_scale': length_scale,
        'noise_var': noise_var,
        'pred_batch_size': args.batch_size,
        'num_pred_batches': n_batches,
        'y_raw_mean': y_train_r_mean.item(),
        'y_raw_std': y_train_r_std.item(),
        'tt_split_hash': tt_split_hsh,
        'expert_split_hash': expert_hsh,
        'model_type': None,
        'pred_time': None,
        'mse': None,
        'rmse': None,
        'nll': None,
        'calibration': None
    })

    # print results
    if not os.path.exists(args.output):
        with open(args.output, 'w') as f:
            f.write(','.join(results.keys()) + '\n')

    with open(args.output, 'a') as f:
        train_info = list(results.values())[:-6]
        for model_info in info:
            f.write(','.join([str(v) for v in train_info + model_info]) + '\n')


if __name__ == '__main__':
    main()
