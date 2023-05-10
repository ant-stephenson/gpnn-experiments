import hashlib

import numpy as np
import torch

SONG_TRAIN = 463715


def standard_scale(x_train_r, y_train_r, x_test_r, y_test_r):
    x_train_r_mean = x_train_r.mean(axis=0)
    x_train_r_std = x_train_r.std(axis=0)
    y_train_r_mean = y_train_r.mean()
    y_train_r_std = y_train_r.std()

    x_train = (x_train_r - x_train_r_mean) / x_train_r_std
    y_train = (y_train_r - y_train_r_mean) / y_train_r_std

    x_test = (x_test_r - x_train_r_mean) / x_train_r_std
    y_test = (y_test_r - y_train_r_mean) / y_train_r_std

    return x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std


def whiten(x_train_r, y_train_r, x_test_r, y_test_r, dim):
    x_train_r_mean = x_train_r.mean(axis=0)
    x_train_centered = x_train_r - x_train_r_mean
    x_train_covar = (x_train_centered.T @ x_train_centered) / x_train_centered.shape[0]
    chol_tri = torch.linalg.cholesky(x_train_covar + torch.eye(x_train_covar.shape[0]) * 1e-5)

    x_train = torch.linalg.solve_triangular(chol_tri, x_train_centered.T, upper=False).T / np.sqrt(dim)
    x_test_centered = x_test_r - x_train_r_mean
    x_test = torch.linalg.solve_triangular(chol_tri, x_test_centered.T, upper=False).T / np.sqrt(dim)

    y_train_r_mean = y_train_r.mean()
    y_train_r_std = y_train_r.std()
    y_train = (y_train_r - y_train_r_mean) / y_train_r_std
    y_test = (y_test_r - y_train_r_mean) / y_train_r_std

    return x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std


def load_raw_data(args):
    data_r = torch.tensor(np.load(args.datafile), dtype=torch.float32)
    x_all_r = data_r[:, :-1]
    y_all_r = data_r[:, -1]
    n_all, dim = x_all_r.shape
    return x_all_r, y_all_r, n_all, dim


def split_train_test(args, x_all_r, y_all_r, n_all):
    inds = np.arange(n_all)
    np.random.seed(args.tt_seed)
    np.random.shuffle(inds)
    hsh = hashlib.md5(np.ascontiguousarray(inds)).hexdigest()
    n_train = int(args.train_prop * n_all)
    n_test = n_all - n_train
    train_inds = inds[:n_train]
    test_inds = inds[n_train:]
    x_train_r = x_all_r[train_inds]
    y_train_r = y_all_r[train_inds]
    x_test_r = x_all_r[test_inds]
    y_test_r = y_all_r[test_inds]
    return x_train_r, y_train_r, x_test_r, y_test_r, n_train, n_test, hsh


def run_preprocessing(args, x_train_r, y_train_r, x_test_r, y_test_r, dim):
    if args.preprocess_method == 'scaling':
        x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std = standard_scale(
            x_train_r, y_train_r, x_test_r, y_test_r)

    elif args.preprocess_method == 'whitening':
        x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std = whiten(
            x_train_r, y_train_r, x_test_r, y_test_r, dim)

    else:
        assert args.preprocess_method == 'none'
        x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std = \
            x_train_r, y_train_r, x_test_r, y_test_r, y_train_r.mean(), y_train_r.std()

    return x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std


def prepare_data(args):
    x_all_r, y_all_r, n_all, dim = load_raw_data(args)
    if 'song' in args.datafile:
        x_train_r, y_train_r, x_test_r, y_test_r, n_train, n_test, hsh = x_all_r[:SONG_TRAIN], y_all_r[:SONG_TRAIN], \
                                                                    x_all_r[SONG_TRAIN:], y_all_r[SONG_TRAIN:], \
                                                                    SONG_TRAIN, len(y_all_r) - SONG_TRAIN, \
                                                                    hashlib.md5(str(SONG_TRAIN)).hexdigest()
    else:
        x_train_r, y_train_r, x_test_r, y_test_r, n_train, n_test, hsh = split_train_test(args, x_all_r, y_all_r, n_all)
    x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std = run_preprocessing(
        args, x_train_r, y_train_r, x_test_r, y_test_r, dim)
    if torch.cuda.is_available():
        x_train, y_train, x_test, y_test = x_train.cuda(), y_train.cuda(), x_test.cuda(), y_test.cuda()
    return x_train, y_train, x_test, y_test, y_train_r_mean, y_train_r_std, hsh
