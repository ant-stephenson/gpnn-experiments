"""
Script to download all datasets and prepare them as numpy array datasets.
"""
import argparse
from functools import partial
import os
import requests
import tarfile
import zipfile


import numpy as np
import pandas as pd

from bayesian_benchmarks.bayesian_benchmarks.data import get_regression_data

TMPDIR = ".tmp_data"

def download_file(url, loc):
    if os.path.exists(loc):
        print("Skipping URL download - already exists.")
        return
    request = requests.get(url, allow_redirects=True)
    with open (loc, "wb") as fout:
        fout.write(request.content)

def download_and_save_numpy(dataset_name, dataset_dir, array_loader):
    out_dir = os.path.join(dataset_dir, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "data.npy")
    if os.path.exists(out_path):
        print(f"Skipping {dataset_name} download - already exists.")
        return
    total_array = array_loader()
    np.save(out_path, total_array)

def bayesian_benchmarks_array_loader(dataset_name):
    dl = get_regression_data(dataset_name)
    x = np.concatenate([dl.X_train, dl.X_test], axis=0)
    y = np.concatenate([dl.Y_train, dl.Y_test], axis=0).reshape((-1, 1))
    total_array = np.concatenate([x, y], axis=1)
    return total_array

def song_array_loader(easy=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00203/YearPredictionMSD.txt.zip"
    download_path = os.path.join(TMPDIR, "song.zip")
    download_file(url, download_path)
    df = pd.read_csv(download_path, header=None)
    reorder_inds = np.concatenate([np.arange(1, len(df.columns)), np.array([0])])
    total_array = df.values[:, reorder_inds]
    return total_array

def bike_array_loader(easy=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
    tmp_download_path = os.path.join(TMPDIR, "bike.zip")
    download_path = os.path.join(TMPDIR, "hour.csv")

    download_file(url, tmp_download_path)

    with zipfile.ZipFile(tmp_download_path, "r") as zip_ref:
        zip_ref.extractall(TMPDIR)
    df = pd.read_csv(download_path, parse_dates=['dteday'])
    df['dteday'] = df['dteday'].apply(lambda x: int(x.strftime('%d')))
    # Instant is just an index field.
    df.drop(columns=["instant"], inplace=True)
    # One should really do this, as it trivialises the problem otherwise.
    df.drop(columns=['registered', "casual"], inplace=True)
    total_array = df.values
    return total_array

def buzz_array_loader(easy=False):
    # TODO: check we're regressing on the correct attribute
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00248/regression.tar.gz"
    tmp_download_path = os.path.join(TMPDIR, "buzz.tar.gz")
    download_path = os.path.join(TMPDIR, "regression", "Twitter", "Twitter.data")

    download_file(url, tmp_download_path)

    tar = tarfile.open(tmp_download_path, "r:gz")
    tar.extractall(TMPDIR)

    df = pd.read_csv(download_path, header=None)
    total_array = df.values
    return total_array

def poletele_array_loader(easy=False):
    url = """https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data"""
    df = pd.read_csv(url)
    if easy:
        df = df.drop(["subject#", "test_time"], axis=1)
        col_inds = np.concatenate([np.array([0, 1, 2]), np.arange(4, df.shape[1]), np.array([3])])
    else:
        df = df.drop(["subject#", "test_time", "motor_UPDRS"], axis=1)
        col_inds = np.concatenate([np.array([0, 1]), np.arange(3, df.shape[1]), np.array([2])])
    data = df.values
    return data[:, col_inds]

def keggdirected_array_loader(easy=False):
    # TODO: check we're regressing on the correct attribute
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00220/Relation%20Network%20(Directed).data"
    df = pd.read_csv(url, header=None)
    df[df == "?"] = np.nan
    df = df.dropna(axis=1)
    data = df.values[:, 1:].astype(float)
    return data

def keggundirected_array_loader(easy=False):
    # TODO: check we're regressing on the correct attribute
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00221/Reaction%20Network%20(Undirected).data"
    df = pd.read_csv(url, header=None)
    df[df == "?"] = np.nan
    df = df.dropna(axis=1)
    data = df.values[:, 1:].astype(float)
    return data


def ctslice_array_loader(easy=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00206/slice_localization_data.zip"
    tmp_download_path = os.path.join(TMPDIR, "ctslice.zip")
    download_path = os.path.join(TMPDIR, "slice_localization_data.csv")

    download_file(url, tmp_download_path)

    with zipfile.ZipFile(tmp_download_path, "r") as zip_ref:
        zip_ref.extractall(TMPDIR)
    df = pd.read_csv(download_path)
    data = df.values[:, 1:]
    return data

def road3d_array_loader(easy=False):
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00246/3D_spatial_network.txt"
    df = pd.read_csv(url, header=None)
    data = df.values[:, 1:]
    return data


BAYESIAN_BENCHMARK_DATASETS = ["protein"]
OTHER_DATASETS = [("song", song_array_loader),
                  ("bike", bike_array_loader),
                  ("buzz", buzz_array_loader),
                  ("poletele", poletele_array_loader),
                  ("keggdirected", keggdirected_array_loader),
                  ("keggundirected", keggundirected_array_loader),
                  ("ctslice", ctslice_array_loader),
                  ("road3d", road3d_array_loader)]




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-harden", action="store_true", help="Some of the datasets contain some features which much reduce the difficulty"
                                                             " of the regression problem. By default these features are included to"
                                                             " ensure similarity with published results, but setting this flag will"
                                                             " remove them.")
    args = parser.parse_args()

    outdir = os.path.join(TMPDIR, "array_datasets")
    os.makedirs(outdir, exist_ok=True)
    for name in BAYESIAN_BENCHMARK_DATASETS:
        download_and_save_numpy(name, outdir, partial(bayesian_benchmarks_array_loader, name))

    for name, loader in OTHER_DATASETS:
        print(f"Getting {name}")
        _loader = loader if args.harden else partial(loader, easy=True)
        download_and_save_numpy(name, outdir, _loader)

