#%%
import os
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
import seaborn as sns
from scipy.stats import mode
from itertools import product as cartesian_prod
import functools

import gpybench.plotting as gplot

# os.chdir(Path(".").absolute().parent.joinpath("gpnn-experiments/simulations"))
from plot_utils import *

sns.set_palette(None)
#%%
do_save_plots: bool = True
uci_datasets = ("ctslice", "song", "buzz", "house_electric", "protein", "road3d")
pathbase = "/Users/anthonystephenson/Documents/GitHub/"
# pathstr =
# "/Users/anthonystephenson/Documents/GitHub/mini-project/experiments/sim_gpnn_limits_results.csv"
# pathstr ="/Users/anthonystephenson/Documents/GitHub/gpnn-experiments/sim_gpnn_limits_results_Dim2.csv"
pathstr = "/Users/anthonystephenson/Documents/GitHub/gpnn-experiments/sim_gpnn_limits_results_ctslice_whiten_dimrescale_Exp.csv"
path = Path(".").joinpath(pathstr)

dtype_dict = {'n': int, 'n_test': int, 'd': int, 'm': int, 'seed': int, 'k_true': str, 'k_model': str, 'ks': np.float32, 'ls': np.float32, 'nv': np.float32, 'assum_ks': np.float32, 'assum_ls': np.float32, 'assum_nv': np.float32, 'varypar': str, 'mse': np.float32, 'nll': np.float32, 'mscal': np.float32}

data = pd.read_csv(path, dtype=dtype_dict, header=0, sep=',')
data = data.round(4)
data = data.fillna(-999)
data = data.groupby(["n","d","m","n_test","assum_ls", "k_model", "k_true", "varypar"]).mean().reset_index()

dataset = "".join([x if x in pathstr else "" for x in ["ciq", "oak"] + list(uci_datasets)])

kernel_type = "".join([x if x in pathstr else "RBF" for x in ["Exp"]])
#%%
if dataset in uci_datasets:
    ls = data.ls.unique()
    ks = data.ks.unique().item()
    nv = data.nv.unique().item()
    kernel = kernel_type
    d = data.d.unique().item()
min_n = 1000

data = data.query('(n >= @min_n) & (ls in @ls) & (ks == @ks) & (k_model ==@kernel) ').reset_index()
#%%
idx_cols = ["n", "n_test" ,"d","m","seed","k_true","k_model","ks","ls","nv","assum_ks","assum_ls","assum_nv","varypar"]

nv = get_nv(data,d, ls)
num_near = get_num_near(data)
nv_hat = get_nvhat(data, d, ls)
ns = get_unique_n_by_ls(data,data.ls.unique())

#%%
if dataset in uci_datasets:
    _ls = ls.item()
    fig, ax = plt.subplots()
    plot_metric_param(data, "mse", "assum_ls", ax, ns, d, _ls, kernel_type)
    ax.set_xlim(0.0,5.0)
    mse_min = data.mse.min()
    mse_max = data.mse.max()
    inc = (mse_max - mse_min) / 10
    ax.set_ylim(mse_min - inc, mse_max + inc)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:-3], labels[:-3], loc='outside right', title="$n$", fontsize=13)
    plt.title(f"MSE ({dataset})")
    if do_save_plots:
        gplot.save_fig(Path(".").joinpath(pathbase).joinpath("mini-project"),
        f"sim_metric_param_{dataset}_d={d}_rescale_{kernel_type}".replace(".","spot"), "pdf",
        show=True, overwrite=True)
# %%
