#%%
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

from plot_utils import *

sns.set_palette(None)
#%%
do_save_plots: bool = False
uci_datasets = ("ctslice", "song", "buzz", "house_electric", "protein", "road3d")
pathbase = "/Users/anthonystephenson/Documents/GitHub/"
pathstr = "/Users/anthonystephenson/Documents/GitHub/gpnn-experiments/sim_gpnn_limits_results_d20.csv"
# pathstr ="/Users/anthonystephenson/Documents/GitHub/gpnn-experiments/sim_gpnn_limits_results_Dim2.csv"
path = Path(".").joinpath(pathstr)

dtype_dict = {'n': int, 'n_test': int, 'd': int, 'm': int, 'seed': int, 'k_true': str, 'k_model': str, 'ks': np.float32, 'ls': np.float32, 'nv': np.float32, 'assum_ks': np.float32, 'assum_ls': np.float32, 'assum_nv': np.float32, 'varypar': str, 'mse': np.float32, 'nll': np.float32, 'mscal': np.float32}

data = pd.read_csv(path, dtype=dtype_dict, header=0, sep=',')
data = data.round(4)
data = data.fillna(-999)
# data = data.groupby(["n","d","m","n_test","assum_ls", "k_model", "k_true", "varypar"]).mean().reset_index()

dataset = "".join([x if x in pathstr else "" for x in ["ciq", "oak", "dim"] + list(uci_datasets)])

#%%
ls = (0.5,0.75,1.0)
ks = 0.9
nv = 0.1
###
d = 20
kernel = "RBF"
min_n = 1000

data = data.query('(n >= @min_n) & (ls in @ls) & (ks == @ks) & (k_model ==@kernel) & (k_model == k_true) & (d==@d)').reset_index()

ns = get_unique_n_by_ls(data,data.ls.unique())

#%% fix old versions
data.replace({"lenscale": "lengthscale", "noisevar": "noise"}, inplace=True)
#%%
ls_grid = 0.5
nrows, ncols = 3,3
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols+1,4*nrows))
for m,(metric, _varpar) in enumerate(cartesian_prod(["mse", "nll", "mscal"],["assum_ls", "assum_ks", "assum_nv"])):
    varypar = get_varypar(_varpar)
    if nrows > 1:
        i, j = set_subplot_loc(m, nrows, ncols)
        ax_m = ax[i,j]
    else:
        ax_m = ax[m]
    this_data = data.query('(k_true == "RBF") & (k_true == k_model) & (varypar == @varypar) & (ls == @ls_grid)').reset_index()
    this_nv = get_nv(this_data, d, ls)
    this_nv_hat = get_nvhat(this_data, d, ls)
    num_near = get_num_near(this_data)
    g = sns.lineplot(data=this_data, x=_varpar, y=metric, hue="n", ax=ax_m, palette=sns.color_palette(palette="flare", n_colors=data.n.nunique()), label="_nolegend_").set(xlabel=None, ylabel=None) 
    try:
        ax_m.get_legend().remove()
    except:
        pass
    # ax_m.set_title(metric.upper())
    if i==2 or nrows==1:
        ax_m.set_xlabel(get_xlabel(_varpar), fontsize=18)
    if j==0 or ncols==1:
        ax_m.set_ylabel(get_title(metric), fontsize=18)
    const = data[_varpar[6:]].iloc[0]
    ax_m.axvline(const, color="k", linestyle="--", label="$\\theta$")
    plot_limit(this_data[_varpar], this_nv, num_near, metric, _varpar, ax_m, "miss", this_nv_hat)
    plot_limit(this_data[_varpar], this_nv, num_near,metric, _varpar, ax_m, "true", this_nv_hat)
    ax_m.set_ylim(0,this_data[metric].max())

handles, labels = ax_m.get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right')
plt.suptitle(f"")
if do_save_plots:
    gplot.save_fig(Path("."), f"sim_metric_param_grid_ls={ls_grid}_d={d}".replace(".","spot"), "pdf", show=True, overwrite=True)

#%% Plot the same lengthscale
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
_ls = 0.75
plot_metric_param(data, "nll", "assum_nv", ax[0], ns, d, _ls)
plot_metric_param(data, "mscal", "assum_nv", ax[1], ns, d, _ls)
plot_metric_param(data, "mse", "assum_ls", ax[2], ns, d, _ls)
# ax[2].set_yscale('log')
# ax[2].set_xscale('log')
# ax[2].set_ylim(-1.0,1.0)

if _ls > 0.5:
    ax[2].set_ylim(0.0,1.5)
    
    zoom = {"oak": 5, "ciq": 4, "": 3}

    axins = zoomed_inset_axes(ax[2], zoom=zoom[dataset], loc=1)
    plot_metric_param(data, "mse", "assum_ls", axins, ns, d, _ls, labels=False)
    # sub region of the original image
    # x1, x2, y1, y2 = 0.0, 0.3, 0.5, 1.5 # mscal ls=0.75
    oak_region = (8.0,10.0,0.2,0.3)
    ciq_region = (0.1,2.0,0.1,0.4)
    sim_region = (0.5, 1.5, 0.05, 0.2)
    regions = {"oak": oak_region, "ciq": ciq_region, "": sim_region}
    x1, x2, y1, y2 = regions[dataset]
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)

    mark_inset(ax[2], axins, loc1=2, loc2=4, fc="none", ec="0.5")

handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right', title="$n$", fontsize=13)
# fig.legend(handles, labels, bbox_to_anchor=(1.01, 0.5), loc="center left", title="$n$", fontsize=13, borderpad=0., handletextpad=0., borderaxespad=0.)

if do_save_plots:
    gplot.save_fig(Path("."),
    f"sim_metric_param_{dataset}_d={d}_ls={_ls}".replace(".","spot"), "pdf",
    show=True, overwrite=True)

#%% Plot mixed lengthscales
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
plot_metric_param(data, "nll", "assum_nv", ax[0], ns, d=d, ls=0.5)
plot_metric_param(data, "mscal", "assum_nv", ax[1], ns, d=d, ls=0.5)
plot_metric_param(data, "mse", "assum_ls", ax[2], ns, d=d, ls=1.0)

ax[2].set_ylim(0.0,1.5)

zoom = {"oak": 5, "ciq": 4, "": 3}

axins = zoomed_inset_axes(ax[2], zoom=zoom[dataset], loc=1)
plot_metric_param(data, "mse", "assum_ls", axins, ns, d, ls=1.0, labels=False)
# sub region of the original image
# x1, x2, y1, y2 = 0.0, 0.3, 0.5, 1.5 # mscal ls=0.75
oak_region = (8.0,10.0,0.2,0.3)
ciq_region = (0.1,2.0,0.1,0.4)
sim_region = (0.5, 1.5, 0.05, 0.2)
regions = {"oak": oak_region, "ciq": ciq_region, "": sim_region}
x1, x2, y1, y2 = regions[dataset]
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

mark_inset(ax[2], axins, loc1=2, loc2=4, fc="none", ec="0.5")

handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right', title="$n$", fontsize=13)

if do_save_plots:
    gplot.save_fig(Path("."),
    f"sim_metric_param_{dataset}_d={d}_ls=mix".replace(".","spot"), "pdf",
    show=True, overwrite=True)
################################################################################