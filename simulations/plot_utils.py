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

def get_varypar(_varpar, is_new=True):
    if is_new:
        varypar = {"assum_ls": "lengthscale", "assum_nv": "noise", "assum_ks": "kernelscale"}[_varpar]
    else:
        varypar = {"assum_ls": "lenscale", "assum_nv": "noisevar", "assum_ks": "kernelscale"}[_varpar]
    return varypar

def get_xlabel(_varpar):
    return {"assum_ls": "$\\hat{l}$", "assum_nv": "$\\hat{\\sigma}_\\xi^2$", "assum_ks": "$\\hat{\\sigma}_f^2$"}[_varpar]

def get_leglabel(_varpar):
    return {"assum_ls": "$l$", "assum_nv": "$\\sigma_\\xi^2$", "assum_ks": "$\\sigma_f^2$"}[_varpar]

def get_title(metric):
    return {"nll": "NLL", "mscal": "Calibration", "mse": "MSE"}[metric]

def get_nvhat(data, d, ls):
    if not isinstance(ls, float):
        return mode(data.query('(d == @d) & (ls in @ls)').assum_nv)[0][0]
    else:
        return mode(data.query('(d == @d) & (ls == @ls)').assum_nv)[0][0]

def get_nv(data, d, ls):
    if not isinstance(ls, float):
        return mode(data.query('(d == @d) & (ls in @ls)').nv)[0][0]
    else:
        return mode(data.query('(d == @d) & (ls == @ls)').nv)[0][0]

def get_num_near(data):
    ms = data.m.unique()
    if len(ms)>1:
        raise ValueError("Expecting only one value.")
    else:
        return ms[0]

def set_subplot_loc(m: int, nrows: int, ncols: int):
    i = int(np.floor(m / ncols))
    j = m - i * nrows - 1
    return i, j
# data.replace({"lenscale": "lengthscale"}, inplace=True)
def get_unique_n_by_ls(data, ls):
    return functools.reduce(np.intersect1d, [data.query('ls == @_ls').n.unique() for _ls in ls])

# %% make sure to fetch correct param for vline
# add horizontal line for theoretical limit
mse_lim_true = lambda nv, m: nv*(1+float(m)**(-1))
mse_lim_miss = lambda nv, nv_hat, m, ks, ks_hat: nv + (nv + 2*nv_hat*ks/ks_hat)*float(m)**(-1)
nll_lim_true = lambda nv, m: 0.5*np.log(nv) + 0.5*(1+float(m)**(-1)) + 0.5*np.log(2*np.pi)
_mscal_lim_miss = lambda nv, nv_hat: nv/nv_hat
_nll_lim_miss = lambda nv, nv_hat, m: nll_lim_true(nv_hat, m) + 0.5*_mscal_lim_miss(nv,nv_hat) - 0.5

def nll_lim_miss(x, **kwargs):
    x = np.sort(x)
    plt.plot(x, _nll_lim_miss(nv, x, num_near), linestyle="--", color="r")

def mscal_lim_miss(x, **kwargs):
    x = np.sort(x)
    plt.plot(x, _mscal_lim_miss(nv, x), linestyle="--", color="r")

def plot_limit(x, nv, num_near, metric="mse", _varpar="assum_ls", ax=None, true_or_miss="miss", nv_hat=None):
    colour = {"true": "g", "miss": "r"}[true_or_miss]
    label = {"true": "$f_\\infty(\\theta)$", "miss": "$f_\\infty(\\hat{\\theta})$"}[true_or_miss]
    if metric == "mse":
        ax.axhline(mse_lim_true(nv,num_near), color=colour, linestyle="--", label=label)
    elif metric == "mscal":
        if true_or_miss == "true":
            mscal_fn = lambda nv, x: np.ones_like(x)
        else:
            mscal_fn = _mscal_lim_miss
        if _varpar == "assum_nv":
            x = np.sort(x)
            ax.plot(x, mscal_fn(nv, x), linestyle="--", color=colour, label=label)
        else:
            ax.axhline(mscal_fn(nv, nv_hat), color=colour, linestyle="--", label=label)
    elif metric == "nll":
        if true_or_miss == "true":
            nll_fn = lambda nv, x, m: np.ones_like(x) * nll_lim_true(nv,m)
        else:
            nll_fn = _nll_lim_miss
        if _varpar == "assum_nv":
            x = np.sort(x)
            ax.plot(x, nll_fn(nv, x, num_near), linestyle="--", color=colour, label=label)
        else:
            ax.axhline(nll_fn(nv, nv_hat, num_near), color=colour, linestyle="--", label=label)

def plot_metric_param(data, metric,_varpar, ax, ns, d, ls, labels=True,):
    this_nv_hat = get_nvhat(data, d, ls)
    this_nv = get_nv(data, d, ls)
    num_near = get_num_near(data)
    varypar = get_varypar(_varpar)
    this_data = data.query('(d == @d) & (k_model == "RBF") & (varypar == @varypar) & (n in @ns) & (ls == @ls)').reset_index()
    this_data = this_data.sort_values("n")
    this_data.n = this_data.n.astype(float).map('{:.2}'.format)
    g = sns.lineplot(data=this_data, x=_varpar, y=metric, hue="n", ax=ax, palette=sns.color_palette(palette="flare", n_colors=this_data.n.nunique())).set(xlabel=None, ylabel=None)
    const = this_data[_varpar[6:]].iloc[0]
    if True:#dataset != "oak":
        ax.axvline(const, color="k", linestyle="--", label="$\\theta$")
    plot_limit(this_data[_varpar], this_nv, num_near, metric, _varpar, ax, "true", this_nv_hat)
    plot_limit(this_data[_varpar], this_nv, num_near, metric, _varpar, ax, "miss", this_nv_hat)
    ax.get_legend().remove()
    if labels:
        ax.set_xlabel(get_xlabel(_varpar), fontsize=18)
        if True:#dataset != "oak":
            ax.set_title(get_title(metric) + f" ($l$={ls})", fontsize=18)
        else:
            ax.set_title(get_title(metric), fontsize=18)
        ax.set_ylim(0,this_data[metric].max())
        ax.tick_params(axis='both', which='major', labelsize=18)
    else:
        ax.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])