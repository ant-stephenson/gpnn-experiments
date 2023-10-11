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

sns.set_palette(None)
#%%
do_save_plots: bool = False
uci_datasets = ("ctslice", "song", "buzz")
# pathstr =
# "/Users/anthonystephenson/Documents/GitHub/mini-project/experiments/sim_gpnn_limits_results.csv"
pathstr ="/Users/anthonystephenson/Documents/GitHub/gpnn-experiments/sim_gpnn_limits_results_Dim2.csv"
# pathstr = "/Users/anthonystephenson/Documents/GitHub/gpnn-experiments/sim_gpnn_limits_results_buzz_whiten_dimrescale.csv"
path = Path(".").joinpath(pathstr)

dtype_dict = {'n': int, 'n_test': int, 'd': int, 'm': int, 'seed': int, 'k_true': str, 'k_model': str, 'ks': np.float32, 'ls': np.float32, 'nv': np.float32, 'assum_ks': np.float32, 'assum_ls': np.float32, 'assum_nv': np.float32, 'varypar': str, 'mse': np.float32, 'nll': np.float32, 'mscal': np.float32}

data = pd.read_csv(path, dtype=dtype_dict, header=0, sep=',')
data = data.round(4)

dataset = "".join([x if x in pathstr else "" for x in ["ciq", "oak"] + list(uci_datasets)])
#%%
if dataset == "ciq":
    ls = (0.5,)
    ks = 0.9
    nv = 0.1
    d = 10
    kernel = "RBF"
elif dataset == "oak":
    # ls = 8.5
    ls = (0.75,)
    # ks = 2.0
    ks = 0.9
    nv = 0.1
    kernel = "Exp"
    d=15
elif dataset in uci_datasets:
    ls = data.ls.unique()
    ks = data.ks.unique().item()
    nv = data.nv.unique().item()
    kernel = "RBF"
    d = data.d.unique().item()
else:
    ls = (0.5,0.75,1.0)
    ks = 0.9
    nv = 0.1
    # d = 20
    kernel = "RBF"
min_n = 1000

data = data.query('(n >= @min_n) & (ls in @ls) & (ks == @ks) & (k_model ==@kernel) ').reset_index()
# data = data.query('(n >= @min_n) & (ls in @ls) & (ks == @ks) & (k_model ==
# @kernel)').reset_index()
#%%

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

def get_nvhat(data, d=d, ls=ls):
    if not isinstance(ls, float):
        return mode(data.query('(d == @d) & (ls in @ls)').assum_nv)[0][0]
    else:
        return mode(data.query('(d == @d) & (ls == @ls)').assum_nv)[0][0]

def set_subplot_loc(m: int, nrows: int, ncols: int):
    i = int(np.floor(m / ncols))
    j = m - i * nrows - 1
    return i, j
# data.replace({"lenscale": "lengthscale"}, inplace=True)
def get_unique_n_by_ls(data, ls):
    return functools.reduce(np.intersect1d, [data.query('ls == @_ls').n.unique() for _ls in ls])

idx_cols = ["n", "n_test" ,"d","m","seed","k_true","k_model","ks","ls","nv","assum_ks","assum_ls","assum_nv","varypar"]
# %% make sure to fetch correct param for vline
# add horizontal line for theoretical limit
mse_lim_true = lambda nv, m: nv*(1+float(m)**(-1))
mse_lim_miss = lambda nv, nv_hat, m, ks, ks_hat: nv + (nv + 2*nv_hat*ks/ks_hat)*float(m)**(-1)
nll_lim_true = lambda nv, m: 0.5*np.log(nv) + 0.5*(1+float(m)**(-1)) + 0.5*np.log(2*np.pi)
_mscal_lim_miss = lambda nv, nv_hat: nv/nv_hat
_nll_lim_miss = lambda nv, nv_hat, m: nll_lim_true(nv_hat, m) + 0.5*_mscal_lim_miss(nv,nv_hat) - 0.5

nv = data.nv.iloc[0]
num_near = data.m.iloc[0]
nv_hat = get_nvhat(data, d, ls)

def nll_lim_miss(x, **kwargs):
    x = np.sort(x)
    plt.plot(x, _nll_lim_miss(nv, x, num_near), linestyle="--", color="r")

def mscal_lim_miss(x, **kwargs):
    x = np.sort(x)
    plt.plot(x, _mscal_lim_miss(nv, x), linestyle="--", color="r")

def plot_limit(x, metric="mse", _varpar="assum_ls", ax=None, true_or_miss="miss", nv_hat=nv_hat):
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

#%% plot for paper: Nll vs nv, Z^2 vs nv (with embedded plot?), MSE vs ls?
ns = get_unique_n_by_ls(data,data.ls.unique())

def plot_metric_param(data, metric,_varpar, ax, labels=True, d=d, ls=ls):
    this_nv_hat = get_nvhat(data, d, ls)
    varypar = get_varypar(_varpar)
    this_data = data.query('(d == @d) & (k_model == "RBF") & (varypar == @varypar) & (n in @ns) & (ls == @ls)').reset_index()
    this_data = this_data.sort_values("n")
    this_data.n = this_data.n.astype(float).map('{:.2}'.format)
    g = sns.lineplot(data=this_data, x=_varpar, y=metric, hue="n", ax=ax, palette=sns.color_palette(palette="flare", n_colors=this_data.n.nunique())).set(xlabel=None, ylabel=None)
    const = this_data[_varpar[6:]].iloc[0]
    if dataset != "oak":
        ax.axvline(const, color="k", linestyle="--", label="$\\theta$")
    plot_limit(this_data[_varpar], metric, _varpar, ax, "true", this_nv_hat)
    plot_limit(this_data[_varpar], metric, _varpar, ax, "miss", this_nv_hat)
    ax.get_legend().remove()
    if labels:
        ax.set_xlabel(get_xlabel(_varpar), fontsize=18)
        if dataset != "oak":
            ax.set_title(get_title(metric) + f" ($l$={ls})", fontsize=18)
        else:
            ax.set_title(get_title(metric), fontsize=18)
        ax.set_ylim(0,this_data[metric].max())
        ax.tick_params(axis='both', which='major', labelsize=18)
    else:
        ax.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

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
    plot_limit(this_data[_varpar], metric, _varpar, ax_m, "miss")
    plot_limit(this_data[_varpar], metric, _varpar, ax_m, "true")
    ax_m.set_ylim(0,this_data[metric].max())

handles, labels = ax_m.get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right')
plt.suptitle(f"")
if do_save_plots:
    gplot.save_fig(Path("."), f"sim_metric_param_grid_ls={ls_grid}_d={d}".replace(".","spot"), "pdf", show=True, overwrite=True)

#%% Plot the same lengthscale
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))
_ls = 0.75
plot_metric_param(data, "nll", "assum_nv", ax[0], d=d, ls=_ls)
plot_metric_param(data, "mscal", "assum_nv", ax[1], d=d, ls=_ls)
plot_metric_param(data, "mse", "assum_ls", ax[2], d=d, ls=_ls)
# ax[2].set_yscale('log')
# ax[2].set_xscale('log')
# ax[2].set_ylim(-1.0,1.0)

if _ls > 0.5:
    ax[2].set_ylim(0.0,1.5)
    
    zoom = {"oak": 5, "ciq": 4, "": 3}

    axins = zoomed_inset_axes(ax[2], zoom=zoom[dataset], loc=1)
    plot_metric_param(data, "mse", "assum_ls", axins, labels=False, d=d, ls=_ls)
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
plot_metric_param(data, "nll", "assum_nv", ax[0], d=d, ls=0.5)
plot_metric_param(data, "mscal", "assum_nv", ax[1], d=d, ls=0.5)
plot_metric_param(data, "mse", "assum_ls", ax[2], d=d, ls=1.0)

ax[2].set_ylim(0.0,1.5)

zoom = {"oak": 5, "ciq": 4, "": 3}

axins = zoomed_inset_axes(ax[2], zoom=zoom[dataset], loc=1)
plot_metric_param(data, "mse", "assum_ls", axins, labels=False, d=d, ls=1.0)
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
#%%
if dataset in uci_datasets:
    _ls = ls.item()
    fig, ax = plt.subplots()
    plot_metric_param(data, "mse", "assum_ls", ax, d=d, ls=_ls)
    ax.set_xlim(0.0,5.0)
    mse_min = data.mse.min()
    mse_max = data.mse.max()
    inc = (mse_max - mse_min) / 10
    ax.set_ylim(mse_min - inc, mse_max + inc)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles[:-3], labels[:-3], loc='outside right', title="$n$", fontsize=13)
    plt.title(f"MSE ({dataset})")
    if do_save_plots:
        gplot.save_fig(Path(".").absolute().parents[1].joinpath("mini-project"),
        f"sim_metric_param_{dataset}_d={d}_rescale".replace(".","spot"), "pdf",
        show=True, overwrite=True)

#%% Dim version
#%% Try and replot with n on x-axis to show equivalent behaviour 
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

def plot_metric_param_by_n(data, metric,_varpar, ax, labels=True, d=d, ls=ls):
    this_nv_hat = get_nvhat(data, d, ls)
    varypar = get_varypar(_varpar)
    this_data = data.query('(d == @d) & (ls == @ls) & (k_true == "RBF") & (varypar == @varypar)').reset_index()
    this_data = this_data.sort_values("n")
    this_data.n = this_data.n.astype(float).map('{:.2}'.format)

    grid = this_data.loc[:,_varpar].unique()[[0,2,4,20,-1]]
    this_data = this_data.loc[this_data.loc[:,_varpar].isin(grid),:]
    this_data.loc[:, _varpar] = (this_data.loc[:,_varpar]/this_data.loc[:,_varpar[6:]]).round(1)

    g = sns.lineplot(data=this_data, x="n", y=metric, hue=_varpar, ax=ax, palette=sns.color_palette(palette="flare", n_colors=this_data.loc[:,_varpar].nunique())).set(xlabel=None, ylabel=None)
    plot_limit(this_data.n, metric, _varpar, ax, "true", this_nv_hat)
    # ax.get_legend().remove()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc='best', title="$\\hat{\\theta}/\\theta$", fontsize=13)
    if labels:
        ax.set_xlabel("$n$", fontsize=18)
        ax.set_title(get_title(metric) + f" ($\\hat{{\\theta}}=${get_xlabel(_varpar)})", fontsize=18)
        ax.set_ylim(0,this_data[metric].max())
        ax.tick_params(axis='both', which='major', labelsize=18)
    else:
        ax.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])
plot_metric_param_by_n(data, "nll", "assum_nv", ax[0], d=d, ls=ls[1])
plot_metric_param_by_n(data, "mscal", "assum_nv", ax[1], d=d, ls=ls[1])
ax[1].get_legend().remove()
plot_metric_param_by_n(data, "mse", "assum_ls", ax[2], d=d, ls=ls[1])
# handles, labels = ax[2].get_legend_handles_labels()
# fig.legend(handles, labels, loc='outside right',
# title="$\\frac{\\hat{\\theta}}{\\theta}$", fontsize=13)
if do_save_plots:
    gplot.save_fig(Path("."), f"sim_metric_by_n_{dataset}_d={d}_ls={ls}".replace(".","spot"), "pdf", show=True, overwrite=True)
#%%
varpar = "ks"
_varpar="assum_"+varpar
xlabel = {"assum_ls": "lengthscale", "assum_nv": "noise variance", "assum_ks": "kernelscale"}[_varpar]
varypar = get_varypar(_varpar)

#%% MSE:
# g = sns.relplot(data=data.loc[data.varypar == varypar,:], x=_varpar, y="mse", hue="n", col="k_model", row="k_true", kind="line")
# (g.map(plt.axvline, x=data.ls[0], color="k", linestyle="--"))
# (g.map(plt.axhline, y=mse_lim_true(nv,num_near), color="r", linestyle="--"))
# g.set_axis_labels(x_var=f"Assumed {xlabel}", y_var="MSE")
# # %% NLL
# g = sns.FacetGrid(data=data.loc[data.varypar == varypar,:], col="k_model", row="k_true", hue="n")
# g.map(sns.lineplot, _varpar, "nll").set(yscale="symlog")
# if varpar == "nv":
#     (g.map(nll_lim_miss, _varpar))
# else:
#     g.map(plt.axhline, y=_nll_lim_miss(nv, nv_hat, num_near), color="r", linestyle="--")
# (g.map(plt.axvline, x=data.loc[0,varpar], color="k", linestyle="--"))
# g.set_axis_labels(x_var=f"Assumed {xlabel}", y_var="NLL")
# # %% Z^2
# g = sns.FacetGrid(data=data.loc[data.varypar == varypar,:], col="k_model", row="k_true", hue="n")
# g.map(sns.lineplot, _varpar, "mscal").set(yscale="symlog")
# if varpar == "nv":
#     (g.map(mscal_lim_miss, _varpar))
# else:
#     nv_hat = mode(data.assum_nv)[0][0]
#     g.map(plt.axhline, y=_mscal_lim_miss(nv, nv_hat), color="r", linestyle="--")
# (g.map(plt.axvline, x=data.loc[0,varpar], color="k", linestyle="--"))
# g.set_axis_labels(x_var=f"Assumed {xlabel}", y_var="$Z^2$")
################################################################################
#%% plot convergence rates?
# ls = 0.75
# assum_nv = 0.2
assum_nv = np.round(data.assum_nv.unique().item(),4)
# get rid of d=2 as too easy: MSE ~ nv already; not sure why the curve is so
# smooth though? Would expect to be just noise
data_1 = data.query('(d>2) &(ls == @_ls) & (k_model == "RBF") & (k_true == k_model | k_true.isnull())').reset_index().copy()
data_1.drop("index", axis=1, inplace=True)
multi_ind = pd.MultiIndex.from_frame(data_1.loc[:,["n","m","seed","k_true","k_model","ks","ls","nv"]])
data_1.index = multi_ind

dgp1 = data_1.groupby(["d", "assum_ks", "assum_ls", "assum_nv"], group_keys=True)
out = dgp1.apply(lambda x: np.polyfit(np.log(x.n), np.log(np.abs(x.mse-x.nv*(1+1/x.m))), 1)).reset_index()
out.loc[:,"slope"] = out.loc[:,0].apply(lambda x: x[0])
out.loc[:,"intercept"] = out.loc[:,0].apply(lambda x: x[1])
out.drop(0, axis=1,inplace=True)
out.assum_ls = out.assum_ls.apply(lambda x: np.round(x, 3))
out.assum_ks = out.assum_ks.apply(lambda x: np.round(x, 3))
out.assum_nv = out.assum_nv.apply(lambda x: np.round(x, 3))

ds = data_1.d.unique()
ds.sort()
palette = sns.color_palette(n_colors=len(ds))
# CHECK COLORS/HORIZONTAL LINES?
from scipy.ndimage.filters import gaussian_filter1d

def plot_slope_vs_param(data, param: str):
    ax = sns.relplot(data=data, x=param, y="slope", hue="d", palette=palette)

    for i,_d in enumerate(ds):
        ind = data.d == _d
        plt.plot(data.loc[ind, param], gaussian_filter1d(data.loc[ind, "slope"], sigma=1), color=palette[i])
        plt.axhline(-2/_d, color=palette[i], linestyle="--")
        # plt.axhline(-2/(_d/2), color=palette[i], linestyle="-.")
    return ax
# %% kshat vs slope
_ks = data_1.index.get_level_values("ks")[0]
tmp = out.query("assum_ls == @ls & assum_nv == @assum_nv")
ax = plot_slope_vs_param(tmp, "assum_ks")
plt.axvline(_ks, color="k", linestyle="--")
#%% nvhat vs slope
_nv = data_1.index.get_level_values("nv")[0]
tmp = out.query("assum_ls == @ls & assum_nv != @assum_nv")
ax = plot_slope_vs_param(tmp, "assum_nv")
plt.axvline(_nv, color="k", linestyle="--")
#%% lshat vs slope - seems to asymptote at 2*-2/d (explanation?)
# l_eff < sqrt(d/d0) * l ? ~sqrt(2) for d/d0=2, but ~1.5-1.6? for d/d0 = 3?
tmp = out.query("assum_nv == @assum_nv & assum_ls != @_ls").sort_values("assum_ls")
ax = plot_slope_vs_param(tmp, "assum_ls")
plt.axvline(ls, color="k", linestyle="--", label="$l$")
plt.axvline(np.sqrt(3)*ls, color="k", linestyle="-.", label="$l_{eff}$")
ax.set(ylim=(out.slope.min()-0.01,out.slope.max()+0.01))
ax.set(xlim=(out.assum_ls.min(),out.assum_ls.max()))
# g = lambda l, lhat, d,eta,a,alpha: a * np.exp(-eta*(lhat-l)) * (l-lhat)**alpha  - 1/np.log(d)
# optim = lambda x: np.sum((g(ls, tmp.loc[tmp.d==10,"assum_ls"], 10, x[0],x[1], x[2]) - tmp.loc[tmp.d==10,"slope"])**2)
# bnds= ((0,None),(0,None),(2,None))
# minimize(optim, [1,0.25,2], bounds=bnds)
# plt.plot(tmp.assum_ls, g(ls, tmp.assum_ls, 10, 3e-6,0.9, 2))
#%%
out.loc[:, "lhat>l"] = out.assum_ls > ls
sns.violinplot(data=out.query("d>2"), x="d", y="slope",palette=palette, hue="lhat>l", split=True)
################################################################################
# %%
# g = sns.relplot(data=data, x="n", y="mse", col="k_model", row="k_true", kind="line").set(yscale="log", xscale="log")
# (g.map(plt.axhline, y=mse_lim_true(nv,num_near), color="r", linestyle="--"))
# # %%
# id_vars=["n","varypar","k_true","k_model", "assum_ks","assum_ls","assum_nv"]
# pdf = data.melt(id_vars=id_vars, value_vars=["mse","nll","mscal"]).pivot(index=id_vars, columns="variable", values="value")
# #%%
# melted = data.melt(id_vars=id_vars, value_vars=["mse","nll","mscal"])
# melted["x"] = 0*melted.assum_nv
# melted.x[melted.varypar == "lenscale"] = melted.assum_ls[melted.varypar == "lenscale"]
# melted.x[melted.varypar == "kernelscale"] = melted.assum_ks[melted.varypar == "kernelscale"]
# melted.x[melted.varypar == "noisevar"] = melted.assum_nv[melted.varypar == "noisevar"]
# #%%
# g = sns.FacetGrid(data=melted.loc[(melted.k_true == "RBF") & (melted.k_model == "RBF"), :], col="varypar", row="variable", hue="n", sharex=False, sharey=False, palette=sns.color_palette("flare"))
# (g.map(sns.lineplot, "x", "value"))
# g.add_legend()
# g.axes[-1,0].set_xlabel("Assumed noise variance")
# g.axes[-1,1].set_xlabel("Assumed lengthscale")
# g.axes[-1,2].set_xlabel("Assumed kernelscale")
# g.axes[0,0].set_ylabel("MSE")
# g.axes[1,0].set_ylabel("NLL")
# g.axes[2,0].set_ylabel("$Z^2$")
# g.set_titles("")

