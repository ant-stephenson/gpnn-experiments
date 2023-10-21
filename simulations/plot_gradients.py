#%%
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from scipy.ndimage.filters import gaussian_filter1d
import seaborn as sns
from scipy.stats import mode
from itertools import product as cartesian_prod
import functools

import gpybench.plotting as gplot

from plot_utils import *

sns.set_palette(None)

#%% plot convergence rates?
def add_slope_cols(data, assum_nv):
    # get rid of d=2 as too easy: MSE ~ nv already; not sure why the curve is so
    # smooth though? Would expect to be just noise
    # data_1 = data.query('(d>2) &(ls == @ls) & (k_model == @k_model) & (k_true
    # == @k_true | k_true.isnull())').reset_index(drop=True).copy()
    data_1 = data
    multi_ind = pd.MultiIndex.from_frame(data_1.loc[:,["n","m","seed","ks","ls","nv"]])
    data_1.index = multi_ind

    dgp1 = data_1.groupby(["d", "assum_ks", "assum_ls", "assum_nv", "k_model", "k_true"], group_keys=True)
    out = dgp1.apply(lambda x: np.polyfit(np.log(x.n), np.log(np.abs(x.mse-x.nv*(1+1/x.m))), 1)).reset_index()
    out.loc[:,"slope"] = out.loc[:,0].apply(lambda x: x[0])
    out.loc[:,"intercept"] = out.loc[:,0].apply(lambda x: x[1])
    out.drop(0, axis=1,inplace=True)
    out.assum_ls = out.assum_ls.apply(lambda x: np.round(x, 3))
    out.assum_ks = out.assum_ks.apply(lambda x: np.round(x, 3))
    out.assum_nv = out.assum_nv.apply(lambda x: np.round(x, 3))

    return data_1, out

def smooth_curve(x,y,d,_d=0,**kwargs):
    ind = d == _d
    plt.plot(x[ind], gaussian_filter1d(y[ind], sigma=1), **kwargs)

def plot_slope_vs_param(data, param: str, ax=None):
    ds = sorted(data.d.unique())
    palette = sns.color_palette(n_colors=len(ds))
    g = sns.relplot(data=data, x=param, y="slope", hue="d", palette=palette, ax=ax, row="k_true", col="k_model")
    p = {"RBF": 2, "Exp": 1}[data.k_true[0]]
    
    (g.map(plt.axvline, x=ls, color="k", linestyle="--", label="$l$"))
    (g.map(plt.axvline, x=np.sqrt(3)*ls, color="k", linestyle="-.", label="$l_{eff}$"))
    for i,_d in enumerate(ds):
        ind = data.d == _d
        (g.map(plt.axhline, y=-p/_d, color=palette[i], linestyle="--"))
        (g.map(smooth_curve, "assum_ls", "slope", "d",_d=_d, color=palette[i]))
        # plt.plot(data.loc[ind, param], gaussian_filter1d(data.loc[ind, "slope"], sigma=1), color=palette[i])
        # plt.axhline(-p/_d, color=palette[i], linestyle="--")
        # plt.axhline(-2/(_d/2), color=palette[i], linestyle="-.")
    return g
#%%

do_save_plots: bool = False
uci_datasets = ("ctslice", "song", "buzz", "house_electric", "protein", "road3d")
pathbase = "/Users/anthonystephenson/Documents/GitHub/"
pathstr = "/Users/anthonystephenson/Documents/GitHub/gpnn-experiments/sim_gpnn_limits_results_misspec_var_l.csv"
# pathstr ="/Users/anthonystephenson/Documents/GitHub/gpnn-experiments/sim_gpnn_limits_results_Dim2.csv"

dataset = "".join([x if x in pathstr else "" for x in ["ciq", "oak", "dim"] + list(uci_datasets)])

#%%
ls = 0.75
ks = 0.9
nv = 0.1
###
k_true = "RBF"
k_model = "RBF"

data = get_sim_data(pathstr, ls, ks, nv, min_n=1000)

ns = get_unique_n_by_ls(data,data.ls.unique())

#%%
assum_nv = np.round(data.assum_nv.unique().item(),4)
data_1, out = add_slope_cols(data, assum_nv)
#%%
# lshat vs slope - seems to asymptote at 2*-2/d (explanation?)
# l_eff < sqrt(d/d0) * l ? ~sqrt(2) for d/d0=2, but ~1.5-1.6? for d/d0 = 3?
tmp = out.query("assum_nv == @assum_nv & assum_ls != @ls").sort_values("assum_ls")
ax = plot_slope_vs_param(tmp, "assum_ls")

ax.set(ylim=(out.slope.min()-0.01,out.slope.max()+0.01))
ax.set(xlim=(out.assum_ls.min(),out.assum_ls.max()))
# g = lambda l, lhat, d,eta,a,alpha: a * np.exp(-eta*(lhat-l)) * (l-lhat)**alpha  - 1/np.log(d)
# optim = lambda x: np.sum((g(ls, tmp.loc[tmp.d==10,"assum_ls"], 10, x[0],x[1], x[2]) - tmp.loc[tmp.d==10,"slope"])**2)
# bnds= ((0,None),(0,None),(2,None))
# minimize(optim, [1,0.25,2], bounds=bnds)
# plt.plot(tmp.assum_ls, g(ls, tmp.assum_ls, 10, 3e-6,0.9, 2))
# #%%
# out.loc[:, "lhat>l"] = out.assum_ls > ls
# sns.violinplot(data=out.query("d>2"), x="d", y="slope", hue="lhat>l", split=True)
#%%
# plot_metric_param_by_n(data, "nll", "assum_nv", ax[0], d=d, ls=ls)
# plot_metric_param_by_n(data, "mscal", "assum_nv", ax[1], d=d, ls=ls)
# ax[1].get_legend().remove()
#%%
fig, ax = plt.subplots()
d = 20

plot_metric_param_by_n(data, "mse", "assum_ls", ax, ns, d, ls, k_model)
if False:
    gplot.save_fig(Path("."), f"sim_metric_by_n_ls={ls}_d={d}".replace(".","spot"), "pdf", show=True, overwrite=True)

#%%
all_k_pairs = list(cartesian_prod([k_true, k_model],[k_true,k_model]))
def plot_kgrid(data, plot_fn, xlabel, ylabel, suptitle=None):
    fig, ax = plt.subplots(2,2, figsize=(12,12), sharex=True)
    for m,pair in enumerate(all_k_pairs):
        j,k = gplot.set_subplot_loc(m, 2,2)
        tmp_data = data.query('k_true == @pair[0] & k_model == @pair[1]')
        plot_fn(tmp_data, ax[j,k])
        
        ax[1,k].set_xlabel(xlabel)
        ax[j,0].set_ylabel(ylabel)
        ax[j,k].set_title(f"{pair[0]}-{pair[1]}")
    plt.suptitle("")
    handles, labels = ax[-1,-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc='outside right', title="$\hat{l}/l$")

def plot_grad_ls(data, ax):
    ls_ = data.assum_ls.unique()[[0,1,2,3,10,-1]]
    palette = sns.color_palette(n_colors=len(ls_))
    for i in range(len(ls_)):
        tmp = data.query("assum_ls == @ls_[@i] & d == @d")
        # log or not? not sure, neither shows clearly
        ax.plot(tmp.n, np.gradient(tmp.mse), label=f"{ls_[i]/ls:.3}", color=palette[i])
        # guess transition is given by ratio of lengthscales to power d ??
        trans = (ls_[i]/ls) ** (-d)
        if trans < ns.max():
            ax.axvline(trans, linestyle="--", color=palette[i])
#%%
plot_kgrid(data, plot_grad_ls, "$n$", "$\Delta$MSE")
if do_save_plots:
    gplot.save_fig(Path(".").joinpath(pathbase).joinpath("mini-project"), f"sim_mse_grad_ktrue={k_true}_kmod={k_model}_ls={ls}_d={d}".replace(".","spot"), "pdf", show=True, overwrite=True)
# %% plot to look for region switching
# k_true = "RBF"
# k_model = "Exp"
# all_k_pairs = cartesian_prod([k_true, k_model],[k_true,k_model])
# fig, ax = plt.subplots(2,2, figsize=(12,12), sharex=True)

# for m,pair in enumerate(all_k_pairs):
#     j,k = gplot.set_subplot_loc(m, 2,2)
#     tmp_data = data.query('k_true == @pair[0] & k_model == @pair[1]')
#     ls_ = tmp_data.assum_ls.unique()[[0,1,2,3,10,-1]]
#     palette = sns.color_palette(n_colors=len(ls_))
#     for i in range(len(ls_)):
#         tmp = tmp_data.query("assum_ls == @ls_[@i] & d == @d")
#         # log or not? not sure, neither shows clearly
#         ax[j,k].plot(tmp.n, np.gradient(tmp.mse), label=f"{ls_[i]/ls:.3}", color=palette[i])
#         # guess transition is given by ratio of lengthscales to power d ??
#         trans = (ls_[i]/ls) ** (-d)
#         if trans < ns.max():
#             ax[j,k].axvline(trans, linestyle="--", color=palette[i])
#     # ax[j,k].legend(title="$\hat{l}/l$", loc="upper right")
#     ax[1,k].set_xlabel("$n$")
#     ax[j,0].set_ylabel("$\Delta$MSE")
#     ax[j,k].set_title(f"{pair[0]}-{pair[1]}")
# plt.suptitle("")
# handles, labels = ax[-1,-1].get_legend_handles_labels()
# fig.legend(handles, labels, loc='outside right', title="$\hat{l}/l$")
# if do_save_plots:
#     gplot.save_fig(Path(".").joinpath(pathbase).joinpath("mini-project"), f"sim_mse_grad_ktrue={k_true}_kmod={k_model}_ls={ls}_d={d}".replace(".","spot"), "pdf", show=True, overwrite=True)
# %%
