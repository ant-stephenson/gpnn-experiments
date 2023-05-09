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

sns.set_palette(None)
#%%
path = Path(".").joinpath("experiments/sim_gpnn_limits_results.csv")

dtype_dict = {'n': int, 'n_test': int, 'd': int, 'm': int, 'seed': int, 'k_true': str, 'k_model': str, 'ks': np.float32, 'ls': np.float32, 'nv': np.float32, 'assum_ks': np.float32, 'assum_ls': np.float32, 'assum_nv': np.float32, 'varypar': str, 'mse': np.float32, 'nll': np.float32, 'mscal': np.float32}

data = pd.read_csv(path, dtype=dtype_dict, header=0, sep=',')

#%%
ls = 1.0
d = 20

# data = data.query('(d == @d) & (ls == @ls)').reset_index()

def get_varypar(_varpar):
    return {"assum_ls": "lenscale", "assum_nv": "noisevar", "assum_ks": "kernelscale"}[_varpar]

def get_xlabel(_varpar):
    return {"assum_ls": "$\\hat{l}$", "assum_nv": "$\\hat{\\sigma}_\\xi^2$", "assum_ks": "$\\hat{\\sigma}_f^2$"}[_varpar]

def get_leglabel(_varpar):
    return {"assum_ls": "$l$", "assum_nv": "$\\sigma_\\xi^2$", "assum_ks": "$\\sigma_f^2$"}[_varpar]

def get_title(metric):
    return {"nll": "NLL", "mscal": "Calibration", "mse": "MSE"}[metric]

def get_nvhat(data, d=d, ls=ls):
    return mode(data.query('(d == @d) & (ls == @ls)').reset_index().assum_nv)[0][0]

def set_subplot_loc(m: int, nrows: int, ncols: int):
    i = int(np.floor(m / ncols))
    j = m - i * nrows - 1
    return i, j
# data.replace({"lenscale": "lengthscale"}, inplace=True)

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
#%%
nrows, ncols = 3,3
fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(4*ncols+1,4*nrows))
for m,(metric, _varpar) in enumerate(cartesian_prod(["mse", "nll", "mscal"],["assum_ls", "assum_ks", "assum_nv"])):
    varypar = get_varypar(_varpar)
    if nrows > 1:
        i, j = set_subplot_loc(m, nrows, ncols)
        ax_m = ax[i,j]
    else:
        ax_m = ax[m]
    this_data = data.query('(k_true == "RBF") & (k_true == k_model) & (varypar == @varypar)').reset_index()
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
# gplot.save_fig(Path("."), f"sim_metric_param_grid_ls={ls}_d={d}".replace(".","spot"), "png", show=True, overwrite=True)

#%% plot for paper: Nll vs nv, Z^2 vs nv (with embedded plot?), MSE vs ls?
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(14, 4))

def plot_metric_param(data, metric,_varpar, ax, labels=True, d=d, ls=ls):
    this_nv_hat = get_nvhat(data, d, ls)
    varypar = get_varypar(_varpar)
    this_data = data.query('(d == @d) & (ls == @ls) & (k_true == "RBF") & (k_true == k_model) & (varypar == @varypar)').reset_index()
    this_data = this_data.sort_values("n")
    this_data.n = this_data.n.astype(float).map('{:.2}'.format)
    g = sns.lineplot(data=this_data, x=_varpar, y=metric, hue="n", ax=ax, palette=sns.color_palette(palette="flare", n_colors=data.n.nunique())).set(xlabel=None, ylabel=None)
    const = this_data[_varpar[6:]].iloc[0]
    ax.axvline(const, color="k", linestyle="--", label="$\\theta$")
    plot_limit(this_data[_varpar], metric, _varpar, ax, "miss", this_nv_hat)
    plot_limit(this_data[_varpar], metric, _varpar, ax, "true", this_nv_hat)
    ax.get_legend().remove()
    if labels:
        ax.set_xlabel(get_xlabel(_varpar), fontsize=18)
        ax.set_title(get_title(metric) + f" ($l$={ls})", fontsize=18)
        ax.set_ylim(0,this_data[metric].max())
    else:
        ax.xaxis.set_ticklabels([])
        ax.axes.yaxis.set_ticklabels([])

plot_metric_param(data, "nll", "assum_nv", ax[0], d=20, ls=0.5)
plot_metric_param(data, "mscal", "assum_nv", ax[1], d=20, ls=0.5)
plot_metric_param(data, "mse", "assum_ls", ax[2], d=20, ls=1.0)

axins = zoomed_inset_axes(ax[2], zoom=3, loc=1)
plot_metric_param(data, "mse", "assum_ls", axins, labels=False)
# sub region of the original image
# x1, x2, y1, y2 = 0.0, 0.3, 0.5, 1.5 # mscal ls=0.75
x1, x2, y1, y2 = 0.5,1.5,0.1,0.25
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

mark_inset(ax[2], axins, loc1=2, loc2=4, fc="none", ec="0.5")

handles, labels = ax[2].get_legend_handles_labels()
fig.legend(handles, labels, loc='outside right', title="$n$", fontsize=13)

# gplot.save_fig(Path("."),
# f"sim_metric_param_ls={ls}_d={d}".replace(".","spot"), "png", show=True,
# overwrite=True)
# gplot.save_fig(Path("."), f"sim_metric_param_mix_ls_d={d}".replace(".","spot"), "png", show=True, overwrite=True)
################################################################################
#%% 
varpar = "ks"
_varpar="assum_"+varpar
xlabel = {"assum_ls": "lengthscale", "assum_nv": "noise variance", "assum_ks": "kernelscale"}[_varpar]
varypar = get_varypar(_varpar)

#%% MSE:
g = sns.relplot(data=data.loc[data.varypar == varypar,:], x=_varpar, y="mse", hue="n", col="k_model", row="k_true", kind="line")
(g.map(plt.axvline, x=data.ls[0], color="k", linestyle="--"))
(g.map(plt.axhline, y=mse_lim_true(nv,num_near), color="r", linestyle="--"))
g.set_axis_labels(x_var=f"Assumed {xlabel}", y_var="MSE")
# %% NLL
g = sns.FacetGrid(data=data.loc[data.varypar == varypar,:], col="k_model", row="k_true", hue="n")
g.map(sns.lineplot, _varpar, "nll").set(yscale="symlog")
if varpar == "nv":
    (g.map(nll_lim_miss, _varpar))
else:
    g.map(plt.axhline, y=_nll_lim_miss(nv, nv_hat, num_near), color="r", linestyle="--")
(g.map(plt.axvline, x=data.loc[0,varpar], color="k", linestyle="--"))
g.set_axis_labels(x_var=f"Assumed {xlabel}", y_var="NLL")
# %% Z^2
g = sns.FacetGrid(data=data.loc[data.varypar == varypar,:], col="k_model", row="k_true", hue="n")
g.map(sns.lineplot, _varpar, "mscal").set(yscale="symlog")
if varpar == "nv":
    (g.map(mscal_lim_miss, _varpar))
else:
    nv_hat = mode(data.assum_nv)[0][0]
    g.map(plt.axhline, y=_mscal_lim_miss(nv, nv_hat), color="r", linestyle="--")
(g.map(plt.axvline, x=data.loc[0,varpar], color="k", linestyle="--"))
g.set_axis_labels(x_var=f"Assumed {xlabel}", y_var="$Z^2$")
################################################################################
# %%
g = sns.relplot(data=data, x="n", y="mse", col="k_model", row="k_true", kind="line")
(g.map(plt.axhline, y=mse_lim_true(nv,num_near), color="r", linestyle="--"))
# %%
id_vars=["n","varypar","k_true","k_model", "assum_ks","assum_ls","assum_nv"]
pdf = data.melt(id_vars=id_vars, value_vars=["mse","nll","mscal"]).pivot(index=id_vars, columns="variable", values="value")
#%%
melted = data.melt(id_vars=id_vars, value_vars=["mse","nll","mscal"])
melted["x"] = 0*melted.assum_nv
melted.x[melted.varypar == "lenscale"] = melted.assum_ls[melted.varypar == "lenscale"]
melted.x[melted.varypar == "kernelscale"] = melted.assum_ks[melted.varypar == "kernelscale"]
melted.x[melted.varypar == "noisevar"] = melted.assum_nv[melted.varypar == "noisevar"]
#%%
g = sns.FacetGrid(data=melted.loc[(melted.k_true == "RBF") & (melted.k_model == "RBF"), :], col="varypar", row="variable", hue="n", sharex=False, sharey=False, palette=sns.color_palette("flare"))
(g.map(sns.lineplot, "x", "value"))
g.add_legend()
g.axes[-1,0].set_xlabel("Assumed noise variance")
g.axes[-1,1].set_xlabel("Assumed lengthscale")
g.axes[-1,2].set_xlabel("Assumed kernelscale")
g.axes[0,0].set_ylabel("MSE")
g.axes[1,0].set_ylabel("NLL")
g.axes[2,0].set_ylabel("$Z^2$")
g.set_titles("")
