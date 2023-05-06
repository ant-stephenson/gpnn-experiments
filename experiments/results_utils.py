import pandas as pd
from pathlib import Path
import re
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import gpybench.plotting as gplot

Idx = pd.IndexSlice

def format_results(df):
    df.columns = df.columns.str.replace(" ", "")
    if "per_predict_time" in df.columns:
        df.rename({"per_predict_time": "Time per prediction"}, axis=1, inplace=True)
        df.loc[:, "Time per prediction"] *= 1000
    else:
        df.loc[:,"Time per prediction"] = df["pred_time"]/df["num_test"] * 1000 # units ms
    # df.loc[:, "train_time"] /= 60 #units: minutes
    df.loc[:, "dataset"] = df.loc[:,"dataset"].str.replace(".npy","").apply(lambda x: x.title())

    # rescale num_points by 1000 for brevity
    # df.num_points = (df.num_points/1000).round(1).astype(str) + "K"
    # df.num_train = (df.num_train/1000).round(1).astype(str) + "K"
    # df.num_test = (df.num_test/1000).round(1).astype(str) + "K"

    # df.num_train = df.num_train.astype(float)
    df.num_train = ((df.num_train/100).round(1)*100).round(1)

    # rename stuff
    df.rename({"dataset": "Dataset", "dimension": "d", "num_train": "n"}, inplace=True, axis=1)
    return df

# def style_to_latex(style):
#     import pypandoc

#     return pypandoc.convert_text(style.render(), to="latex", format="html")

def reformat(df):
    midx = df.index.droplevel(["pre_processing", "traintest_seed"]).unique()
    df = df.unstack()
    df.columns = df.columns.droplevel(0).rename(None)
    df = df.set_index(midx)
    return df

def round_to_n(x, n):
    if np.isnan(x):
        return np.nan
    return round(x, -int(np.floor(np.log10(np.abs(x)))) + (n - 1))

def combine_mean_std(mean, std):
    means = mean.applymap(lambda x: round_to_n(x, 3)).astype(str)
    stds = std.applymap(lambda x: round_to_n(x, 2)).astype(str)
    out = means.applymap(lambda x: x + " ± ") + stds
    return out

def summarise(df, idx):
    df_grp = df.groupby(idx)
    out = combine_mean_std(df_grp.mean(), df_grp.std())
    return out

def uncertain_str_to_float(s):
    if isinstance(s, str):
        return float(s.split("±")[0])
    else:
        return s

def column_map(x):
    x = (x[0].replace("_"," "), x[1].replace("_", " "))
    return  (x[0].title() if len(x[0]) > 5 else x[0].upper(),x[1].title() if len(x[1]) > 5 else x[1].upper())

def format_cols(df):
    df.columns = df.columns.map(column_map)
    df.columns.names = [None, None]
    return df

def set_nidx_notation(df):
    df["n"] = df.index.get_level_values("n")
    df.index = df.index.droplevel("n")
    df["n"] = df["n"].map('{:.2}'.format)
    df.set_index("n", append=True, inplace=True)
    new_order = ["Dataset", "n", "d"] + list(set(df.index.names).difference(["Dataset", "n", "d"]))
    df = df.reorder_levels(new_order)
    return df

# use to get best overall distributed method
def combine_dist_and_join_svgp(distdf, svgpdf, agg_map, pick_min_mse=False, min_idx=None):
    dist_longdf = distdf.stack().loc[Idx[:,:,:,"whitening"]]
    if min_idx is not None:
        dist_longdf = dist_longdf.loc[min_idx, :]
    if pick_min_mse:
        min_idx = dist_longdf.groupby(level=list(range(dist_longdf.index.nlevels-1))).RMSE.agg(uncertain_str_min_idx)
        dist_longdf = dist_longdf.loc[min_idx, :]
        #rm index with chosen model type (if desired)
        dist_longdf = dist_longdf.droplevel(-1)
    else:
        dist_longdf = dist_longdf.groupby(level=list(range(dist_longdf.index.nlevels-1))).agg(agg_map)
        min_idx = dist_longdf.index
    dist_longdf.columns = pd.MultiIndex.from_product([dist_longdf.columns, ['Distributed']]).map(column_map)
    joint = svgpdf.loc[Idx[:,:,:,"whitening"],:].droplevel("pre_processing").join(dist_longdf)
    joint = joint.sort_index(level="n")
    # if we want exp float notation for n.
    joint = set_nidx_notation(joint)
    return joint, min_idx

def calibration_min(x):
    idx = ((x.apply(uncertain_str_to_float) - 1)**2).idxmin()
    return x[idx]

def uncertain_str_min_idx(x):
    return (x.apply(uncertain_str_to_float)).idxmin()

def uncertain_str_min(x):
    idx = uncertain_str_min_idx(x)
    return x[idx]

def best_by_group(col, func):
    """ Gets "best" model by index group and makes bold. Should ideally include
    uncertainty in choice of "best". Atm just does by mean.

    Args:
        col (_type_): _description_

    Returns:
        _type_: _description_
    """
    # if col.name == "Calibration":
    #     func = calibration_min
    # else: 
    #     func = uncertain_str_min
    f = lambda x: x
    col_min = col.groupby(level=0).transform(func)
    return (col.apply(f) == col_min).map({True: 'font-weight: bold', False: ''})

def add_midrules(table_str):
    return re.sub(r"^\\multirow", r"\\midrule\n\\multirow", table_str, flags=re.MULTILINE)

def tablewidth_adjust(table_str, caption):
    table_str = re.sub(r"^\\begin{tab", fr"\\begin{{adjustbox}}{{max width=\\textwidth, caption={{{caption}}}, float=table}}\n\\begin{{tab", table_str, flags=re.MULTILINE, count=1)
    table_str = re.sub(r"^(\\end{tabl.*$)", r"\1\n\\end{adjustbox}", table_str, flags=re.MULTILINE, count=1)
    return table_str

def mathify(table_str):
    table_str = re.sub(r"\bn\b", r"\(n\)", table_str, flags=re.MULTILINE)
    table_str = re.sub(r"\bd\b", r"\(d\)", table_str, flags=re.MULTILINE)
    return table_str

def rm_table_env(table_str):
    table_str = re.sub(r"\\begin{table}\n", r"", table_str, flags=re.MULTILINE)
    table_str = re.sub(r"\\end{table}\n", r"", table_str, flags=re.MULTILINE)
    return table_str

def format_style(df, axis=0):
    df_style = df.style

    # set best model to bold
    if "Calibration" in df_style.columns:
        df_style.apply(lambda x: best_by_group(x, calibration_min), axis=axis, subset=["Calibration"])
        if "RMSE" in df_style.columns and "NLL" in df_style.columns:
            df_style.apply(lambda x: best_by_group(x, uncertain_str_min), axis=axis, subset=["RMSE", "NLL"])
    else:
        df_style.apply(lambda x: best_by_group(x, uncertain_str_min), axis=axis)
    return df_style

def style_to_latex(df_style, caption="", label="", adjust_tablewidth=False, **kwargs):
    df_style.format_index("\\textbf{{{}}}", escape="latex", axis=1, level=0)
    if not adjust_tablewidth:
        latex = df_style.to_latex(convert_css=True, hrules=True,caption=caption,label=label, multicol_align="l", environment="")
    else:
        latex = df_style.to_latex(convert_css=True, hrules=True,label=label, multicol_align="l", environment="", **kwargs)
    latex = add_midrules(latex)
    latex = mathify(latex)
    if adjust_tablewidth:
        latex = tablewidth_adjust(latex, caption)
        latex = rm_table_env(latex)
    return latex

def format_to_latex(df, axis=0, caption="", label="", adjust_tablewidth=False, **kwargs):
    df_style = format_style(df, axis)
    return style_to_latex(df_style)

def create_summary_table(df, values):
    idx_cols = ["Dataset","n","d","pre_processing","traintest_seed"]
    new_df = df.pivot_table(index=idx_cols, columns="model_type", values=values)
    return summarise(format_cols(new_df), idx_cols[:-1])

def combine_results(dist, svgp, mop, metric_map, pick_min_mse=False, min_idx=None):
    summary, min_idx = combine_dist_and_join_svgp(dist, svgp, metric_map, pick_min_mse=pick_min_mse, min_idx=min_idx)
    summary = summary.join(set_nidx_notation(mop.droplevel("pre_processing"))).sort_index(level=0, axis=1)

    summary.rename({"Variational": "SVGP"}, inplace=True, axis=1)
    return summary, min_idx

def plot_model_metric(data, metric = "calibration", file_type = "png", xticksize=15, yticksize=15, save=True, plot_fn=sns.barplot, rescale=False):
    plot_data = data.copy()
    if rescale:
        plot_data[metric] = plot_data[metric].groupby(data.index).transform(lambda x: x /x.abs().max())
        rescale_filename = "_rescaled"
        rescale_title = " (Relative to extreme)"
    else:
        rescale_filename = ""
        rescale_title = ""
    plt.close()
    with mpl.rc_context({"xtick.labelsize": xticksize, "ytick.labelsize": yticksize, "legend.fontsize": 15}):
        fg = plot_fn(data=plot_data.loc[:,:].reset_index(), x="Dataset", y=metric,
        hue="model_type")
        fg.legend_.set_title(None)
        ax = plt.gca()
        fontsize = 18
        if metric == "calibration":
            plt.title(metric.title())
            plt.axhline(1, linestyle="--", color="k")
            ylim_u = np.min([2,plot_data.loc[:,metric].max()])
        else:
            plt.title(metric.upper())
            ylim_u = np.min([5,plot_data.loc[:,metric].max()])
        ylim_l = plot_data.loc[:,metric].min()
        plt.ylim(ylim_l, ylim_u)
        plt.xlabel(None)
        plt.ylabel(None)
        if metric in ["calibration", "nll"]:
            plt.yscale("symlog")
            ax.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
            plt.title(f"symlog({ax.get_title()})")
        if metric == "calibration":
            plt.yticks([1.0])
        for tick in ax.get_xticklabels():
            tick.set_rotation(45)
        if metric == "rmse":
            plt.legend(bbox_to_anchor=(1,1), handlelength=0.75, borderpad=0.2, handletextpad=0.4, borderaxespad=0.25)
        else:
            ax.get_legend().remove()
        plt.title(f"{ax.get_title()}" + rescale_title, fontsize=fontsize)
        if save:
            gplot.save_fig(Path("."), f"{metric}_vs_model_whitening" + rescale_filename, file_type, show=True, overwrite=True)