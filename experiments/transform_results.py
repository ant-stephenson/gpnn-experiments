#%%
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from results_utils import *

Idx = pd.IndexSlice
pd.set_option('display.float_format', lambda x: '%0.4g' % x)

def get_results(path, kernel_type="RBF"):
    results = pd.read_csv(path)
    if "partition_seed" in results.columns:
        results.traintest_seed = results.partition_seed
    results = format_results(results)
    if isinstance(kernel_type, list):
        results.model_type += results.kernel_type.apply(lambda x: f" ({x})")
    else:
        results.query('kernel_type == @kernel_type', inplace=True)
    summary = create_summary_table(results, values=["rmse","nll","calibration"])
    timings = create_summary_table(results, values=["Time per prediction","train_time"])
    return results,summary, timings

#%% get results and add normalised prediction time
dist_loc = Path("/Users/anthonystephenson/Documents/Distributed_Results_April_24th.csv")
svgp_loc = Path("/Users/anthonystephenson/Documents/SVGP_Results_April_21st.csv")
mop_loc = Path("/Users/anthonystephenson/Documents/csv_results_output_file2c_with_dscal.csv")


#TODO: UPDATED RESULTS WOULD REMOVE THIS. IF TRUE, REPLACES WRONG n VALUE WITH
#SAME AS EVERYTHING ELSE SO WE CAN JOIN, BUT TECHNICALLY NOT FAIR
# USE_WRONG_GRBCM = False
# if USE_WRONG_GRBCM:
#     idx = dist_results.query('Dataset == "Song" & traintest_seed == 1 & pre_processing == "whitening" & model_type == "grbcm"').index
#     dist_results.loc[idx, "n"] = dist_results.loc[dist_results.loc[:,"Dataset"] == "Song", "n"].mode()

# choose whitening or scaling
standard = "whitening"

#%%
# kernel_type = "Exp"

dist_results, summary_dist, summary_timings_dist = get_results(dist_loc)
svgp_results, summary_svgp, summary_timings_svgp = get_results(svgp_loc)
mop_results, summary_mop, summary_timings_mop = get_results(mop_loc, "RBF")
mop_exp, summary_mop_exp, _ = get_results(mop_loc, "Exp")
mop_mat, summary_mop_mat, _ = get_results(mop_loc, "Matern")
mop_all, summary_mop_all, _ = get_results(mop_loc, ["RBF", "Exp", "Matern"])

#%%
metric_map = {"Calibration": calibration_min, "RMSE": uncertain_str_min, "NLL": uncertain_str_min}
best_summary, min_idx = combine_results(summary_dist, summary_svgp, summary_mop, metric_map, True)

best_summary_kernels, _ = combine_results(summary_dist, summary_svgp, summary_mop_all, metric_map, True)

# how do we do timings; use whatever we used for metrics
timings_summary, _ = combine_results(summary_timings_dist, summary_timings_svgp, summary_timings_mop, uncertain_str_min, min_idx = min_idx)

#%%
# Drop Buzz and Calibration for now
best_summary = best_summary.drop("Buzz")
best_summary = best_summary.drop("Calibration", axis=1)
best_summary_kernels = best_summary_kernels.drop("Buzz")
timings_summary = timings_summary.drop("Buzz")

#%%
summary_all = summary_svgp.join(summary_dist).join(summary_mop)
summary_all.drop("Buzz", inplace=True)
summary_all.rename({"Variational": "SVGP"}, inplace=True, axis=1)
summary_all_long = summary_all.stack().loc[Idx[:,:,:,standard]]
summary_all_long.index.names = [name if name is not None else "Model" for name in summary_all_long.index.names]
summary_all_long = set_nidx_notation(summary_all_long)
# %% formatting and style
# all methods in long format
caption_metrics_all = ("Results for all methods on all metrics.", "")
latex_table_all = format_to_latex(summary_all_long, label="tab:metrics_all_long", caption=caption_metrics_all)
print(latex_table_all)
# %% best combined distributed + svgp + mop
caption_metrics_best = ("RMSE and NLL results (mean and standard deviation over 3 runs) for the best distributed method (w.r.t. MSE), SVGP and our method.")
latex_best = format_to_latex(best_summary, axis=1, label="tab:metrics_best_dist", caption=caption_metrics_best, adjust_tablewidth=True)
print(latex_best)
#%% best dist + svgp (timings). Drop prediction for now
caption_timings_best = (r"Corresponding timings (with mean and standard deviation) associated to the metrics in \\autoref{tab:metrics_best_dist}.")
timings_summary.rename({"Time Per Prediction": "Time per prediction/ms", "Train Time": "Train time/s"}, inplace=True, axis=1)
timings_summary.drop("Time per prediction/ms", axis=1, inplace=True)
latex_timings = format_to_latex(timings_summary, axis=1, label="tab:timings_best_dist", caption=caption_timings_best, adjust_tablewidth=True)
print(latex_timings)

#%% tables for comparing kernels
caption_kernels = ("@METRIC results for the best distributed method, SVGP and our method run on an RBF, Matérn-3/2 and Exponential kernel.")

for metric in ["Calibration", "RMSE", "NLL"]:
    # print(metric)
    # print(best_summary_kernels.loc[:, Idx[metric, :]])
    latex_k = format_to_latex(best_summary_kernels.loc[:, Idx[metric, :]], axis=1, label=f"tab:{metric.lower()}_best", caption=caption_kernels.replace("@METRIC", metric), adjust_tablewidth=True)
    print(latex_k)
#%%
if kernel_type != "RBF":
    raise Exception
# %% plots
# first combine results and do some modifying
all_results = pd.concat([dist_results, svgp_results, mop_results])
all_results = all_results.set_index(["Dataset","n","d","pre_processing"])
all_results = all_results.loc[Idx[:,:,:,standard],:].droplevel("pre_processing")
all_results = all_results.drop("Buzz")
all_results.model_type = all_results.model_type.apply(lambda x: x.upper())
all_results.model_type.replace({"VARIATIONAL": "SVGP"}, inplace=True)

#%%plot timings
tplot_results = all_results.replace({'BCM': 'Dist.', 'GPOE': 'Dist.', 'GRBCM': 'Dist.', 'POE': 'Dist.', 'RBCM': 'Dist.'})

tmp = tplot_results.query("model_type == 'OURS'").copy()
# tmp.replace({"OURS": "OURS\\textsuperscript{\\textdagger}"}, inplace=True)
tmp.replace({"OURS": r"OURS${}^\dagger$"}, inplace=True)
tmp.train_time = tmp.train_time - tmp.time_phase2
tplot_results = pd.concat([tplot_results, tmp]).reset_index()

tplot_results.rename({"model_type": "Model"}, axis=1, inplace=True)
tplot_results.d = tplot_results.d.replace({13: "<20", 8: "<20", 19: "<20", 9: "<20", 2: "<20", 90: ">20", 378: ">20"})

fg = sns.relplot(
    data=tplot_results,
    x="n", y="train_time", 
    hue="Model",
    size="d", size_order=[">20","<20"]
).set(ylabel=None)
fg.map_dataframe(sns.lineplot, "n", "train_time", hue="Model", style="d")
fg.ax.patch.set_edgecolor('black') 
fg.ax.patch.set_linewidth(1)
plt.yscale('log')
plt.xscale('log')
plt.ylabel(None)
plt.xlabel("$n$", fontsize=18)
sns.move_legend(fg, loc="upper left", bbox_to_anchor=(0.2, 0.9), fontsize=12, borderpad=0., handletextpad=0., borderaxespad=0.)
#%%
fg = sns.relplot(
    data=all_results, kind="line",
    x="n", y="Time per prediction", 
    hue="model_type", 
)
plt.yscale('log')
plt.xscale('log')
plt.ylabel("Time per prediction/ms")
fg.legend.set_title(None)
#%%
#%%
plot_model_metric(all_results, "calibration", save=True, rescale=False)
#%%
plot_model_metric(all_results, "rmse", save=True, rescale=False)
#%%
plot_model_metric(all_results, "nll", save=True, rescale=True)
#%% get gradients of log-log plots
# def approx_grad(df, model_type, ylabel, xlabel):
#     y_summary = df.loc[df.model_type == model_type, ylabel].describe()
#     x_summary = df.reset_index().loc[df.reset_index().model_type == model_type, xlabel].describe()
#     return (np.log(y_summary.loc["max"]) - np.log(y_summary.loc["min"]))/(np.log(x_summary.loc["max"]) - np.log(x_summary.loc["min"]))

# approx_grad(all_results, "grbcm", "Time per prediction", "n")
# # %% tables for github. nothing works properly... to_html() best so far.
# md_summary = (
#         best_summary.reset_index()
#           .rename(columns={name:'' for name in best_summary.index.names})
#           .to_markdown(tablefmt='github', index=False)
#       )
best_summary.to_html().replace("\n", "")
# %%
