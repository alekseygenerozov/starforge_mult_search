import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import pickle
import seaborn as sns
colorblind_palette = sns.color_palette("colorblind")

from starforge_mult_search.analysis.analyze_stack import npz_stack
from starforge_mult_search.analysis.labelLine import labelLines
from starforge_mult_search.analysis.figures.figure_preamble import *

bfb_filter = my_data["same_sys_at_fst"].astype(bool)

fig,ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax.set_xlim(-1, 1)
ax.set_yscale('log')
ax.set_ylabel("PDF")
ax.set_xlabel(r"$cos(\theta)$")
ax.plot([-2, -2], [1,1], "k--", label="Not BFB Bins")
ax.legend()
bins = np.arange(-1,1.01,0.1)

vangs = my_data["vangs"]
vangs_prim = my_data["vangs_prim"]
quasi_filter = my_data["quasi_filter"]

n, bins, patches = ax.hist(vangs[quasi_filter], bins=bins, histtype='step', label="Cluster Frame", linewidth=4, density=True)
bar_color = patches[0].get_edgecolor()
ax.hist(vangs[quasi_filter & ~bfb_filter], bins=bins, histtype='step', linewidth=4, color=bar_color, linestyle="--", density=True)
ax.annotate(f'Cluster\nFrame', xy=(0.88, 2.70), color=bar_color, va="bottom", ha="right")
n, bins, patches = ax.hist(vangs_prim[quasi_filter], bins=bins, histtype='step', label="Primary\nFrame", linewidth=4, density=True)
bar_color = patches[0].get_edgecolor()
ax.hist(vangs_prim[quasi_filter & ~bfb_filter], bins=bins, histtype='step', color=bar_color, linewidth=4, linestyle="--", density=True)
ax.annotate(f'Rel vel & sep', xy=(-1, 2.5), color=bar_color, va="bottom")

iso_height =  1. / (0.1) / len(bins)
l1,=ax.plot([-1, 0.5, 0.75, 1], [iso_height, iso_height, iso_height, iso_height], "-.",  label="Isotropic")
labelLines([l1], xvals=[0])

fig.savefig("fig3a.pdf")

##Getting control age differences seed-by-seed
##Save this info with the data files...
snap_interval = 2.47e4
delta_snapc = np.zeros(len(my_data["bin_ids"]))
idx = 0
for seed in seeds:
    tmp_data_path = base_new + str(seed) + suff_new
    tmp_data = np.load(tmp_data_path + f"/dat_coll{suff}.npz", allow_pickle=True)
    with open(tmp_data_path + "/path_lookup.p", "rb") as ff:
        tmp_path = pickle.load(ff)

    tmp_path_keys = list(tmp_path.keys())
    for ii, uid in enumerate(tmp_data["bin_ids"]):
        k1, k2 = np.random.choice(tmp_path_keys, 2, replace=False)
        bin_list = list(uid)
        pathc1 = tmp_path[k1]
        pathc2 = tmp_path[k2]

        snapc1 = np.where(~np.isinf(pathc1[:, 0]))[0][0]
        snapc2 = np.where(~np.isinf(pathc2[:, 0]))[0][0]
        delta_snapc[idx] = np.abs(snapc2 - snapc1) * snap_interval
        idx += 1

from starforge_mult_search.analysis.plotting import annotate_multiple_ecdf

delta_snap = my_data["delta_snap"]
quasi_filter = my_data["quasi_filter"]

##Caption: Cumulative distribution of age differences between binary stars in the simulation (black) compared to the age differences
##of randomly chosen pairs. Most binary stars form close together in time.
fig,ax = plt.subplots(figsize=(8,8), constrained_layout=True)
ax.set_ylabel("Fraction (Cumulative)")
ax.set_xlabel(r"Age Difference [Myr]")
# ax.legend(title=f"$f_t={int(my_ft)}$")

d1 = delta_snap[~np.isinf(delta_snap) & (quasi_filter)] / 1e6
# d1b = delta_snap[~np.isinf(delta_snap) & (quasi_filter) & (final_bin_filt)] / 1e6
d2 = delta_snapc[~np.isinf(delta_snapc)] / 1e6
annotate_multiple_ecdf((d1,  d2), labels=("Binaries", "Control"), x_offset=(0.3, 0.32))

ax.annotate("Binaries", (0.8, 800))
ax.annotate("Control", (5.5, 800), color='red')
fig.savefig("fig3b.pdf")
