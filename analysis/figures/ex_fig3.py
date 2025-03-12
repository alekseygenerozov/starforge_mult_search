import ast
from collections import defaultdict
import copy
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sci_analysis.plotting import annotate_multiple_ecdf
from scipy.stats import ks_2samp
import seaborn as sns

colorblind_palette = sns.color_palette("colorblind")
import cgs_const as cgs
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import npz_stack
from starforge_mult_search.analysis.high_multiples_analysis import make_hier, get_pair_state, add_node_to_orbit_tab, add_node_to_orbit_tab_streamlined

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_SMA = 6
LOOKUP_ECC = 7
LOOKUP_Q = 8

sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))
sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
vxcol = np.where(sink_cols == "vx")[0][0]
vycol = np.where(sink_cols == "vy")[0][0]
vzcol = np.where(sink_cols == "vz")[0][0]
hcol = np.where(sink_cols == "h")[0][0]
mcol = np.where(sink_cols == "m")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]
scol = np.where(sink_cols == "sys_id")[0][0]

##Stacking data
my_ft = 1.0
my_tides = False
base_new = "../analysis_pipeline_experiment/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_"
seeds = (1, 2, 42)
seeds_idx = (0, 1, 2)
end_snaps = np.array((464, 423, 489))
start_snaps = np.array((44, 48, 48))
# seeds = (42,)
# seeds_idx = (0,)
# end_snaps = np.array((489,))
suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
npzs_list = []
# for seed in seeds:
##DO WE NEED TO SEPARATE dat_coll.npz and dat_coll_mult.npz
suff = "_mult"
npzs_list = [base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz" for seed in seeds]
seeds_lookup = np.concatenate(
    [[seed] * len(np.load(base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz", allow_pickle=True)["bin_ids"]) for
     seed in seeds])
seeds_lookup_idx = np.concatenate([[seeds_idx[seed_idx]] * len(
    np.load(base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz", allow_pickle=True)["bin_ids"]) for seed_idx, seed
                                   in enumerate(seeds)])
my_data = npz_stack(npzs_list)
coll_full_df_life = pd.concat([pd.read_parquet(base_new + str(seed) + suff_new + f"/mults_flat.pq") for seed in seeds])

path_lookup = {}
spin_lookup = {}
lookup_dict = {}
for seed in seeds:
    tmp_dat_path = base_new + str(seed) + suff_new
    with open(tmp_dat_path + "/path_lookup.p", "rb") as ff:
        tmp_path_pickle = pickle.load(ff)
        assert not np.any(np.isin(tmp_path_pickle.keys(), path_lookup.keys()))
        path_lookup.update(tmp_path_pickle)

    with open(tmp_dat_path + "/spin_lookup.p", "rb") as ff:
        tmp_path_pickle = pickle.load(ff)
        assert not np.any(np.isin(tmp_path_pickle.keys(), spin_lookup.keys()))
        spin_lookup.update(tmp_path_pickle)

    with open(tmp_dat_path + "/lookup_dict.p", "rb") as ff:
        tmp_path_pickle = pickle.load(ff)
        assert not np.any(np.isin(tmp_path_pickle.keys(), spin_lookup.keys()))
        lookup_dict.update(tmp_path_pickle)

snap_interval = my_data["snap_interval"][0]
#########################################################################################################
lookup_dict_keys = lookup_dict.keys()
lookup_dict_keys = list(lookup_dict_keys)
first_mult = np.ones(len(lookup_dict_keys)) * np.inf
f1 = coll_full_df_life["frac_of_orbit"]
n1 = coll_full_df_life["nbound_snaps"]
tmp_sel = coll_full_df_life.loc[(f1>1) & (n1>1)]
##Filter for selecting first instance of each index
filt = ~tmp_sel.index.get_level_values("id").duplicated(keep="first")
tmp_sel = tmp_sel.loc[filt]

for ii,kk in enumerate(lookup_dict_keys):
    tmp_filt = tmp_sel.index.get_level_values("id").str.contains(rf"\b{int(kk)}\b")
    tmp_delay = tmp_sel.loc[tmp_filt].index.get_level_values("t").min()

    if not np.isnan(tmp_delay):
        first_mult[ii] = tmp_delay
#########################################################################################################
lookup_dict_keys = lookup_dict.keys()
n1 = len(lookup_dict_keys)
mass_end = np.zeros(n1)
delay_to_mult = np.ones(n1) * np.inf

##Better to have some sort of persistence filter here even if it is a basic one??
for idx,kk in enumerate(lookup_dict_keys):
    tmp = lookup_dict[kk]
    delay_to_mult[idx] = first_mult[idx] - tmp[0, LOOKUP_SNAP]
    m_series = path_lookup[f"{int(kk)}"][:, mcol]
    mass_end[idx] = m_series[~np.isinf(m_series)][-1]
#########################################################################################################
import matplotlib.colors as mcolors
import matplotlib.cm as cm

fig,ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
# ax.set_title(f"Explicit tides={my_tides}, ft={my_ft}")
ax.set_xlim(0.01, 1)
ax.set_ylim(0., 1)
ax.set_ylabel("CDF")
ax.set_xlabel("Delay to multiple [Myr]")
from matplotlib.lines import Line2D

bins = np.linspace(-1, 1, 10)
# seq_palette = sns.color_palette("Blues", len(bins))
cmap = sns.color_palette("Blues", as_cmap=True)  # Convert seaborn palette to a colormap
norm = mcolors.Normalize(vmin=bins[0], vmax=bins[-1])  # Normalize bins for color mapping

legend_handles = []
for ii in range(1, len(bins)):
    col = cmap(norm(bins[ii]))
    ax.ecdf(delay_to_mult[(np.log10(mass_end)<bins[ii]) & (np.log10(mass_end)>bins[ii-1])] * snap_interval / 1e6,
           color=col)

sm = cm.ScalarMappable(cmap=cmap, norm=norm)
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax, label=r"$log(m_f)$")
plt.show()
fig.savefig("delay_to_mult.pdf")