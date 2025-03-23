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
import matplotlib.patches as mpatches

dummy_patch = mpatches.Patch(color='white', label='')

from IPython.core.debugger import set_trace

colorblind_palette = sns.color_palette("colorblind")
import cgs_const as cgs
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import npz_stack
from starforge_mult_search.analysis.high_multiples_analysis import make_hier, get_pair_state, add_node_to_orbit_tab, add_node_to_orbit_tab_streamlined

##CAN HAVE THIS AS AN IMPORT -- NOT THE BEST PRACTICE, BUT THEN AT LEAST WE DON'T HAVE THE SAME CODE OVER AND OVER AGAIN
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
suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
npzs_list = []
# for seed in seeds:
##DO WE NEED TO SEPARATE dat_coll.npz and dat_coll_mult.npz
suff = "_mult"
npzs_list = [base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz" for seed in seeds]
my_data = npz_stack(npzs_list)

npzs_list = [base_new + str(seed) + suff_new + f"/fates_corr.npz" for seed in seeds]
fates_corr = npz_stack(npzs_list)
coll_full_df_life = pd.concat([pd.read_parquet(base_new + str(seed) + suff_new + f"/mults_flat.pq") for seed in seeds])
coll_full_df_life = coll_full_df_life.loc[(high_df["frac_of_orbit"]>=1) & (high_df["nbound_snaps"]>1)]

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


#########################################################################################################
end_states = fates_corr["end_states"]
same_sys_filt = fates_corr["same_sys_filt"]
quasi_filter = my_data["quasi_filter"]
#########################################################################################################
d1 = len(end_states[quasi_filter])
#########################################################################################################
##Tallying all the non-surviving states.
ns1 = len(end_states[(end_states=="1 1") & (quasi_filter) ]) / d1
ns2 = len(end_states[(end_states=="1 2") & (quasi_filter) ]) / d1
ns3 = len(end_states[(end_states=="1 3") & (quasi_filter) ]) / d1
ns4 = len(end_states[(end_states=="1 4") & (quasi_filter) ]) / d1
ns5 = []
ns5.append(len(end_states[(end_states=="2 2") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="2 3") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="2 4") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="3 3") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="3 4") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="4 4") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns = len(end_states[~(same_sys_filt) & (quasi_filter) ]) / d1


print(f"S S:{ns1} B S:{ns2} T S:{ns3} Q S:{ns4} M M:{np.sum(ns5)} NS Tot: {ns1 + ns2 + ns3 + ns4 + np.sum(ns5)} NS Tot CK: {ns}")
print((ns) * d1)
#########################################################################################################
##Tallying all the surviving states.
ss1 = len(end_states[(end_states=="2 2") & (quasi_filter) ]) / d1
ss2 = len(end_states[(end_states=="3 3") & (quasi_filter) ]) / d1
ss3 = len(end_states[(end_states=="4 4") & (quasi_filter) ]) / d1

print(f"B:{ss1} T:{ss2} Q:{ss3} S Tot:{ss1 + ss2 + ss3}")
print(f"B:{ss1 * d1} T:{ss2 * d1} Q:{ss3 * d1}")
print((ss1 + ss2 + ss3) * d1)

##Also save seed info here--will be useful for the table...
# np.savez(suff_new.replace("/", "") + "_fates_corr.npz", end_states=end_states, same_sys_filt=same_sys_filt)

##Good place to put the book-keeping about single stars -- Already done in ex_fig8.py
