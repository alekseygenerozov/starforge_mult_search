import pickle

import numpy as np
import pandas as pd
import seaborn as sns
colorblind_palette = sns.color_palette("colorblind")

from starforge_mult_search.analysis.analyze_stack import npz_stack

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_M = 5
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

suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
npzs_list = []
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
##Getting the final multiplicity of the binary stars, and whether they are in the same multiple system at the end.
npzs_list = [base_new + str(seed) + suff_new + f"/fates_corr.npz" for seed in seeds]
fates_corr = npz_stack(npzs_list)
same_sys_filt = fates_corr["same_sys_filt"]
end_states = fates_corr["end_states"]