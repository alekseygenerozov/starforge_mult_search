import pickle
import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns

colorblind_palette = sns.color_palette("colorblind")

from starforge_mult_search.analysis.analyze_stack import npz_stack,subtract_path,max_w_infinite,get_min_dist_binary
from starforge_mult_search.analysis import analyze_multiples_part2
from labelLine import labelLines

##Try to get rid of this import...
from sci_analysis import plotting
from scipy.stats import ks_2samp

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
##DONT USE FATES HERE -- DOES NOT HAVE THE CORRECT BOOKKEEPING FOR SINGLE STARS.

seeds = (1, 2, 42)
my_ft = 1.0
my_tides = False
base_new = "../new_analysis/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_"
suff = "_mult"

suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
npzs_list = []
npzs_list = [base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz" for seed in seeds]
my_data = npz_stack(npzs_list)

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
fates_corr = np.load(suff_new.replace("/", "") + "_fates_corr.npz")
same_sys_filt = fates_corr["same_sys_filt"]
end_states = fates_corr["end_states"]

bin_ids = my_data["bin_ids"]
quasi_filter = my_data["quasi_filter"]
##Replace with updated filter
##Surviving binaries including those in higher order multiples(!!)
bin_ids_11 = bin_ids[quasi_filter &  (end_states=="1 1")]
bin_ids_subset = bin_ids_11
norm_sep = np.zeros(len(bin_ids_subset))

for idx, uid in enumerate(bin_ids_subset):
    bin_list = list(uid)
    tmp_row = np.array(bin_list).astype(str)
    b1, b2, xxxxx = analyze_multiples_part2.get_bound_snaps(lookup_dict[bin_list[0]], lookup_dict[bin_list[1]])

    tmp_times = b1[:,0].astype(int)
    fb, lb = tmp_times[0], tmp_times[-1]
    path_diff_all = get_min_dist_binary(path_lookup, tmp_row)
    norm_sep[idx] = min(path_diff_all[lb][0], path_diff_all[lb + 1][0]) / (2 * b1[-1, LOOKUP_SMA])

norm_sep_og = np.copy(norm_sep)
#########################################################################################################
bin_ids = my_data["bin_ids"]
quasi_filter = my_data["quasi_filter"]
final_bound_snaps_norm = my_data["final_bound_snaps_norm"]
no_mult_before_bin = my_data["mults_filt"]
##NOTE: Deliberately take different filters here to have a clean sample
bin_ids_surv = bin_ids[quasi_filter & (final_bound_snaps_norm==1) & (no_mult_before_bin)]
print(len(bin_ids_surv))
norm_sep = np.zeros(len(bin_ids_surv))
bin_ids_subset = bin_ids_surv

for idx, uid in enumerate(bin_ids_subset):
    bin_list = list(uid)
    tmp_row = np.array(bin_list).astype(str)
    b1, b2, xxxxx = analyze_multiples_part2.get_bound_snaps(lookup_dict[bin_list[0]], lookup_dict[bin_list[1]])

    path_diff_all = get_min_dist_binary(path_lookup, tmp_row)
    ##Minimum distance for all surviving binaries
    norm_sep[idx] = np.min(path_diff_all[b1[:,0].astype(int)][:,0] / (2 * b1[:, LOOKUP_SMA]))
#########################################################################################################
fig,ax = plt.subplots(figsize=(8,8), constrained_layout=True)
ax.set_xlim(0.05, 1000)
ax.set_xscale("log")
ax.set_xlabel("Min[$d_{ext} / (2 a_{bin})$]")
ax.set_ylabel("Fraction")
pval = ks_2samp(norm_sep, norm_sep_og).pvalue
ax.legend(title=f"KS p-value={pval:.2g}", loc="upper left", frameon=True)

plotting.annotate_multiple_ecdf((norm_sep, norm_sep_og),\
                       ("Surviving\n(no mult\ninteractions)", "Ionized",  "Min(Lb and Lb+1)", "Lb", "traj_extrap"), ax=ax,
                       levels=(60, 60, 50, 75, 80), ha=["left", "right"], x_offset=(6, -.6), y_offset=-0.04, colors=['0.5', None, None, None], linestyles=["--", None, None, None])
fig.savefig("close_encounter_plot.pdf")
#########################################################################################################
bin_ids = my_data["bin_ids"]
quasi_filter = my_data["quasi_filter"]
no_mult_before_bin = my_data["mults_filt"]
##Replace with updated filter
# same_sys_final_norm = my_data["same_sys_final_norm"]
final_bound_snaps_norm = my_data["final_bound_snaps_norm"]
final_pair_mass_no_halo = my_data["mfinal_pair"]

bins = np.arange(-1, 1.21, 0.2)
##Does not include the small fraction of binaries with deleted stars.
vd_b, b1, tmp1 = plt.hist(np.log10((final_pair_mass_no_halo[quasi_filter & (same_sys_filt)])), bins=bins,
                       histtype='step')
vs_b, b2, tmp2 = plt.hist(np.log10(final_pair_mass_no_halo[quasi_filter & (same_sys_filt)]), bins=bins,
                       histtype='step')

fig,ax = plt.subplots(figsize=(8,8), constrained_layout=True)
ax.set_ylabel("$N_{surv}$ / $N_{dis}$")
ax.set_xlabel("log($m_{pair, f}$ [$M_{\odot}$])")
plt.plot(0.5 * (b1[1:] + b1[:-1]), vs_b / vd_b, "s-")

fig.savefig("nsurv_mass.pdf")
