import pickle
import tqdm

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

colorblind_palette = sns.color_palette("colorblind")

from starforge_mult_search.analysis.analyze_stack import npz_stack,subtract_path,max_w_infinite,get_min_dist_binary
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.analysis.high_multiples_analysis import lookup_star_mult, parse_mult_id
from labelLine import labelLines

##Try to get rid of this import...
from sci_analysis import plotting

from starforge_mult_search.analysis.figures.figure_preamble import *
#########################################################################################################
##One of the stars was in a persistent multiple before the 2 stars became *binary* -- can also see what
##what happens if we construct a filter based on 1st time the 2 stars became *system*

##Move this code to high_multiples_analysis.py(!)
bin_ids = my_data["bin_ids"]
quasi_filter = my_data["quasi_filter"]
high_df = pd.concat([pd.read_parquet(base_new + str(seed) + suff_new + f"/mults_flat.pq") for seed in seeds])
high_df = high_df.loc[(high_df["frac_of_orbit"] >= 1) & (high_df["nbound_snaps"] > 1)]

mult_ids = high_df.index.get_level_values("id")
mult_ids_set = mult_ids.to_series().apply(parse_mult_id)
high_df["mult_ids_set"] = mult_ids_set.to_list()

tval = high_df.index.get_level_values("t")
high_df["tval"] = tval
pmult_filt = np.zeros(len(bin_ids)).astype(bool)
##
for ii, row in tqdm.tqdm(enumerate(bin_ids)):
    ##Don't care about non-persistent binaries so we can skip them
    if not quasi_filter[ii]:
        continue
    bin_list = list(row)
    ibs = my_data["init_bound_snaps"][ii]
    tmp_sel = high_df.query(f"tval < {ibs}")
    mult_ids_set = tmp_sel["mult_ids_set"]

    # mult_ids_set = np.unique(np.concatenate(mult_ids_set.tolist()))
    ck1 = [bin_list[0] in mult_id for mult_id in mult_ids_set]
    ck1 = np.any(ck1)
    ck2 = [bin_list[1] in mult_id for mult_id in mult_ids_set]
    ck2 = np.any(ck2)

    pmult_filt[ii] = ck1 or ck2

#########################################################################################################
npzs_list = [base_new + str(seed) + suff_new + f"/fates_corr.npz" for seed in seeds]
fates_corr = npz_stack(npzs_list)
same_sys_filt = fates_corr["same_sys_filt"]
end_states = fates_corr["end_states"]

bin_ids = my_data["bin_ids"]
quasi_filter = my_data["quasi_filter"]
##Replace with updated filter
##Surviving binaries including those in higher order multiples(!!)
bin_ids_11 = bin_ids[quasi_filter &  (end_states=="1 1")]
bin_ids_subset = bin_ids_11
norm_sep = np.zeros(len(bin_ids_subset))
mult_after_destruction = np.zeros(len(bin_ids_subset))

for idx, uid in tqdm.tqdm(enumerate(bin_ids_subset)):
    bin_list = list(uid)
    tmp_row = np.array(bin_list).astype(str)
    sys1_info = lookup_dict[bin_list[0]]
    sys2_info = lookup_dict[bin_list[1]]
    b1, b2, xxxxx = analyze_multiples_part2.get_bound_snaps(sys1_info, sys2_info)

    tmp_times = b1[:,0].astype(int)
    fb, lb = tmp_times[0], tmp_times[-1]
    path_diff_all = get_min_dist_binary(path_lookup, tmp_row)
    norm_sep[idx] = min(path_diff_all[lb][0], path_diff_all[lb + 1][0]) / (2 * b1[-1, LOOKUP_SMA])
    try:
        mult1 = sys1_info[sys1_info[:,LOOKUP_SNAP]==lb+1][0, LOOKUP_MULT]
        mult2 = sys2_info[sys2_info[:,LOOKUP_SNAP]==lb+1][0, LOOKUP_MULT]
    except IndexError:
        breakpoint()

    mult1 = lookup_star_mult(high_df, bin_list[0], lb + 1, pre_filtered=False)
    mult2 = lookup_star_mult(high_df, bin_list[1], lb + 1, pre_filtered=False)
    mult_after_destruction[idx] = max(mult1[1], mult2[1])

norm_sep_og = np.copy(norm_sep)
print(f"Frac in mult after destruction: {len(mult_after_destruction[mult_after_destruction > 1]) / len(mult_after_destruction)}")
#########################################################################################################
bin_ids = my_data["bin_ids"]
quasi_filter = my_data["quasi_filter"]
final_bound_snaps_norm = my_data["final_bound_snaps_norm"]
##May want to revisit this filter
no_mult_before_bin = my_data["mults_filt"]
##NOTE: Deliberately take different filters here to have a clean sample
##Makes signal much weaker(!)
bin_ids_surv = bin_ids[quasi_filter & (final_bound_snaps_norm==1) & ~(pmult_filt)]
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
vd_b, b1, tmp1 = plt.hist(np.log10((final_pair_mass_no_halo[quasi_filter & ~(same_sys_filt)])), bins=bins,
                       histtype='step')
vs_b, b2, tmp2 = plt.hist(np.log10(final_pair_mass_no_halo[quasi_filter & (same_sys_filt)]), bins=bins,
                       histtype='step')

fig,ax = plt.subplots(figsize=(8,8), constrained_layout=True)
ax.set_ylabel("$N_{surv}$ / $N_{dis}$")
ax.set_xlabel("log($m_{pair, f}$ [$M_{\odot}$])")
plt.plot(0.5 * (b1[1:] + b1[:-1]), vs_b / vd_b, "s-")

fig.savefig("nsurv_mass.pdf")
