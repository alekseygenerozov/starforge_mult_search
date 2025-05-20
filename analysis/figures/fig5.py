import copy
import pickle
import tqdm

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

colorblind_palette = sns.color_palette("colorblind")

from starforge_mult_search.analysis.analyze_stack import npz_stack,subtract_path,max_w_infinite,get_min_dist_binary,get_soft_times,get_bound_snaps_adjust
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.analysis.high_multiples_analysis import lookup_star_mult, parse_mult_id
from labelLine import labelLines

##Try to get rid of this import...
from sci_analysis import plotting
from starforge_mult_search.analysis.figures.figure_preamble import *

#########################################################################################################
## Constructing new filter: whether
## one of the stars was in a persistent multiple before the 2 stars became *binary*
bin_ids = my_data["bin_ids"]
quasi_filter = my_data[f"quasi_filter{contig_suff}"]
high_df = pd.concat([pd.read_parquet(base_new + str(seed) + suff_new + f"/mults{flat_suff}.pq") for seed in seeds])
high_df = high_df.loc[(high_df[f"frac_of_orbit{contig_suff}"] >= 1) & (high_df[f"nbound_snaps{contig_suff}"] > 1)]

mult_ids = high_df.index.get_level_values("id")
mult_ids_set = mult_ids.to_series().apply(parse_mult_id)
high_df["mult_ids_set"] = mult_ids_set.to_list()

tval = high_df.index.get_level_values("t")
high_df["tval"] = tval
pmult_filt = np.zeros(len(bin_ids)).astype(bool)
##ex_time is like a "floor" for the exchange time -- if the pair is bound after this time it is considered an exchange binary(!)
##Want separate data with the true exchange time...
ex_time = np.ones(len(bin_ids)) * np.inf
ex_time_end = np.ones(len(bin_ids)) * np.inf
ex_time_max = np.ones(len(bin_ids)) * np.inf
ex_time_max_end = np.ones(len(bin_ids)) * np.inf
bins_first_bound = np.ones(len(bin_ids)) * np.inf
bins_last_bound = np.ones(len(bin_ids)) * np.inf
print(two_body)
for ii, row in tqdm.tqdm(enumerate(bin_ids)):
    ##Don't care about non-persistent binaries so we can skip them
    if not quasi_filter[ii]:
        continue
    bin_list = list(row)
    ##Save bound_snaps data to save time(!!)
    ##Use high_df table to get more stringent binary snapshots(!!!)
    curr_bin_list = list(bin_ids[ii])
    curr_bin_list.sort()
    bin_sel = high_df.loc[str(curr_bin_list)]
    ##Try/except -- should no longer be necessary -- the binary contiguous filter should be consistent.
    # try:
    #     bin_sel = high_df.loc[str(curr_bin_list)]
    # except KeyError:
    #     continue
    bs = bin_sel["tval"].to_numpy()
    ibs = bs[0]
    bins_first_bound[ii] = bs[0]
    bins_last_bound[ii] = bs[-1]

    fst = my_data["fst"][ii]
    tmp_sel = high_df.loc[(tval >= fst) & (tval < bs[-1])]
    ##IDEAS: Require binary * physically closer to another one...
    bin_exclude = ~tmp_sel["tval"].isin(bs)
    tmp_sel = tmp_sel.loc[bin_exclude]
    mult_ids_set = tmp_sel["mult_ids_set"]
    ##Additional filtering here--only select those cases where multiple is quasi-persistent based on prior snapshots.
    ck1 = [bin_list[0] in mult_id for mult_id in mult_ids_set]
    # ck1 = np.any(ck1)
    ck2 = [bin_list[1] in mult_id for mult_id in mult_ids_set]
    # ck2 = np.any(ck2)
    tmp_sel2a = tmp_sel.loc[ck1]
    tmp_sel2b = tmp_sel.loc[ck2]

    # potential_ck(tmp_sel2a, bin_list[0], bin_list[1])
    # potential_ck(tmp_sel2b, bin_list[1], bin_list[0])
    soft_times = get_soft_times(bin_list[0], bin_list[1], path_lookup)

    # tmp_sel2a = tmp_sel2a.loc[~np.isin(tmp_sel2a["tval"], soft_times)]
    # tmp_sel2b = tmp_sel2b.loc[~np.isin(tmp_sel2b["tval"], soft_times)]
    if len(tmp_sel2a) > 0:
        mult_a_times = tmp_sel2a["tval"]
        bs_after_mult = bs[bs > mult_a_times.min()][0]
        ex_time[ii] = mult_a_times[mult_a_times < bs_after_mult].max()
        ex_time_max[ii] = mult_a_times.max()
    if len(tmp_sel2b) > 0:
        mult_b_times = tmp_sel2b["tval"]
        bs_after_mult = bs[bs > mult_b_times.min()][0]
        ex_time_b = mult_b_times[mult_b_times < bs_after_mult].max()
        ex_time[ii] = min(ex_time[ii], ex_time_b)
        ex_time_max[ii] = min(ex_time_max[ii], mult_b_times.max())
    if ~np.isinf(ex_time[ii]):
        ex_time_end[ii] = bs[bs > ex_time[ii]][0]
        ex_time_max_end[ii] = bs[bs > ex_time_max[ii]][0]

    pmult_filt[ii] = ex_time[ii] >= ibs

##Need to get time of the first exchange as well -- this is not quite ex_time
np.savez(f"pmult_before_bin_{my_ft}{flat_suff}{contig_suff}.npz", pmult_filt=pmult_filt, ex_time=ex_time, ex_time_max=ex_time_max,
         ex_time_end=ex_time_end, ex_time_max_end=ex_time_max_end)
#########################################################################################################
#Loading data -- Note different persistence filter was used for this file(!!!) Will have to "unify" the
#persistence filters.
npzs_list = [base_new + str(seed) + suff_new + f"/fates_corr{flat_suff}{contig_suff}.npz" for seed in seeds]
fates_corr = npz_stack(npzs_list)
same_sys_filt = fates_corr["same_sys_filt"]
end_states = fates_corr["end_states"]
bin_ids = my_data["bin_ids"]
#########################################################################################################
##Ionized binaries and encounters -- those that end up as single stars
bin_ids_11 = bin_ids[quasi_filter &  (end_states=="1 1")]
bin_ids_subset = bin_ids_11
norm_sep = np.zeros(len(bin_ids_subset))
mult_after_destruction = np.zeros(len(bin_ids_subset))
# bins_first_bound_subset = bins_first_bound[quasi_filter &  (end_states=="1 1")]
# bins_last_bound_subset = bins_last_bound[quasi_filter &  (end_states=="1 1")]

for idx, uid in tqdm.tqdm(enumerate(bin_ids_subset)):
    bin_list = list(uid)
    tmp_row = np.array(bin_list).astype(str)
    sys1_info = lookup_dict[bin_list[0]]
    sys2_info = lookup_dict[bin_list[1]]
    path_diff_all, path_diff_all_order = get_min_dist_binary(path_lookup, tmp_row, two_body)
    bin_sel = get_bound_snaps_adjust(bin_list, high_df)
    lb = int(bin_sel["tval"].iloc[-1])
    lsma = bin_sel["a"].iloc[-1]
    try:
        norm_sep[idx] = min(path_diff_all[lb], path_diff_all[lb + 1]) / (2 * lsma)
    except IndexError:
        breakpoint()
    try:
        mult1 = sys1_info[sys1_info[:,LOOKUP_SNAP]==lb+1][0, LOOKUP_MULT]
        mult2 = sys2_info[sys2_info[:,LOOKUP_SNAP]==lb+1][0, LOOKUP_MULT]
    except IndexError:
        breakpoint()

    mult1 = lookup_star_mult(high_df, bin_list[0], lb + 1, pre_filtered=False)
    mult2 = lookup_star_mult(high_df, bin_list[1], lb + 1, pre_filtered=False)
    mult_after_destruction[idx] = max(mult1[1], mult2[1])

norm_sep_og = np.copy(norm_sep)
##TO FIX: Not right filtering!
print(f"Frac in mult after destruction: {len(mult_after_destruction[mult_after_destruction > 1]) / len(mult_after_destruction)}")
#########################################################################################################
##Surviving binaries and encounters -- those that end up as single stars
bin_ids = my_data["bin_ids"]
##Checking if the stars are bound at the last snapshot both exist(!!)
final_bound_snaps_norm = bins_last_bound / my_data["end_stars"]
##May also filter out cases where "exchange" occurs after the initial formation -- but then we may be putting in the answer with our sample selection...
no_mult_before_bin = (pmult_filt) ##Since this will be looking at final binaries we can just check that ex_time is infinite(?)
##NOTE: Deliberately taking stricter 'survival' sample. Need the stars to remain in orbit of one another for the analysis
##to make sense.
bin_ids_surv = bin_ids[quasi_filter & (final_bound_snaps_norm==1) & (no_mult_before_bin)]
# bins_last_bound_subset = bins_first_bound[quasi_filter & (final_bound_snaps_norm==1) & (no_mult_before_bin)]
print(len(bin_ids_surv))
norm_sep = np.zeros(len(bin_ids_surv))
bin_ids_subset = bin_ids_surv

for idx, uid in enumerate(bin_ids_subset):
    bin_list = list(uid)
    tmp_row = np.array(bin_list).astype(str)
    bin_sel = get_bound_snaps_adjust(bin_list, high_df)
    path_diff_all, path_diff_all_order = get_min_dist_binary(path_lookup, tmp_row, two_body)
    ##Minimum distance for all surviving binaries
    norm_sep[idx] = np.min(path_diff_all[bin_sel["tval"].astype(int)] / (2 * bin_sel["a"]))
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
fig.savefig(f"fig5a_{two_body}.pdf")
np.savez(f"fig5_data_{two_body}.npz", norm_sep=norm_sep, norm_sep_og=norm_sep_og, bin_ids_surv=bin_ids_surv, bin_ids_11=bin_ids_11)
#########################################################################################################
final_pair_mass_no_halo = my_data["mfinal_pair"]

bins = np.arange(-1, 1.21, 0.2)
vd_b, b1, tmp1 = plt.hist(np.log10((final_pair_mass_no_halo[quasi_filter & ~(same_sys_filt)])), bins=bins,
                       histtype='step')
vs_b, b2, tmp2 = plt.hist(np.log10(final_pair_mass_no_halo[quasi_filter & (same_sys_filt)]), bins=bins,
                       histtype='step')

fig,ax = plt.subplots(figsize=(8,8), constrained_layout=True)
ax.set_ylabel("$N_{surv}$ / $N_{dis}$")
ax.set_xlabel("log($m_{pair, f}$ [$M_{\odot}$])")
plt.plot(0.5 * (b1[1:] + b1[:-1]), vs_b / vd_b, "s-")

fig.savefig("fig5b.pdf")
