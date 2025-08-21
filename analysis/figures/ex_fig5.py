from collections import defaultdict
import os
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from starforge_mult_search.analysis.plotting import annotate_multiple_ecdf
from scipy.stats import ks_2samp
import seaborn as sns

colorblind_palette = sns.color_palette("colorblind")

from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import npz_stack
from starforge_mult_search.analysis.high_multiples_analysis import make_hier, get_mult, get_pair_state, add_node_to_orbit_tab_streamlined
from starforge_mult_search.analysis.high_multiples_analysis import lookup_star_mult_b, filter_maximal_sets
from starforge_mult_search.analysis.labelLine import labelLines
from starforge_mult_search.analysis import cgs_const as cgs
from starforge_mult_search.analysis.power_fit import fit_power
from starforge_mult_search.analysis.figures.figure_preamble import *

#########################################################################################################
bin_ids = my_data["bin_ids"]
quasi_filter = my_data[f"quasi_filter{contig_suff}"]
bin_ids_quasi_list = np.array([list(row) for row in bin_ids[quasi_filter]]).ravel()
#########################################################################################################
star_ids = np.array(list(path_lookup.keys()))
star_mult_label_final = np.ones(len(star_ids))
star_final_mass = np.ones(len(star_ids)) * np.inf
star_form_time = np.ones(len(star_ids)) * np.inf

##Filtering confusion
##Much of this could be refactored into its own function
##SNe 'snapshot' problem again...Here SNe are excluded.
for ii,star_id in tqdm.tqdm(enumerate(star_ids)):
    star_times = path_lookup[star_id][:,0]
    star_masses = path_lookup[star_id][:, mcol]

    star_times = star_times[~np.isinf(star_times)]
    star_masses = star_masses[~np.isinf(star_masses)]
    star_final_mass[ii] = star_masses[-1]
    star_end_snap = int(star_times[-1])
    star_form_time[ii] = star_times[0]

    tmp_sel = coll_full_df_life.xs(star_end_snap, level="t")
    tmp_sel = tmp_sel.loc[(tmp_sel[f"nbound_snaps{contig_suff}"]>1) & (tmp_sel[f"frac_of_orbit{contig_suff}"] >= 1)]
    star_in_mult = tmp_sel.index.get_level_values("id").str.contains(rf"\b{star_id}\b")
    mults_with_star = tmp_sel.loc[star_in_mult]
    if len(mults_with_star)==0:
        continue
    tmp_mults = mults_with_star.groupby("id", sort=False).apply(lambda x: get_mult(x.name)).values
    star_mult_label_final[ii] = max(tmp_mults)
#########################################################################################################
single_filter = (star_mult_label_final==1)
np.savez("single_filter.npz", np.transpose((star_ids, single_filter)))
single_final_masses = star_final_mass[single_filter]
all_masses = star_final_mass#[star_mult_label_final!=-1]
from_bins_filt = np.isin(star_ids[single_filter].astype(int), bin_ids_quasi_list)
print(f"Number of final singles: {len(single_final_masses)}")
# print(f"Number of final singles from bins: {len(single_final_masses[from_bins_filt])}")
# print(f"Frac from bin: {len(single_final_masses[from_bins_filt]) / len(single_final_masses)}")
# print(f"Frac from bin (ms > 1 Msun): {len(single_final_masses[(from_bins_filt) & (single_final_masses > 1)]) / len(single_final_masses[(single_final_masses > 1)])}")
##Get the number that were in *persistent multiples*
#########################################################################################################
f1 = coll_full_df_life[f"frac_of_orbit{contig_suff}"]
n1 = coll_full_df_life[f"nbound_snaps{contig_suff}"]
def parse_mult_id(id_str):
    return list(map(int, id_str.replace("[", "").replace("]", "").split(",")))

##Make f1 >= 1 for consistency, but should not matter.
tmp_sel = coll_full_df_life.loc[(f1>=1) & (n1>1)]
tmp_sel["tval"] = tmp_sel.index.get_level_values("t")
##Filter -- only get maximal multiple(!)
mult_ids = tmp_sel.index.get_level_values("id")
mult_ids_set = mult_ids.to_series().apply(parse_mult_id)

single_star_in_mult = []
single_star_in_soft_mult = []
mult_ids_set = np.unique(np.concatenate(mult_ids_set.tolist()))
for star_id in tqdm.tqdm(star_ids[single_filter]):
    single_star_in_mult.append(int(star_id) in mult_ids_set)

# min_soft_ratio_by_id = tmp_sel.groupby("id")["soft_ratio"].min()
# tmp_sel_soft = tmp_sel.loc[min_soft_ratio_by_id[min_soft_ratio_by_id <= 2].index]
# mult_ids = tmp_sel_soft.index.get_level_values("id")
# mult_ids_set = mult_ids.to_series().apply(parse_mult_id)
# for star_id in tqdm.tqdm(star_ids[single_filter]):
#     single_star_in_soft_mult.append(int(star_id) in mult_ids_set)
# max_multiples_only = []
# single_star_in_iso_bin = []
# single_star_in_higher = []
#########################################################################################################
##Getting table without subsystems
# times_all = tmp_sel.index.get_level_values("t").unique()
# for tt in tqdm.tqdm(times_all):
#     tmp_slice = tmp_sel.xs(tt, level="t")
#     ##Could refactor -- get sets for the full frame and then we can simplify this [Priority: Low...]
#     tmp_mult_ids_set = tmp_slice.index.get_level_values("id").to_series().apply(parse_mult_id)
#     tmp_filt = filter_maximal_sets(tmp_mult_ids_set)
#     tmp_slice.loc[tmp_filt]
#     max_multiples_only.append(tmp_slice)
# max_multiples_only = pd.concat(max_multiples_only)
# #########################################################################################################
# mult_ids_iso_bins = max_multiples_only.loc[max_multiples_only["mult"]==2]
# mult_ids_set_iso_bins = mult_ids_iso_bins.index.get_level_values("id").to_series().apply(parse_mult_id)
# mult_ids_set_iso_bins = np.unique(np.concatenate(mult_ids_set_iso_bins.tolist()))
# #########################################################################################################
# #########################################################################################################
# mult_ids_higher = max_multiples_only.loc[max_multiples_only["mult"] > 2]
# mult_ids_set_higher = mult_ids_higher.index.get_level_values("id").to_series().apply(parse_mult_id)
# mult_ids_set_higher = np.unique(np.concatenate(mult_ids_set_higher.tolist()))
# #########################################################################################################
# for star_id in tqdm.tqdm(star_ids[single_filter]):
#     single_star_in_iso_bin.append(int(star_id) in mult_ids_set_iso_bins)
#     single_star_in_higher.append(int(star_id) in mult_ids_set_higher)
# max_ids = max_multiples_only.index.get_level_values("id").str
# max_ids_times = max_multiples_only["tval"].to_numpy()
# #########################################################################################################
# last_mults = np.zeros(len(star_ids[single_filter]))
# for ii,star_id in tqdm.tqdm(enumerate(star_ids[single_filter])):
#     tmp_filt = max_ids.contains(rf"\b{star_id}\b")
#     tmp_slice = max_multiples_only.loc[tmp_filt]
#     if len(tmp_slice) > 0:
#         last_mults[ii] = tmp_slice["mult"].iloc[-1]
# #########################################################################################################
# first_mults = np.ones(len(star_ids[single_filter]))
# star_form_time_single = star_form_time[single_filter]
# for ii,star_id in tqdm.tqdm(enumerate(star_ids[single_filter])):
#     tmp_max_multiples_only_slice = max_multiples_only.loc[(max_ids_times >= star_form_time_single[ii]) & (max_ids_times < star_form_time_single[ii] + 20)]
#     tmp_max_ids = tmp_max_multiples_only_slice.index.get_level_values("id").str
#     tmp_filt = tmp_max_ids.contains(rf"\b{star_id}\b")
#     tmp_slice = tmp_max_multiples_only_slice.loc[tmp_filt]
#     if len(tmp_slice) > 0:
#         first_mults[ii] = tmp_slice["mult"].iloc[0]
##For 1st snapshot we can do a similar loop but filter slice to be within 5e5 yr of appearance of the star...
#########################################################################################################
single_star_in_mult = np.array(single_star_in_mult).astype(bool)
single_star_in_soft_mult = np.array(single_star_in_mult).astype(bool)
# np.savez("single_in_mult.npz", np.transpose((star_ids[single_filter], single_star_in_mult, first_mults, last_mults)))

print(f"Frac from mult: {len(single_final_masses[single_star_in_mult]) / len(single_final_masses)}")
print(f"Frac from mult (ms > 1 Msun): {len(single_final_masses[single_star_in_mult & (single_final_masses > 1)]) / len(single_final_masses[single_final_masses > 1])}")

# print(f"Frac from soft mult: {len(single_final_masses[single_star_in_soft_mult]) / len(single_final_masses)}")
# print(f"Frac from soft mult (ms > 1 Msun): {len(single_final_masses[single_star_in_soft_mult & (single_final_masses > 1)]) / len(single_final_masses[single_final_masses > 1])}")
#########################################################################################################
fig,ax = plt.subplots()
# ax.set_title(r"Singles Final MF")
ax.set_yscale("log")
ax.set_ylabel("PDF")
ax.set_xlabel("Log(Mass [$M_{\odot}$])")
bsize = 0.1
bins = np.arange(-2, 1.81, bsize)
ax.annotate(r"IMF", xy=(0.01, 0.99), xycoords="axes fraction", va="top", ha="left", fontsize=7)

ax.hist(np.log10(all_masses), histtype='step', density=True, bins=bins, label="All stars", linewidth=1, color="0.5")
ax.hist(np.log10(single_final_masses), histtype='step', density=True, bins=bins, label="All singles", linewidth=1)
ax.hist(np.log10(single_final_masses[~single_star_in_mult]), histtype='step', density=True, bins=bins, label="Always single", linewidth=2.5/4)
ax.hist(np.log10(single_final_masses[from_bins_filt]), histtype='step', bins=bins, label="From binaries", density=True,
       linewidth=2.5/4)
ax.hist(np.log10(single_final_masses[(single_star_in_mult) & ~(from_bins_filt)]), bins=bins, label="From higher\nmultiples", density=True,
       linewidth=2.5/4, linestyle="-.", alpha=0.2)

ax.legend(fontsize=5)# ,loc="upper right" , bbox_to_anchor=(0.6, 0.35))
absc = np.geomspace(0.3, 10**1.8, 500)

def lighten_color(color, factor=0.5):
    """Lightens the given color by blending it with white."""
    return tuple(1 - factor * (1 - c) for c in color)

f0 = fit_power(all_masses[all_masses > 0.3], 1.1)[0]
f1 = fit_power(single_final_masses[single_final_masses > 0.3], 1.1)[0]
f2 = fit_power(single_final_masses[from_bins_filt & (single_final_masses > 0.3)], 1.1)[0]
print(f"Power law fits {f1} {f2}")

fit_colors = [lighten_color(c, factor=0.7) for c in colorblind_palette]
# l0,=ax.plot(np.log10(absc), 0.9 * (absc / 0.3)**(-f0 + 1), color="0.5", linestyle="--", label=f"$dN/dm \\propto m^{{-{f0:.2f}}}$")
l1,=ax.plot(np.log10(absc), 0.9 * (absc / 0.3)**(-f1 + 1), color=fit_colors[0], linestyle="--", label=f"$dN/dm \\propto m^{{-{f1:.2f}}}$")
l2,=ax.plot(np.log10(absc), 0.9 * (absc / 0.3)**(-f2 + 1), color=fit_colors[2], linestyle="--", label=f"$dN/dm \\propto m^{{-{f2:.2f}}}$")
# labelLines([l0], fontsize=16, xvals=(0.25,), ha='left', va='top', ang=0, y_offset=0.22, align=False)
labelLines([l1], fontsize=6, xvals=(0.1,), ha='right', va='top', ang=-55, y_offset=-0.3)
labelLines([l2], fontsize=6, xvals=(np.log10(0.3),), ang=0, y_offset=0.15, align=False)

fig.savefig("ex_fig5.pdf")