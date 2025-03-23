from collections import defaultdict
import os
import tqdm

import matplotlib.pyplot as plt
import numpy as np
import pickle
import pandas as pd
from sci_analysis.plotting import annotate_multiple_ecdf
from scipy.stats import ks_2samp
import seaborn as sns

colorblind_palette = sns.color_palette("colorblind")

import cgs_const as cgs
from labelLine import labelLines
from power_fit import fit_power
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import npz_stack
from starforge_mult_search.analysis.high_multiples_analysis import make_hier, get_mult, get_pair_state, add_node_to_orbit_tab_streamlined

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_SMA = 6
LOOKUP_ECC = 7
LOOKUP_Q = 8
snap_interval = 2.47e4

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

#########################################################################################################

##Stacking data
my_ft = 1.0
my_tides = False
suff = "_mult"

base_new = "../analysis_pipeline_experiment/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_"
seeds = (1, 2, 42)
suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
npzs_list = []
npzs_list = [base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz" for seed in seeds]

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

bin_ids = my_data["bin_ids"]
mprims = np.zeros(len(bin_ids))
for ii, row in enumerate(bin_ids):
    bin_list = list(row)
    tmp_m1 = path_lookup[f"{bin_list[0]}"][:, mcol]
    tmp_m2 = path_lookup[f"{bin_list[1]}"][:, mcol]
    mprims[ii] = max(tmp_m1[(~np.isinf(tmp_m1)) & (~np.isinf(tmp_m2))][-1],
                     tmp_m2[(~np.isinf(tmp_m1)) & (~np.isinf(tmp_m2))][-1])

fig, ax = plt.subplots()
ax.set_ylabel("CDF")
ax.set_xlabel("log(-PE/ KE)")

mults_filt = my_data["mults_filt"]
ax.plot([0, 0], [0, 1], "--", color="0.5")
ax.arrow(0, 0.5, -1, 0, width=0.02, color='0.5')
ax.annotate("Unbound", [-2, 0.55], color='0.5')

higher_mult_filt = (my_data["same_sys_at_fst"].astype(bool)) & (my_data["init_bound_snaps"] != my_data["fst"])
plt.ecdf(np.log10(my_data["ens"][(my_data["quasi_filter"])]), label="No gas")
plt.ecdf(np.log10(my_data["ens_gas"][my_data["quasi_filter"]]), label="Gas")
plt.ecdf(np.log10(my_data["ens_gas"][(my_data["quasi_filter"]) & (~higher_mult_filt)]), label="Gas")

ax.set_title(r"$f_t=$" + f"{my_ft}")
ax.legend()

#########################################################################################################

bin_ids = my_data["bin_ids"]
quasi_filter = my_data["quasi_filter"]
bin_ids_quasi_list = np.array([list(row) for row in bin_ids[quasi_filter]]).ravel()

#########################################################################################################

star_ids = np.array(list(path_lookup.keys()))
star_mult_label_final = np.ones(len(star_ids))
star_final_mass = np.ones(len(star_ids)) * np.inf

##Much of this could be refactored into its own function
##SNe 'snapshot' problem again...Here SNe are excluded.
for ii,star_id in tqdm.tqdm(enumerate(star_ids)):
    star_times = path_lookup[star_id][:,0]
    star_masses = path_lookup[star_id][:, mcol]

    star_times = star_times[~np.isinf(star_times)]
    star_masses = star_masses[~np.isinf(star_masses)]
    star_final_mass[ii] = star_masses[-1]
    star_end_snap = int(star_times[-1])
    if star_end_snap < len(path_lookup[star_id][:,0]) - 1:
        star_mult_label_final[ii] = -1
        continue

    tmp_sel = coll_full_df_life.xs(star_end_snap, level="t")
    tmp_sel = tmp_sel.loc[(tmp_sel["nbound_snaps"]>1) & (tmp_sel["frac_of_orbit"] >= 1)]
    star_in_mult = tmp_sel.index.get_level_values("id").str.contains(rf"\b{star_id}\b")
    ##May fail if we an issue with substring...
    mults_with_star = tmp_sel.loc[star_in_mult]
    if len(mults_with_star)==0:
        continue
    tmp_mults = mults_with_star.groupby("id", sort=False).apply(lambda x: get_mult(x.name)).values
    star_mult_label_final[ii] = max(tmp_mults)

single_filter = (star_mult_label_final==1)
single_final_masses = star_final_mass[single_filter]
from_bins_filt = np.isin(star_ids[single_filter].astype(int), bin_ids_quasi_list)
print(f"Number of final singles: {len(single_final_masses)}")
print(f"Number of final singles from bins: {len(single_final_masses[from_bins_filt])}")
print(f"Frac from bin: {len(single_final_masses[from_bins_filt]) / len(single_final_masses)}")
print(f"Frac from bin (ms > 1 Msun): {len(single_final_masses[(from_bins_filt) & (single_final_masses > 1)]) / len(single_final_masses[(single_final_masses > 1)])}")
##Get the number that were in *persistent multiples*
coll_full_df_life = pd.concat([pd.read_parquet(base_new + str(seed) + suff_new + f"/mults_flat.pq") for seed in seeds])
f1 = coll_full_df_life["frac_of_orbit"]
n1 = coll_full_df_life["nbound_snaps"]
def parse_mult_id(id_str):
    return list(map(int, id_str.replace("[", "").replace("]", "").split(",")))

##Make f1 >= 1 for consistency, but should not matter.
tmp_sel = coll_full_df_life.loc[(f1>=1) & (n1>1)]
mult_ids = tmp_sel.index.get_level_values("id")
mult_ids_set = mult_ids.to_series().apply(parse_mult_id)
single_star_in_mult = []

mult_ids_set = np.unique(np.concatenate(mult_ids_set.tolist()))

for star_id in tqdm.tqdm(star_ids[single_filter]):
    single_star_in_mult.append(int(star_id) in mult_ids_set)
single_star_in_mult = np.array(single_star_in_mult).astype(bool)
print(f"Frac from mult: {len(single_final_masses[single_star_in_mult]) / len(single_final_masses)}")
print(f"Frac from mult (ms > 1 Msun): {len(single_final_masses[single_star_in_mult & (single_final_masses > 1)]) / len(single_final_masses[single_final_masses > 1])}")


fig,ax = plt.subplots()
ax.set_title(r"Singles Final MF")
ax.set_yscale("log")
ax.set_ylabel("PDF")
ax.set_xlabel("Log(Mass [$M_{\odot}$])")
bsize = 0.1

ax.hist(np.log10(single_final_masses), histtype='step', density=True, bins=np.arange(-2, 1.2, bsize), label="All", linewidth=4)
ax.hist(np.log10(single_final_masses[~single_star_in_mult]), histtype='step', density=True, bins=np.arange(-2, 1.2, bsize), label="Always single", linewidth=2.5)
# ax.hist(np.log10(sings[~tmp_filt][:, LOOKUP_MTOT]), histtype='step', bins=np.arange(-2, 1.2, bsize), label="Not from bins", density=True)
ax.hist(np.log10(single_final_masses[from_bins_filt]), histtype='step', bins=np.arange(-2, 1.2, bsize), label="From binaries", density=True,
       linewidth=2.5)
ax.hist(np.log10(single_final_masses[(single_star_in_mult) & ~(from_bins_filt)]), histtype='step', bins=np.arange(-2, 1.2, bsize), label="From higher\nmultiples", density=True,
       linewidth=2.5, linestyle="--")

ax.legend()

absc = np.geomspace(0.3, 10, 500)

def lighten_color(color, factor=0.5):
    """Lightens the given color by blending it with white."""
    return tuple(1 - factor * (1 - c) for c in color)

f1 = fit_power(single_final_masses[single_final_masses > 0.3], 1.1)[0]
f2 = fit_power(single_final_masses[from_bins_filt & (single_final_masses > 0.3)], 1.1)[0]
print(f"Power law fits {f1} {f2}")

fit_colors = [lighten_color(c, factor=0.7) for c in colorblind_palette]
l1,=ax.plot(np.log10(absc), 0.9 * (absc / 0.3)**(-f1 + 1), color=fit_colors[0], linestyle="--", label=f"$dN/dm \\propto m^{{-{f1:.2f}}}$")
l2,=ax.plot(np.log10(absc), 0.9 * (absc / 0.3)**(-f2 + 1), color=fit_colors[2], linestyle="--", label=f"$dN/dm \\propto m^{{-{f2:.2f}}}$")
labelLines([l1], fontsize=16, xvals=(0.1,), ha='right', va='top', ang=-55, y_offset=-0.16)
labelLines([l2], fontsize=16, xvals=(-0.1,), ang=-45, y_offset=0.22)

fig.savefig("single_mf.pdf")