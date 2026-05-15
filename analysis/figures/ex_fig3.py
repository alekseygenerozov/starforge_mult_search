import matplotlib.pyplot as plt
import numpy as np

from starforge_mult_search.analysis.analyze_stack import make_binned_data
from starforge_mult_search.analysis.figures.figure_preamble import *

# mpl.rcParams["figure.figsize"] = (3.3, 3.3)
# mpl.rcParams["ps.fonttype"] = 42
# plt.style.use("nature")
# mpl.rcParams["font.sans-serif"] = "Arial"

##Mass bins to use.
bins = np.linspace(-1, 2, 6)
bins_center = 0.5 * (bins[1:] + bins[:-1])
same_sys_at_ist = my_data["same_sys_at_fst"]
quasi_filter = my_data[f"quasi_filter{contig_suff}"]

tmp_filt_part1 = (quasi_filter) & (same_sys_filt)
absc, ords = (
    np.log10(my_data["mfinal_primary"][tmp_filt_part1]),
    same_sys_at_ist.astype(int)[tmp_filt_part1],
)
n1, n1u, d1, te1 = make_binned_data(absc, ords, bins)

tmp_filt_part1 = (quasi_filter) & ~(same_sys_filt)
absc, ords = (
    np.log10(my_data["mfinal_primary"][tmp_filt_part1]),
    same_sys_at_ist.astype(int)[tmp_filt_part1],
)
n2, n2u, d2, te2 = make_binned_data(absc, ords, bins)

tmp_filt_part1 = quasi_filter
absc, ords = (
    np.log10(my_data["mfinal_primary"][tmp_filt_part1]),
    same_sys_at_ist.astype(int)[tmp_filt_part1],
)
n3, n3u, d3, te3 = make_binned_data(absc, ords, bins)

from labelLine import labelLines

fig, ax = plt.subplots(figsize=(16, 16), constrained_layout=True)
ax.set_xlim(-0.8, 1.8)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r"log($M_{prim, f} / M_{\odot}$)")
ax.set_ylabel("BFB Fraction")

ax.errorbar(
    bins_center,
    n1 / d1,
    yerr=te1,
    marker="o",
    linestyle="",
    alpha=0.7,
    label="Survivors",
    markerfacecolor="none",
    markeredgecolor="C0",
    markeredgewidth=2,
)
# for bin_idx in range(len(n1)):
#     ax.text(max(bins_center[bin_idx], -0.6), n1[bin_idx] / d1[bin_idx] + 0.05, f"{int(n1[bin_idx])}/{int(d1[bin_idx])}", fontsize=5, ha="center", color=colorblind_palette[0])


ax.errorbar(
    bins_center,
    n2 / d2,
    yerr=te2,
    marker="v",
    linestyle="",
    alpha=0.7,
    label="Non-survivors",
    markersize=15,
)
# for bin_idx in range(len(n1)):
#     ax.text(max(bins_center[bin_idx], -0.6), n2[bin_idx] / d2[bin_idx] - 0.07, f"{int(n2[bin_idx])}/{int(d2[bin_idx])}", fontsize=5, ha="center", color=colorblind_palette[1])


print(n1, n2)
print(d1, d2)
ax.legend(loc="lower left")
fig.savefig("ex_fig3.pdf")

fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax.set_xlim(-0.8, 1.8)
ax.set_ylim(0, 1.0)
ax.set_xlabel(r"log($m_{max, f} / M_{\odot}$)")
ax.set_ylabel("BFB Fraction")

ax.errorbar(bins_center, n3 / d3, yerr=te3, marker="s", linestyle="", alpha=0.7)
print(n3)

print(d3)
# ax.legend(title=r"$f_t=$"+f"{my_ft}")
fig.savefig("fig2b.pdf")
