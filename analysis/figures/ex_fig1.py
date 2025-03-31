import numpy as np
import matplotlib.pyplot as plt

from starforge_mult_search.analysis.analyze_stack import make_binned_data
from starforge_mult_search.analysis.figures.figure_preamble import *

##Mass bins to use.
bins = np.linspace(-1, 2, 6)
bins_center = 0.5 * (bins[1:] + bins[:-1])
same_sys_at_ist = my_data["same_sys_at_fst"]

tmp_filt_part1 = (my_data["quasi_filter"]) & (same_sys_filt)
absc, ords = np.log10(my_data["mfinal_primary"][tmp_filt_part1]), same_sys_at_ist.astype(int)[tmp_filt_part1]
n1, n1u, d1 = make_binned_data(absc, ords, bins)

tmp_filt_part1 = (my_data["quasi_filter"]) & ~(same_sys_filt)
absc, ords = np.log10(my_data["mfinal_primary"][tmp_filt_part1]), same_sys_at_ist.astype(int)[tmp_filt_part1]
n2, n2u, d2 = make_binned_data(absc, ords, bins)

tmp_filt_part1 = (my_data["quasi_filter"])
absc, ords = np.log10(my_data["mfinal_primary"][tmp_filt_part1]), same_sys_at_ist.astype(int)[tmp_filt_part1]
n3, n3u, d3 = make_binned_data(absc, ords, bins)

from labelLine import labelLines

fig, ax = plt.subplots()
ax.set_xlim(-0.8, 1.8)
ax.set_ylim(0, 1.)
ax.set_xlabel(r"$log(M_{prim, f} / M_{\odot})$")
ax.set_ylabel("BFB Fraction")

ax.errorbar(bins_center, n1 / d1, \
            yerr=n1u / d1, marker="s", linestyle="", alpha=0.7, label="Survivors")
ax.errorbar(bins_center, n2 / d2, \
            yerr=n2u / d2, marker="s", linestyle="", alpha=0.7, label="Non-survivors")
fig.savefig("ex_fig1.pdf")

fig, ax = plt.subplots()
ax.set_xlim(-0.8, 1.8)
ax.set_ylim(0, 1.)
ax.set_xlabel(r"$log(M_{prim, f} / M_{\odot})$")
ax.set_ylabel("BFB Fraction")

ax.errorbar(bins_center, n3 / d3, \
            yerr=n3u / d3, marker="s", linestyle="", alpha=0.7, label="Survivors")

print(n3 /d3)
# ax.legend(title=r"$f_t=$"+f"{my_ft}")
ax.legend(loc="lower left")
fig.savefig("fig2b.pdf")