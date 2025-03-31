import ast
from collections import defaultdict
import copy
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from starforge_mult_search.analysis.plotting import annotate_multiple_ecdf
from scipy.stats import ks_2samp
import seaborn as sns

colorblind_palette = sns.color_palette("colorblind")
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import npz_stack
from starforge_mult_search.analysis.high_multiples_analysis import make_hier, get_pair_state, add_node_to_orbit_tab_streamlined
from starforge_mult_search.analysis import cgs_const as cgs

from starforge_mult_search.analysis.figures.figure_preamble import *
#########################################################################################################
lookup_dict_keys = lookup_dict.keys()
lookup_dict_keys = list(lookup_dict_keys)
first_mult = np.ones(len(lookup_dict_keys)) * np.inf
f1 = coll_full_df_life["frac_of_orbit"]
n1 = coll_full_df_life["nbound_snaps"]
##Make f1 >= 1 for consistency, but should not matter.
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

##Wide figure to accomodate the colorbar
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