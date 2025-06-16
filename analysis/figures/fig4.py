import ast
from collections import defaultdict
import copy
import os
import pickle
import tqdm

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from starforge_mult_search.analysis.plotting import annotate_multiple_ecdf
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.patches as mpatches

dummy_patch = mpatches.Patch(color='white', label='')

colorblind_palette = sns.color_palette("colorblind")
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import npz_stack
from starforge_mult_search.analysis import cgs_const as cgs
from starforge_mult_search.analysis.high_multiples_analysis import make_hier, get_pair_state, add_node_to_orbit_tab_streamlined
from starforge_mult_search.analysis.high_multiples_analysis import lookup_star_mult_with_mass


from starforge_mult_search.analysis.figures.figure_preamble import *



bin_ids = my_data["bin_ids"]
min_mass_b = np.zeros(len(bin_ids))
mthres = 0
high_df = coll_full_df_life
high_df = high_df.loc[(high_df[f"frac_of_orbit{contig_suff}"] >= 1) & (high_df[f"nbound_snaps{contig_suff}"] > 1)]

end_states = np.zeros(len(bin_ids)).astype(str)
same_sys_filt = np.zeros(len(bin_ids)).astype(bool)
for ii, row in tqdm.tqdm(enumerate(bin_ids)):
    bin_list = list(row)
    id1 = bin_list[0]
    id2 = bin_list[1]

    ##Want min mass of the pair before the binary is destroyed?
    end_time1 = lookup_dict[float(int(id1))][-1, 0]
    end_time2 = lookup_dict[float(int(id2))][-1, 0]
    end_time = int(min(end_time1, end_time2))
    ## Evaluating min mass at the end of its lifetime...
    look1 = lookup_dict[float(int(id1))]
    look2 = lookup_dict[float(int(id2))]
    end_bin_time = analyze_multiples_part2.get_bound_snaps(look1, look2)[0][-1][0]
    end_bin_time = int(end_bin_time)
    min_mass_b[ii] = min(path_lookup[str(id1)][end_bin_time, mcol], path_lookup[str(id2)][end_bin_time, mcol])

    tmp_df = high_df.xs(end_time, level="t")
    tmp_mult1 = lookup_star_mult_with_mass(tmp_df, id1, end_time, path_lookup, pre_filtered=True, contig_suff=contig_suff)
    tmp_mult2 = lookup_star_mult_with_mass(tmp_df, id2, end_time, path_lookup, pre_filtered=True, contig_suff=contig_suff)
    same_sys_filt[ii] = (tmp_mult1[0] == tmp_mult2[0])

    tmp_mult1 = tmp_mult1[2]
    tmp_mult2 = tmp_mult2[2]

    tmp_mult1 = len(tmp_mult1[tmp_mult1 > mthres])
    tmp_mult2 = len(tmp_mult2[tmp_mult2 > mthres])
    end_states[ii] = f"{min(tmp_mult1, tmp_mult2)} {max(tmp_mult1, tmp_mult2)}"

#########################################################################################################
end_states_ck = fates_corr["end_states"]
same_sys_filt_ck = fates_corr["same_sys_filt"]
quasi_filter = my_data[f"quasi_filter{contig_suff}"]

# assert(np.all(end_states_ck==end_states))
assert(np.all(same_sys_filt==same_sys_filt_ck))
##Need to apply completeness correction to higher multiples table to get a corrected endState??
#########################################################################################################
d1 = len(end_states[quasi_filter & (min_mass_b > mthres)])
#########################################################################################################
##Tallying all the non-surviving states.
ns1 = len(end_states[(end_states=="1 1") & (quasi_filter) & (min_mass_b > mthres) ]) / d1
ns2 = len(end_states[(end_states=="1 2") & (quasi_filter) & (min_mass_b > mthres) ]) / d1
ns3 = len(end_states[(end_states=="1 3") & (quasi_filter) & (min_mass_b > mthres) ]) / d1
ns4 = len(end_states[(end_states=="1 4") & (quasi_filter) & (min_mass_b > mthres) ]) / d1
ns5 = []
ns5.append(len(end_states[(end_states=="2 2") &  ~(same_sys_filt) & (quasi_filter) & (min_mass_b > mthres) ]) / d1)
ns5.append(len(end_states[(end_states=="2 3") &  ~(same_sys_filt) & (quasi_filter) & (min_mass_b > mthres) ]) / d1)
ns5.append(len(end_states[(end_states=="2 4") &  ~(same_sys_filt) & (quasi_filter) & (min_mass_b > mthres) ]) / d1)
ns5.append(len(end_states[(end_states=="3 3") &  ~(same_sys_filt) & (quasi_filter) & (min_mass_b > mthres) ]) / d1)
ns5.append(len(end_states[(end_states=="3 4") &  ~(same_sys_filt) & (quasi_filter) & (min_mass_b > mthres) ]) / d1)
ns5.append(len(end_states[(end_states=="4 4") &  ~(same_sys_filt) & (quasi_filter) & (min_mass_b > mthres) ]) / d1)
ns = len(end_states[~(same_sys_filt) & (quasi_filter) & (min_mass_b > mthres) ]) / d1


print(f"S S:{ns1:.4f} B S:{ns2:.4f} T S:{ns3:.4f} Q S:{ns4:.4f} M M:{np.sum(ns5):.4f} NS Tot: {ns1 + ns2 + ns3 + ns4 + np.sum(ns5):.4f}")
#########################################################################################################
##Tallying all the surviving states.
ss1 = len(end_states[(end_states=="2 2") & (quasi_filter) & (min_mass_b > mthres) ]) / d1
ss2 = len(end_states[(end_states=="3 3") & (quasi_filter) & (min_mass_b > mthres) ]) / d1
ss3 = len(end_states[(end_states=="4 4") & (quasi_filter) & (min_mass_b > mthres) ]) / d1

print(f"B:{ss1:.4f} T:{ss2:.4f} Q:{ss3:.4f} S Tot:{ss1 + ss2 + ss3:.4f}")
# print(f"B:{ss1 * d1} T:{ss2 * d1} Q:{ss3 * d1}")

