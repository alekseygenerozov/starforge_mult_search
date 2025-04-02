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
import matplotlib.patches as mpatches

dummy_patch = mpatches.Patch(color='white', label='')

colorblind_palette = sns.color_palette("colorblind")
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import npz_stack
from starforge_mult_search.analysis import cgs_const as cgs
from starforge_mult_search.analysis.high_multiples_analysis import make_hier, get_pair_state, add_node_to_orbit_tab_streamlined

from starforge_mult_search.analysis.figures.figure_preamble import *
#########################################################################################################
end_states = fates_corr["end_states"]
same_sys_filt = fates_corr["same_sys_filt"]
quasi_filter = my_data["quasi_filter"]
#########################################################################################################
d1 = len(end_states[quasi_filter])
#########################################################################################################
##Tallying all the non-surviving states.
ns1 = len(end_states[(end_states=="1 1") & (quasi_filter) ]) / d1
ns2 = len(end_states[(end_states=="1 2") & (quasi_filter) ]) / d1
ns3 = len(end_states[(end_states=="1 3") & (quasi_filter) ]) / d1
ns4 = len(end_states[(end_states=="1 4") & (quasi_filter) ]) / d1
ns5 = []
ns5.append(len(end_states[(end_states=="2 2") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="2 3") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="2 4") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="3 3") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="3 4") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns5.append(len(end_states[(end_states=="4 4") &  ~(same_sys_filt) & (quasi_filter) ]) / d1)
ns = len(end_states[~(same_sys_filt) & (quasi_filter) ]) / d1


print(f"S S:{ns1:.4f} B S:{ns2:.4f} T S:{ns3:.4f} Q S:{ns4:.4f} M M:{np.sum(ns5):.4f} NS Tot: {ns1 + ns2 + ns3 + ns4 + np.sum(ns5):.4f}")
#########################################################################################################
##Tallying all the surviving states.
ss1 = len(end_states[(end_states=="2 2") & (quasi_filter) ]) / d1
ss2 = len(end_states[(end_states=="3 3") & (quasi_filter) ]) / d1
ss3 = len(end_states[(end_states=="4 4") & (quasi_filter) ]) / d1

print(f"B:{ss1:.4f} T:{ss2:.4f} Q:{ss3:.4f} S Tot:{ss1 + ss2 + ss3:.4f}")
# print(f"B:{ss1 * d1} T:{ss2 * d1} Q:{ss3 * d1}")

