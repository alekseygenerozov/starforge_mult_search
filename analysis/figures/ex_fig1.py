import ast
from collections import defaultdict
import copy
import pickle
import os

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sci_analysis.plotting import annotate_multiple_ecdf
from scipy.stats import ks_2samp
import seaborn as sns
import matplotlib.patches as mpatches

dummy_patch = mpatches.Patch(color='white', label='')

from IPython.core.debugger import set_trace

colorblind_palette = sns.color_palette("colorblind")
import cgs_const as cgs
from starforge_mult_search.analysis import analyze_multiples_part2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import npz_stack
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


print(f"S S:{ns1} B S:{ns2} T S:{ns3} Q S:{ns4} M M:{np.sum(ns5)} NS Tot: {ns1 + ns2 + ns3 + ns4 + np.sum(ns5)} NS Tot CK: {ns}")
print((ns) * d1)
#########################################################################################################
##Tallying all the surviving states.
ss1 = len(end_states[(end_states=="2 2") & (quasi_filter) ]) / d1
ss2 = len(end_states[(end_states=="3 3") & (quasi_filter) ]) / d1
ss3 = len(end_states[(end_states=="4 4") & (quasi_filter) ]) / d1

print(f"B:{ss1} T:{ss2} Q:{ss3} S Tot:{ss1 + ss2 + ss3}")
print(f"B:{ss1 * d1} T:{ss2 * d1} Q:{ss3 * d1}")
print((ss1 + ss2 + ss3) * d1)

##Also save seed info here--will be useful for the table...
# np.savez(suff_new.replace("/", "") + "_fates_corr.npz", end_states=end_states, same_sys_filt=same_sys_filt)

##Good place to put the book-keeping about single stars -- Already done in ex_fig8.py
