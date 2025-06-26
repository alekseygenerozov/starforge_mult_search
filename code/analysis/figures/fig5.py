import pickle
import tqdm

import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt
import seaborn as sns

colorblind_palette = sns.color_palette("colorblind")

from analysis.analyze_stack import npz_stack,subtract_path,max_w_infinite,get_min_dist_binary
from analysis import analyze_multiples_part2
from analysis.high_multiples_analysis import lookup_star_mult, parse_mult_id
from analysis.labelLine import labelLines

##Try to get rid of this import...
from analysis import plotting

from analysis.figures.figure_preamble import *
#########################################################################################################
#########################################################################################################
##Loading data
npzs_list = [base_new + str(seed) + suff_new + f"/fates_corr.npz" for seed in seeds]
fates_corr = npz_stack(npzs_list)
same_sys_filt = fates_corr["same_sys_filt"]
quasi_filter = my_data["quasi_filter"]
#########################################################################################################
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

fig.savefig("fig5.pdf")
