import numpy as np
from matplotlib import pyplot as plt

from starforge_mult_search.analysis.figures.figure_preamble import *
from starforge_mult_search.analysis.high_multiples_analysis import (
    apply_persistence_filter,
    get_maximal_multiples,
)
from starforge_mult_search.analysis.sink_cols import *

high_df = coll_full_df_life
high_df_filt = apply_persistence_filter
high_df_filt_max = get_maximal_multiples(high_df)
end_masses = []
end_mults = []
##Get full path of each star and get the final mass
##INCLUDES STARS DESTROYED BY SNE BEFORE THE END OF THE SIMULATION...
for kk in path_lookup.keys():
    tmp_path = path_lookup[kk]
    tmp_path = tmp_path[~np.isinf(tmp_path[:, 0])]
    end_masses.append(tmp_path[-1, mcol])
    end_time = tmp_path[-1, 0]
    tmp_high_df = high_df_filt_max.xs(end_time, level="t")
    end_mult = 1
    if int(kk) in tmp_high_df.index:
        end_mult = tmp_high_df.loc[int(kk)]["mult"]
    end_mults.append(end_mult)


end_masses_sorted = sorted(end_masses)
masses_cumul = np.cumsum(end_masses_sorted)

fig, ax = plt.subplots()
ax.set_xlabel("log[M / $M_{\odot}$]")
ax.set_ylabel("Fraction of total mass")
ax.plot(np.log10(end_masses_sorted), masses_cumul / masses_cumul[-1])
fig.savefig("mass_dist.pdf")

from scipy.interpolate import interp1d

print(interp1d(masses_cumul / masses_cumul[-1], end_masses_sorted)(0.5))


end_masses = np.array(end_masses)
end_mults = np.array(end_mults)
for ii in range(1, 5):
    print(np.sum(end_masses[end_mults == ii]))
