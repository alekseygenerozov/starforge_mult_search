import matplotlib.ticker as mticker
import matplotlib.pyplot as plt

from starforge_mult_search.analysis.figures.figure_preamble import *
from starforge_mult_search.analysis.analyze_stack import max_w_infinite, subtract_path_1d

path_lookup_keys = path_lookup.keys()

mstars_final = [max_w_infinite(path_lookup[uu][:, mcol]) for uu in path_lookup_keys]
mhalos_max = [max_w_infinite(subtract_path_1d(path_lookup[uu][:, mtotcol], path_lookup[uu][:, mcol])) for uu in path_lookup_keys]

mhalos_max = np.array(mhalos_max)
mstars_final = np.array(mstars_final)
fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, gridspec_kw={'height_ratios': [1,5]})
ax1.set_ylim(1.01,3.01)
ax1.set_yticks([2,3])
ax2.set_ylim(-0.02,1.01)
ax1.set_xlim(-0.02, 2)
ax2.set_xlim(-0.02, 2)
ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))
ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))


ax1.spines.bottom.set_visible(False)
ax2.spines.top.set_visible(False)
ax1.xaxis.tick_top()
ax1.tick_params(labeltop=False)  # don't put tick labels at the top
ax2.xaxis.tick_bottom()

d = .5  # proportion of vertical to horizontal extent of the slanted line
kwargs = dict(marker=[(-1, -d), (1, d)], markersize=12,
              linestyle="none", color='k', mec='k', mew=1, clip_on=False)
ax1.plot([0, 1], [0, 0], transform=ax1.transAxes, **kwargs)
ax2.plot([0, 1], [1, 1], transform=ax2.transAxes, **kwargs)

ax2.set_ylabel('PDF/CDF')
ax2.set_xlabel("Max halo mass/Final star mass")
ax2.set_title(f"$f_t={my_ft}$")
fig.subplots_adjust(hspace=0.05)  # adjust space between Axes
# plot the same data on both Axes
# ax1.ecdf(mhalos_max / mstars_final, color=colorblind_palette[0])
print(np.median(mhalos_max / mstars_final)**-1.)
ax2.ecdf(mhalos_max / mstars_final, color=colorblind_palette[0])
ax1.hist(mhalos_max / mstars_final, histtype='step', bins=np.arange(0, 2.01, 0.05), linewidth=4,
         weights=[1 / len(mstars_final) * 10] * len(mstars_final),
        color=colorblind_palette[1])
ax2.hist(mhalos_max / mstars_final, histtype='step', bins=np.arange(0, 2.01, 0.05), linewidth=4,
         weights=[1 / len(mstars_final) * 10] * len(mstars_final),
         color=colorblind_palette[1])

ax2.annotate("CDF", (.7, 0.6), color=colorblind_palette[0])
ax2.annotate(r"$PDF\times Constant$", (10, 0.1), color=colorblind_palette[1], ha='right')

fig.savefig(f"ex_fig7a.pdf")