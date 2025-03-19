from labelLine import labelLines
from matplotlib.ticker import LogFormatterSciNotation
from matplotlib.ticker import FuncFormatter

import matplotlib.pyplot as plt

from starforge_mult_search.analysis.figures.figure_preamble import *

import cgs_const as cgs

bin_ids = my_data["bin_ids"]
fst = my_data["fst"]
# bin_ids_example = [{3920731, 13654613}, {13245844, 19648925}, {7647001, 9938318}, {12261108, 12102006},
#                    {5312318, 3832908}]
bin_ids_example = [{3920731, 13654613}, {13245844, 19648925}]
xlims = [(5.31, 5.8), (1.8, 5), None, None, None]
for bb, uid in enumerate(bin_ids_example):
    jj = np.where(bin_ids == uid)[0][0]

    cols = [np.array((129, 50, 168)) / 256, colorblind_palette[1]]
    alphas = [0.5, 0.5]

    fig1, ax0 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    fig2, ax1 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    fig3, ax2 = plt.subplots(figsize=(8, 8), constrained_layout=True)
    axs = (ax0, ax1, ax2)

    ax = axs[0]
    ax.annotate(f"Example {bb + 1}", xycoords="axes fraction", xy=(0.99, 0.99), va='top', ha='right')
    if not (xlims[bb] is None):
        ax.set_xlim(xlims[bb][0], xlims[bb][1])

    # ax.set_ylim(10, 1e4)
    ax.set_xlabel('t [Myr]')
    ax.set_ylabel('a [au]')

    ax = axs[1]
    if not (xlims[bb] is None):
        ax.set_xlim(xlims[bb][0], xlims[bb][1])
    ax.set_ylim(0, 1)
    ax.set_xlabel('t [Myr]')
    ax.set_ylabel('e')
    # ax.tick_params(axis="x", which="both", rotation=45)

    ax = axs[2]
    if not (xlims[bb] is None):
        ax.set_xlim(xlims[bb][0], xlims[bb][1])
    ax.set_ylim(0.04, 5)
    ax.set_xlabel('t [Myr]')
    ax.set_ylabel('m $[M_{\odot}]$')
    ax.plot([7e6, 7e6], [100, 100], 'k-', label="Sink\n+Gas Halo")
    ax.plot([7e6, 7e6], [100, 100], 'k--', label="Sink")

    ax.legend()

    ##Could clean this up(!!) -- e.g. by using the lookup table generated in captures_com_trajectory...
    test_id1, test_id2 = list(bin_ids[jj])
    # sys1_info = lookup[test_id1 == lookup[:, LOOKUP_PID].astype(int)]
    # sys2_info = lookup[test_id2 == lookup[:, LOOKUP_PID].astype(int)]
    sys1_info = lookup_dict[test_id1]
    sys2_info = lookup_dict[test_id2]
    sys1_tag = ["{0}_{1}".format(row[LOOKUP_SNAP], row[2]) for row in sys1_info]
    sys2_tag = ["{0}_{1}".format(row[LOOKUP_SNAP], row[2]) for row in sys2_info]
    ##NOT ENOUGH!! STARS COULD BE IN THE SAME MULTIPLE BUT NOT BOUND--HAVE TO DO FURTHER FILTERING BASED ON SMA
    same_sys_filt1 = np.in1d(sys1_tag, sys2_tag)
    same_sys_filt2 = np.in1d(sys2_tag, sys1_tag)
    sys1_info = sys1_info[same_sys_filt1]
    sys2_info = sys2_info[same_sys_filt2]
    ##Making the stars have the same sma -- implies they are bound to each other...
    bound_filt = sys1_info[:, LOOKUP_SMA] == sys2_info[:, LOOKUP_SMA]
    tmp_info = np.copy(sys1_info)
    to_replace = tmp_info[~bound_filt]
    tmp_info[~bound_filt] = np.ones(to_replace.shape) * np.inf
    axs[0].plot(tmp_info[:, LOOKUP_SNAP] * snap_interval / 1e6, tmp_info[:, LOOKUP_SMA] * cgs.pc / cgs.au, marker="s",
                color="brown", linewidth=5)
    axs[1].plot(tmp_info[:, LOOKUP_SNAP] * snap_interval / 1e6, tmp_info[:, LOOKUP_ECC], marker="s", color="brown",
                linewidth=5)

    ls_mass = []
    for kk in (0, 1):
        test_id = list(bin_ids[jj])[kk]
        tmp_sys_idx = lookup_dict[test_id]

        l1, = axs[0].semilogy(
            [tmp_sys_idx[1, LOOKUP_SNAP] * snap_interval / 1e6, tmp_sys_idx[-2, LOOKUP_SNAP] * snap_interval / 1e6,
             tmp_sys_idx[-1, LOOKUP_SNAP] * snap_interval / 1e6], [20, 20, 20], color='0.5', label="Softening")
        clean_filt = (tmp_sys_idx[:, LOOKUP_SNAP] >= fst[jj])

        axs[0].semilogy(tmp_sys_idx[:, LOOKUP_SNAP][clean_filt] * snap_interval / 1e6,
                        tmp_sys_idx[:, LOOKUP_SMA][clean_filt] * cgs.pc / cgs.au, "-",
                        color=cols[kk], alpha=alphas[kk], linewidth=2)
        axs[1].plot(tmp_sys_idx[:, LOOKUP_SNAP][clean_filt] * snap_interval / 1e6,
                    tmp_sys_idx[:, LOOKUP_ECC][clean_filt],
                    color=cols[kk], alpha=alphas[kk], linewidth=2)
        tmp_line, = axs[2].semilogy(tmp_sys_idx[:, LOOKUP_SNAP] * snap_interval / 1e6, tmp_sys_idx[:, LOOKUP_MTOT],
                                    color=cols[kk], label=f"Star {kk + 1}")
        axs[2].semilogy(tmp_sys_idx[:, LOOKUP_SNAP] * snap_interval / 1e6, tmp_sys_idx[:, LOOKUP_M], color=cols[kk],
                        linestyle='--')
        # ls_mass.append(tmp_line)

    # axs[2].ticklabel_format(axis='both', style='plain')
    labelLines([l1])


    # Create a custom formatter
    # Define a custom formatter function
    def log_formatter(y, pos):
        if y >= 1:
            return f'{int(y)}'  # For values 1 and above, show as integers
        else:
            return f'{y:.1f}'  # For values below 1, show one decimal place


    # Apply the custom formatter to the y-axis
    axs[2].yaxis.set_major_formatter(FuncFormatter(log_formatter))


    fig1.savefig(f"bin_example_a_{bb}.pdf")
    fig2.savefig(f"bin_example_e_{bb}.pdf")
    fig3.savefig(f"bin_example_m_{bb}.pdf")
