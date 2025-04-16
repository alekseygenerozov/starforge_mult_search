from starforge_mult_search.analysis.figures.figure_preamble import *
from bash_command import bash_command as bc
import matplotlib.pyplot as plt

def get_com_winf(paths):
    paths2 = np.copy(paths)
    paths2[np.isinf(paths2)] = 0

    tmp_ms = paths2[:, :, mcol]
    tmp_ms.shape = (len(tmp_ms), -1, 1)
    com = path_divide_3d(np.sum((tmp_ms * paths2[:, :, pxcol:pzcol + 1]), axis=0), np.sum(tmp_ms, axis=0))

    return com

def path_divide_3d(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:,0])) & (~np.isinf(p2[:,0]))
    print(p1[filt].shape, p2[filt].shape)
    diff[filt] = p1[filt] / p2[filt]

    return diff

def get_com(paths):
    tmp_ms = paths[:, :, mcol]
    tmp_ms.shape = (len(tmp_ms), -1, 1)
    com = path_divide_3d(np.sum((tmp_ms * paths[:, :, pxcol:pzcol + 1]), axis=0), np.sum(tmp_ms, axis=0))

    return com

def plot_movie(ps, c1, t1, c2, t2, tt, start_time, end_time, com_flag=0, ax1=0, ax2=1, tag="", cols=None, ls=None,
               base_save="tmp_fig/", annotation=None):
    bc.bash_command(f"mkdir -p {base_save}")
    size_scale = 0.02
    paths = np.array([path_lookup[str(pp)] for pp in ps])
    paths_T = np.transpose(paths, axes=(1,0,2))
    tt = int(tt)

    if start_time < 0:
        start_time = get_fst(paths)
    # print(start_time)
    if cols is None:
        cols = ["k"] * len(ps)
    if ls is None:
        ls = ["-"] * len(ps)
    # print(start_time)

    coms = get_com_winf(paths)
    delta = np.zeros((end_time + 1, 3))
    if com_flag == 1:
        delta = np.array([coms[start_time]] * len(delta))
    elif com_flag > 0:
        delta = np.array(coms)

    # for tt in range(start_time, end_time + 1):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.set_xlabel("x [pc]")
    ax.set_ylabel("y [pc]")
    max_sep = 0.04
    ax.set_xlim(coms[tt, ax1] - delta[tt, ax1] - 10 * max_sep, coms[tt, ax1] - delta[tt, ax1] + 10 * max_sep)
    ax.set_ylim(coms[tt, ax2] - delta[tt, ax2] - 10 * max_sep, coms[tt, ax2] - delta[tt, ax2] + 10 * max_sep)

    tmp_hier = ""
    if annotation is not None:
        tmp_hier += annotation[tt]
    for pidx in range(len(ps)):
        ax.plot(paths_T[:tt + 1, pidx, pxcol + ax1] - delta[:tt + 1, ax1],
                paths_T[:tt + 1, pidx, pxcol + ax2] - delta[:tt + 1, ax2],
                color=cols[pidx])
        ax.scatter(paths_T[tt, pidx, pxcol + ax1] - delta[tt, ax1], paths_T[tt, pidx, pxcol + ax2] - delta[tt, ax2],
                marker="s",
                color=cols[pidx], s=paths_T[tt, pidx, mtotcol] / size_scale)

    ax.legend(title=tmp_hier, title_fontsize=16)

    ##All companions plotted together with hallow symbols
    comps = np.concatenate((np.unique(np.concatenate(c1)), np.unique(np.concatenate(c2))))
    comps = comps[(comps!=ps[0]) & (comps!=ps[1])]
    paths_extra = np.array([path_lookup[str(pp)] for pp in comps])
    paths_extra_T = np.transpose(paths_extra, axes=(1, 0, 2))
    if len(comps) > 0:
        ax.scatter(paths_extra_T[tt, :, pxcol + ax1] - delta[tt, ax1], paths_extra_T[tt, :, pxcol + ax2] - delta[tt, ax2],
                marker="o", linestyle="", alpha=0.5, facecolors='none', edgecolors=colorblind_palette[0], s=paths_extra_T[tt, :, mtotcol] / size_scale)
    ##Separate style for current companions...
    t_group = (t1, t2)
    for ii, cc in enumerate((c1, c2)):
        comps_curr = np.array(cc, dtype=object)[np.array(t_group[ii]) == tt]
        if len(comps_curr) > 0:
            comps_curr = comps_curr[0]
            comps_curr = comps_curr[(comps_curr!=ps[0]) & (comps_curr!=ps[1])]
            if len(comps_curr) > 0:
                print(comps_curr)
                paths_extra = np.array([path_lookup[str(pp)] for pp in comps_curr])
                paths_extra_T = np.transpose(paths_extra, axes=(1, 0, 2))
                ax.scatter(paths_extra_T[tt, :, pxcol + ax1] - delta[tt, ax1],
                           paths_extra_T[tt, :, pxcol + ax2] - delta[tt, ax2],
                           marker="o", linestyle="", alpha=0.5, facecolors=colorblind_palette[0], edgecolors=colorblind_palette[0],
                           s=paths_extra_T[tt, :, mtotcol] / size_scale)



    ##Separate style for other stars--use flag for this.
    fig.savefig(f"tmp_{tt:03d}.png")

def movie(df, tmp_bin_idx, tt, case_label=None):
    unique_binaries = df.index.get_level_values("binary").unique()
    my_bin = df.loc[unique_binaries[tmp_bin_idx]]
    print(my_bin)
    ps = my_bin.index.to_list()
    tmp_end_snap = int(lookup_dict[ps[0]][0, -1])

    comps1 = my_bin.iloc[0]["comps"]
    comps2 = my_bin.iloc[1]["comps"]
    times1 = my_bin.iloc[0]["times"]
    times2 = my_bin.iloc[1]["times"]
    hiers1 = my_bin.iloc[0]["hiers"]
    hiers2 = my_bin.iloc[1]["hiers"]
    first_star_snap = int(min(min(times1), min(times2)))

    annotations1 = [""] * 490
    for ii, ttt in enumerate(times1):
        annotations1[int(ttt)] = hiers1[ii]

    annotations2 = [""] * 490
    for ii, ttt in enumerate(times2):
        annotations2[int(ttt)] = hiers2[ii]

    annotations = [f"{annotations1[ii]}\n{annotations2[ii]}" for ii in range(len(annotations1))]
    if case_label is None:
        case_label = tmp_bin_idx
    plot_movie(ps, comps1, times1, comps2, times2, tt, first_star_snap, tmp_end_snap, annotation=annotations,
               base_save=f"exchange_followup/case_bi{case_label}")

if __name__=="__main__":
    df = pd.read_hdf("binary_data.h5", key="data")
    movie(df, 72, 300)