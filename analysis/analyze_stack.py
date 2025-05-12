from collections import defaultdict
import copy
import glob

import numpy as np
from numba import njit

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_SMA = 6
LOOKUP_ECC = 7
LOOKUP_Q = 8

sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))
sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
vxcol = np.where(sink_cols == "vx")[0][0]
vycol = np.where(sink_cols == "vy")[0][0]
vzcol = np.where(sink_cols == "vz")[0][0]
hcol = np.where(sink_cols == "h")[0][0]
mcol = np.where(sink_cols == "m")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]
scol = np.where(sink_cols == "sys_id")[0][0]

def npz_stack(npz_list):
    """
    Stack data from different seeds
    """
    # Dictionary to hold lists of arrays for each key
    data_dict = defaultdict(list)

    # Loop over .npz files in the directory
    for filename in npz_list:
        # Load the .npz file
        data = np.load(filename, allow_pickle=True)

        # Iterate over keys in the .npz file
        for key in data.keys():
            data_dict[key].append(data[key])  # Append data for this key
    concatenated_data_dict = {key: np.concatenate(arrays) for key, arrays in data_dict.items()}

    return concatenated_data_dict

# @njit
def subtract_path(p1, p2):
    """
    Function to get displacement of 2 stars accounting for infinity placeholders
    """
    assert len(p1)==len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:,0])) & (~np.isinf(p2[:,0]))
    diff[filt] = p1[filt] - p2[filt]

    return diff

@njit
def get_peri(x, y, z, vx, vy, vz, mtot):
    """
    Compute 2-body pericenter--Given coordinates of relative positions and velocities.
    """
    sep = np.sqrt(x * x + y * y + z * z)
    vrel = np.sqrt(vx * vx + vy * vy + vz * vz)
    en = -sfc.GN * mtot / (sep) + 0.5 * vrel * vrel
    ell = np.cross((x, y, z), (vx, vy, vz))
    ell = np.sqrt(ell[0] * ell[0] + ell[1] * ell[1] + ell[2] * ell[2])

    return -sfc.GN * mtot / (2. * en) * (1. - np.sqrt(1. + 2. * en * ell**2. / (sfc.GN * mtot)**2.))

@njit
def subtract_path_opt(p1, p2):
    """
    Efficiently compute p1 - p2, skipping rows where either is [inf, inf, inf]
    """
    n = p1.shape[0]
    # diff = np.empty((n, 3))
    d = np.empty(n)
    angs = np.empty(n)

    for i in range(n):
        if np.isinf(p1[i, 0]) or np.isinf(p2[i, 0]):
            d[i] = np.inf
            angs[i] = 0
        else:
            dx = p1[i, 0] - p2[i, 0]
            dy = p1[i, 1] - p2[i, 1]
            dz = p1[i, 2] - p2[i, 2]
            dvx = p1[i, 3] - p2[i, 3]
            dvy = p1[i, 4] - p2[i, 4]
            dvz = p1[i, 5] - p2[i, 5]
            mtot = p1[i, 6] + p2[i, 6]
            angs[i] = dx * dvx + dy * dvy + dz * dvz
            if (i > 0) and (angs[i] * angs[i-1] < 0):
                d[i] = get_peri(dx, dy, dz, dvx, dvy, dvz, mtot)
            else:
                d[i] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return d


def subtract_path_1d(p1, p2):
    assert len(p1)==len(p2)
    diff = np.ones(len(p1)) * np.inf
    filt = (~np.isinf(p1)) & (~np.isinf(p2))
    diff[filt] = p1[filt] - p2[filt]

    return diff

def max_w_infinite(p1):
    """
    Maximum of ID array with infinity placeholders.
    """
    if np.all(np.isinf(p1)):
        return np.inf
    else:
        return np.max(p1[~np.isinf(p1)])

def get_min_dist_binary(path_lookup, tmp_row):
    """
    Get time series of separations between binary and other stars
    """
    p1_raw = path_lookup[tmp_row[0]]
    p2_raw = path_lookup[tmp_row[1]]
    path_lookup_keys = path_lookup.keys()

    path_diff_all = []
    for ii, uu in enumerate(path_lookup_keys):
        #Want only closest approach of stars external to the binary.
        if uu in tmp_row:
            continue
        ##Filtering out other seeds? Could be done more robustly/elegantly
        if len(path_lookup[uu]) != len(p1_raw):
            continue

        ##Displacement from binary com
        path_diff1 = subtract_path_opt(path_lookup[uu][:, pxcol:pzcol + 1], p1_raw[:, pxcol:pzcol + 1])
        # path_diff1 = np.sum(path_diff1 * path_diff1, axis=1)**.5
        path_diff2 = subtract_path_opt(path_lookup[uu][:, pxcol:pzcol + 1], p2_raw[:, pxcol:pzcol + 1])
        # path_diff2 = np.sum(path_diff2 * path_diff2, axis=1)**.5
        path_diff = np.min((path_diff1, path_diff2), axis=0)
        path_diff_all.append(path_diff)

    path_diff_all = np.array(path_diff_all).T
    closest_idx = np.argmin(path_diff_all, axis=1)
    closest_val = path_diff_all[np.arange(path_diff_all.shape[0]), closest_idx]
    del path_diff_all

    # path_diff_all_order = np.argsort(path_diff_all, axis=1)
    # path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)

    return closest_val, closest_idx

def get_closest_star_time_series(path_lookup, my_key):
    p1_raw = path_lookup[my_key]
    ##Filtering out other seeds? Could be done more robustly/elegantly
    path_lookup_keys = np.array(list(path_lookup.keys()))
    nsnaps = np.array([len(path_lookup[kk]) for kk in path_lookup_keys])
    path_lookup_keys = path_lookup_keys[nsnaps==len(p1_raw)]

    path_diff_all = []
    for ii, uu in enumerate(path_lookup_keys):
        ##Exclude the star itself
        if uu==my_key:
            continue

        ##Getting separations for all particles...
        tmp_path1 = path_lookup[uu][:, [pxcol, pycol, pzcol, vxcol, vycol, vzcol, mcol]]
        tmp_path2 = p1_raw[:, [pxcol, pycol, pzcol, vxcol, vycol, vzcol, mcol]]
        path_diff = subtract_path_opt(tmp_path1, tmp_path2)
        # path_diff = np.sum(path_diff * path_diff, axis=1)**.5
        path_diff_all.append(path_diff)
    path_diff_all = np.array(path_diff_all).T
    # path_diff_all_order = np.argsort(path_diff_all, axis=1)
    # path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)
    closest_idx = np.argmin(path_diff_all, axis=1)
    closest_val = path_diff_all[np.arange(path_diff_all.shape[0]), closest_idx]
    del path_diff_all
    keys = path_lookup_keys[path_lookup_keys!=my_key][closest_idx]
    closest_comp = [[my_key, keys[ii], path_lookup[keys[ii]][ii, mcol], path_lookup[keys[ii]][ii, mtotcol], closest_val[ii]] for ii in range(len(keys))]
    closest_comp = np.array(closest_comp)
    filt = ~np.isinf(closest_comp[:,-1].astype(float))

    return closest_comp[filt]

def get_closest_star_time_series_mem_opt(path_lookup, my_key):
    p1_raw = path_lookup[my_key]
    ##Filtering out other seeds? Could be done more robustly/elegantly
    path_lookup_keys = np.array(list(path_lookup.keys()))
    nsnaps = np.array([len(path_lookup[kk]) for kk in path_lookup_keys])
    path_lookup_keys = path_lookup_keys[nsnaps==len(p1_raw)]

    # path_diff_all = []
    min_dists = np.full(len(p1_raw), np.inf)
    min_keys = np.full(len(p1_raw), "", dtype=object)
    for ii, uu in enumerate(path_lookup_keys):
        ##Exclude the star itself
        if uu==my_key:
            continue
        path_diff = subtract_path_opt(path_lookup[uu][:, pxcol:pzcol + 1], p1_raw[:, pxcol:pzcol + 1])
        update_mask = path_diff < min_dists

        min_dists[update_mask] = path_diff[update_mask]
        min_keys[update_mask] = uu

    # print(min_keys)
    closest_comp = [
        [my_key, min_keys[ii], path_lookup[min_keys[ii]][ii, mcol], path_lookup[min_keys[ii]][ii, mtotcol], min_dists[ii]]
        for ii in range(len(min_keys)) if not np.isinf(min_dists[ii])
    ]

    return np.array(closest_comp)


##Only do 1 seed at a time
# def get_closest_star_time_series_transposed(path_lookup_time, my_key):
#     p1_raw = path_lookup[my_key]
    ##Filtering out other seeds? Could be done more robustly/elegantly
    #
    # path_diff_all = []
    # for ii, uu in enumerate(path_lookup_keys):
    #     ##Getting separations for all particles...
    #     path_diff = subtract_path_opt(path_lookup[uu][:, pxcol:pzcol + 1], p1_raw[:, pxcol:pzcol + 1])
    #     # path_diff = np.sum(path_diff * path_diff, axis=1)**.5
    #     path_diff_all.append(path_diff)
    # path_diff_all = np.array(path_diff_all).T
    # # path_diff_all_order = np.argsort(path_diff_all, axis=1)
    # # path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)
    # closest_idx = np.argmin(path_diff_all, axis=1)
    # closest_val = path_diff_all[np.arange(path_diff_all.shape[0]), closest_idx]
    #
    # keys = path_lookup_keys[path_lookup_keys!=my_key][closest_idx]
    # closest_comp = [[my_key, keys[ii], path_lookup[keys[ii]][ii, mcol], path_lookup[keys[ii]][ii, mtotcol], closest_val[ii]] for ii in range(len(keys))]
    # closest_comp = np.array(closest_comp)
    # filt = ~np.isinf(closest_comp[:,-1].astype(float))

    # return closest_comp[filt]

def get_closest_star_time_series_T(path_lookup, my_key, t):
    p1_raw = path_lookup[my_key]
    if np.isinf(p1_raw[t, 0]):
        return "blank", np.inf
    path_lookup_keys = np.array(list(path_lookup.keys()))
    nsnaps = np.array([len(path_lookup[kk]) for kk in path_lookup_keys])
    path_lookup_keys = path_lookup_keys[nsnaps==len(p1_raw)]

    pos_all = np.array([path_lookup[kk][t, pxcol:pzcol+1] for kk in path_lookup_keys])
    delta = pos_all - p1_raw[t][pxcol:pzcol+1]
    delta = np.sum(delta * delta, axis=1)**.5
    order = np.argsort(delta)

    return path_lookup_keys[order[1]], delta[order[1]]
#
# def get_closest_star_time_series_exp(path_lookup, my_key):
#     p1_raw = path_lookup[my_key]
#
#     path_lookup_keys = np.array(list(path_lookup.keys()))
#     nsnaps = np.array([len(path_lookup[kk]) for kk in path_lookup_keys])
#     path_lookup_keys = path_lookup_keys[nsnaps == len(p1_raw)]
#
#     pos_all = np.array([path_lookup[kk][:, pxcol:pzcol + 1] for kk in path_lookup_keys])
#     delta = pos_all - p1_raw[:, pxcol:pzcol + 1]
#     delta = np.sum(delta * delta, axis=2) ** .5
#     delta = delta.T
#     order = np.argsort(delta, axis=1)
#
#     return path_lookup_keys[order[:, 1]], np.take_along_axis(delta, order, axis=1)[:, 1]

def make_binned_data(absc, ords, bins):
    """
    Binning of (boolean) ords according to absc and bins
    """
    binned_num = np.zeros(len(bins) - 1)
    binned_den = np.zeros(len(bins) - 1)
    binned_numu = np.zeros(len(bins) - 1)
    for bidx in range(1, len(bins)):
        tmp_filt = (absc >= bins[bidx - 1]) & (absc < bins[bidx])
        tmp_ords = ords[tmp_filt]

        binned_num[bidx - 1] = len(tmp_ords[tmp_ords > 0])
        binned_numu[bidx - 1] = len(tmp_ords[tmp_ords > 0]) ** .5
        binned_den[bidx - 1] = len(tmp_ords)

    return binned_num, binned_numu, binned_den

def get_soft_times(id1, id2, path_lookup):
    d12 = subtract_path(path_lookup[f"{id1}"][:, pxcol:pzcol+1], path_lookup[f"{id2}"][:, pxcol:pzcol+1])
    d12 = np.sum(d12 * d12, axis=1)**.5
    hmax = np.max((path_lookup[f"{id1}"][:, hcol], path_lookup[f"{id2}"][:, hcol]), axis=0)
    soft_times = np.where(d12 < hmax)[0]

    return soft_times

def get_fpaths(base_path, cloud_tag, seed, analysis_tag, v_str="."):
    """
    Auxiliary function for generating file paths.
    """
    sim_tag = f"{cloud_tag}_{seed}"
    cloud_tag_split = cloud_tag.split("_")
    cloud_tag0 = f"{cloud_tag_split[0]}_{cloud_tag_split[1]}"

    base = base_path + f"/{v_str}/{cloud_tag0}/{sim_tag}/"
    r1 = base_path + f"/{v_str}/{cloud_tag0}/{sim_tag}/{cloud_tag_split[0]}_snapshot_"
    r2 = "_" + analysis_tag
    base_sink = base + f"/sinkprop/{cloud_tag_split[0]}_snapshot_"

    return base, base_sink, r1, r2, cloud_tag0, sim_tag


def get_snap_info(base, base_sink):
    """
    Getting info about snapshot files -- cadence (difference between snapshot numbers), snapshot time intervel (yr),
    start_snap (first snapshot number), end_snap (last snapshot number)
    """
    snaps = [xx.replace(base_sink, "").replace(".sink", "") for xx in glob.glob(base_sink + "*.sink")]
    snaps = np.array(snaps).astype(int)
    cadence = np.diff(np.sort(snaps))[0]
    snap_interval = np.atleast_1d(np.genfromtxt(base + "/sinkprop/snap_interval")).astype(float)
    ##Get snapshot numbers automatically
    start_snap = min(snaps)
    end_snap = max(snaps)

    return cadence, snap_interval, start_snap, end_snap

def get_end_time_set(my_set, path_lookup):
    """
    Get final time each of star of my_set exists and the maximum mass (primary of that set)...
    """
    ps = [path_lookup[str(ss)][:, [0, mcol]] for ss in my_set]
    ps = np.array(ps)

    ps = np.swapaxes(ps, 0, 1)
    end_stars_row = ps[~np.isinf(np.mean(ps[:, :, 0], axis=1))][-1]
    return end_stars_row[0, 0], max(end_stars_row[:, 1])

def get_bound_snaps_adjust(bin_list, high_df):
    ##Use high_df table to get more stringent binary snapshots(!!!)
    curr_bin_list = copy.copy(bin_list)
    curr_bin_list.sort()
    bin_sel = high_df.loc[str(curr_bin_list)]

    return bin_sel

