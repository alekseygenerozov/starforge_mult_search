from collections import defaultdict
import numpy as np

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_SMA = 6
LOOKUP_ECC = 7
LOOKUP_Q = 8
snap_interval = 2.47e4

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

def subtract_path(p1, p2):
    """
    Function to get displacement of 2 stars accounting for infinity placeholders
    """
    assert len(p1)==len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:,0])) & (~np.isinf(p2[:,0]))
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
        path_diff1 = subtract_path(path_lookup[uu][:, pxcol:pzcol + 1], p1_raw[:, pxcol:pzcol + 1])
        path_diff1 = np.sum(path_diff1 * path_diff1, axis=1)**.5
        path_diff2 = subtract_path(path_lookup[uu][:, pxcol:pzcol + 1], p2_raw[:, pxcol:pzcol + 1])
        path_diff2 = np.sum(path_diff2 * path_diff2, axis=1)**.5
        path_diff = np.min((path_diff1, path_diff2), axis=0)
        path_diff_all.append(path_diff)

    path_diff_all = np.array(path_diff_all).T
    path_diff_all_order = np.argsort(path_diff_all, axis=1)
    path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)

    return path_diff_all


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