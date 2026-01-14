import pickle

import numpy as np
import tqdm
from scipy.interpolate import interp1d
from scipy.spatial import cKDTree

##REFACTOR INTO ITS OWN FILE(!!!)
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


##Could come up with a better version of this using the bhswallow files--at least useful for consistency check...
def get_mass_thres_time(path_lookup, thres):
    path_lookup_keys = path_lookup.keys()
    time_thres_lookup = {kk: np.inf for kk in path_lookup_keys}
    for kk in tqdm.tqdm(path_lookup_keys):
        tmp_path = path_lookup[kk]
        tmp_path = tmp_path[~np.isinf(tmp_path[:, 0])]
        idx = np.where(tmp_path[:, mcol] > thres)[0]
        if len(idx) == 0:
            continue
        elif idx[0] == 0:
            time_thres_lookup[kk] = 0
        else:
            ##This is the *time* when the star crosses the mass threshold
            tinterp = interp1d(
                (tmp_path[idx[0] - 1, mcol], tmp_path[idx[0], mcol]),
                (tmp_path[idx[0] - 1, 0], tmp_path[idx[0], 0]),
            )(thres)
            time_thres_lookup[kk] = float(tinterp)
    return time_thres_lookup


def parse_data(
    sim_params,
    seed,
    snapshot,
    mass_thres=0.0,
    radial_bins=np.geomspace(1e-3, 5, 15),
    age_cut=0.2,
):
    base_sink = f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/{sim_params}_{seed}/sinkprop/"
    dat = np.genfromtxt(base_sink + f"M2e4_snapshot_{snapshot:03d}.sink")
    # pos = dat[:, 1:4]
    adat = np.genfromtxt(base_sink + f"M2e4_snapshot_{snapshot:03d}.age")

    with open(base_sink + "/path_lookup.p", "rb") as ff:
        tmp_path_pickle = pickle.load(ff)
    time_thres_lookup = get_mass_thres_time(tmp_path_pickle, mass_thres)

    tcorr = np.array([time_thres_lookup[str(int(kk))] for kk in dat[:, 0]])
    # with open(base_sink + "/snap_interval") as snap_interval_file:
    #     snap_interval = float(snap_interval_file.read())
    snap_interval = 2.4703e4
    ##Conditional will handle the case where the mass does not cross through the chosen snapshots within the existing simulation snapshots...
    ##Current age relative to the time star reaches 0.1 Msun (will be negative if the sink is below this mass). If sink never reach 0.1 Msun, then the age will be -inf
    adat = np.array(
        [
            (
                (snapshot - tcorr[ii]) * snap_interval / 1e6
                if tcorr[ii] > 0
                else adat[ii]
            )
            for ii in range(len(adat))
        ]
    )

    # dat = dat[(dat[:, -1] > mass_min)]
    dat = dat[(adat > 0) & (adat < age_cut)]
    ##xy positions -- converted to angles using Taurus-Auriga distance: 140 pc [?]
    taurus_dist = 140.0
    pos = dat[:, 1:3] / taurus_dist * 180.0 / np.pi

    tree1 = cKDTree(pos)
    my_counts = tree1.count_neighbors(tree1, radial_bins)
    return my_counts, time_thres_lookup, len(dat), adat
