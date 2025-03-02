import glob
import numpy as np
import matplotlib.pyplot as plt
##See if we can do the filtering in a cleaner/simpler way np.in1d can be a little uninituitive in some cases
import sys
import pickle
import h5py
import tqdm

from starforge_mult_search.code import find_multiples_new2, halo_masses_single_double_par
from starforge_mult_search.code.find_multiples_new2 import cluster,system

# from find_multiples_new2 import cluster, system
sys.path.append("/home/aleksey/code/python")
from bash_command import bash_command as bc
import cgs_const as cgs
import pandas as pd

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MTOT = 4
LOOKUP_M = 5
LOOKUP_SMA = 6
LOOKUP_ECC = 7
##TO FIX(!!)
# snap_interval = 2.47e4

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

def snap_lookup(tmp_dat, pid, ID_COLUMN=0):
    tmp_idx = np.where(tmp_dat[:, ID_COLUMN].astype(int) == pid)[0][0]
    return tmp_dat[tmp_idx], tmp_idx


def get_energy_from_sink_table(tmp_dat, row_idx):
    tmp_dat_select = tmp_dat[row_idx]
    tmp_pos = tmp_dat_select[:, 1:4]
    tmp_vel = tmp_dat_select[:, 4:7]
    tmp_h = tmp_dat_select[:, -2]
    tmp_mass = tmp_dat_select[:, -1]

    pe1 = halo_masses_single_double_par.PE(tmp_pos, tmp_mass, tmp_h)
    ke1 = halo_masses_single_double_par.KE(tmp_pos, tmp_mass, tmp_vel, np.zeros(len(tmp_pos)))

    return pe1 + ke1

def get_unique_binaries(r1, r2, start_snap, end_snap, cadence):
    """
    Get the unique binaries from a series of snapshots...

    :param string r1: Bases of pickle file name
    :param string r2: End of pickle file name
    :param int start_snap: Starting snapshot index
    :param int end_snap: Ending snapshot index

    :return: (i) List of unique binary ids (ii) Array with the following
     columns (a) Snapshot of first appearance
     (b) Iniitial smas (c) Initial eccentricities (d) Initial com position
     (e) Initial masses (f) Initial separation
    :rtype: Tuple
    """
    bin_ids = []
    first_appearance = []
    sma_i = []
    ecc_i = []
    mass_i1 = []
    mass_i2 = []
    pos_ix = []
    pos_iy = []
    pos_iz = []
    sep_i = []

    bin_ids_all = []
    times_all = []
    nsys = []

    for ss in range(start_snap, end_snap + 1, cadence):
        try:
            with open(r1 + "{0:03d}".format(ss) + r2, "rb") as ff:
                print(r1 + "{0:03d}".format(ss) + r2)
                cl_a = pickle.load(ff)
        except FileNotFoundError:
            print("Not found", ss)
            continue
        mults_a = np.array([sys1.multiplicity for sys1 in cl_a.systems])
        ids_a = np.array([set(sys1.ids) for sys1 in cl_a.systems], dtype=object)
        bin_ids_all.append(ids_a[mults_a == 2])
        times_all.append(np.ones(ids_a[mults_a == 2].size) * ss)
        nsys.append([ss, len(ids_a), len(ids_a[mults_a == 2]), len(ids_a[mults_a == 3]), len(ids_a[mults_a == 4])])

        for jj, pp_set in enumerate(ids_a):
            if (mults_a[jj] == 2) and ~np.isin(pp_set, bin_ids):
                bin_ids.append(pp_set)
                first_appearance.append(ss)
                sma_i.append(cl_a.systems[jj].orbits[0, 0])
                ecc_i.append(cl_a.systems[jj].orbits[0, 1])
                pos_ix.append(cl_a.systems[jj].orbits[0, 4])
                pos_iy.append(cl_a.systems[jj].orbits[0, 5])
                pos_iz.append(cl_a.systems[jj].orbits[0, 6])
                mass_i1.append(cl_a.systems[jj].orbits[0, 10])
                mass_i2.append(cl_a.systems[jj].orbits[0, 11])
                sep_i.append(np.linalg.norm(np.diff(cl_a.systems[jj].sub_pos, axis=0)))

    return bin_ids, np.transpose((first_appearance, sma_i, ecc_i, pos_ix, pos_iy, pos_iz,
                                  mass_i1, mass_i2, sep_i)), np.concatenate(bin_ids_all), np.concatenate(times_all),\
        nsys

def create_sys_lookup_table(r1, r2, base_sink, start_snap, end_snap, cadence):
    """
    Get the a lookup table for parent system/orbit of each star

    :param r1 string: Bases of pickle file name
    :param r2 string: End of pickle file name
    :param start_snap int: Starting snapshot index
    :param end_snap int: Ending snapshot index

    :return: numpy array where columns are (i) snapshot (ii) star id (iii) index of parent system (iv) multiplicity
    mass (with and without gas), semimajor axis, eccentricity.
    :rtype: np.ndarray
    """
    lookup = []
    for ss in range(start_snap, end_snap + 1, cadence):
        try:
            with open(r1 + "{0:03d}".format(ss) + r2, "rb") as ff:
                cl = pickle.load(ff)
        except FileNotFoundError:
            continue
        ids_a = np.array([sys1.ids for sys1 in cl.systems], dtype=object)
        tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))

        for ii in range(len(ids_a)):
            mprim = max(cl.systems[ii].sub_mass)
            mprim_id = ids_a[ii][np.argmax(cl.systems[ii].sub_mass)]
            masses_sorted = np.sort(cl.systems[ii].sub_mass)[::-1]
            for jj, elem1 in enumerate(ids_a[ii]):
                m1 = cl.systems[ii].sub_mass[jj]
                star_order = np.where(masses_sorted==m1)[0][0]
                tmp_orb = cl.systems[ii].orbits
                w1_row, w1_idx = snap_lookup(tmp_sink, elem1)
                if len(tmp_orb) == 0:
                    sma1 = -1
                    ecc1 = -1
                    q1 = -1
                ##Using mass to identify particles -- In principle may not give a unique match...
                ##Perhaps use assert to check this assumption(!!!)
                else:
                    sel1 = np.isclose(m1, tmp_orb[:, 10:12])
                    sel1 = np.array([row[0] or row[1] for row in sel1])
                    sma1 = tmp_orb[sel1][0][0]
                    ecc1 = tmp_orb[sel1][0][1]
                    q1 = m1 / (np.sum(tmp_orb[sel1][:,10:12]) - m1)
                lookup.append([ss, elem1, ii, len(ids_a[ii]), m1, w1_row[-1], sma1, ecc1, q1, mprim, mprim_id, star_order])

    return np.array(lookup)

def get_first_snap_idx(base_sink, start_snap, end_snap, cadence):
    """
    Get lookup table of the first snapshot index used
    """
    tmp_snap_idx = []
    for ss in range(start_snap, end_snap + 1, cadence):
        tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
        tmp_idx = tmp_sink[:, 0].astype(int)
        tmp_snap = [ss for ii in range(len(tmp_idx))]
        tmp_snap_idx.append(np.transpose([tmp_idx, tmp_sink[:, 1], tmp_sink[:, 2], tmp_sink[:, 3], tmp_snap]))

    tmp_snap_idx = np.vstack(tmp_snap_idx)
    tmp_uu, tmp_ui = np.unique(tmp_snap_idx[:, 0], return_index=True)
    first_snap_idx = tmp_snap_idx[tmp_ui]

    return first_snap_idx

def get_age_diff(bin_ids, first_snap_idx):
    """
    Age differences between the binary stars
    """
    delta_snap = np.zeros(len(bin_ids))
    for ii, pp in enumerate(bin_ids):
        row = list(pp)
        w1_row, w1 = snap_lookup(first_snap_idx, row[0])
        snap1 = first_snap_idx[w1, -1]

        w1_row, w1 = snap_lookup(first_snap_idx, row[1])
        snap2 = first_snap_idx[w1, -1]

        delta_snap[ii] = max(snap1, snap2) - min(snap1, snap2)
    return delta_snap

def get_fst(first_snapshot_idx, uids):
    fst_idx = np.zeros(len(uids)).astype(int)
    for ii, row in enumerate(uids):
        row_li = list(row)
        for tmp_item in row_li:
            tmp_snap1 = snap_lookup(first_snapshot_idx, tmp_item)[0][-1]
            fst_idx[ii] = max(tmp_snap1, fst_idx[ii])

    return fst_idx

def get_paths(base_sink, save_path, lookup, start_snap, end_snap, cadence):
    sinks_all = []
    spins_all = []
    ts = []
    tags = []
    accels = []
    nsinks = np.zeros(end_snap + 1)
    for ss in range(start_snap, end_snap + 1, cadence):
        tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
        sinks_all.append(tmp_sink)
        tmp_spin = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.spin".format(ss)))
        spins_all.append(tmp_spin)
        # tmp_accel_stars = pytreegrav.Accel($$$, $$$, $$$, theta=0.5, G=sfc.GN, method='bruteforce')
        # accels.append(tmp_accel_stars)

        ts.append(ss * np.ones(len(tmp_sink)))

    sinks_all = np.vstack(sinks_all)
    ts = np.concatenate(ts)
    ts.shape = (-1, 1)
    sinks_all = np.hstack((ts, sinks_all))
    sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))
    spins_all = np.vstack(spins_all)
    spins_all = np.hstack((ts, spins_all))

    tags = ["{0}_{1}".format(sinks_all[ii, 1], sinks_all[ii, 0]) for ii in range(len(ts))]
    sinks_all = sinks_all[np.argsort(tags)]
    spins_all = spins_all[np.argsort(tags)]
    tags2 = ["{0}_{1}".format(lookup[ii, 1], lookup[ii, 0]) for ii in range(len(ts))]
    lookup_sorted = lookup[np.argsort(tags2)]
    sinks_all = np.hstack((sinks_all, lookup_sorted[:, [2, LOOKUP_MTOT, LOOKUP_SMA, LOOKUP_ECC]]))
    sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
    assert (np.all(np.array(tags)[np.argsort(tags)] == np.array(tags2)[np.argsort(tags2)]))
    ######Saving a path for each particle
    utags = np.unique(sinks_all[:, 1])
    utags_str = utags.astype(int).astype(str)
    utimes = np.unique(sinks_all[:, 0])
    path_lookup = {}
    spin_lookup = {}
    path_lookup_times = {}
    for ii, uu in enumerate(utags):
        tmp_sel = sinks_all[sinks_all[:, 1] == uu]
        tmp_path1 = np.ones((end_snap + 1, len(sink_cols))) * np.inf
        tmp_path1[tmp_sel[:, 0].astype(int)] = tmp_sel
        path_lookup[utags_str[ii]] = tmp_path1
    for ii, uu in enumerate(utags):
        tmp_sel = spins_all[sinks_all[:, 1] == uu]
        tmp_path1 = np.ones((end_snap + 1, 4)) * np.inf
        tmp_path1[tmp_sel[:, 0].astype(int)] = tmp_sel
        spin_lookup[utags_str[ii]] = tmp_path1

    for ii, uu in enumerate(utimes):
        tmp_sel = sinks_all[sinks_all[:, 0] == uu]
        path_lookup_times[int(uu)] = tmp_sel

    with open(save_path + "/path_lookup.p", "wb") as ff:
        pickle.dump(path_lookup, ff)

    with open(save_path + "/spin_lookup.p", "wb") as ff:
        pickle.dump(spin_lookup, ff)

    with open(save_path + "/path_lookup_times.p", "wb") as ff:
        pickle.dump(path_lookup_times, ff)

    return path_lookup

def get_acc(r1, r2, start_snap, end_snap, cadence):
    ts_all = []
    ids_all = []
    accs_all = []

    for ss in tqdm.tqdm(range(start_snap, end_snap + 1, cadence)):
        with open(f"{r1}{ss:03d}{r2}", "rb") as ff:
            cl = pickle.load(ff)
        tmp_ids = np.concatenate([ss.ids for ss in cl.systems])
        tmp_ts = [ss] * len(tmp_ids)
        tmp_accels = np.concatenate([ss.sub_accel for ss in cl.systems])

        ts_all.append(tmp_ts)
        ids_all.append(tmp_ids)
        accs_all.append(tmp_accels)

    ids_all = np.concatenate(ids_all)
    ids_all.shape = (-1, 1)
    ts_all = np.concatenate(ts_all)
    ts_all.shape = (-1, 1)
    accs_all = np.vstack(accs_all)
    accs_all = np.hstack((ts_all, ids_all, accs_all))

    ncols = accs_all.shape[1]
    utags = np.unique(accs_all[:, 1])
    utags_str = utags.astype(int).astype(str)
    acc_lookup = {}
    for ii, uu in enumerate(utags):
        tmp_sel = accs_all[accs_all[:, 1] == uu]
        tmp_path1 = np.ones((end_snap + 1, ncols)) * np.inf
        tmp_path1[tmp_sel[:, 0].astype(int)] = tmp_sel

        acc_lookup[utags_str[ii]] = tmp_path1

    return acc_lookup

def main():
    cloud_tag = sys.argv[1]
    sim_tag = f"{cloud_tag}_{sys.argv[2]}"
    cloud_tag_split = cloud_tag.split("_")
    cloud_tag0 = f"{cloud_tag_split[0]}_{cloud_tag_split[1]}"
    base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/{0}/{1}/".format(cloud_tag0, sim_tag)
    r1 = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/{0}/{1}/M2e4_snapshot_".format(cloud_tag0, sim_tag)
    r2 = sys.argv[3]
    # base_sink = base + "/sinkprop/M2e4_snapshot_"
    base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
    print(base_sink)
    r2_nosuff = r2.replace(".p", "")
    snaps = [xx.replace(base_sink, "").replace(".sink", "") for xx in glob.glob(base_sink + "*.sink")]
    snaps = np.array(snaps).astype(int)
    cadence = np.diff(np.sort(snaps))[0]
    print(cadence)

    ##Get snapshot numbers automatically
    start_snap_sink = min(snaps)
    start_snap = min(snaps)
    end_snap = max(snaps)
    print(end_snap)
    aa = "analyze_multiples_output_{0}/".format(r2_nosuff)
    save_path = f"{cloud_tag0}/{sim_tag}/{aa}"
    # ####################################################################################################
    bc.bash_command(f"mkdir -p {save_path}")
    with open(save_path + "/mult_data_path", "w") as ff:
        ff.write(r1 + "\n")
        ff.write(r2 + "\n")
    with open(save_path + "/save_data_path", "w") as ff:
        ff.write(save_path + "\n")

    ##Getting list of unique binaries--TO DO: ALSO GET THOSE THAT WERE PART OF HIGHER ORDER MULTIPLES.
    bin_ids, ic, bin_ids_all, times_all, nsys = get_unique_binaries(r1, r2, start_snap, end_snap, cadence)
    np.savez(save_path + "/unique_bin_ids", bin_ids, ic)

    ##List of fst for these binaries
    first_snap_idx = get_first_snap_idx(base_sink, start_snap_sink, end_snap, cadence)
    fst = get_fst(first_snap_idx, bin_ids)
    np.savez(save_path + "/fst", fst)

    ##System lookup table
    lookup = create_sys_lookup_table(r1, r2, base_sink, start_snap, end_snap, cadence)
    lookup = np.hstack((lookup, np.ones(len(lookup))[:, np.newaxis] * end_snap))
    np.savez(save_path + "/system_lookup_table", lookup)
    lookup_dict = {}
    for uu in np.unique(lookup[:, 1]):
        lookup_dict[uu] = lookup[lookup[:, LOOKUP_PID] == uu]
    with open(save_path + f"/lookup_dict.p", "wb") as ff:
        pickle.dump(lookup_dict, ff)
    ##Particle paths...
    start_snap = int(min(lookup[:, LOOKUP_SNAP]))
    path_lookup = get_paths(base_sink, save_path, lookup, start_snap, end_snap, cadence)

    acc_lookup = get_acc(r1, r2, start_snap, end_snap, cadence)
    with open(save_path + "/acc_lookup.p", "wb") as ff:
        pickle.dump(acc_lookup, ff)


    #################################################################################
    ##Getting binaries that have *only* been in multiples...
    lookup_pd = pd.DataFrame(lookup, columns=("time", "pid", "sid", "mult", "ms+mhalo", "x",
                                              "sma", "ecc", "q", "mprim+mhalo", "mprim_id", "order", "tf"))

    ##Should get all the binaries ever -- including those in higher order multiples...
    sys_group = lookup_pd.groupby(["time", "sid", "sma"])[["time", "pid", "mult"]].apply(lambda group: [list(group['time'])[0]] + list(group["pid"]) if len(group) == 2 and group["mult"].min() >= 2 else None).dropna()
    sys_group = sys_group.to_list()
    ##Can try assert here to be sure that the array is sorted in time
    sys_group = np.array(sys_group)
    tfirst_bin_in_mult = sys_group[:,0]
    bin_in_mult = sys_group[:, [1, 2]].astype(int)
    bin_in_mult_str = [str(np.sort(row)) for row in bin_in_mult]
    tmp, tmp_uidx = np.unique(bin_in_mult_str, return_index=True)

    bin_in_mult = bin_in_mult[tmp_uidx]
    tfirst_bin_in_mult = tfirst_bin_in_mult[tmp_uidx]
    bin_in_mult = np.array([set(row) for row in bin_in_mult])

    np.savez(save_path + "/unique_bin_ids_mult", bin_in_mult, tfirst_bin_in_mult[:, np.newaxis])
    fst = get_fst(first_snap_idx, bin_in_mult)
    np.savez(save_path + "/fst_mult", fst)


#######################################################################################################################################################################
#######################################################################################################################################################################



if __name__ == "__main__":
    main()