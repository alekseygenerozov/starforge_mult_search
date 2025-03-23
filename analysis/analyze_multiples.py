import glob
import os
import pickle
import sys

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
from starforge_mult_search.code import (find_multiples_new2,
                                        halo_masses_single_double_par)
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import get_fpaths, get_snap_info, LOOKUP_PID, LOOKUP_SNAP, sink_cols

import pandas as pd


def snap_lookup(tmp_dat, pid, ID_COLUMN=0):
    tmp_idx = np.where(tmp_dat[:, ID_COLUMN].astype(int) == pid)[0][0]
    return tmp_dat[tmp_idx], tmp_idx

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
                ##Each star will have to appear *at least* once in the orbits--the first time it appears it
                ##will be as a single star.
                else:
                    sel1 = np.isclose(elem1, tmp_orb[:, 12:14])
                    sel1 = np.array([row[0] or row[1] for row in sel1])
                    tmp_orb = tmp_orb[sel1][0]
                    sma1 = tmp_orb[0]
                    ecc1 = tmp_orb[1]
                    q1 = m1 / (np.sum(tmp_orb[10:12]) - m1)
                lookup.append([ss, elem1, ii, len(ids_a[ii]), m1, w1_row[-1], sma1, ecc1, q1, mprim, mprim_id, star_order])

    return np.array(lookup)

def get_sink_df(base_sink, start_snap, end_snap, cadence):
    """
    Store sink data within Pandas dataframe.

    :param base_sink: Path and prefix of sink files
    :param start_snap: First snapshot
    :param end_snap: Last snapshot
    :param cadence: Gap between snapshot times.

    :return: Dataframe containing sink data.
    :rtype: Pandas dataframe
    """
    sinks_df = []
    for ss in range(start_snap, end_snap + 1, cadence):
        tmp_sink = pd.DataFrame(np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss))),
                                columns=["pid", "x", "y", "z", "vx", "vy", "vz", "h", "m"])
        tmp_sink.insert(0, "t", np.ones(len(tmp_sink)) * ss)
        sinks_df.append(tmp_sink)
    sinks_df = pd.concat(sinks_df).reset_index(drop=True)
    return sinks_df

def get_spin_df(base_sink, start_snap, end_snap, cadence):
    """
    Store spin data within Pandas dataframe.

    :param base_sink: Path and prefix of sink files
    :param start_snap: First snapshot
    :param end_snap: Last snapshot
    :param cadence: Gap between snapshot times.

    :return: Dataframe containing spins.
    :rtype: Pandas dataframe
    """
    spins_df = []
    for ss in range(start_snap, end_snap + 1, cadence):
        tmp_sink = np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.sink".format(ss)))
        tmp_spin = pd.DataFrame(np.atleast_2d(np.genfromtxt(base_sink + "{0:03d}.spin".format(ss))),
                                columns=["sx", "sy", "sz"])
        tmp_spin.insert(0, "pid", tmp_sink[:,0])
        tmp_spin.insert(0, "t", np.ones(len(tmp_spin)) * ss)
        spins_df.append(tmp_spin)
    spins_df = pd.concat(spins_df).reset_index(drop=True)
    return spins_df

def get_fst(first_snapshot_idx, uids):
    """
    Get first snapshot both stars exist for a list of stellar pairs.
    """
    fst_idx = np.zeros(len(uids)).astype(int)
    for ii, row in enumerate(uids):
        row_li = list(row)
        for tmp_item in row_li:
            tmp_snap1 = first_snapshot_idx.loc[float(tmp_item)]["t"]
            fst_idx[ii] = max(tmp_snap1, fst_idx[ii])

    return fst_idx

def get_paths(sinks_df, spins_df, lookup_df, save_path, end_snap):
    sinks_all = pd.concat([sinks_df, lookup_df[["sys_id", "mtot", "sma", "ecc"]]], axis=1)
    ##Collecting all tags and particle ids.
    utags = sinks_all["pid"].unique()
    utags = np.sort(utags)
    utags_str = utags.astype(int).astype(str)
    utimes = sinks_all["t"].unique()
    utimes = np.sort(utimes)

    path_lookup = {}
    spin_lookup = {}
    path_lookup_times = {}
    ##Saving a path for each particle
    for ii, uu in enumerate(utags):
        tmp_sel = sinks_all.loc[sinks_all["pid"] == uu]
        tmp_path1 = np.ones((end_snap + 1, len(sink_cols))) * np.inf
        tmp_path1[tmp_sel.iloc[:, 0].astype(int)] = tmp_sel
        path_lookup[utags_str[ii]] = tmp_path1
    for ii, uu in enumerate(utags):
        tmp_sel = spins_df.loc[spins_df["pid"] == uu]
        del tmp_sel["pid"]
        tmp_path1 = np.ones((end_snap + 1, 4)) * np.inf
        tmp_path1[tmp_sel.iloc[:, 0].astype(int)] = tmp_sel
        spin_lookup[utags_str[ii]] = tmp_path1
    for ii, uu in enumerate(utimes):
        tmp_sel = sinks_all.loc[sinks_all["t"] == uu]
        ##Changing format and ordering for backwards compatibility
        tmp_sel = np.array(tmp_sel.values.tolist())
        tmp_order = np.argsort(tmp_sel[:, 1].astype(str))
        tmp_sel = tmp_sel[tmp_order]

        path_lookup_times[int(uu)] = tmp_sel

    with open(save_path + "/path_lookup.p", "wb") as ff:
        pickle.dump(path_lookup, ff)

    with open(save_path + "/spin_lookup.p", "wb") as ff:
        pickle.dump(spin_lookup, ff)

    with open(save_path + "/path_lookup_times.p", "wb") as ff:
        pickle.dump(path_lookup_times, ff)

    return path_lookup

@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def main(params):
    base, base_sink, r1, r2, cloud_tag0, sim_tag = get_fpaths(params["base_path"], params["cloud_tag"], params["seed"], params["analysis_tag"], v_str=params["v_str"])
    r2_nosuff = r2.replace(".p", "")
    v_str = params["v_str"]
    cadence, snap_interval, start_snap, end_snap = get_snap_info(base, base_sink)
    #
    aa = "analyze_multiples_output_{0}/".format(r2_nosuff)
    save_path = f"{v_str}/{cloud_tag0}/{sim_tag}/{aa}"
    ###################################################################################################################
    ##Replace with more flexible command
    os.makedirs(save_path, exist_ok=True)
    # bc.bash_command(f"mkdir -p {save_path}")
    ##Store info about the surviving
    with open(save_path + "/mult_data_path", "w") as ff:
        ff.write(r1 + "\n")
        ff.write(r2 + "\n")

    ##System lookup table
    lookup = create_sys_lookup_table(r1, r2, base_sink, start_snap, end_snap, cadence)
    #Add the final snapshot -- Useful for when we have to stack multiple seeds.
    lookup = np.hstack((lookup, np.ones(len(lookup))[:, np.newaxis] * end_snap))
    np.savez(save_path + "/system_lookup_table", lookup)
    lookup_dict = {}
    for uu in np.unique(lookup[:, 1]):
        lookup_dict[uu] = lookup[lookup[:, LOOKUP_PID] == uu]
    with open(save_path + f"/lookup_dict.p", "wb") as ff:
        pickle.dump(lookup_dict, ff)
    ##Particle paths...
    start_snap_b = int(min(lookup[:, LOOKUP_SNAP]))
    assert start_snap == start_snap_b
    ###################################################################################################################
    ##Look pairs that are in the same system with the same semi-major axis.
    ##Should get all the binaries ever -- including those in higher order multiples.
    ##Semi-major axes are from same underlying data so don't have to worry about floating point issues.
    lookup_df = pd.DataFrame(lookup, columns=("time", "pid", "sys_id", "mult", "mtot", "x",
                                              "sma", "ecc", "q", "mprim+mhalo", "mprim_id", "order", "tf"))
    sys_group = lookup_df.groupby(["time", "sys_id", "sma"])[["time", "pid", "mult"]].apply(lambda group: [list(group['time'])[0]] + list(group["pid"]) if len(group) == 2 and group["mult"].min() >= 2 else None).dropna()
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
    ###################################################################################################################
    np.savez(save_path + "/unique_bin_ids_mult", bin_in_mult, tfirst_bin_in_mult[:, np.newaxis])
    ##Getting the initial snapshot together for all the binary pairs.
    sinks_df = get_sink_df(base_sink, start_snap, end_snap, cadence)
    first_snap_idx = sinks_df.groupby("pid").first()
    fst = get_fst(first_snap_idx, bin_in_mult)
    np.savez(save_path + "/fst_mult", fst)
    ###################################################################################################################
    spins_df = get_spin_df(base_sink, start_snap, end_snap, cadence)
    sinks_df = sinks_df.sort_values(["pid", "t"])
    spins_df = spins_df.sort_values(["pid", "t"])
    lookup_df = lookup_df.sort_values(["pid", "time"])
    sinks_df.reset_index(inplace=True, drop=True)
    spins_df.reset_index(inplace=True, drop=True)
    lookup_df.reset_index(inplace=True, drop=True)
    get_paths(sinks_df, spins_df, lookup_df, save_path, end_snap)
    ##################################################################################################################



if __name__ == "__main__":
    main()