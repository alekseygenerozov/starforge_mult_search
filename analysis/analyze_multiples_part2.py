import glob
import h5py
import hydra
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import pickle

from starforge_mult_search.code import find_multiples_new2, halo_masses_single_double_par
from starforge_mult_search.code.find_multiples_new2 import cluster,system
from starforge_mult_search.analysis.analyze_stack import get_fpaths, get_snap_info, LOOKUP_PID, LOOKUP_SNAP, sink_cols
from starforge_mult_search.analysis import cgs_const as cgs

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MTOT = 4
LOOKUP_M = 5
LOOKUP_SMA = 6
LOOKUP_ECC = 7

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
#

def get_bound_snaps(sys1_info, sys2_info):
    sys1_tag = ["{0}_{1}".format(row[LOOKUP_SNAP], row[2]) for row in sys1_info]
    sys2_tag = ["{0}_{1}".format(row[LOOKUP_SNAP], row[2]) for row in sys2_info]
    ##STARS COULD BE IN THE SAME MULTIPLE BUT NOT BOUND--HAVE TO DO FURTHER FILTERING BASED ON SMA
    same_sys_filt1 = np.in1d(sys1_tag, sys2_tag)
    same_sys_filt2 = np.in1d(sys2_tag, sys1_tag)
    sys1_info = sys1_info[same_sys_filt1]
    sys2_info = sys2_info[same_sys_filt2]
    ##Semi-major axes are from same underlying data so no floating point issues.
    bound_filt = sys1_info[:, LOOKUP_SMA] == sys2_info[:, LOOKUP_SMA]

    bound_snaps1 = sys1_info[bound_filt]
    bound_snaps2 = sys2_info[bound_filt]

    return bound_snaps1, bound_snaps2, sys1_info[:, LOOKUP_SNAP]


def get_quasi(bin_ids, lookup_dict, fst, snap_interval, path_lookup):
    """
    Get info about time each binary pair was bound.

    :param bin_ids: List of binary pairs, where each pair is a set
    :param lookup_dict: Lookup table (dict) of properties (e.g. system id, multiplicity, sma, etc.), indexed by particle id
    :param fst: Array with the initial snapshot together for each pair
    :param snap_interval: Interval between simulation snapshot (float)
    :param path_lookup: Lookup table (dict) of properties of positions and masses for each binary pair.

    :return: Dictionary containing: persistence filter (True if binary is persistent), final snapshot binary is bound,
    final bound snapshot over the end snapshot (1 if the two stars are in a binary at the end), bound_time (number of
    snapshots stars are bound together as a binary), bound_time_norm (bound time normalized to the binary period),
    initial snapshot binary is bound,..., age difference between multiple stars, final snapshot
    stars in the same multiple system over the final snapshot in simulation (1 if stars are in the same multiple system
    in the end).
    :rtype: dict

    """
    bound_time = np.zeros(len(bin_ids))
    bound_time_norm = np.zeros(len(bin_ids))
    init_bound_snaps = np.zeros(len(bin_ids))
    final_bound_snaps = np.zeros(len(bin_ids))
    final_bound_snaps_norm = np.zeros(len(bin_ids))
    age_diff = np.zeros(len(bin_ids))
    same_sys_final_norm = np.zeros(len(bin_ids))
    final_pair_mass_lbin = np.zeros(len(bin_ids))
    final_pair_mass_lsys = np.zeros(len(bin_ids))
    bound_time_pers = []

    for ii, uid in enumerate(bin_ids):
        bin_list = list(uid)
        ##Getting the final snapshot stars are bound to each other--refactor into its own function...
        sys1_info = lookup_dict[bin_list[0]]
        sys2_info = lookup_dict[bin_list[1]]
        ##Filter to check multiple history prior to the initial snapshot together.
        age_diff[ii] = np.abs(sys1_info[0,0] - sys2_info[0,0]) * snap_interval[0]

        ##Get snapshots where the two stars are bound together.
        bound_snaps1, bound_snaps2, same_sys_snap = get_bound_snaps(sys1_info, sys2_info)
        ##Get total number of snapshots!
        nsnaps = bound_snaps1[0, -1]
        tmp_final_bound_snap = bound_snaps2[:, LOOKUP_SNAP][-1]
        final_bound_snaps[ii] = tmp_final_bound_snap
        final_bound_snaps_norm[ii] = tmp_final_bound_snap / nsnaps
        init_bound_snaps[ii] = bound_snaps1[:, LOOKUP_SNAP][0]
        same_sys_final_norm[ii] = same_sys_snap[-1] / nsnaps

        #Mass of pair at final binary/system snapshots
        path1 = path_lookup[f"{bin_list[0]}"]
        path2 = path_lookup[f"{bin_list[1]}"]
        fpm = path1[path1[:, 0] == tmp_final_bound_snap][0, mcol] + path2[path2[:, 0] == tmp_final_bound_snap][0, mcol]
        final_pair_mass_lbin[ii] = fpm
        fpm = path1[path1[:, 0] == same_sys_snap[-1]][0, mcol] + path2[path2[:, 0] == same_sys_snap[-1]][0, mcol]
        final_pair_mass_lsys[ii] = fpm

        ##Total time binaries are bound normalized to the period.
        tmp_bound_time_pers = (bound_snaps1[:, LOOKUP_SMA] * cgs.pc / cgs.au) ** 1.5 / (bound_snaps1[:, LOOKUP_MTOT] + bound_snaps2[:, LOOKUP_MTOT]) ** .5
        bound_time_pers.append(tmp_bound_time_pers)
        bound_time[ii] = len(bound_snaps1)
        bound_time_norm[ii] = np.sum(snap_interval / tmp_bound_time_pers)

    return {"quasi_filter": (bound_time_norm >= 1) & (bound_time > 1), "final_bound_snaps": final_bound_snaps,"final_bound_snaps_norm": final_bound_snaps_norm,
            "bound_time":bound_time, "bound_time_norm":bound_time_norm, "init_bound_snaps":init_bound_snaps,
            "age_diff": age_diff, "same_sys_final_norm": same_sys_final_norm, "final_pair_mass_lbin": final_pair_mass_lbin,
            "final_pair_mass_lsys": final_pair_mass_lsys, "bound_time_pers": bound_time_pers}

def get_energy(bin_ids, fst, lookup_dict, path_lookup):
    ens = np.ones(len(bin_ids)) * np.inf
    ens_gas = np.ones(len(bin_ids)) * np.inf
    same_sys_at_fst = np.ones(len(bin_ids)) * np.inf
    bin_at_fst = np.ones(len(bin_ids)) * np.inf
    vangs = np.ones(len(bin_ids)) * np.inf
    vangs_prim = np.ones(len(bin_ids)) * np.inf
    mfinal_primary = np.ones(len(bin_ids)) * np.inf
    mfinal_pair = np.ones(len(bin_ids)) * np.inf

    for ii, uid in enumerate(bin_ids):
        fst_idx = fst[ii]
        bin_list = list(uid)
        path1 = path_lookup[f"{bin_list[0]}"]
        path2 = path_lookup[f"{bin_list[1]}"]

        look1 = lookup_dict[bin_list[0]]
        look1 = look1[look1[:, LOOKUP_SNAP]==fst_idx]
        look2 = lookup_dict[bin_list[1]]
        look2 = look2[look2[:, LOOKUP_SNAP]==fst_idx]

        ##Check if pair is in the same multiple/same binary at fst.
        same_sys_at_fst[ii] = (look1[:,2] == look2[:, 2])[0]
        bin_at_fst[ii] = ((look1[:,2] == look2[:, 2]) and (look1[:, LOOKUP_SMA]==look2[:, LOOKUP_SMA]))[0]

        pos1 = path1[fst_idx, 2:5]
        pos2 = path2[fst_idx, 2:5]
        vel1 = path1[fst_idx, 5:8]
        vel2 = path2[fst_idx, 5:8]
        m1 = path1[fst_idx, mcol]
        m2 = path2[fst_idx, mcol]
        mtot1 = look1[0, LOOKUP_MTOT]
        mtot2 = look2[0, LOOKUP_MTOT]
        h1 = path1[fst_idx, hcol]
        h2 = path2[fst_idx, hcol]

        tmp_en_gas = find_multiples_new2.get_energy(pos1, pos2, vel1, vel2, mtot1, mtot2, h1=h1, h2=h2)
        tmp_en = find_multiples_new2.get_energy(pos1, pos2, vel1, vel2, m1, m2, h1=h1, h2=h2)
        ens_gas[ii] = tmp_en_gas[0] / tmp_en_gas[1]
        ens[ii] = tmp_en[0] / tmp_en[1]
        vangs[ii] = np.dot(vel1, vel2) / np.linalg.norm(vel1) / np.linalg.norm(vel2)
        vangs_prim[ii] = np.dot(vel1 - vel2, pos1 - pos2) / np.linalg.norm(vel1 - vel2) / np.linalg.norm(pos1 - pos2)

        ##Get masses of stars at the last snapshot both exist, which will be the last snapshot, unless there is a supernova
        ##Select snapshots where star exists by filtering infs
        mfilt = np.where((~np.isinf(path1[:, mcol])) & (~np.isinf(path2[:, mcol])))
        m1end = path1[mfilt][-1, mcol]
        m2end = path2[mfilt][-1, mcol]
        mfinal_primary[ii] = max(m1end, m2end)
        mfinal_pair[ii] = m1end + m2end

    return {"ens": ens, "ens_gas":ens_gas, "same_sys_at_fst":same_sys_at_fst, "bin_at_fst": bin_at_fst,
            "vangs": vangs, "vangs_prim": vangs_prim, "mfinal_primary": mfinal_primary, "mfinal_pair": mfinal_pair}

@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def main(params):
    base, base_sink, r1, r2, cloud_tag0, sim_tag = get_fpaths(params["base_path"], params["cloud_tag"], params["seed"], params["analysis_tag"], v_str=params["v_str"])
    r2_nosuff = r2.replace(".p", "")
    v_str = params["v_str"]
    cadence, snap_interval, start_snap, end_snap = get_snap_info(base, base_sink)

    aa = "analyze_multiples_output_{0}/".format(r2_nosuff)
    save_path = f"{v_str}/{cloud_tag0}/{sim_tag}/{aa}"
    analysis_suff = "_mult"
    ####################################################################################################
    #################################################################################

    bin_ids = np.load(save_path + f"/unique_bin_ids{analysis_suff}.npz", allow_pickle=True)["arr_0"]
    fst = np.load(save_path + f"/fst{analysis_suff}.npz")["arr_0"]

    with open(save_path + f"/lookup_dict.p", "rb") as ff:
        lookup_dict = pickle.load(ff)
    with open(save_path + f"/path_lookup.p", "rb") as ff:
        path_lookup = pickle.load(ff)

    ##Quasi-persistent filter and other info about time that binaries are bound
    bound_time_data = get_quasi(bin_ids, lookup_dict, fst, snap_interval, path_lookup)

    ##Information about initial state--energies, angles, etc. -- much of this data is not used in the final analysis
    en_data = get_energy(bin_ids, fst, lookup_dict, path_lookup)
    np.savez(save_path + f"/dat_coll{analysis_suff}.npz", bin_ids=bin_ids, fst=fst,
             ens=en_data["ens"], ens_gas=en_data["ens_gas"],
             same_sys_at_fst=en_data["same_sys_at_fst"], bin_at_fst=en_data["bin_at_fst"],
             vangs=en_data["vangs"], vangs_prim=en_data["vangs_prim"],
             mfinal_primary=en_data["mfinal_primary"],
             mfinal_pair=en_data["mfinal_pair"],
             quasi_filter=bound_time_data["quasi_filter"],
             final_bound_snaps_norm=bound_time_data["final_bound_snaps_norm"],
             final_bound_snaps=bound_time_data["final_bound_snaps"],
             init_bound_snaps=bound_time_data["init_bound_snaps"],
             bound_time=bound_time_data["bound_time"],
             bound_time_norm=bound_time_data["bound_time_norm"],
             delta_snap=bound_time_data["age_diff"],
             same_sys_final_norm=bound_time_data["same_sys_final_norm"],
             final_pair_mass_lbin=bound_time_data["final_pair_mass_lbin"],
             final_pair_mass_lsys=bound_time_data["final_pair_mass_lsys"],
             bound_time_pers=np.array(bound_time_data["bound_time_pers"], dtype=object),
             snap_interval=snap_interval)
#######################################################################################################################################################################
#######################################################################################################################################################################



if __name__ == "__main__":
    main()