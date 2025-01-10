import glob
import numpy as np
import matplotlib.pyplot as plt
##See if we can do the filtering in a cleaner/simpler way np.in1d can be a little uninituitive in some cases
import sys
import pickle

from starforge_mult_search.code import find_multiples_new2, halo_masses_single_double_par
from starforge_mult_search.code.find_multiples_new2 import cluster,system

# from find_multiples_new2 import cluster, system
import h5py
sys.path.append("/home/aleksey/code/python")
from bash_command import bash_command as bc
import cgs_const as cgs

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

def get_fate(r1, r2, row, ss):
    """
    :param string r1: Bases of pickle file name
    :param string r2: End of pickle file name
    :param list row: List (or other subscriptable) with binary elements
    :param int ss: Snapshot index

    :return: Fate of binary (specified by row) in snapshot number ss. (i) d = At least on star deleted
    (ii) s[2-4] = In system of multiplicity i (s2 means system is surviving as a binary) (iii) i = ionized
    (iv) mm = Stars are in separate multiples (v) ms = One star is in a single, while the other is in a multiple
    :rtype: string
    """
    with open(r1 + "{0:03d}".format(int(ss)) + r2, "rb") as ff:
        cl = pickle.load(ff)
    mults_a = np.array([sys1.multiplicity for sys1 in cl.systems])
    ids_a = np.array([sys1.ids for sys1 in cl.systems], dtype=object)

    try:
        idx1 = np.where(np.concatenate([np.isin([row[0]], tmp_row) for tmp_row in ids_a]))[0][0]
        idx2 = np.where(np.concatenate([np.isin([row[1]], tmp_row) for tmp_row in ids_a]))[0][0]
    except IndexError:
        return 'd'

    if idx1 == idx2:
        mult_sys = len(ids_a[idx1])
        return ('s' + str(mult_sys))
    else:
        if len(ids_a[idx1]) == 1 and len(ids_a[idx2]) == 1:
            return 'i'
        elif len(ids_a[idx1]) > 1 and len(ids_a[idx2]) > 1:
            return 'mm'
        elif (len(ids_a[idx1]) == 1 and len(ids_a[idx2]) == 2) or (len(ids_a[idx1]) == 2 and len(ids_a[idx2]) == 1):
            return 'bs'
        elif (len(ids_a[idx1]) == 1 and len(ids_a[idx2]) > 2) or (len(ids_a[idx1]) > 2 and len(ids_a[idx2]) == 1):
            return 'ms'

def get_mult_filt(bin_ids, lookup_dict, ic):
    """
    Checking to see if multiples had previously been single prior to their first appearance

    :param bin_ids Array-like: List of binary ids to check
    :param sys_lookup Dict-like: Lookup table for stellar properties
    :param ic Array-like: Lookup table for initial binary properties

    :return: Array of booleans. True=Multiple stars had previously been in multiple
    :rtype: np.ndarray
    """
    t_first = ic[:, 0]
    mults_filt = np.zeros(len(bin_ids)).astype(bool)
    for ii, row in enumerate(bin_ids):
        row_list = list(row)
        sys_lookup_sel0 = lookup_dict[(row_list[0])]
        sys_lookup_sel0 = sys_lookup_sel0[(sys_lookup_sel0[:,0] < t_first[ii])]
        sys_lookup_sel1 = lookup_dict[(row_list[1])]
        sys_lookup_sel1 = sys_lookup_sel1[(sys_lookup_sel1[:,0] < t_first[ii])]
        mults_filt[ii] = (np.all(sys_lookup_sel0[:, 3] == 1) and np.all(sys_lookup_sel1[:, 3] == 1))

    return mults_filt

def get_fst(first_snapshot_idx, uids):
    fst_idx = np.zeros(len(uids)).astype(int)
    for ii, row in enumerate(uids):
        row_li = list(row)
        for tmp_item in row_li:
            tmp_snap1 = snap_lookup(first_snapshot_idx, tmp_item)[0][-1]
            fst_idx[ii] = max(tmp_snap1, fst_idx[ii])

    return fst_idx

def get_bound_snaps(sys1_info, sys2_info):
    sys1_tag = ["{0}_{1}".format(row[LOOKUP_SNAP], row[2]) for row in sys1_info]
    sys2_tag = ["{0}_{1}".format(row[LOOKUP_SNAP], row[2]) for row in sys2_info]
    ##STARS COULD BE IN THE SAME MULTIPLE BUT NOT BOUND--HAVE TO DO FURTHER FILTERING BASED ON SMA
    ##This code block repeats frequently -- refactor into its own function...
    same_sys_filt1 = np.in1d(sys1_tag, sys2_tag)
    same_sys_filt2 = np.in1d(sys2_tag, sys1_tag)
    sys1_info = sys1_info[same_sys_filt1]
    sys2_info = sys2_info[same_sys_filt2]
    bound_filt = sys1_info[:, LOOKUP_SMA] == sys2_info[:, LOOKUP_SMA]
    bound_snaps1 = sys1_info[bound_filt]
    bound_snaps2 = sys2_info[bound_filt]

    return bound_snaps1, bound_snaps2

def get_quasi(bin_ids, lookup_dict, fst, snap_interval):
    """
    Quasi-persistent filter for binaries.
    """
    bound_time = np.zeros(len(bin_ids))
    bound_time_norm = np.zeros(len(bin_ids))
    init_bound_snaps = np.zeros(len(bin_ids))
    final_bound_snaps = np.zeros(len(bin_ids))
    final_bound_snaps_norm = np.zeros(len(bin_ids))
    mults_filt_corr = np.zeros(len(bin_ids))
    age_diff = np.zeros(len(bin_ids))

    for ii, uid in enumerate(bin_ids):
        bin_list = list(uid)
        ##Getting the final snapshot stars are bound to each other--refactor into its own function...
        sys1_info = lookup_dict[bin_list[0]]
        sys2_info = lookup_dict[bin_list[1]]
        # fst_idx = max(sys1_info[0,0], sys2_info[0, 0])
        # assert fst_idx==fst[ii]
        fst_idx = fst[ii]
        sys_lookup_sel0 = sys1_info[sys1_info[:,0] < fst_idx]
        sys_lookup_sel1 = sys2_info[sys2_info[:,0] < fst_idx]
        mults_filt_corr[ii] = (np.all(sys_lookup_sel0[:, 3] == 1) and np.all(sys_lookup_sel1[:, 3] == 1))
        age_diff[ii] = np.abs(sys1_info[0,0] - sys2_info[0,0]) * snap_interval[0]

        bound_snaps1, bound_snaps2 = get_bound_snaps(sys1_info, sys2_info)
        tmp_final_bound_snap = bound_snaps2[:, LOOKUP_SNAP][-1]
        final_bound_snaps[ii] = tmp_final_bound_snap
        final_bound_snaps_norm[ii] = tmp_final_bound_snap / bound_snaps1[0, -1]
        init_bound_snaps[ii] = bound_snaps1[:, LOOKUP_SNAP][0]

        bound_time_pers = (bound_snaps1[:, LOOKUP_SMA] * cgs.pc / cgs.au) ** 1.5 / (bound_snaps1[:, LOOKUP_MTOT] + bound_snaps2[:, LOOKUP_MTOT]) ** .5
        bound_time[ii] = len(bound_snaps1)
        bound_time_norm[ii] = np.sum(snap_interval / bound_time_pers)

    return {"quasi_filter": (bound_time_norm >= 1) & (bound_time > 1), "final_bound_snaps": final_bound_snaps,"final_bound_snaps_norm": final_bound_snaps_norm,
            "bound_time":bound_time, "bound_time_norm":bound_time_norm, "init_bound_snaps":init_bound_snaps, "mults_filt_corr": mults_filt_corr,
            "age_diff": age_diff}

def get_energy(bin_ids, fst, lookup_dict, path_lookup):
    ens = np.ones(len(bin_ids)) * np.inf
    ens_gas = np.ones(len(bin_ids)) * np.inf
    same_sys_at_fst = np.ones(len(bin_ids)) * np.inf
    bin_at_fst = np.ones(len(bin_ids)) * np.inf
    vangs = np.ones(len(bin_ids)) * np.inf
    vangs_prim = np.ones(len(bin_ids)) * np.inf

    for ii, uid in enumerate(bin_ids):
        fst_idx = fst[ii]
        bin_list = list(uid)
        path1 = path_lookup[f"{bin_list[0]}"]
        path2 = path_lookup[f"{bin_list[1]}"]

        look1 = lookup_dict[bin_list[0]]
        look1 = look1[look1[:, LOOKUP_SNAP]==fst_idx]
        look2 = lookup_dict[bin_list[1]]
        look2 = look2[look2[:, LOOKUP_SNAP]==fst_idx]

        ##Could also be true if the pair is in the sam multiple system(!)
        same_sys_at_fst[ii] = (look1[:,2] == look2[:, 2])[0]
        bin_at_fst[ii] = ((look1[:,2] == look2[:, 2]) and (look1[:, LOOKUP_SMA]==look2[:, LOOKUP_SMA]))[0]

        pos1 = path1[fst_idx, 2:5]
        pos2 = path2[fst_idx, 2:5]
        vel1 = path1[fst_idx, 5:8]
        vel2 = path2[fst_idx, 5:8]
        ##Corrected Indexing
        m1 = path1[fst_idx, mcol]
        m2 = path2[fst_idx, mcol]
        mtot1 = look1[0, LOOKUP_MTOT]
        mtot2 = look2[0, LOOKUP_MTOT]
        h1 = path1[fst_idx, hcol]
        h2 = path2[fst_idx, hcol]

        # tmp_orb_gas = find_multiples_new2.get_orbit(pos1, pos2, vel1, vel2, mtot1, mtot2, h1=h1, h2=h2)
        # tmp_orb = find_multiples_new2.get_orbit(pos1, pos2, vel1, vel2, m1, m2, h1=h1, h2=h2)
        tmp_en_gas = find_multiples_new2.get_energy(pos1, pos2, vel1, vel2, mtot1, mtot2, h1=h1, h2=h2)
        tmp_en = find_multiples_new2.get_energy(pos1, pos2, vel1, vel2, m1, m2, h1=h1, h2=h2)
        ens_gas[ii] = tmp_en_gas[0] / tmp_en_gas[1]
        ens[ii] = tmp_en[0] / tmp_en[1]
        vangs[ii] = np.dot(vel1, vel2) / np.linalg.norm(vel1) / np.linalg.norm(vel2)
        vangs_prim[ii] = np.dot(vel1 - vel2, pos1 - pos2) / np.linalg.norm(vel1 - vel2) / np.linalg.norm(pos1 - pos2)

    return {"ens": ens, "ens_gas":ens_gas, "same_sys_at_fst":same_sys_at_fst, "bin_at_fst": bin_at_fst,
            "vangs": vangs, "vangs_prim": vangs_prim}

def get_exchange_filter(bin_ids, bound_time_data):
    quasi_filter = bound_time_data["quasi_filter"]
    tmp_bins = np.array([list(row) for row in bin_ids[(quasi_filter)]]).ravel()
    my_counts = np.zeros(len(bin_ids))

    # for ii, uid in enumerate(bin_ids[(quasi_filter) & (final_bound_snaps_norm==1)]):
    for ii, uid in enumerate(bin_ids):
        bin_list = list(uid)
        tmp_count1 = len(np.where(tmp_bins == bin_list[0])[0])
        tmp_count2 = len(np.where(tmp_bins == bin_list[1])[0])

        my_counts[ii] = max(tmp_count1, tmp_count2)

    exchange_filt_b = my_counts > 1
    return exchange_filt_b

def main():
    cloud_tag = sys.argv[1]
    sim_tag = f"{cloud_tag}_{sys.argv[2]}"
    cloud_tag_split = cloud_tag.split("_")
    cloud_tag0 = f"{cloud_tag_split[0]}_{cloud_tag_split[1]}"
    base = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/{0}/{1}/".format(cloud_tag0, sim_tag)
    r1 = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/{0}/{1}/M2e4_snapshot_".format(cloud_tag0, sim_tag)
    r2 = sys.argv[3]
    base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
    snap_interval = np.atleast_1d(np.genfromtxt(base + "/sinkprop/snap_interval")).astype(float)
    r2_nosuff = r2.replace(".p", "")
    snaps = [xx.replace(base_sink, "").replace(".sink", "") for xx in glob.glob(base_sink + "*.sink")]
    snaps = np.array(snaps).astype(int)

    ##Get snapshot numbers automatically
    start_snap_sink = min(snaps)
    start_snap = min(snaps)
    end_snap = max(snaps)
    print(end_snap)
    aa = "analyze_multiples_output_{0}/".format(r2_nosuff)
    save_path = f"{cloud_tag0}/{sim_tag}/{aa}"
    ####################################################################################################
    #################################################################################

    bin_ids = np.load(save_path + "/unique_bin_ids.npz", allow_pickle=True)["arr_0"]
    ic = np.load(save_path + "/unique_bin_ids.npz")["arr_1"]
    fst = np.load(save_path + "/fst.npz")["arr_0"]

    with open(save_path + f"/lookup_dict.p", "rb") as ff:
        lookup_dict = pickle.load(ff)
    with open(save_path + f"/path_lookup.p", "rb") as ff:
        path_lookup = pickle.load(ff)

    ####This will be part 2???
    ##Quasi-persistent filter
    bound_time_data = get_quasi(bin_ids, lookup_dict, fst, snap_interval)
    # np.savez(save_path + "/quasi", quasi_filter)

    ##Multiplcity filter 1
    # np.savez(save_path + "/mults_filt")
    mults_filt = get_mult_filt(bin_ids, lookup_dict, ic)

    ##Binary fates
    # fates = [get_fate(r1, r2, list(row), end_snap) for row in bin_ids]
    fates = []

    ##Exchange filter
    exchange_filt_b = get_exchange_filter(bin_ids, bound_time_data)

    ##Energies/Angles...
    en_data = get_energy(bin_ids, fst, lookup_dict, path_lookup)
    np.savez(save_path + "/dat_coll.npz", bin_ids=bin_ids, fst=fst,
             ens=en_data["ens"], ens_gas=en_data["ens_gas"],
             same_sys_at_fst=en_data["same_sys_at_fst"], bin_at_fst=en_data["bin_at_fst"],
             vangs=en_data["vangs"], vangs_prim=en_data["vangs_prim"],
             quasi_filter=bound_time_data["quasi_filter"],
             final_bound_snaps_norm=bound_time_data["final_bound_snaps_norm"],
             final_bound_snaps=bound_time_data["final_bound_snaps"],
             init_bound_snaps=bound_time_data["init_bound_snaps"],
             mults_filt_corr=bound_time_data["mults_filt_corr"],
             bound_time=bound_time_data["bound_time"],
             bound_time_norm=bound_time_data["bound_time_norm"],
             delta_snap=bound_time_data["age_diff"],
             mults_filt=mults_filt, fates=fates, exchange_filt_b=exchange_filt_b,
             snap_interval=snap_interval)
#######################################################################################################################################################################
#######################################################################################################################################################################



if __name__ == "__main__":
    main()