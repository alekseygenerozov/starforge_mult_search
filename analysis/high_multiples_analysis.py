import ast
import copy
import glob
import hydra
import numpy as np
import os
import pandas as pd
import pickle
import sys
import tqdm

from starforge_mult_search.code.find_multiples_new2 import cluster, system, PE, KE, get_orbit
from starforge_mult_search.analysis.analyze_stack import get_fpaths, get_snap_info, get_end_time_set, pxcol, pzcol, vxcol, vzcol, mcol, mtotcol, hcol
from starforge_mult_search.analysis import cgs_const as cgs

from bash_command import bash_command as bc

class SystemNode:
    """
    Node representing a single component of the system hierarchy.

    :param data: Dictionary storing properties (e.g., mass, position, velocity) of this node.
    :param children: List of child nodes representing substructures (orbits or particles).
    """

    def __init__(self, data, children=None):
        self.data = data  # Dictionary of properties
        self.children = children or []  # List of child nodes
        self.leaves = []

    def add_child(self, child_node):
        """Add a child node to this node."""
        self.children.append(child_node)
        # If the child node has no children, it's a leaf
        if child_node.data["orbit"] is None:
            self.leaves.append(child_node)

    def get_property(self, key):
        """Get a property at this node."""
        return self.data.get(key, None)

    def get_all_properties(self, key):
        """Recursively get a property from all nodes."""
        values = [self.data.get(key, None)]
        for child in self.children:
            values.extend(child.get_all_properties(key))
        return [v for v in values if v is not None]

    def __repr__(self, level=0):
        """Pretty-print the node and its hierarchy."""
        ret = "  " * level + repr(self.data) + "\n"
        for child in self.children:
            ret += child.__repr__(level + 1)
        return ret


def rec_sort(my_list):
    my_list_ = copy.deepcopy(my_list)
    if isinstance(my_list_, (int, np.integer)):
        return my_list_

    p1 = my_list_.pop()
    p2 = my_list_.pop()

    if isinstance(p1, (int, np.integer)) and isinstance(p2, (int, np.integer)):
        return [min(p1, p2), max(p1, p2)]
    else:
        return [rec_sort(p2), rec_sort(p1)]


def removeNestings(l, output):
    for i in l:
        if type(i) == list:
            removeNestings(i, output)
        else:
            output.append(i)

def get_energy_wrap(p1, p2, p_dict, v_dict, m_dict, h_dict):
    p1_flat = []
    p2_flat = []

    removeNestings([p1], p1_flat)
    removeNestings([p2], p2_flat)

    pos_flat1 = [p_dict[pp] for pp in p1_flat]
    v_flat1 = [v_dict[pp] for pp in p1_flat]
    m_flat1 = [m_dict[pp] for pp in p1_flat]
    h_flat1 = [h_dict[pp] for pp in p1_flat]

    pos_flat1 = np.average(pos_flat1, weights=m_flat1, axis=0)
    v_flat1 = np.average(v_flat1, weights=m_flat1, axis=0)
    h_flat1 = np.sum(h_flat1)

    pos_flat2 = [p_dict[pp] for pp in p2_flat]
    v_flat2 = [v_dict[pp] for pp in p2_flat]
    m_flat2 = [m_dict[pp] for pp in p2_flat]
    h_flat2 = [h_dict[pp] for pp in p2_flat]

    pos_flat2 = np.average(pos_flat2, weights=m_flat2, axis=0)
    v_flat2 = np.average(v_flat2, weights=m_flat2, axis=0)
    h_flat2 = np.sum(h_flat2)

    m_flat1 = np.sum(m_flat1)
    m_flat2 = np.sum(m_flat2)
    return (PE(np.array([pos_flat1, pos_flat2]), np.array([m_flat1, m_flat2]), np.array([h_flat1, h_flat2])),
            KE(np.array([pos_flat1, pos_flat2]), np.array([m_flat1, m_flat2]), np.array([v_flat1, v_flat2]), np.array([0, 0])))


def make_hier(hier1, orbs1, p_dict, v_dict, m_dict, h_dict, flat_id=False):
    """
    Recursively builds a hierarchy tree from the input hierarchy and orbit data.

    :param hier1: Hierarchical structure (list or int) representing the system.
    :param orbs1: List of orbit parameters corresponding to the hierarchy.
    :param p_dict: Dictionary of particle positions in the system
    :param v_dict: Dictionary of particle velocities in the system
    :param m_dict: Dictionary of particle masses in the system
    :param h_dict: Dictionary of particle softening lengths
    :param flat_id: Make ids hierarchy agnostic. If True then systems with the same stars, but different hierarchies
     will have the same id. False by default.

    :return: A `SystemNode` representing the root of the hierarchy tree.
    """
    import copy

    # Make deep copies to avoid modifying the original data
    h_copy = copy.deepcopy(hier1)
    orbs_copy = copy.deepcopy(orbs1)

    # Initialize the node with the current hierarchy level
    node = SystemNode(data={"id": h_copy, "orbit": None})

    if isinstance(h_copy, list):
        # Get the corresponding orbit
        tmp_orb = orbs_copy.pop()
        tmp_flat = []
        removeNestings(h_copy, tmp_flat)
        ## Can flatten and sort the hierarchy instead...
        sid = rec_sort(h_copy)
        sid_full = copy.deepcopy(sid)
        if flat_id:
            sid = []
            removeNestings(h_copy, sid)
            sid.sort()
        node = SystemNode(
            data={"id": sid, "orbit": tmp_orb[[0, 1, 10, 11]], "pos": tmp_orb[4:7], "vel": tmp_orb[7:10],
                  "mult": len(tmp_flat), "hier": sid_full})
        # Extract the last two components from the hierarchy
        p1 = h_copy.pop()
        p2 = h_copy.pop()
        tmp_pe, tmp_ke = get_energy_wrap(p1, p2, p_dict, v_dict, m_dict, h_dict)
        node.data["pe"] = tmp_pe
        node.data["ke"] = tmp_ke

        # Handle nested structures recursively
        if isinstance(p1, list):
            child1, orbs_copy = make_hier(p1, orbs_copy, p_dict, v_dict, m_dict, h_dict, flat_id=flat_id)
            node.add_child(child1)
        else:
            node.add_child(
                SystemNode(data={"id": p1, "orbit": None, "pos": p_dict[p1], "vel": v_dict[p1], "mass": m_dict[p1]}))

        if isinstance(p2, list):
            child2, orbs_copy = make_hier(p2, orbs_copy, p_dict, v_dict, m_dict, h_dict, flat_id=flat_id)
            node.add_child(child2)
        else:
            node.add_child(
                SystemNode(data={"id": p2, "orbit": None, "pos": p_dict[p2], "vel": v_dict[p2], "mass": m_dict[p2]}))

    return node, orbs_copy


def get_inc_trip(i1, i2, i3, tmp_path_lookup, snap, inc_halo=False):
    """
    Get relative inclination of triple system (with inner binary i1, i2, and outer
    tertiary i3 from lookup table of positions and velocities (tmp_path_lookup), at
    snapshot snap. The angular momentum of the tertiary is calculate with respect
    to the center of mass of the inner binary. This is by default calculated without
    the halo mass corrections, but if inc_halo=True, the halo masses are included
    in the calculation of the com...TO DO: GET OTHER ORBITAL ELEMENTS AS WELL...
    """
    my_mcol = mcol
    if inc_halo:
        my_mcol = mtotcol
    p1 = tmp_path_lookup[i1][snap]
    p2 = tmp_path_lookup[i2][snap]
    p3 = tmp_path_lookup[i3][snap]
    bin_r = p1[pxcol:pzcol + 1] - p2[pxcol:pzcol + 1]
    bin_v = p1[vxcol:vzcol + 1] - p2[vxcol:vzcol + 1]

    bin_com = (p1[my_mcol] * p1[pxcol:vzcol + 1] + p2[my_mcol] * p2[pxcol:vzcol + 1]) / (p1[my_mcol] + p2[my_mcol])
    t_r = p3[pxcol:pzcol + 1] - bin_com[:3]
    t_v = p3[vxcol:vzcol + 1] - bin_com[3:]

    jhat_1 = np.cross(bin_r, bin_v)
    jhat_1 = jhat_1 / np.linalg.norm(jhat_1)
    jhat_2 = np.cross(t_r, t_v)
    jhat_2 = jhat_2 / np.linalg.norm(jhat_2)
    orb = get_orbit(p1[pxcol:pzcol + 1], p2[pxcol:pzcol + 1], p1[vxcol:vzcol + 1], p2[vxcol:vzcol + 1], p1[my_mcol], p2[my_mcol], p1[hcol], p2[hcol])

    return {"ang": np.dot(jhat_1, jhat_2), "inner_sep":np.linalg.norm(bin_r), "outer_sep":np.linalg.norm(t_r),
             "inner_a":orb[0], "inner_e":orb[1], "inner_sep_proj":np.linalg.norm(bin_r[:-1]), "outer_sep_proj":np.linalg.norm(t_r[:-1])}

def get_q_trip(i1, i2, i3, tmp_path_lookup, snap, inc_halo=False):
    """
    Get mass ratios of triple system (with inner binary i1, i2, and outer
    tertiary i3 from lookup table of positions and velocities (tmp_path_lookup), at
    snapshot snap. The angular momentum of the tertiary is calculate with respect
    to the center of mass of the inner binary. This is by default calculated without
    the halo mass corrections, but if inc_halo=True, the halo masses are included
    in the calculation of the com...
    """
    my_mcol = mcol
    if inc_halo:
        my_mcol = mtotcol
    p1 = tmp_path_lookup[i1][snap]
    p2 = tmp_path_lookup[i2][snap]
    p3 = tmp_path_lookup[i3][snap]

    ##Mass ratio of inner binary: min / max < 1 by definition
    m1 = max(p1[my_mcol], p2[my_mcol])
    m2 = min(p1[my_mcol], p2[my_mcol])
    q1 = m2 / m1
    ##Tertiary / Inner binary.
    q2 = p3[my_mcol] / (p1[my_mcol] + p2[my_mcol])

    return {"q1":q1, "q2":q2, "m1":m1, "m2":m2, "m3":p3[my_mcol]}

def add_node_to_orbit_tab_streamlined(n1, snap, coll_full, end_snap, sub_sys=False):
    if n1.data["orbit"] is None:
        return
    else:
        tab_dat = []
        tab_dat.append(str(n1.data["id"]))
        tab_dat.append(snap)
        tab_dat.append(end_snap)

        tmp_orb = n1.data["orbit"]
        ##Data for outer orbit
        tab_dat.append(tmp_orb[0])
        tab_dat.append(tmp_orb[1])

        tmp_per = (tmp_orb[0] * cgs.pc / cgs.au) ** 1.5 / (tmp_orb[2] + tmp_orb[3]) ** .5
        tab_dat.append(tmp_per)
        tab_dat.append(sub_sys)
        tab_dat.append(str(n1.data["hier"]))
        tab_dat.append(n1.data["pe"])
        tab_dat.append(n1.data["ke"])
        tab_dat.append(tmp_orb[2])
        tab_dat.append(tmp_orb[3])
        coll_full.append(tab_dat)

        add_node_to_orbit_tab_streamlined(n1.children[0], snap, coll_full, end_snap, sub_sys=True)
        add_node_to_orbit_tab_streamlined(n1.children[1], snap, coll_full, end_snap, sub_sys=True)

def get_mult(my_id):
    kk_flat=[]
    removeNestings(ast.literal_eval(my_id), kk_flat)
    return len(kk_flat)

def lookup_star_mult(my_df, star_id, target, pre_filtered=False, contig_suff=""):
    """
    Get the multiplicity and id of max multiplicity
    system, containing star_id at time target.

    """
    star_id = str(int(star_id))
    tmp_sel = my_df
    if not pre_filtered:
        tmp_sel = my_df.xs(target, level="t")
        tmp_sel = tmp_sel.loc[(tmp_sel[f"nbound_snaps{contig_suff}"]>1) & (tmp_sel[f"frac_of_orbit{contig_suff}"] >= 1)]
    star_in_mult = tmp_sel.index.get_level_values("id").str.contains(rf"\b{star_id}\b")
    mults_with_star = tmp_sel.loc[star_in_mult]
    if len(mults_with_star)==0:
        return star_id, 1
    tmp_mults = mults_with_star.groupby("id", sort=False).apply(lambda x: get_mult(x.name)).values

    tmp_idx = np.where(tmp_mults==np.max(tmp_mults))[0][0]
    return mults_with_star.index.get_level_values("id")[tmp_idx], tmp_mults[tmp_idx]

def lookup_star_mult_with_mass(my_df, star_id, target, path_lookup, pre_filtered=False, contig_suff=""):
    """
    Get the multiplicity and id of max multiplicity
    system, containing star_id at time target.

    """
    star_id = str(int(star_id))
    tmp_sel = my_df
    if not pre_filtered:
        tmp_sel = my_df.xs(target, level="t")
        tmp_sel = tmp_sel.loc[(tmp_sel[f"nbound_snaps{contig_suff}"]>1) & (tmp_sel[f"frac_of_orbit{contig_suff}"] >= 1)]
    star_in_mult = tmp_sel.index.get_level_values("id").str.contains(rf"\b{star_id}\b")
    mults_with_star = tmp_sel.loc[star_in_mult]
    if len(mults_with_star)==0:
        return star_id, 1, np.array([path_lookup[star_id][target, mcol]])
    tmp_mults = mults_with_star.groupby("id", sort=False).apply(lambda x: get_mult(x.name)).values
    tmp_idx = np.where(tmp_mults==np.max(tmp_mults))[0][0]
    host_sys = mults_with_star.iloc[tmp_idx]
    tmp_mult, tmp_time = host_sys["mult_ids_list"], target
    tmp_masses = [path_lookup[str(uu)][tmp_time, mcol] for uu in tmp_mult]

    return host_sys.name, tmp_mults[tmp_idx], np.array(tmp_masses)

def lookup_star_mult_b(my_df, star_id):
    """
    Get multiple of star with id star_id from DataFrame my_df
    """
    mult = 1
    for row in my_df.iterrows():
        if (row[1]["mult"] > mult) and (star_id in row[1]["mult_ids_list"]):
            mult = row[1]["mult"]
        if mult==4:
            break
    
    return mult

def filter_maximal_sets(sets):
    result = []
    for s in sets:
        if not any(s < other for other in sets):  # proper subset
            result.append(True)
        else:
            result.append(False)
    return result

def get_pair_state(my_df, id1, id2, target, **kwargs):
    """
    Get multiplicity of stars id1 and id2 from dataframe my_df at time target. Also, find out
    if the stars are in the same system.
    """
    s1, m1 = lookup_star_mult(my_df, id1, target, **kwargs)
    s2, m2 = lookup_star_mult(my_df, id2, target, **kwargs)

    return (f"{min(m1, m2)} {max(m1, m2)}"), s1==s2

def parse_mult_id(id_str):
    return set(map(int, id_str.replace("[", "").replace("]", "").split(",")))

def parse_mult_id_list(id_str):
    return list(map(int, id_str.replace("[", "").replace("]", "").split(",")))

def subset_count(ids1, ids):
    subsets = []
    for row in ids:
        subsets.append(np.all([ii in row for ii in ids1]))
    subsets = np.array(subsets)

    return len(subsets[subsets])

def filter_top_level(my_df):
    """
    Get only the top level of the multiples
    """
    mult_ids = my_df.index.get_level_values("id")
    mult_ids_set = mult_ids.to_series().apply(parse_mult_id)
    my_counts = np.array([subset_count(row, mult_ids_set.to_list()) for row in mult_ids_set.to_list()])

    return my_df.loc[my_counts==1]

def assign_contiguous_segments(group, cadence=1):
    diffs = np.diff(group.index.get_level_values("t"))
    segment_ids = np.zeros(len(group), dtype=int)
    segment_ids[1:] = np.cumsum(diffs != cadence)
    return pd.Series(segment_ids, index=group.index, name="segment")

@hydra.main(version_base=None, config_path=os.getcwd(), config_name="config")
def main(params):
    base, base_sink, r1, r2, cloud_tag0, sim_tag = get_fpaths(params["base_path"], params["cloud_tag"], params["seed"], params["analysis_tag"], v_str=params["v_str"])
    r2_nosuff = r2.replace(".p", "")
    v_str = params["v_str"]
    cadence, snap_interval, start_snap, end_snap = get_snap_info(base, base_sink)
    flat_id = False
    tail_out = ""
    if ("flat_id" in params):
        flat_id = params["flat_id"]
        tail_out = "_flat"


    coll_full = []
    aa = "analyze_multiples_output_{0}/".format(r2_nosuff)
    save_path = f"{v_str}/{cloud_tag0}/{sim_tag}/{aa}"
    os.makedirs(save_path, exist_ok=True)
    with open(save_path + f"/path_lookup.p", "rb") as ff:
        path_lookup = pickle.load(ff)

    for snap in range(start_snap, end_snap + 1, cadence):
        with open(
                f"{r1}{snap:03d}{r2}", "rb") as ff:
            cl = pickle.load(ff)
        sidx = 0
        for ss in cl.systems:
            if ss.multiplicity >= 2:
                h1, o1 = list(ss.hierarchy), list(ss.orbits)
                p_dict = {ss.ids[ii]: ss.sub_pos[ii] for ii in range(len(ss.ids))}
                v_dict = {ss.ids[ii]: ss.sub_vel[ii] for ii in range(len(ss.ids))}
                m_dict = {ss.ids[ii]: ss.sub_mass[ii] for ii in range(len(ss.ids))}
                h_dict = {ss.ids[ii]: ss.sub_soft[ii] for ii in range(len(ss.ids))}


                n1, x1 = make_hier(h1, o1, p_dict, v_dict, m_dict, h_dict, flat_id=flat_id)
                ##Could we simply add the full node to the table??
                add_node_to_orbit_tab_streamlined(n1, snap, coll_full, end_snap, sub_sys=False)
                sidx += 1

    coll_full_df = pd.DataFrame(coll_full, columns=("id", "t", "tf", "a", "e", "p", "ss", "hier", "pe", "ke", "m1", "m2"))
    coll_full_df.set_index(["id", "t"], inplace=True)
    ##TO DO: Try to homogenize this code...##group_keys is true by default, so it may be unnecessary.
    frac_of_orbit = coll_full_df.groupby("id", group_keys=True).apply(lambda x: np.sum(snap_interval / x["p"])).rename("frac_of_orbit")
    nbound_snaps = coll_full_df.groupby("id", group_keys=True).apply(lambda x: len(x)).rename("nbound_snaps")
    coll_full_df_life = coll_full_df.join(frac_of_orbit, on="id")
    coll_full_df_life = coll_full_df_life.join(nbound_snaps, on="id")
    ##Getting cumulative number of snapshots and orbits
    tmp1 = coll_full_df_life.groupby("id", group_keys=True)[["tf"]].transform(lambda x: list(range(len(x))))
    tmp2 = coll_full_df_life.groupby("id")[["p"]].transform(lambda x: (snap_interval / x).cumsum())
    coll_full_df_life = pd.merge(coll_full_df_life, tmp1, left_index=True, right_index=True)
    coll_full_df_life = pd.merge(coll_full_df_life, tmp2, left_index=True, right_index=True)

    coll_full_df_life.rename(columns={"p_x": "p", "tf_x": "tf", "p_y": "cumul_frac", "tf_y": "cumul_snaps"}, inplace=True)
    ##Get orbits and bound snapshots by segments...
    coll_full_df_life["segment"] = coll_full_df_life.groupby("id", group_keys=False).apply(lambda x: assign_contiguous_segments(x, cadence=cadence))
    frac_of_orbit = coll_full_df_life.groupby(["id", "segment"]).apply(lambda x: np.sum(snap_interval / x["p"])).rename("frac_of_orbit_seg")
    nbound_snaps = coll_full_df_life.groupby(["id", "segment"]).apply(lambda x: len(x)).rename("nbound_snaps_seg")
    coll_full_df_life = coll_full_df_life.join(frac_of_orbit, on=["id", "segment"])
    coll_full_df_life = coll_full_df_life.join(nbound_snaps, on=["id", "segment"])

    ##Convenience columns....e.g. Multiplicity
    mult_hiers = coll_full_df_life["hier"]
    mult_ids_list = mult_hiers.apply(parse_mult_id_list)
    coll_full_df_life["mult_ids_list"] = mult_ids_list
    coll_full_df_life["mult"] = coll_full_df_life["mult_ids_list"].apply(lambda ss: len(ss))
    ##Getting end times for all stars, and final primary mass for multiple.
    ##TO DO: Also store version with halo mass.
    coll_full_df_life[["end_stars", "mult_prim_final"]] = coll_full_df_life["mult_ids_list"].apply(lambda ss: pd.Series(get_end_time_set(ss, path_lookup)))
    ##Write out dataframe with the higher order multiples.
    coll_full_df_life.to_parquet(save_path + f"/mults{tail_out}.pq")

    analysis_suff = "_mult"
    ##Maybe we should do both versions here -- with and without segment...
    bin_ids = np.load(save_path + f"/unique_bin_ids{analysis_suff}.npz", allow_pickle=True)["arr_0"]
    my_data = np.load(save_path + f"/dat_coll{analysis_suff}.npz", allow_pickle=True)

    with open(save_path + f"/lookup_dict.p", "rb") as ff:
        lookup_dict = pickle.load(ff)
    ##Getting state of binaries at end of simulation using tabulated persistent multiples
    f1 = coll_full_df_life["frac_of_orbit_seg"]
    n1 = coll_full_df_life["nbound_snaps_seg"]
    tmp_sel = coll_full_df_life.loc[(f1 >= 1) & (n1 > 1)]
    end_states = []
    same_sys_filt = np.empty(len(bin_ids)).astype(bool)

    for ii, row in tqdm.tqdm(enumerate(bin_ids)):
        bin_list = list(row)
        id1 = bin_list[0]
        id2 = bin_list[1]
        end_time1 = lookup_dict[id1][-1, 0]
        end_time2 = lookup_dict[id2][-1, 0]

        end_time = min(end_time1, end_time2)
        es, ss = get_pair_state(tmp_sel.xs(end_time, level="t"), id1, id2, end_time, pre_filtered=True)
        end_states.append(es)
        same_sys_filt[ii] = ss

    np.savez(save_path + f"/fates_corr{tail_out}_seg.npz", end_states=end_states, same_sys_filt=same_sys_filt)

    bin_list = tmp_sel[tmp_sel["mult"]==2]
    bin_set = set(bin_list.index.get_level_values(level="id"))
    quasi_filter_contig = np.zeros(len(bin_ids)).astype(bool)

    ##Adding contiguous persistence filter for binaries.
    for ii, row in enumerate(bin_ids):
        tmp_id = list(row)
        tmp_id.sort()
        quasi_filter_contig[ii] = (str(tmp_id) in bin_set)
    my_data = dict(my_data)
    my_data["quasi_filter_seg"] = quasi_filter_contig
    bc.bash_command("cp " + save_path + f"/dat_coll{analysis_suff}.npz " + save_path + f"/dat_coll{analysis_suff}_bk.npz")
    np.savez(save_path + f"/dat_coll{analysis_suff}.npz", **my_data)
    ######################################################################################################
    f1 = coll_full_df_life["frac_of_orbit"]
    n1 = coll_full_df_life["nbound_snaps"]
    tmp_sel = coll_full_df_life.loc[(f1 >= 1) & (n1 > 1)]
    end_states = []
    same_sys_filt = np.empty(len(bin_ids)).astype(bool)

    for ii, row in tqdm.tqdm(enumerate(bin_ids)):
        bin_list = list(row)
        id1 = bin_list[0]
        id2 = bin_list[1]
        end_time1 = lookup_dict[id1][-1, 0]
        end_time2 = lookup_dict[id2][-1, 0]

        end_time = min(end_time1, end_time2)
        es, ss = get_pair_state(tmp_sel.xs(end_time, level="t"), id1, id2, end_time, pre_filtered=True)
        end_states.append(es)
        same_sys_filt[ii] = ss

    np.savez(save_path + f"/fates_corr{tail_out}.npz", end_states=end_states, same_sys_filt=same_sys_filt)


if __name__ == "__main__":
    main()