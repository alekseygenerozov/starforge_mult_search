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

from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.analysis.analyze_stack import get_fpaths, get_snap_info
from starforge_mult_search.analysis import cgs_const as cgs

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

def make_hier(hier1, orbs1, p_dict, v_dict, m_dict, flat_id=False):
    """
    Recursively builds a hierarchy tree from the input hierarchy and orbit data.

    :param hier1: Hierarchical structure (list or int) representing the system.
    :param orbs1: List of orbit parameters corresponding to the hierarchy.
    :param p_dict: Dictionary of particle positions in the system
    :param v_dict: Dictionary of particle velocities in the system
    :param m_dict: Dictionary of particle masses in the system
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
        if flat_id:
            sid = []
            removeNestings(h_copy, sid)
            sid.sort()
        node = SystemNode(
            data={"id": sid, "orbit": tmp_orb[[0, 1, 10, 11]], "pos": tmp_orb[4:7], "vel": tmp_orb[7:10],
                  "mult": len(tmp_flat)})
        # Extract the last two components from the hierarchy
        p1 = h_copy.pop()
        p2 = h_copy.pop()

        # Handle nested structures recursively
        if isinstance(p1, list):
            child1, orbs_copy = make_hier(p1, orbs_copy, p_dict, v_dict, m_dict, flat_id=flat_id)
            node.add_child(child1)
        else:
            node.add_child(
                SystemNode(data={"id": p1, "orbit": None, "pos": p_dict[p1], "vel": v_dict[p1], "mass": m_dict[p1]}))

        if isinstance(p2, list):
            child2, orbs_copy = make_hier(p2, orbs_copy, p_dict, v_dict, m_dict, flat_id=flat_id)
            node.add_child(child2)
        else:
            node.add_child(
                SystemNode(data={"id": p2, "orbit": None, "pos": p_dict[p2], "vel": v_dict[p2], "mass": m_dict[p2]}))

    return node, orbs_copy


def get_inc_trip(i1, i2, i3, tmp_path_lookup):
    """
    Get relative inclination of triple system from lookup table of positions and velocities.
    """
    p1 = tmp_path_lookup[i1][-1]
    p2 = tmp_path_lookup[i2][-1]
    p3 = tmp_path_lookup[i3][-1]
    bin_r = p1[pxcol:pzcol + 1] - p2[pxcol:pzcol + 1]
    bin_v = p1[vxcol:vzcol + 1] - p2[vxcol:vzcol + 1]

    bin_com = (p1[mcol] * p1[pxcol:vzcol + 1] + p2[mcol] * p2[pxcol:vzcol + 1]) / (p1[mcol] + p2[mcol])
    t_r = p3[pxcol:pzcol + 1] - bin_com[:3]
    t_v = p3[vxcol:vzcol + 1] - bin_com[3:]

    jhat_1 = np.cross(bin_r, bin_v)
    jhat_1 = jhat_1 / np.linalg.norm(jhat_1)
    jhat_2 = np.cross(t_r, t_v)
    jhat_2 = jhat_2 / np.linalg.norm(jhat_2)

    return np.dot(jhat_1, jhat_2)

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
        coll_full.append(tab_dat)

        add_node_to_orbit_tab_streamlined(n1.children[0], snap, coll_full, end_snap, sub_sys=True)
        add_node_to_orbit_tab_streamlined(n1.children[1], snap, coll_full, end_snap, sub_sys=True)

def get_mult(my_id):
    kk_flat=[]
    removeNestings(ast.literal_eval(my_id), kk_flat)
    return len(kk_flat)

def lookup_star_mult(my_df, star_id, target, pre_filtered=False):
    """
    Get the multiplicity and id of max multiplicity
    system, containing star_id at time target.

    """
    star_id = str(int(star_id))
    tmp_sel = my_df
    if not pre_filtered:
        tmp_sel = my_df.xs(target, level="t")
        tmp_sel = tmp_sel.loc[(tmp_sel["nbound_snaps"]>1) & (tmp_sel["frac_of_orbit"] >= 1)]
    star_in_mult = tmp_sel.index.get_level_values("id").str.contains(rf"\b{star_id}\b")
    mults_with_star = tmp_sel.loc[star_in_mult]
    if len(mults_with_star)==0:
        return star_id, 1
    tmp_mults = mults_with_star.groupby("id", sort=False).apply(lambda x: get_mult(x.name)).values

    tmp_idx = np.where(tmp_mults==np.max(tmp_mults))[0][0]
    return mults_with_star.index.get_level_values("id")[tmp_idx], tmp_mults[tmp_idx]

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

                n1, x1 = make_hier(h1, o1, p_dict, v_dict, m_dict, flat_id=flat_id)
                add_node_to_orbit_tab_streamlined(n1, snap, coll_full, end_snap, sub_sys=False)
                sidx += 1

    coll_full_df = pd.DataFrame(coll_full, columns=("id", "t", "tf", "a", "e", "p", "ss"))
    coll_full_df.set_index(["id", "t"], inplace=True)
    frac_of_orbit = coll_full_df.groupby("id", group_keys=True).apply(lambda x: np.sum(snap_interval / x["p"])).rename("frac_of_orbit")
    nbound_snaps = coll_full_df.groupby("id", group_keys=True).apply(lambda x: len(x)).rename("nbound_snaps")
    coll_full_df_life = coll_full_df.join(frac_of_orbit, on="id")
    coll_full_df_life = coll_full_df_life.join(nbound_snaps, on="id")
    coll_full_df_life.to_parquet(save_path + f"/mults{tail_out}.pq")

    analysis_suff = "_mult"
    bin_ids = np.load(save_path + f"/unique_bin_ids{analysis_suff}.npz", allow_pickle=True)["arr_0"]
    ##Getting state of binaries at end of simulation using tabulated persistent multiples
    f1 = coll_full_df_life["frac_of_orbit"]
    n1 = coll_full_df_life["nbound_snaps"]
    tmp_sel = coll_full_df_life.loc[(f1 >= 1) & (n1 > 1)]
    end_states = []
    same_sys_filt = np.empty(len(bin_ids)).astype(bool)
    with open(save_path + f"/lookup_dict.p", "rb") as ff:
        lookup_dict = pickle.load(ff)

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

    np.savez(save_path + "/fates_corr.npz", end_states=end_states, same_sys_filt=same_sys_filt)

if __name__ == "__main__":
    main()