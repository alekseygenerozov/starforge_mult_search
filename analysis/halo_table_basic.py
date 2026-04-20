import glob
import pickle
import re
import sys

import h5py
import numpy as np
import pandas as pd
import tqdm

from starforge_mult_search.analysis.analyze_stack import (
    get_first_snap_table,
    get_star_map_bins,
)
from starforge_mult_search.analysis.high_multiples_analysis import (
    apply_persistence_filter,
    get_maximal_multiples,
)


def get_companions(pid, high_df):
    comps = np.array([pid])
    if pid in high_df.index.get_level_values(level="mult_ids_list"):
        comps = np.unique(
            np.concatenate(
                high_df.xs(int(pid), level="mult_ids_list")[
                    "mult_ids_list_og"
                ].to_numpy()
            )
        )
    return comps


my_ft = sys.argv[1]
tag = sys.argv[2]
my_pattern = f"halo_masses_{tag}_npTrue_c0.5_*_compFalse_tf{my_ft}.hdf5"
halo_files = glob.glob(my_pattern)
halo_files = sorted(halo_files)
##Multiplicity data--TO DO: REMOVE THE HARD-CODING HERE (_FLAT and _SEG). FLAT DOES NOT MATTER [?]-- DOUBLE CHECK THROUGH ASSERTION(!)
high_df = pd.read_parquet(f"{sys.argv[3]}/mults_flat.pq")
high_df_filt = apply_persistence_filter(high_df, "_seg")
high_df_filt_max = get_maximal_multiples(high_df_filt)
star_map_closest_all = get_star_map_bins(high_df_filt)
star_map_idx = star_map_closest_all.index.get_level_values(
    level="mult_ids_list"
).unique()

with open(f"{sys.argv[3]}/path_lookup.p", "rb") as ff:
    path_lookup = pickle.load(ff)
first_snap_table = get_first_snap_table(path_lookup)


###Binary companion lookup
################################################################
##TO DO: SIMPLIFY
mlist = star_map_closest_all.index
blabel = np.array(star_map_closest_all["bin_halo_label"]).astype(int)
mlist = np.array([np.array(row) for row in mlist]).astype(int)
blabel_cut = blabel[blabel != mlist[:, 1]]
mlist_cut = mlist[blabel != mlist[:, 1]]
blookup1 = {tuple(mlist_cut[ii]): blabel_cut[ii] for ii in range(len(mlist_cut))}
blookup2 = {
    (mlist_cut[ii, 0], blabel_cut[ii]): mlist_cut[ii, 1] for ii in range(len(mlist_cut))
}
blookup = {**blookup1, **blookup2}
#################################################################

halo_tab_all = []
for hf in tqdm.tqdm(halo_files):
    snap = re.search("c0.5_[0-9]+_compFalse", hf).group(0)
    snap = snap.replace("c0.5_", "")
    snap = int(snap.replace("_compFalse", ""))
    halo_tab_snap = []
    with h5py.File(hf, "r") as ff:
        all_keys = list(ff.keys())
        star_ids = np.unique(
            [
                re.match("halo_[0-9]+", kk).group(0).replace("halo_", "")
                for kk in all_keys
            ]
        )
        star_ids = star_ids.astype(int)
        # [hf[f"halo_{star_id}"][:] for star_id in star_ids]
        for star_id in star_ids:
            # companions = get_companions(star_id, high_df_filt_max)
            halo_id = star_id
            comp_id = blookup.get((snap, star_id))

            if comp_id:
                halo_id = str(sorted((star_id, comp_id)))
                # companions = np.concatenate(
                #     (companions, get_companions(comp_id, high_df_filt_max))
                # )
            else:
                comp_id = star_id
            pid1 = min(comp_id, star_id)
            pid2 = max(comp_id, star_id)

            if np.sum(ff[f"halo_{star_id}"][:]) == 0:
                continue
            else:
                tmp_dat = ff[f"halo_{star_id}"][:]
                tmp_masses = ff[f"halo_{star_id}_m"][:]
                tmp_pos = ff[f"halo_{star_id}_x"][:]
                tmp_vel = ff[f"halo_{star_id}_v"][:]
                tmp_ids = ff[f"halo_{star_id}_pid"][:]
                tmp_outflow = np.ones(tmp_dat.shape) * np.inf
                if f"halo_{star_id}_outflow" in ff.keys():
                    tmp_outflow = ff[f"halo_{star_id}_outflow"][:]
                halo_tab_snap.append(
                    np.transpose(
                        (
                            np.ones(len(tmp_dat)) * int(snap),
                            np.ones(len(tmp_dat)) * pid1,
                            np.ones(len(tmp_dat)) * pid2,
                            # [companions] * len(tmp_dat),
                            tmp_ids,
                            tmp_dat,
                            tmp_masses,
                            tmp_pos[:, 0],
                            tmp_pos[:, 1],
                            tmp_pos[:, 2],
                            tmp_vel[:, 0],
                            tmp_vel[:, 1],
                            tmp_vel[:, 2],
                            tmp_outflow,
                        )
                    )
                )
    if len(halo_tab_snap) > 0:
        halo_tab_snap = pd.DataFrame(
            data=np.vstack(halo_tab_snap),
            columns=[
                "snap",
                "pid1",
                "pid2",
                "gas_id",
                "mask",
                "mass",
                "x",
                "y",
                "z",
                "vx",
                "vy",
                "vz",
                "outflow",
            ],
        )
        halo_tab_snap.set_index(["snap", "pid1", "pid2"], inplace=True)
        halo_tab_all.append(halo_tab_snap)
halo_tab_all = pd.concat(halo_tab_all)
halo_tab_all.sort_index(level=[0, 1], inplace=True)
halo_tab_all.to_parquet(f"halo_table_{tag}_{my_ft}_new.pq")
halo_tab_all.to_parquet(f"halo_table_{tag}_{my_ft}_new.pq")
