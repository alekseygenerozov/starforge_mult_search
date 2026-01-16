import glob
import re
import sys

import h5py
import numpy as np
import pandas as pd
import tqdm

my_ft = sys.argv[1]
tag = sys.argv[2]
my_pattern = f"halo_masses_{tag}_npTrue_c0.5_*_compFalse_tf{my_ft}.hdf5"
halo_files = glob.glob(my_pattern)
halo_files = sorted(halo_files)

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
        # [hf[f"halo_{star_id}"][:] for star_id in star_ids]
        for star_id in star_ids:
            if np.sum(ff[f"halo_{star_id}"][:]) == 0:
                continue
            else:
                tmp_dat = ff[f"halo_{star_id}"][:]
                tmp_masses = ff[f"halo_{star_id}_m"][:]
                tmp_pos = ff[f"halo_{star_id}_x"][:]
                tmp_ids = ff[f"halo_{star_id}_pid"][:]
                halo_tab_snap.append(
                    np.transpose(
                        (
                            np.ones(len(tmp_dat)) * int(snap),
                            np.ones(len(tmp_dat)) * int(star_id),
                            tmp_ids,
                            tmp_dat,
                            tmp_masses,
                            tmp_pos[:, 0],
                            tmp_pos[:, 1],
                            tmp_pos[:, 2],
                        )
                    )
                )
    halo_tab_snap = pd.DataFrame(
        data=np.vstack(halo_tab_snap),
        columns=["snap", "pid", "gas_id", "mask", "mass", "x", "y", "z"],
    )
    halo_tab_snap.set_index(["snap", "pid"], inplace=True)
    halo_tab_all.append(halo_tab_snap)
halo_tab_all = pd.concat(halo_tab_all)
halo_tab_all.to_parquet(f"halo_table_{tag}_{my_ft}.pq")
