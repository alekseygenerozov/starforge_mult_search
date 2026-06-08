import glob
import os
import subprocess
import sys

import numpy as np
from omegaconf import OmegaConf

from starforge_mult_search.code.run_batch_aux import bash_command, get_cadence


# def bash_command(cmd, **kwargs):
# 	'''Run command from the bash shell'''
# 	process = subprocess.Popen(['/bin/bash', '-c', cmd],  **kwargs)
# 	return process.communicate()[0]
#
def load_config(user_config_path="halo.yaml"):
    default_config = OmegaConf.create(
        {
            "start": 0,
            "end": -1,
            "tides_factor": 8.0,
            "iter2": False,
            "iter2_config": "fig_config4.yaml",
            "extra_flags": "",
            "halo_select": "",
        }
    )

    if os.path.exists(user_config_path):
        user_config = OmegaConf.load(user_config_path)
        return OmegaConf.merge(default_config, user_config)
    return default_config


cfg = load_config()

with open("data_loc", "r") as ff:
    snap_base = ff.read()
    snap_base = snap_base.strip()
snaps = glob.glob(snap_base + "*hdf5")

cadence = get_cadence(snaps)
start = cfg.start
end = cfg.end
if end < 0:
    end = (len(snaps) - 1) * cadence

flags = f"--non_pair --tides_factor {cfg.tides_factor} {cfg.extra_flags}"
halo_snaps = range(start, end + 1, cadence)
if cfg.halo_select:
    print(cfg.halo_select)
    flags += f"--halo_select {cfg.halo_select} "
    halo_dat = np.genfromtxt(cfg.halo_select).astype(int)
    halo_dat = halo_dat[(halo_dat[:, 0] >= start) & (halo_dat[:, 0] <= end)]
    halo_snaps = np.unique(halo_dat[:, 0])

script = "starforge_mult_search/code/halo_masses_single_double_par.py"
if cfg.iter2:
    script = script.replace("single_double_par", "iter2")
    flags += f"--config_file {cfg.iter2_config}"

for ii in halo_snaps:
    my_cmd = f"python3 {script} {flags}  --snap_base {snap_base}  {ii}"
    print(my_cmd)
    bash_command(my_cmd)
