import glob
import subprocess
import sys

import numpy as np

from starforge_mult_search.code.run_batch_aux import bash_command, get_cadence

# def bash_command(cmd, **kwargs):
# 	'''Run command from the bash shell'''
# 	process = subprocess.Popen(['/bin/bash', '-c', cmd],  **kwargs)
# 	return process.communicate()[0]
#

with open("data_loc", "r") as ff:
    snap_base = ff.read()
    snap_base = snap_base.strip()
snaps = glob.glob(snap_base + "*hdf5")

cadence = get_cadence()
start = int(sys.argv[1])
end = int(sys.argv[2])
if end < 0:
    end = (len(snaps) - 1) * cadence

halo_dat = np.genfromtxt(sys.argv[-1]).astype(int)
halo_dat = halo_dat[(halo_dat[:, 0] >= start) & (halo_dat[:, 0] <= end)]
halo_snaps = np.unique(halo_dat[:, 0])

for ii in range(halo_snaps):
    my_cmd = f"python3 starforge_mult_search/code/halo_masses_single_double_select.py --non_pair --tides_factor {sys.argv[3]}  --halo_select {sys.argv[-1]}  --snap_base {snap_base}  {ii}"
    print(my_cmd)
    bash_command(my_cmd)
