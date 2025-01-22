import sys
import subprocess
import glob

import multiprocessing
from run_batch_aux import bash_command, get_cadence

# def bash_command(cmd, **kwargs):
#         '''Run command from the bash shell'''
#         process=subprocess.Popen(['/bin/bash', '-c',cmd],  **kwargs)
#         return process.communicate()[0]

def run_find_bins(ii, snap_base, tides_factor):
    cmd = f"python3 find_multiples_new2.py {ii} --halo_mass_file halo_masses/M2e4halo_masses_sing_npTrue_c0.5 --ngrid 1 --snap_base {snap_base} --tides_factor {tides_factor}"
    bash_command(cmd)

with open("data_loc", "r") as ff:
	snap_base = ff.read().strip()
snaps = glob.glob(snap_base + "*hdf5")

cadence = get_cadence()
start = int(sys.argv[1])
end = int(sys.argv[2])
if end < 0:
	end = (len(snaps) - 1) * cadence
with multiprocessing.Pool(10) as pool:
    pool.starmap(run_find_bins, [(ii, snap_base, sys.argv[3]) for ii in range(start, end + 1, cadence)])