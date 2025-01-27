import sys
import subprocess
import glob

import multiprocessing
from run_batch_aux import bash_command, get_cadence

def bash_command(cmd, **kwargs):
        '''Run command from the bash shell'''
        process=subprocess.Popen(['/bin/bash', '-c',cmd],  **kwargs)
        return process.communicate()[0]

def run_sink(ii, snap_base):
        cmd = f"python3 starforge_mult_search/code/sink_data.py {ii}  --snap_base {snap_base}"
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
    pool.starmap(run_sink, [(ii, snap_base) for ii in range(start, end + 1, cadence)])