import sys
import subprocess
import glob

from run_batch_aux import bash_command, get_cadence

# def bash_command(cmd, **kwargs):
# 	'''Run command from the bash shell'''
# 	process = subprocess.Popen(['/bin/bash', '-c', cmd],  **kwargs)
# 	return process.communicate()[0]
#

with open("data_loc", "r") as ff:
	snap_base = ff.read()
snaps = glob.glob(snap_base + "*hdf5")

cadence = get_cadence()
start = int(sys.argv[1])
end = int(sys.argv[2])
if end < 0:
	end = (len(snaps) - 1) * candence
for ii in range(start, end + 1, cadence):
	print(f"python3 halo_masses_single_double_par.py --non_pair --tides_factor {sys.argv[3]} --snap_base {snap_base}  {ii}")
	bash_command(f"python3 halo_masses_single_double_par.py --non_pair --tides_factor {sys.argv[3]} --snap_base {snap_base}  {ii}")
