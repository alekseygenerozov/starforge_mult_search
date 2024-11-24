import sys
import subprocess
import glob

def bash_command(cmd, **kwargs):
        '''Run command from the bash shell'''
        process=subprocess.Popen(['/bin/bash', '-c',cmd],  **kwargs)
        return process.communicate()[0]


with open("data_loc", "r") as ff:
	snap_base = ff.read().strip()
snaps = glob.glob(snap_base + "*hdf5")

start = int(sys.argv[1])
end = int(sys.argv[2])
if end < 0:
	end = len(snaps) - 1
for ii in range(start, end + 1, 1):
        cmd = f"python3 find_multiples_new2.py {ii} --halo_mass_file halo_masses/M2e4halo_masses_sing_npTrue_c0.5 --ngrid 1 --snap_base {snap_base} --tides_factor {sys.argv[3]}"
        print(cmd)
        bash_command(cmd)
