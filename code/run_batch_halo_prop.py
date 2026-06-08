import glob
import multiprocessing
import os
import subprocess
import sys

from run_batch_aux import bash_command, get_cadence


def bash_command(cmd, **kwargs):
    """Run command from the bash shell"""
    process = subprocess.Popen(["/bin/bash", "-c", cmd], **kwargs)
    return process.communicate()[0]


def run_hp(ii, snap_base):
    cmd = f"python3 ../starforge_mult_search/analysis/halo_props.py {ii}  --snap_base {snap_base} --tides_factor {sys.argv[3]}"
    bash_command(cmd)


with open("data_loc", "r") as ff:
    snap_base = ff.read().strip()
snaps = glob.glob(snap_base + "*hdf5")

os.chdir("halo_masses/")
cadence = get_cadence(snaps)
start = int(sys.argv[1])
end = int(sys.argv[2])
if end < 0:
    end = (len(snaps) - 1) * cadence
with multiprocessing.Pool(10) as pool:
    pool.starmap(run_hp, [(ii, snap_base) for ii in range(start, end + 1, cadence)])
