import argparse
import glob
import numpy as np
import pandas as pd
import subprocess

from starforge_mult_search.code import find_multiples_new2

def bash_command(cmd, **kwargs):
    '''Run command from the bash shell'''
    process = subprocess.Popen(['/bin/bash', '-c', cmd],  **kwargs)
    return process.communicate()[0]

def main():
    parser = argparse.ArgumentParser(description="Get repreating gas ids.")
    parser.add_argument("--snap_base", default="snapshot", help="First part of snapshot name")
    args = parser.parse_args()

    snaps = glob.glob(args.snap_base + "*hdf5")
    start = 0 
    end = len(snaps) 

    repeaters = []
    ##Iterate over all snapshots
    for ss in range(start, end):
        print(ss)
        snapshot_file = args.snap_base + '_{0:03d}.hdf5'.format(int(ss))
        ##Getting all the counts of all the ids.
        gas_ids = find_multiples_new2.load_gas_ids(snapshot_file, res_limit=1e-3)
        ##Getting the count
        ordered_count = pd.Series(gas_ids).value_counts().loc[gas_ids].to_numpy()
        gas_ids[ordered_count > 1]
        repeaters.append(gas_ids)
        np.savez("repeaters.npz", repeaters)

if __name__ == "__main__":
    main()