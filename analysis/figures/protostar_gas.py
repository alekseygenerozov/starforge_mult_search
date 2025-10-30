import argparse
import numpy as np
import pandas as pd
import subprocess

from starforge_mult_search.code import find_multiples_new2

def bash_command(cmd, **kwargs):
    '''Run command from the bash shell'''
    process = subprocess.Popen(['/bin/bash', '-c', cmd],  **kwargs)
    return process.communicate()[0]
def main():
    print("test")
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("acc_data_lookup", help="Index of snapshot to read")
    parser.add_argument("--snap_base", default="snapshot", help="First part of snapshot name")

    args = parser.parse_args()
    acc_data_lookup = pd.read_parquet(args.acc_data_lookup)
    first_times = acc_data_lookup.index.get_level_values(level="first_time").unique()

    dat_all = []
    ##Iterate over all snapshots
    for ss in first_times:
        snapshot_file = args.snap_base + '_{0:03d}.hdf5'.format(int(ss))
        den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base, partspin =\
        find_multiples_new2.load_data(snapshot_file, res_limit=1e-3)
        gas_ids = find_multiples_new2.load_gas_ids(snapshot_file, res_limit=1e-3)

        xuniq, indx = np.unique(x, return_index=True, axis=0)
        muniq = m[indx]
        huniq = h[indx]
        vuniq = v[indx]
        uuniq = u[indx]
        denuniq = den[indx]
        gas_ids = gas_ids[indx]

        vuniq = vuniq.astype(np.float64)
        xuniq = xuniq.astype(np.float64)
        muniq = muniq.astype(np.float64)
        huniq = huniq.astype(np.float64)
        uuniq = uuniq.astype(np.float64)
        denuniq = denuniq.astype(np.float64)
        # partpos = partpos.astype(np.float64)
        # partmasses = partmasses.astype(np.float64)
        # partsink = partsink.astype(np.float64)

        acc_data_lookup_select = acc_data_lookup.loc[ss]
        forming_stars = acc_data_lookup_select.index.unique()
        ##Iterate over all particles in snapshot
        for star in forming_stars:
            gas_star = acc_data_lookup_select.loc[star]
            filt = np.isin(gas_ids, gas_star)
            mfilt = muniq[filt]
            dfilt = denuniq[filt]
            reff = (3. * np.sum(mfilt / dfilt) / (4. * np.pi))**(1. / 3.)
            mtot = np.sum(mfilt)
            dat_all.append((star, reff, mtot))
        
        np.savez("dat_all.npz")

if __name__ == "__main__":
    main()