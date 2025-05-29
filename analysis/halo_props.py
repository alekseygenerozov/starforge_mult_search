import pickle
import numpy as np
import sys
##Code uses functionality in find_multiples_new2
# sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
import pytreegrav
import argparse
import h5py
import multiprocessing
import functools
import time
import subprocess

from starforge_mult_search.code import find_multiples_new2

def bash_command(cmd, **kwargs):
    '''Run command from the bash shell'''
    process = subprocess.Popen(['/bin/bash', '-c', cmd],  **kwargs)
    return process.communicate()[0]
def main():
    print("test")
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("snap", help="Index of snapshot to read")
    parser.add_argument("--snap_base", default="snapshot", help="First part of snapshot name")
    parser.add_argument("--non_pair", action="store_true", help="Flag to turn on non-pairwise algorithm")
    parser.add_argument("--compress", action="store_true", help="Filter out compressive tidal forces")
    parser.add_argument("--tides_factor", type=float, default=8.0, help="Prefactor for check of tidal criterion (8.0)")
    parser.add_argument("--cutoff", type=float, default=0.5, help="Outer cutoff to look for bound gas (0.5 pc)")
    parser.add_argument("--name_tag", default="M2e4", help="Extension for saving.")


    args = parser.parse_args()
    print(args)

    snap_idx = args.snap
    cutoff = args.cutoff
    non_pair = args.non_pair
    name_tag = args.name_tag

    snapshot_file = args.snap_base + '_{0:03d}.hdf5'.format(int(args.snap))
    snapshot_num = f'{int(args.snap):03d}'

    den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base =\
    find_multiples_new2.load_data(snapshot_file, res_limit=1e-3)
    halo_ids = find_multiples_new2.load_gas_ids(snapshot_file, res_limit=1e-3)

    xuniq, indx = np.unique(x, return_index=True, axis=0)
    muniq = m[indx]
    huniq = h[indx]
    vuniq = v[indx]
    uuniq = u[indx]
    denuniq = den[indx]
    vuniq = vuniq.astype(np.float64)
    xuniq = xuniq.astype(np.float64)
    muniq = muniq.astype(np.float64)
    huniq = huniq.astype(np.float64)
    uuniq = uuniq.astype(np.float64)
    denuniq = denuniq.astype(np.float64)
    partpos = partpos.astype(np.float64)
    partmasses = partmasses.astype(np.float64)
    partsink = partsink.astype(np.float64)
    halo_ids = halo_ids[indx]

    halo_mass_name = "halo_masses_sing_np{0}_c{1}_{2}_comp{3}_tf{4}".format(non_pair, cutoff, snap_idx, args.compress,
                                                                               args.tides_factor)
    with h5py.File(halo_mass_name + ".hdf5", 'a') as gas_dat_h5:
        for ii in range(len(partids)):
            halo_idx = gas_dat_h5["halo_{0}".format(partids[ii])]
            gas_dat_h5.create_dataset("halo_{0}_pid".format(partids[ii]), data=halo_ids[halo_idx])
            # gas_dat_h5.create_dataset("halo_{0}_h".format(partids[ii]), data=huniq[halo_idx])
            # gas_dat_h5.create_dataset("halo_{0}_rho".format(partids[ii]), data=denuniq[halo_idx])
            # gas_dat_h5.create_dataset("halo_{0}_x".format(partids[ii]), data=xuniq[halo_idx])
            # gas_dat_h5.create_dataset("halo_{0}_v".format(partids[ii]), data=vuniq[halo_idx])
            # gas_dat_h5.create_dataset("halo_{0}_u".format(partids[ii]), data=uuniq[halo_idx])
            # gas_dat_h5.create_dataset("halo_{0}_m".format(partids[ii]), data=muniq[halo_idx])

if __name__ == "__main__":
    main()