import argparse
import functools
import multiprocessing
import pickle
import subprocess
import sys
import time

import h5py
import numpy as np

##Code uses functionality in find_multiples_new2
# sys.path.append("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge")
import pytreegrav

from starforge_mult_search.code import find_multiples_new2


def bash_command(cmd, **kwargs):
    """Run command from the bash shell"""
    process = subprocess.Popen(["/bin/bash", "-c", cmd], **kwargs)
    return process.communicate()[0]


def main():
    print("test")
    parser = argparse.ArgumentParser(
        description="Parse starforge snapshot, and get multiple data."
    )
    parser.add_argument("snap", help="Index of snapshot to read")
    parser.add_argument(
        "--snap_base", default="snapshot", help="First part of snapshot name"
    )
    parser.add_argument(
        "--non_pair", action="store_true", help="Flag to turn on non-pairwise algorithm"
    )
    parser.add_argument(
        "--compress", action="store_true", help="Filter out compressive tidal forces"
    )
    parser.add_argument(
        "--tides_factor",
        type=float,
        default=8.0,
        help="Prefactor for check of tidal criterion (8.0)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=0.5,
        help="Outer cutoff to look for bound gas (0.5 pc)",
    )
    parser.add_argument("--name_tag", default="M2e4", help="Extension for saving.")
    parser.add_argument(
        "--star_age_key", default="ProtoStellarAge", help="Key for stellar age"
    )

    args = parser.parse_args()
    print(args)
    snap_idx = args.snap
    cutoff = args.cutoff
    non_pair = args.non_pair
    star_age_key = args.star_age_key

    snapshot_file = args.snap_base + "_{0:03d}.hdf5".format(int(args.snap))

    out = find_multiples_new2.load_data(
        snapshot_file, res_limit=1e-3, star_age_key=star_age_key
    )
    den = out["den"]
    x = out["x"]
    m = out["m"]
    h = out["h"]
    u = out["u"]
    v = out["v"]
    b = out["b"]
    gas_ids = out["gas_ids"]
    outflow_frac = out["outflow_frac"]
    partids = out["partids"]
    ##TO DO: REFACTOR THIS SANITIZATION TO ITS OWN FUNCTION.
    xuniq, indx = np.unique(x, return_index=True, axis=0)
    muniq = m[indx]
    huniq = h[indx]
    vuniq = v[indx]
    uuniq = u[indx]
    buniq = b[indx]
    denuniq = den[indx]
    gas_ids_uniq = gas_ids[indx]
    outflow_frac_uniq = outflow_frac[indx]

    vuniq = vuniq.astype(np.float64)
    xuniq = xuniq.astype(np.float64)
    muniq = muniq.astype(np.float64)
    huniq = huniq.astype(np.float64)
    uuniq = uuniq.astype(np.float64)
    buniq = buniq.astype(np.float64)
    denuniq = denuniq.astype(np.float64)
    outflow_frac_uniq = outflow_frac_uniq.astype(np.float64)

    halo_mass_name = "halo_masses/halo_masses_sing_np{0}_c{1}_{2}_comp{3}_tf{4}".format(
        non_pair, cutoff, snap_idx, args.compress, args.tides_factor
    )
    with h5py.File(halo_mass_name + ".hdf5", "a") as gas_dat_h5:
        for ii in range(len(partids)):
            halo_idx = gas_dat_h5["halo_{0}".format(partids[ii])]
            gas_dat_h5.require_dataset(
                "halo_{0}_b".format(partids[ii]),
                data=buniq[halo_idx],
                shape=buniq[halo_idx].shape,
                dtype=buniq[halo_idx].dtype,
            )
            gas_dat_h5.require_dataset(
                "halo_{0}_pid".format(partids[ii]),
                data=gas_ids_uniq[halo_idx],
                shape=gas_ids_uniq[halo_idx].shape,
                dtype=gas_ids_uniq[halo_idx].dtype,
            )
            gas_dat_h5.require_dataset(
                "halo_{0}_h".format(partids[ii]),
                data=huniq[halo_idx],
                shape=huniq[halo_idx].shape,
                dtype=huniq[halo_idx].dtype,
            )
            gas_dat_h5.require_dataset(
                "halo_{0}_rho".format(partids[ii]),
                data=denuniq[halo_idx],
                shape=denuniq[halo_idx].shape,
                dtype=denuniq[halo_idx].dtype,
            )
            gas_dat_h5.require_dataset(
                "halo_{0}_x".format(partids[ii]),
                data=xuniq[halo_idx],
                shape=xuniq[halo_idx].shape,
                dtype=xuniq[halo_idx].dtype,
            )
            gas_dat_h5.require_dataset(
                "halo_{0}_v".format(partids[ii]),
                data=vuniq[halo_idx],
                shape=vuniq[halo_idx].shape,
                dtype=vuniq[halo_idx].dtype,
            )
            gas_dat_h5.require_dataset(
                "halo_{0}_u".format(partids[ii]),
                data=uuniq[halo_idx],
                shape=uuniq[halo_idx].shape,
                dtype=uuniq[halo_idx].dtype,
            )
            gas_dat_h5.require_dataset(
                "halo_{0}_m".format(partids[ii]),
                data=muniq[halo_idx],
                shape=muniq[halo_idx].shape,
                dtype=muniq[halo_idx].dtype,
            )
            gas_dat_h5.require_dataset(
                "halo_{0}_outflow".format(partids[ii]),
                data=outflow_frac_uniq[halo_idx],
                shape=muniq[halo_idx].shape,
                dtype=muniq[halo_idx].dtype,
            )


if __name__ == "__main__":
    main()
