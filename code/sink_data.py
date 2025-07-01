import numpy as np
import argparse

from starforge_mult_search.code.find_multiples_new2 import load_data

def main():
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("snap", help="Name of snapshot to read")
    parser.add_argument("--snap_base", default="snapshot", help="First part of snapshot name")
    parser.add_argument("--name_tag", default="M2e4", help="Extension for saving.")
    parser.add_argument("--star_age_key", default="ProtoStellarAge", help="Key for stellar age")
    args = parser.parse_args()

    star_age_key = args.star_age_key
    snapshot_file = args.snap_base + '_{0:03d}.hdf5'.format(int(args.snap))
    name_tag = args.name_tag
    snapshot_num = f'{int(args.snap):03d}'

    den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base, partspin = load_data(snapshot_file, star_age_key=star_age_key)
    if len(partpos)==0:
        print("No particles!")
        return

    nsinks = len(partpos)
    partids.shape = (nsinks, -1)
    partsink.shape = (nsinks, -1)
    partmasses.shape = (nsinks, -1)

    np.savetxt(name_tag+"_snapshot_"+snapshot_num+".sink" , np.hstack((partids, partpos, partvels, partsink, partmasses)))
    np.savetxt(name_tag+"_snapshot_"+snapshot_num+".spin", partspin)
    np.savetxt(name_tag+"_snapshot_"+snapshot_num+".age", tage_myr)

if __name__ == "__main__":
    main()
