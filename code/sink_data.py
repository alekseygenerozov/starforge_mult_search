import numpy as np
import h5py
from itertools import combinations
import pickle
import argparse

import warnings


def load_data(file, res_limit=0.0):
    """ file - h5pdf5 STARFORGE snapshot
        res_limit - minimum mass resolution to include in analyis (in code units)
    """
    # Load snapshot data
    f = h5py.File(file, 'r')

    # Mask to remove any cells with mass below the cell resolution
    # (implemented specifically to remove feedback cells if desired)
    mask = (f['PartType0']['Masses'][:] >= res_limit * 0.999)
    mask3d = np.array([mask, mask, mask]).T

    # Read in gas properties
    # Mass density
    den = f['PartType0']['Density'][:] * mask
    # Spatial positions
    x = f['PartType0']['Coordinates'] * mask3d

    # Mass of each cell/partical
    m = f['PartType0']['Masses'][:] * mask
    # Calculation smoothing length, useful for weighting and/or visualization
    h = f['PartType0']['SmoothingLength'][:] * mask
    # Internal (thermal) energy
    u = f['PartType0']['InternalEnergy'][:] * mask
    v = f['PartType0']['Velocities'] * mask3d
    b = f['PartType0']['MagneticField'][:] * mask3d
    #t = f['PartType0']['Temperature'][:] * mask
    # Fraction of molecular material in each cell
    fmol = f['PartType0']['MolecularMassFraction'][:] * mask
    # To get molecular gas density do: den*fmol*fneu*(1-helium_mass_fraction)/(2.0*mh), helium_mass_fraction=0.284
    fneu = f['PartType0']['NeutralHydrogenAbundance'][:] * mask

    if 'PartType5' in f.keys():
        partpos = f['PartType5']['Coordinates'][:]
        partmasses = f['PartType5']['Masses'][:]
        partvels = f['PartType5']['Velocities'][:]
        partids = f['PartType5']['ParticleIDs'][:]
        partsink = (f['PartType5']['SinkRadius'][:])
        partspin = (f['PartType5']['BH_Specific_AngMom'][:])
    else:
        partpos = []
        partmasses = [0]
        partids = []
        partvels = [0, 0, 0]
        partsink = []
        partspin = []

    time = f['Header'].attrs['Time']
    unitlen = f['Header'].attrs['UnitLength_In_CGS']
    unitmass = f['Header'].attrs['UnitMass_In_CGS']
    unitvel = f['Header'].attrs['UnitVelocity_In_CGS']
    unitb = 1e4  # f['Header'].attrs['UnitMagneticField_In_CGS'] If not defined

    unit_base = {'UnitLength': unitlen, 'UnitMass': unitmass, 'UnitVel': unitvel, 'UnitB': unitb}

    # Unit base information specifies conversion between code units and CGS
    # Example: To convert to density in units of g/cm^3 do: den*unit_base['UnitMass']/unit_base['UnitLength']**3

    tsnap_myr = time * (unit_base['UnitLength'] / unit_base['UnitVel']) / (3600.0 * 24.0 * 365.0 * 1e6)
    tstar_form_Myr = f['PartType5']['ProtoStellarAge'][...] * (unit_base['UnitLength'] / unit_base['UnitVel']) / (3600.0 * 24.0 * 365.0 * 1e6)
    tage_myr = tsnap_myr - tstar_form_Myr
    print("Snapshot time in %f Myr" % (tsnap_myr))

    del f
    return den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base, \
        partspin

def main():
    parser = argparse.ArgumentParser(description="Parse starforge snapshot, and get multiple data.")
    parser.add_argument("snap", help="Name of snapshot to read")
    parser.add_argument("--snap_base", default="snapshot", help="First part of snapshot name")
    parser.add_argument("--name_tag", default="M2e4", help="Extension for saving.")
    args = parser.parse_args()

    snapshot_file = args.snap_base + '_{0:03d}.hdf5'.format(int(args.snap))
    name_tag = args.name_tag
    snapshot_num = snapshot_file[-8:-5].replace("_", "")

    den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base, partspin\
        = load_data(args.snap)

    nsinks = len(partpos)
    partids.shape = (nsinks, -1)
    partsink.shape = (nsinks, -1)
    partmasses.shape = (nsinks, -1)

    np.savetxt(name_tag+"_snapshot_"+snapshot_num+".sink" , np.hstack((partids, partpos, partvels, partsink, partmasses)))
    np.savetxt(name_tag+"_snapshot_"+snapshot_num+".spin", partspin)
    np.savetxt(name_tag+"_snapshot_"+snapshot_num+".age", tage_myr)

if __name__ == "__main__":
    main()
