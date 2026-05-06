import argparse
import numpy as np
import pandas as pd
import subprocess

from starforge_mult_search.code import find_multiples_new2
import h5py
import sys
import pickle


def load_data(file, res_limit=0.0, star_age_key="ProtoStellarAge"):
    """file - h5pdf5 STARFORGE snapshot
    res_limit - minimum mass resolution to include in analyis (in code units)
    """
    # Load snapshot data
    f = h5py.File(file, "r")

    # Mask to remove any cells with mass below the cell resolution
    # (implemented specifically to remove feedback cells if desired)
    mask = f["PartType0"]["Masses"][:] >= res_limit * 0.999

    # Read in gas properties
    # Mass density
    den = f["PartType0"]["Density"][:][mask]
    # Spatial positions
    x = f["PartType0"]["Coordinates"][:][mask]

    # Mass of each cell/partical
    m = f["PartType0"]["Masses"][:][mask]
    # Calculation smoothing length, useful for weighting and/or visualization
    h = f["PartType0"]["SmoothingLength"][:][mask]
    # Internal (thermal) energy
    u = f["PartType0"]["InternalEnergy"][:][mask]
    v = f["PartType0"]["Velocities"][:][mask]
    b = f["PartType0"]["MagneticField"][:][mask]
    outflow_frac = f["PartType0"]["Metallicity"][:, 11][mask]
    # t = f['PartType0']['Temperature'][:] * mask
    # Fraction of molecular material in each cell
    try:
        fmol = f["PartType0"]["MolecularMassFraction"][:][mask]
    except KeyError:
        fmol = np.ones_like(u) * np.inf
    # To get molecular gas density do: den*fmol*fneu*(1-helium_mass_fraction)/(2.0*mh), helium_mass_fraction=0.284
    fneu = f["PartType0"]["NeutralHydrogenAbundance"][:][mask]
    gas_ids = f["PartType0"]["ParticleIDs"][:][mask]

    ## Units and snapshot time
    try:
        unitlen = f["Header"].attrs["UnitLength_In_CGS"]
        unitmass = f["Header"].attrs["UnitMass_In_CGS"]
        unitvel = f["Header"].attrs["UnitVelocity_In_CGS"]
    ##Fallback for units...
    except KeyError:
        unitlen = 3.085678e18
        unitmass = 1.989e33
        unitvel = 100.0
    unitb = 1e4  # f['Header'].attrs['UnitMagneticField_In_CGS'] If not defined
    unit_base = {
        "UnitLength": unitlen,
        "UnitMass": unitmass,
        "UnitVel": unitvel,
        "UnitB": unitb,
    }
    time = f["Header"].attrs["Time"]
    tsnap_myr = (
        time
        * (unit_base["UnitLength"] / unit_base["UnitVel"])
        / (3600.0 * 24.0 * 365.0 * 1e6)
    )

    if "PartType5" in f.keys():
        partpos = f["PartType5"]["Coordinates"][:]
        partmasses = f["PartType5"]["Masses"][:]
        partvels = f["PartType5"]["Velocities"][:]
        partids = f["PartType5"]["ParticleIDs"][:]
        partsink = f["PartType5"]["SinkRadius"][:]
        partspin = f["PartType5"]["BH_Specific_AngMom"][:]
        tstar_form_Myr = (
            f["PartType5"][star_age_key][...]
            * (unit_base["UnitLength"] / unit_base["UnitVel"])
            / (3600.0 * 24.0 * 365.0 * 1e6)
        )
        tage_myr = tsnap_myr - tstar_form_Myr
    ##Had some non-empty values here...
    else:
        partpos = []
        partmasses = []
        partids = []
        partvels = []
        partsink = []
        partspin = []
        tage_myr = []

    print("Snapshot time in %f Myr" % (tsnap_myr))

    del f
    return {
        "den": den,
        "x": x,
        "m": m,
        "h": h,
        "u": u,
        "b": b,
        "v": v,
        "fmol": fmol,
        "fneu": fneu,
        "gas_ids": gas_ids,
        "partpos": partpos,
        "partmasses": partmasses,
        "partvels": partvels,
        "partids": partids,
        "partsink": partsink,
        "tage_myr": tage_myr,
        "unit_base": unit_base,
        "partspin": partspin,
        "outflow_frac": outflow_frac,
    }


def bash_command(cmd, **kwargs):
    """Run command from the bash shell"""
    process = subprocess.Popen(["/bin/bash", "-c", cmd], **kwargs)
    return process.communicate()[0]


def main():
    print("test")
    parser = argparse.ArgumentParser(
        description="Parse starforge snapshot, and get multiple data."
    )
    parser.add_argument("acc_data_lookup", help="Index of snapshot to read")
    parser.add_argument(
        "--snap_base", default="snapshot", help="First part of snapshot name"
    )

    args = parser.parse_args()
    acc_data_lookup = pd.read_parquet(args.acc_data_lookup).set_index(
        ["first_time", "pid"]
    )
    first_times = acc_data_lookup.index.get_level_values(level="first_time").unique()

    star_first_distance = {}
    ##Iterate over all snapshots
    for ss in first_times:
        print(ss)
        sys.stdout.flush()
        snapshot_file = args.snap_base + "_{0:03d}.hdf5".format(int(ss))
        out = load_data(
            snapshot_file,
            res_limit=1e-3,
        )
        xuniq = out["x"]
        gas_ids = out["gas_ids"]
        partpos = out["partpos"]
        partids = out["partids"]
        xuniq = xuniq.astype(np.float64)

        acc_data_lookup_select = acc_data_lookup.loc[ss]
        forming_stars = acc_data_lookup_select.index.unique()
        ##Iterate over all particles in snapshot
        for star in forming_stars:
            gas_star = acc_data_lookup_select.loc[star]
            filt = np.isin(gas_ids, gas_star)
            xfilt = xuniq[filt]

            star_pos = partpos[partids == star]
            delta = xfilt - star_pos
            delta = np.linalg.norm(delta, axis=1)
            star_first_distance[star] = delta

        with open("star_first_distance.p", "wb") as ff:
            pickle.dump(star_first_distance, ff)


if __name__ == "__main__":
    main()
