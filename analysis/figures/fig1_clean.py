import ast
import configparser
import sys

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from meshoid import Meshoid
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_data(file, res_limit=0.0, star_age_key="ProtoStellarAge"):
    """file - h5pdf5 STARFORGE snapshot
    res_limit - minimum mass resolution to include in analyis (in code units)
    """
    # Load snapshot data
    f = h5py.File(file, "r")

    # Mask to remove any cells with mass below the cell resolution
    # (implemented specifically to remove feedback cells if desired)
    mask = f["PartType0"]["Masses"][:] >= res_limit * 0.999
    mask3d = np.array([mask, mask, mask]).T

    # Read in gas properties
    # Mass density
    den = f["PartType0"]["Density"][:] * mask
    # Spatial positions
    x = f["PartType0"]["Coordinates"] * mask3d

    # Mass of each cell/partical
    m = f["PartType0"]["Masses"][:] * mask
    # Calculation smoothing length, useful for weighting and/or visualization
    h = f["PartType0"]["SmoothingLength"][:] * mask
    # Internal (thermal) energy
    u = f["PartType0"]["InternalEnergy"][:] * mask
    v = f["PartType0"]["Velocities"] * mask3d
    b = f["PartType0"]["MagneticField"][:] * mask3d
    # t = f['PartType0']['Temperature'][:] * mask
    # Fraction of molecular material in each cell
    try:
        fmol = f["PartType0"]["MolecularMassFraction"][:] * mask
    except KeyError:
        fmol = np.ones_like(u) * np.inf
    # To get molecular gas density do: den*fmol*fneu*(1-helium_mass_fraction)/(2.0*mh), helium_mass_fraction=0.284
    fneu = f["PartType0"]["NeutralHydrogenAbundance"][:] * mask

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

    print("Snapshot time is %f Myr" % (tsnap_myr))

    del f
    return (
        den,
        x,
        m,
        h,
        u,
        b,
        v,
        fmol,
        fneu,
        partpos,
        partmasses,
        partvels,
        partids,
        partsink,
        tage_myr,
        unit_base,
        partspin,
    )

############################################################################################################
##PARSING CONFIG OPTIONS
config = configparser.ConfigParser()
config.read(f"config_{sys.argv[1]}")
##snapshot number to plot
snap_idx = config.getint("params", "snap_idx")
##id of star 1
bin_id1 = config.getint("params", "bin1")
##id of star 2
bin_id2 = config.getint("params", "bin2")
##base of simulation path
base = config.get("params", "base", fallback=f"./")
##defines region over which we compute the surface density
rmax = config.getfloat("params", "rmax", fallback=0.5)
##resolution of grid used to reconstruct surface density
res = config.getint("params", "res", fallback=800)
##parameters for colorscale
vmin = config.getfloat("params", "vmin", fallback=1.0)
vmax = config.getfloat("params", "vmax", fallback=3e4)
##axis limits: (-plimit, plimit) if positive, (-rmax, rmax) if negative
plimit = config.getfloat("params", "plimit", fallback=-1)
##fixed center (if none use com of star 1 and star 2)
center = config.get("params", "center", fallback=None)
savetype = config.get("params", "savetype", fallback="png")
############################################################################################################
##PARSING SIMULATION DATA
snap_file = base + f"snapshot_{snap_idx:03d}.hdf5"
(
    den,
    x,
    m,
    h,
    u,
    b,
    v,
    fmol,
    fneu,
    partpos,
    partmasses,
    partvels,
    partids,
    partsink,
    tage_myr,
    unit_base,
    partspin,
) = load_data(snap_file, res_limit=1e-3)

##Filtering duplicates
xuniq, indx = np.unique(x, return_index=True, axis=0)
##Gas data: muniq: mass, huniq: softening, vuniq: velocity, uuniq: internal energy, denuniq: density
##Data that is unused is commented out
muniq = m[indx]
huniq = h[indx]
# vuniq = v[indx]
# uuniq = u[indx]
# denuniq = den[indx]
xuniq = xuniq.astype(np.float64)
muniq = muniq.astype(np.float64)
huniq = huniq.astype(np.float64)
# vuniq = vuniq.astype(np.float64)
# uuniq = uuniq.astype(np.float64)
# denuniq = denuniq.astype(np.float64)
##sink particle data partpos: particle, partvels: velocities, partmasses: masses, partsink: softening length
partpos = partpos.astype(np.float64)
parvels = partvels.astype(np.float64)
partmasses = partmasses.astype(np.float64)
# partsink = partsink.astype(np.float64)
############################################################################################################
##GETTING CENTER
##User can either specify the center with the "center" argument or by giving "particle ids"
##in the latter case the code will use the com of the stars for the center.
##If you would like to center it
if center is not None:
    center = ast.literal_eval(center)
else:
    star_pos = np.concatenate(
        (partpos[partids == bin_id1], partvels[partids == bin_id1])
    ).ravel()
    star_mass = partmasses[partids == bin_id1]
    star_pos2 = np.concatenate(
        (partpos[partids == bin_id2], partvels[partids == bin_id2])
    ).ravel()
    star_mass2 = partmasses[partids == bin_id2]
    center = (star_mass * star_pos + star_mass2 * star_pos2) / (star_mass + star_mass2)

center = np.array(center)
print("Center:", center)
############################################################################################################
##ONLY SELECT GAS IN VOXEL AROUND STARS
sel2 = np.abs(xuniq - center[:3])
sel2 = (sel2[:, 0] < rmax) & (sel2[:, 1] < rmax) & (sel2[:, 2] < rmax)
############################################################################################################
##GETTING SURFACE DENSITY VIA THE MESHOID PACKAGE
xuniq_center = xuniq - center[:3]
M = Meshoid(xuniq_center[sel2], muniq[sel2], huniq[sel2])
X = np.linspace(-rmax, rmax, res)
Y = np.linspace(-rmax, rmax, res)
X, Y = np.meshgrid(X, Y, indexing="ij")
sigma_gas_msun_pc2 = M.SurfaceDensity(
    M.m, size=2 * rmax, res=res, center=np.array((0, 0, 0))
)
############################################################################################################
##PLOTTING THE SURFACE DENSITY
fig, ax = plt.subplots(figsize=(9.5, 8), constrained_layout=True)
ax.set_xlabel("x [pc]")
ax.set_ylabel("y [pc]")
p = ax.pcolormesh(
    X,
    Y,
    sigma_gas_msun_pc2,
    norm=colors.LogNorm(vmin=vmin, vmax=vmax),
    cmap="viridis",
    linewidth=0,
    rasterized=True,
)
if plimit > 0:
    ax.set_xlim(-plimit, plimit)
    ax.set_ylim(-plimit, plimit)
else:
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)

plt.colorbar(p, label=r"$\Sigma$ [$M_{\odot}$ pc$^{-2}$]")
fig.savefig(f"surface_density_{snap_idx}." + savetype, dpi=300)
############################################################################################################
