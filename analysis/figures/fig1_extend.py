import ast
import configparser
import pickle
import sys

import h5py
import matplotlib.pyplot as plt
import matplotlib.units as units
import numpy as np
import seaborn as sns
from matplotlib import colors
from matplotlib.colors import LogNorm
import matplotlib.patches as patches
from meshoid import Meshoid
from mpl_toolkits.axes_grid1 import make_axes_locatable
import pandas as pd

from starforge_mult_search.analysis import cgs_const as cgs
from starforge_mult_search.code import find_multiples_new2
from starforge_mult_search.code import starforge_constants as sfc
from starforge_mult_search.code.find_multiples_new2 import cluster, system

snap_interval = 2.47e4
conv = cgs.pc / cgs.au / 1e4
# Define a custom unit
class AUnit(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        "Convert a datetime value to a scalar or array."
        return (value) * cgs.pc / cgs.au

def subtract_path(p1, p2):
    assert len(p1)==len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:,0])) & (~np.isinf(p2[:,0]))
    diff[filt] = p1[filt] - p2[filt]

    return diff

def get_phalo(base, aa, snap_idx, bin_id1, bin_id2, my_ft):
    with open(base.replace("/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/", "") + aa + "/path_lookup.p", "rb") as ff:
        path_lookup = (pickle.load(ff))

    tmp_pos = path_lookup[f"{bin_id1}"][snap_idx, pxcol:vzcol+1]
    tmp_mass = path_lookup[f"{bin_id1}"][snap_idx, mcol]
    with h5py.File(base + f"/halo_masses/halo_masses_sing_npTrue_c0.5_{snap_idx}_compFalse_tf{my_ft}.hdf5") as hf:
        tmp_halo_pos = np.hstack((hf[f"halo_{bin_id1}_x"][...], hf[f"halo_{bin_id1}_v"][...]))
        tmp_halo_mass = (hf[f"halo_{bin_id1}_m"][...])
        tmp_halo_rho = (hf[f"halo_{bin_id1}_rho"][...])

    tmp_pos2 = path_lookup[f"{bin_id2}"][snap_idx, pxcol:vzcol+1]
    tmp_mass2 = path_lookup[f"{bin_id2}"][snap_idx, mcol]
    with h5py.File(base + f"/halo_masses/halo_masses_sing_npTrue_c0.5_{snap_idx}_compFalse_tf{my_ft}.hdf5") as hf:
        tmp_halo_pos2 = np.hstack((hf[f"halo_{bin_id2}_x"][...], hf[f"halo_{bin_id2}_v"][...]))
        tmp_halo_mass2 = (hf[f"halo_{bin_id2}_m"][...])
        tmp_halo_rho2 = (hf[f"halo_{bin_id2}_rho"][...])

    center = (tmp_mass * tmp_pos + tmp_mass2 * tmp_pos2) / (tmp_mass + tmp_mass2)
    tmp_pos_center = tmp_pos - center
    tmp_halo_pos_center = tmp_halo_pos - center
    tmp_pos2_center = tmp_pos2 - center
    tmp_halo_pos2_center = tmp_halo_pos2 - center

    ###Getting com of each star + halo
    com_w_halo = (tmp_pos * tmp_mass + np.sum(tmp_halo_mass[:, np.newaxis] * tmp_halo_pos, axis=0)) / (tmp_mass + np.sum(tmp_halo_mass))
    com2_w_halo = (tmp_pos2 * tmp_mass2 + np.sum(tmp_halo_mass2[:, np.newaxis] * tmp_halo_pos2, axis=0)) / (tmp_mass2 + np.sum(tmp_halo_mass2))

    ##Be careful with the different coordinates here...
    return center, tmp_pos_center, tmp_halo_pos_center, tmp_pos2_center, tmp_halo_pos2_center, com_w_halo - center, com2_w_halo - center

def get_phalo_limits(base, aa, snap_idx, bin_id1, bin_id2):
    center, tmp_pos_center, tmp_halo_pos_center, tmp_pos2_center, tmp_halo_pos2_center, com_w_halo, com2_w_halo = get_phalo(base, aa, snap_idx,
                                                                                                   bin_id1, bin_id2, "8.0")
    ##Automatically set axis extent based on the size of the halos -- TO DO PLOT CONSI
    halos_x = (np.concatenate(
        (tmp_halo_pos_center[:, 0], tmp_halo_pos2_center[:, 0], [tmp_pos_center[0]], [tmp_pos2_center[0]])))
    halos_y = (np.concatenate(
        (tmp_halo_pos_center[:, 1], tmp_halo_pos2_center[:, 1], [tmp_pos_center[1]], [tmp_pos2_center[1]])))
    xmin, xmax = min(halos_x), max(halos_x)
    ymin, ymax = min(halos_y), max(halos_y)

    return xmin, xmax, ymin, ymax

def get_initial_orbit(tmp1, tmp2):
    tmp_filt = (~np.isinf(tmp1[:, 0])) & (~np.isinf(tmp2[:,0]))
    tmp1_fst = tmp1[tmp_filt][0]
    tmp2_fst = tmp2[tmp_filt][0]
    tmp_orb = find_multiples_new2.get_orbit(tmp1_fst[pxcol:pzcol+1], tmp2_fst[pxcol:pzcol+1],\
                                            tmp1_fst[vxcol:vzcol+1], tmp2_fst[vxcol:vzcol+1],\
                                            tmp1_fst[mtotcol], tmp2_fst[mtotcol],\
                                           tmp1_fst[hcol], tmp2_fst[hcol])

    return tmp_orb


def add_colorbar_to_axes(ax, mappable, label='', orientation='vertical', size='5%', pad=0.05):
    """
    Add a colorbar to an existing axes.
`
    Parameters:
    - ax: The axes to which the colorbar should be added.
    - mappable: The image or plot object to which the colorbar applies (e.g., the result of ax.imshow()).
    - label: The label for the colorbar.
    - orientation: The orientation of the colorbar ('vertical' or 'horizontal').
    - size: The size of the colorbar relative to the axes.
    - pad: The padding between the axes and the colorbar.
    """
    divider = make_axes_locatable(ax)
    if orientation == 'vertical':
        cax = divider.append_axes("right", size=size, pad=pad)
    else:
        cax = divider.append_axes("bottom", size=size, pad=pad)

    cbar = plt.colorbar(mappable, cax=cax, orientation=orientation)
    cbar.set_label(label, rotation=90 if orientation == 'vertical' else 0, labelpad=15)
    return cbar


def sigmoid(x):
    return 0.5 * (1. + x / (1. + x**2.)**.5)

def ad_index(u):
    delta = (-.38, 0.22, -0.068, -0.42, 0.65)
    a = (5.95, 6,18, 10.26, 7.71, 98.87)
    b = (9.25, 9.89, 10.24, 11.13, 14.28)

    u_cgs = u * 100**2.
    gamma = 5. / 3.
    for kk in range(5):
        gamma += delta[kk] * sigmoid(a[kk] * (np.log10(u_cgs) - b[kk]))

    return gamma

def u_to_cs(u1):
    gamma_eff = ad_index(u1)
    # print("gamma:",gamma_eff)
    return u1**.5 * (gamma_eff * (gamma_eff - 1))**.5


units.registry["au"] = AUnit()
colorblind_palette = sns.color_palette("colorblind")
# Set the matplotlib color cycle to the seaborn colorblind palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colorblind_palette)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['patch.linewidth'] = 3
col1 = np.array((129, 50, 168)) / 256
col2 = colorblind_palette[1]

sink_cols = np.array(("t", "id", "px", "py", "pz", "vx", "vy", "vz", "h", "m"))
sink_cols = np.concatenate((sink_cols, ["sys_id", "mtot", "sma", "ecc"]))
mcol = np.where(sink_cols == "m")[0][0]
pxcol = np.where(sink_cols == "px")[0][0]
pycol = np.where(sink_cols == "py")[0][0]
pzcol = np.where(sink_cols == "pz")[0][0]
vxcol = np.where(sink_cols == "vx")[0][0]
vycol = np.where(sink_cols == "vy")[0][0]
vzcol = np.where(sink_cols == "vz")[0][0]
hcol = np.where(sink_cols == "h")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]
scol = np.where(sink_cols == "sys_id")[0][0]


config = configparser.ConfigParser()
config.read(f"config_{sys.argv[1]}")

snap_idx = config.getint("params","snap_idx")
bin_id1 = config.getint("params","bin1")
bin_id2 = config.getint("params", "bin2")
my_ft = config.get("params","ft", fallback="1.0")
seed = config.getint("params","seed", fallback=42)
rmax = config.getfloat("params", "rmax", fallback=0.5)
res = config.getint("params", "res", fallback=800)
savetype = config.get("params","savetype", fallback="png")
vmin = config.getfloat("params", "vmin", fallback=1.0)
vmax = config.getfloat("params", "vmin", fallback=3e4)
plimit = config.getfloat("params", "plimit", fallback=-1)
ins = config.getfloat("params", "ins", fallback=-1.0)
ins_loc = config.get("params", "ins_loc", fallback="upper right")
annot = config.get("params", "annot", fallback="")
v_rescale = config.getfloat("params", "v_rescale", fallback=2)
center = config.get("params", "center", fallback=None)
base = config.get("params", "base", fallback=f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_")
snap_loc = config.get("params", "snap_loc", fallback=None)
tracer_file = config.get("params", "tracers", fallback="")
down_sample = config.getint("params", "down_sample", fallback=1)
arrow_opacity = config.getfloat("params", "arrow_opacity", fallback=0.8)

if center is not None:
    center = ast.literal_eval(center)

v_scale = 100. / cgs.pc * cgs.year * v_rescale
d_cut = rmax
base = base + f"{seed}/"

# r2 = f"_TidesFalse_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse.p".replace(".p", "")
# aa = "analyze_multiples_output_" + r2 + "/"
if snap_loc is None:
    snap_loc = base
snap_file = snap_loc + f"snapshot_{snap_idx:03d}.hdf5"

den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base, partspin = \
find_multiples_new2.load_data(snap_file, res_limit=1e-3)
gas_ids = find_multiples_new2.load_gas_ids(snap_file, res_limit=1e-3)

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
partpos = partpos.astype(np.float64)
parvels = partvels.astype(np.float64)
partmasses = partmasses.astype(np.float64)
partsink = partsink.astype(np.float64)

##Hack for xz plane--flip y and z...in all the arrays??? For some reason cannot seem to set plane in Meshoid?
##xuniq, vuniq, partpos, partvels

if center is None:
    # center, tmp_pos_center, tmp_halo_pos_center, tmp_pos2_center, tmp_halo_pos2_center, com_w_halo, com2_w_halo = get_phalo(base, aa, snap_idx,
    #                                                                                        bin_id1, bin_id2, my_ft)
    tmp_pos = np.concatenate((partpos[partids==bin_id1], partvels[partids==bin_id1])).ravel()
    tmp_mass = partmasses[partids==bin_id1]
    tmp_pos2 = np.concatenate((partpos[partids==bin_id2], partvels[partids==bin_id2])).ravel()
    tmp_mass2 = partmasses[partids==bin_id2]
    center = (tmp_mass * tmp_pos + tmp_mass2 * tmp_pos2) / (tmp_mass + tmp_mass2)

center = np.array(center)
##ONLY SELECT GAS IN VOXEL AROUND STARS
sel2 = np.abs(xuniq - center[:3])
sel2 = (sel2[:,0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:,2] < d_cut)
sel2_gas = np.copy(sel2)

##GETTING SURFACE DENSITY VIA THE MESHOID PACKAGE
xuniq_center = xuniq - center[:3]
M = Meshoid(xuniq_center[sel2], muniq[sel2], huniq[sel2])
X = np.linspace(- rmax, rmax, res)
Y = np.linspace(- rmax, rmax, res)
X, Y = np.meshgrid(X, Y, indexing='ij')
sigma_gas_msun_pc2 = M.SurfaceDensity(M.m,  size=2 * rmax, res=res, center=np.array((0,0,0)))  # *1e4

############################################################################################################

fig,ax = plt.subplots(figsize=(8,8), constrained_layout=True)
ax.set_xlabel("x [pc]")
ax.set_ylabel("y [pc]")
# ax.annotate(f"Example {annot}", (0.01, 0.99), xycoords='axes fraction', va="top", ha="left")

p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", linewidth=0, rasterized=True)
# ax.scatter(tmp_pos_center[0], tmp_pos_center[1],  marker="X", color="k", s=40)
# ax.scatter(tmp_pos2_center[0], tmp_pos2_center[1],  marker="X", color="k", s=40)
if plimit > 0:
    ax.set_xlim(-plimit, plimit)
    ax.set_ylim(-plimit, plimit)
else:
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)

fig.savefig(f"fig1_{sys.argv[1]}a_{snap_idx}." + savetype, dpi=300)
############################################################################################################
# dist_center = partpos - center[:3]
# dist_center = np.sum(dist_center * dist_center, axis=1)**.5
sel2 = np.abs(partpos - center[:3])
##Only include star partciles in the box...
dist_filter = (sel2[:,0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:,2] < d_cut)
partpos_filt  = partpos[dist_filter]
partvel_filt = partvels[dist_filter]
partids_filt = partids[dist_filter]
partmasses_filt = partmasses[dist_filter]
#####Overlays of star paticles and stars
for ii in range(len(partpos_filt)):
    center_x, center_y = partpos_filt[ii, 0] - center[0], partpos_filt[ii, 1] - center[1]
    ax.plot(center_x, center_y, "kX", markersize= 4 * np.log(partmasses_filt[ii] / 0.001))
    # Create the circular patch comparable to the accretion radius...Really this is an upper bound(!)
    # hl_radius = 2. * sfc.GN * partmasses_filt[ii] / np.linalg.norm(partvel_filt[ii])**2.
    ##Arbitrary cutoff for gas neighbors...
    gas_neighbors = np.linalg.norm(xuniq[sel2_gas] - partpos_filt[ii], axis=1) < 0.01
    if len(xuniq[sel2_gas][gas_neighbors]) == 0:
        continue
    gas_neighbors_vel = np.mean(vuniq[sel2_gas][gas_neighbors], axis=0)
    gas_neighbors_cs = np.mean(u_to_cs(uuniq[sel2_gas][gas_neighbors]))
    bhl_radius = 2. * sfc.GN * partmasses_filt[ii] / (np.linalg.norm(partvel_filt[ii] - gas_neighbors_vel)**2. + gas_neighbors_cs**2.)
    print(gas_neighbors_cs / 1e3, gas_neighbors_vel / 1e3, np.linalg.norm(partvel_filt[ii] - gas_neighbors_vel) / 1e3, np.linalg.norm(partvel_filt[ii]) / 1e3)

    circle = patches.Circle((center_x, center_y), bhl_radius, color='orange', fill=False, linewidth=2, label='Circle')
    # Add the circle to the axes
    ax.add_patch(circle)
arr_index1 = np.where(partids_filt.astype(int)==bin_id1)[0]
arr_index2 = np.where(partids_filt.astype(int)==bin_id2)[0]
ax.plot(partpos_filt[arr_index1, 0] - center[0], partpos_filt[arr_index1, 1] - center[1], "ro")
ax.plot(partpos_filt[arr_index2, 0] - center[0], partpos_filt[arr_index2, 1] - center[1], "ro")


fig.savefig(f"fig1_{sys.argv[1]}b_{snap_idx}." + savetype, dpi=300)

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
colors = ["w", "gold"]

if tracer_file:
    tracer_data = np.genfromtxt(tracer_file)
    tracer_ids = tracer_data[:, 0]
    # is_accreted = tracer_data[:, 1].astype(bool)
    tracer_filt = np.isin(gas_ids, tracer_ids)
    tmp_halo_pos = np.hstack((xuniq[tracer_filt], vuniq[tracer_filt]))
    is_accreted = pd.DataFrame(tracer_data, columns=("id", "acc"), dtype=int).set_index("id").loc[gas_ids[tracer_filt]]
    is_accreted = is_accreted["acc"].to_numpy().astype(bool)

    ##Only include halo particles in the Voxel
    sel2 = np.abs(tmp_halo_pos[:, :3] - center[:3])
    dist_filter = (sel2[:,0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:,2] < d_cut)
    tmp_halo_pos = tmp_halo_pos[dist_filter]
    is_accreted = is_accreted[dist_filter]
    random_selection = np.random.choice(range(len(tmp_halo_pos)), len(tmp_halo_pos) // down_sample, replace=False)
    tmp_halo_pos = tmp_halo_pos[random_selection]
    is_accreted = is_accreted[random_selection]

    arrow_cols = [colors[0] if row else colors[1] for row in is_accreted]
    try:
        ax.quiver(tmp_halo_pos[:, 0] - center[0], tmp_halo_pos[:, 1] - center[1],
                    (tmp_halo_pos[:, 3]  - center[3]) * v_scale * snap_interval,
                    (tmp_halo_pos[:, 4]  - center[4]) * v_scale * snap_interval,
                    scale=1, scale_units="xy", angles="xy",alpha=arrow_opacity, color=arrow_cols)#color=colors[int(partids_filt[ii]) % len(colors)])
    except IndexError:
        breakpoint()
fig.savefig(f"fig1_{sys.argv[1]}d_{snap_idx}." + savetype, dpi=300)

# for ii in range(len(partids_filt)):
#     with h5py.File(base + f"/halo_masses/halo_masses_sing_npTrue_c0.5_{snap_idx}_compFalse_tf{my_ft}.hdf5") as hf:
#         tmp_halo_arr_id = hf[f"halo_{partids_filt[ii]}"][...]
#         tmp_halo_pos = np.hstack((hf[f"halo_{partids_filt[ii]}_x"][...], hf[f"halo_{partids_filt[ii]}_v"][...]))
#         if np.sum(tmp_halo_arr_id)==0:
#             continue
#         try:
#             ax.quiver(tmp_halo_pos[:, 0] - center[0], tmp_halo_pos[:, 1] - center[1],
#                       (tmp_halo_pos[:, 3]  - partvel_filt[ii, 0]) * v_scale * snap_interval,
#                       (tmp_halo_pos[:, 4]  - partvel_filt[ii, 1]) * v_scale * snap_interval,
#                       scale=1, scale_units="xy", angles="xy",alpha=0.2, color=colors[int(partids_filt[ii]) % len(colors)])
#         except IndexError:
#             breakpoint()
# fig.savefig(f"fig1_{sys.argv[1]}d_{snap_idx}." + savetype, dpi=300)


