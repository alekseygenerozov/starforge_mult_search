import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LogNorm
import seaborn as sns
import pickle
import h5py

from meshoid import Meshoid

import cgs_const as cgs

# sys.path.append("/home/aleksey/code/python/star_forge_analysis/")
# from analyze_multiples import snap_lookup
from starforge_mult_search.code import find_multiples_new2
from starforge_mult_search.code.find_multiples_new2 import cluster, system
from starforge_mult_search.code import starforge_constants as sfc
import configparser
import matplotlib.units as units
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Create new colormap
# newcolors = plt.cm.viridis(np.linspace(0, 1, 256))
# # Reduce blue in the first half to avoid conflict with blue scatter points
# newcolors[:128, 2] *= np.linspace(0.3, 1, 128)
# # Reduce red and green in the second half to avoid conflict with orange scatter points
# newcolors[128:, 0] *= np.linspace(1, 0.5, 128)
# newcolors[128:, 1] *= np.linspace(1, 0.7, 128)
#
# custom_cmap = colors.ListedColormap(newcolors)
# Adjust scatter plot colors

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
    with open(base + aa + "/path_lookup.p", "rb") as ff:
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




units.registry["au"] = AUnit()
colorblind_palette = sns.color_palette("colorblind")
# Set the matplotlib color cycle to the seaborn colorblind palette
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=colorblind_palette)
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['patch.linewidth'] = 3
# col2 = (1, 0.2, 0.2)  # More vibrant orange
# col1 =  (0.8, 0.3, 0.3)    # More vibrant blue
# col1 = colorblind_palette[0]
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
config.read("config")

snap_idx = config.getint("params","snap_idx")
bin_id1 = config.getint("params","bin1")
bin_id2 = config.getint("params", "bin2")
my_ft = config.get("params","ft", fallback="1.0")
seed = config.getint("params","seed", fallback=42)
rmax = config.getfloat("params", "rmax", fallback=0.5)
res = config.getint("params", "res", fallback=800)
savetype = config.get("params","savetype", fallback="pdf")
vmin = config.getfloat("params", "vmin", fallback=1.0)
vmax = config.getfloat("params", "vmin", fallback=3e4)
plimit = config.getfloat("params", "plimit", fallback=-1)
ins = config.getfloat("params", "ins", fallback=-1.0)
ins_loc = config.get("params", "ins_loc", fallback="upper right")
annot = config.get("params", "annot", fallback="")

d_cut = rmax
# base = f"/work2/09543/aleksey2/frontera/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{seed}/"
base = f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{seed}/"

# with open(base + "/data_loc", "r") as ff:
# 	snap_base = ff.read().strip()
r2 = f"_TidesTrue_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse.p".replace(".p", "")
aa = "analyze_multiples_output_" + r2 + "/"

# snap_file = base + f"_{snap_idx:03d}.hdf5"
snap_file = base + f"snapshot_{snap_idx:03d}.hdf5"

den, x, m, h, u, b, v, fmol, fneu, partpos, partmasses, partvels, partids, partsink, tage_myr, unit_base = \
find_multiples_new2.load_data(snap_file, res_limit=1e-3)
##WOULD HAVE BEEN BETTER TO PUT THIS FUNCTIONALITY IN LOAD DATA!!!
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

##GET HALO AND PARTICLE POSITIONS CENTERED ON THE FIRST STAR
center, tmp_pos_center, tmp_halo_pos_center, tmp_pos2_center, tmp_halo_pos2_center, com_w_halo, com2_w_halo = get_phalo(base, aa, snap_idx,
                                                                                       bin_id1, bin_id2, my_ft)
##ONLY SELECT GAS IN VOXEL AROUND STARS
sel2 = np.abs(xuniq - center[:3])
sel2 = (sel2[:,0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:,2] < d_cut)

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
ax.annotate(f"Example {annot}", (0.01, 0.99), xycoords='axes fraction', va="top", ha="left")

p = ax.pcolormesh(X, Y, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", linewidth=0, rasterized=True)
# sc1 = ax.scatter(tmp_pos_center[0], tmp_pos_center[1], marker="X", color="k")#, edgecolors="k", facecolors="none")
sc1 = ax.quiver(tmp_pos_center[0], tmp_pos_center[1], tmp_pos_center[3], tmp_pos_center[4])#, edgecolors="k", facecolors="none")
ax.scatter(tmp_pos2_center[0], tmp_pos2_center[1],  marker="X", color="k")#, edgecolors="k", facecolors="none")
# plt.colorbar(p, label=r"$\Sigma$ [$M_{\odot} pc^{-2}$]")
if plimit > 0:
    ax.set_xlim(-plimit, plimit)
    ax.set_ylim(-plimit, plimit)

fig.savefig("pretty." + savetype)

############################################################################################################

fig,ax = plt.subplots(figsize=(9.5,8), constrained_layout=True)
# ax.ticklabel_format(style='sci', axis='both', scilimits=(4,4), useMathText=True)

ax.set_xlabel("x [$10^4$ au]")
ax.set_ylabel("y [$10^4$ au]")

xmin, xmax, ymin, ymax = get_phalo_limits(base, aa, snap_idx, bin_id1, bin_id2)
buff = 0.05
xmin *= conv
xmax *= conv
ymin *= conv
ymax *= conv


p = ax.pcolormesh(X * conv, Y * conv, sigma_gas_msun_pc2, norm=colors.LogNorm(vmin=vmin, vmax=vmax), cmap="viridis", linewidth=0, rasterized=True)
ax.quiver(tmp_halo_pos_center[:, 0] * conv, tmp_halo_pos_center[:, 1] * conv, tmp_halo_pos_center[:,3], tmp_halo_pos_center[:,4], color=col1, alpha=0.4)
ax.quiver(tmp_halo_pos2_center[:, 0] * conv, tmp_halo_pos2_center[:, 1] * conv, tmp_halo_pos2_center[:,3], tmp_halo_pos2_center[:,4], color="#A52A2A", alpha=0.4)

center_b, tmp_pos_center_b, tmp_halo_pos_center_b, tmp_pos2_center_b, tmp_halo_pos2_center_b, com_w_halo_b, com2_w_halo_b = get_phalo(base, aa, snap_idx,
                                                                                       bin_id1, bin_id2, 8.0)
ax.quiver(tmp_halo_pos_center_b[:, 0] * conv, tmp_halo_pos_center_b[:, 1] * conv, tmp_halo_pos_center_b[:,3], tmp_halo_pos_center_b[:,4], color=col1, alpha=0.2)
ax.quiver(tmp_halo_pos2_center_b[:, 0] * conv, tmp_halo_pos2_center_b[:, 1] * conv, tmp_halo_pos2_center_b[:,3], tmp_halo_pos2_center_b[:,4], color="#A52A2A", alpha=0.2)
plt.colorbar(p, label=r"$\Sigma$ [$M_{\odot} pc^{-2}$]")

#####################################################################################################
with open(base + aa + "/path_lookup.p", "rb") as ff:
    path_lookup = (pickle.load(ff))
# ##Getting com over time for pair -- make sure that the replacement here will not cause errors
tmp1 = path_lookup[f"{bin_id1}"]
tmp2 = path_lookup[f"{bin_id2}"]
ms1 = tmp1[:, mcol]
ms1[np.isinf(ms1)] = 1
ms1.shape = (-1, 1)
ms2 = tmp2[:, mcol]
ms2[np.isinf(ms2)] = 1
ms2.shape = (-1, 1)
coms = (tmp1[:, pxcol:pzcol + 1] * ms1 + tmp2[:, pxcol:pzcol + 1] * ms2) / (ms1 + ms2)
##Positions in the evolving com frame
p1 = subtract_path(tmp1[:, pxcol:pzcol + 1], coms)
p2 = subtract_path(tmp2[:, pxcol:pzcol + 1], coms)

tmp_orb = get_initial_orbit(tmp1, tmp2)
# add_colorbar_to_axes(ax, p, label=r"$\Sigma$ [$M_{\odot} pc^{-2}$]")
########################################################################################
if ins > 0:
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset, inset_axes

    # Define the region to zoom in on
    x1, x2, y1, y2 = -ins, ins, -ins, ins
    # Create an inset of the zoomed region
    # axins = zoomed_inset_axes(ax, zoom=4, loc='upper right', borderpad=2)
    axins = ax.inset_axes([0.7, 0.1, 0.2, 0.2])
    axins.tick_params(axis="x", which="major", labelsize=16, rotation=45)
    axins.tick_params(axis="y", which="major", labelsize=16)
    width, height = "30%", "30%"  # specify the width and height of the inset in relative terms
    # axins = inset_axes(ax, width=width, height=height, loc=ins_loc, borderpad=2, bbox_transform = ax.transAxes)
    axins.set_aspect('equal')
    # Update the view to reflect the new units
    # ax.autoscale_view()
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticks([-ins/2, ins/2])
    axins.set_yticks([-ins/2, ins/2])
    axins.plot(p1[:, 0] * conv, p1[:, 1] * conv, color=col1)
    axins.plot(p2[:, 0] * conv, p2[:, 1] * conv, color=col2)
    # mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5", linewidth=2)

ax.set_xlim(xmin -  buff * (xmax - xmin), xmax + buff * (xmax - xmin))
ax.set_ylim(ymin -  buff * (ymax - ymin), ymax + buff * (ymax - ymin))
ax.annotate(r"$a_i = {0:.0f}$ au, $e_i$ = {1:.2g}".format(tmp_orb[0] * conv * 1e4, tmp_orb[1]),
           (0.01, 0.99), ha='left', va='top', xycoords='axes fraction')
start_pt = tmp_pos_center[0] * conv, tmp_pos_center[1] * conv
v_rescale = 3.
vel1 = tmp_pos_center[3] * 100. / cgs.au / 1e4 * cgs.year * v_rescale , tmp_pos_center[4] * 100. / cgs.au / 1e4 * cgs.year * v_rescale
# vel1 = com_w_halo[3] * 100. / cgs.au * cgs.year , com_w_halo[4] * 100. / cgs.au * cgs.year
end_pt = start_pt[0] + vel1[0] * snap_interval, start_pt[1] + vel1[1] * snap_interval
ax.scatter(start_pt[0], start_pt[1], c="k", marker="X", s=40)
# ax.arrow(start_pt[0], start_pt[1],
#          end_pt[0] - start_pt[0], end_pt[1] - start_pt[1],
#          color='k', head_width=1000, head_length=500,
#          length_includes_head=True, linewidth=1.5)
ax.annotate('', xy=(end_pt[0], end_pt[1]), xytext=(start_pt[0], start_pt[1]),
             arrowprops=dict(arrowstyle='->', color="k", linewidth=4))
# ax.scatter(tmp_pos2_center[0] * conv, tmp_pos2_center[1] * conv, c="k", marker="X")

start_pt = tmp_pos2_center[0] * conv, tmp_pos2_center[1] * conv
vel1 = tmp_pos2_center[3] * 100. / cgs.au / 1e4 * cgs.year * v_rescale, tmp_pos2_center[4] * 100. / cgs.au / 1e4 * cgs.year * v_rescale
# vel1 = com2_w_halo[3] * 100. / cgs.au * cgs.year , com2_w_halo[4] * 100. / cgs.au * cgs.year
end_pt = start_pt[0] + vel1[0] * snap_interval, start_pt[1] + vel1[1] * snap_interval
ax.scatter(start_pt[0], start_pt[1], c="k", marker="X", s=40)
# ax.plot([start_pt[0], end_pt[0]], [start_pt[1], end_pt[1]], "k")
# ax.arrow(start_pt[0], start_pt[1],
#          end_pt[0] - start_pt[0], end_pt[1] - start_pt[1],
#          color='k', head_width=1000, head_length=500,
#          length_includes_head=True, linewidth=1.5)
ax.annotate('', xy=(end_pt[0], end_pt[1]), xytext=(start_pt[0], start_pt[1]),
             arrowprops=dict(arrowstyle='->', color="k", linewidth=4))

# plt.tight_layout()
# fig.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
# from matplotlib.transforms import Bbox
# bbox = Bbox.from_bounds(0, 0, 8.5, 8)
fig.savefig(f"prettyb_{snap_idx:03d}." + savetype)


