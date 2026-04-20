import ast
import configparser
import gc
import hashlib
import pickle
import sys

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.units as units
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import colors
from matplotlib.colors import LogNorm
from meshoid import Meshoid
from mpl_toolkits.axes_grid1 import make_axes_locatable

from starforge_mult_search.analysis import cgs_const as cgs
from starforge_mult_search.analysis.analyze_stack import get_blookup
from starforge_mult_search.analysis.high_multiples_analysis import get_maximal_multiples
from starforge_mult_search.code import find_multiples_new2
from starforge_mult_search.code import starforge_constants as sfc
from starforge_mult_search.code.find_multiples_new2 import cluster, system

# Get a colormap with highly distinct colors (tab20 has 20 distinct colors)
cmap = plt.get_cmap("Dark2")
num_colors = cmap.N
snap_interval = 2.47e4
conv = cgs.pc / cgs.au / 1e4


# Define a custom unit
class AUnit(units.ConversionInterface):
    @staticmethod
    def convert(value, unit, axis):
        "Convert a datetime value to a scalar or array."
        return (value) * cgs.pc / cgs.au


def subtract_path(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:, 0])) & (~np.isinf(p2[:, 0]))
    diff[filt] = p1[filt] - p2[filt]

    return diff


def add_colorbar_to_axes(
    ax, mappable, label="", orientation="vertical", size="5%", pad=0.05
):
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
    if orientation == "vertical":
        cax = divider.append_axes("right", size=size, pad=pad)
    else:
        cax = divider.append_axes("bottom", size=size, pad=pad)

    cbar = plt.colorbar(mappable, cax=cax, orientation=orientation)
    cbar.set_label(label, rotation=90 if orientation == "vertical" else 0, labelpad=15)
    return cbar


def sigmoid(x):
    return 0.5 * (1.0 + x / (1.0 + x**2.0) ** 0.5)


def ad_index(u):
    delta = (-0.38, 0.22, -0.068, -0.42, 0.65)
    a = (5.95, 6, 18, 10.26, 7.71, 98.87)
    b = (9.25, 9.89, 10.24, 11.13, 14.28)

    u_cgs = u * 100**2.0
    gamma = 5.0 / 3.0
    for kk in range(5):
        gamma += delta[kk] * sigmoid(a[kk] * (np.log10(u_cgs) - b[kk]))

    return gamma


def u_to_cs(u1):
    gamma_eff = ad_index(u1)
    # print("gamma:",gamma_eff)
    return u1**0.5 * (gamma_eff * (gamma_eff - 1)) ** 0.5


def get_persistent_color(pid1, pid2):
    """
    Maps a pair of pids to a consistent color using a stable hash.
    Using hashlib ensures the color remains exactly the same even if you
    restart your Python session/script entirely.
    """
    # Create a unique string identifier for this pair
    pair_id = f"{pid1}_{pid2}".encode("utf-8")

    # Create a stable integer hash from the string
    # We use MD5, grab the first 8 hex characters, and convert to an integer
    hash_int = int(hashlib.md5(pair_id).hexdigest()[:8], 16)

    # Modulo the hash by the number of available colors to get an index
    color_index = hash_int % num_colors

    # Return the RGBA color from the colormap
    return cmap(color_index)


def lookup_mult(mult_df, snap_idx, id):
    if (snap_idx, id) in mult_df.index:
        return mult_df.loc[(snap_idx, id)]["mult_ids_list_og"]
    else:
        return [id]


def get_com_wrapper(snap_idx, bin_id1, bin_id2, mult_lookup, particle_data):
    particle_ids = (bin_id1, bin_id2)
    if (bin_id2 == -999) and (len(mult_lookup) > 0):
        particle_ids = lookup_mult(mult_lookup, snap_idx, bin_id1)
    return get_com(particle_ids, particle_data)


def get_com(ids, part_data):
    (partpos, partvels, partmasses, partids) = part_data
    tmp_sel = np.where(np.isin(partids, ids))[0]
    tmp_pos = partpos[tmp_sel]
    tmp_vel = partvels[tmp_sel]
    tmp_mass = partmasses[tmp_sel]
    tmp_pos_vel = np.hstack((tmp_pos, tmp_vel))

    if len(tmp_mass) == 0:
        return tmp_pos_vel

    return np.average(tmp_pos_vel, axis=0, weights=tmp_mass)


def point_size_function(sep_pc, rmax):
    rmax = 0.2
    x = np.log(sep_pc)

    r0 = 1e3 * cgs.au / cgs.pc
    r1 = rmax * 3.0**0.5
    x0 = np.log(r0)
    x1 = np.log(r1)
    min_size = 6.0
    max_size = 20.0
    interp_size = np.exp(
        (x - x0) / (x1 - x0) * np.log(min_size)
        + (x - x1) / (x0 - x1) * np.log(max_size)
    )

    return np.clip(interp_size, min_size, max_size)


units.registry["au"] = AUnit()
colorblind_palette = sns.color_palette("colorblind")
# Set the matplotlib color cycle to the seaborn colorblind palette
plt.rcParams["axes.prop_cycle"] = plt.cycler(color=colorblind_palette)
plt.rcParams["lines.linewidth"] = 3
plt.rcParams["patch.linewidth"] = 3
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

snap_idx = config.getint("params", "snap_idx")
bin_id1 = config.getint("params", "bin1")
bin_id2 = config.getint("params", "bin2")
my_ft = config.get("params", "ft", fallback="1.0")
seed = config.getint("params", "seed", fallback=42)
rmax = config.getfloat("params", "rmax", fallback=0.5)
res = config.getint("params", "res", fallback=800)
savetype = config.get("params", "savetype", fallback="png")
vmin = config.getfloat("params", "vmin", fallback=1.0)
vmax = config.getfloat("params", "vmax", fallback=3e4)
plimit = config.getfloat("params", "plimit", fallback=-1)
ins = config.getfloat("params", "ins", fallback=-1.0)
ins_loc = config.get("params", "ins_loc", fallback="upper right")
annot = config.get("params", "annot", fallback="")
v_rescale = config.getfloat("params", "v_rescale", fallback=2)
center_file = config.get("params", "center_file", fallback=None)
center = config.get("params", "center", fallback=None)
center_time = config.getint("params", "center_time", fallback=snap_idx)
base = config.get(
    "params",
    "base",
    fallback=f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_",
)
snap_loc = config.get("params", "snap_loc", fallback=None)
tracer_file = config.get("params", "tracers", fallback="")
halo_lookup = config.get("params", "halo_lookup", fallback="")
mult_lookup = config.get("params", "mult_lookup", fallback="")
down_sample = config.getint("params", "down_sample", fallback=1)
arrow_opacity = config.getfloat("params", "arrow_opacity", fallback=0.8)
ms = config.getfloat("params", "ms", fallback=1)
ma = config.getfloat("params", "ma", fallback=1)


v_scale = 100.0 / cgs.pc * cgs.year * v_rescale
##snapshot interval in code units.
snap_time_code = 2.47e4 * cgs.year / (cgs.pc / 100.0)
d_cut = rmax
base = base + f"{seed}/"

# r2 = f"_TidesFalse_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse.p".replace(".p", "")
# aa = "analyze_multiples_output_" + r2 + "/"
if snap_loc is None:
    snap_loc = base
snap_file = snap_loc + f"snapshot_{snap_idx:03d}.hdf5"

out = find_multiples_new2.load_data(snap_file, res_limit=1e-3)
den = out["den"]
x = out["x"]
m = out["m"]
h = out["h"]
u = out["u"]
v = out["v"]
b = out["b"]
gas_ids = out["gas_ids"]

partpos = out["partpos"]
partmasses = out["partmasses"]
partvels = out["partvels"]
partids = out["partids"]
partsink = out["partsink"]
tage_myr = out["tage_myr"]

##NOTE THE CHANGE IN INDEXING HERE(!)
# xuniq, indx = np.unique(x, return_index=True, axis=0)
# 1. Create a boolean mask for rows that are entirely zeros
zero_rows_mask = (x == 0).all(axis=1)
indx = np.where(~zero_rows_mask)[0]
xuniq = x[indx]
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
if center_file is not None:
    centers = np.genfromtxt(center_file)
    center = centers[np.where(centers[:, 0] == snap_idx)[0][0]][1:]
elif center is not None:
    center = ast.literal_eval(center)
    center = np.array(center)
    center[:3] += center[3:] * (snap_idx - center_time) * snap_time_code


blookup = {}
if mult_lookup:
    mult_lookup = pd.read_parquet(mult_lookup)
    blookup = get_blookup(mult_lookup)
    mult_lookup = get_maximal_multiples(mult_lookup)

bin_center = get_com_wrapper(
    snap_idx, bin_id1, bin_id2, mult_lookup, (partpos, partvels, partmasses, partids)
)
if center is None:
    center = bin_center
center = np.array(center)

##ONLY SELECT GAS IN VOXEL AROUND STARS
sel2 = np.abs(xuniq - center[:3])
sel2 = (sel2[:, 0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:, 2] < d_cut)
sel2_gas = np.copy(sel2)

##GETTING SURFACE DENSITY VIA THE MESHOID PACKAGE
xuniq_center = xuniq - center[:3]
M = Meshoid(xuniq_center[sel2], muniq[sel2], huniq[sel2])
X = np.linspace(-rmax, rmax, res)
Y = np.linspace(-rmax, rmax, res)
X, Y = np.meshgrid(X, Y, indexing="ij")
sigma_gas_msun_pc2 = M.SurfaceDensity(
    M.m, size=2 * rmax, res=res, center=np.array((0, 0, 0))
)  # *1e4
############################################################################################################

fig, ax = plt.subplots(figsize=(8, 8), constrained_layout=True)
ax.set_xlabel("x [pc]")
ax.set_ylabel("y [pc]")
# ax.annotate(f"Example {annot}", (0.01, 0.99), xycoords='axes fraction', va="top", ha="left")

p = ax.pcolormesh(
    X,
    Y,
    sigma_gas_msun_pc2,
    norm=colors.LogNorm(vmin=vmin, vmax=vmax),
    cmap="viridis",
    linewidth=0,
    rasterized=True,
)
# ax.scatter(tmp_pos_center[0], tmp_pos_center[1],  marker="X", color="k", s=40)
# ax.scatter(tmp_pos2_center[0], tmp_pos2_center[1],  marker="X", color="k", s=40)
if plimit > 0:
    ax.set_xlim(-plimit, plimit)
    ax.set_ylim(-plimit, plimit)
else:
    ax.set_xlim(-rmax, rmax)
    ax.set_ylim(-rmax, rmax)

# fig.savefig(f"fig1_{sys.argv[1]}a_{snap_idx}." + savetype, dpi=300)
############################################################################################################
# dist_center = partpos - center[:3]
# dist_center = np.sum(dist_center * dist_center, axis=1)**.5
sel2 = np.abs(partpos - center[:3])
##Only include star partciles in the box...
dist_filter = (sel2[:, 0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:, 2] < d_cut)
partpos_filt = partpos[dist_filter]
partvel_filt = partvels[dist_filter]
partids_filt = partids[dist_filter]
partmasses_filt = partmasses[dist_filter]

if halo_lookup:
    halo_lookup = pd.read_parquet(halo_lookup)

#####Overlays of star paticles and stars
for ii in range(len(partpos_filt)):

    center_x, center_y = (
        partpos_filt[ii, 0] - center[0],
        partpos_filt[ii, 1] - center[1],
    )

    pid1_for_star_plot = partids_filt[ii]
    pid2_for_star_plot = blookup.get(
        ((int(snap_idx), int(pid1_for_star_plot))), pid1_for_star_plot
    )
    group_color_for_star = "k"
    if len(halo_lookup) > 0 and (
        (pid1_for_star_plot in halo_lookup["pid1"].to_numpy())
        or (pid1_for_star_plot in halo_lookup["pid2"].to_numpy())
    ):
        group_color1_for_star = np.array(
            get_persistent_color(pid1_for_star_plot, pid1_for_star_plot)
        )
        group_color2_for_star = np.array(
            get_persistent_color(pid2_for_star_plot, pid2_for_star_plot)
        )
        group_color_for_star = 0.5 * (group_color1_for_star + group_color2_for_star)

    if (bin_id1 in (pid1_for_star_plot, pid2_for_star_plot)) or (
        bin_id2 in (pid1_for_star_plot, pid2_for_star_plot)
    ):
        group_color_for_star = "red"
    if group_color_for_star != "k":
        print(
            snap_idx,
            "star color",
            pid1_for_star_plot,
            pid2_for_star_plot,
            group_color_for_star,
        )

    ax.plot(
        center_x,
        center_y,
        "X",
        color=group_color_for_star,
        # markersize=ms * np.log(partmasses_filt[ii] / 0.001),
        markersize=point_size_function(
            np.linalg.norm(partpos_filt[ii] - center[:3]), rmax
        ),
        # alpha=ma,
    )
    # # Create the circular patch comparable to the accretion radius...Really this is an upper bound(!)
    # hl_radius = 2. * sfc.GN * partmasses_filt[ii] / np.linalg.norm(partvel_filt[ii])**2.
    ##Arbitrary cutoff for gas neighbors...
    # gas_neighbors = np.linalg.norm(xuniq[sel2_gas] - partpos_filt[ii], axis=1) < 0.01
    # if len(xuniq[sel2_gas][gas_neighbors]) == 0:
    #     continue
    # gas_neighbors_vel = np.mean(vuniq[sel2_gas][gas_neighbors], axis=0)
    # gas_neighbors_cs = np.mean(u_to_cs(uuniq[sel2_gas][gas_neighbors]))
    # bhl_radius = (
    #     2.0
    #     * sfc.GN
    #     * partmasses_filt[ii]
    #     / (
    #         np.linalg.norm(partvel_filt[ii] - gas_neighbors_vel) ** 2.0
    #         + gas_neighbors_cs**2.0
    #     )
    # )
    # print(
    #     gas_neighbors_cs / 1e3,
    #     gas_neighbors_vel / 1e3,
    #     np.linalg.norm(partvel_filt[ii] - gas_neighbors_vel) / 1e3,
    #     np.linalg.norm(partvel_filt[ii]) / 1e3,
    # )

    # circle = patches.Circle(
    #     (center_x, center_y),
    #     bhl_radius,
    #     color="orange",
    #     fill=False,
    #     linewidth=2,
    #     label="Circle",
    # )
    # # Add the circle to the axes
    # ax.add_patch(circle)
arr_index1 = np.where(partids_filt.astype(int) == bin_id1)[0]
arr_index2 = np.where(partids_filt.astype(int) == bin_id2)[0]
ax.plot(
    partpos_filt[arr_index1, 0] - center[0],
    partpos_filt[arr_index1, 1] - center[1],
    "ro",
)
ax.plot(
    partpos_filt[arr_index2, 0] - center[0],
    partpos_filt[arr_index2, 1] - center[1],
    "ro",
)


# fig.savefig(f"fig1_{sys.argv[1]}b_{snap_idx}." + savetype, dpi=300)

# prop_cycle = plt.rcParams['axes.prop_cycle']
# colors = prop_cycle.by_key()['color']
colors = ["gold", "w"]
del sel2
del sel2_gas
del den
del denuniq
del uuniq
del huniq
# del muniq
gc.collect()
if tracer_file:
    tracer_data = np.genfromtxt(tracer_file)
    tracer_ids = tracer_data[:, 0]
    # is_accreted = tracer_data[:, 1].astype(bool)
    tracer_filt = np.isin(gas_ids, tracer_ids)
    ##Getting positions of tracer gas particles
    tmp_halo_pos = np.hstack(
        (xuniq[tracer_filt], vuniq[tracer_filt], muniq[tracer_filt, np.newaxis])
    )
    ##Making sure ids are in the same order...
    tracer_ids = gas_ids[tracer_filt]
    is_accreted = (
        pd.DataFrame(tracer_data, columns=("id", "acc"), dtype=int)
        .set_index("id")
        .loc[gas_ids[tracer_filt]]
    )
    is_accreted = is_accreted["acc"].to_numpy()  # .astype(bool)

    ##Only include halo particles in the Voxel
    sel2 = np.abs(tmp_halo_pos[:, :3] - center[:3])
    dist_filter = (sel2[:, 0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:, 2] < d_cut)
    try:
        tracer_ids = tracer_ids[dist_filter]
    except IndexError:
        breakpoint()

    tmp_halo_pos = tmp_halo_pos[dist_filter]
    is_accreted = is_accreted[dist_filter]
    if len(tmp_halo_pos) > 0:
        random_selection = np.random.choice(
            range(len(tmp_halo_pos)), len(tmp_halo_pos) // down_sample, replace=False
        )
        tmp_halo_pos = tmp_halo_pos[random_selection]
        is_accreted = is_accreted[random_selection]

        halo_com = np.average(tmp_halo_pos[:, :-1], axis=0, weights=tmp_halo_pos[:, -1])
        arrow_cols = [colors[row] for row in is_accreted.astype(int)]
        v_offset_x = halo_com[3]
        v_offset_y = halo_com[4]
        if len(bin_center) > 0:
            v_offset_x = bin_center[3]
            v_offset_y = bin_center[4]
        try:
            ##Change the velocity to always be relative to the star(?) Even if center is not in the star frame
            ax.quiver(
                tmp_halo_pos[:, 0] - center[0],
                tmp_halo_pos[:, 1] - center[1],
                (tmp_halo_pos[:, 3] - v_offset_x) * v_scale * snap_interval,
                (tmp_halo_pos[:, 4] - v_offset_y) * v_scale * snap_interval,
                scale=1,
                scale_units="xy",
                angles="xy",
                alpha=arrow_opacity,
                color=arrow_cols,
            )  # color=colors[int(partids_filt[ii]) % len(colors)])
        except IndexError:
            breakpoint()

        ##Coloring by halo
        ##MAKE SURE THAT HALO_LOOKUP HERE ONLY INCLUDES UNRELATED STARS(!)
        if len(halo_lookup) > 0:
            tmp_tracers_halo = halo_lookup.loc[tracer_ids]
            tmp_tracers_halo = tmp_tracers_halo.loc[
                tmp_tracers_halo["snap"] == int(snap_idx)
            ]
            tmp_tracers_halo_grouped = tmp_tracers_halo.groupby(["pid1", "pid2"])

            ##MAKE SURE ACCRETING STAR IS (!)
            # Iterate through the group name (pid1, pid2) and the actual group dataframe (group_df)
            for (pid1, pid2), group_df in tmp_tracers_halo_grouped:
                col = None
                ##IDEA: HAVE MAPPING BETWEEN PARTICLE ID AND COLOR...

                # 1. Plot the dataframe coordinates and save the line object
                # (Added marker='o' and linestyle='' assuming these are discrete points, remove if they are continuous lines)
                # lines = ax.plot(
                #     group_df["x"] - center[0],
                #     group_df["y"] - center[1],
                #     marker="o",
                #     linestyle="",
                #     color=col,
                # )
                group_color1 = np.array(get_persistent_color(pid1, pid1))
                group_color2 = np.array(get_persistent_color(pid2, pid2))
                group_color = 0.5 * (group_color1 + group_color2)
                if (bin_id1 in (pid1, pid2)) or (bin_id2 in (pid1, pid2)):
                    group_color = "red"
                print(snap_idx, "gas color", pid1, pid2, group_color)
                ##Change the velocity to always be relative to the star(?) Even if center is not in the star frame
                ax.quiver(
                    group_df["x"] - center[0],
                    group_df["y"] - center[1],
                    (group_df["vx"] - v_offset_x) * v_scale * snap_interval,
                    (group_df["vy"] - v_offset_y) * v_scale * snap_interval,
                    scale=1,
                    scale_units="xy",
                    angles="xy",
                    color=group_color,
                )  # color=colors[int(partids_filt[ii]) % len(colors)])

                # Extract the color matplotlib automatically assigned to this group
                tmp_star_pos1 = partpos[partids == pid1]
                tmp_star_pos2 = partpos[partids == pid2]

                # 2. Plot the path for pid1 using the exact same color
                # ax.scatter(
                #     tmp_star_pos1[0, 0] - center[0],
                #     tmp_star_pos1[0, 1] - center[1],
                #     color=group_color,
                #     alpha=0.7,  # Optional: Make the paths slightly transparent to distinguish them from the points
                #     marker="X",
                # )

                # # 3. Plot the path for pid2 using the exact same color
                # ax.scatter(
                #     tmp_star_pos2[0, 0] - center[0],
                #     tmp_star_pos2[0, 1] - center[1],
                #     color=group_color,
                #     alpha=0.7,
                #     marker="X",
                # )


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
# fig.savefig(f"fig1_{sys.argv[1]}d_{snap_idx}." + savetype, dpi=300)
