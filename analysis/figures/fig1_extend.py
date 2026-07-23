import ast
import colorsys
import configparser
import gc
import hashlib
import logging
import os
import pickle
import sys

import h5py
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
cmap = plt.get_cmap("turbo")
# num_colors = cmap.N
snap_interval = 2.47e4
conv = cgs.pc / cgs.au / 1e4


def deterministic_tracer_sample(tracer_ids, target_fraction=0.10, seed=42):
    """
    Selects a deterministic subset of tracers using an integer hash on tracer IDs.

    GEMINI
    """
    ids = np.asarray(tracer_ids, dtype=np.uint64)

    # Simple, fast vectorized bitwise hash (e.g., SplitMix64 style)
    x = ids ^ np.uint64(seed)
    x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    x = x ^ (x >> np.uint64(31))

    # Normalize hashed values to [0.0, 1.0]
    max_uint64 = float(np.iinfo(np.uint64).max)
    normalized = x.astype(np.float64) / max_uint64

    # Mask indicating which tracers to keep
    return normalized < target_fraction


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


# def get_persistent_color(pid1, pid2):
#     """
#     Maps a pair of pids to a consistent color using a stable hash.
#     Using hashlib ensures the color remains exactly the same even if you
#     restart your Python session/script entirely.
#     """
#     pid1 = int(pid1)
#     pid2 = int(pid2)
#     # Create a unique string identifier for this pair
#     pair_id = f"{pid1}_{pid2}".encode("utf-8")

#     # Create a stable integer hash from the string
#     # We use MD5, grab the first 8 hex characters, and convert to an integer
#     hash_int = int(hashlib.md5(pair_id).hexdigest()[:8], 16)

#     # Modulo the hash by the number of available colors to get an index
#     color_index = hash_int % num_colors

#     # Return the RGBA color from the colormap
#     return cmap(color_index)


# def get_persistent_color(pid):
#     """Generates an infinite variety of colors with a 'Dark2' vibe."""
#     pid_str = str(int(pid)).encode("utf-8")

#     # Generate an integer hash
#     hash_int = int(hashlib.md5(pid_str).hexdigest()[:8], 16)

#     # 1. Map the hash to a float between 0.0 and 1.0 to pick a Hue
#     # 0xFFFFFFFF is the maximum possible value for an 8-character hex string
#     hue = hash_int / 0xFFFFFFFF

#     # 2. Hardcode Saturation and Lightness to get that 'Dark2' aesthetic
#     # Lightness: 0.45 keeps it slightly dark. Saturation: 0.7 keeps it rich but not neon.
#     lightness = 0.45
#     saturation = 0.70

#     # 3. Convert back to RGB for matplotlib
#     r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)

#     return (r, g, b, 1.0)  # Return RGBA

# def get_persistent_color(pid1, pid2):
#     """
#     Maps a pair of pids to a consistent color using a stable hash.
#     Using hashlib ensures the color remains exactly the same even if you
#     restart your Python session/script entirely.
#     """
#     # Create a unique string identifier for this pair
#     pair_id = f"{pid1}_{pid2}".encode('utf-8')

#     # Create a stable integer hash from the string
#     # We use MD5, grab the first 8 hex characters, and convert to an integer
#     hash_int = int(hashlib.md5(pair_id).hexdigest()[:8], 16)

#     # Modulo the hash by the number of available colors to get an index
#     color_index = hash_int % num_colors

#     # Return the RGBA color from the colormap
#     return cmap(color_index)


def get_persistent_color(pid):
    pid_str = str(int(pid)).encode("utf-8")
    hash_int = int(hashlib.md5(pid_str).hexdigest()[:8], 16)

    # Modulo 256 since continuous colormaps typically have 256 bins
    color_index = hash_int % cmap.N

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
    partpos, partvels, partmasses, partids = part_data
    tmp_sel = np.where(np.isin(partids, ids))[0]
    tmp_pos = partpos[tmp_sel]
    tmp_vel = partvels[tmp_sel]
    tmp_mass = partmasses[tmp_sel]
    tmp_pos_vel = np.hstack((tmp_pos, tmp_vel))

    if len(tmp_mass) == 0:
        return tmp_pos_vel

    return np.average(tmp_pos_vel, axis=0, weights=tmp_mass)


# def point_size_function(sep_pc, rmax):
#     min_size = 6.0
#     max_size = 20.0
#     rmax = 0.2
#     if sep_pc == 0:
#         return max_size
#     x = np.log(sep_pc)

#     r0 = 1e3 * cgs.au / cgs.pc
#     r1 = rmax * 3.0**0.5
#     x0 = np.log(r0)
#     x1 = np.log(r1)

#     interp_size = np.exp(
#         (x - x0) / (x1 - x0) * np.log(min_size)
#         + (x - x1) / (x0 - x1) * np.log(max_size)
#     )

#     return np.clip(interp_size, min_size, max_size)


def point_size_function(sep_pc, rmax):
    min_size = 6.0 * 0.44
    max_size = 20.0 * 0.44
    # rmax = 0.2  # Un-comment if you strictly want to force 0.2!

    # Ensure input is at least a 1D array
    sep_pc = np.atleast_1d(sep_pc)

    # Mask to avoid log(0)
    is_zero = sep_pc == 0
    safe_sep = np.where(is_zero, 1.0, sep_pc)
    x = np.log(safe_sep)

    r0 = 1e3 * cgs.au / cgs.pc
    r1 = rmax * 3.0**0.5
    x0 = np.log(r0)
    x1 = np.log(r1)

    interp_size = np.exp(
        (x - x0) / (x1 - x0) * np.log(min_size)
        + (x - x1) / (x0 - x1) * np.log(max_size)
    )

    out_size = np.clip(interp_size, min_size, max_size)
    out_size[is_zero] = max_size  # Re-apply max size for zero-distance

    # Return scalar if a single float was passed, otherwise array
    return out_size[0] if out_size.size == 1 else out_size


def get_multiple_shift_info(
    pid1, snap_idx, mult_lookup, target_ids, particle_tuple, center
):
    """
    Checks if a star is part of a multiple system.
    If so, returns its shifted position (Center of Mass relative to plot center)
    and a boolean indicating whether a target star is in the system.

    Returns (None, False) if the star is not in a multiple system.
    GEMINI GENERATED--CHECK
    """
    if len(mult_lookup) == 0 or (int(snap_idx), pid1) not in mult_lookup.index:
        return None, False

    # Extract the list of IDs in this multiple system
    mult_row = mult_lookup.loc[(int(snap_idx), pid1)]
    mult_row_list = np.array(mult_row["mult_ids_list_og"]).astype(int)

    # Calculate the Center of Mass for the multiple
    mult_center = get_com(mult_row_list, particle_tuple)
    shifted_pos = mult_center - center

    # Check if any target star is inside this multiple system
    contains_target = any(t_id in mult_row_list for t_id in target_ids if t_id != -999)

    return shifted_pos, contains_target


def main():
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

    config = configparser.ConfigParser()
    config.read(f"config_{sys.argv[1]}")

    snap_idx = config.getint("params", "snap_idx")
    bin_id1 = config.getint("params", "bin1")
    bin_id2 = config.getint("params", "bin2")
    # Read bin3
    bin_id3 = config.getint("params", "bin3", fallback=-999)
    # my_ft = config.get("params", "ft", fallback="1.0")
    seed = config.getint("params", "seed", fallback=42)
    rmax = config.getfloat("params", "rmax", fallback=0.5)
    res = config.getint("params", "res", fallback=800)
    savetype = config.get("params", "savetype", fallback="png")
    vmin = config.getfloat("params", "vmin", fallback=1.0)
    vmax = config.getfloat("params", "vmax", fallback=3e4)
    plimit = config.getfloat("params", "plimit", fallback=-1)
    # ins = config.getfloat("params", "ins", fallback=-1.0)
    # ins_loc = config.get("params", "ins_loc", fallback="upper right")
    # annot = config.get("params", "annot", fallback="")
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
    trace_future = config.getboolean("params", "trace_future", fallback=False)
    logging.basicConfig(filename="my_log.log", level=logging.INFO)
    logger = logging.getLogger(__name__)

    v_scale = 100.0 / cgs.pc * cgs.year * v_rescale
    ##snapshot interval in code units.
    # snap_time_code = 2.47e4 * cgs.year / (cgs.pc / 100.0)
    d_cut = rmax
    base = base + f"{seed}/"

    # r2 = f"_TidesFalse_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse.p".replace(".p", "")
    # aa = "analyze_multiples_output_" + r2 + "/"
    if snap_loc is None:
        snap_loc = base
    snap_file = snap_loc + f"snapshot_{snap_idx:03d}.hdf5"
    snap_file_next = snap_loc + f"snapshot_{snap_idx:03d}.hdf5"

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
    tsnap_myr = out["tsnap_myr"]

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
    # parvels = partvels.astype(np.float64)
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
        center[:3] += center[3:] * (snap_idx - center_time)  # * snap_time_code

    # blookup = {}
    if mult_lookup:
        mult_lookup = pd.read_parquet(mult_lookup)
        # blookup = get_blookup(mult_lookup)
        mult_lookup = get_maximal_multiples(mult_lookup)

    bin_center = get_com_wrapper(
        snap_idx,
        bin_id1,
        bin_id2,
        mult_lookup,
        (partpos, partvels, partmasses, partids),
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

    fig, ax = plt.subplots()
    ax.set_xlabel("x [pc]")
    ax.set_ylabel("y [pc]")
    ax.tick_params(axis="both", which="both", color="0.5", labelcolor="k")
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(prune="lower"))
    # Force Matplotlib to draw the figure internally so tick labels exist
    fig.canvas.draw()

    # Grab the tick labels and hide the first one
    labels = ax.get_xticklabels()
    if labels:
        labels[0].set_visible(False)
    # ax.annotate(f"Example {annot}", (0.01, 0.99), xycoords='axes fraction', va="top", ha="left")
    ax.yaxis.set_ticks(np.arange(-rmax, rmax + 0.01, 0.2))

    ax.pcolormesh(
        X,
        Y,
        sigma_gas_msun_pc2,
        norm=colors.LogNorm(vmin=vmin, vmax=vmax),
        cmap="viridis",
        linewidth=0,
        rasterized=True,
    )
    # Add to fig1_extend.py after ax.pcolormesh
    ax.text(
        0.05,
        0.95,
        f"t = {tsnap_myr:.2f} Myr",
        transform=ax.transAxes,
        color="white",
        fontsize=10,
        fontweight="bold",
        va="top",
        ha="left",
    )

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
    # partvel_filt = partvels[dist_filter]
    partids_filt = partids[dist_filter]
    # partmasses_filt = partmasses[dist_filter]

    if halo_lookup:
        halo_lookup = pd.read_parquet(halo_lookup)

    star_data = []
    # 1. Calculate all distances and sizes simultaneously
    bin_id1_select = np.where(partids_filt == bin_id1)[0]
    if len(bin_id1_select) == 0:
        logger.info("Warning bin_id1 falls outside of domain! Skipping plot")
        return -1
    bin_id1_pos = partpos_filt[bin_id1_select[0]]
    dists = np.linalg.norm(partpos_filt - bin_id1_pos, axis=1)
    sizes = point_size_function(dists, rmax) ** 2

    # 2. Initialize coordinate and color arrays
    plot_x = partpos_filt[:, 0] - center[0]
    plot_y = partpos_filt[:, 1] - center[1]
    plot_z = partpos_filt[:, 2] - center[2]
    colors_arr = np.full(len(partpos_filt), "k", dtype=object)
    target_ids = {bin_id1, bin_id2, bin_id3}
    particle_tuple = (partpos, partvels, partmasses, partids.astype(int))

    # 3. Handle multiples logic
    for ii in range(len(partpos_filt)):
        pid1 = int(partids_filt[ii])
        # Color red if it's the exact target star
        if pid1 in target_ids:
            colors_arr[ii] = "r"

        # Fetch multiple system info if applicable
        shifted_pos, contains_target = get_multiple_shift_info(
            pid1, snap_idx, mult_lookup, target_ids, particle_tuple, center
        )

        # Apply the shift and color if the star is in a multiple system
        if shifted_pos is not None:
            plot_x[ii], plot_y[ii], plot_z[ii] = shifted_pos
            if contains_target:
                colors_arr[ii] = "r"

    # 4. Split into target and background arrays to control rendering order (zorder)
    target_mask = colors_arr == "r"
    bg_mask = ~target_mask

    # Plot Background Stars
    if np.any(bg_mask):
        ax.scatter(
            plot_x[bg_mask],
            plot_y[bg_mask],
            marker="X",
            c=colors_arr[bg_mask],
            edgecolors="black",
            linewidths=1.5,
            s=sizes[bg_mask],
            zorder=10,
        )

    # Plot Target Stars (Zorder 11 keeps them on top)
    if np.any(target_mask):
        ax.scatter(
            plot_x[target_mask],
            plot_y[target_mask],
            marker="X",
            c=colors_arr[target_mask],
            edgecolors="black",
            linewidths=1.5,
            s=sizes[target_mask],
            zorder=11,
        )

    # 5. Compile star_data exactly as you had it before
    star_data = list(
        zip(partids_filt, plot_x, plot_y, plot_z, colors_arr, np.sqrt(sizes))
    )

    # fig.savefig(f"fig1_{sys.argv[1]}b_{snap_idx}." + savetype, dpi=300)

    # prop_cycle = plt.rcParams['axes.prop_cycle']
    # colors = prop_cycle.by_key()['color']
    cols = ["gold", "w"]
    sel2 = None
    sel2_gas = None
    den = None
    denuniq = None
    uuniq = None
    huniq = None

    tracer_pv = []
    if tracer_file:
        ##TO DO: ADJUST SUB-SAMPLING HERE!
        tracer_data = np.genfromtxt(tracer_file)
        tracer_ids = tracer_data[:, 0]
        ##Deterministic tracer downsampling--commutative with filtering
        # tracer_ids = tracer_ids[
        #     deterministic_tracer_sample(
        #         tracer_ids, target_fraction=1.0 / down_sample, seed=42
        #     )
        # ]
        # is_accreted = tracer_data[:, 1].astype(bool)
        tracer_filt = np.isin(gas_ids, tracer_ids)
        ##Getting positions of tracer gas particles
        tmp_halo_pos = np.hstack(
            (xuniq[tracer_filt], vuniq[tracer_filt], muniq[tracer_filt, np.newaxis])
        )
        ##Making sure ids are in the same order...
        tracer_ids = gas_ids[tracer_filt]
        np.savez(
            f"tracers_full_{snap_idx}.npz",
            tracer_ids=tracer_ids,
            tracer_pos=tmp_halo_pos[:, :-1],
            center=center,
        )

        is_accreted = (
            pd.DataFrame(tracer_data, columns=("id", "acc"), dtype=int)
            .set_index("id")
            .loc[gas_ids[tracer_filt]]
        )
        is_accreted = is_accreted["acc"].to_numpy()  # .astype(bool)

        ##Only include halo particles in the Voxel
        sel2 = np.abs(tmp_halo_pos[:, :3] - center[:3])
        dist_filter = (sel2[:, 0] < d_cut) & (sel2[:, 1] < d_cut) & (sel2[:, 2] < d_cut)

        tmp_halo_pos = tmp_halo_pos[dist_filter]
        tracer_ids = tracer_ids[dist_filter]
        is_accreted = is_accreted[dist_filter]
        if len(tmp_halo_pos) > 0:
            # random_selection = np.random.choice(
            #     range(len(tmp_halo_pos)),
            #     len(tmp_halo_pos) // down_sample,
            #     replace=False,
            # )
            tracer_mask = deterministic_tracer_sample(
                tracer_ids, 1.0 / down_sample, seed=42
            )
            random_selection = np.where(tracer_mask)
            ##Dummy code--Since downsampling is already done, we keep all particles.
            # random_selection = np.array(range(len(tmp_halo_pos)))
            tmp_halo_pos = tmp_halo_pos[random_selection]
            tracer_ids = tracer_ids[random_selection]
            is_accreted = is_accreted[random_selection]

            # halo_com = np.average(tmp_halo_pos[:, :-1], axis=0, weights=tmp_halo_pos[:, -1])
            arrow_cols = np.array([cols[row] for row in is_accreted.astype(int)])
            v_offset_x = center[3]
            v_offset_y = center[4]

            if os.path.exists(f"tracers_full_{snap_idx + 1}.npz") and trace_future:
                tracers_future = np.load(f"tracers_full_{snap_idx + 1}.npz")

                ##TO DO: POINT TO STAR IF ACCRETED?
                ##Future tracer positions.
                future_pos = tracers_future["tracer_pos"] - tracers_future["center"]
                # Find the common IDs and get their exact indices in BOTH arrays.
                # assume_unique=True speeds this up since particle IDs are unique.
                common_ids, ind_curr, ind_fut = np.intersect1d(
                    tracer_ids,
                    tracers_future["tracer_ids"],
                    # assume_unique=True,
                    return_indices=True,
                )

                current_pos = (
                    tmp_halo_pos[:, 0][ind_curr] - center[0],
                    tmp_halo_pos[:, 1][ind_curr] - center[1],
                )
                arrow_cols = arrow_cols[ind_curr]
                delta_gas = (
                    future_pos[:, 0][ind_fut] - current_pos[0],
                    future_pos[:, 1][ind_fut] - current_pos[1],
                )

            else:
                current_pos = (
                    (tmp_halo_pos[:, 0] - center[0]),
                    (tmp_halo_pos[:, 1] - center[1]),
                )
                delta_gas = (
                    (tmp_halo_pos[:, 3] - v_offset_x) * v_scale * snap_interval,
                    (tmp_halo_pos[:, 4] - v_offset_y) * v_scale * snap_interval,
                )

            try:
                ##Change the velocity to always be relative to the star(?) Even if center is not in the star frame
                # tracer_pv.append(np.transpose((tmp_halo_pos[:, :6] - center), is_accreted))
                ax.quiver(
                    current_pos[0],
                    current_pos[1],
                    delta_gas[0],
                    delta_gas[1],
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

                for (pid1, pid2), group_df in tmp_tracers_halo_grouped:
                    col = None
                    ##IDEA: HAVE MAPPING BETWEEN PARTICLE ID AND COLOR...
                    group_color1 = np.array(get_persistent_color(pid1))
                    group_color2 = np.array(get_persistent_color(pid2))
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
                    # tmp_star_pos1 = partpos[partids == pid1]
                    # tmp_star_pos2 = partpos[partids == pid2]

    np.savez(
        "fig1_data_" + sys.argv[1] + f"_{snap_idx}.npz",
        star_data=star_data,
        tracer_pv=tracer_pv,
    )
    fig.savefig(f"fig1_{sys.argv[1]}d_{snap_idx}." + savetype, dpi=300)
    original_position = ax.get_position()
    ax.set_ylabel("")
    ax.yaxis.set_ticklabels([])
    ax.set_position(original_position)
    fig.savefig(f"fig1_{sys.argv[1]}d_{snap_idx}_noy." + savetype, dpi=300)


if __name__ == "__main__":
    main()
