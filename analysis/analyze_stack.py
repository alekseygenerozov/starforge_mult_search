import copy
import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd
from astropy.io import ascii
from astropy.table import vstack as astro_vstack
from numba import njit
from pytreegrav.kernel import PotentialKernel
from scipy.interpolate import interp1d

import starforge_mult_search.code.starforge_constants as sfc

LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_SMA = 6
LOOKUP_ECC = 7
LOOKUP_Q = 8

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
mcol = np.where(sink_cols == "m")[0][0]
mtotcol = np.where(sink_cols == "mtot")[0][0]
scol = np.where(sink_cols == "sys_id")[0][0]


def npz_stack(npz_list):
    """
    Stack data from different seeds
    """
    # Dictionary to hold lists of arrays for each key
    data_dict = defaultdict(list)

    # Loop over .npz files in the directory
    for filename in npz_list:
        # Load the .npz file
        data = np.load(filename, allow_pickle=True)

        # Iterate over keys in the .npz file
        for key in data.keys():
            data_dict[key].append(data[key])  # Append data for this key
    concatenated_data_dict = {
        key: np.concatenate(arrays) for key, arrays in data_dict.items()
    }

    return concatenated_data_dict


# @njit
def subtract_path(p1, p2):
    """
    Function to get displacement of 2 stars accounting for infinity placeholders
    """
    assert len(p1) == len(p2)
    diff = np.ones((len(p1), 3)) * np.inf
    filt = (~np.isinf(p1[:, 0])) & (~np.isinf(p2[:, 0]))
    diff[filt] = p1[filt] - p2[filt]

    return diff


def divide_path(p1, p2):
    """
    Function to get displacement of 2 stars accounting for infinity placeholders
    """
    assert len(p1) == len(p2)
    diff = np.ones((p1.shape[0], p1.shape[1])) * np.inf
    filt = (~np.isinf(p1[:, 0])) & (~np.isinf(p2[:, 0]))
    diff[filt] = p1[filt] / p2[filt]

    return diff


##Use different variable instead of mtot here...
@njit
def get_peri(x, y, z, vx, vy, vz, mtot, eps):
    """
    Compute 2-body pericenter--Given coordinates of relative positions and velocities.
    """
    GN = 4.301e3

    sep = np.sqrt(x * x + y * y + z * z)
    vrel = np.sqrt(vx * vx + vy * vy + vz * vz)
    ##Account for softening here
    en = -GN * mtot / (sep) + 0.5 * vrel * vrel
    ell = np.cross((x, y, z), (vx, vy, vz))
    ell = np.sqrt(ell[0] * ell[0] + ell[1] * ell[1] + ell[2] * ell[2])

    ##This formula must also be adjusted for softening--solve numerically, but watch out for multiple roots
    return (
        -GN
        * mtot
        / (2.0 * en)
        * (1.0 - np.sqrt(1.0 + 2.0 * en * ell**2.0 / (GN * mtot) ** 2.0))
    )


@njit
def phi_softened(r, mtot, eps):
    GN = 4.301e3
    return GN * mtot * PotentialKernel(r, eps)


@njit
def eff_pot(r, L2, mtot, eps):
    return 0.5 * L2 / (r * r) + phi_softened(r, mtot, eps)


@njit
def root_function(r, E, L2, mtot, eps):
    return E - eff_pot(r, L2, mtot, eps)


@njit
def bisect_root(E, L2, mtot, eps, a, b):
    rtol = 1e-8
    maxiter = 300
    G = 4.301e3

    fa = root_function(a, E, L2, mtot, eps)
    fb = root_function(b, E, L2, mtot, eps)

    if fa * fb > 0:
        return np.nan  # No bracketed root

    log_a = np.log10(a)
    log_b = np.log10(b)

    for _ in range(maxiter):
        log_mid = 0.5 * (log_a + log_b)
        mid = 10.0**log_mid
        fc = root_function(mid, E, L2, mtot, eps)

        if abs(log_b - log_a) < rtol:
            return mid

        if fa * fc < 0.0:
            log_b = log_mid
            fb = fc
        else:
            log_a = log_mid
            fa = fc

    return 10.0**log_mid  # return last midpoint if no convergence


@njit
def get_peri_softened_numba(x, y, z, vx, vy, vz, mtot, eps):
    r0 = np.sqrt(x * x + y * y + z * z)
    v2 = vx * vx + vy * vy + vz * vz
    phi = phi_softened(r0, mtot, eps)
    E = 0.5 * v2 + phi

    # Angular momentum squared
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    L2 = Lx * Lx + Ly * Ly + Lz * Lz

    rmin = 1e-20  # Avoid divide-by-zero
    rmax = r0  # Assume current sep is outside pericenter

    return bisect_root(E, L2, mtot, eps, rmin, rmax)


@njit
def get_apo_softened_numba(x, y, z, vx, vy, vz, mtot, eps):
    r0 = np.sqrt(x * x + y * y + z * z)
    v2 = vx * vx + vy * vy + vz * vz
    phi = phi_softened(r0, mtot, eps)
    E = 0.5 * v2 + phi

    # Angular momentum squared
    Lx = y * vz - z * vy
    Ly = z * vx - x * vz
    Lz = x * vy - y * vx
    L2 = Lx * Lx + Ly * Ly + Lz * Lz

    rmin = r0
    rmax = 10 * r0

    return bisect_root(E, L2, mtot, eps, rmin, rmax)


@njit
def subtract_path_opt(p1, p2, dir=1):
    """
    Efficiently compute p1 - p2, skipping rows where either is [inf, inf, inf]
    """
    n = p1.shape[0]
    # diff = np.empty((n, 3))
    d = np.ones(n) * np.inf
    angs = np.zeros(n)

    # First pass: compute angs and mark invalid entries
    for i in range(n):
        if np.isinf(p1[i, 0]) or np.isinf(p2[i, 0]):
            angs[i] = 0
        else:
            dx = p1[i, 0] - p2[i, 0]
            dy = p1[i, 1] - p2[i, 1]
            dz = p1[i, 2] - p2[i, 2]
            dvx = p1[i, 3] - p2[i, 3]
            dvy = p1[i, 4] - p2[i, 4]
            dvz = p1[i, 5] - p2[i, 5]
            angs[i] = dx * dvx + dy * dvy + dz * dvz

    for i in range(n):
        if np.isinf(p1[i, 0]) or np.isinf(p2[i, 0]):
            d[i] = np.inf
        else:
            dx = p1[i, 0] - p2[i, 0]
            dy = p1[i, 1] - p2[i, 1]
            dz = p1[i, 2] - p2[i, 2]
            dvx = p1[i, 3] - p2[i, 3]
            dvy = p1[i, 4] - p2[i, 4]
            dvz = p1[i, 5] - p2[i, 5]
            mtot = p1[i, 6] + p2[i, 6]
            eps = max(p1[i, 7], p2[i, 7])
            ##Addition criterion: if bound and orbital period is the less than interval(!!)--Need a way to compute the softened orbital period...
            ##Need ability to do both forward and backward integration...
            if (dir > 0) and (i < n) and (angs[i] * angs[i + 1] < 0):
                d[i] = get_peri_softened_numba(dx, dy, dz, dvx, dvy, dvz, mtot, hcol)
            elif (dir < 0) and (i > 0) and (angs[i] * angs[i - 1] < 0):
                d[i] = get_peri_softened_numba(dx, dy, dz, dvx, dvy, dvz, mtot, hcol)
            else:
                d[i] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return d


@njit
def subtract_path_opt_vanilla(p1, p2):
    """
    Efficiently compute p1 - p2, skipping rows where either is [inf, inf, inf]
    """
    n = p1.shape[0]
    # diff = np.empty((n, 3))
    d = np.empty(n)

    for i in range(n):
        if np.isinf(p1[i, 0]) or np.isinf(p2[i, 0]):
            d[i] = np.inf
        else:
            dx = p1[i, 0] - p2[i, 0]
            dy = p1[i, 1] - p2[i, 1]
            dz = p1[i, 2] - p2[i, 2]
            d[i] = (dx * dx + dy * dy + dz * dz) ** 0.5
    return d


def subtract_path_1d(p1, p2):
    assert len(p1) == len(p2)
    diff = np.ones(len(p1)) * np.inf
    filt = (~np.isinf(p1)) & (~np.isinf(p2))
    diff[filt] = p1[filt] - p2[filt]

    return diff


def max_w_infinite(p1):
    """
    Maximum of ID array with infinity placeholders.
    """
    if np.all(np.isinf(p1)):
        return np.inf
    else:
        return np.max(p1[~np.isinf(p1)])


def get_min_dist_binary(path_lookup, tmp_row, two_body):
    """
    Get time series of separations between binary and other stars
    """
    p1_raw = path_lookup[tmp_row[0]]
    p2_raw = path_lookup[tmp_row[1]]
    path_lookup_keys = path_lookup.keys()

    my_subtract_func = subtract_path_opt_vanilla
    if two_body:
        my_subtract_func = subtract_path_opt

    path_diff_all = []
    keys_all = []
    for ii, uu in enumerate(path_lookup_keys):
        # Want only closest approach of stars external to the binary.
        if uu in tmp_row:
            continue
        ##Filtering out other seeds? Could be done more robustly/elegantly
        if len(path_lookup[uu]) != len(p1_raw):
            continue

        ##Displacement from binary com
        path_diff1 = my_subtract_func(
            path_lookup[uu][:, pxcol : pzcol + 1], p1_raw[:, pxcol : pzcol + 1]
        )
        # path_diff1 = np.sum(path_diff1 * path_diff1, axis=1)**.5
        path_diff2 = my_subtract_func(
            path_lookup[uu][:, pxcol : pzcol + 1], p2_raw[:, pxcol : pzcol + 1]
        )
        # path_diff2 = np.sum(path_diff2 * path_diff2, axis=1)**.5
        path_diff = np.min((path_diff1, path_diff2), axis=0)
        path_diff_all.append(path_diff)
        keys_all.append(uu)

    keys_all = np.array(keys_all)
    ##Why is this transposition necessary? Couldn't we just change to axis=0 in the following line...
    path_diff_all = np.array(path_diff_all).T
    closest_idx = np.argmin(path_diff_all, axis=1)
    closest_val = path_diff_all[np.arange(path_diff_all.shape[0]), closest_idx]
    del path_diff_all

    return closest_val, closest_idx, keys_all[closest_idx]


def get_sigma(vels):
    """
    Get velocity distribution from list of velocities

    :vels (list-like): List

    :return: 3D velocity dispersion
    """
    sigma_x = np.std(vels[:, 0])
    sigma_y = np.std(vels[:, 1])
    sigma_z = np.std(vels[:, 2])

    return (sigma_x**2.0 + sigma_y**2.0 + sigma_z**2.0) ** 0.5


def get_com_series(path_lookup, tmp_row):
    """
    Get velocity distribution from list of velocities

    :path_lookup (dict): Lookup dictionary for particle paths over time...
    :tmp_row (list-like): List of particles.

    :return: Time series of COM positions and velocities.
    """
    coms = np.zeros((len(path_lookup[tmp_row[0]]), 6))
    tot_mass = np.zeros(len(path_lookup[tmp_row[0]]))
    for part in tmp_row:
        coms += (
            path_lookup[part][:, pxcol : vzcol + 1]
            * path_lookup[part][:, mcol][:, np.newaxis]
        )
        tot_mass += path_lookup[part][:, mcol]
    coms = divide_path(coms, tot_mass[:, np.newaxis])
    return coms, tot_mass


##TO DO: GENERALIZE FOR ARBITRARY COLLECTIONS OF STARS[?]
##TO DO: CLEAN UP INDEXING--MAKE IT EXPLICIT THAT COMPANIONS_FIRST_STAR ARE INTS
def get_dynamics_binary(path_lookup, tmp_row, two_body, nneighbors=16, mult_table=None):
    """
    Get time series of separations between binary and other stars
    """
    p_raw = [path_lookup[part] for part in tmp_row]
    path_lookup_keys = path_lookup.keys()

    my_subtract_func = subtract_path_opt_vanilla
    if two_body:
        my_subtract_func = subtract_path_opt

    path_diff_all = []
    keys_all = []
    if mult_table is not None:
        companions_first_star = []
        # print(mult_table)
        if int(tmp_row[0]) in mult_table.index.get_level_values(level="mult_ids_list"):
            mult_table = mult_table.xs(int(tmp_row[0]), level="mult_ids_list").copy()
            mult_table["t"] = mult_table.index
            mult_table = mult_table.explode("mult_ids_list_og").set_index(
                ["t", "mult_ids_list_og"]
            )
            companions_first_star = mult_table.index.get_level_values(
                "mult_ids_list_og"
            )
    for ii, uu in enumerate(path_lookup_keys):
        # Want only closest approach of stars external to the group.
        if uu in tmp_row:
            continue
        ##Filtering out other seeds. Could be done more robustly/elegantly --
        ##e.g. to deal with the edge case that different seeds could have the same number of snapshots
        if len(path_lookup[uu]) != len(p_raw[0]):
            continue
        ##Filtering out higher multiples.
        overlap_times = np.array([], dtype=int)
        if mult_table is not None:
            ##Could be cleaner / more symmetric[?]
            ##Useful for filtering out higher multiples -- assuming tmp_row corresponds to a bound pair...
            ##Could move the first selection out of the loop for efficiency.a
            # overlap_times = mult_table.loc[
            #     lambda df: df["mult_ids_list_og"].apply(lambda lst: uu in lst)].index.values
            if int(uu) in companions_first_star:
                overlap_times = mult_table.xs(
                    int(uu), level="mult_ids_list_og"
                ).index.values

        ##Displacement from binary stars
        path_diff = [
            my_subtract_func(
                path_lookup[uu][:, pxcol : pzcol + 1], tmp_path[:, pxcol : pzcol + 1]
            )
            for tmp_path in p_raw
        ]
        # path_diff1 = np.sum(path_diff1 * path_diff1, axis=1)**.5
        # path_diff2 = my_subtract_func(path_lookup[uu][:, pxcol:pzcol + 1], p2_raw[:, pxcol:pzcol + 1])
        ##Patch for higher companions??

        ##Take minimum of distances from two stars...
        path_diff = np.min(path_diff, axis=0)
        ##Trick to exclude particles that are in the same multiple...
        path_diff[overlap_times] = np.inf

        path_diff_all.append(path_diff)
        keys_all.append(uu)

    ##Getting coms of stars over all times...
    coms_row, tot_mass_row = get_com_series(path_lookup, tmp_row)
    keys_all = np.array(keys_all)
    path_diff_all = np.array(path_diff_all).T
    ##Note argmpartition will *not* give the sorted order.
    partition = np.argpartition(path_diff_all, nneighbors)
    keys_closest = keys_all[partition][:, :nneighbors]

    ##Placeholder for everything 0...(i.e. the particles does not exist yet or there are not enough neighbors)
    ndens = np.zeros((len(keys_closest), nneighbors))
    # sigmas = np.ones(len(keys_closest)) * np.inf
    mass_tot_closest = np.zeros((len(keys_closest), nneighbors))
    mass_closest = np.zeros((len(keys_closest), nneighbors))
    sigma = np.zeros((len(keys_closest), nneighbors))
    coll_rate = np.zeros((len(keys_closest), nneighbors))
    coll_rate_focused = np.zeros((len(keys_closest), nneighbors))

    ##Iterating over all times
    for ii, row in enumerate(keys_closest):
        dist_neighbors = path_diff_all[ii, partition[ii, :nneighbors]]
        order = np.argsort(dist_neighbors)
        ##Trying to do n-densities simultaneously
        ndens[ii] = np.array(
            [
                (nn + 1) / (4.0 * np.pi / 3.0) / dist_neighbors[order[nn]] ** 3
                for nn in range(nneighbors)
            ]
        )
        v_neighbors = np.array([path_lookup[kk][ii, vxcol : vzcol + 1] for kk in row])[
            order
        ]

        mass_neighbors = np.array([path_lookup[kk][ii, mcol] for kk in row])[order]
        mass_closest[ii] = np.array(
            [np.mean(mass_neighbors[: nn + 1]) for nn in range(nneighbors)]
        )
        mass_neighbors = np.array([path_lookup[kk][ii, mtotcol] for kk in row])[order]
        mass_tot_closest[ii] = np.array(
            [np.mean(mass_neighbors[: nn + 1]) for nn in range(nneighbors)]
        )
        # mass_tot_closest[ii] = np.mean([path_lookup[kk][ii, mtotcol] for kk in row])
        ##Need to add the velocity dispersion of of the star itself...

        # sigma[ii] = np.array(
        #     [
        #         get_sigma(np.vstack((v_neighbors[: nn + 1], coms_row[ii, 3:])))
        #         for nn in range(nneighbors)
        #     ]
        # )
        # ##Hard-coded for a target size of 1e4 au
        # b = 0.048
        # coll_rate[ii] = ndens[ii] * sigma[ii] * np.pi * b**2.0
        # coll_rate_focused[ii] = coll_rate[ii] * (
        #     1
        #     + 2.0
        #     * sfc.GN
        #     * (tot_mass_row[ii] + mass_closest[ii])
        #     / (b * sigma[ii] ** 2.0)
        # )

    return {
        "mass_closest": mass_closest,
        "mass_tot_closest": mass_tot_closest,
        "keys_closest": keys_closest,
        "ndens": ndens,
        # "sigma": sigma,
        # "coll_rate": coll_rate,
        # "coll_rate_focused": coll_rate_focused,
    }
    # return {"sigma": sigmas, "mass_closest": mass_closest, "mass_tot_closest":mass_tot_closest, "keys_closest":keys_closest, "ndens":ndens}


# def get_min_dist_binary_og(path_lookup, tmp_row):
#     """
#     Get time series of separations between binary and other stars
#     """
#     p1_raw = path_lookup[tmp_row[0]]
#     p2_raw = path_lookup[tmp_row[1]]
#     path_lookup_keys = path_lookup.keys()
#
#     path_diff_all = []
#     for ii, uu in enumerate(path_lookup_keys):
#         #Want only closest approach of stars external to the binary.
#         if uu in tmp_row:
#             continue
#         ##Filtering out other seeds? Could be done more robustly/elegantly
#         if len(path_lookup[uu]) != len(p1_raw):
#             continue
#
#         ##Displacement from binary com
#         path_diff1 = subtract_path(path_lookup[uu][:, pxcol:pzcol + 1], p1_raw[:, pxcol:pzcol + 1])
#         path_diff1 = np.sum(path_diff1 * path_diff1, axis=1)**.5
#         path_diff2 = subtract_path(path_lookup[uu][:, pxcol:pzcol + 1], p2_raw[:, pxcol:pzcol + 1])
#         path_diff2 = np.sum(path_diff2 * path_diff2, axis=1)**.5
#         path_diff = np.min((path_diff1, path_diff2), axis=0)
#         path_diff_all.append(path_diff)
#
#     path_diff_all = np.array(path_diff_all).T
#     path_diff_all_order = np.argsort(path_diff_all, axis=1)
#     path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)
#
#     return path_diff_all


def get_closest_star_time_series(path_lookup, my_key, two_body=False, dir=1):
    p1_raw = path_lookup[my_key]
    ##Filtering out other seeds? Could be done more robustly/elegantly
    path_lookup_keys = np.array(list(path_lookup.keys()))
    nsnaps = np.array([len(path_lookup[kk]) for kk in path_lookup_keys])
    path_lookup_keys = path_lookup_keys[nsnaps == len(p1_raw)]

    path_diff_all = []
    for ii, uu in enumerate(path_lookup_keys):
        ##Exclude the star itself
        if uu == my_key:
            continue

        ##Getting separations for all particles...
        tmp_path1 = path_lookup[uu][
            :, [pxcol, pycol, pzcol, vxcol, vycol, vzcol, mcol, hcol]
        ]
        tmp_path2 = p1_raw[:, [pxcol, pycol, pzcol, vxcol, vycol, vzcol, mcol, hcol]]
        if two_body:
            path_diff = subtract_path_opt(tmp_path1, tmp_path2, dir=dir)
        else:
            path_diff = subtract_path_opt_vanilla(tmp_path1, tmp_path2)
        # path_diff = np.sum(path_diff * path_diff, axis=1)**.5
        path_diff_all.append(path_diff)
    path_diff_all = np.array(path_diff_all).T
    # path_diff_all_order = np.argsort(path_diff_all, axis=1)
    # path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)
    closest_idx = np.argmin(path_diff_all, axis=1)
    closest_val = path_diff_all[np.arange(path_diff_all.shape[0]), closest_idx]
    del path_diff_all
    keys = path_lookup_keys[path_lookup_keys != my_key][closest_idx]
    closest_comp = [
        [
            my_key,
            keys[ii],
            path_lookup[keys[ii]][ii, mcol],
            path_lookup[keys[ii]][ii, mtotcol],
            closest_val[ii],
            path_lookup[keys[ii]][ii, 0],
        ]
        for ii in range(len(keys))
    ]
    closest_comp = np.array(closest_comp)
    filt = ~np.isinf(closest_comp[:, -2].astype(float))

    return closest_comp[filt]


def get_closest_star_time_series_mem_opt(path_lookup, my_key):
    p1_raw = path_lookup[my_key]
    ##Filtering out other seeds? Could be done more robustly/elegantly
    path_lookup_keys = np.array(list(path_lookup.keys()))
    nsnaps = np.array([len(path_lookup[kk]) for kk in path_lookup_keys])
    path_lookup_keys = path_lookup_keys[nsnaps == len(p1_raw)]

    # path_diff_all = []
    min_dists = np.full(len(p1_raw), np.inf)
    min_keys = np.full(len(p1_raw), "", dtype=object)
    for ii, uu in enumerate(path_lookup_keys):
        ##Exclude the star itself
        if uu == my_key:
            continue
        path_diff = subtract_path_opt(
            path_lookup[uu][:, pxcol : pzcol + 1], p1_raw[:, pxcol : pzcol + 1]
        )
        update_mask = path_diff < min_dists

        min_dists[update_mask] = path_diff[update_mask]
        min_keys[update_mask] = uu

    # print(min_keys)
    closest_comp = [
        [
            my_key,
            min_keys[ii],
            path_lookup[min_keys[ii]][ii, mcol],
            path_lookup[min_keys[ii]][ii, mtotcol],
            min_dists[ii],
        ]
        for ii in range(len(min_keys))
        if not np.isinf(min_dists[ii])
    ]

    return np.array(closest_comp)


# def get_t90(path_lookup, my_key):
#     p1_raw = path_lookup[my_key]
#     p1_raw = p1_raw[~np.isinf(p1_raw[:,0])]
#
#     m_end = p1_raw[-1, mcol]
#     m_series = p1_raw[:, mcol]
#     t_series = p1_raw[:, 0]
#     idx_crit = np.where(m_series < 0.9 * m_end)[0]
#     if len(idx_crit)==0:
#         return t_series[0], 0
#     else:
#         return interp1d([m_series[idx_crit[-1]], m_series[idx_crit[-1] + 1]], [t_series[idx_crit[-1]], t_series[idx_crit[-1] + 1]])


def get_t90(path_lookup, my_key):
    p1_raw = path_lookup[my_key]
    p1_raw = p1_raw[~np.isinf(p1_raw[:, 0])]

    m_end = p1_raw[-1, mcol]
    m_series = p1_raw[:, mcol]
    t_series = p1_raw[:, 0]
    idx_crit = np.where(m_series < 0.9 * m_end)[0]
    if len(idx_crit) == 0:
        return t_series[0], 0, t_series[0]
    else:
        t90_abs = interp1d(
            [m_series[idx_crit[-1]], m_series[idx_crit[-1] + 1]],
            [t_series[idx_crit[-1]], t_series[idx_crit[-1] + 1]],
        )(0.9 * m_end)
        return t90_abs, t90_abs - t_series[0], t_series[0]


def get_t90_series(t_series, m_series):
    m_end = m_series[-1]
    idx_crit = np.where(m_series < 0.9 * m_end)[0]
    if len(idx_crit) == 0:
        return t_series[0], 0, t_series[0]
    else:
        t90_abs = interp1d(
            [m_series[idx_crit[-1]], m_series[idx_crit[-1] + 1]],
            [t_series[idx_crit[-1]], t_series[idx_crit[-1] + 1]],
        )(0.9 * m_end)
        return t90_abs, t90_abs - t_series[0], t_series[0]


##Only do 1 seed at a time
# def get_closest_star_time_series_transposed(path_lookup_time, my_key):
#     p1_raw = path_lookup[my_key]
##Filtering out other seeds? Could be done more robustly/elegantly
#
# path_diff_all = []
# for ii, uu in enumerate(path_lookup_keys):
#     ##Getting separations for all particles...
#     path_diff = subtract_path_opt(path_lookup[uu][:, pxcol:pzcol + 1], p1_raw[:, pxcol:pzcol + 1])
#     # path_diff = np.sum(path_diff * path_diff, axis=1)**.5
#     path_diff_all.append(path_diff)
# path_diff_all = np.array(path_diff_all).T
# # path_diff_all_order = np.argsort(path_diff_all, axis=1)
# # path_diff_all = np.take_along_axis(path_diff_all, path_diff_all_order, axis=1)
# closest_idx = np.argmin(path_diff_all, axis=1)
# closest_val = path_diff_all[np.arange(path_diff_all.shape[0]), closest_idx]
#
# keys = path_lookup_keys[path_lookup_keys!=my_key][closest_idx]
# closest_comp = [[my_key, keys[ii], path_lookup[keys[ii]][ii, mcol], path_lookup[keys[ii]][ii, mtotcol], closest_val[ii]] for ii in range(len(keys))]
# closest_comp = np.array(closest_comp)
# filt = ~np.isinf(closest_comp[:,-1].astype(float))

# return closest_comp[filt]


def get_closest_star_time_series_T(path_lookup, my_key, t):
    p1_raw = path_lookup[my_key]
    if np.isinf(p1_raw[t, 0]):
        return "blank", np.inf
    path_lookup_keys = np.array(list(path_lookup.keys()))
    nsnaps = np.array([len(path_lookup[kk]) for kk in path_lookup_keys])
    path_lookup_keys = path_lookup_keys[nsnaps == len(p1_raw)]

    pos_all = np.array(
        [path_lookup[kk][t, pxcol : pzcol + 1] for kk in path_lookup_keys]
    )
    delta = pos_all - p1_raw[t][pxcol : pzcol + 1]
    delta = np.sum(delta * delta, axis=1) ** 0.5
    order = np.argsort(delta)

    return path_lookup_keys[order[1]], delta[order[1]]


#
# def get_closest_star_time_series_exp(path_lookup, my_key):
#     p1_raw = path_lookup[my_key]
#
#     path_lookup_keys = np.array(list(path_lookup.keys()))
#     nsnaps = np.array([len(path_lookup[kk]) for kk in path_lookup_keys])
#     path_lookup_keys = path_lookup_keys[nsnaps == len(p1_raw)]
#
#     pos_all = np.array([path_lookup[kk][:, pxcol:pzcol + 1] for kk in path_lookup_keys])
#     delta = pos_all - p1_raw[:, pxcol:pzcol + 1]
#     delta = np.sum(delta * delta, axis=2) ** .5
#     delta = delta.T
#     order = np.argsort(delta, axis=1)
#
#     return path_lookup_keys[order[:, 1]], np.take_along_axis(delta, order, axis=1)[:, 1]

# def var_g23(N, k):
#     return (N - k + 1.) * ( k + 1.) / (N + 3.) / (N + 2.)**2.


def var_g23(N, k):
    return (N - k + 1.0) * (k + 1.0) / (N + 3.0) / (N + 2.0) ** 2.0


def make_binned_data_cont(absc, ords, bins):
    """
    Binning of (boolean) ords according to absc and bins
    """
    binned_num = np.zeros(len(bins) - 1)
    binned_err = np.zeros(len(bins) - 1)
    binned_err2 = np.zeros(len(bins) - 1)

    for bidx in range(1, len(bins)):
        tmp_filt = (absc >= bins[bidx - 1]) & (absc < bins[bidx])
        tmp_ords = ords[tmp_filt]
        tmp_ords = tmp_ords[~np.isinf(tmp_ords)]

        binned_num[bidx - 1] = np.mean(tmp_ords)
        binned_err[bidx - 1] = np.std(tmp_ords)
        binned_err2[bidx - 1] = np.std(tmp_ords) / (len(tmp_ords)) ** 0.5

    return binned_num, binned_err, binned_err2


def make_binned_data_cont_rev_err(absc, ords, bins):
    """
    Binning of (boolean) ords according to absc and bins
    """
    binned_num = np.zeros(len(bins) - 1)
    binned_err = np.zeros(len(bins) - 1)
    binned_err2 = np.zeros(len(bins) - 1)

    for bidx in range(1, len(bins)):
        tmp_filt = (absc >= bins[bidx - 1]) & (absc < bins[bidx])
        tmp_ords = ords[tmp_filt]
        tmp_ords = tmp_ords[~np.isinf(tmp_ords)]

        if len(tmp_ords > 0):
            binned_num[bidx - 1] = np.median(tmp_ords)
            binned_err[bidx - 1] = np.percentile(tmp_ords, 15.86)
            binned_err2[bidx - 1] = np.percentile(tmp_ords, 84.13)
        else:
            binned_num[bidx - 1] = np.inf
            binned_err[bidx - 1] = np.inf
            binned_err2[bidx - 1] = np.inf

    return binned_num, binned_err, binned_err2


def make_binned_data(absc, ords, bins):
    """
    Binning of (boolean) ords according to absc and bins
    """
    binned_num = np.zeros(len(bins) - 1)
    binned_den = np.zeros(len(bins) - 1)
    binned_numu = np.zeros(len(bins) - 1)
    true_err = np.zeros(len(bins) - 1)
    for bidx in range(1, len(bins)):
        tmp_filt = (absc >= bins[bidx - 1]) & (absc < bins[bidx])
        tmp_ords = ords[tmp_filt]

        binned_num[bidx - 1] = len(tmp_ords[tmp_ords > 0])
        binned_numu[bidx - 1] = len(tmp_ords[tmp_ords > 0]) ** 0.5
        binned_den[bidx - 1] = len(tmp_ords)
        true_err[bidx - 1] = var_g23(len(tmp_ords), len(tmp_ords[tmp_ords > 0])) ** 0.5

    return binned_num, binned_numu, binned_den, true_err


def get_soft_times(id1, id2, path_lookup):
    d12 = subtract_path(
        path_lookup[f"{id1}"][:, pxcol : pzcol + 1],
        path_lookup[f"{id2}"][:, pxcol : pzcol + 1],
    )
    d12 = np.sum(d12 * d12, axis=1) ** 0.5
    hmax = np.max(
        (path_lookup[f"{id1}"][:, hcol], path_lookup[f"{id2}"][:, hcol]), axis=0
    )
    soft_times = np.where(d12 < hmax)[0]

    return soft_times


def get_fpaths(base_path, cloud_tag, seed, analysis_tag, v_str="."):
    """
    Auxiliary function for generating file paths.
    """
    sim_tag = f"{cloud_tag}_{seed}"
    cloud_tag_split = cloud_tag.split("_")
    cloud_tag0 = f"{cloud_tag_split[0]}_{cloud_tag_split[1]}"

    base = base_path + f"/{v_str}/{cloud_tag0}/{sim_tag}/"
    r1 = base_path + f"/{v_str}/{cloud_tag0}/{sim_tag}/{cloud_tag_split[0]}_snapshot_"
    r2 = "_" + analysis_tag
    base_sink = base + f"/sinkprop/{cloud_tag_split[0]}_snapshot_"

    return base, base_sink, r1, r2, cloud_tag0, sim_tag


def get_snap_info(base, base_sink):
    """
    Getting info about snapshot files -- cadence (difference between snapshot numbers), snapshot time intervel (yr),
    start_snap (first snapshot number), end_snap (last snapshot number)
    """
    snaps = [
        xx.replace(base_sink, "").replace(".sink", "")
        for xx in glob.glob(base_sink + "*.sink")
    ]
    snaps = np.array(snaps).astype(int)
    cadence = np.diff(np.sort(snaps))[0]
    snap_interval = np.atleast_1d(
        np.genfromtxt(base + "/sinkprop/snap_interval")
    ).astype(float)
    ##Get snapshot numbers automatically
    start_snap = min(snaps)
    end_snap = max(snaps)

    return cadence, snap_interval, start_snap, end_snap


def get_end_time_set(my_set, path_lookup):
    """
    Get final time each of star of my_set exists and the maximum mass (primary of that set)...
    """
    ps = [path_lookup[str(ss)][:, [0, mcol]] for ss in my_set]
    ps = np.array(ps)

    ps = np.swapaxes(ps, 0, 1)
    end_stars_row = ps[~np.isinf(np.mean(ps[:, :, 0], axis=1))][-1]
    return end_stars_row[0, 0], max(end_stars_row[:, 1])


def get_bound_snaps_adjust(bin_list, high_df):
    ##Use high_df table to get more stringent binary snapshots(!!!)
    curr_bin_list = copy.copy(bin_list)
    curr_bin_list.sort()
    bin_sel = high_df.loc[str(curr_bin_list)]

    return bin_sel


def get_star_mapping_closest(high_df):
    """Transform multiples table to be indexed by stars, picking out minimal multiple for each one.

    :param high_df: Multiples data from starforge simulation
    :type high_df: Pandas dataframe
    :return: "Exploded" dataframe indexed by star id. Each id will have one row that corresponds to "minimal" multiples containing that id
    :rtype: Pandas dataframe
    """
    df = high_df.copy()
    df["mult_len"] = df["mult_ids_list"].apply(len)

    # Step 2: Explode to have one row per star
    df_exploded = df.explode("mult_ids_list")
    df_exploded["mult_ids_list_og"] = high_df["mult_ids_list"].copy()

    # Step 3: Sort so the longest lists come first
    df_exploded = df_exploded.sort_values("mult_len", ascending=True)
    # Step 6: Create a mapping: star_id → row with longest mult_ids_list containing it
    star_to_row = df_exploded.drop_duplicates(
        subset="mult_ids_list", keep="first"
    ).set_index(
        "mult_ids_list"
    )  # or .set_index("id") if you prefer row IDs

    # star_mapping = star_to_row.groupby(
    #     "mult_ids_list"
    # ).first()  # .to_dict(orient="index")
    return star_to_row


def get_star_mapping(high_df, keep_index=True):
    """Transform multiples table to be indexed by stars, picking out maximal multiple for each one.

    :param high_df: Multiples data from starforge simulation
    :type high_df: Pandas dataframe
    :return: "Exploded" dataframe indexed by star id. Each id will have one row that corresponds to "maximal" multiples containing that id
    :rtype: Pandas dataframe
    """
    df = high_df.copy()
    df["mult_len"] = df["mult_ids_list"].apply(len)

    # Step 1: Explode to have one row per star
    df_exploded = df.explode("mult_ids_list")
    df_exploded["mult_ids_list_og"] = high_df["mult_ids_list"].copy()

    # Step 2: Sort so the longest lists come first
    df_exploded = df_exploded.sort_values("mult_len", ascending=False)

    # Step 3: Create a mapping: star_id → row with longest mult_ids_list containing it
    star_to_row = df_exploded.drop_duplicates(
        subset="mult_ids_list", keep="first"
    ).set_index("mult_ids_list")
    ## Step 4: Reset to the original indexing with one multiple per row if desired.
    if keep_index:
        star_to_row.set_index(star_to_row["mult_ids_list_og"].astype(str), inplace=True)
        star_to_row = star_to_row.loc[~star_to_row.index.duplicated(keep="first")]

    return star_to_row


def get_first_snap_table(path_lookup):
    """_summary_

    :param path_lookup: dictionary of numpy arrays containing particles pos, vel, mass, etc. over times.
    :type path_lookup: dict
    :return: Pandas dataframe with columns id, snap, initial pos, final mass, initial mass, mass at 1 Myr, initial halo mass
    :rtype: Pandas dataframe
    """
    path_lookup_keys = path_lookup.keys()
    first_snap_table = []
    for kk in path_lookup_keys:
        star_filt_path = path_lookup[kk][~np.isinf(path_lookup[kk][:, 0])]
        mf = star_filt_path[-1][mcol]
        mMyr = star_filt_path[min(40, len(star_filt_path) - 1), mcol]

        mstar = star_filt_path[0][mcol]
        mhalo = star_filt_path[0][mtotcol]
        first_snap_table.append(
            (
                int(kk),
                star_filt_path[0][0],
                star_filt_path[0][pxcol],
                star_filt_path[0][pycol],
                star_filt_path[0][pzcol],
                mf,
                mstar,
                mMyr,
                mhalo,
            )
        )
    first_snap_table = pd.DataFrame(
        first_snap_table,
        columns=("pid", "snap", "x", "y", "z", "mf", "mstar", "mMyr", "mhalo"),
    )
    return first_snap_table


##IS SOMEWHAT REDUNDANT WITH GET_STAR_MAPPING_CLOSEST?
def get_star_map_bins(high_df_filt):
    """_summary_

    :param high_df_filt: Dataframe with multiples
    :type high_df_filt: Pandas dataframe
    :return: Pandas dataframe of binaries indexed by stars. Contains a column with binary halo labels (this is the main use of this function).
    """
    snaps = high_df_filt.index.get_level_values(level="t").unique()

    star_map_closest_all = []
    for snap in snaps:
        star_map_closest = get_star_mapping_closest(high_df_filt.xs(snap, level="t"))
        star_map_closest = star_map_closest.loc[star_map_closest["mult"] == 2]
        bin_halo_label = star_map_closest["mult_ids_list_og"].apply(lambda x: x[0])
        star_map_closest["bin_halo_label"] = bin_halo_label
        star_map_closest_all.append(star_map_closest)
    star_map_closest_all = pd.concat(star_map_closest_all, keys=snaps)
    return star_map_closest_all


def read_bh_swallow(base_swallow):
    """_summary_

    :param base_swallow: path to bh_swallow file
    :type base_swallow: str
    :return: bh_swallow as pandas dataframe with ids and times sorted
    :rtype: _type_
    """
    if os.path.isfile(base_swallow + "/bhswallow.pq"):
        bh_swallow_df = pd.read_parquet(base_swallow + "/bhswallow.pq")
    else:
        bh_swallow = []
        tmp_swallow = ascii.read(base_swallow + "/bhswallow.dat")
        bh_swallow.append(tmp_swallow)
        bh_swallow = astro_vstack(bh_swallow)
        bh_swallow_df = bh_swallow.to_pandas()
        bh_swallow_df.rename(
            columns={"col1": "time", "col2": "id", "col3": "sink_mass", "col7": "hid"},
            inplace=True,
        )
        bh_swallow_df.sort_values(by=["id", "time"], inplace=True)
        bh_swallow_df.to_parquet(base_swallow + "/bhswallow.pq")

    return bh_swallow_df
