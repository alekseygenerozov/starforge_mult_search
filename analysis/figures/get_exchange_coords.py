import pickle

import numpy as np
import pandas as pd

from starforge_mult_search.analysis.figures.figure_preamble import (
    my_data,
    path_lookup,
    lookup_dict,
    contig_suff,
    flat_suff,
    my_ft,
)
from starforge_mult_search.code.find_multiples_new2 import get_orbit


def pair_dist(g):
    p0 = g.iloc[0]
    p1 = g.iloc[1]

    return np.sqrt(
        (p0["x"] - p1["x"]) ** 2.0
        + (p0["y"] - p1["y"]) ** 2.0
        + (p0["z"] - p1["z"]) ** 2.0
    )


def check_soft(state, lookup_dict, path_lookup, time):
    is_soft = False
    for ii in range(len(state)):
        for jj in range(ii + 1, len(state)):
            soft_pair = max(state[ii, -1], state[jj, -1])
            if np.linalg.norm(state[ii, 2:5] - state[jj, 2:5]) < soft_pair:
                return True
            my_look1 = lookup_dict[state[ii, 1]]
            my_look2 = lookup_dict[state[jj, 1]]
            my_sma1 = my_look1[my_look1[:, 0] == time][0, LOOKUP_SMA]
            my_ecc1 = my_look1[my_look1[:, 0] == time][0, LOOKUP_ECC]
            my_sma2 = my_look2[my_look2[:, 0] == time][0, LOOKUP_SMA]
            my_ecc2 = my_look2[my_look2[:, 0] == time][0, LOOKUP_ECC]
            if (
                (my_sma1 > 0)
                and (my_sma1 == my_sma2)
                and (my_sma1 * (1.0 - my_ecc1) < soft_pair)
            ):
                return True

    return False


LOOKUP_SNAP = 0
LOOKUP_PID = 1
LOOKUP_MULT = 3
LOOKUP_MTOT = 4
LOOKUP_SMA = 6
LOOKUP_ECC = 7
LOOKUP_Q = 8
##Columns to use
############################################################################################################
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
##############################################################################################################
##Have this come from the config file instead(!)
# my_ft = 1.0
# flat_suff = ""
# contig_suff = "_seg"

ex_time_max = np.load(f"pmult_before_bin_{my_ft}{flat_suff}{contig_suff}.npz")[
    "ex_time_max"
]
ex_time_max_end = np.load(f"pmult_before_bin_{my_ft}{flat_suff}{contig_suff}.npz")[
    "ex_time_max_end"
]
ex_filt = ~np.isinf(ex_time_max)
ex_filt = ex_filt & my_data[f"quasi_filter{contig_suff}"]
ex_index = np.where(ex_filt)[0]
# paths = np.array([path_lookup[str(pp)] for pp in path_lookup.keys()])

with open(f"companions{flat_suff}{contig_suff}.p", "rb") as ff:
    comps_dict = pickle.load(ff)

init_state_all = []
end_state_all = []
orbs_all = []
init_table_columns = [
    "t",
    "id",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "mcol",
    "mtotcol",
    "hcol",
    "nonew_comps",
    "is_soft",
]
end_table_columns = [
    "t",
    "id",
    "x",
    "y",
    "z",
    "vx",
    "vy",
    "vz",
    "mcol",
    "mtotcol",
    "hcol",
    "bdist",
    "sma",
    "ecc",
    "is_soft",
]
for bin_input in range(len(ex_index)):
    tmp_bin_idx = ex_index[bin_input]
    my_bin = my_data["bin_ids"][tmp_bin_idx]
    ps = list(my_bin)
    path1 = path_lookup[str(ps[0])]
    path2 = path_lookup[str(ps[1])]

    start_encounter_time, end_encounter_time = int(ex_time_max[tmp_bin_idx]), int(
        ex_time_max_end[tmp_bin_idx]
    )
    end_encounter_time = (
        start_encounter_time + (end_encounter_time - start_encounter_time) * 3
    )
    ##Edge case -- going beyond the boundary of the simulation(!!) - Is this adequate??
    if end_encounter_time >= len(path1):
        dummy = np.atleast_2d(np.ones(len(init_table_columns)) * np.inf)
        init_state_all.append(
            pd.DataFrame(dummy, columns=init_table_columns).reset_index(drop=True)
        )
        dummy = np.atleast_2d(np.ones(len(end_table_columns)) * np.inf)
        end_state_all.append(
            pd.DataFrame(dummy, columns=end_table_columns).reset_index(drop=True)
        )
        continue

    c1 = comps_dict["comps_a_ids_flat"][tmp_bin_idx]
    c2 = comps_dict["comps_b_ids_flat"][tmp_bin_idx]
    times1 = comps_dict["comps_a_times"][tmp_bin_idx]
    times2 = comps_dict["comps_b_times"][tmp_bin_idx]

    t_group = (times1, times2)
    comps_start = []
    comps_end = []
    for ii, cc in enumerate((c1, c2)):
        tmp_comps_start = np.array(cc, dtype=object)[
            np.array(t_group[ii]) == start_encounter_time
        ]
        tmp_comps_end = np.array(cc, dtype=object)[
            np.array(t_group[ii]) == end_encounter_time
        ]

        if len(tmp_comps_start) > 0:
            comps_start.append(tmp_comps_start[0])
        if len(tmp_comps_end) > 0:
            comps_end.append(tmp_comps_end[0])

    comps_start = np.concatenate(comps_start)
    comps_start = np.unique(
        comps_start[(comps_start != ps[0]) & (comps_start != ps[1])]
    )
    if len(comps_end) > 0:
        comps_end = np.concatenate(comps_end)
        comps_end = np.unique(comps_end[(comps_end != ps[0]) & (comps_end != ps[1])])

    start_stars = np.concatenate((ps, comps_start))
    end_stars = np.concatenate((ps, comps_end))

    init_state = [
        path_lookup[str(tmp)][
            start_encounter_time,
            [0, 1, pxcol, pycol, pzcol, vxcol, vycol, vzcol, mcol, mtotcol, hcol],
        ]
        for tmp in start_stars
    ]
    ##For simplicity save the start stars -- we will filter out cases where new stars appear at the end
    end_state = [
        path_lookup[str(tmp)][
            end_encounter_time,
            [0, 1, pxcol, pycol, pzcol, vxcol, vycol, vzcol, mcol, mtotcol, hcol],
        ]
        for tmp in start_stars
    ]
    my_look1 = lookup_dict[ps[0]]
    my_look2 = lookup_dict[ps[1]]
    # sma_end1 = my_look1[my_look1[:, 0] == end_encounter_time][0, LOOKUP_SMA]
    # ecc_end1 = my_look1[my_look1[:, 0] == end_encounter_time][0, LOOKUP_ECC]
    # sma_end2 = my_look2[my_look2[:, 0] == end_encounter_time][0, LOOKUP_SMA]
    # ecc_end2 = my_look2[my_look2[:, 0] == end_encounter_time][0, LOOKUP_ECC]
    # assert sma_end1==sma_end2
    # assert ecc_end1==ecc_end2
    end_orbit = get_orbit(
        path1[end_encounter_time, pxcol : pzcol + 1],
        path2[end_encounter_time, pxcol : pzcol + 1],
        path1[end_encounter_time, vxcol : vzcol + 1],
        path2[end_encounter_time, vxcol : vzcol + 1],
        path1[end_encounter_time, mtotcol],
        path2[end_encounter_time, mtotcol],
        h1=path1[end_encounter_time, hcol],
        h2=path2[end_encounter_time, hcol],
    )
    sma_end1, ecc_end1 = end_orbit[0], end_orbit[1]

    ##Augmenting info for initial state
    init_state = np.array(init_state)
    no_new_comps = np.all(np.isin(comps_end, comps_start))
    is_softened_start = check_soft(
        init_state, lookup_dict, path_lookup, start_encounter_time
    )
    init_extras = np.array(
        [[no_new_comps, is_softened_start] for ii in range(len(init_state))]
    )
    # no_new_comps = np.ones((len(init_state), 1)) * no_new_comps
    init_state = np.hstack((init_state, init_extras))

    ##Augmenting info for end state..
    end_state = np.array(end_state)
    is_softened_end = check_soft(
        end_state, lookup_dict, path_lookup, end_encounter_time
    )
    end_pair_dist = np.linalg.norm(end_state[0, 2:5] - end_state[1, 2:5])
    end_extras = np.array(
        [
            [end_pair_dist, sma_end1, ecc_end1, is_softened_end]
            for ii in range(len(end_state))
        ]
    )
    end_state = np.hstack((end_state, end_extras))

    init_state_all.append(
        pd.DataFrame(init_state, columns=init_table_columns).reset_index(drop=True)
    )
    end_state_all.append(
        pd.DataFrame(end_state, columns=end_table_columns).reset_index(drop=True)
    )

ncases = len(init_state_all)
ic = pd.concat(init_state_all, keys=range(ncases))
es = pd.concat(end_state_all, keys=range(ncases))
##Fixing indiices
ic.index = ic.index.get_level_values(0)
es.index = es.index.get_level_values(0)

##Could do this earlier(!) -- In the construction of the original DF
frac_halo = np.abs((ic["mcol"] - ic["mtotcol"]) / ic["mcol"])
ic["frac_halo"] = frac_halo
ic["frac_halo_max"] = ic.groupby(ic.index)["frac_halo"].max()


# ic.to_parquet("init_state_all.pq")
es.to_parquet("end_state_all.pq")
