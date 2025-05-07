import pickle

import numpy as np
import pandas as pd

from starforge_mult_search.analysis.figures.figure_preamble import my_data, path_lookup

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
my_ft = 1.0
flat_suff = ""
contig_suff = "_seg"

ex_time_max = np.load(f"pmult_before_bin_{my_ft}{flat_suff}{contig_suff}.npz")["ex_time_max"]
ex_time_max_end = np.load(f"pmult_before_bin_{my_ft}{flat_suff}{contig_suff}.npz")["ex_time_max_end"]
ex_filt = ~np.isinf(ex_time_max)
ex_filt = ex_filt & my_data[f"quasi_filter{contig_suff}"]
ex_index = np.where(ex_filt)[0]
# paths = np.array([path_lookup[str(pp)] for pp in path_lookup.keys()])

with open(f"companions{flat_suff}{contig_suff}.p", "rb") as ff:
    comps_dict = pickle.load(ff)

init_state_all = []
end_state_all = []
for bin_input in range(len(ex_index)):
    tmp_bin_idx = ex_index[bin_input]
    my_bin = my_data["bin_ids"][tmp_bin_idx]
    ps = list(my_bin)

    start_encounter_time, end_encounter_time = int(ex_time_max[tmp_bin_idx]), int(ex_time_max_end[tmp_bin_idx])
    c1 = comps_dict["comps_a_ids_flat"][tmp_bin_idx]
    c2 = comps_dict["comps_b_ids_flat"][tmp_bin_idx]
    times1 = comps_dict["comps_a_times"][tmp_bin_idx]
    times2 = comps_dict["comps_b_times"][tmp_bin_idx]

    t_group = (times1, times2)
    comps_start = []
    comps_end = []
    for ii, cc in enumerate((c1, c2)):
        tmp_comps_start = np.array(cc, dtype=object)[np.array(t_group[ii]) == start_encounter_time]
        tmp_comps_end = np.array(cc, dtype=object)[np.array(t_group[ii]) == end_encounter_time]

        if len(tmp_comps_start) > 0:
            comps_start.append(tmp_comps_start[0])
        if len(tmp_comps_end) > 0:
            comps_end.append(tmp_comps_end[0])

    comps_start = np.concatenate(comps_start)
    comps_start = np.unique(comps_start[(comps_start != ps[0]) & (comps_start != ps[1])])
    comps_end = np.concatenate(comps_end)
    comps_end = np.unique(comps_end[(comps_end != ps[0]) & (comps_end != ps[1])])

    start_stars = np.concatenate((ps, comps_start))
    end_stars = np.concatenate((ps, comps_end))

    init_state = [path_lookup[str(tmp)][start_encounter_time, [0, 1, pxcol, pycol, pzcol, vxcol, vycol, vzcol, mcol, mtotcol]] for tmp in start_stars]
    end_state = [path_lookup[str(tmp)][end_encounter_time, [0, 1, pxcol, pycol, pzcol, vxcol, vycol, vzcol, mcol, mtotcol]] for tmp in end_stars]

    init_state_all.append(pd.DataFrame(init_state, columns=["t", "id", "x", "y", "z", "vx", "vy", "vz", "mcol", "mtotcol"]))
    end_state_all.append(pd.DataFrame(end_state, columns=["t", "id", "x", "y", "z", "vx", "vy", "vz", "mcol", "mtotcol"]))

init_state_all = pd.concat(init_state_all)
end_state_all = pd.concat(end_state_all)
init_state_all.to_parquet("init_state_all.pq")
end_state_all.to_parquet("end_state_all.pq")











