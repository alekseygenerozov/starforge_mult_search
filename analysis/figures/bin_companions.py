from starforge_mult_search.analysis.figures.figure_preamble import *
import tqdm

def removeNestings_wrap(l):
    output = []
    removeNestings(l, output)
    return output

##Table of higher multiples
high_df = pd.concat([pd.read_parquet(base_new + str(seed) + suff_new + f"/mults.pq") for seed in seeds])
tval = high_df.index.get_level_values("t")
high_df["tval"] = tval
high_df = high_df.loc[(high_df["frac_of_orbit_seg"] >= 1) & (high_df["nbound_snaps_seg"] > 1)]
bin_ids = my_data["bin_ids"]

comps_a_ids = []
comps_a_times = []
comps_a_ids_flat = []

comps_b_ids = []
comps_b_times = []
comps_b_ids_flat = []
for row in tqdm.tqdm(bin_ids):
    bin_list = list(row)

    mult_ids_list = high_df["mult_ids_list"]
    tmp_sel2a = high_df.loc[high_df.index.get_level_values("id").str.contains(rf"\b{bin_list[0]}\b")]
    tmp_sel2b = high_df.loc[high_df.index.get_level_values("id").str.contains(rf"\b{bin_list[1]}\b")]

    comps_a_ids = []
    comps_a_times = []
    comps_a_ids_flat = []

    comps_b_ids = []
    comps_b_times = []
    comps_b_ids_flat = []

    if len(tmp_sel2a) > 0:
        comps_a = tmp_sel2a.groupby("t")[["tval", "mult", "mult_ids_list"]].apply(lambda g: g[g["mult"] == g["mult"].max()])
        comps_a_ids.append(comps_a.index.get_level_values("id").to_numpy())
        comps_a_times.append(comps_a["tval"].values)
        comps_a_ids_flat.append(comps_a["mult_ids_list"].values)
    else:
        comps_a_ids.append([])
        comps_a_times.append([])
        comps_a_ids_flat.append([])
    if len(tmp_sel2b) > 0:
        comps_b = tmp_sel2b.groupby("t")[["tval", "mult", "mult_ids_list"]].apply(lambda g: g[g["mult"] == g["mult"].max()])
        comps_b_ids.append(comps_b.index.get_level_values("id").to_numpy())
        comps_b_times.append(comps_b["tval"].values)
        comps_b_ids_flat.append(comps_b["mult_ids_list"].values)
    else:
        comps_b_ids.append([])
        comps_b_times.append([])
        comps_b_ids_flat.append([])


np.savez("bin_persistent_comps.npz", comps_a_ids=comps_a_ids, comps_a_times=comps_a_times, comps_a_ids_flat=comps_a_ids_flat,
         comp_b_ids=comps_b_ids, comps_b_times=comps_b_times, comps_b_ids_flat=comps_b_ids_flat)