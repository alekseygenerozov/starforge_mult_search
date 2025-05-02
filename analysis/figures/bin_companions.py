from starforge_mult_search.analysis.figures.figure_preamble import *
import tqdm

def removeNestings_wrap(l):
    output = []
    removeNestings(l, output)
    return output

##Table of higher multiples
high_df = pd.concat([pd.read_parquet(base_new + str(seed) + suff_new + f"/mults{flat_suff}.pq") for seed in seeds])
tval = high_df.index.get_level_values("t")
high_df["tval"] = tval
high_df = high_df.loc[(high_df[f"frac_of_orbit{contig_suff}"] >= 1) & (high_df[f"nbound_snaps{contig_suff}"] > 1)]
bin_ids = my_data["bin_ids"]
quasi_filter = my_data[f"quasi_filter{contig_suff}"]

comps_a_ids = [[]  for _ in range(len(bin_ids))]
comps_a_times = [[]  for _ in range(len(bin_ids))]
comps_a_ids_flat = [[]  for _ in range(len(bin_ids))]

comps_b_ids = [[]  for _ in range(len(bin_ids))]
comps_b_times = [[]  for _ in range(len(bin_ids))]
comps_b_ids_flat = [[]  for _ in range(len(bin_ids))]
for ii, row in tqdm.tqdm(enumerate(bin_ids)):
    bin_list = list(row)
    if not quasi_filter[ii]:
        continue

    mult_ids_list = high_df["mult_ids_list"]
    tmp_sel2a = high_df.loc[high_df.index.get_level_values("id").str.contains(rf"\b{bin_list[0]}\b")]
    tmp_sel2b = high_df.loc[high_df.index.get_level_values("id").str.contains(rf"\b{bin_list[1]}\b")]

    if len(tmp_sel2a) > 0:
        comps_a = tmp_sel2a.groupby("t")[["tval", "mult", "mult_ids_list"]].apply(lambda g: g[g["mult"] == g["mult"].max()])
        comps_a_ids[ii] = (comps_a.index.get_level_values("id").to_numpy())
        comps_a_times[ii] = (comps_a["tval"].values)
        comps_a_ids_flat[ii] = (comps_a["mult_ids_list"].values)

    if len(tmp_sel2b) > 0:
        comps_b = tmp_sel2b.groupby("t")[["tval", "mult", "mult_ids_list"]].apply(lambda g: g[g["mult"] == g["mult"].max()])
        comps_b_ids[ii] = (comps_b.index.get_level_values("id").to_numpy())
        comps_b_times[ii] = (comps_b["tval"].values)
        comps_b_ids_flat[ii] = (comps_b["mult_ids_list"].values)

##Fix the name here!!!
comp = dict(comps_a_ids=comps_a_ids, comps_a_times=comps_a_times, comps_a_ids_flat=comps_a_ids_flat,
         comp_b_ids=comps_b_ids, comps_b_times=comps_b_times, comps_b_ids_flat=comps_b_ids_flat)
##TODO: Tag this with analysis tags contig_suff and flat_suff
with open("companions.p", "wb") as ff:
    pickle.dump(comp, ff)