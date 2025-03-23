import pickle

import numpy as np
import pandas as pd

from starforge_mult_search.analysis.high_multiples_analysis import lookup_star_mult

from starforge_mult_search.analysis.high_multiples_analysis import filter_top_level

pd.set_option("display.precision", 2)
# Data for the DataFrame


my_ft = 1.0
my_tides = False

base_new_all = ["../analysis_pipeline_experiment/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_",
        "../analysis_pipeline_experiment/M2e4_R10/M2e4_R10_S0_T1_B0.01_Res271_n2_sol0.5_",
        "../analysis_pipeline_experiment/M2e4_R10/M2e4_R10_S0_T1_B1_Res271_n2_sol0.5_",
        "../analysis_pipeline_experiment/M2e4_R10/M2e4_R10_S0_T0.5_B0.01_Res271_n2_sol0.5_",
        "../analysis_pipeline_experiment/M2e4_R10/M2e4_R10_S0_T2_B0.01_Res271_n2_sol0.5_",
        "../analysis_pipeline_experiment/v1.2/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_"
                ]
seeds_all =[ (1, 2, 42), (1,), (1,), (1,), (1,), (42,)]
grand_total_stars = 0
grand_total_bins_a = 0
grand_total_bins_b = 0
grand_total_mults = 0
grand_total_mults_b = 0
for ii in range(len(base_new_all)):
    base_new = base_new_all[ii]
    suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
    npzs_list = []
    seeds = seeds_all[ii]
    suff = "_mult"

    c0s = []
    c1s = []
    f1s = []
    c2s = []
    c3s = []
    f3s = []

    for seed in seeds:
        print(base_new, seed)
        # npzs_list = [base_new+str(seed)+suff_new+f"/dat_coll{suff}.npz" for seed in seeds]
        # my_data = npz_stack(npzs_list)
        my_data = np.load(base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz", allow_pickle=True)
        # final_sys_filter = (my_data["same_sys_final_norm"] == 1)
        final_sys_filter =  np.load(base_new + str(seed) + suff_new + f"/fates_corr.npz", allow_pickle=True)["same_sys_filt"]
        with open(base_new + str(seed) + suff_new + "/path_lookup.p", "rb") as ff:
            tmp_path_pickle = pickle.load(ff)
            tmp_nstar = len(tmp_path_pickle.keys())
        grand_total_stars += tmp_nstar

        bin_ids = my_data["bin_ids"]
        quasi_filter = my_data["quasi_filter"]
        # quasi_filter = get_quasi_filter_modb(my_data, 10)
        ss_fst = my_data["same_sys_at_fst"].astype(bool)
        # final_bin_filter = (my_data["fates"] == "s2")
        # final_bin_filter_wm = (my_data["final_bound_snaps_norm"] == 1.)
        tot1 = len(bin_ids[quasi_filter])
        tot2 = len(bin_ids[quasi_filter & final_sys_filter])
        # tot3 = len(ens_gas[quasi_filter & final_bin_filter_wm])

        c0s.append(tot1)
        c1s.append(len(bin_ids[(quasi_filter) & (ss_fst)]))
        print(c1s[-1])
        f1s.append(c1s[-1] / tot1)
        c2s.append(len(bin_ids[quasi_filter & final_sys_filter]))
        # f2s.append(c2s[-1] / tot2)
        c3s.append(len(bin_ids[quasi_filter & ss_fst & final_sys_filter]))
        f3s.append(c3s[-1] / tot2)

        ##Incorporate final multiple filter into the core data tables for easy filtering...
        data = [
            {"Condition": "Total", "Count": tot1, "Fraction": 1.0},
            # {"Condition": "Bound at IST (no gas)", "Count": len(bin_ids[(quasi_filter) & (ss_fst)]),
            #  "Fraction": len(bin_ids[(quasi_filter) & (ss_fst)]) / tot1},
            {"Condition": "Bound at IST", "Count": c1s[-1],
             "Fraction": f1s[-1]},
            {"Condition": "Survivors", "Count": tot2,
             "Fraction": 1.0},
            {"Condition": "Bound at IST", "Count": c3s[-1],
             "Fraction": f3s[-1]}
        ]

        # Create the DataFrame
        df = pd.DataFrame(data)

        # Display the DataFrame
        df.set_index("Condition", inplace=True)
        print(df)
        grand_total_bins_a += tot1
        grand_total_bins_b += tot2

        coll_full_df_life = pd.read_parquet(base_new + str(seed) + suff_new + f"/mults_flat.pq")
        coll_full_df_life = coll_full_df_life.loc[(coll_full_df_life["frac_of_orbit"] >= 1) & (coll_full_df_life["nbound_snaps"] > 1)]
        high_df_final = coll_full_df_life[(coll_full_df_life["tf"]==coll_full_df_life.index.get_level_values("t"))]
        high_df_final = filter_top_level(high_df_final)
        nmults = len(high_df_final)
        grand_total_mults += nmults
        #
        high_df_final = coll_full_df_life[(coll_full_df_life["tf"] == coll_full_df_life.index.get_level_values("t"))]
        final_sys_filter = np.load(base_new + str(seed) + suff_new + f"/fates_corr.npz")["same_sys_filt"]
        tmp_bin_list = my_data["bin_ids"][my_data["quasi_filter"] & final_sys_filter]
        tmp_mult_coll = []
        tmp_ids = high_df_final.index.get_level_values(level="id")
        for row in tmp_bin_list:
            tmp_row_list = list(row)
            tmp_id1 = lookup_star_mult(high_df_final, tmp_row_list[0], -1, pre_filtered=True)#[0]
            tmp_id2 = lookup_star_mult(high_df_final, tmp_row_list[1], -1, pre_filtered=True)#[0]
            if tmp_id1[0]!=tmp_id2[0]:
                continue
            tmp_mult_coll.append(tmp_id1[0])
        nmults = len(np.unique(tmp_mult_coll))
        grand_total_mults_b += nmults

    print(np.mean(c0s), np.mean(c1s), np.mean(c2s), np.mean(c3s))
    print(1, np.mean(f1s), 1, np.mean(f3s))

print(grand_total_stars, grand_total_bins_a, grand_total_bins_b, grand_total_mults, grand_total_mults_b)