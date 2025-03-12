import numpy as np
import pandas as pd

pd.set_option("display.precision", 2)
# Data for the DataFrame


my_ft = 1.0
my_tides = False
base_new = "../new_analysis/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_"
suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
npzs_list = []
seeds = (1, 2, 42)
suff = "_mult"

c0s = []
c1s = []
f1s = []
c2s = []
f2s = []
c3s = []

fates_corr = np.load(suff_new.replace("/", "") + "_fates_corr.npz")
final_sys_filter_all = fates_corr["same_sys_filt"]
seeds_lookup = np.concatenate(
    [[seed] * len(np.load(base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz", allow_pickle=True)["bin_ids"])
     for
     seed in seeds])

for seed in seeds:
    # npzs_list = [base_new+str(seed)+suff_new+f"/dat_coll{suff}.npz" for seed in seeds]
    # my_data = npz_stack(npzs_list)
    my_data = np.load(base_new + str(seed) + suff_new + f"/dat_coll{suff}.npz", allow_pickle=True)
    # final_sys_filter = (my_data["same_sys_final_norm"] == 1)
    final_sys_filter = final_sys_filter_all[seeds_lookup==seed]

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
    c3s.append(len(bin_ids[quasi_filter & ss_fst & final_sys_filter]))
    f2s.append(c2s[-1] / tot2)

    ##Incorporate final multiple filter into the core data tables for easy filtering...
    data = [
        {"Condition": "Total", "Count": tot1, "Fraction": 1.0},
        # {"Condition": "Bound at IST (no gas)", "Count": len(bin_ids[(quasi_filter) & (ss_fst)]),
        #  "Fraction": len(bin_ids[(quasi_filter) & (ss_fst)]) / tot1},
        {"Condition": "Bound at IST", "Count": c1s[-1],
         "Fraction": f1s[-1]},
        {"Condition": "Survivors", "Count": tot2,
         "Fraction": 1.0},
        {"Condition": "Bound at IST", "Count": c2s[-1],
         "Fraction": f2s[-1]}
    ]

    # Create the DataFrame
    df = pd.DataFrame(data)

    # Display the DataFrame
    df.set_index("Condition", inplace=True)
    print(df)

print(np.mean(c0s), np.mean(c1s), np.mean(c2s), np.mean(c3s))
print(np.mean(f1s), np.mean(f2s))