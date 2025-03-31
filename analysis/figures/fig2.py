import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
import seaborn as sns
colorblind_palette = sns.color_palette("colorblind")

from starforge_mult_search.analysis.analyze_stack import npz_stack
from labelLine import labelLines

##Stacking data
my_tides = False
suff = "_mult"
base_new =  "M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_"
seeds = (1,2,42)

# Compute ECDF
def ecdf(data):
    x = np.sort(data)
    y = np.arange(1, len(data) + 1) / len(data)
    return x, y

my_ft = 1.0
suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
npzs_list = []
npzs_list = [base_new+str(seed)+suff_new+f"/dat_coll{suff}.npz" for seed in seeds]
base_data = npz_stack(npzs_list)

my_ft = 8.0
suff_new = f"/analyze_multiples_output__Tides{my_tides}_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse"
npzs_list = []
npzs_list = [base_new+str(seed)+suff_new+f"/dat_coll{suff}.npz" for seed in seeds]
comp = npz_stack(npzs_list)

# Load data
quasi_filter = base_data["quasi_filter"]
ens = base_data["ens"]
ens_gas = base_data["ens_gas"]

comp_quasi_filter = comp["quasi_filter"]
comp_ens = comp["ens"]
comp_ens_gas = comp["ens_gas"]

fig,ax = plt.subplots(constrained_layout=True)
ax.set_xlim(-2.1, 2)
ax.set_ylabel("Fraction (Cumulative)")
ax.set_xlabel(r"log(-PE / KE)")

log_ens = np.log10(ens[quasi_filter])
log_ens_b = np.log10(ens_gas[quasi_filter])
log_ens_comp = np.log10(comp_ens[comp_quasi_filter])
log_ens_gas = np.log10(ens_gas[quasi_filter])
log_ens_gas_comp = np.log10(comp_ens_gas[comp_quasi_filter])

ax.ecdf(log_ens)
ax.ecdf(log_ens_b, color="r", linestyle="--")
ax.ecdf(log_ens_comp, color=colorblind_palette[0])
ax.ecdf(log_ens_gas, label="Gas", color=colorblind_palette[1])
ax.ecdf(log_ens_gas_comp, label="Gas", color=colorblind_palette[1])

x_gas1, y_gas1 = ecdf(log_ens)
x_gas2, y_gas2 = ecdf(log_ens_comp)

print(interp1d(x_gas1, y_gas1)(0))
print(interp1d(x_gas2, y_gas2)(0))

x_common = np.linspace(min(min(x_gas1), min(x_gas2)), max(max(x_gas1), max(x_gas2)), 500)
y_gas1_interp = np.interp(x_common, x_gas1, y_gas1)
y_gas2_interp = np.interp(x_common, x_gas2, y_gas2)

ax.fill_between(x_common, y_gas1_interp,  y_gas2_interp, color=colorblind_palette[0], alpha=0.3)


x_gas1, y_gas1 = ecdf(log_ens_gas)
x_gas2, y_gas2 = ecdf(log_ens_gas_comp)

x_common = np.linspace(min(min(x_gas1), min(x_gas2)), max(max(x_gas1), max(x_gas2)), 500)
y_gas1_interp = np.interp(x_common, x_gas1, y_gas1)
y_gas2_interp = np.interp(x_common, x_gas2, y_gas2)

print(interp1d(x_gas1, y_gas1)(0))
print(interp1d(x_gas2, y_gas2)(0))

# Plot ECDF
# ax.plot(x_gas, y_gas, label="Gas", color='blue')
ax.fill_between(x_common, y_gas1_interp,  y_gas2_interp, color=colorblind_palette[1], alpha=0.3)
ax.plot([0, 0], [0,1], '0.5', linestyle=':')
ax.annotate("No gas", (-0.2, 0.6), ha='right', color=colorblind_palette[0]) 
ax.annotate("Gas", (0.95, 0.75), ha='left', color=colorblind_palette[1])
ax.annotate("Unbound", (-0.05,0.96), ha='right', color='0.5')
ax.annotate("Bound", (0.03,0.96), ha='left', color='0.5')

fig.savefig("energy.pdf")


##Be wary of flips here....
ens_ck_seed1 = np.load("en_col_seed1_1.0.npz")["arr_0"]
ens_ck_seed2 = np.load("en_col_seed2_1.0.npz")["arr_0"]
ens_ck_seed42 = np.load("en_col_seed42_1.0.npz")["arr_0"]
ens_ck = np.concatenate((ens_ck_seed1, ens_ck_seed2, ens_ck_seed42))

log_ens_ratio_ft1 = np.log10(-ens_ck[:, 0] / ens_ck[:, 1])[quasi_filter]
ax.ecdf(log_ens_ratio_ft1, color=colorblind_palette[2])

ens_ck_seed1 = np.load("en_col_seed1_8.0.npz")["arr_0"]
ens_ck_seed2 = np.load("en_col_seed2_8.0.npz")["arr_0"]
ens_ck_seed42 = np.load("en_col_seed42_8.0.npz")["arr_0"]
ens_ck = np.concatenate((ens_ck_seed1, ens_ck_seed2, ens_ck_seed42))
# print(len(ens_ck))

log_ens_ratio_ft8 = np.log10(-ens_ck[:, 0] / ens_ck[:, 1])[comp_quasi_filter]
ax.ecdf(log_ens_ratio_ft8, color=colorblind_palette[2])

x_gas1, y_gas1 = ecdf(log_ens_ratio_ft8)
print(interp1d(x_gas1, y_gas1)(0))
x_gas2, y_gas2 = ecdf(log_ens_ratio_ft1)
print(interp1d(x_gas2, y_gas2)(0))

x_common = np.linspace(min(min(x_gas1), min(x_gas2)), max(max(x_gas1), max(x_gas2)), 500)
y_gas1_interp = np.interp(x_common, x_gas1, y_gas1)
y_gas2_interp = np.interp(x_common, x_gas2, y_gas2)

# # Plot ECDF
ax.fill_between(x_common, y_gas1_interp,  y_gas2_interp, color=colorblind_palette[2], alpha=0.3)
ax.annotate("Gas corrected", (0.55, 0.5), ha='left', color=colorblind_palette[2]) 
plt.show()
fig.savefig("energy_space.pdf")
