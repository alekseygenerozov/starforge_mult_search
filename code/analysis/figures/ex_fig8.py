import numpy as np
import matplotlib.pyplot as plt

my_ft = 8.0
dat = np.load(f"halo_sizes_{my_ft}.npz")

fig,ax = plt.subplots()
ax.set_xlabel(r"$log(r_{halo} / r_{jeans})$")
ax.set_ylabel("N")
ax.hist(np.log10(dat["rhalos"] / dat["rjeans"]), histtype="step", bins=50, linewidth=4)
ax.legend(title=r"$f_t=$" + f"{int(my_ft)}")

print(np.median(dat["rhalos"] / dat["rjeans"]), np.median(dat["rhalos"]), np.median(dat["rjeans"]))
fig.savefig(f"halo_size_plot_a_{my_ft}.pdf")


fig,ax = plt.subplots()
ax.set_xlabel(r"Axis ratio")
ax.set_ylabel("N")
ax.hist(dat["a1s"] / dat["a3s"], histtype="step", bins=50, label="$a_1/a_3$", linewidth=4)
ax.hist(dat["a2s"] / dat["a3s"], histtype="step", bins=50, label="$a_2/a_3$", linewidth=4)
ax.legend(title=r"$f_t=$" + f"{int(my_ft)}")

fig.savefig(f"halo_size_plot_b_{my_ft}.pdf")
