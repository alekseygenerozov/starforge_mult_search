import pickle
import sys

import h5py
import hydra
import matplotlib.pyplot as plt
import numpy as np
import tqdm

# sys.path.append("/home/aleksey/code/python/star_forge_analysis/")
import starforge_mult_search.code.starforge_constants as sfc

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

delta = (-0.38, 0.22, -0.068, -0.42, 0.65)
a = (5.95, 6, 18, 10.26, 7.71, 98.87)
b = (9.25, 9.89, 10.24, 11.13, 14.28)


def sigmoid(x):
    return 0.5 * (1.0 + x / (1.0 + x**2.0) ** 0.5)


def ad_index(u):
    u_cgs = u * 100**2.0
    gamma = 5.0 / 3.0
    for kk in range(5):
        gamma += delta[kk] * sigmoid(a[kk] * (np.log10(u_cgs) - b[kk]))

    return gamma


def u_to_cs(u1):
    gamma_eff = ad_index(u1)
    # print("gamma:",gamma_eff)
    return u1**0.5 * (gamma_eff * (gamma_eff - 1)) ** 0.5


def get_shape_eigen(dxc):
    """dxc - distance from center (density max) position"""
    ## Return length of principle axes
    evals, evecs = np.linalg.eig(np.cov(dxc.T))  # This seems very slow ...
    ord1 = np.argsort(evals)

    return evals[ord1] ** 0.5, evecs[ord1]


def jeans(rho, cs):
    return cs / (sfc.GN * rho) ** 0.5


# @hydra.main(config_path=".", config_name="halo_size_config", version_base=None)
def main():
    rhalos = []
    rjeans = []
    final_masses = []
    a1s = []
    a2s = []
    a3s = []
    ##Missing seed 1
    clean_keys = []
    npts = []

    for seed in (1, 2, 42):
        my_ft = 8.0
        sim_tag = f"M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{seed}"
        base = f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{seed}/"
        r1 = "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/{0}/M2e4_snapshot_".format(
            sim_tag
        )
        r2 = f"_TidesFalse_smaoFalse_mult4_ngrid1_hmTrue_ft{my_ft}_coFalse.p".replace(
            ".p", ""
        )
        aa = "analyze_multiples_output_" + r2 + "/"
        base_sink = base + "/sinkprop/{0}_snapshot_".format(sim_tag)
        base2 = base.replace(
            "/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/", ""
        )
        bin_ids = np.load(base2 + aa + "/unique_bin_ids_mult.npz", allow_pickle=True)[
            "arr_0"
        ]
        dat_coll = np.load(base2 + aa + f"/dat_coll_mult.npz")

        fst = np.load(base2 + aa + "/fst_mult.npz", allow_pickle=True)["arr_0"]
        with open(base2 + aa + "/path_lookup.p", "rb") as ff:
            path_lookup = pickle.load(ff)
        for ii, pid in enumerate(tqdm.tqdm(path_lookup.keys())):
            path1 = path_lookup[pid]
            path1 = path1[~np.isinf(path1[:, 0])]
            ##Should use time of maximum halo...for consistency
            ss = int(path1[0, 0])
            # final_masses.append()
            fmass = path1[-1, mcol]

            with h5py.File(
                base
                + f"/halo_masses/halo_masses_sing_npTrue_c0.5_{ss}_compFalse_tf{my_ft}.hdf5",
                "r",
            ) as ff:
                sink_pos = path_lookup[str(pid)][ss, 2:5]

                kk0 = f"halo_{pid}"
                kk = f"halo_{pid}_x"
                kk_rho = f"halo_{pid}_rho"
                kk_u = f"halo_{pid}_u"
                kk_mass = f"halo_{pid}_m"

                tmp_bound = ff[kk0][...]
                tmp_pos = ff[kk][...]
                tmp_rho = ff[kk_rho][...]
                tmp_u = ff[kk_u][...]
                tmp_mass = ff[kk_mass][...]
                assert len(tmp_bound) == len(tmp_pos)

                if len(tmp_pos.shape) != 2:
                    breakpoint()
                    print("bad1", tmp_bound)
                elif np.sum(tmp_pos) == 0:
                    if len(tmp_bound) > 0:
                        breakpoint()
                    print("bad2", tmp_bound)
                elif len(tmp_pos) < 1:
                    print("bad3", tmp_bound)
                else:
                    # dx = tmp_pos
                    # dx = tmp_pos - np.median(tmp_pos, axis=0)
                    rho_mean = np.mean(tmp_rho)
                    cs_mean = u_to_cs(np.mean(tmp_u))
                    dx = tmp_pos - sink_pos
                    rs = np.sum(dx * dx, axis=1) ** 0.5
                    # print("cs:", cs_mean * 100 / 1e5)
                    npts.append(len(tmp_pos))

                    evals, evecs = get_shape_eigen(dx)
                    a1, a2, a3 = evals
                    a1s.append(a1)
                    a2s.append(a2)
                    a3s.append(a3)
                    rhalo = (a1 * a2 * a3) ** (1.0 / 3.0)
                    rhalos.append(rhalo)
                    ##Volume-weighted average of the density...
                    # rho_mean2 = np.sum(tmp_mass[rs <= rhalo]) / (4. * np.pi / 3. * rhalo**3.)
                    # if np.isinf(jeans(rho_mean2, cs_mean)):
                    #     breakpoint()
                    rjeans.append(jeans(rho_mean, cs_mean))
                    final_masses.append(fmass)
                    clean_keys.append(pid)

                    # fig, axs = plt.subplots(figsize=(16, 8), ncols = 2, constrained_layout=True)
                    # ax = axs[0]
                    # ax.set_xlabel("x [pc]")
                    # ax.set_ylabel("y [pc]")
                    # ax.set_title(f"{a1:.3g} {a2:.3g} {a3:.3g} {rhalos[-1]:.3g}")
                    # ax.scatter(dx[:,0], dx[:,1])
                    # ax.plot([0, a1 * evecs[0,0]], [0, a1 * evecs[0,1]], "r--", linewidth=4)
                    # ax.plot([0, a2 * evecs[1,0]], [0, a2 * evecs[1,1]], "r--", linewidth=4)
                    # ax.plot([0, a3 * evecs[2,0]], [0, a3 * evecs[2,1]], "r--", linewidth=4)
                    #
                    # ax = axs[1]
                    # ax.set_xlabel("x [pc]")
                    # ax.set_ylabel("z [pc]")
                    # ax.scatter(dx[:,0], dx[:,2], c='k')
                    # ax.plot([0, a1 * evecs[0,0]], [0, a1 * evecs[0,2]], "r--", linewidth=4)
                    # ax.plot([0, a2 * evecs[1,0]], [0, a2 * evecs[1,2]], "r--", linewidth=4)
                    # ax.plot([0, a3 * evecs[2,0]], [0, a3 * evecs[2,2]], "r--", linewidth=4)
                    #
                    # fig.savefig(f"tmp_{seed}_{my_ft}_{ii:03d}.png")
                    # plt.close()

        np.savez(
            f"halo_sizes_{my_ft}.npz",
            rhalos=rhalos,
            rjeans=rjeans,
            final_masses=final_masses,
            a1s=a1s,
            a2s=a2s,
            a3s=a3s,
            clean_keys=clean_keys,
            npts=npts,
        )


if __name__ == "__main__":
    main()
