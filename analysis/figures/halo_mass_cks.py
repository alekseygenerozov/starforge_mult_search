from starforge_mult_search.analysis.figures.figure_preamble import *
import h5py
from astropy.io import ascii

print(my_ft)

mfs = np.array([[path_lookup[kk][-1, 1], path_lookup[kk][-1, mcol]] for kk in path_lookup.keys()])
mfs = mfs[(mfs[:,1] >= 10) & ~np.isinf(mfs[:,1])]

pid = str(int(mfs[-1][0]))
path1 = path_lookup[pid]
path1 = path1[~np.isinf(path1[:, 0])]

snaps = path1[:,0]

seed = 42
base = f"/home/aleksey/Dropbox/projects/Hagai_projects/star_forge/M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_{seed}/"
halo_lookup = {}
halo_snap_lookup = {}
halo_snap_mask = {}
bh_swallow = ascii.read(base + "/bhswallow.dat")
bh_swallow = bh_swallow[bh_swallow["col2"]==int(pid)]
accreted = bh_swallow["col7"]

for ss in snaps:
    with h5py.File(base + f"/halo_masses/halo_masses_sing_npTrue_c0.5_{int(ss)}_compFalse_tf{my_ft}.hdf5", "r") as ff:
        kk0 = f'halo_{pid}'
        kk = f'halo_{pid}_x'
        tmp_bound = ff[kk0][...]
        tmp_pos = ff[kk][...]
        if np.sum(tmp_bound) == 0:
            continue
        halo_snap_lookup[ss] = tmp_bound
        halo_snap_mask[ss] = [(xx in accreted) for xx in tmp_bound]


        for ii, row in enumerate(tmp_bound):
            try:
                if (int(row) in halo_lookup):
                    halo_lookup[int(row)].append(ss)
                else:
                    halo_lookup[int(row)] = [ss]
            except TypeError:
                breakpoint()

