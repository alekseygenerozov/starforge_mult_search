

# starforge_mult_search

Idenfies all of the binaries in a starforge simulation snapshot. Here are the necessary
steps:

(0) Make a file called data_loc in your current working directory with the location
of the starforge snapshots to be analyzed.

(1) Calculate the halo masses using "python3 run_batch_halo_only_par.py START END FT,"
where START is the starting snapshot, END is the ending snapshot, and FT is the desired
ft. To analyze all of the snapshots, use python3 run_batch_halo_only_par.py 0 -1 FT.

(2) This will run the script halo_masses_single_double_par.py and produce two set output files: (i) Plain text files M2e4halo_masses*, with
tabular data, containing the list of halo masses in the first column and 
(ii) halo_masses*hdf5, which stores information on which gas particle's are in
each star particle's halo.

(3) Move these output files to a separate directory called "halo_masses."

(4) Use python3 run_batch.py START END FT to identify multiples in the simulation.
START is the starting snapshot, END is the ending snapshot, and FT is the desired ft.
To analyze all of the snapshots, use python3 run_batch.py 0 -1 FT.
This will run the script find_multiples_new2.py, and output pickle files for all of the snapshots containing information about the multiples
in the simulation. The data is stored in a "cluster" object that contains a list of "systems."
Each system contains several fields encoding its properties, e.g.
(i) multiplicity (ii) ids [ids of stars in the system] (iii) sub_mass [masses of
stars in system] (iv) sub_pos [positions of stars in the system.] (v) orbital data, etc. The code below shows an example of using the 
of reading a pickle file, and getting a numpy array of the system multiplicities.


```
with open(f"example.p", "rb") as ff:
    clb = pickle.load(ff)
    mb = np.array([ss.multiplicity for ss in clb.systems])
```

(5) User can adjust options in find_multiples_new2.py to tweak the multiplication id 
algorithm 

```
usage: find_multiples_new2.py [-h] [--snap_base SNAP_BASE]
                              [--name_tag NAME_TAG] [--sma_order]
                              [--halo_mass_file HALO_MASS_FILE]
                              [--mult_max MULT_MAX] [--ngrid NGRID]
                              [--compress] [--tides_factor TIDES_FACTOR]
                              [--nhalo] [--ntides]
                              snap

Parse starforge snapshot, and get multiple data.

positional arguments:
  snap                  Snapshot index

options:
  -h, --help            show this help message and exit
  --snap_base SNAP_BASE
                        First part of snapshot name
  --name_tag NAME_TAG   Extension for saving.
  --sma_order           Assemble hierarchy by sma instead of binding energy
  --halo_mass_file HALO_MASS_FILE
                        Start of the file containing gas halo mass around sink
                        particles
  --mult_max MULT_MAX   Multiplicity cut (4).
  --ngrid NGRID         Number of subgrids to use. Higher number will be
                        faster, but less accurate (1)
  --compress            Filter out compressive tidal forces
  --tides_factor TIDES_FACTOR
                        Prefactor for check of tidal criterion (8.0)
  --nhalo               Turn off halo
  --ntides              Turn off tides


  ```
# Requirements

NumPy, SciPy, pytreegrav (https://github.com/mikegrudic/pytreegrav), h5py, numba
Versions used in paper: NumPy (1.24.4), SciPy (1.6.1), pytreegrav (commit:b38de42e, similar to version 1.1), h5py (3.2.1), numba (0.57.1)
Also tested demo with: NumPy (1.26.0), SciPy (1.13.1), pytreegrav (1.1.4), h5py (3.12.1), numba (0.60.0)


# Installation and Demo
We provide a code capsule to run a demo of the code directly on Code Ocean, so that local installation is not required. Environment setup on
Code Ocean takes approximately 20 seconds. 

The code capsule include a demo dataset of a single snapshot from a simulation of a
smaller, 2x10^3 solar mass cloud.

The total runtime for for the demo is 3 minutes on 10 core Lenovo E15 laptop and 4 minutes on Code Ocean

The demo will produce the output files described in #starforge_mult_search above, as well as a file called 'mult_summary' with
the multiple statistics in the demo data. The contents should be as below:

Multiplicity 1 count: 111\
Multiplicity 2 count: 17\
Multiplicity 3 count: 2\
Multiplicity 4 count: 6






