

# starforge_mult_search

Idenfies all the binaries in a starforge simulation snapshot. Here are the necessary
steps:

(0) Make a file called data_loc in your current working directory with the location
of the starforge snapshots to be analyzed.

(1) Calculate the halo masses using "python3 run_batch_halo_only_par.py START END FT,"
where START is the starting snapshot, END is the ending snapshot, and FT is the desired
ft. To analyze all the snapshots, use python3 run_batch_halo_only_par.py 0 -1 FT.

(2) This will run the script halo_masses_single_double_par.py and produce two set output files: (i) Plain text files M2e4halo_masses*, with
tabular data, containing the list of halo masses in the first column and 
(ii) halo_masses*hdf5, which stores information on which gas particle's are in
each star particle's halo.

(3) Move these output files to a separate directory called "halo_masses."

(4) Use python3 run_batch.py START END FT to identify multiples in the simulation.
START is the starting snapshot, END is the ending snapshot, and FT is the desired ft.
To analyze all the snapshots, use python3 starforge_mult_search/code/run_batch.py 0 -1 "1.0 --ntides".
This will run the script find_multiples_new2.py, and output pickle files for all the snapshots containing information about the multiples
in the simulation. The data is stored in a "cluster" object that contains a list of "systems."
Each system contains several fields encoding its properties, e.g.
(i) multiplicity (ii) ids [ids of stars in the system] (iii) sub_mass [masses of
stars in system] (iv) sub_pos [positions of stars in the system.] (v) orbital data, etc. The code below shows an example of using the 
of reading a pickle file, and getting a numpy array of the system multiplicities.
Note that the masses stores in the output data, include the halo mass contribution.

```
with open(f"example.p", "rb") as ff:
    clb = pickle.load(ff)
    mb = np.array([ss.multiplicity for ss in clb.systems])
```

(5) User can adjust options in find_multiples_new2.py to tweak the multiplication id 
algorithm. Note that we have specified the --ntides flag in our analysis to turn off
the tidal criterion in the multiple identification.

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

NumPy, SciPy, pytreegrav (https://github.com/mikegrudic/pytreegrav), h5py, numba, astropy, hydra, pandas, seaborn
Versions used in paper: NumPy (1.24.4), SciPy (1.6.1), pytreegrav (commit:b38de42e, similar to version 1.1), h5py (3.2.1), numba (0.57.1),
astropy (7.0.0), hydra (1.3.2), pandas (2.1.4), seaborn (0.13.2), meshoid (1.46.0) [TO DO: Generate environment that is like
TACC and run analysis with this?! To avoid any inconsistencies. TO: Add github for meshoid.]
Also tested demo with: NumPy (1.26.0), SciPy (1.13.1), pytreegrav (1.1.4), h5py (3.12.1), numba (0.60.0)


# Installation and Demos
We provide a code capsule to run a demo of the code directly on Code Ocean, so that local installation is not required. Environment setup on
Code Ocean takes approximately 20 seconds. 

The code capsule include a demo dataset of a single snapshot from a simulation of a
smaller, 2x10^3 solar mass cloud.

The total runtime for the demo is 3 minutes on a 10 core Lenovo E15 laptop and 4 minutes on Code Ocean

The demo will produce the output files described in #starforge_mult_search above, as well as a file called 'mult_summary' with
the multiple statistics in the demo data. The contents should be as below:

Multiplicity 1 count: 111\
Multiplicity 2 count: 17\
Multiplicity 3 count: 2\
Multiplicity 4 count: 6

# Reproducing paper results and figures
Running the full analysis pipeline for the actual simulations data takes several days of compute on an HPC.
Moreover, it would be impractical to upload the required simulation snapshots (of order 1 TB for a particular cloud).

However, the paper figures can be reproduced in reasonable time (within ~20 minutes) if the output of the 
multiple search algorithm has been prepared in advance. We have upload this output, so 
that this can be done. The scripts for this part of the analysis are stored in the 
"analysis" subdirectory. In summary this directory contains the 

1. Scripts that post-process these files (e.g. analyze_multiples.py, analyze_multiples_part2.py,
high_multiples_analysis.py) and reogranize the data into tables for easier analysis. For completeness,
we describe these scripts' outputs in README_ANALYSIS.md, but the user does not need to interact 
with these intermediate outputs to reproduce the paper figures.
2. Scripts that produce Figures in the analysis/figures, using the above data tables. 
So fig2.py generates figure 2 (fig2.pdf), fig3.py generates figure 3 (fig3.pdf), etc. 
fig2.py also generates ex_fig5.py, and ex_fig1.py generates the right panel of fig2. (As fig2b.pdf)
fig4.py generates the numbers in the schematic in fig. 4
3. Tables.py generate the numbers in the Tables.

The script "run_pipeline_all" runs the analysis steps above, and generates pdf files 
for the figures in the paper. The intermediate data tables (described in README_ANALYSIS.MD)
are stored in directories with names like the following:

M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_1/analyze_multiples_output__TidesFalse_smaoFalse_mult4_ngrid1_hmTrue_ft1.0_coFalse.

The first part of name (M2e4_R10/M2e4_R10_S0_T1_B0.1_Res271_n2_sol0.5_1) indicates the cloud parameters of the underlying 
starforge simulation. The suffix at the end (TidesFalse_smaoFalse_mult4_ngrid1_hmTrue_ft1.0_coFalse) indicates the parameters used for the halo multiple identification code 