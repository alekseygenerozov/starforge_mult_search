python3 run_batch_halo_only_par.py 250 250 1
mkdir halo_masses
mv M2e4halo_masses* halo_masses*hdf5 halo_masses
python3 run_batch.py 250 250 1
python3 mult_summary.py M2e4_snapshot_250_TidesTrue_smaoFalse_mult4_ngrid1_hmTrue_ft1.0_coFalse.p
mv halo_masses/* /results
mv M2e4_snapshot*.p /results
mv mult_summary /results