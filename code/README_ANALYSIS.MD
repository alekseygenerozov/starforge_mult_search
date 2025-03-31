#### `analyze_multiples.py`
Processes the multiple search output files to extract all binaries, and properties across time.

Output files:
- `lookup_dict.p`: Dictionary mapping particle IDs to arrays describing their multiple system properties. The columns are:
  - `LOOKUP_SNAP`: Snapshot number.
  - `LOOKUP_PID`: Particle ID.
  - `LOOKUP_SID`: Parent system index.
  - `LOOKUP_MULT`: Particle multiplicity.
  - `LOOKUP_MTOT`: Mass including gas halo.
  - `LOOKUP_M`: Mass excluding gas halo.
  - `LOOKUP_SMA`: Semi-major axis.
  - `LOOKUP_ECC`: Eccentricity.
  - `LOOKUP_Q`: Mass ratio.
  - `LOOKUP_MPRIM`: Primary mass.
  - `LOOKUP_MPRIM_ID`: Primary star index.
  - `LOOKUP_STAR_ORDER`: Rank in multiple (1 = Primary).
  - Note: If single, `LOOKUP_SMA`, `LOOKUP_ECC`, and `LOOKUP_Q` are -1.
- `unique_bin_ids_mult.npz`: Unique binary IDs throughout the simulation, including those in higher-order multiples.
- `fst_mult.npz`: Initial snapshot data for all binary pairs.
- `path_lookup.p`: Pickle file containing particle positions and masses. The columns are:
  - `t`: Snapshot number
  - `id`: Particle ID. 
  - `px, py, pz`: Position components. 
  - `vx, vy, vz`: Velocity components.
  - `h`: Softening length. 
  - `m`: Mass of particle (without halo).
  - `sys_id`: System index (in systems list)
  - `mtot`: Mass of particle + halo
  - `sma`: Semi-major axis [pc].
  - `ecc`: Eccentricity.
#### `analyze_multiples_part2.py`
Refines and extends the analysis with additional properties.

Output files:
- `dat_coll_mult.npz`: Contains various binary properties:
  - `bin_ids`: Binary pairs.
  - `fst`: Initial snapshot together (IST) for binaries.
  - `ens`: -PE/KE at IST (without gas).
  - `ens_gas`: -PE/KE at IST (with gas).
  - `same_sys_at_fst`: Whether pair is in same multiple at IST.
  - `bin_at_fst`: Whether pair is a binary at IST.
  - `vangs`: Angle between velocity vectors.
  - `vangs_prim`: Angle between separation and secondary velocity.
  - `mfinal_primary`: Max primary mass at last snapshot both exist.
  - `mfinal_pair`: Total pair mass at last snapshot both exist.
  - `quasi_filter`: Mask for selecting persistent binaries.
  - `init_bound_snaps`: First snapshot pair is binary.
  - `final_bound_snaps`: Last snapshot pair is binary.
  - `bound_time`: Number of snapshots pair remains bound.
  - `delta_snap`: Time difference between star formation.
  - `same_sys_final_norm`: Whether pair remains in same system at the end. CAUTION: Do not
            use this filter to check survival, as it does not require the multiple to be perisistent
  - `final_pair_mass_lbin`: Total pair mass at last binary snapshot.
  - `final_pair_mass_lsys`: Total pair mass at last system snapshot.
  - `snap_interval`: Time interval between snapshots.
  - Note: Mass fields *exclude* gas halos.

#### `high_multiples_analysis.py`
Processes higher-order multiples.

Output files:
- `mults_flat.pq`: Pandas DataFrame with multiple systems and their longevity. Stores all high order multiples and sub-systems
  - Columns: `id`, `t`, `tf`, `a`, `e`, `p`, `ss`, `frac_of_orbit`, `nbound_snaps`.
  - `id`: Particle IDs in multiple (Index). 
  - `t`: Snapshot number (Index).
  - `tf`: Final snapshot number
  - `a`: Semi-major axis [pc]
  - `e`: Eccentricity
  - `p`: Period [yr]
  - `ss`: Whether system is a sub-system of another.
  - `frac_of_orbit`: Fraction of orbit systems remains bound (see eq. 1). 
  - `nbound_snaps`: Number of snapshots system remains bound.
  
- `fates_corr.npz`:
  - `end_states`: Final multiplicity of binary pairs.
  - `same_sys_filt`: Whether pairs remain in persistent multiples. THIS is what is used to select survivors

#### `analyze_stack.py`
Aggregates data across multiple seeds, and contains auxillary functions for post-processing data.

