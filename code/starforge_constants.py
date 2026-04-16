import os

# Default GN value (e.g., in physical units)
GN = 4.301e3
##Length and time scales
sfc_pc = 3.085678e+18
sfc_time_to_snap = sfc_pc / 100 / (3600.0 * 24.0 * 365.0) / 2.4703e4

# Check for override file
override_file = "GN_override.txt"  # or any other name you like
if os.path.exists(override_file):
    try:
        with open(override_file, "r") as f:
            GN = float(f.read().strip())
            print(f"[sfc] GN overridden to {GN} from {override_file}")
    except Exception as e:
        print(f"[sfc] Failed to read GN override file: {e}")
