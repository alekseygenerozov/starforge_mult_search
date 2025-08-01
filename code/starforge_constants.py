import os

# Default GN value (e.g., in physical units)
GN = 4.301e3  
# Check for override file
override_file = "GN_override.txt"  # or any other name you like
if os.path.exists(override_file):
    try:
        with open(override_file, "r") as f:
            GN = float(f.read().strip())
            print(f"[sfc] GN overridden to {GN} from {override_file}")
    except Exception as e:
        print(f"[sfc] Failed to read GN override file: {e}")

