import argparse
import glob
import multiprocessing
import os
import subprocess
import sys

from run_batch_aux import bash_command, get_cadence

from starforge_mult_search.code import find_multiples_new2


def bash_command(cmd, **kwargs):
    """Run command from the bash shell"""
    process = subprocess.Popen(["/bin/bash", "-c", cmd], **kwargs)
    return process.communicate()[0]


def run_sink(ii, snap_base, star_age_key):
    cmd = f"python3 starforge_mult_search/code/sink_data.py {ii}  --snap_base {snap_base} --star_age_key {star_age_key}"
    bash_command(cmd)


def main():
    parser = argparse.ArgumentParser(
        description="Parse starforge snapshot, and get multiple data."
    )
    parser.add_argument("start", help="start snapshot")
    parser.add_argument("end", help="end snapshot")
    parser.add_argument(
        "--star_age_key", default="ProtoStellarAge", help="Key for stellar age"
    )
    args = parser.parse_args()
    start = int(args.start)
    end = int(args.end)
    star_age_key = args.star_age_key

    with open("data_loc", "r") as ff:
        snap_base = ff.read().strip()
    snaps = glob.glob(snap_base + "*hdf5")
    cadence = get_cadence(snaps)
    print(f"{cadence = }")

    out1 = find_multiples_new2.load_data(
        snap_base + "_000.hdf5", res_limit=1e-3, star_age_key=star_age_key
    )
    out2 = find_multiples_new2.load_data(
        snap_base + f"_{cadence:03d}.hdf5", res_limit=1e-3, star_age_key=star_age_key
    )
    if not os.path.exists("sinkprop"):
        bash_command(f"mkdir sinkprop")
    snap_interval = (out2["tage_myr"] - out1["tage_myr"]) * 1e6
    print(f"{snap_interval = }")
    with open("sinkprop/snap_interval") as ff:
        ff.write(snap_interval)

    if end < 0:
        end = (len(snaps) - 1) * cadence
    with multiprocessing.Pool(10) as pool:
        pool.starmap(
            run_sink,
            [(ii, snap_base, star_age_key) for ii in range(start, end + 1, cadence)],
        )


if __name__ == "__main__":
    main()
