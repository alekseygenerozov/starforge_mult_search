import multiprocessing
import os
import subprocess
import sys

import numpy as np

sys.path.append("../")


def bash_command(cmd, **kwargs):
    """Run command from the bash shell"""
    process = subprocess.Popen(["/bin/bash", "-c", cmd], **kwargs)
    return process.communicate()[0]


def process_example(ii):
    target_dir = f"example_{ii}"
    bash_script_cmd = (
        "python3 ../starforge_mult_search/analysis/figures/fig1_extend.py 0"
    )

    tracers = np.genfromtxt(os.path.join(target_dir, "tracers"))
    if len(tracers) < 30:
        return
    times = np.genfromtxt(os.path.join(target_dir, "times")).astype(int)

    with open(os.path.join(target_dir, "config_template"), "r") as ff:
        config_template = ff.read()

    with open(os.path.join(target_dir, "config_template_b"), "r") as ff:
        config_template_b = ff.read()

    for ss in range(times[0], times[1]):
        config = config_template.replace("SS", str(ss))
        with open(os.path.join(target_dir, "config_0"), "w") as fout:
            fout.write(config)
        bash_command(f"cd {target_dir} && {bash_script_cmd}")

    for ss in range(times[1], times[2] + 1):
        config = config_template_b.replace("SS", str(ss))
        with open(os.path.join(target_dir, "config_0"), "w") as fout:
            fout.write(config)
        bash_command(f"cd {target_dir} && {bash_script_cmd}")


def main():
    # process_example(sys.argv[1])
    ##Parallel image productions(!)
    with multiprocessing.Pool(10) as pool:
        pool.map(process_example, range(int(sys.argv[1])))


if __name__ == "__main__":
    main()
