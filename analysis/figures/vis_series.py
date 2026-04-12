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
    os.chdir(f"example_{ii}")
    tracers = np.genfromtxt("tracers")
    if len(tracers) < 30:
        os.chdir("../")
        return
    times = np.genfromtxt("times").astype(int)
    # comps = np.genfromtxt("companion_lists").astype(int)

    # with open("config_template", "r") as ff:
    #     config_template = ff.read()

    with open("config_template_b", "r") as ff:
        config_template_b = ff.read()

    # for ss in range(times[0], times[1]):
    #     config = config_template.replace("SS", str(ss))
    #     with open("config_0", "w") as fout:
    #         fout.write(config)
    #     bash_command(
    #         "python3 ../starforge_mult_search/analysis/figures/fig1_extend.py 0"
    #     )

    for ss in range(times[1], times[2] + 1):
        config = config_template_b.replace("SS", str(ss))
        # config = config.replace("COMP", str(comps[ss]))
        with open("config_0", "w") as fout:
            fout.write(config)
        bash_command(
            "python3 ../starforge_mult_search/analysis/figures/fig1_extend.py 0"
        )
    os.chdir("../")


def main():
    # process_example(sys.argv[1])
    ##Parallel image productions(!)
    examples_file = sys.argv[1]
    examples = np.genfromtxt(examples_file).astype(int)

    with multiprocessing.Pool(5) as pool:
        pool.map(process_example, examples)


if __name__ == "__main__":
    main()
