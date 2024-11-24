import pickle
import sys

import numpy as np

from find_multiples_new2 import cluster, system

with open(sys.argv[1], "rb") as ff:
    clb = pickle.load(ff)
    mb = np.array([ss.multiplicity for ss in clb.systems])

unique, counts = np.unique(mb, return_counts=True)
##Breakdown of muliplicities
with open("mult_summary", "w") as ff:
    for uu, cc in zip(unique, counts):
        ff.write(f"Multiplicity {uu} count: {cc}\n")