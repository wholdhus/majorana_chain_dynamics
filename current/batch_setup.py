import os
import numpy as np
import yaml

cwd = os.getcwd()

L          = 32
maxdim     = 200
bc         = "OBC"
kdim       = 15
nsweeps    = 20
cutoff     = 1e-9
energy_tol = 1e-8
noise      = [0.0]
nstates    = 1

steps = 101
gs = np.round(np.linspace(0.5, 1.5, steps), 4)

parities = {"even": 1,
            "odd": -1}

folder = f"L{L}_{bc}_m{maxdim}"
os.mkdir(cwd + "/" + folder)

for g in gs:
    for p in parities:
        dirname = cwd + "/" + folder + f"/g{g}_p{p}"
        os.mkdir(dirname)
        fname = dirname + "/output.h5"
        data = {"L": L,
                "bc": bc,
                "parity": parities[p],
                "t": float(1.0),
                "g": float(g),
                "nstates": nstates,
                "fname": fname,
                "dmrg_params": {"m": maxdim,
                                "nsweeps": nsweeps,
                                "cutoff": cutoff,
                                "dmrg_tol": energy_tol,
                                "noise": noise,
                                "eigsolve_krylovdim": kdim,
                                }
                }
        with open(dirname + "/input.yml", 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print("This data: ")
        print(data)
        print("Dumped data to file : {}".format(dirname + "/input.yml"))
        print()