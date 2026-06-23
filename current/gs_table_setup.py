import os
import numpy as np
import yaml

cwd = os.getcwd()

# Parameters should match the TEBD run we want excess energy for
L                = 4
bc               = "OBC"
t1               = 1.0
omega            = 5.0
g                = 0.2
steps_per_period = 20

# DMRG parameters
maxdim     = 400
kdim       = 15
nsweeps    = 20
cutoff     = 1e-9
energy_tol = 1e-8
noise      = [0.0]
nstates    = 2
weight     = 1000

parities = {"even": 1, "odd": -1}

# Compute the endpoint t1 values matching TEBD's energy_t measurement times
T              = 2 * np.pi / omega
tau            = T / steps_per_period
endpoint_times = [tau * n for n in range(steps_per_period + 1)]
t1_values      = np.unique(np.round([t1 * np.cos(omega * t) for t in endpoint_times], 10))

folder = f"L{L}_{bc}_gs_table_t1{t1}_g{g}_om{omega}"
os.mkdir(cwd + "/" + folder)

for t1_val in t1_values:
    for p in parities:
        dirname = cwd + "/" + folder + f"/t1{t1_val}_p{p}"
        os.mkdir(dirname)
        fname = dirname + "/output.h5"
        data = {"L": L,
                "bc": bc,
                "parity": parities[p],
                "t": float(t1_val),
                "g": float(g),
                "nstates": nstates,
                "fname": fname,
                "dmrg_params": {"m": maxdim,
                                "nsweeps": nsweeps,
                                "cutoff": cutoff,
                                "dmrg_tol": energy_tol,
                                "noise": noise,
                                "eigsolve_krylovdim": kdim,
                                "weight": weight,
                                }
                }
        with open(dirname + "/input.yml", 'w') as f:
            yaml.dump(data, f, sort_keys=False)
        print(f"Written: {dirname}/input.yml")