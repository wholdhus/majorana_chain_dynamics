import os
import numpy as np
import yaml

cwd = os.getcwd()

# Parameters should match the TEBD run we excess energy for
L              = 32
bc             = "OBC"
t1             = 1.0
omega          = 5.0
g              = 0.5
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

# Compute the midpoint t1 values used in the TEBD gate construction
T   = 2 * np.pi / omega
tau = T / steps_per_period
midpoint_times = [tau * (n - 0.5) for n in range(1, steps_per_period + 1)]
t1_values = np.round([t1 * np.cos(omega * t) for t in midpoint_times], 10)

folder = f"L{L}_{bc}_m{maxdim}_t1sweep_g{g}"
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