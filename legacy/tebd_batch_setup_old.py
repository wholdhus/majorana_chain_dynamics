import os
import numpy as np
import yaml

cwd = os.getcwd()

# Parameters
L      = 32
bc     = "OBC"
t1     = 1.0
omega  = 5.0
periods = 40

# Set as a single value or a list to sweep
g       = 0.5
# g     = np.round(np.linspace(-0.5, 1.5, 201), 4)
ac_site = 16
# ac_site = list(range(1, L+1))

parities = {"even": 1, "odd": -1}

# TEBD/DMRG Parameters
maxdim             = 400
steps_per_period   = 20
corr_every         = 1
cutoff             = 1e-9
nsweeps            = 20
dmrg_tol           = 1e-8
eigsolve_krylovdim = 15
noise              = [0.0]

# Batch name
batch_folder = "L32_OBC_m400_t11.0_g0.5_om5.0"
os.mkdir(cwd + "/" + batch_folder)

for g_val in np.atleast_1d(g):
    for site_val in np.atleast_1d(ac_site):
        for p_str, p_int in parities.items():
            dirname = (cwd + "/" + batch_folder +
                       f"/g{g_val}_site{site_val}_p{p_str}")
            os.mkdir(dirname)
            fname = dirname + "/output.h5"
            data = {
                "L":       L,
                "bc":      bc,
                "parity":  p_int,
                "t1":      float(t1),
                "g":       float(g_val),
                "omega":   float(omega),
                "periods": int(periods),
                "ac_site": int(site_val),
                "fname":   fname,
                "tebd_params": {
                    "maxdim":             maxdim,
                    "steps_per_period":   steps_per_period,
                    "corr_every":         corr_every,
                    "cutoff":             cutoff,
                    "nsweeps":            nsweeps,
                    "dmrg_tol":           dmrg_tol,
                    "eigsolve_krylovdim": eigsolve_krylovdim,
                    "noise":              noise,
                }
            }
            with open(dirname + "/input.yml", "w") as f:
                yaml.dump(data, f, sort_keys=False)
            print(f"Written: {dirname}/input.yml")
