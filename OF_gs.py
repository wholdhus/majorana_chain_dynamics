import os
import numpy as np
import yaml

cwd = os.getcwd()

L = 24
maxdim = 200
folder = f"L{L}_m{maxdim}_gs"
os.mkdir(cwd+"/"+folder)

steps = 41
gs = np.round(np.linspace(-0.5, 3.5, steps), 4)
parities = {"even": 1,
            "odd": -1}
nstates = 5
bc = "OBC"
kdim = 15
nsweeps = 40
cutoff = 1E-8
energy_tol = 1E-8
noise = [1e-9, 1e-9, 1e-6, 1e-6, 1e-5, 1e-5,
         1e-6, 1e-7, 1e-8, 1e-9, 0.0]

for g in gs:
    for p in parities:
        dirname = cwd + "/" + folder + f"/L{L}_g{g}_p_{p}"
        os.mkdir(dirname)
        data = {"L": L,
                "bc": bc,
                "parity": parities[p],
                "t": float(1.0),
                "g": float(g),
                "nstates": nstates,
                "fname": dirname+"/output.h5",
                "dmrg_params": {"nsweeps": nsweeps,
                                "m": maxdim,
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
