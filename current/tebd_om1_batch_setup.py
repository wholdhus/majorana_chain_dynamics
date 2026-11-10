import os
import math
import shutil
import numpy as np
import yaml

L        = 24
bc       = "OBC"
omega    = 1.0
maxdims  = [200, 400, 600, 800]
gs       = np.round(np.concatenate([np.linspace(0.05, 0.30, 6), np.linspace(0.85, 1.10, 6)]), 4)
sites    = [1, L // 4, L // 2]
parities = {"even": 1, "odd": -1}

t1         = 1.0
periods    = 20
tau_target = 0.05
corr_every = 2

cutoff             = 1e-9
nsweeps            = 20
dmrg_tol           = 1e-8
eigsolve_krylovdim = 15
noise              = [0.0]

request_cpus  = 8
mem_gb        = {200: 4, 400: 4, 600: 8, 800: 8}
priority      = {200: (35, 15), 400: (35, 15), 600: (25, 10), 800: (25, 10)}
exec_path     = "/cluster/research-groups/rahmani/Liam/majorana_chain_dynamics/current/tebd_run.sh"
summer_folder = "tebd_batches_2026-05"
parent_folder = "tebd_om1_batches_2026-11"

sp = int(2 * math.pi / (omega * tau_target))
summer_gs = np.round(np.linspace(0.0, 2.0, 21), 4)
cwd = os.path.join(os.getcwd(), parent_folder)

for maxdim in maxdims:
    batch_folder = f"L{L}_{bc}_m{maxdim}"
    batch_path = os.path.join(cwd, batch_folder)
    os.makedirs(batch_path, exist_ok=True)
    first, rest, ncopied = [], [], 0
    for g in gs:
        for site in sites:
            for p_str, p_int in parities.items():
                job = f"om{omega}_g{g}_site{site}_p{p_str}"
                dirname = os.path.join(batch_path, job)
                if maxdim in (200, 400) and g in summer_gs:
                    if not os.path.exists(dirname):
                        shutil.copytree(os.path.join(os.getcwd(), summer_folder, batch_folder, job), dirname)
                    ncopied += 1
                    continue
                os.makedirs(dirname, exist_ok=True)
                data = {
                    "L":       L,
                    "bc":      bc,
                    "parity":  p_int,
                    "t1":      float(t1),
                    "g":       float(g),
                    "omega":   float(omega),
                    "periods": int(periods),
                    "ac_site": int(site),
                    "fname":   os.path.join(dirname, "output.h5"),
                    "tebd_params": {
                        "maxdim":             maxdim,
                        "steps_per_period":   sp,
                        "corr_every":         corr_every,
                        "cutoff":             cutoff,
                        "nsweeps":            nsweeps,
                        "dmrg_tol":           dmrg_tol,
                        "eigsolve_krylovdim": eigsolve_krylovdim,
                        "noise":              noise,
                    }
                }
                with open(os.path.join(dirname, "input.yml"), "w") as f:
                    yaml.dump(data, f, sort_keys=False)
                (first if site == 1 and p_str == "even" else rest).append(dirname)

    for name, jobs, prio in [("site1_peven", first, priority[maxdim][0]), ("rest", rest, priority[maxdim][1])]:
        with open(os.path.join(cwd, f"{batch_folder}_{name}.batch"), "w") as f:
            f.write(
                f"Universe   = vanilla\n"
                f"Executable = {exec_path}\n"
                f"Output     = out.$(Process)\n"
                f"Error      = err.$(Process)\n"
                f"Log        = condor.log\n"
                f"Request_Cpus   = {request_cpus}\n"
                f"Request_Memory = {mem_gb[maxdim]}GB\n"
                f"Priority = {prio}\n"
                f"Initialdir = $(dirname)\n"
                f"Queue dirname from (\n" + "\n".join(jobs) + "\n)\n"
            )

    print(f"{batch_folder}: {len(first)} site1_peven + {len(rest)} rest new jobs, {ncopied} copied, {mem_gb[maxdim]} GB")

print("Done.")
