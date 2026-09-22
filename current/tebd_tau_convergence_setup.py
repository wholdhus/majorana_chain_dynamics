import os
import yaml

configs  = [(24, 400, 1e-9, 4)]
bc       = "OBC"
omega    = 0.1
gs       = [0.2, 1.0]
parities = {"even": 1, "odd": -1}
sps      = [1280, 640, 320, 160] # steps per period

t1         = 1.0

nsweeps            = 20
dmrg_tol           = 1e-8
eigsolve_krylovdim = 15
noise              = [0.0]

request_cpus  = 8
request_mem   = 8
exec_path     = "/cluster/research-groups/rahmani/Liam/majorana_chain_dynamics/current/tebd_run.sh"
parent_folder = "tebd_tau_convergence_2026-09"

root = os.path.abspath(parent_folder)

for L, maxdim, cutoff, periods in configs:
    batch_folder = f"L{L}_{bc}_m{maxdim}_om{omega}_tau_convergence"
    batch_path   = os.path.join(root, batch_folder)
    site = L // 2
    dirs = []
    for sp in sps:
        for g in gs:
            for p_str, p_int in parities.items():
                dirname = os.path.join(batch_path, f"om{omega}_g{g}_site{site}_p{p_str}_sp{sp}")
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
                        "corr_every":         sp // 16,
                        "cutoff":             cutoff,
                        "nsweeps":            nsweeps,
                        "dmrg_tol":           dmrg_tol,
                        "eigsolve_krylovdim": eigsolve_krylovdim,
                        "noise":              noise,
                    }
                }
                with open(os.path.join(dirname, "input.yml"), "w") as f:
                    yaml.dump(data, f, sort_keys=False)
                dirs.append(dirname)

    with open(os.path.join(root, f"{batch_folder}.batch"), "w") as f:
        f.write(
            f"Universe   = vanilla\n"
            f"Executable = {exec_path}\n"
            f"Output     = out.$(Process)\n"
            f"Error      = err.$(Process)\n"
            f"Log        = condor.log\n"
            f"Request_Cpus   = {request_cpus}\n"
            f"Request_Memory = {request_mem}GB\n"
            f"Initialdir = $(dirname)\n"
            f"Queue dirname from (\n" + "\n".join(dirs) + "\n)\n"
        )
    print(f"{batch_folder}: {len(dirs)} jobs, {sum(sp * periods for sp in sps) * len(gs) * len(parities)} total Trotter steps")

print("Done.")
