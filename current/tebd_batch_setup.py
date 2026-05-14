import os
import math
import numpy as np
import yaml

# Sweep configuration
Ls       = [24, 32, 48]
maxdims  = [200, 400, 800]
bcs      = ["OBC", "PBC"]
omegas   = [0.1, 1.0, 10.0, 100.0]
gs       = np.round(np.linspace(0.0, 2.0, 21), 4)
parities = {"even": 1, "odd": -1}

# Fixed parameters
t1            = 1.0
periods       = 20
tau_target    = 0.05
min_sp        = 10   # Floor at 10 steps/period when high-frequency
max_sp        = 200  # Cap on low-frequency end; tau grows above tau_target there
corr_every    = 1

# DMRG ground-state parameters (used inside TEBD)
cutoff             = 1e-9
nsweeps            = 20
dmrg_tol           = 1e-8
eigsolve_krylovdim = 15
noise              = [0.0]

# Cluster parameters
request_cpus  = 8
julia_threads = 8
exec_path     = "/cluster/research-groups/rahmani/Liam/majorana_chain_dynamics/current/tebd_run.sh"
batch_root    = "/cluster/research-groups/rahmani/Liam/majorana_chain_dynamics/current"
parent_folder = "tebd_batches_2026-05"

cwd = os.path.join(os.getcwd(), parent_folder)
os.makedirs(cwd, exist_ok=True)


# Helpers
def steps_per_period(omega):
    # Using N = int(2*pi/(omega*tau)) from the whiteboard with min_sp and max_sp
    n = int(2 * math.pi / (omega * tau_target))
    return max(min_sp, min(n, max_sp))

def ac_sites_for(L, bc):
    if bc == "OBC":
        return [1, L // 4, L // 2]
    else:  # PBC
        return [1, L // 2]

def mem_gb(L, maxdim):
    # RAM scales as L * maxdim^2.
    return max(4, math.ceil(2.5e-7 * L * maxdim**2))

# Main loop: handles one batch folder per (L, maxdim, bc)
for L in Ls:
    for maxdim in maxdims:
        for bc in bcs:
            batch_folder = f"L{L}_{bc}_m{maxdim}"
            batch_path   = os.path.join(cwd, batch_folder)
            os.makedirs(batch_path, exist_ok=True)

            ac_sites = ac_sites_for(L, bc)

            njobs = 0
            for omega in omegas:
                sp = steps_per_period(omega)
                for g in gs:
                    for site in ac_sites:
                        for p_str, p_int in parities.items():
                            dirname = os.path.join(
                                batch_path,
                                f"om{omega}_g{g}_site{site}_p{p_str}"
                            )
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
                            njobs += 1

            # Write the matching HTCondor .batch file
            batch_file = os.path.join(cwd, f"{batch_folder}.batch")
            mem = mem_gb(L, maxdim)
            with open(batch_file, "w") as f:
                f.write(
                    f"Universe   = vanilla\n"
                    f"Executable = {exec_path}\n"
                    f"Output     = out.$(Process)\n"
                    f"Error      = err.$(Process)\n"
                    f"Log        = condor.log\n"
                    f"Request_Cpus   = {request_cpus}\n"
                    f"Request_Memory = {mem}GB\n"
                    f"Priority = 5\n"
                    f"Initialdir = $(dirname)\n"
                    f"Queue dirname matching dirs {batch_root}/{parent_folder}/{batch_folder}/*\n"
                )

            print(f"{batch_folder}: {njobs} jobs, {mem} GB requested")

print("Done.")
