import os
import glob
import numpy as np
import yaml

# Point this at the TEBD batch folder (relative or absolute)
batch_dir = "tebd_batches_2026-05/L24_OBC_m200"

# DMRG params for the lookup table
kdim       = 15
nsweeps    = 20
cutoff     = 1e-9
energy_tol = 1e-8
noise      = [0.0]
nstates    = 1
weight     = 1000  # unused when nstates=1, kept for consistency with dmrg_run.jl

parities = {"even": 1, "odd": -1}

# Discover all input.yml files in the batch, collect (L, bc, maxdim, t1, g, omega, parity) tuples
configs = set()
for ymlpath in glob.glob(os.path.join(batch_dir, "**", "input.yml"), recursive=True):
    with open(ymlpath) as f:
        p = yaml.safe_load(f)
    L      = p["L"]
    bc     = p["bc"]
    t1     = float(p["t1"])
    g      = float(p["g"])
    omega  = float(p["omega"])
    parity = int(p["parity"])
    maxdim = int(p["tebd_params"]["maxdim"])
    steps_per_period = int(p["tebd_params"]["steps_per_period"])
    configs.add((L, bc, maxdim, t1, g, omega, steps_per_period, parity))

if not configs:
    raise RuntimeError(f"No input.yml files found under {batch_dir}")

# Sanity check: L, bc, maxdim should be unique across the batch
Ls      = {c[0] for c in configs}
bcs     = {c[1] for c in configs}
maxdims = {c[2] for c in configs}
assert len(Ls) == 1 and len(bcs) == 1 and len(maxdims) == 1, \
    f"Batch has mixed L/bc/maxdim: L={Ls}, bc={bcs}, maxdim={maxdims}"
L, bc, maxdim = Ls.pop(), bcs.pop(), maxdims.pop()

# Build the set of (t1_endpoint, g, parity) triples we need
needed = set()
for _, _, _, t1, g, omega, steps_per_period, parity in configs:
    tau = (2 * np.pi / omega) / steps_per_period
    for n in range(steps_per_period + 1):
        t1_eff = round(t1 * np.cos(omega * tau * n), 10)
        needed.add((t1_eff, g, parity))

print(f"Batch: {batch_dir}")
print(f"  L={L}, bc={bc}, maxdim={maxdim}")
print(f"  {len(configs)} TEBD configs -> {len(needed)} (t1, g, parity) jobs")

# Write jobs
batch_name = os.path.basename(os.path.normpath(batch_dir))
folder = f"{batch_name}_gs_table"
os.mkdir(folder)

parity_label = {1: "even", -1: "odd"}
for t1_val, g_val, p_val in sorted(needed):
    dirname = f"{folder}/t1{t1_val}_g{g_val}_p{parity_label[p_val]}"
    os.mkdir(dirname)
    data = {
        "L": L,
        "bc": bc,
        "parity": p_val,
        "t": float(t1_val),
        "g": float(g_val),
        "nstates": nstates,
        "fname": f"{dirname}/output.h5",
        "dmrg_params": {
            "m": maxdim,
            "nsweeps": nsweeps,
            "cutoff": cutoff,
            "dmrg_tol": energy_tol,
            "noise": noise,
            "eigsolve_krylovdim": kdim,
            "weight": weight,
        },
    }
    with open(f"{dirname}/input.yml", "w") as f:
        yaml.dump(data, f, sort_keys=False)

print(f"Wrote {len(needed)} jobs to {folder}/")
