import os
import glob
import math
import argparse
import numpy as np
import yaml

# Parent folder holding all the TEBD batches
PARENT = "tebd_batches_2026-05"

# Absolute cluster paths baked into the generated .batch files.
EXEC_PATH  = "/cluster/research-groups/rahmani/Jack/dmrg_run.sh"
BATCH_ROOT = "/cluster/research-groups/rahmani/Jack"

# DMRG params for the lookup table (ground state only)
KDIM       = 15
NSWEEPS    = 20
CUTOFF     = 1e-9
ENERGY_TOL = 1e-8
NOISE      = [0.0]
NSTATES    = 1
WEIGHT     = 1000

REQUEST_CPUS  = 2
PARITY_LABEL  = {1: "even", -1: "odd"}

TABLE_PRIORITY = 100


def mem_gb(L, maxdim):
    return max(4, math.ceil(2.5e-7 * L * maxdim**2))


def needed_triples(batch_dir):
    configs = set()
    for ymlpath in glob.glob(os.path.join(batch_dir, "**", "input.yml"), recursive=True):
        with open(ymlpath) as f:
            p = yaml.safe_load(f)
        configs.add((
            p["L"], p["bc"], int(p["tebd_params"]["maxdim"]),
            float(p["t1"]), float(p["g"]), float(p["omega"]),
            int(p["tebd_params"]["steps_per_period"]), int(p["parity"]),
        ))
    if not configs:
        return None

    Ls      = {c[0] for c in configs}
    bcs     = {c[1] for c in configs}
    maxdims = {c[2] for c in configs}
    assert len(Ls) == len(bcs) == len(maxdims) == 1, \
        f"{batch_dir}: mixed L/bc/maxdim L={Ls} bc={bcs} m={maxdims}"
    L, bc, maxdim = Ls.pop(), bcs.pop(), maxdims.pop()

    needed = set()
    skipped_trivial = False
    for _, _, _, t1, g, omega, spp, parity in configs:
        tau = (2 * np.pi / omega) / spp
        for n in range(spp + 1):
            t1_eff = round(t1 * np.cos(omega * tau * n), 10)
            if t1_eff == 0.0 and g == 0.0:
                skipped_trivial = True
                continue
            needed.add((t1_eff, g, parity))

    return L, bc, maxdim, sorted(needed), skipped_trivial


def write_table(parent, batch_name, L, bc, maxdim, triples):
    """Write the table directory tree and the matching .batch file."""
    folder = os.path.join(parent, f"{batch_name}_gs_table")
    os.mkdir(folder)

    for t1_val, g_val, p_val in triples:
        dirname = os.path.join(folder, f"t1{t1_val}_g{g_val}_p{PARITY_LABEL[p_val]}")
        os.mkdir(dirname)
        data = {
            "L": L, "bc": bc, "parity": p_val,
            "t": float(t1_val), "g": float(g_val),
            "nstates": NSTATES,
            "fname": os.path.abspath(os.path.join(dirname, "output.h5")),
            "dmrg_params": {
                "m": maxdim, "nsweeps": NSWEEPS, "cutoff": CUTOFF,
                "dmrg_tol": ENERGY_TOL, "noise": NOISE,
                "eigsolve_krylovdim": KDIM, "weight": WEIGHT,
            },
        }
        with open(os.path.join(dirname, "input.yml"), "w") as f:
            yaml.dump(data, f, sort_keys=False)

    # Matching HTCondor submit file, similar to the table folder.
    batch_file = os.path.join(parent, f"{batch_name}_gs_table.batch")
    mem = mem_gb(L, maxdim)
    with open(batch_file, "w") as f:
        f.write(
            f"Universe   = vanilla\n"
            f"Executable = {EXEC_PATH}\n"
            f"Output     = out.$(Process)\n"
            f"Error      = err.$(Process)\n"
            f"Log        = condor.log\n"
            f"Request_Cpus   = {REQUEST_CPUS}\n"
            f"Request_Memory = {mem}GB\n"
            f"Priority = {TABLE_PRIORITY}\n"
            f"Initialdir = $(dirname)\n"
            f"Queue dirname matching dirs "
            f"{BATCH_ROOT}/{PARENT}/{batch_name}_gs_table/*\n"
        )
    return folder, batch_file, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print counts only; write nothing")
    args = ap.parse_args()

    # Find TEBD batches: folders named L*_*_m* that are NOT already a table.
    batch_dirs = sorted(
        d for d in glob.glob(os.path.join(PARENT, "L*_*_m*"))
        if os.path.isdir(d) and not d.endswith("_gs_table")
    )
    if not batch_dirs:
        raise RuntimeError(f"No TEBD batches found under {PARENT}/")

    total_jobs = 0
    submit_cmds = []
    print(f"{'batch':22} {'L':>3} {'bc':>4} {'m':>4} {'jobs':>6} {'mem':>5}  trivial-skipped")
    for batch_dir in batch_dirs:
        batch_name = os.path.basename(batch_dir)
        table_folder = os.path.join(PARENT, f"{batch_name}_gs_table")
        if os.path.exists(table_folder):
            print(f"{batch_name:22}  SKIP (table folder already exists)")
            continue

        res = needed_triples(batch_dir)
        if res is None:
            print(f"{batch_name:22}  SKIP (no input.yml found)")
            continue
        L, bc, maxdim, triples, skipped = res
        total_jobs += len(triples)

        if args.dry_run:
            mem = mem_gb(L, maxdim)
            print(f"{batch_name:22} {L:>3} {bc:>4} {maxdim:>4} "
                  f"{len(triples):>6} {mem:>4}G  {skipped}")
        else:
            folder, bf, mem = write_table(PARENT, batch_name, L, bc, maxdim, triples)
            print(f"{batch_name:22} {L:>3} {bc:>4} {maxdim:>4} "
                  f"{len(triples):>6} {mem:>4}G  {skipped}")
            submit_cmds.append(f"condor_submit {bf}")
    print(f"TOTAL DMRG jobs across all tables: {total_jobs}")

    if not args.dry_run:
        print("\nTo submit every table (priority already set above TEBD):")
        for c in submit_cmds:
            print(f"  {c}")

if __name__ == "__main__":
    main()
