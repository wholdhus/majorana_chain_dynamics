#!/bin/bash
echo "Starting conda?"
source /cluster/home/castilw2/miniconda3/etc/profile.d/conda.sh
conda activate
echo "Starting Julia?"
/cluster/home/castilw2/.juliaup/bin/julia -t 8 /cluster/research-groups/rahmani/Liam/majorana_chain_dynamics/current/tebd_run.jl input.yml