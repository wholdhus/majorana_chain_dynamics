#!/bin/bash
echo "Starting conda?"
source /cluster/home/castilw2/miniconda3/etc/profile.d/conda.sh
conda activate
echo "Starting Julia?"
julia -t 8 /cluster/home/castilw2/majorana_chain_dynamics/dmrg_run.jl input.yml