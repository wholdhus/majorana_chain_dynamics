#!/bin/bash
echo "Starting conda?"
source /cluster/home/holdhuw/miniconda3/etc/profile.d/conda.sh
conda activate
echo "Starting Julia?"
julia -t 8 /cluster/home/holdhuw/majorana_chain_dynamics/dmrg_sim.jl input.yml
