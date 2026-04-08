#!/bin/bash
echo "Starting conda?"
source /cluster/home/USERNAME/miniconda3/etc/profile.d/conda.sh
conda activate
echo "Starting Julia?"
julia -t 8 /cluster/home/USERNAME/majorana_chain_dynamics/dmrg_sim.jl input.yml
