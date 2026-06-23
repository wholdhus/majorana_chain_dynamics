#!/bin/bash
echo "Starting conda?"
source /cluster/home/harri290/miniconda3/etc/profile.d/conda.sh
conda activate
echo "Starting Julia?"
/cluster/home/harri290/.juliaup/bin/julia -t 2 /cluster/research-groups/rahmani/Jack/tebd_run.jl input.yml