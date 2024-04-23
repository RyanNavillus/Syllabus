#!/bin/bash

declare -a curricula=("SP" "FSP" "PFSP")

for curriculum in "${curricula[@]}"; do

    for seed in {0..10}; do
        sbatch <<EOF
#!/bin/bash
#SBATCH --job-name=lasertag_${curriculum}_${seed}
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=1

python ./lasertag_dr.py --track True --save-agent-checkpoints True --total-episodes 400000 --checkpoint-frequency 10000 --seed $seed --agent-curriculum $curriculum
EOF
    done
done
