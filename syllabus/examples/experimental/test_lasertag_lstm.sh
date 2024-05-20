#!/bin/bash

declare -a curricula=("SP" "FSP" "PFSP")
declare -a env_curricula=("DR", "SP")

for curriculum in "${curricula[@]}"; do
    for env_curriculum in "${env_curricula[@]}"; do
        python ./lasertag_rnn_lstm.py \\
        --track True \\
        --save-agent-checkpoints True\\ 
        --total-updates 10000 \\
        --agent-update-frequency 2000 \\
        --checkpoint-frequency 2000 \\
        --seed $seed \\
        --agent-curriculum $curriculum \\ 
        --env-curriculum $env_curriculum
    done
done
