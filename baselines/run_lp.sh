myArray=("bigfish" "bossfight" "caveflyer" "chaser" "climber" "coinrun" "dodgeball" "fruitbot" "heist" "jumper" "leaper" "maze" "miner" "ninja" "plunder" "starpilot")
seeds=5
for env in ${myArray[@]}; do
  for seed in $(seq $seeds); do
    sbatch lp.slurm_template $env $seed
  done
done