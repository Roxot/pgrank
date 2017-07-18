#!/bin/bash

set -x

ks=( "2" "4" "5" "10" )
reg_strengths=( "0.0" "1e-5" "1e-4" "1e-3" "1e-2" "1e-1" "1e-0" )
epsilons=( "0.0" "0.1" "0.5" "0.9" )
exploration_types=( "uniform" "oracle" )

for k in "${ks[@]}"
do
  for reg_str in "${reg_strengths[@]}"
  do
    for epsilon in "${epsilons[@]}"
    do
      for exp_type in "${exploration_types[@]}"
      do
        # Hacky way to do a grid search over all learning rates, reg strengths, epsilons, and exploration types.
        srun -p gpu --gres=gpu:1  --mem 40GB -c 2 python axis_search.py --tune learning_rate --k $k --exploration_type $exp_type --epsilon $epsilon --weight_reg_str $reg_str --num_epochs 10 --average_over 30 >"experiments/results/k=${k}_exp_type=${exp_type}_epsilon=${epsilon}_reg_str=${reg_str}"
      done
    done
  done
done
