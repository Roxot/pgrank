import numpy as np
import tensorflow as tf
import sys
import argparse
import itertools

from datetime import datetime

from utils import run_model

print("Started at %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Grid search params.
learning_rates = np.power(10., np.arange(-7, 1))                        # 10e-7 -- 1.0
reg_strengths = np.concatenate([[0.], np.power(10., np.arange(-5, 1))]) # 10e-5 -- 1.0 & 0.
epsilons = [0.1, 0.5, 0.9]
baselines = [0., 0.5, 0.7, 0.9]

# Unchangeable params.
grad_clip_norm = 5.0
sample_weight_lower = 0.01
sample_weight_upper = 5.0
batch_normalization = True

# Arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--tune", type=str, default=None)
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--average_over", type=int, default=30)

# Parameter values if not tuned.
parser.add_argument("--learning_rate", type=float, default=1e-4)
parser.add_argument("--weight_reg_str", type=float, default=0.)
parser.add_argument("--epsilon", type=float, default=0.)
parser.add_argument("--baseline", type=float, default=0.)
parser.add_argument("--exploration_type", type=str, default=None)
args = parser.parse_args()

if args.tune is None:
    print("No tunable parameter given, please use --tune learning_rate for example.")
    sys.exit()

# Set up the tunable parameters.
if args.tune == 'learning_rate':
    tunable_params = learning_rates
elif args.tune == 'weight_reg_str':
    tunable_params = reg_strengths
elif args.tune == 'epsilon':
    tunable_params = epsilons
elif args.tune == 'baseline':
    tunable_params = baselines
else:
    print("Unknown tunable parameter.")
    sys.exit()
tunable_param_name = args.tune

print("Finding best parameters for k = %d" % args.k)
print("Num epochs = %d" % args.num_epochs)
print("Averaging over %d runs" % args.average_over)
print("Tuning the parameter: %s\n" % args.tune)

print("If not tuned the following parameter values are used:")
print("Learning rate = %f" % args.learning_rate)
print("Epsilon = %f" % args.epsilon)
print("Exploration type = %s" % args.exploration_type)
print("Weight regularization strength = %s" % args.weight_reg_str)
print("Baseline = %s" % args.baseline)
print("=========")

cur_best_avg_ndcgs = [-1, -1, -1]
cur_best_settings = [(), (), ()]

for tunable_param in tunable_params:

    params = {
        'grad_clip_norm': grad_clip_norm,
        'sample_weight_lower': sample_weight_lower,
        'sample_weight_upper': sample_weight_upper,
        'baseline': args.baseline,
        'exploration_type': args.exploration_type,
        'epsilon': args.epsilon,
        'learning_rate': args.learning_rate,
        'weight_reg_str': args.weight_reg_str,
        'num_epochs': args.num_epochs,
        'average_over': args.average_over,
        'batch_normalization': batch_normalization
    }
    params[tunable_param_name] = tunable_param

    avg, std, min, max = run_model(args.k, **params)
    print("Current setup %s explored, avg = %f, std = %f, min = %f, max = %f" % \
            ((tunable_param), avg, std, min, max))

    # Save the best 3 scores to see the variety.
    for i, cur_best_avg_ndcg in enumerate(cur_best_avg_ndcgs):
        if avg > cur_best_avg_ndcg:
            for j in range(i+1, len(cur_best_avg_ndcgs))[::-1]:
                cur_best_avg_ndcgs[j] = cur_best_avg_ndcgs[j-1]
                cur_best_settings[j] = cur_best_settings[j-1]

            cur_best_avg_ndcgs[i] = avg
            cur_best_settings[i] = (tunable_param)
            break

    # Print the current best setups.
    print("========")
    print("Currently the best setups are %s, which got average scores of %s" % (cur_best_settings, cur_best_avg_ndcgs))
    print("========")

print("Done at %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
