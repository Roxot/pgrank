import numpy as np
import tensorflow as tf
import argparse
import itertools

from datetime import datetime

from utils import run_model

print("Started at %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# Grid search params.
learning_rates = np.power(10., np.arange(-7, 1))                        # 10e-7 -- 1.0
reg_strengths = np.concatenate([[0.], np.power(10., np.arange(-5, 1))]) # 10e-5 -- 1.0 & 0.
epsilons = [0.1, 0.5, 0.9]

# Unchangeable params.
grad_clip_norm = 5.0
sample_weight_lower = 0.01
sample_weight_upper = 5.0
baseline = 0.7
batch_normalization = True

# Arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--k", type=int, default=2)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--average_over", type=int, default=10)
parser.add_argument("--exploration_type", type=str, default=None)
args = parser.parse_args()

print("Finding best parameters for k = %d" % args.k)
print("Num epochs = %d" % args.num_epochs)
print("Exploration type = %s" % args.exploration_type)
print("Averaging over %d runs" % args.average_over)
print("=========")

cur_best_avg_ndcgs = [-1, -1, -1]
cur_best_settings = [(), (), ()]

if args.exploration_type is None:

    # Without grid searching epsilons.
    for learning_rate, reg_str in itertools.product(learning_rates, reg_strengths):
        avg, std, min, max = run_model(args.k, grad_clip_norm=grad_clip_norm, sample_weight_lower=sample_weight_lower, \
                sample_weight_upper=sample_weight_upper, baseline=baseline, exploration_type=args.exploration_type, \
                epsilon=0., learning_rate=learning_rate, weight_reg_str=reg_str, num_epochs=args.num_epochs, \
                average_over=args.average_over, batch_normalization=batch_normalization)
        print("Current setup %s explored, avg = %f, std = %f, min = %f, max = %f" % \
                ((learning_rate, reg_str), avg, std, min, max))

        # Save the best 3 scores to see the variety.
        for i, cur_best_avg_ndcg in enumerate(cur_best_avg_ndcgs):
            if avg > cur_best_avg_ndcg:
                for j in range(i+1, len(cur_best_avg_ndcgs))[::-1]:
                    cur_best_avg_ndcgs[j] = cur_best_avg_ndcgs[j-1]
                    cur_best_settings[j] = cur_best_settings[j-1]

                cur_best_avg_ndcgs[i] = avg
                cur_best_settings[i] = (learning_rate, reg_str)
                break

        # Print the current best setups.
        print("========")
        print("Currently the best setups are %s, which got average scores of %s" % (cur_best_settings, cur_best_avg_ndcgs))
        print("========")

else:

    # With grid searching epsilons.
    for learning_rate, reg_str, epsilon in itertools.product(learning_rates, reg_strengths, epsilons):
        avg, std, min, max = run_model(args.k, grad_clip_norm=grad_clip_norm, sample_weight_lower=sample_weight_lower, \
                sample_weight_upper=sample_weight_upper, baseline=baseline, exploration_type=args.exploration_type, \
                epsilon=epsilon, learning_rate=learning_rate, weight_reg_str=reg_str, num_epochs=args.num_epochs, \
                average_over=args.average_over, batch_normalization=batch_normalization)
        print("Current setup %s explored, avg = %f, std = %f, min = %f, max = %f" % \
                ((learning_rate, reg_str, epsilon), avg, std, min, max))

        # Save the best 3 scores to see the variety.
        for i, cur_best_avg_ndcg in enumerate(cur_best_avg_ndcgs):
            if avg > cur_best_avg_ndcg:
                for j in range(i+1, len(cur_best_avg_ndcgs))[::-1]:
                    cur_best_avg_ndcgs[j] = cur_best_avg_ndcgs[j-1]
                    cur_best_settings[j] = cur_best_settings[j-1]

                cur_best_avg_ndcgs[i] = avg
                cur_best_settings[i] = (learning_rate, reg_str, epsilon)
                break

        # Print the current best setups.
        print("========")
        print("Currently the best setups are %s, which got average scores of %s" % (cur_best_settings, cur_best_avg_ndcgs))
        print("========")

print("Done at %s" % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
