#!/bin/bash

# K = 10
# python pglistsearch_2.py --lr 1e-6 --k 10 --log_dir lr_tuning_k10/1e-6
# python pglistsearch_2.py --lr 1e-5 --k 10 --log_dir lr_tuning_k10/1e-5
# python pglistsearch_2.py --lr 1e-4 --k 10 --log_dir lr_tuning_k10/1e-4
# python pglistsearch_2.py --lr 1e-3 --k 10 --log_dir lr_tuning_k10/1e-3
# python pglistsearch_2.py --lr 1e-2 --k 10 --log_dir lr_tuning_k10/1e-2
# python pglistsearch_2.py --lr 1e-1 --k 10 --log_dir lr_tuning_k10/1e-1

# K = 4
# python pglistsearch_2.py --lr 1e-6 --k 4 --log_dir lr_tuning_k4/1e-6
# python pglistsearch_2.py --lr 1e-5 --k 4 --log_dir lr_tuning_k4/1e-5
# python pglistsearch_2.py --lr 1e-4 --k 4 --log_dir lr_tuning_k4/1e-4
# python pglistsearch_2.py --lr 1e-3 --k 4 --log_dir lr_tuning_k4/1e-3
# python pglistsearch_2.py --lr 1e-2 --k 4 --log_dir lr_tuning_k4/1e-2
# python pglistsearch_2.py --lr 1e-1 --k 4 --log_dir lr_tuning_k4/1e-1

# K = 5
# python pglistsearch_2.py --lr 1e-6 --k 5 --log_dir lr_tuning_k5/1e-6
# python pglistsearch_2.py --lr 1e-5 --k 5 --log_dir lr_tuning_k5/1e-5
# python pglistsearch_2.py --lr 1e-4 --k 5 --log_dir lr_tuning_k5/1e-4
# python pglistsearch_2.py --lr 1e-3 --k 5 --log_dir lr_tuning_k5/1e-3
# python pglistsearch_2.py --lr 1e-2 --k 5 --log_dir lr_tuning_k5/1e-2
# python pglistsearch_2.py --lr 1e-1 --k 5 --log_dir lr_tuning_k5/1e-1

# K = 3
# python pglistsearch_2.py --lr 1e-6 --k 3 --log_dir lr_tuning_k3/1e-6
python pglistsearch_2.py --lr 1e-5 --k 3 --log_dir lr_tuning_k3/1e-5 --num_iter 25000
python pglistsearch_2.py --lr 1e-4 --k 3 --log_dir lr_tuning_k3/1e-4 --num_iter 25000
python pglistsearch_2.py --lr 1e-3 --k 3 --log_dir lr_tuning_k3/1e-3 --num_iter 25000
# python pglistsearch_2.py --lr 1e-2 --k 3 --log_dir lr_tuning_k3/1e-2
# python pglistsearch_2.py --lr 1e-1 --k 3 --log_dir lr_tuning_k3/1e-1

# K = 2
# python pglistsearch_2.py --lr 1e-6 --k 2 --log_dir lr_tuning_k2/1e-6
python pglistsearch_2.py --lr 1e-5 --k 2 --log_dir lr_tuning_k2/1e-5 --num_iter 25000
python pglistsearch_2.py --lr 1e-4 --k 2 --log_dir lr_tuning_k2/1e-4 --num_iter 25000
python pglistsearch_2.py --lr 1e-3 --k 2 --log_dir lr_tuning_k2/1e-3 --num_iter 25000
# python pglistsearch_2.py --lr 1e-2 --k 2 --log_dir lr_tuning_k2/1e-2
# python pglistsearch_2.py --lr 1e-1 --k 2 --log_dir lr_tuning_k2/1e-1
