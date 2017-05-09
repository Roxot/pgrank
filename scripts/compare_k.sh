#!/bin/bash

# K = 2
# python pglistsearch_2.py --lr 1e-4 --k 2 --log_dir "compare_k/k=2" --num_iter 100000

# K = 3
# python pglistsearch_2.py --lr 1e-4 --k 3 --log_dir "compare_k/k=3" --num_iter 100000
# python pglistsearch_2.py --lr 1e-5 --k 3 --log_dir "compare_k/k=3,lr=1e-5" --num_iter 100000

# K = 4
# python pglistsearch_2.py --lr 1e-4 --k 4 --log_dir "compare_k/k=4" --num_iter 100000

# K = 5
# python pglistsearch_2.py --lr 1e-4 --k 5 --log_dir "compare_k/k=5" --num_iter 100000

# K = 6
# python pglistsearch.py --lr 1e-4 --k 6 --log_dir "compare_k/k=6" --num_iter 100000

# K = 7
# python pglistsearch.py --lr 1e-4 --k 7 --log_dir "compare_k/k=7" --num_iter 100000

# K = 8
# python pglistsearch.py --lr 1e-4 --k 8 --log_dir "compare_k/k=8" --num_iter 100000

# K = 9
python pglistsearch.py --lr 1e-4 --k 9 --log_dir "compare_k/k=9" --num_iter 100000

# K = 10
python pglistsearch.py --lr 1e-4 --k 10 --log_dir "compare_k/k=10" --num_iter 100000
