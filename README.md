# Policy Gradient Ranker

How to run:

```
python run_pg.py
```

For run options use:

```
python run_pg.py --help
```

Do a grid search:

```
python grid_search.py --k [list_length]
```

The grid search ranges are predefined in `grid_search.py`. For additional options use the `--help` command.

Do an axis search:

```
python axis_search --tune [param_to_run_an_axis_search_over] --k [list_length]
```

To average each setting over 10 runs use `--average_over 10` (default is 30). The tunable parameters are `learning_rate`, `weight_reg_str`, `baseline` and `epsilon` and their ranges are defined in the script. The type of exploration can be set using `--exploration_type` which can take the values `uniform` and `oracle`. For parameters over which no axis search is done, they can be set to a value as desired as `--[param_name] param_value`. For additional options use the `--help` command.

For example:

```
python axis_search.py --tune baseline --k 5 --average_over 10 --learning_rate 1e-4 --weight_reg_str 0.0001
```

Will perform an axis search over the baseline values defined in `axis_search.py` for lists of length 5. It will use a learning rate of 1e-4 and a weight regularization strength of 0.0001 for all experiments in the axis search. It will perform each setting 10 times and report the setting with the highest average.


During the grid and axis searches the average, standard deviation, minimum and maximum for all setups will be reported in the standard output, for the number of runs each setting is run for according to the `--average_over` parameter.
