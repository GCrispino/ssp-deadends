# ssp-deadends

Implementation of algorithms to solve the criteria eGUBS, MCMP, $\alpha$-MCMP, fSSPUDE and discounted cost criterion, as well as the and standard criterion for probabilistic planning problems using PDDLGym.

This code is used in the second set of experiments in the paper "GUBS criterion: arbitrary trade-offs between cost and probability-to-goal in stochastic planning based on Expected Utility Theory" ([link](https://www.sciencedirect.com/science/article/pii/S0004370222001886)), and in the paper " $\alpha$-MCMP: trade-offs between probability and cost in SSPs with the MCMP criterion", accepted for publishing in the [BRACIS 2023](https://www.bracis.dcc.ufmg.br/home) conference.



## Installing

1. Create and activate virtual environment:
```
$ python -m venv testenv
$ source testenv/bin/activate
```
2. Install dependencies
```
$ pip install -r requirements.txt
```

## Running
Run the `experiment.py` file to solve all criteria for specified values to be used as parameter for each criterion.

For example, the following command solves the first instance of the Triangle Tireworld domain with 0.1, 0.5 and 1 as values of the maximum probability value for MCMP, 0, 25 and 50 as values of the finite penalty in the fSSPUDE criterion, and 0.8, 0.9 and 0.99 as the discount values for the discounted cost criterion.
Also, it sets the value of epsilon as `1e-3`, and `1e-10` and `0.3` the values of the eGUBS parameters `k_g` and `lambda`, respectively:

```
$ python experiment.py --env PDDLEnvTireworld-v0 --problem_index 0 --pmax_vals 0.1 0.5 1 --penalty_vals 0 25 50 --gamma_vals 0.8 0.9 0.99 --epsilon 1e-3 --k_g 1e-10 --lambda -0.3 --render_and_save
```

Running `$ python experiment.py --help` will print a description of each possible parameter that can be set:
```
❯ python experiment.py --help
usage: experiment.py [-h] --env ENV [--problem_index PROBLEM_INDEX]
                     [--algorithm {vi,mcmp}] [--epsilon EPSILON]
                     [--vi_mode {discounted,penalty}] [--gamma GAMMA]
                     [--gamma_vals [GAMMA_VALS ...]] [--penalty PENALTY]
                     [--penalty_vals [PENALTY_VALS ...]]
                     [--pmax_vals [PMAX_VALS ...]] [--alpha_vals [ALPHA_VALS ...]]
                     [--lambda_vals [LAMB_VALS ...]] [--batch]
                     [--batch_size BATCH_SIZE] [--limit_time] [--compare_policies]
                     [--lambda LAMB] [--k_g K_G] [--init_param_val INIT_PARAM_VAL]
                     [--simulate] [--render_and_save] [--output_dir OUTPUT_DIR]
                     [--print_sim_history] [--plot_stats]
                     [--no_run_alpha_experiments]
                     [--no_run_gubs_comparison_experiments]

Implementation of different algorithms for solving SSPs with deadends described as
PDDLGym environments.

options:
  -h, --help            show this help message and exit
  --env ENV             PDDLGym environment to solve
  --problem_index PROBLEM_INDEX
                        Chosen environment's problem index to solve (default: 0)
  --algorithm {vi,mcmp}
                        Algorithm (default: vi)
  --epsilon EPSILON     Epsilon used for convergence (default: 0.1)
  --vi_mode {discounted,penalty}
                        VI algorithm mode (default: discounted)
  --gamma GAMMA         Discount factor (default: 0.99)
  --gamma_vals [GAMMA_VALS ...]
                        Specific discount factor values to run experiments for
                        (default: [])
  --penalty PENALTY     Penalty cost to quit when mode is 'penalty' (default: 10)
  --penalty_vals [PENALTY_VALS ...]
                        Specific penalty values to run experiments for (default: [])
  --pmax_vals [PMAX_VALS ...]
                        Specific p_max values to run experiments on MCMP for
                        (default: [1])
  --alpha_vals [ALPHA_VALS ...]
                        Specific alpha values to run experiments on alphaMCMP for
                        (default: [0.999])
  --lambda_vals [LAMB_VALS ...]
                        Specific lambda values to run for eGUBS when running
                        experiments by alpha values (default: None)
  --batch               Defines whether or not to solve for several parameters
                        (default: False)
  --batch_size BATCH_SIZE
                        Size of batch in batch mode (default: 20)
  --limit_time          Defines whether or not to limit solving of alternate
                        criteria by the time it takes to solve the problem for the
                        eGUBS criterion (default: False)
  --compare_policies    Defines whether or not to run policy comparison analysis
                        (default: False)
  --lambda LAMB         Risk factor (default: -0.1)
  --k_g K_G             Constant goal utility (default: -0.1)
  --init_param_val INIT_PARAM_VAL
                        Initial value for param being varied when in batch mode
                        (default: 0.01)
  --simulate            Defines whether or not to run a simulation in the problem by
                        applying the algorithm's resulting policy (default: False)
  --render_and_save     Defines whether or not to render and save the received
                        observations during execution to a file (default: False)
  --output_dir OUTPUT_DIR
                        Simulation's output directory (default: ./output)
  --print_sim_history   Defines whether or not to print chosen actions during
                        simulation (default: False)
  --plot_stats          Defines whether or not to run a series of episodes with both
                        a random policy and the policy returned by the algorithm and
                        plot stats about these runs (default: False)
  --no_run_alpha_experiments
                        Defines whether or not to run experiments based on values of
                        alpha (default: False)
  --no_run_gubs_comparison_experiments
                        Defines whether or not to run experiments to compare
                        different criteria to eGUBS (default: False)
```

NOTE: currently running solvers via the `main.py` is not working.

## Testing
To run the test suite, run the following:

```
$ PYTHONPATH=. pytest
```

## Experiments in papers

### $\alpha$-MCMP: trade-offs between probability and cost in SSPs with the MCMP criterion

The commands used for the experiments of this paper were the following:

#### Navigation domain
```
$ python experiment.py --env PDDLEnvNavigation10-v0 --problem_index 0 --epsilon 1e-8 --lambda_vals -0.01 -0.02 -0.03 -0.04 -0.05 -0.06 -0.07 -0.08 -0.09 -0.1 -0.2 --alpha_vals 1e-5 0.0001 0.001 0.01 0.1 0.5 0.999 --no_run_gubs_comparison_experiments --render_and_save --plot
```

#### River domain
```
$ python experiment.py --env PDDLEnvRiver-alt-v0 --problem_index 4 --epsilon 1e-8 --lambda_vals -0.01 -0.1 -0.2 -0.3 -0.4 -0.5 --alpha_vals 1e-5 0.0001 0.001 0.01 0.1 0.5 0.999 --render_and_save --no_run_gubs_comparison_experiments
```

#### Triangle Tireworld domain
```
$ python experiment.py --env PDDLEnvTireworld-v0 --problem_index 2 --epsilon 1e-8 --lambda_vals -0.1 -0.2 -0.25 -0.3 -0.35 -0.4 --alpha_vals 1e-5 0.0001 0.001 0.01 0.1 0.5 0.999 --render_and_save --no_run_gubs_comparison_experiments
```

The final data files used for generating the charts included in the paper are included in the `data/bracis2023` directory.
The only difference between them and the files generated by the commands above is that in the data for the Navigation domain, the resuls for the lambda values of -0.01, -0.02, -0.03 and -0.04 were removed for better clarity since they are the same to the results of -0.05.

To generate the charts, you can use the `plot.py` script and pass the data file as a parameter.
For example, for doing this for the River domain data, run the following command:
```
$ python plot.py --log_scale_x_alpha --data_file river.json
```

### GUBS criterion: arbitrary trade-offs between cost and probability-to-goal in stochastic planning based on Expected Utility Theory
To be included.
