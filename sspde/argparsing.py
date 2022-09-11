import argparse

DEFAULT_PROB_INDEX = 0
DEFAULT_ALGORITHM = "vi"
DEFAULT_EPSILON = 0.1
DEFAULT_GAMMA = 0.99
DEFAULT_PENALTY = 10
DEFAULT_INIT_PARAM_VALUE = 0.01
DEFAULT_VI_MODE = "discounted"
DEFAULT_BATCH = False
DEFAULT_BATCH_SIZE = 20
DEFAULT_LAMBDA = -0.1
DEFAULT_KG = 1
DEFAULT_SIMULATE = False
DEFAULT_RENDER_AND_SAVE = False
DEFAULT_PRINT_SIM_HISTORY = False
DEFAULT_PLOT_STATS = False
DEFAULT_OUTPUT_DIR = "./output"


def parse_args():
    parser = argparse.ArgumentParser(
        description='Value Iteration algorithm for PDDLGym environments.')

    parser.add_argument('--env',
                        dest='env',
                        required=True,
                        help="PDDLGym environment to solve")
    parser.add_argument(
        '--problem_index',
        type=int,
        default=DEFAULT_PROB_INDEX,
        dest='problem_index',
        help="Chosen environment's problem index to solve (default: %s)" %
        str(DEFAULT_PROB_INDEX))
    parser.add_argument('--algorithm',
                        dest='algorithm',
                        choices=['vi', 'mcmp'],
                        default=DEFAULT_ALGORITHM,
                        help="Algorithm (default: %s)" % DEFAULT_ALGORITHM)
    parser.add_argument('--epsilon',
                        dest='epsilon',
                        type=float,
                        default=DEFAULT_EPSILON,
                        help="Epsilon used for convergence (default: %s)" %
                        str(DEFAULT_EPSILON))
    parser.add_argument('--vi_mode',
                        dest='vi_mode',
                        choices=['discounted', 'penalty'],
                        default=DEFAULT_VI_MODE,
                        help="VI algorithm mode (default: %s)" % DEFAULT_VI_MODE)
    parser.add_argument('--gamma',
                        dest='gamma',
                        type=float,
                        default=DEFAULT_GAMMA,
                        help="Discount factor (default: %s)" %
                        str(DEFAULT_GAMMA))
    parser.add_argument(
        '--penalty',
        dest='penalty',
        type=float,
        default=DEFAULT_PENALTY,
        help="Penalty cost to quit when mode is 'penalty' (default: %s)" %
        str(DEFAULT_PENALTY))
    parser.add_argument(
        '--batch',
        dest='batch',
        default=DEFAULT_BATCH,
        action="store_true",
        help=
        "Defines whether or not to solve for several parameters (default: %s)"
        % DEFAULT_BATCH)
    parser.add_argument(
        '--batch_size',
        dest='batch_size',
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=
        "Size of batch in batch mode (default: %s)"
        % DEFAULT_BATCH_SIZE)
    parser.add_argument('--lambda',
                        dest='lamb',
                        type=float,
                        default=DEFAULT_LAMBDA,
                        help="Risk factor (default: %s)" % str(DEFAULT_LAMBDA))
    parser.add_argument('--k_g',
                        dest='k_g',
                        type=float,
                        default=DEFAULT_KG,
                        help="Constant goal utility (default: %s)" %
                        str(DEFAULT_LAMBDA))
    parser.add_argument(
        '--init_param_val',
        dest='init_param_val',
        type=float,
        default=DEFAULT_INIT_PARAM_VALUE,
        help="Initial value for param being varied when in batch mode (default: %s)" %
        str(DEFAULT_INIT_PARAM_VALUE))
    parser.add_argument(
        '--simulate',
        dest='simulate',
        default=DEFAULT_SIMULATE,
        action="store_true",
        help=
        "Defines whether or not to run a simulation in the problem by applying the algorithm's resulting policy (default: %s)"
        % DEFAULT_SIMULATE)
    parser.add_argument(
        '--render_and_save',
        dest='render_and_save',
        default=DEFAULT_RENDER_AND_SAVE,
        action="store_true",
        help=
        "Defines whether or not to render and save the received observations during execution to a file (default: %s)"
        % DEFAULT_RENDER_AND_SAVE)
    parser.add_argument('--output_dir',
                        dest='output_dir',
                        default=DEFAULT_OUTPUT_DIR,
                        help="Simulation's output directory (default: %s)" %
                        DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        '--print_sim_history',
        dest='print_sim_history',
        action="store_true",
        default=DEFAULT_PRINT_SIM_HISTORY,
        help=
        "Defines whether or not to print chosen actions during simulation (default: %s)"
        % DEFAULT_PRINT_SIM_HISTORY)

    parser.add_argument(
        '--plot_stats',
        dest='plot_stats',
        action="store_true",
        default=DEFAULT_PLOT_STATS,
        help=
        "Defines whether or not to run a series of episodes with both a random policy and the policy returned by the algorithm and plot stats about these runs (default: %s)"
        % DEFAULT_PLOT_STATS)

    return parser.parse_args()