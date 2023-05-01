import json

import matplotlib.pyplot as plt

import sspde.argparsing as argparsing
import sspde.plot as plot


def try_key(data, key):
    if key not in data:
        raise KeyError(f"key {key} not in data file!")
    return data[key]


args = argparsing.parse_plot_args()

data_file_path = args.data_file

with open(data_file_path) as f:
    data = json.load(f)

    penalty_param_vals = try_key(data, 'penalty_param_vals')
    penalty_vals = try_key(data, 'penalty_result_vals')

    discounted_param_vals = try_key(data, 'discounted_param_vals')
    discounted_vals = try_key(data, 'discounted_result_vals')

    mcmp_p_vals = try_key(data, 'mcmp_p_vals')
    mcmp_vals = try_key(data, 'mcmp_result_vals')
    mcmp_costs = try_key(data, 'mcmp_costs')

    alpha_vals = try_key(data, 'alpha_vals')
    alpha_mcmp_vals = try_key(data, 'alpha_mcmp_result_vals')
    alpha_mcmp_costs = try_key(data, 'alpha_mcmp_costs')

    egubs_alpha_vals = try_key(data, 'egubs_alpha_vals')
    egubs_alpha_result_vals_by_lamb = try_key(data, 'egubs_alpha_result_vals_by_lamb')
    egubs_alpha_result_probs_by_lamb = try_key(data, 'egubs_alpha_result_probs_by_lamb')

    p_max = try_key(data, 'p_max')
    v_gubs = try_key(data, 'v_gubs')

    plot.plot_data(
        penalty_param_vals,
        penalty_vals,
        discounted_param_vals,
        discounted_vals,
        mcmp_p_vals,
        mcmp_vals,
        mcmp_costs,
        alpha_vals,
        alpha_mcmp_vals,
        alpha_mcmp_costs,
        egubs_alpha_vals,
        egubs_alpha_result_vals_by_lamb,
        egubs_alpha_result_probs_by_lamb,
        p_max,
        v_gubs,
        log_alpha=args.log_scale_x_alpha,
    )

    plt.show()
