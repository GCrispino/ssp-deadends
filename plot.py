import json

import matplotlib.pyplot as plt

import sspde.argparsing as argparsing
import sspde.experiments.output as output
import sspde.plot as plot
import sspde.utils as utils

args = argparsing.parse_plot_args()

data_file_path = args.data_file

with open(data_file_path) as f:
    data = json.load(f)

    gubs_comparison_expr_vals_json = utils.try_key(
        data, 'gubs_comparison_expr_vals')
    alpha_expr_vals_json = utils.try_key(data, 'alpha_expr_vals')

    gubs_comparison_expr_vals = None if gubs_comparison_expr_vals_json is None else output.GUBSComparisonExprOutput.from_json(
        {
            **gubs_comparison_expr_vals_json, 'p_max': data['p_max']
        })
    alpha_expr_vals = None if alpha_expr_vals_json is None else output.AlphaExprOutput.from_json(
        alpha_expr_vals_json)

    p_max = utils.try_key(data, 'p_max')

    plot.plot_data(
        gubs_comparison_expr_vals,
        alpha_expr_vals,
        p_max,
        log_alpha=args.log_scale_x_alpha,
    )

    plt.show()
