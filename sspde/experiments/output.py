from dataclasses import dataclass, asdict
from typing import Dict

import sspde.utils as utils


@dataclass
class Result:

    def to_json(self):
        return asdict(self)


@dataclass
class GUBSComparisonExprOutput(Result):
    penalty_result_vals: list
    penalty_param_vals: list
    discounted_result_vals: list
    discounted_param_vals: list
    mcmp_result_vals: list
    mcmp_p_vals: list
    mcmp_costs: list
    p_max: float
    v_gubs: float

    @staticmethod
    def from_json(data):
        penalty_param_vals = utils.try_key(data, 'penalty_param_vals')
        penalty_vals = utils.try_key(data, 'penalty_result_vals')

        discounted_param_vals = utils.try_key(data, 'discounted_param_vals')
        discounted_vals = utils.try_key(data, 'discounted_result_vals')

        mcmp_p_vals = utils.try_key(data, 'mcmp_p_vals')
        mcmp_vals = utils.try_key(data, 'mcmp_result_vals')
        mcmp_costs = utils.try_key(data, 'mcmp_costs')

        v_gubs = utils.try_key(data, 'v_gubs')

        p_max = utils.try_key(data, 'p_max')

        return GUBSComparisonExprOutput(penalty_vals, penalty_param_vals,
                                        discounted_vals, discounted_param_vals,
                                        mcmp_vals, mcmp_p_vals, mcmp_costs,
                                        p_max, v_gubs)


@dataclass
class AlphaExprOutput(Result):
    alpha_vals: list
    alpha_mcmp_result_vals: list
    alpha_mcmp_costs: list
    egubs_alpha_vals: list
    egubs_alpha_result_vals_by_lamb: Dict[float, list]
    egubs_alpha_result_probs_by_lamb: Dict[float, list]

    @staticmethod
    def from_json(data):
        alpha_vals = utils.try_key(data, 'alpha_vals')
        alpha_mcmp_vals = utils.try_key(data, 'alpha_mcmp_result_vals')
        alpha_mcmp_costs = utils.try_key(data, 'alpha_mcmp_costs')

        egubs_alpha_vals = utils.try_key(data, 'egubs_alpha_vals')
        egubs_alpha_result_vals_by_lamb = utils.try_key(
            data, 'egubs_alpha_result_vals_by_lamb')
        egubs_alpha_result_probs_by_lamb = utils.try_key(
            data, 'egubs_alpha_result_probs_by_lamb')

        return AlphaExprOutput(alpha_vals, list(alpha_mcmp_vals),
                               alpha_mcmp_costs, egubs_alpha_vals,
                               egubs_alpha_result_vals_by_lamb,
                               egubs_alpha_result_probs_by_lamb)
