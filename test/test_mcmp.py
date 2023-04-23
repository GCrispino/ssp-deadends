import os
import unittest
from copy import deepcopy

import gym.envs.registration as registration
from pddlgym.structs import Predicate

import sspde.mdp.mcmp as mcmp
import sspde.mdp.general as general
import sspde.utils as utils

dir_path = os.path.dirname(os.path.realpath(__file__))

# test domain
env_test_domain, problem = utils.create_problem_instance_from_file(
    dir_path, 'test_domain')

spec_obj = registration.EnvSpec(id="PDDLEnvTest_domain-v0", entry_point="")
setattr(env_test_domain, "spec", spec_obj)

obs, *_ = env_test_domain.reset()
goal_test_domain = problem.goal
A = list(env_test_domain.action_space.all_ground_literals(obs))
move1_operator_pred = Predicate('move1', 0)
move2_operator_pred = Predicate('move2', 0)

mdp_graph = general.build_mdp_graph(obs, A, env_test_domain, 0)
S = list(sorted([s for s in mdp_graph]))
S_i = {s: i for i, s in enumerate(S)}

variable_map, in_flow, out_flow = mcmp.get_lp_data(env_test_domain, S, A,
                                                   mdp_graph)

mcmp_cost_fn = general.create_cost_fn(mdp_graph, False)


class TestMCMP(unittest.TestCase):

    def test_mcmp_pmax_1(self):
        env_test_domain.reset()
        p_max = 1

        mincost, *_ = mcmp.mcmp(obs, S_i, variable_map, in_flow, out_flow,
                                p_max, mcmp_cost_fn, env_test_domain,
                                mdp_graph)

        var_map = deepcopy(variable_map)

        pi_func = mcmp.create_pi_func_prob(env_test_domain, var_map, in_flow,
                                           out_flow, obs, A, p_max)

        assert mincost == 2
        assert pi_func(obs, move1_operator_pred()) == 1
        assert pi_func(obs, move2_operator_pred()) == 0

    def test_mcmp_pmax_095(self):
        env_test_domain.reset()
        p_max = 0.95

        mincost, *_ = mcmp.mcmp(obs, S_i, variable_map, in_flow, out_flow,
                                p_max, mcmp_cost_fn, env_test_domain,
                                mdp_graph)

        var_map = deepcopy(variable_map)

        pi_func = mcmp.create_pi_func_prob(env_test_domain, var_map, in_flow,
                                           out_flow, obs, A, p_max)

        # non-maxprob action is now optimal
        assert mincost == 1
        assert pi_func(obs, move1_operator_pred()) == 0
        assert pi_func(obs, move2_operator_pred()) == 1

    def test_mcmp_pmax_098(self):
        env_test_domain.reset()
        p_max = 0.98

        mincost, *_ = mcmp.mcmp(obs, S_i, variable_map, in_flow, out_flow,
                                p_max, mcmp_cost_fn, env_test_domain,
                                mdp_graph)

        var_map = deepcopy(variable_map)

        pi_func = mcmp.create_pi_func_prob(env_test_domain, var_map, in_flow,
                                           out_flow, obs, A, p_max)

        # since there's no deterministic policy for this MDP that has 0.98
        # probability-to-goal, it yields a probabilistic policy
        assert mincost == 1.6
        assert pi_func(obs, move1_operator_pred()) == 0.6
        assert pi_func(obs, move2_operator_pred()) == 0.4


class TestAlphaMCMP(unittest.TestCase):
    # p_max is fixed as the MDP's maximum probability-to-goal
    p_max = 1

    def test_alpha_mcmp_alpha1(self):
        """
        alpha == 1 should return the same result as maxprob
        """
        env_test_domain.reset()
        alpha = 1

        mincost, *_ = mcmp.alpha_mcmp(obs, S_i, variable_map, in_flow,
                                      out_flow, alpha, self.p_max,
                                      mcmp_cost_fn, env_test_domain, mdp_graph)

        var_map = deepcopy(variable_map)

        pi_func = mcmp.create_pi_func_prob(env_test_domain, var_map, in_flow,
                                           out_flow, obs, A, self.p_max)

        assert mincost == 2
        assert pi_func(obs, move1_operator_pred()) == 1
        assert pi_func(obs, move2_operator_pred()) == 0

    def test_alpha_mcmp_alpha095(self):
        """
        alpha == 0.95 should return the same result as vanilla MCMP with p_max == 0.95
        """
        env_test_domain.reset()
        alpha = 0.95

        mincost, *_ = mcmp.alpha_mcmp(obs, S_i, variable_map, in_flow,
                                      out_flow, alpha, self.p_max,
                                      mcmp_cost_fn, env_test_domain, mdp_graph)

        var_map = deepcopy(variable_map)

        pi_func = mcmp.create_pi_func_prob(env_test_domain, var_map, in_flow,
                                           out_flow, obs, A, self.p_max)

        assert mincost == 1
        assert pi_func(obs, move1_operator_pred()) == 0
        assert pi_func(obs, move2_operator_pred()) == 1
