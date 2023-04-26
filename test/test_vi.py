import os
import unittest

import sspde.mdp.vi as vi
import sspde.mdp.general as general
import sspde.pddl as pddl
import sspde.utils as utils

import gym.envs.registration as registration
import numpy as np
from pddlgym.structs import Predicate

dir_path = os.path.dirname(os.path.realpath(__file__))
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
V_i = {s: i for i, s in enumerate(S)}

mdp_graph_penalty = general.build_mdp_graph(obs,
                                            A,
                                            env_test_domain,
                                            0,
                                            penalty=True)
S_penalty = list(sorted([s for s in mdp_graph_penalty]))
S_i_penalty = {s: i for i, s in enumerate(S_penalty)}
V_i_penalty = {s: i for i, s in enumerate(S_penalty)}

epsilon = 1e-5


class TestVI(unittest.TestCase):

    def test_vi(self):
        env_test_domain.reset()
        succ_states = vi.get_succ_states("discounted", A, mdp_graph)

        V, pi, P, _ = vi.vi(S, succ_states, A, V_i, goal_test_domain,
                            env_test_domain, epsilon, mdp_graph)
        assert 1 < V[V_i[obs]] < 2
        assert pi[V_i[obs]] == move1_operator_pred()
        assert P[V_i[obs]] == 1

    def test_vi_penalty(self):
        env_test_domain.reset()
        penalty = 1
        succ_states = vi.get_succ_states("penalty", A, mdp_graph_penalty)

        V, pi, P, _ = vi.vi(S,
                            succ_states,
                            A,
                            V_i_penalty,
                            goal_test_domain,
                            env_test_domain,
                            epsilon,
                            mdp_graph_penalty,
                            mode="penalty",
                            penalty=penalty)

        assert V[V_i_penalty[obs]] == 1
        assert pi[V_i_penalty[obs]] == pddl.quit_action
        assert P[V_i_penalty[obs]] == 0
