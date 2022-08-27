import math
import numpy as np

from pddlgym.core import get_successor_states, InvalidAction
from pddlgym.inference import check_goal

import sspde.pddl as pddl
import sspde.rendering as rendering


def get_all_reachable(s, A, env, reach=None):
    reach = {} if not reach else reach

    reach[s] = {}
    for a in A:
        try:
            succ = get_successor_states(s,
                                        a,
                                        env.domain,
                                        raise_error_on_invalid_action=True,
                                        return_probs=True)
        except InvalidAction:
            succ = {s: 1.0}
        reach[s][a] = {s_: prob for s_, prob in succ.items()}
        for s_ in succ:
            if s_ not in reach:
                reach.update(get_all_reachable(s_, A, env, reach))
    return reach


def build_mdp_graph(s, A, env, problem_index, penalty=False):
    problem = env.problems[problem_index]
    goal = problem.goal

    mdp = {}

    if penalty:
        mdp[pddl.quit_state] = {
            "goal": True,
            "actions":
            {a: {
                pddl.quit_state: 1
            }
             for a in A + [pddl.quit_action]}
        }

    def fn(s, reach_s):
        nonlocal mdp
        is_goal = check_goal(s, goal)

        reach_s_final = reach_s
        if penalty:
            #print("state:", rendering.get_state_id(env, s))
            reach_s_final = {**reach_s, pddl.quit_action: {pddl.quit_state: 1}}

        mdp[s] = {
            "goal": is_goal,
            "actions": reach_s_final if not is_goal else {}
        }
        # do something

    pddl.traverse_all_reachable(s, A, env, fn)

    return mdp


def Pr(s, a, s_, mdp):
    s_mdp = mdp[s]
    if a not in s_mdp['actions']:
        return 0

    a_outcomes = s_mdp['actions'][a]

    if s_ not in a_outcomes:
        return 0
    return a_outcomes[s_]


def get_succs(s, mdp):
    s_mdp = mdp[s]

    succs = set()
    for a, outcomes in s_mdp['actions'].items():
        succs.update(set(outcomes.keys()))

    return succs


def sample_state(s, a, mdp):
    outcomes = mdp[s]['actions'][a]
    probs_next_states = list(outcomes.values())
    next_states = list(outcomes.keys())

    i_next_state = np.random.choice(len(next_states), p=probs_next_states)

    return next_states[i_next_state]


def create_cost_fn(mdp_graph, has_penalty, penalty=None):

    def cost_fn(s, a):
        if mdp_graph[s]['goal']:
            return 0

        if has_penalty:
            if a == pddl.quit_action:
                return math.inf

        return 1

    return cost_fn


def create_pi_func(pi, V_i):

    def pi_func(s):
        pi_ = np.copy(pi)
        if s == pddl.quit_state:
            return pddl.quit_action
        return pi_[V_i[s]]

    return pi_func
