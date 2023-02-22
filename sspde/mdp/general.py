import math
import numpy as np

from pddlgym.core import get_successor_states, InvalidAction
from pddlgym.inference import check_goal

import sspde.pddl as pddl
import sspde.rendering as rendering


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


def to_no_penalty_mdp_graph(mdp_graph):
    return {
        s: {
            **v,
            **{
                'actions': {
                    a: outcomes
                    for a, outcomes in v['actions'].items() if a != pddl.quit_action
                }
            }
        }
        for s, v in mdp_graph.items() if s != pddl.quit_state
    }


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
    for _, outcomes in s_mdp['actions'].items():
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


def traverse_mdp_graph_with_policy(s0,
                                   pi,
                                   mdp_graph,
                                   nonstat=False,
                                   C_maxs=None):

    if nonstat:
        if C_maxs is None:
            raise ValueError("C_maxs cannot be None")

    # TODO -> change to deque if it makes a difference
    if nonstat:
        to_visit = [(s0, 0)]
    else:
        to_visit = [s0]

    visited = {}

    while len(to_visit) != 0:
        if nonstat:
            s, C = to_visit.pop()
            a = pi(s, C)
            c = 0 if mdp_graph[s]['goal'] else 1

            if s not in visited:
                visited[s] = {}

            visited[s][C] = a
            if C >= C_maxs(s):
                continue

            C_ = C + c
        else:
            s = to_visit.pop()
            a = pi(s)
            c = 1
            visited[s] = a

        if mdp_graph[s]['goal']:
            continue

        succs = mdp_graph[s]['actions'][a].keys()

        for succ in succs:
            if nonstat:
                succ_not_visited = succ not in visited
                succ_C_not_visited = (not succ_not_visited) and (
                    C_ not in visited[succ])
                #if (not succ_not_visited) or (not succ_C_not_visited):
                if (succ_not_visited) or (succ_C_not_visited):
                    to_visit.append((succ, C_))
            else:
                if succ not in visited:
                    to_visit.append(succ)

    return visited


def compare_policies(s0, pi_ns, pi, C_maxs, mdp_graph, env):
    """
        Compares a non-stationary policy pi_ns with a stationary policy pi
    """
    reachable_stat = traverse_mdp_graph_with_policy(s0, pi, mdp_graph)
    reachable_non_stat = traverse_mdp_graph_with_policy(s0,
                                                        pi_ns,
                                                        mdp_graph,
                                                        nonstat=True,
                                                        C_maxs=C_maxs)

    log = False
    if log:
        print("stat policy:")
        for s, a in reachable_stat.items():
            s_id = rendering.get_state_id(env, s)
            print(f"  pistat({s_id}): {a}")

        print("non stat policy:")
        for s, cs in reachable_non_stat.items():
            s_id = rendering.get_state_id(env, s)
            print(f"  pinonstat({s_id}, C):")
            for c, a in cs.items():
                print(f"    pinonstat({s_id}, {c}): {a}")

    diffs = {}
    for s in reachable_stat:
        if mdp_graph[s]['goal']:
            continue
        if s not in reachable_non_stat:
            diffs[s] = -1
            continue

        s_id = rendering.get_state_id(env, s)
        for C in sorted(reachable_non_stat[s]):
            a = reachable_non_stat[s][C]
            if a != pi(s):
                assert a == pi_ns(s, C)

                if s not in diffs or diffs[s] > C:
                    diffs[s] = C
                #break

    for s in set(reachable_non_stat) - set(reachable_stat):
        diffs[s] = -2

    return diffs, reachable_stat, reachable_non_stat
