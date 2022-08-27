import numpy as np

import sspde.pddl as pddl

from sspde.mdp.general import sample_state
from sspde.mdp.rs import rs_lexicographic_eval


def eval_policy(s0, succ_states, pi, cost_fn, P, lamb, k_g, epsilon, mdp_graph, env, V_i=None):
    if V_i is None:
        S = list(sorted([s for s in mdp_graph]))
        V_i = {s: i for i, s in enumerate(S)}

    V_rs, _ = rs_lexicographic_eval(succ_states, V_i, pi, cost_fn, lamb, epsilon, mdp_graph, env)

    print("Value of rs lexicographic policy", V_rs[V_i[s0]])
    print("Prob to goal of rs lexicographic policy", P[V_i[s0]])

    return np.exp(lamb) * V_rs[V_i[s0]] + k_g * P[V_i[s0]]


# TODO -> debug this when using penalty such that
# -> the found_goal array is containing True values even when the policy has probability-to-goal equal to 0
def eval_policy_sim(s0,
                    mdp_graph,
                    pi,
                    cost_fn,
                    u,
                    k_g,
                    n_episodes=100,
                    horizon=100):
    cumcosts = np.zeros(n_episodes)
    found_goal = np.zeros(n_episodes, dtype=bool)
    for i in range(n_episodes):
        s = s0

        for _ in range(horizon):
            a = pi(s)

            c = cost_fn(s, a)
            cumcosts[i] += c

            next_state = sample_state(s, a, mdp_graph)
            # if s == pddl.quit_state:
            #     print(s)
            #     exit("exit")

            s = next_state

            if mdp_graph[s]['goal'] and s != pddl.quit_state:
                found_goal[i] = True
                break

    # TODO -> cumcosts seem way too high for some cases when using penalty
    res = u(cumcosts) + found_goal
    mean_eval = np.mean(res)
    std_eval = np.std(res)
    print("ops")
    print(cumcosts)
    print(found_goal)
    print("ops")

    return mean_eval, std_eval
