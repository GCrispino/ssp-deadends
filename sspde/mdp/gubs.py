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


def egubs_vi(V_rs,
             P_rs,
             pi_rs,
             C_max,
             lamb,
             k_g,
             V_i,
             S,
             goal,
             succ_states,
             A,
             mdp_graph,
             c=1):

    def u(c):
        return np.exp(c * lamb)

    G_i = [V_i[s] for s in V_i if mdp_graph[s]['goal']]
    n_states = len(S)
    n_actions = len(A)

    C_max_plus = max(C_max, 0)

    V = np.zeros((n_states, C_max_plus + 1))
    V_rs_C = np.zeros((n_states, C_max_plus + 2))
    P = np.zeros((n_states, C_max_plus + 2))
    pi = np.full((n_states, C_max_plus + 2), None)

    #print(C_max, len(G_i), V_rs_C.shape)
    for i in range(V_rs_C.shape[1]):
        V_rs_C[G_i, i] = V_rs[G_i]

    #V_rs_C[G_i, :] = V_rs[G_i]
    P[G_i, :] = 1
    V_rs_C[:, C_max_plus + 1] = V_rs.T
    P[:, C_max_plus + 1] = P_rs.T
    pi[:, C_max_plus + 1] = pi_rs.T

    n_updates = 0
    for C in reversed(range(C_max_plus + 1)):
        Q = np.zeros(n_actions)
        P_a = np.zeros(n_actions)
        for s in S:
            i_s = V_i[s]
            n_updates += 1
            for i_a, a in enumerate(A):
                #c__ = 0 if check_goal(utils.from_literals(s), goal) else c
                c__ = 0 if mdp_graph[s]['goal'] else c
                c_ = C + c__
                successors = succ_states[s, a]

                # Get value
                gen_q = [
                    p * V_rs_C[V_i[s_], c_] for s_, p in successors.items()
                ]
                #print(' gen_q:', gen_q, lamb, c__)
                Q[i_a] = u(c__) * \
                    np.sum(np.fromiter(gen_q, dtype=float))

                # Get probability
                gen_p = (p * P[V_i[s_], c_] for s_, p in successors.items())
                P_a[i_a] = np.sum(np.fromiter(gen_p, dtype=float))
            i_a_opt = np.argmax(u(C) * Q + k_g * P_a)
            a_opt = A[i_a_opt]
            pi[i_s, C] = a_opt

            P[i_s, C] = P_a[i_a_opt]
            V_rs_C[i_s, C] = Q[i_a_opt]
            V[i_s, C] = V_rs_C[i_s, C] + k_g * P[i_s, C]

    return V, V_rs_C, P, pi


def get_P_diff_W(s, a, P1, P2, V_i, k_g, succ_s):
    return k_g * (np.sum(
        np.fromiter((p * P1[V_i[s_]]
                     for s_, p in succ_s.items()), dtype=float)) - P2[V_i[s]])


def get_cmax(V, V_i, P, S, succ_states, A, lamb, k_g, u, c=1):
    X = get_X(V, V_i, lamb, S, succ_states, A, u)
    W = np.full(len(X), -math.inf)

    for i, ((s, a), x) in enumerate(X):
        denominator = get_P_diff_W(s, a, P, P, V_i, k_g, succ_states[s, a])
        if denominator == 0:
            W[i] = -math.inf
        else:
            W[i] = -(1 / lamb) * np.log(x / denominator)

    try:
        C_max = np.max(W[np.invert(np.isnan(W))])
    except:
        return -math.inf
    if C_max == np.inf:
        raise Exception("this shouldn't happen")

    return int(np.ceil(C_max)) if C_max != -math.inf else -math.inf


def get_X(V, V_i, lamb, S, succ_states, A, u, c=1):

    # TODO -> parece estar falhando quando s é meta.
    # verificar código que gera o succ_states e comparar com o do outro repositório
    list_X = [((s, a), (V[V_i[s]] - np.sum(
        np.fromiter((p * u(c) * V[V_i[s_]]
                     for s_, p in succ_states[s, a].items()),
                    dtype=float)))) for (s, a) in itertools.product(S, A)]

    X = np.array(list_X, dtype=object)

    return X[X.T[1] < 0]


def rs_and_egubs_vi(s0, S, A, succ_states, V_i, goal, k_g, lamb, epsilon, h,
                    mdp_graph):

    def u(c):
        return np.exp(c * lamb)

    print("Finding optimal risk sensitive lexicographic policy")
    V_rs, P_rs, pi_rs, _ = rs_lexicographic(lamb,
                                            V_i,
                                            S,
                                            h,
                                            goal,
                                            succ_states,
                                            A,
                                            mdp_graph,
                                            epsilon=epsilon)
    v_rs = V_rs[V_i[s0]]
    p_rs = P_rs[V_i[s0]]
    a_opt_rs = pi_rs[V_i[s0]]

    print("Best action at initial state for rs criterion:", a_opt_rs)
    print("Probability to goal at initial state for rs criterion:", p_rs)
    print("Optimal value at initial state for rs criterion:", v_rs)
    print()
    print("Computing C_max...")
    C_max = get_cmax(V_rs, V_i, P_rs, S, succ_states, A, lamb, k_g, u)
    print("C_max:", C_max)

    print("Running eGUBS-VI to find optimal eGUBS policy")
    return egubs_vi(V_rs, P_rs, pi_rs, C_max, lamb, k_g, V_i, S, goal,
                    succ_states, A, mdp_graph)
