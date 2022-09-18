import numpy as np

import sspde.rendering as rendering


# TODO -> parece estar errado pro critério de penalidade em valores pequenos
#           - verificar se em outros valores faz sentido também
def rs_lexicographic_eval(succ_states,
                          V_i,
                          A,
                          pi,
                          cost_fn,
                          lamb,
                          epsilon,
                          mdp_graph,
                          env,
                          prob_policy=False):

    G_i = [V_i[s] for s in V_i if mdp_graph[s]['goal']]
    not_goal = [s for s in mdp_graph if not mdp_graph[s]['goal']]
    n_states = len(mdp_graph)

    # initialize
    V = np.zeros(n_states, dtype=float)

    # TODO -> add heuristics here?
    #for s in not_goal:
    #    V[V_i[s]] = h_v(s)
    #P = np.zeros(n_states, dtype=float)
    V[G_i] = -np.sign(lamb)
    #P[G_i] = 1

    i = 1

    pi_cache = {}

    def pi_cache_fn(s, a):
        if (s, a) not in pi_cache:
            pi_cache[s, a] = pi(s, a)
        return pi_cache[s, a]

    while True:
        V_ = np.copy(V)
        #P_ = np.copy(P)
        for s in not_goal:

            def Q(s, a):
                all_reachable = succ_states[s, a]

                c = cost_fn(s, a)
                return np.sum([
                    np.exp(lamb * c) * V[V_i[s_]] * p
                    for s_, p in all_reachable.items()
                ])

            if prob_policy:
                action_result = sum(pi_cache_fn(s, a) * Q(s, a) for a in A)
            else:
                # deterministic policy
                action_result = Q(s, pi(s))

            V_[V_i[s]] = action_result

        v_norm = np.linalg.norm(V_ - V, np.inf)

        # print("Iteration", i)
        # print(' delta1:', v_norm, p_norm, v_norm + p_norm)
        #if v_norm + p_norm < epsilon:
        if v_norm < epsilon:
            break
        V = V_
        #P = P_
        i += 1

    #print(f'{i} iterations')
    return V, i


def rs_lexicographic(lamb,
                     V_i,
                     S,
                     h_v,
                     goal,
                     succ_states,
                     A,
                     mdp_graph,
                     c=1,
                     epsilon=1e-3,
                     n_iter=None):

    def u(c):
        return np.exp(lamb * c)

    G_i = [V_i[s] for s in V_i if mdp_graph[s]['goal']]
    #G_i = [V_i[s] for s in V_i if check_goal(utils.from_literals(s), goal)]
    not_goal = [s for s in mdp_graph if not mdp_graph[s]['goal']]
    #not_goal = [s for s in S if not check_goal(utils.from_literals(s), goal)]
    n_states = len(S)

    # initialize
    V = np.zeros(n_states, dtype=float)
    for s in not_goal:
        V[V_i[s]] = h_v(s)
    pi = np.full(n_states, None)
    P = np.zeros(n_states, dtype=float)
    V[G_i] = -np.sign(lamb)
    P[G_i] = 1
    if not isinstance(A, np.ndarray):
        A = np.array(A)

    i = 1

    P_not_max_prob = np.copy(P)
    while True:
        V_ = np.copy(V)
        P_ = np.copy(P)
        for s in not_goal:

            all_reachable = [succ_states[s, a] for a in A]
            actions_results_p = np.array([
                np.sum([P[V_i[s_]] * p for s_, p in all_reachable[i].items()])
                for i, a in enumerate(A)
            ])

            # set maxprob
            max_prob = np.max(actions_results_p)
            P_[V_i[s]] = max_prob
            i_A_max_prob = np.argwhere(
                actions_results_p == max_prob).reshape(-1)
            A_max_prob = A[i_A_max_prob]
            not_max_prob_actions_results = actions_results_p[
                actions_results_p != max_prob]

            P_not_max_prob[V_i[s]] = P[V_i[s]] if len(
                not_max_prob_actions_results) == 0 else np.max(
                    not_max_prob_actions_results)

            actions_results = np.array([
                np.sum([
                    u(c) * V[V_i[s_]] * p
                    for s_, p in all_reachable[j].items()
                ]) for j in i_A_max_prob
            ])

            i_a = np.argmax(actions_results)
            V_[V_i[s]] = actions_results[i_a]
            pi[V_i[s]] = A_max_prob[i_a]

        v_norm = np.linalg.norm(V_ - V, np.inf)
        p_norm = np.linalg.norm(P_ - P, np.inf)

        P_diff = P_ - P_not_max_prob
        arg_min_p_diff = np.argmin(P_diff)
        min_p_diff = P_diff[arg_min_p_diff]

        if n_iter and i == n_iter:
            break
        # print("Iteration", i)
        # print(' delta1:', v_norm, p_norm, v_norm + p_norm)
        # print(' delta2:', min_p_diff)
        #print('prob:', P, P_)
        if v_norm + p_norm < epsilon and min_p_diff >= 0:
            break
        V = V_
        P = P_
        i += 1

    #print(f'{i} iterations')
    return V, P, pi, i
