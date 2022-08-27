import numpy as np

import sspde.rendering as rendering

# TODO -> parece estar errado pro critério de penalidade em valores pequenos
#           - verificar se em outros valores faz sentido também
def rs_lexicographic_eval(succ_states, V_i, pi, cost_fn, lamb, epsilon, mdp_graph, env):

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

    while True:
        V_ = np.copy(V)
        #P_ = np.copy(P)
        for s in not_goal:
            a = pi(s)
            all_reachable = succ_states[s, a]
            #action_result_p = np.sum(
            #    [P[V_i[s_]] * p for s_, p in all_reachable.items()])

            # set maxprob
            # P_[V_i[s]] = action_result_p

            c = cost_fn(s, a)
            action_result = np.sum(
                [np.exp(lamb * c) * V[V_i[s_]] * p for s_, p in all_reachable.items()])

            V_[V_i[s]] = action_result

        v_norm = np.linalg.norm(V_ - V, np.inf)
        #p_norm = np.linalg.norm(P_ - P, np.inf)

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
