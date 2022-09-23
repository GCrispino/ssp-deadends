import time

import numpy as np

import sspde.pddl as pddl
import sspde.rendering as rendering


def vi(S,
       succ_states,
       A,
       V_i,
       goal,
       env,
       epsilon,
       mdp_graph,
       mode='discounted',
       gamma=0.99,
       penalty=None,
       start=None,
       time_limit=None):

    V = np.zeros(len(V_i))
    P = np.zeros(len(V_i))
    pi = np.full(len(V_i), None)

    G_i = [V_i[s] for s in V_i if mdp_graph[s]['goal']]
    P[G_i] = 1

    i = 0
    diff = np.inf
    diff_p = np.inf
    while True:
        if time_limit is not None:
            if start is None:
                raise ValueError(start)

            if time.perf_counter() - start > time_limit:
                return V, pi, P, True
        print('Iteration', i, diff + diff_p)
        V_ = np.copy(V)
        P_ = np.copy(P)

        for s in S:
            if mdp_graph[s]['goal']:
                continue
            Q = np.zeros(len(A))
            Q_p = np.zeros(len(A))
            cost = 1
            for i_a, a in enumerate(A):
                succ = succ_states[s, a]

                probs = np.fromiter(iter(succ.values()), dtype=float)
                succ_i = [V_i[succ_s] for succ_s in succ_states[s, a]]

                discount = gamma if mode == 'discounted' else 1
                Q[i_a] = cost + np.dot(probs, discount * V_[succ_i])

                Q_p[i_a] = np.dot(probs, P_[succ_i])

            V[V_i[s]] = np.min(Q)
            if mode == 'penalty':
                if penalty < V[V_i[s]]:
                    V[V_i[s]] = min(penalty, V[V_i[s]])
                    pi[V_i[s]] = pddl.quit_action
                    P[V_i[s]] = 0
                else:
                    i_a_best = np.argmin(Q)
                    pi[V_i[s]] = A[i_a_best]
                    P[V_i[s]] = Q_p[i_a_best]
            else:
                i_a_best = np.argmin(Q)
                pi[V_i[s]] = A[i_a_best]
                P[V_i[s]] = Q_p[i_a_best]

        diff = np.linalg.norm(V_ - V, np.inf)
        diff_p = np.linalg.norm(P_ - P, np.inf)

        #if diff < epsilon:
        if diff + diff_p < epsilon:
            break
        i += 1

    return V, pi, P, False


def get_succ_states(vi_mode, A, mdp_graph):
    succ_states = {s: {} for s in mdp_graph}
    for s in mdp_graph:
        A_set = A
        if vi_mode == "penalty":
            A_set = A_set + [pddl.quit_action]
        for a in A_set:
            if mdp_graph[s]['goal']:
                succ_states[s, a] = {s: 1}
            else:
                succ_states[s, a] = mdp_graph[s]['actions'][a]

    return succ_states
