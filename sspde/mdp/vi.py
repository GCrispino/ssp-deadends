import numpy as np

import sspde.pddl as pddl
import sspde.rendering as rendering

def vi(S,
       succ_states,
       A,
       V_i,
       G_i,
       goal,
       env,
       epsilon,
       mdp_graph,
       mode='discounted',
       gamma=0.99,
       penalty=None):

    V = np.zeros(len(V_i))
    P = np.zeros(len(V_i))
    pi = np.full(len(V_i), None)
    # print(len(S), len(V_i), len(G_i), len(P))
    # print(G_i)
    P[G_i] = 1

    i = 0
    diff = np.inf
    while True:
        print('Iteration', i, diff)
        V_ = np.copy(V)
        P_ = np.copy(P)

        for s in S:
            # if check_goal(s, goal):
            #     continue
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
                    # P[V_i[s]] = np.max(Q_p)
                    P[V_i[s]] = Q_p[i_a_best]
            else:
                i_a_best = np.argmin(Q)
                pi[V_i[s]] = A[i_a_best]
                # P[V_i[s]] = np.max(Q_p)
                P[V_i[s]] = Q_p[i_a_best]

        diff = np.linalg.norm(V_ - V, np.inf)
        diff_p = np.linalg.norm(P_ - P, np.inf)
        #if diff < epsilon:
        if diff + diff_p < epsilon:
            break
        i += 1
    return V, pi, P
