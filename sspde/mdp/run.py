import math
import time
from copy import deepcopy

import numpy as np
import pulp

import sspde.mdp.general as general
import sspde.mdp.gubs as gubs
import sspde.mdp.mcmp as mcmp
import sspde.mdp.vi as vi
import sspde.rendering as rendering


def eval_gubs(env,
              obs,
              succ_states,
              V_i,
              A,
              mode,
              lamb,
              k_g,
              epsilon,
              param_vals,
              reses,
              mdp_graph,
              prob_policy=False):
    vals = []

    for i, (V, pi_func, p_max) in enumerate(reses):
        param_cost_fn = general.create_cost_fn(mdp_graph, mode == "penalty",
                                               param_vals[i])
        v = gubs.eval_policy(obs,
                             succ_states,
                             pi_func,
                             param_cost_fn,
                             p_max,
                             lamb,
                             k_g,
                             epsilon,
                             mdp_graph,
                             A,
                             env,
                             V_i=V_i,
                             prob_policy=prob_policy)
        vals.append(v)
        print(
            f"Evaluated value of the optimal policy at s0 under the eGUBS criterion with param val = {param_vals[i]}:",
            v)
        print()

    return vals


def run_vi_and_eval_gubs(env,
                         obs,
                         goal,
                         mode,
                         init_val,
                         val,
                         S,
                         A,
                         V_i,
                         G_i,
                         succ_states,
                         k_g,
                         lamb,
                         epsilon,
                         mdp_graph,
                         time_limit,
                         batch_size=5):
    start = time.perf_counter()

    if mode == "discounted":
        param = "gamma"
    elif mode == "penalty":
        param = "penalty"

    # param_vals = [getattr(args, param)]
    param_vals = [val]

    n_vals = batch_size
    if mode == "discounted":
        # half = n_vals / 2
        # n_linear_vals = math.floor(half)
        # n_log_vals = math.ceil(half)
        percentage_log_vals = 0.75
        n_log_vals = math.floor(n_vals * percentage_log_vals)
        n_linear_vals = n_vals - n_log_vals

        param_vals = np.concatenate(
            (np.linspace(init_val, 0.9, n_linear_vals + 1)[:-1],
             (float(0.9)**np.logspace(0, -10, num=n_log_vals))))

    elif mode == "penalty":
        param_vals = np.linspace(init_val, val, n_vals)

    kwargs_list = [{"mode": mode, param: val} for val in param_vals]

    reses = []
    for kwargs in kwargs_list:
        elapsed = time.perf_counter() - start
        print(f"  elapsed: {elapsed}, time limit: {time_limit}")
        if elapsed > time_limit:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit}")
            break
        print(f"running for param val {param}={kwargs[param]}:")

        V, pi, P, timed_out = vi.vi(S,
                                    succ_states,
                                    A,
                                    V_i,
                                    G_i,
                                    goal,
                                    env,
                                    epsilon,
                                    mdp_graph,
                                    start=start,
                                    time_limit=time_limit,
                                    **kwargs)

        if timed_out:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit} during value iteration"
            )
            continue

        pi_func = general.create_pi_func(pi, V_i)

        reses.append((V[V_i[obs]], pi_func, P[V_i[obs]]))
        print("Value at initial state:", V[V_i[obs]])
        print("Probability to goal at initial state:", P[V_i[obs]])
        print("Best action at initial state:", pi[V_i[obs]])
        print()

    vals = eval_gubs(env, obs, succ_states, V_i, A, mode, lamb, k_g, epsilon,
                     param_vals, reses, mdp_graph)

    return vals, param_vals


def run_mcmp_and_eval_gubs(env,
                           obs,
                           init_pval,
                           S,
                           A,
                           V_i,
                           succ_states,
                           lamb,
                           k_g,
                           epsilon,
                           mdp_graph,
                           time_limit,
                           batch_size=5):

    start = time.perf_counter()

    # Initialize variables
    variables = []
    variable_map = {}
    for i, s in enumerate(S):
        for a in A:
            s_id_ = rendering.get_state_id(env, s)
            s_id = s_id_ if s_id_ != "" else i
            var = pulp.LpVariable(name=f"x_({s_id}-{a})", lowBound=0)
            variables.append(var)
            variable_map[(s, a)] = var
    in_flow = mcmp.get_in_flow(variable_map, mdp_graph)
    out_flow = mcmp.get_out_flow(variable_map, mdp_graph)

    S_i = {s: i for i, s in enumerate(S)}
    p_max, model_prob = mcmp.maxprob_lp(obs, S_i, in_flow, out_flow, env,
                                        mdp_graph)

    # TODO -> put time check here and early return if time is up

    n_vals = batch_size
    ps = np.linspace(init_pval, p_max, n_vals)

    mcmp_cost_fn = general.create_cost_fn(mdp_graph, False)

    last_mcmp_cost = None
    reses = []
    for p in ps:
        elapsed = time.perf_counter() - start
        print(f"  elapsed: {elapsed}, time limit: {time_limit}")
        if elapsed > time_limit:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit}")
            break

        print(f"running for param val p_max={p_max}:")
        mincost, model_cost, timed_out = mcmp.mcmp(obs,
                                                   S_i,
                                                   variable_map,
                                                   in_flow,
                                                   out_flow,
                                                   p,
                                                   mcmp_cost_fn,
                                                   env,
                                                   mdp_graph,
                                                   start=start,
                                                   log_solver=False)

        last_mcmp_cost = mincost
        var_map = deepcopy(variable_map)
        pi_func = mcmp.create_pi_func_prob(var_map, obs, A, p)

        if timed_out:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit} during value iteration"
            )
            continue

        reses.append((mincost, pi_func, p))
        print("Value at initial state:", mincost)
        print("Probability to goal at initial state:", p)
        print("Action probabilities initial state:",
              {a: pi_func(obs, a)
               for a in A})
        print()

    vals = eval_gubs(env,
                     obs,
                     succ_states,
                     V_i,
                     A,
                     "mcmp",
                     lamb,
                     k_g,
                     epsilon,
                     ps,
                     reses,
                     mdp_graph,
                     prob_policy=True)

    return vals, ps, last_mcmp_cost
