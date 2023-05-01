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
                         param_vals,
                         S,
                         A,
                         V_i,
                         succ_states,
                         k_g,
                         lamb,
                         epsilon,
                         mdp_graph,
                         pi_gubs=None,
                         C_maxs=None,
                         time_limit=None,
                         compare_policies=False,
                         batch_size=5):
    has_time_limit = bool(time_limit)
    start = time.perf_counter()

    if mode == "discounted":
        param = "gamma"
    elif mode == "penalty":
        param = "penalty"

    kwargs_list = [{"mode": mode, param: val} for val in param_vals]

    reses = []
    for kwargs in kwargs_list:
        elapsed = time.perf_counter() - start
        print(f"  elapsed: {elapsed}, time limit: {time_limit}")
        if has_time_limit and elapsed > time_limit:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit}")
            break
        print(f"running for param val {param}={kwargs[param]}:")

        V, pi, P, timed_out = vi.vi(S,
                                    succ_states,
                                    A,
                                    V_i,
                                    goal,
                                    env,
                                    epsilon,
                                    mdp_graph,
                                    start=start,
                                    time_limit=time_limit,
                                    **kwargs)

        if has_time_limit and timed_out:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit} during value iteration"
            )
            continue

        pi_func = general.create_pi_func(pi, V_i)

        if compare_policies and pi_gubs:
            diffs, reachable_stat, reachable_non_stat = general.compare_policies(
                obs, pi_gubs, pi_func, C_maxs, mdp_graph, env)

            print("policies comparison:")
            if len(diffs) == 0:
                print("  policies are equal!")
            for s_, C in diffs.items():
                print(
                    f"  difference in {rendering.get_state_id(env, s_)}: {C}")
                print(
                    f"    pi_stat({rendering.get_state_id(env, s_)}) = {pi_func(s_)}"
                )
                if C >= 0:
                    print(
                        f"    pi_gubs({rendering.get_state_id(env, s_)}, {C}) = {pi_gubs(s_, C)}"
                    )

        reses.append((V[V_i[obs]], pi_func, P[V_i[obs]]))

        if compare_policies:
            for s__, a in reachable_stat.items():
                print()
                print("state:", rendering.get_state_id(env, s__))
                print("  best action:", pi_func(s__))

            for s__, vals in reachable_non_stat.items():
                print()
                print("state:", rendering.get_state_id(env, s__))
                for C_, a in vals.items():
                    print("  cost:", C_)
                    #print("value:", V[V_i[s]])
                    #print("prob-to-goal:", P[V_i[s]])
                    print("  best action:", pi_gubs(s__, C_))

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
                           p_maxs=None,
                           time_limit=None,
                           batch_size=5):

    has_time_limit = bool(time_limit)
    start = time.perf_counter()

    # Initialize variables
    variable_map, in_flow, out_flow = mcmp.get_lp_data(env, S, A, mdp_graph)

    S_i = {s: i for i, s in enumerate(S)}
    p_max, _ = mcmp.maxprob_lp(obs, S_i, in_flow, out_flow, env, mdp_graph)
    if p_maxs is None:
        # TODO -> put time check here and early return if time is up

        n_vals = batch_size
        ps = np.linspace(init_pval, p_max, n_vals)
    else:
        ps = p_maxs

    mcmp_cost_fn = general.create_cost_fn(mdp_graph, False)

    reses = []
    vals = []
    mcmp_costs = []
    mincost_maxprob = None
    for p in reversed(ps):
        elapsed = time.perf_counter() - start
        print(f"  elapsed: {elapsed}, time limit: {time_limit}")
        if has_time_limit and elapsed > time_limit:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit}")
            break

        print(f"running for param val p_max={p}:")
        mincost, _, timed_out = mcmp.mcmp(obs,
                                          S_i,
                                          variable_map,
                                          in_flow,
                                          out_flow,
                                          p,
                                          mcmp_cost_fn,
                                          env,
                                          mdp_graph,
                                          start=start)

        if mincost_maxprob is None:
            mincost_maxprob = mincost

        var_map = deepcopy(variable_map)
        pi_func = mcmp.create_pi_func_prob(env, var_map, in_flow, out_flow,
                                           obs)

        if has_time_limit and timed_out:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit} during value iteration"
            )
            continue

        reses.append((mincost, pi_func, p))
        mcmp_costs.append(mincost)
        print("Value at initial state:", mincost)
        print("Probability to goal at initial state:", p)

        action_probs = {a: pi_func(obs, a) for a in A}
        print("Action probabilities at initial state:",
              {a: prob
               for a, prob in action_probs.items() if prob > 0})
        print()

        # eval policy under eGUBS
        param_cost_fn = general.create_cost_fn(mdp_graph, False, p)
        v = gubs.eval_policy(obs,
                             succ_states,
                             pi_func,
                             param_cost_fn,
                             p,
                             lamb,
                             k_g,
                             epsilon,
                             mdp_graph,
                             A,
                             env,
                             V_i=V_i,
                             prob_policy=True)

        vals.append(v)
        print(
            f"Evaluated value of the optimal policy at s0 under the eGUBS criterion with param val = {p}:",
            v)
        print()

    reses = reses[::-1]
    vals = vals[::-1]
    mcmp_costs = mcmp_costs[::-1]

    return vals, ps, mincost_maxprob, p_max, mcmp_costs


def run_alpha_mcmp_and_eval_gubs(env,
                                 obs,
                                 alphas,
                                 S,
                                 A,
                                 V_i,
                                 succ_states,
                                 lamb,
                                 k_g,
                                 epsilon,
                                 mdp_graph,
                                 p_max=None,
                                 time_limit=None,
                                 batch_size=5):

    has_time_limit = bool(time_limit)
    start = time.perf_counter()

    # Initialize variables
    variable_map, in_flow, out_flow = mcmp.get_lp_data(env, S, A, mdp_graph)

    S_i = {s: i for i, s in enumerate(S)}
    if p_max is None:
        p_max, model_prob = mcmp.maxprob_lp(obs, S_i, in_flow, out_flow, env,
                                            mdp_graph)

    mcmp_cost_fn = general.create_cost_fn(mdp_graph, False)

    reses = []
    vals = []
    mcmp_costs = []
    for alpha in alphas:
        elapsed = time.perf_counter() - start
        print(f"  elapsed: {elapsed}, time limit: {time_limit}")
        if has_time_limit and elapsed > time_limit:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit}")
            break

        print(f"running for param val alpha={alpha}:")
        mincost, _, timed_out = mcmp.alpha_mcmp(obs,
                                                S_i,
                                                variable_map,
                                                in_flow,
                                                out_flow,
                                                alpha,
                                                p_max,
                                                mcmp_cost_fn,
                                                env,
                                                mdp_graph,
                                                start=start)

        var_map = deepcopy(variable_map)
        pi_func = mcmp.create_pi_func_prob(env, var_map, in_flow, out_flow,
                                           obs)

        if has_time_limit and timed_out:
            print(
                f"  elapsed time of {elapsed} exceeded limit of {time_limit} during value iteration"
            )
            continue

        reses.append((mincost, pi_func, alpha * p_max))
        mcmp_costs.append(mincost)
        print("Value at initial state:", mincost)
        print("Probability to goal at initial state:", alpha * p_max)

        action_probs = {a: pi_func(obs, a) for a in A}
        print("Action probabilities at initial state:",
              {a: prob
               for a, prob in action_probs.items() if prob > 0})
        print()

        actual_pmax = alpha * p_max
        # eval policy under eGUBS
        param_cost_fn = general.create_cost_fn(mdp_graph, False, actual_pmax)
        v = gubs.eval_policy(obs,
                             succ_states,
                             pi_func,
                             param_cost_fn,
                             actual_pmax,
                             lamb,
                             k_g,
                             epsilon,
                             mdp_graph,
                             A,
                             env,
                             V_i=V_i,
                             prob_policy=True)

        vals.append(v)
        print(
            f"Evaluated value of the optimal policy at s0 under the eGUBS criterion with param val = {alpha}:",
            v)
        print()

    return vals, alphas, mcmp_costs


def run_egubs_for_alphas(obs,
                         alphas,
                         S,
                         A,
                         V_i,
                         succ_states,
                         goal,
                         lamb,
                         lamb_vals,
                         epsilon,
                         mdp_graph,
                         time_limit=None):

    has_time_limit = bool(time_limit)
    start = time.perf_counter()


    lamb_vals = lamb_vals if lamb_vals is not None else [lamb]
    # compute eGUBS for all alphas
    vals_by_lamb = {}
    probs_by_lamb = {}
    for lamb in lamb_vals:
        vals = []
        probs = []
        for alpha in alphas:
            elapsed = time.perf_counter() - start
            print(f"  elapsed: {elapsed}, time limit: {time_limit}")
            if has_time_limit and elapsed > time_limit:
                print(
                    f"  elapsed time of {elapsed} exceeded limit of {time_limit}"
                )
                break

            k_g = alpha / (1 - alpha)
            print(f"running for param val alpha={alpha}, k_g={k_g} and lambda={lamb}:")
            V_gubs, _, P_gubs, pi_gubs, _ = gubs.rs_and_egubs_vi(
                obs, S, A, succ_states, V_i, goal, k_g, lamb, epsilon,
                general.h_1, mdp_graph)

            v_gubs = V_gubs[V_i[obs], 0]
            p_gubs = P_gubs[V_i[obs], 0]
            a_opt_gubs = pi_gubs(obs, 0)

            vals.append(v_gubs)
            probs.append(p_gubs)

            print("Value at initial state:", v_gubs)
            print("Probability to goal at initial state:", p_gubs)
            print("Best action at initial state:", a_opt_gubs)
            print()

            # eval policy under eGUBS

            print(
                f"Evaluated value of the optimal policy at s0 under the eGUBS criterion with param val = {alpha}:",
                v_gubs)
            print()

        vals_by_lamb[lamb] = list(vals)
        probs_by_lamb[lamb] = list(probs)

    return vals_by_lamb, alphas, probs_by_lamb
