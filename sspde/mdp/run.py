import math
import time
from copy import deepcopy

import numpy as np

import sspde.argparsing as argparsing
import sspde.experiments.output as output
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
                           S_i,
                           variable_map,
                           in_flow,
                           out_flow,
                           A,
                           V_i,
                           succ_states,
                           lamb,
                           k_g,
                           epsilon,
                           mdp_graph,
                           p_max,
                           p_maxs=None,
                           time_limit=None,
                           batch_size=5):

    has_time_limit = bool(time_limit)
    start = time.perf_counter()

    # Initialize variables
    variable_map, in_flow, out_flow = mcmp.get_lp_data(env, S, A, mdp_graph)

    if p_maxs is None:
        # TODO -> put time check here and early return if time is up

        n_vals = batch_size
        ps = list(np.linspace(init_pval, p_max, n_vals))
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
            print(
                f"running for param val alpha={alpha}, k_g={k_g} and lambda={lamb}:"
            )
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


def run_experiments_eval_gubs(env, obs, S, S_i, no_penalty_S, variable_map, in_flow,
                              out_flow, A, general_succ_states, V_i,
                              no_penalty_V_i, goal, k_g, lamb, epsilon,
                              penalty, penalty_vals, gamma_vals, p_max,
                              pmax_vals, mdp_graph, no_penalty_mdp_graph,
                              time_limit, batch_size, compare_policies):
    # compute eGUBS optimal policy
    # begin time
    start = time.perf_counter()

    V_gubs, V_rs_C, P_gubs, pi_gubs, C_maxs = gubs.rs_and_egubs_vi(
        obs, S, A, general_succ_states, V_i, goal, k_g, lamb, epsilon,
        general.h_1, mdp_graph)
    v_gubs = V_gubs[V_i[obs], 0]
    p_gubs = P_gubs[V_i[obs], 0]
    a_opt_gubs = pi_gubs(obs, 0)

    stop = time.perf_counter()

    # get elapsed
    elapsed_gubs = stop - start
    time_limit = None if time_limit is False else elapsed_gubs
    print("Elapsed time to compute optimal policy for eGUBS:", elapsed_gubs)

    # Compute MCMP
    mcmp_vals, mcmp_p_vals, mincost_maxprob, p_max, mcmp_costs = run_mcmp_and_eval_gubs(
        env,
        obs,
        argparsing.DEFAULT_INIT_PARAM_VALUE,
        no_penalty_S,
        S_i,
        variable_map,
        in_flow,
        out_flow,
        A,
        no_penalty_V_i,
        general_succ_states,
        lamb,
        k_g,
        epsilon,
        no_penalty_mdp_graph,
        p_max,
        p_maxs=pmax_vals,
        time_limit=time_limit,
        batch_size=batch_size)

    #mcmp_vals = np.array(mcmp_vals)
    n_mcmp_vals = len(mcmp_vals)
    print("mcmp values used:", mcmp_p_vals[-n_mcmp_vals:])

    # TODO -> o valor do eGUBS pro primeiro valor de desconto dá 0, enquanto que rodando o main.py pro mesmo valor, retorna 0.13101062.
    #         quando não usamos as variáveis "no_penalty" aqui, o mesmo valor é retornado, o que talvez indique que o outro script esteja errado porque __talvez__ usa as variáveis de penalidade mesmo quando está no desconto
    #         ou o experiment.py ta errado
    start = time.perf_counter()
    discounted_succ_states = vi.get_succ_states("discounted", A, mdp_graph)

    if gamma_vals:
        discounted_vals = np.array(gamma_vals)
    else:
        percentage_log_vals = 0.75
        n_log_vals = math.floor(batch_size * percentage_log_vals)
        n_linear_vals = batch_size - n_log_vals

        discounted_vals = np.concatenate(
            (np.linspace(argparsing.DEFAULT_INIT_PARAM_VALUE, 0.9,
                         n_linear_vals + 1)[:-1],
             (float(0.9)**np.logspace(0, -10, num=n_log_vals))))

    discounted_vals, discounted_param_vals = run_vi_and_eval_gubs(
        env,
        obs,
        goal,
        "discounted",
        discounted_vals,
        no_penalty_S,
        A,
        no_penalty_V_i,
        discounted_succ_states,
        k_g,
        lamb,
        epsilon,
        no_penalty_mdp_graph,
        pi_gubs=pi_gubs,
        C_maxs=C_maxs,
        time_limit=time_limit,
        compare_policies=compare_policies,
        batch_size=batch_size)

    n_discounted_vals = len(discounted_vals)
    #discounted_vals = np.array(discounted_vals)
    print("discount factor values used:",
          discounted_param_vals[:n_discounted_vals])
    discounted_param_vals = list(-np.log2(1 - discounted_param_vals))

    if penalty != None:
        max_penalty = penalty
    else:
        max_penalty = mincost_maxprob * 5

    penalty_succ_states = vi.get_succ_states("penalty", A, mdp_graph)
    if not penalty_vals:
        penalty_vals = np.linspace(argparsing.DEFAULT_INIT_PARAM_VALUE,
                                   max_penalty, batch_size)
    penalty_vals = list(penalty_vals)

    penalty_vals, penalty_param_vals = run_vi_and_eval_gubs(
        env,
        obs,
        goal,
        "penalty",
        penalty_vals,
        S,
        A,
        V_i,
        penalty_succ_states,
        k_g,
        lamb,
        epsilon,
        mdp_graph,
        pi_gubs=pi_gubs,
        C_maxs=C_maxs,
        time_limit=time_limit,
        batch_size=batch_size,
    )

    n_penalty_vals = len(penalty_vals)
    #penalty_vals = np.array(penalty_vals)
    print("penalty values used:", penalty_param_vals[:n_penalty_vals])

    return output.GUBSComparisonExprOutput(penalty_vals, penalty_param_vals,
                                           discounted_vals,
                                           discounted_param_vals, mcmp_vals,
                                           mcmp_p_vals, mcmp_costs, p_max,
                                           v_gubs)


def run_experiments_for_alphas(env, obs, goal, alpha_vals, S, no_penalty_S, A,
                               V_i, no_penalty_V_i, general_succ_states, lamb,
                               lamb_vals, k_g, epsilon, mdp_graph,
                               no_penalty_mdp_graph, batch_size):
    # Compute alphaMCMP
    alpha_mcmp_vals, alpha_vals, alpha_mcmp_costs = run_alpha_mcmp_and_eval_gubs(
        env,
        obs,
        alpha_vals,
        no_penalty_S,
        A,
        no_penalty_V_i,
        general_succ_states,
        lamb,
        k_g,
        epsilon,
        no_penalty_mdp_graph,
        batch_size=batch_size)

    #alpha_mcmp_vals = np.array(alpha_mcmp_vals)
    n_alpha_mcmp_vals = len(alpha_mcmp_vals)
    print("alpha_mcmp values used:", alpha_mcmp_vals[-n_alpha_mcmp_vals:])

    # Compute eGUBS for alphas
    egubs_alpha_result_vals_by_lamb, egubs_alpha_vals, egubs_alpha_probs_by_lamb = run_egubs_for_alphas(
        obs, alpha_vals, S, A, V_i, general_succ_states, goal, lamb, lamb_vals,
        epsilon, mdp_graph)

    return output.AlphaExprOutput(alpha_vals, list(alpha_mcmp_vals),
                                  alpha_mcmp_costs, egubs_alpha_vals,
                                  egubs_alpha_result_vals_by_lamb,
                                  egubs_alpha_probs_by_lamb)
