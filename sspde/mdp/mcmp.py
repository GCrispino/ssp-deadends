import math

import pulp

import sspde.rendering as rendering
from sspde.mdp.general import get_succs, Pr

def get_variables(env, S, A):
    """
        creates a map of variables to use in the linear programming model
    """
    variables = []
    variable_map = {}
    for i, s in enumerate(S):
        for a in A:
            s_id_ = rendering.get_state_id(env, s)
            s_id = s_id_ if s_id_ != "" else i
            var = pulp.LpVariable(name=f"x_({s_id}-{a})", lowBound=0)
            variables.append(var)
            variable_map[(s, a)] = var

    return variable_map

def get_in_flow(variable_map, mdp):
    ins = {}
    succs_map = {}
    for s in mdp:
        succs = get_succs(s, mdp)
        succs_map[s] = succs
        for succ in succs:
            succs_s = get_succs(succ, mdp)
            if set(succs_s) == set({succ}):
                continue
            #if succ == "sd":
            #    continue
            for a in mdp[s]['actions']:
                variable = variable_map[s, a]
                prob = Pr(s, a, succ, mdp)
                if succ not in ins:
                    ins[succ] = []
                ins[succ].append(variable * prob)
    for s in mdp:
        # if s == "sd":
        #     continue
        if set(get_succs(s, mdp)) == set({s}):
            continue
        ins[s] = pulp.lpSum(ins[s] if s in ins else [0])
    return ins


def get_out_flow(variable_map, mdp):
    outs = {}
    for s in mdp:
        if set(get_succs(s, mdp)) == set({s}):
            continue

        if mdp[s]['goal']:
            continue
        out = []
        for a in mdp[s]['actions']:
            variable = variable_map[s, a]
            out.append(variable)
        outs[s] = pulp.lpSum(out)
    return outs

def get_lp_data(env, S, A, mdp):
    variable_map = get_variables(env, S, A)
    in_flow = get_in_flow(variable_map, mdp)
    out_flow = get_out_flow(variable_map, mdp)

    return variable_map, in_flow, out_flow

def mcmp(s_0,
         S_i,
         variable_map,
         in_flow,
         out_flow,
         p_max,
         cost_fn,
         env,
         mdp,
         log_solver=False,
         start=None):
    # Create the model
    model_cost = pulp.LpProblem(name="mcmp", sense=pulp.LpMinimize)

    # Add the constraints to the models
    for s, s_obj in mdp.items():
        i = S_i[s]
        # s_id = i if S_ids is None else S_ids[s]
        s_id_ = rendering.get_state_id(env, s)
        s_id = s_id_ if s_id_ != "" else i
        if set(get_succs(s, mdp)) == set({s}):
            continue
        if not s_obj["goal"] and s != s_0:
            model_cost += (out_flow[s] - in_flow[s] <= 0, f"flow_{s_id}")

    # constraint for initial state
    i_s_0 = S_i[s_0]
    s_0_id_ = rendering.get_state_id(env, s_0)
    s_0_id = s_0_id_ if s_0_id_ != "" else i_s_0
    model_cost += (out_flow[s_0] - in_flow[s_0] <= 1, f"flow_init_{s_0_id}")

    # get goal state
    # TODO -> multiple goal staes might exist
    gs = [s for s, v in mdp.items() if v['goal']]
    in_flow_gs = pulp.lpSum([in_flow[g] for g in gs])

    # constraint for goal state
    model_cost += (in_flow_gs == p_max, f"flow_goals")

    # Add the objective function to the model
    obj_func = None
    # for s, a in itertools.product(S, A):
    for s, s_obj in mdp.items():
        for a in s_obj['actions']:
            x = variable_map[s, a]
            v = x * cost_fn(s, a)

            if obj_func is None:
                obj_func = v
            else:
                obj_func += v

    #print("objective function:", obj_func)

    model_cost += obj_func

    # Solve the problem
    model_cost.solve(pulp.PULP_CBC_CMD(msg=log_solver))

    return model_cost.objective.value(), model_cost, False

    #print_model_status(model_cost)


def maxprob_lp(s_0, S_i, in_flow, out_flow, env, mdp):
    # Create the model
    model_prob = pulp.LpProblem(name="maxprob", sense=pulp.LpMaximize)

    # Add the constraints to the models
    for s in mdp:
        i = S_i[s]
        # if s == "sd":
        #     continue
        if set(get_succs(s, mdp)) == set({s}):
            continue
        if not mdp[s]["goal"] and s != s_0:
            s_id_ = rendering.get_state_id(env, s)
            s_id = s_id_ if s_id_ != "" else i
            model_prob += (out_flow[s] - in_flow[s] == 0, f"flow_{s_id}")

    # constraint for initial state
    i_s_0 = S_i[s_0]
    #s_0_id = i_s_0 if S_ids is None else S_ids[s_0]
    s_0_id_ = rendering.get_state_id(env, s_0)
    s_0_id = s_0_id_ if s_0_id_ != "" else i_s_0
    model_prob += (out_flow[s_0] - in_flow[s_0] == 1, f"flow_init_{s_0_id}")

    # get goal state
    gs = [s for s, v in mdp.items() if v['goal']]
    in_flow_gs = pulp.lpSum([in_flow[g] for g in gs])

    #obj_func_prob = in_flow[g]
    obj_func_prob = in_flow_gs

    model_prob += obj_func_prob

    model_prob.solve()

    return model_prob.objective.value(), model_prob


def create_pi_func(variable_map, A):

    def pi_func(s):
        best = None
        max_val = -math.inf
        for a in A:
            if (val_a := variable_map[s, a].value()) > max_val:
                max_val = val_a
                best = a

        return best

    return pi_func


def create_pi_func_prob(env, variable_map, in_flow, out_flow, s0, A, p_max):

    def pi_func(s, a):
        if s not in out_flow:
            return 0

        out_val = out_flow[s].value()

        if out_val == 0:
            return 0

        in_val = in_flow[s].value()
        out_diff = out_val - in_val
        if s.literals != s0.literals:
            if out_diff < 0 and math.fabs(out_diff) > 1e-5:
                # TODO - what to do when out(s) - in(s) < 0 ? 
                print(rendering.get_state_id(env, s))
                print("  ", out_val, in_val, out_val - in_val)
                assert False
            return variable_map[s, a].value() / out_val

        # if s == s0, chooses action a with probability (out(s) - in(s)) * (x_s_a / out(s))
        return out_diff * (variable_map[s, a].value() / out_val)

    return pi_func


def print_model_status(model):
    print(model)

    print(f"status: {model.status}, {pulp.LpStatus[model.status]}")

    print(f"objective: {model.objective.value()}")

    for var in model.variables():
        print(f"{var.name}: {var.value()}")

    for name, constraint in model.constraints.items():
        print(f"{name}: {constraint.value()}")
