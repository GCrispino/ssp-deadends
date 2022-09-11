import math
import os
import sys

import gym
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pddlgym
import pulp

import sspde.argparsing as argparsing
import sspde.mdp.gubs as gubs
import sspde.mdp.mcmp as mcmp
import sspde.pddl as pddl
import sspde.rendering as rendering

from datetime import datetime

from sspde.mdp.general import build_mdp_graph, create_cost_fn, create_pi_func
from sspde.mdp.vi import vi

#matplotlib.use('agg')

sys.setrecursionlimit(5000)
np.random.seed(42)


def run_episode(pi,
                env,
                n_steps=100,
                render_and_save=False,
                output_dir=".",
                print_history=False):
    obs, _ = env.reset()

    # Create folder to save images
    if render_and_save:
        output_outdir = args.output_dir
        domain_name = env.domain.domain_name
        problem_name = domain_name + str(problem_index)
        output_dir = os.path.join(output_outdir, domain_name, problem_name,
                                  f"{str(datetime.now().timestamp())}")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)

    if render_and_save:
        img = env.render()
        imageio.imsave(os.path.join(output_dir, "frame1.png"), img)

    cum_reward = 0
    for i in range(1, n_steps + 1):
        old_obs = obs
        obs, reward, done, _ = env.step(pi(obs))
        cum_reward += reward
        if print_history:
            print(pi(old_obs), reward)

        if render_and_save:
            img = env.render()
            imageio.imsave(os.path.join(output_dir, f"frame{i + 1}.png"), img)

        if done:
            break
    return i, cum_reward


args = argparsing.parse_args()

env = gym.make(args.env)
problem_index = args.problem_index
env.fix_problem_index(problem_index)
problem = env.problems[problem_index]
goal = problem.goal
prob_objects = frozenset(problem.objects)

obs, _ = env.reset()
A = list(sorted(env.action_space.all_ground_literals(obs, valid_only=False)))

print(' calculating list of states...')
has_penalty = args.vi_mode == "penalty"
mdp_graph = build_mdp_graph(obs, A, env, problem_index, penalty=has_penalty)
S = list(sorted([s for s in mdp_graph]))
print('Number of states:', len(S))

if args.vi_mode == "discounted":
    param = "gamma"
elif args.vi_mode == "penalty":
    param = "penalty"
param_vals = [getattr(args, param)]

succ_states = {s: {} for s in mdp_graph}
for s in mdp_graph:
    if mdp_graph[s]['goal']:
        continue
    A_set = A
    if args.vi_mode == "penalty":
        A_set = A_set + [pddl.quit_action]
    for a in A_set:
        succ_states[s, a] = mdp_graph[s]['actions'][a]

if args.algorithm == "vi":

    V_i = {s: i for i, s in enumerate(S)}
    G_i = [V_i[s] for s in V_i if mdp_graph[s]['goal']]

    print('obtaining optimal policy')

    n_vals = 20
    if args.batch:
        init_val = args.init_param_val
        n_vals = args.batch_size
        if args.vi_mode == "discounted":
            param_vals = np.linspace(init_val, args.gamma, n_vals)
        elif args.vi_mode == "penalty":
            param_vals = np.linspace(init_val, args.penalty, n_vals)

    kwargs_default = {
        "mode": args.vi_mode,
        "gamma": args.gamma,
        "penalty": args.penalty,
    }
    if args.batch:
        kwargs_list = [{
            **kwargs_default,
            **{
                param: val
            }
        } for val in param_vals]
    else:
        kwargs_list = [kwargs_default]
    print(kwargs_list)
    reses = []
    for kwargs in kwargs_list:
        print(f"running for param val {param}={kwargs[param]}:")
        V, pi, P = vi(S, succ_states, A, V_i, G_i, goal, env, args.epsilon,
                      mdp_graph, **kwargs)

        pi_func = create_pi_func(pi, V_i)

        reses.append((V, pi_func, P))
        print("Value at initial state:", V[V_i[obs]])
        print("Probability to goal at initial state:", P[V_i[obs]])
        print("Best action at initial state:", pi[V_i[obs]])

elif args.algorithm == "mcmp":

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
    #p_max *= 0.9
    mcmp_cost_fn = create_cost_fn(mdp_graph, False)
    mincost, model_cost = mcmp.mcmp(obs, S_i, variable_map, in_flow, out_flow,
                               p_max, mcmp_cost_fn, env, mdp_graph)

    # p_vals = np.linspace(args.init_param_val, p_max, args.batch_size)
    # reses = []
    # for p in p_vals:
    #     print(f"running for param val p_max={p}:")
    #     mincost, model_cost = mcmp.mcmp(obs,
    #                                     S_i,
    #                                     variable_map,
    #                                     in_flow,
    #                                     out_flow,
    #                                     p,
    #                                     mcmp_cost_fn,
    #                                     env,
    #                                     mdp_graph,
    #                                     log_solver=False)

    #     pi_func = mcmp.create_pi_func(variable_map, A)

    #     mcmp.print_model_status(model_cost)
    #     reses.append((mincost, pi_func, p))
    #     print("Value at initial state:", mincost)
    #     print("Probability to goal at initial state:", p)
    #     print("Best action at initial state:", pi_func(obs))
    #     print()
    #print("Value at initial state:", mincost)
    #print("Probability to goal at initial state:", p_max)

    print("s0:", rendering.get_state_id(env, obs))

    #print("s0:", obs)


    def pi_func(s):
        best = None
        max_val = -math.inf
        for a in A:
            if (val_a := variable_map[s, a].value()) > max_val:
                max_val = val_a
                best = a

        return best


lamb = args.lamb
k_g = args.k_g


def u(c):
    return np.exp(c * lamb)


vals = []
stds = []
reses_pi = [pi_func for (_, pi_func, P) in reses]

for i, (V, pi_func, P) in enumerate(reses):
    param_cost_fn = create_cost_fn(mdp_graph, has_penalty, param_vals[i])
    v = gubs.eval_policy(obs,
                         succ_states,
                         pi_func,
                         param_cost_fn,
                         P,
                         lamb,
                         k_g,
                         args.epsilon,
                         mdp_graph,
                         env,
                         V_i=V_i)
    vals.append(v)
    print(
        f"Evaluated value of the optimal policy at s0 under the eGUBS criterion with param val = {param_vals[i]}:",
        v)
    print()

means = np.array(vals)
stds = np.array(stds)

plt.plot(param_vals, means)
if len(stds) > 0:
    plt.fill_between(param_vals, means - stds, means + stds, alpha=0.5)
plt.show()

n_episodes = 1000

print("Optimal action at initial state:", pi_func(obs))

plot = False
if args.plot_stats:
    print('running episodes with optimal policy')
    steps1 = []
    rewards1 = []
    for i in range(n_episodes):
        n_steps, reward = run_episode(pi_func, env)
        steps1.append(n_steps)
        rewards1.append(reward)

    print('running episodes with random policy')
    steps2 = []
    rewards2 = []
    for i in range(n_episodes):
        n_steps, reward = run_episode(lambda s: env.action_space.sample(s),
                                      env)
        steps2.append(n_steps)
        rewards2.append(reward)
    rewards2 = np.array(rewards2)

    plt.title('Reward')
    plt.plot(range(len(rewards1)), np.cumsum(rewards1), label="optimal")
    plt.plot(range(len(rewards1)), np.cumsum(rewards2), label="random")
    plt.legend()
    plt.figure()

    plt.title('steps')
    plt.plot(range(len(steps1)), np.cumsum(steps1), label="optimal")
    plt.plot(range(len(steps1)), np.cumsum(steps2), label="random")
    plt.legend()
    plt.show()

if args.simulate:
    _, goal = run_episode(pi_func,
                          env,
                          n_steps=50,
                          render_and_save=args.render_and_save,
                          output_dir=args.output_dir,
                          print_history=args.print_sim_history)
