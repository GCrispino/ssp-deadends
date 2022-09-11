import math
import os
import sys
import time

import gym
import imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pddlgym
import pulp

import sspde.argparsing as argparsing
import sspde.mdp.general as general
import sspde.mdp.gubs as gubs
import sspde.mdp.run as run
import sspde.mdp.rs as rs
import sspde.pddl as pddl
import sspde.rendering as rendering

from datetime import datetime

from sspde.mdp.vi import get_succ_states, vi

#matplotlib.use('agg')

sys.setrecursionlimit(5000)
np.random.seed(42)
np.set_printoptions(precision=16)

################################################################

################################################################

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

mdp_graph = general.build_mdp_graph(obs, A, env, problem_index, penalty=True)
no_penalty_mdp_graph = general.to_no_penalty_mdp_graph(mdp_graph)
no_penalty_S = list(sorted([s for s in no_penalty_mdp_graph]))
S = list(sorted([s for s in mdp_graph]))
print('Number of states:', len(S))

V_i = {s: i for i, s in enumerate(S)}
no_penalty_V_i = {s: i for i, s in enumerate(no_penalty_mdp_graph)}
G_i = [V_i[s] for s in V_i if mdp_graph[s]['goal']]
no_penalty_G_i = [
    V_i[s] for s in no_penalty_V_i if no_penalty_mdp_graph[s]['goal']
]

lamb = args.lamb
k_g = args.k_g

general_succ_states = get_succ_states("discounted", A, mdp_graph)
penalty_succ_states = get_succ_states("penalty", A, mdp_graph)


def h_1(s):
    return 1


# compute eGUBS optimal policy
# TODO -> use eGUBS-AO*
# begin time
start = time.perf_counter()

V_gubs, V_rs_C, P_gubs, pi_gubs = gubs.rs_and_egubs_vi(obs, S, A,
                                                       general_succ_states,
                                                       V_i, goal, k_g, lamb,
                                                       args.epsilon, h_1,
                                                       mdp_graph)
v_gubs = V_gubs[V_i[obs], 0]
p_gubs = P_gubs[V_i[obs], 0]
a_opt_gubs = pi_gubs[V_i[obs], 0]

stop = time.perf_counter()

# get elapsed
elapsed_gubs = stop - start

# Compute MCMP
# TODO -> get mincost from MCMP and use that as heuristic somehow for choosing a max penalty val
v_gubs_mcmp, mincost_mcmp, p_max = run.run_mcmp_and_eval_gubs(
    env, obs, no_penalty_S, A, no_penalty_V_i, general_succ_states, lamb, k_g,
    args.epsilon, no_penalty_mdp_graph)

# TODO -> o valor do eGUBS pro primeiro valor de desconto dá 0, enquanto que rodando o main.py pro mesmo valor, retorna 0.13101062.
#         quando não usamos as variáveis "no_penalty" aqui, o mesmo valor é retornado, o que talvez indique que o outro script esteja errado porque __talvez__ usa as variáveis de penalidade mesmo quando está no desconto
#         ou o experiment.py ta errado
start = time.perf_counter()
discounted_succ_states = get_succ_states("discounted", A, mdp_graph)
discounted_vals, discounted_param_vals = run.run_vi_and_eval_gubs(
    env,
    obs,
    goal,
    "discounted",
    argparsing.DEFAULT_INIT_PARAM_VALUE,
    args.gamma,
    no_penalty_S,
    A,
    no_penalty_V_i,
    no_penalty_G_i,
    discounted_succ_states,
    k_g,
    lamb,
    args.epsilon,
    no_penalty_mdp_graph,
    elapsed_gubs,
    batch_size=args.batch_size)

discounted_vals = np.array(discounted_vals)
n_discounted_vals = len(discounted_vals)
print("discount factor values used:", discounted_param_vals)
discounted_param_vals = -np.log(1 - discounted_param_vals)

max_penalty = args.penalty
max_penalty = mincost_mcmp * 5
penalty_succ_states = get_succ_states("penalty", A, mdp_graph)
penalty_vals, penalty_param_vals = run.run_vi_and_eval_gubs(
    env,
    obs,
    goal,
    "penalty",
    argparsing.DEFAULT_INIT_PARAM_VALUE,
    max_penalty,
    S,
    A,
    V_i,
    G_i,
    penalty_succ_states,
    k_g,
    lamb,
    args.epsilon,
    mdp_graph,
    elapsed_gubs,
    batch_size=args.batch_size,
)

penalty_vals = np.array(penalty_vals)
n_penalty_vals = len(penalty_vals)
print("penalty values used:", penalty_param_vals)
penalty_param_vals = np.log(penalty_param_vals)

#plt.title("penalty")
#plt.axhline(y=v_gubs, color='r', linestyle='-', label="eGUBS optimal")
#plt.axhline(y=v_gubs_mcmp, color='b', linestyle='-', label="MCMP")
#print("MCMP mincost value:", mincost_mcmp)
#plt.plot(penalty_param_vals[:n_penalty_vals], penalty_vals)
#plt.legend()

#plt.figure()
#plt.title("discounted")
#plt.axhline(y=v_gubs, color='r', linestyle='-', label="eGUBS optimal")
##plt.axhline(y=v_gubs_mcmp, color='b', linestyle='-', label="MCMP")
#plt.plot(discounted_param_vals[:n_discounted_vals], discounted_vals)
#plt.legend()

fig, ax = plt.subplots()
ax.set_title("eGUBS criterion vs. other criteria")
ax.axhline(y=v_gubs, color='r', linestyle='-', label="eGUBS optimal")
plt.axhline(y=v_gubs_mcmp,
            color='b',
            linestyle='-',
            label=r"MCMP $p_{max}=$" + f"{p_max}")
ax.plot(penalty_param_vals[:n_penalty_vals],
        penalty_vals,
        label="penalty",
        marker="x")
ax.set_xlabel(r"$\log(D)$")
ax.set_ylabel(r"Policy evaluation according to eGUBS at $s_0$")

ax2 = ax.twiny()
ax2.plot(discounted_param_vals[:n_discounted_vals],
         discounted_vals,
         label="discounted",
         color="tab:green",
         marker="x")
ax2.set_xlabel(r"$-\log(1 - \gamma)$")

#secax = ax.secondary_xaxis('top', functions=(forward, inverse))
#ax.plot(inverse(discounted_param_vals)[:n_discounted_vals], discounted_vals, label="discounted inverse")
#ax.plot(forward(discounted_param_vals)[:n_discounted_vals], discounted_vals, label="discounted forward")
#secax.set_xlabel("discounted")

fig.legend()

domain_name = env.domain.domain_name
problem_name = domain_name + str(problem_index)
output_outdir = args.output_dir
output_dir = os.path.join(output_outdir, domain_name, problem_name,
                          f"{str(datetime.now().timestamp())}")
if args.render_and_save:
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    plt.savefig(os.path.join(output_dir, "criteria.png"))
if args.plot_stats:
    plt.show()
