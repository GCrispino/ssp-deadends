import math
import os
import sys
import time

import gym
import matplotlib.pyplot as plt

import numpy as np

import sspde.argparsing as argparsing
import sspde.mdp.general as general
import sspde.mdp.gubs as gubs
import sspde.mdp.run as run
import sspde.plot as plot
import sspde.utils as utils

from datetime import datetime

from matplotlib.backends.backend_pdf import PdfPages

from sspde.mdp.vi import get_succ_states


def save_fig_page(fig, path):
    pp = PdfPages(path)
    fig.savefig(pp, format="pdf")
    pp.close()


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

# compute eGUBS optimal policy
# begin time
start = time.perf_counter()

V_gubs, V_rs_C, P_gubs, pi_gubs, C_maxs = gubs.rs_and_egubs_vi(
    obs, S, A, general_succ_states, V_i, goal, k_g, lamb, args.epsilon,
    general.h_1, mdp_graph)
v_gubs = V_gubs[V_i[obs], 0]
p_gubs = P_gubs[V_i[obs], 0]
# a_opt_gubs = pi_gubs[V_i[obs], 0]
a_opt_gubs = pi_gubs(obs, 0)

stop = time.perf_counter()

# get elapsed
elapsed_gubs = stop - start
time_limit = None if args.limit_time is False else elapsed_gubs
print("Elapsed time to compute optimal policy for eGUBS:", elapsed_gubs)

# Compute MCMP
mcmp_vals, mcmp_p_vals, mincost_maxprob, p_max, mcmp_costs = run.run_mcmp_and_eval_gubs(
    env,
    obs,
    argparsing.DEFAULT_INIT_PARAM_VALUE,
    no_penalty_S,
    A,
    no_penalty_V_i,
    general_succ_states,
    lamb,
    k_g,
    args.epsilon,
    no_penalty_mdp_graph,
    p_maxs=args.pmax_vals,
    time_limit=time_limit,
    batch_size=args.batch_size)

mcmp_vals = np.array(mcmp_vals)
n_mcmp_vals = len(mcmp_vals)
print("mcmp values used:", mcmp_p_vals[-n_mcmp_vals:])

# Compute alphaMCMP
alpha_mcmp_vals, alpha_vals, alpha_mcmp_costs = run.run_alpha_mcmp_and_eval_gubs(
    env,
    obs,
    args.alpha_vals,
    no_penalty_S,
    A,
    no_penalty_V_i,
    general_succ_states,
    lamb,
    k_g,
    args.epsilon,
    no_penalty_mdp_graph,
    time_limit=time_limit,
    batch_size=args.batch_size)

alpha_mcmp_vals = np.array(alpha_mcmp_vals)
n_alpha_mcmp_vals = len(alpha_mcmp_vals)
print("alpha_mcmp values used:", alpha_mcmp_vals[-n_alpha_mcmp_vals:])

# TODO -> o valor do eGUBS pro primeiro valor de desconto dá 0, enquanto que rodando o main.py pro mesmo valor, retorna 0.13101062.
#         quando não usamos as variáveis "no_penalty" aqui, o mesmo valor é retornado, o que talvez indique que o outro script esteja errado porque __talvez__ usa as variáveis de penalidade mesmo quando está no desconto
#         ou o experiment.py ta errado
start = time.perf_counter()
discounted_succ_states = get_succ_states("discounted", A, mdp_graph)

if args.gamma_vals:
    discounted_vals = np.array(args.gamma_vals)
else:
    percentage_log_vals = 0.75
    n_log_vals = math.floor(args.batch_size * percentage_log_vals)
    n_linear_vals = args.batch_size - n_log_vals

    discounted_vals = np.concatenate(
        (np.linspace(argparsing.DEFAULT_INIT_PARAM_VALUE, 0.9,
                     n_linear_vals + 1)[:-1],
         (float(0.9)**np.logspace(0, -10, num=n_log_vals))))

discounted_vals, discounted_param_vals = run.run_vi_and_eval_gubs(
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
    args.epsilon,
    no_penalty_mdp_graph,
    pi_gubs=pi_gubs,
    C_maxs=C_maxs,
    time_limit=time_limit,
    compare_policies=args.compare_policies,
    batch_size=args.batch_size)

n_discounted_vals = len(discounted_vals)
discounted_vals = np.array(discounted_vals)
print("discount factor values used:",
      discounted_param_vals[:n_discounted_vals])
discounted_param_vals = -np.log2(1 - discounted_param_vals)

if args.penalty != None:
    max_penalty = args.penalty
else:
    max_penalty = mincost_maxprob * 5

penalty_succ_states = get_succ_states("penalty", A, mdp_graph)
if args.penalty_vals:
    penalty_vals = np.array(args.penalty_vals)
else:
    penalty_vals = np.linspace(argparsing.DEFAULT_INIT_PARAM_VALUE,
                               max_penalty, args.batch_size)

penalty_vals, penalty_param_vals = run.run_vi_and_eval_gubs(
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
    args.epsilon,
    mdp_graph,
    pi_gubs=pi_gubs,
    C_maxs=C_maxs,
    time_limit=time_limit,
    batch_size=args.batch_size,
)

n_penalty_vals = len(penalty_vals)
penalty_vals = np.array(penalty_vals)
print("penalty values used:", penalty_param_vals[:n_penalty_vals])

fig_criteria, fig_mcmp = plot.plot_data(
    penalty_param_vals,
    penalty_vals,
    discounted_param_vals,
    discounted_vals,
    mcmp_p_vals,
    mcmp_vals,
    mcmp_costs,
    alpha_vals,
    alpha_mcmp_vals,
    alpha_mcmp_costs,
    p_max,
    v_gubs,
)

domain_name = env.domain.domain_name
problem_name = domain_name + str(problem_index)
output_outdir = args.output_dir
output_dir = os.path.join(output_outdir, domain_name, problem_name,
                          f"{str(datetime.now().timestamp())}")
if args.render_and_save:
    # save rendered figure
    plot_file_path = os.path.join(output_dir, "criteria.pdf")
    print(f"writing plot figure to {plot_file_path}")
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
    save_fig_page(fig_criteria, plot_file_path)

    # save output data in JSON format
    output_filename = str(datetime.time(datetime.now())) + '.json'
    output_data = {
        **vars(args),
        'penalty_param_vals': list(penalty_param_vals),
        'penalty_result_vals': list(penalty_vals),
        'discounted_param_vals': list(discounted_param_vals),
        'discounted_result_vals': list(discounted_vals),
        'mcmp_p_vals': list(mcmp_p_vals),
        'mcmp_result_vals': list(mcmp_vals),
        'mcmp_costs': list(mcmp_costs),
        'alpha_vals': list(alpha_vals),
        'alpha_mcmp_result_vals': list(alpha_mcmp_vals),
        'alpha_mcmp_costs': list(alpha_mcmp_costs),
        'p_max': p_max,
        'v_gubs': v_gubs,
    }
    output_file_path = utils.output(output_filename,
                                    output_data,
                                    output_dir=output_dir)
    if output_file_path:
        print("Output JSON data written to ", output_file_path)
if args.plot_stats:
    plt.show()
