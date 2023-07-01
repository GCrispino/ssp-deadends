import os
import sys

import gym
import matplotlib.pyplot as plt

import numpy as np

import sspde.argparsing as argparsing
import sspde.mdp.general as general
import sspde.mdp.mcmp as mcmp
import sspde.mdp.run as run
import sspde.plot as plot
import sspde.utils as utils

from datetime import datetime

from sspde.mdp.vi import get_succ_states

#matplotlib.use('agg')

sys.setrecursionlimit(5000)
np.random.seed(42)
np.set_printoptions(precision=16)

################################################################

################################################################

# Setup
args = argparsing.parse_args()

run_alpha_expr = not args.no_run_alpha_expr
run_gubs_comparison_expr = not args.no_run_gubs_comparison_expr
print(run_alpha_expr, run_gubs_comparison_expr)

if not run_alpha_expr and not run_gubs_comparison_expr:
    raise ValueError(
        "--not_run_alpha_experiments and --not_run_gubs_comparison_experiments can't both be set"
    )

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


variable_map, in_flow, out_flow = mcmp.get_lp_data(env, no_penalty_S, A, no_penalty_mdp_graph)
S_i = {s: i for i, s in enumerate(no_penalty_S)}
p_max, _ = mcmp.maxprob_lp(obs, S_i, in_flow, out_flow, env, no_penalty_mdp_graph)


gubs_comparison_expr_vals = None
alpha_expr_vals = None
# Experiments for different criteria evaluated under eGUBS
if run_gubs_comparison_expr:
    gubs_comparison_expr_vals = run.run_experiments_eval_gubs(
        env, obs, S, S_i, no_penalty_S, variable_map, in_flow, out_flow, A, general_succ_states, V_i, no_penalty_V_i,
        goal, k_g, lamb, args.epsilon, args.penalty, args.penalty_vals,
        args.gamma_vals, p_max, args.pmax_vals, mdp_graph, no_penalty_mdp_graph,
        args.limit_time, args.batch_size, args.compare_policies)

# Experiments based on alpha values:
if run_alpha_expr:
    alpha_expr_vals = run.run_experiments_for_alphas(
        env, obs, goal, args.alpha_vals, S, no_penalty_S, A, V_i,
        no_penalty_V_i, general_succ_states, lamb, args.lamb_vals, k_g,
        args.epsilon, mdp_graph, no_penalty_mdp_graph, args.batch_size)

# Create charts
fig_criteria, fig_mcmp = plot.plot_data(
    gubs_comparison_expr_vals,
    alpha_expr_vals,
    p_max,
)

if args.plot_stats:
    plt.show()

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
    if fig_criteria is not None:
        plot.save_fig_page(fig_criteria, plot_file_path)

    # save output data in JSON format
    output_filename = str(datetime.time(datetime.now())) + '.json'
    output_data = {
        **vars(args),
        "p_max": p_max,
        "alpha_expr_vals":
        alpha_expr_vals.to_json() if alpha_expr_vals is not None else None,
        "gubs_comparison_expr_vals":
        gubs_comparison_expr_vals.to_json()
        if gubs_comparison_expr_vals is not None else None,
    }
    output_file_path = utils.output(output_filename,
                                    output_data,
                                    output_dir=output_dir)
    if output_file_path:
        print("Output JSON data written to ", output_file_path)
