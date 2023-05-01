import matplotlib.pyplot as plt
import numpy as np


def plot_data(
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
    egubs_alpha_vals,
    egubs_alpha_result_vals_by_lamb,
    egubs_alpha_result_probs_by_lamb,
    p_max,
    v_gubs,
):
    n_penalty_vals = len(penalty_vals)
    n_discounted_vals = len(discounted_vals)
    n_mcmp_vals = len(mcmp_vals)
    n_alpha_mcmp_vals = len(alpha_mcmp_vals)

    fig, ax = plt.subplots()
    ax.set_title("eGUBS criterion vs. other criteria")
    ax.axhline(y=v_gubs, color='r', linestyle='-', label="eGUBS optimal")
    pl_penalty, = ax.plot(penalty_param_vals[:n_penalty_vals],
                          penalty_vals,
                          label="penalty",
                          marker="^")
    ax.set_xlabel(r"$D$")
    ax.set_ylabel(r"Policy evaluation according to eGUBS at $s_0$")

    ax2 = ax.twiny()
    pl_discounted, = ax2.plot(discounted_param_vals[:n_discounted_vals],
                              discounted_vals,
                              label="discounted",
                              color="tab:green",
                              marker="P")
    ax2.set_xlabel(r"$-\log_2(1 - \gamma)$")

    ax3 = ax.twiny()
    ax3.spines['top'].set_position(("axes", 1.15))
    pl_mcmp, = ax3.plot(mcmp_p_vals[-n_mcmp_vals:],
                        mcmp_vals,
                        color="tab:orange",
                        label="MCMP",
                        marker="X")
    ax3.set_xlabel(r"$p_{max}$")
    ax3.set_ylabel(r"Policy evaluation according to eGUBS at $s_0$")

    ax4 = ax.twiny()
    ax4.spines['top'].set_position(("axes", 1.30))
    pl_alpha_mcmp, = ax4.plot(alpha_vals[-n_alpha_mcmp_vals:],
                              alpha_mcmp_vals,
                              color="tab:brown",
                              label=r"$\alpha$-MCMP",
                              marker="o")
    ax4.set_xlabel(r"$\alpha$")
    ax4.set_ylabel(r"Policy evaluation according to eGUBS at $s_0$")

    # set axis colors
    ax.xaxis.label.set_color(pl_penalty.get_color())
    ax2.xaxis.label.set_color(pl_discounted.get_color())
    ax3.xaxis.label.set_color(pl_mcmp.get_color())

    ax.tick_params(axis='x', colors=pl_penalty.get_color())
    ax2.tick_params(axis='x', colors=pl_discounted.get_color())
    ax3.tick_params(axis='x', colors=pl_mcmp.get_color())
    ax4.tick_params(axis='x', colors=pl_alpha_mcmp.get_color())

    fig.legend()
    plt.subplots_adjust(top=0.75)

    fig2, ax = plt.subplots()
    ax.set_title(r"$\alpha$-MCMP and eGUBS")

    ax.plot(alpha_vals,
            np.array(alpha_vals) * p_max,
            label=r"$\alpha$-MCMP",
            marker="o")

    for lamb, egubs_alpha_result_probs in egubs_alpha_result_probs_by_lamb.items():
        label = r"eGUBS - $\lambda =" + str(lamb) + "$"
        ax.plot(egubs_alpha_vals, egubs_alpha_result_probs, label=label, marker="x")
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel(r"$P^{\pi}_G(s_0)$")
    fig2.legend()

    return fig, fig2
