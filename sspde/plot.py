import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages


def save_fig_page(fig, path):
    pp = PdfPages(path)
    fig.savefig(pp, format="pdf")
    pp.close()


def plot_data(
    gubs_comparison_expr_vals,
    alpha_expr_vals,
    p_max,
    log_alpha=False,
    output_file_path=None,
):

    fig, fig2 = None, None
    if gubs_comparison_expr_vals:
        penalty_vals = gubs_comparison_expr_vals.penalty_result_vals
        penalty_param_vals = gubs_comparison_expr_vals.penalty_param_vals
        discounted_vals = gubs_comparison_expr_vals.discounted_result_vals
        discounted_param_vals = gubs_comparison_expr_vals.discounted_param_vals
        mcmp_vals = gubs_comparison_expr_vals.mcmp_result_vals
        mcmp_p_vals = gubs_comparison_expr_vals.mcmp_p_vals
        p_max = gubs_comparison_expr_vals.p_max
        v_gubs = gubs_comparison_expr_vals.v_gubs

        n_penalty_vals = len(penalty_vals)
        n_discounted_vals = len(discounted_vals)
        n_mcmp_vals = len(mcmp_vals)

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

        # set axis colors
        ax.xaxis.label.set_color(pl_penalty.get_color())
        ax2.xaxis.label.set_color(pl_discounted.get_color())
        ax3.xaxis.label.set_color(pl_mcmp.get_color())

        ax.tick_params(axis='x', colors=pl_penalty.get_color())
        ax2.tick_params(axis='x', colors=pl_discounted.get_color())
        ax3.tick_params(axis='x', colors=pl_mcmp.get_color())

        fig.legend()
        plt.subplots_adjust(top=0.75)

    if alpha_expr_vals:
        alpha_vals = alpha_expr_vals.alpha_vals
        egubs_alpha_vals = alpha_expr_vals.egubs_alpha_vals
        egubs_alpha_result_probs_by_lamb = alpha_expr_vals.egubs_alpha_result_probs_by_lamb

        fig2, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.set_title(r"$\alpha$-MCMP and eGUBS")

        probs = np.array(alpha_vals) * p_max
        x_label = r"$\alpha$"
        if log_alpha:
            x_label = r"$\log_{10}\alpha$"
            alpha_vals = np.log10(alpha_vals)
            egubs_alpha_vals = np.log10(egubs_alpha_vals)
        ax.plot(alpha_vals, probs, label=r"$\alpha$-MCMP", marker="o")

        for lamb, egubs_alpha_result_probs in egubs_alpha_result_probs_by_lamb.items(
        ):
            label = r"eGUBS - $\lambda =" + str(lamb) + "$"
            ax.plot(egubs_alpha_vals,
                    egubs_alpha_result_probs,
                    label=label,
                    marker="x")
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$P^{\pi}_G(s_0)$")
        ax.legend(bbox_to_anchor=(1.0, 0.7), loc='upper left')
        fig2.tight_layout()
        if output_file_path:
            save_fig_page(fig2, output_file_path)

    return fig, fig2
