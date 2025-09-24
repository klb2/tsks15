import marimo

__generated_with = "0.15.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Receiver Operating Characteristic (ROC)

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebook illustrates the receiver operating characteristic (ROC) concept for a simple Gaussian model.

    In particular, we have a detection problem, in which the data $y$ is distributed according to a normal distribution with different mean values and unit variance:

    \begin{equation*}
    \begin{cases}
    H_0: & \mathcal{N}(\mu_0, 1)\\
    H_1: & \mathcal{N}(\mu_1, 1)
    \end{cases}
    \end{equation*}
    """
    )
    return


@app.cell
def _(
    gamma,
    md_graph,
    mo,
    np,
    pdf1,
    pdf2,
    plt,
    prob_fa,
    prob_tp,
    rv1,
    rv2,
    slider_mean1,
    slider_mean2,
    slider_threshold,
    x,
):
    _fig, _axs = plt.subplots(1, 2)
    _ax_pdf, _ax_roc = _axs
    _ax_pdf.plot(x, pdf1, label="$\\mathcal{N}(\\mu_0, 1)$", c="b")
    _ax_pdf.plot(x, pdf2, label="$\\mathcal{N}(\\mu_1, 1)$", c="orange")
    _ax_pdf.vlines(
        gamma, 0, 1.05 * np.maximum(np.max(pdf1), np.max(pdf2)), ls="--", color="k"
    )
    _idx_gamma = np.argmax(x > gamma)
    _ax_pdf.fill_between(x[_idx_gamma:], pdf1[_idx_gamma:], color="b", alpha=0.25)
    _ax_pdf.fill_between(
        x[:_idx_gamma], pdf2[:_idx_gamma], color="orange", alpha=0.25
    )
    _ax_pdf.set_xlabel("$y$")
    _ax_pdf.set_ylabel("PDF")
    # _ax_pdf.legend()

    _ax_roc.plot(prob_fa, prob_tp)
    _ax_roc.plot([rv1.sf(gamma)], [rv2.sf(gamma)], "o")
    _ax_roc.plot([0, 1], [0, 1], "k--")
    _ax_roc.set_xlabel(r"$P_{\text{FA}}$")
    _ax_roc.set_ylabel(r"$P_{\text{TP}}$")

    _fig.tight_layout()

    mo.hstack(
        [
            mo.vstack(
                [
                    slider_mean1,
                    slider_mean2,
                    slider_threshold,
                    mo.mpl.interactive(_fig),
                ]
            ),
            mo.md(md_graph),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(mo):
    slider_mean1 = mo.ui.slider(-2, -0.5, 0.1, -1, label="Mean $\\mu_0$")
    slider_mean2 = mo.ui.slider(0.5, 2, 0.1, 1, label="Mean $\\mu_1$")
    slider_threshold = mo.ui.slider(-3, 3, 0.1, 0, label="Threshold $\\gamma$")
    return slider_mean1, slider_mean2, slider_threshold


@app.cell
def _(slider_mean1, stats, x):
    mean1 = slider_mean1.value
    rv1 = stats.norm(mean1)
    pdf1 = rv1.pdf(x)
    prob_tp = rv1.cdf(x)
    return pdf1, prob_tp, rv1


@app.cell
def _(slider_mean2, stats, x):
    mean2 = slider_mean2.value
    rv2 = stats.norm(mean2)
    pdf2 = rv2.pdf(x)
    prob_fa = rv2.cdf(x)
    return pdf2, prob_fa, rv2


@app.cell
def _(slider_threshold):
    gamma = slider_threshold.value
    return (gamma,)


@app.cell
def _(np):
    x = np.linspace(-5, 5, 500)
    return (x,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


@app.cell
def _():
    md_graph = r"""
    ## PDFs

    The first graph shows the PDFs under the two hypotheses $H_0$ and $H_1$. Their mean values $\mu_0$ and $\mu_1$ can be adjusted via the sliders above the figure.
    The black dashed line indicates the threshold $\gamma$, which can also be changed using the corresponding slider.

    For measurement values $y$ that fall below the threshold, we decide for $H_0$, and therefore, we get a false alarm (FA) if $y>\gamma$ while $H_0$ is actually true.
    The probability of this happening is $P_{\text{FA}} = \Pr(y>\gamma; H_0)$, which corresponds to the blue area.

    Similarly, if $H_1$ is true but $y<\gamma$, we wrongly decide for $H_0$ and missing a detection. This probability is indicated by the orange area.


    ## ROC Curve

    The second graph shows the tradeoff between false alarm probability and detection probability, where each point on the curve corresponds to a value of $\gamma$. The orange marker indicates the operating point corresponding to the current value of $\gamma$ set by the slider.
    """
    return (md_graph,)


if __name__ == "__main__":
    app.run()
