import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Bayes Risk and Probability of Error

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebooks provides an interactive visualization of the Bayesian approach to hypothesis testing/detection problems.
    It follows the examples given in (S. Kay, Detection Theory, Sections 3.6-3.7).
    In particular, we have the following distributions conditioned on hypotheses $\mathcal{H}_0$ and $\mathcal{H}_1$:
    $$\mathcal{H}_0: \mathcal{N}(0, 1)\\\mathcal{H}_1: \mathcal{N}(A, 1)$$
    with prior probabilities $\Pr(\mathcal{H}_0)$ and $\Pr(\mathcal{H}_1) = 1-\Pr(\mathcal{H}_0)$, respectively.

    The optimal threshold (in terms of the likelihood ratio) is given as
    $$\gamma = \frac{\Pr(\mathcal{H}_0)}{\Pr(\mathcal{H}_1)},$$
    which translates to the optimal threshold in terms of the sample value $x$ as
    $$t = \frac{A}{2} + \frac{\ln{\gamma}}{A},$$
    and we decide for $\mathcal{H}_1$ if $x>t$.
    """
    )
    return


@app.cell
def _(
    md_summary_parameters,
    mo,
    np,
    opt_x_threshold,
    pdf0,
    pdf1,
    plt,
    slider_mean_h1,
    slider_prior,
    slider_threshold,
    x,
    x_threshold,
):
    _fig, _ax_pdf = plt.subplots()
    _ax_pdf.plot(x, pdf0, label="PDF under $\\mathcal{H}_0$", c="b")
    _ax_pdf.plot(x, pdf1, label="PDF under $\\mathcal{H}_1$", c="orange")
    _ax_pdf.vlines(
        opt_x_threshold,
        0,
        1.05 * np.maximum(np.max(pdf0), np.max(pdf1)),
        ls="--",
        color="k",
        label="Optimal Threshold",
    )
    _ax_pdf.vlines(
        x_threshold,
        0,
        1.05 * np.maximum(np.max(pdf0), np.max(pdf1)),
        ls="-.",
        color="gray",
        label="Selected Threshold",
    )

    # _idx_gamma = np.argmax(x > gamma)
    # _ax_pdf.fill_between(x[_idx_gamma:], pdf1[_idx_gamma:], color="b", alpha=0.25)
    # _ax_pdf.fill_between(
    #    x[:_idx_gamma], pdf2[:_idx_gamma], color="orange", alpha=0.25
    # )

    _ax_pdf.legend()
    mo.hstack(
        [
            mo.mpl.interactive(_fig),
            mo.vstack(
                [
                    slider_prior,
                    slider_mean_h1,
                    slider_threshold,
                    mo.md(md_summary_parameters),
                ]
            ),
        ],
        widths=[1.75, 1],
    )
    # plt.show()
    return


@app.cell
def _(errors, mo):
    mo.md(
        f"""
    ## Error Probabilities

    With the selected parameters, you obtain the following total error probabilities $P_{{\\textnormal{{err}}}}$ with
    $$P_{{\\textnormal{{err}}}} = \\Pr(\\mathcal{{H}}_1 \\mid \\mathcal{{H}}_0) \\Pr(\\mathcal{{H}}_0) + \\Pr(\\mathcal{{H}}_0 \\mid \\mathcal{{H}}_1) \\Pr(\\mathcal{{H}}_1).$$
    This value corresponds to the Bayes risk for the weights $C_{{00}}=C_{{11}}=0$ and $C_{{01}} = C_{{10}} = 1$.

    | Threshold | $P_{{\\textnormal{{err}}}}$ |
    |:---|---:|
    | Optimal threshold | $P_{{\\textnormal{{err}}}}={errors["opt"]:.3f}$ |
    | Selected threshold | $P_{{\\textnormal{{err}}}}={errors["selected"]:.3f}$ |
    """
    )
    return


@app.cell
def _(mean_h1, opt_x_threshold, prior_h0, x_threshold):
    md_summary_parameters = f"""
    ## Summary of Parameters

    | Parameter | Value |
    |:---|---:|
    | Prior probability of $\\mathcal{{H}}_0$ | $\\Pr(\\mathcal{{H}}_0)={prior_h0:.3f}$|
    | Prior probability of $\\mathcal{{H}}_1$ | $\\Pr(\\mathcal{{H}}_1)=1-\\Pr(\\mathcal{{H}}_0)={1 - prior_h0:.3f}$|
    | Mean under $\\mathcal{{H}}_1$ | $A={mean_h1:.3f}$|
    | Optimal threshold | $x_{{\\textnormal{{opt}}}}={opt_x_threshold:.3f}$|
    | Selected threshold | $x_{{\\textnormal{{sel}}}}={x_threshold:.3f}$|
    """
    return (md_summary_parameters,)


@app.cell
def _(
    opt_x_threshold,
    prior_h0,
    prior_h1,
    rv_cond_h0,
    rv_cond_h1,
    x_threshold,
):
    thresholds = {
        "opt": opt_x_threshold,
        "selected": x_threshold,
    }
    errors = {}
    for name, threshold in thresholds.items():
        error_1 = rv_cond_h0.sf(threshold) * prior_h0
        error_2 = rv_cond_h1.cdf(threshold) * prior_h1
        errors[name] = error_1 + error_2
    return (errors,)


@app.cell
def _(np):
    x = np.linspace(-5, 5, 250)
    return (x,)


@app.cell
def _(mo):
    slider_prior = mo.ui.slider(
        0, 0.95, 0.05, 0.5, label="Prior probability $\\Pr(\\mathcal{H}_0)$"
    )
    slider_threshold = mo.ui.slider(-5, 5, 0.1, 0.5, label="Decision Threshold")
    slider_mean_h1 = mo.ui.slider(
        0.5, 3, 0.2, 1, label="Mean under $\\mathcal{H}_1$"
    )
    return slider_mean_h1, slider_prior, slider_threshold


@app.cell
def _(slider_mean_h1, slider_prior, slider_threshold):
    prior_h0 = slider_prior.value
    x_threshold = slider_threshold.value
    mean_h1 = slider_mean_h1.value
    return mean_h1, prior_h0, x_threshold


@app.cell
def _(mean_h1, np, prior_h0):
    prior_h1 = 1.0 - prior_h0
    opt_lr_threshold = prior_h0 / prior_h1
    opt_x_threshold = mean_h1 / 2 + np.log(opt_lr_threshold) / mean_h1
    return opt_x_threshold, prior_h1


@app.cell
def _(mean_h1, stats, x):
    rv_cond_h0 = stats.norm(0, 1)
    rv_cond_h1 = stats.norm(mean_h1, 1)

    pdf0 = rv_cond_h0.pdf(x)
    pdf1 = rv_cond_h1.pdf(x)
    return pdf0, pdf1, rv_cond_h0, rv_cond_h1


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


if __name__ == "__main__":
    app.run()
