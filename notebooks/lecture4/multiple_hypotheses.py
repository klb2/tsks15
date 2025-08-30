import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Multiple Hypothesis Testing

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebooks provides an interactive visualization of multiple hypothesis testing for detection problems.
    It follows the examples given in (S. Kay, Detection Theory, Section 3.8).
    In particular, we have $N$ hypothesis $\mathcal{H}_{i}$ with the following distributions conditioned on hypotheses $\mathcal{H}_{i}$:
    $$\mathcal{H}_i: \mathcal{N}\left(A\left(i-\left\lfloor\frac{N}{2}\right\rfloor\right), 1\right)$$
    with mean value $\mu_i = A\left(i-\left\lfloor\frac{N}{2}\right\rfloor\right)$.
    Each hypothesis has the prior probability $\Pr(\mathcal{H}_i)$.

    According to [Kay, Eq. (3.22)], we decide for hypothesis $k$ with the maximum a posteriori probability $\Pr(\mathcal{H}_k \mid x)$. By [Kay, Eq. (3.23)], this corresponds to $$\hat{i} = \argmax_{i} p(x \mid \mathcal{H}_i) \Pr(\mathcal{H}_i).$$
    With our Gaussian model, we have $$p(x \mid \mathcal{H}_i) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left[-\frac{1}{2\sigma^2}(x-\mu_i)^2\right].$$
    """
    )
    return


@app.cell
def _(
    decisions,
    idx_decision_boundaries,
    mo,
    np,
    plt,
    rvs,
    slider_a,
    slider_num_levels,
    sliders_priors,
    text_priors,
    x,
):
    _fig, _axs = plt.subplots(2, 1, squeeze=True, sharex=True)
    _ax_pdf, _ax_decision = _axs
    for _rv in rvs:
        _ax_pdf.plot(x, _rv.pdf(x), label=f"$\\mu_i={_rv.mean():.3f}$")
    _ax_pdf.vlines(
        x[idx_decision_boundaries],
        0,
        np.max(rvs[0].pdf(x)),
        color="k",
        ls="--",
        label="Decision Boundaries",
    )
    _ax_pdf.legend()
    _ax_pdf.set_ylabel("PDF")

    _ax_decision.step(x, decisions)
    _ax_decision.set_ylabel("Index of Selected Hypothesis")

    _fig.tight_layout()

    mo.hstack(
        [
            mo.mpl.interactive(_fig),
            mo.vstack(
                [slider_num_levels, slider_a, sliders_priors, mo.md(text_priors)]
            ),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(dc_value, mo, num_levels, priors, prob_correct, prob_error):
    mo.md(
        f"""
    ## Error Probability

    The probability of making an error $P_{{\\textnormal{{E}}}}$, i.e., deciding for the wrong hypothesis, can be calculated through $$P_{{\\textnormal{{E}}}} = 1- P_{{\\textnormal{{C}}}},$$ where $P_{{\\textnormal{{C}}}}$ is the probability of making the correct decision.


    ### Summary of Parameters

    | Parameter | Value |
    |:---|---:|
    | Number of hypotheses | $N={num_levels:d}$ |
    | Mean level/factor | $A = {dc_value:.3f}$ |
    | Prior probabilities of $\\mathcal{{H}}_i$ | {priors} |

    ### Resulting Probabilities

    | Parameter | Value |
    |:---|---:|
    | Correct Decision | $P_{{\\textnormal{{C}}}} = {prob_correct:.3f}$ |
    | Error Probability | $P_{{\\textnormal{{E}}}} = {prob_error:.3f}$ |
    """
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


@app.cell
def _(mo):
    slider_a = mo.ui.slider(0.2, 5, 0.2, 1, label="Mean Value $A$")
    slider_num_levels = mo.ui.slider(3, 7, 2, 3, label="Number of Hypotheses")
    return slider_a, slider_num_levels


@app.cell
def _(slider_a, slider_num_levels):
    dc_value = slider_a.value
    num_levels = slider_num_levels.value
    return dc_value, num_levels


@app.cell
def _(mo, num_levels):
    sliders_priors = mo.ui.array(
        [mo.ui.slider(0, 1, 0.02, 1 / num_levels)] * (num_levels - 1),
        label="Prior Probabilities (the last one is calculated through 1-sum(priors))",
    )
    return (sliders_priors,)


@app.cell
def _(dc_value, np, num_levels, sliders_priors, stats):
    x = np.linspace(-num_levels * dc_value, num_levels * dc_value, 250)

    rvs = [
        stats.norm(loc=(k - num_levels // 2) * dc_value, scale=1)
        for k in range(num_levels)
    ]
    priors = np.array(sliders_priors.value)
    priors = np.concatenate((priors, [1 - np.sum(priors)]))
    return priors, rvs, x


@app.cell
def _(decisions, np, priors, rvs, x):
    prob_correct_individual = [
        (
            rv.cdf(x[np.max(np.where(decisions == idx)[0], initial=0)])
            - rv.cdf(x[np.min(np.where(decisions == idx)[0], initial=0)])
        )
        * prior
        for idx, (rv, prior) in enumerate(zip(rvs, priors))
    ]
    prob_correct = np.sum(prob_correct_individual)
    prob_error = 1.0 - prob_correct
    return prob_correct, prob_error


@app.cell
def _(np, priors, rvs, x):
    posteriors = np.array([rv.pdf(x) * prior for (rv, prior) in zip(rvs, priors)])
    decisions = np.argmax(posteriors, axis=0)
    idx_decision_boundaries = np.where(np.diff(decisions))[0]
    return decisions, idx_decision_boundaries


@app.cell
def _(np, priors):
    if np.any(priors < 0):
        text_priors = f"""
        /// danger | Invalid prior probabilities!

        The selected prior probabilities are not a valid probability distribution: {priors}.
        ///
        """
        print(f"Invalid prior probabilities: {priors}")
        # raise ValueError(f"Invalid prior probabilities: {priors}")
    else:
        _priors = ", ".join([f"{k:.2f}" for k in priors])
        text_priors = f"""
        Selected prior probabilities: ({_priors})
        """
        print(f"Prior probabilities: {priors}")
    return (text_priors,)


if __name__ == "__main__":
    app.run()
