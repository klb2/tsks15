import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Best Linear Unbiased Estimator (BLUE) for Non-Gaussian Noise

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)

    This notebook illustrates the use of the best linear unbiased estimator (BLUE) for the estimation of a DC level $A$ in non-Gaussian noise $w$.
    In particular, the signal model is $$y[n] = A + w[n],$$ where the noise samples $w[n]$ are independent and distributed according to a Laplace distribution with variance $\sigma^2_n$.

    The BLUE is given as $$\hat{A}_{\text{BLUE}} = \dfrac{\displaystyle\sum_{n=0}^{N-1} \frac{y[n]}{\sigma^2_n}}{\displaystyle\sum_{n=0}^{N-1} \sigma^2_n}$$ with variance $$\mathrm{var}(\hat{A}_{\text{BLUE}}) = \dfrac{1}{\displaystyle\sum_{n=0}^{N-1} \sigma^2_n}.$$
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Sliders

    In the following, you can update the value of $A$ and the variance of the fourth sample $\sigma^2_3$ through the sliders.
    The plot will show the signal $y$ together with the true value of $A$ (gray dashed line) and its estimate $\hat{A}_{\text{BLUE}}$ (black dash-dotted line).
    """
    )
    return


@app.cell
def _(
    dc_estimate,
    dc_level,
    md_results_est,
    mo,
    num_steps,
    plt,
    slider_a,
    slider_var,
    t,
    y,
):
    _fig, _axs = plt.subplots()
    _axs.plot(t, y, "o-")
    _axs.plot([3], [y[3]], "o")
    _axs.hlines(dc_level, 0, num_steps, "gray", "--")
    _axs.hlines(dc_estimate, 0, num_steps, "black", "-.")
    _axs.set_xlabel("Sample Index $n$")
    _axs.set_ylabel("Signal $y[n]$")

    mo.hstack(
        [
            mo.mpl.interactive(_fig),
            mo.vstack([slider_a, slider_var, mo.md(md_results_est)]),
        ],
        widths=[2, 1],
    )
    return


@app.cell
def _(md_results_var, mo):
    mo.md(md_results_var)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


@app.cell
def _(mo, np):
    slider_a = mo.ui.slider(0.5, 5, 0.1, 1, label="DC Level $A$")
    slider_var = mo.ui.slider(
        steps=np.logspace(-3, 4, 8),
        value=1,
        label="Variance at $n=3$",
    )
    return slider_a, slider_var


@app.cell
def _(np, stats):
    num_steps = 10
    t = np.arange(num_steps)
    noise_variances = stats.gamma(2, scale=0.5).rvs(num_steps)
    return noise_variances, num_steps, t


@app.cell
def _():
    # dc_level = slider_a.value  # move this command into a different cell, if you want the noise samples to be updated every time you change the DC value
    return


@app.cell
def _(noise_variances, np, slider_a, slider_var, stats):
    dc_level = slider_a.value  # move this command to its own cell if you do not want the noise samples to be updated every time you change the DC value
    var3 = slider_var.value
    noise_variances[3] = var3
    noise = stats.laplace(scale=np.sqrt(noise_variances / 2)).rvs()
    return dc_level, noise


@app.cell
def _(dc_level, noise, noise_variances, np):
    y = dc_level + noise
    var_estimate = 1 / np.sum(1 / noise_variances)
    weighted_samples = y / noise_variances
    dc_estimate = np.sum(weighted_samples) * var_estimate
    return dc_estimate, var_estimate, weighted_samples, y


@app.cell
def _(dc_estimate, dc_level, noise_variances, var_estimate, weighted_samples):
    md_results_est = f"""
    ## Results

    ### Estimation
    For the samples in the left plot, we obtain the following estimation results using the BLUE:

    \\begin{{align}}
    A &= {dc_level:.3f}\\\\
    \\hat{{A}}_{{\\text{{BLUE}}}} &= {dc_estimate:.3f}\\\\
    \\mathrm{{var}}(\\hat{{A}}_{{\\text{{BLUE}}}}) &= {var_estimate:.3f}
    \\end{{align}}
    """

    md_results_var = f"""
    ### Variances and Weighting
    The variances of the individual Laplace-noise samples are:

    \\begin{{equation*}}
    \\mathrm{{cov}}(w) = \\mathrm{{diag}}\\begin{{pmatrix}}{" & ".join(f"{k:.2f}" for k in noise_variances)}\\end{{pmatrix}}
    \\end{{equation*}}

    and the corresponding weighted samples:

    \\begin{{equation*}}
    \\frac{{y}}{{\\sigma^2}} = \\begin{{pmatrix}}{" & ".join(f"{k:.3f}" for k in weighted_samples)}\\end{{pmatrix}}
    \\end{{equation*}}


    From this, you can see that the samples are weighted according to the corresponding noise variance.
    Pay attention to the fourth sample ($n=3$) as you can control its variance through the slider above.
    If the variance is very large, the weighted sample will be close to zero, i.e., be ignored in the estimate.
    On the contrary, if the variance is very small, a lot of weight is put on this sample as it is virtually noise-free.
    """
    return md_results_est, md_results_var


if __name__ == "__main__":
    app.run()
