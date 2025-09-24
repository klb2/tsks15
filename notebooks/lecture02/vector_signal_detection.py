import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Detection of Vector Signals in Gaussian Noise

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebook provides a simple illustration of detecting a vector signal in Gaussian noise.
    In particular, we have the following signal model

    \begin{equation*}
    \begin{cases}
    \mathcal{H}_0 &: y = a + w\\
    \mathcal{H}_1 &: y = b + w\\
    \end{cases}
    \end{equation*}

    with known vector signals $a$ and $b$, and Gaussian noise $w\sim\mathcal{N}(0, \sigma^2 I)$.
    """
    )
    return


@app.cell
def _(
    X,
    Y,
    a,
    b,
    md_figure,
    mo,
    pdf_h0,
    pdf_h1,
    plt,
    slider_a,
    slider_b,
    slider_sigma,
):
    _fig, _axs = plt.subplots(1, 2, squeeze=True)
    _fig.set_tight_layout(True)
    _ax1, _ax2 = _axs
    for _ax in (_ax1, _ax2):
        _ax.set_xlabel(r"First component $y[1]$")
        _ax.set_ylabel(r"Second component $y[2]$")
    _ax1.set_title(r"Likelihood under $\mathcal{H}_0$ with center $a$")
    _ax1.contourf(X, Y, pdf_h0)
    _ax1.scatter(*a, marker="^", c="r")
    _ax2.set_title(r"Likelihood under $\mathcal{H}_1$ with center $b$")
    _ax2.contourf(X, Y, pdf_h1)
    _ax2.scatter(*b, c="k")

    mo.hstack(
        [
            mo.mpl.interactive(_fig),
            mo.vstack([mo.md(md_figure), slider_a, slider_b, slider_sigma]),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(md_roc, mo, plt, prob_detect, prob_fa):
    _fig, _ax = plt.subplots()
    _fig.set_tight_layout(True)
    _ax.plot(prob_fa, prob_detect)
    _ax.plot(prob_fa, prob_fa, "--", c="gray")
    _ax.set_xlabel("Probability of False Alarm $P_{\\text{FA}}$")
    _ax.set_ylabel("Probability of Detection $P_{\\text{D}}$")

    mo.hstack([mo.mpl.interactive(_fig), mo.md(md_roc)], widths=[1.75, 1])
    return


@app.cell
def _(X, Y, a, b, gamma, md_bayes, mo, plt, slider_prob_h0, statistic):
    _fig, _ax = plt.subplots()
    _fig.set_tight_layout(True)
    _plot = _ax.contourf(X, Y, statistic > gamma)
    _fig.colorbar(_plot)
    _ax.scatter(*a, marker="^", c="r")
    _ax.scatter(*b, c="k")
    _ax.set_xlabel("First component $y[1]$")
    _ax.set_ylabel("Second component $y[2]$")

    mo.hstack(
        [mo.mpl.interactive(_fig), mo.vstack([mo.md(md_bayes), slider_prob_h0])],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(mo):
    slider_sigma = mo.ui.slider(0.5, 5, 0.1, 1, label="Noise variance $\\sigma^2$")
    slider_a = mo.ui.array(
        [
            mo.ui.slider(-3, 3, 0.1, 1, label="$a_x$"),
            mo.ui.slider(-3, 3, 0.1, 1, label="$a_y$"),
        ],
        label="Point $a$",
    )
    slider_b = mo.ui.array(
        [
            mo.ui.slider(-3, 3, 0.1, -1, label="$b_x$"),
            mo.ui.slider(-3, 3, 0.1, -2, label="$b_y$"),
        ],
        label="Point $b$",
    )
    slider_prob_h0 = mo.ui.slider(
        0.01, 0.99, 0.01, 0.5, label="Prior probability $\\Pr(\\mathcal{H}_0)$"
    )
    return slider_a, slider_b, slider_prob_h0, slider_sigma


@app.cell
def _(a, b, grid, np, prob_fa, rv_h0, rv_h1, sigma, stats):
    pdf_h0 = rv_h0.pdf(grid)
    pdf_h1 = rv_h1.pdf(grid)

    prob_detect = stats.norm.sf(
        stats.norm.isf(prob_fa) - np.sqrt(np.linalg.norm(b - a) ** 2 / sigma)
    )
    prob_error = stats.norm.sf(np.linalg.norm(b - a) / (2 * np.sqrt(sigma)))
    statistic = grid @ (b - a)
    return pdf_h0, pdf_h1, prob_detect, statistic


@app.cell
def _(a, b, np, sigma, stats):
    rv_h0 = stats.multivariate_normal(mean=a, cov=sigma * np.identity(2))
    rv_h1 = stats.multivariate_normal(mean=b, cov=sigma * np.identity(2))
    return rv_h0, rv_h1


@app.cell
def _(np):
    x = np.linspace(-5, 5, 100)
    prob_fa = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, x)
    grid = np.dstack((X, Y))
    return X, Y, grid, prob_fa


@app.cell
def _(np, slider_a, slider_b, slider_sigma):
    a = np.array(slider_a.value)
    b = np.array(slider_b.value)
    sigma = slider_sigma.value
    return a, b, sigma


@app.cell
def _(a, b, np, sigma, slider_prob_h0):
    prob_h0 = slider_prob_h0.value
    prob_h1 = 1 - prob_h0

    gamma = (
        2 * sigma * np.log(prob_h0 / prob_h1)
        + (np.linalg.norm(b) ** 2 - np.linalg.norm(a) ** 2) / 2
    )
    return (gamma,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


@app.cell
def _():
    md_figure = r"""
    ## Illustration

    The figure shows the density of $y$ under both hypotheses with their respective mean values ($a$ and $b$).

    The sliders below allow adjusting the components of $a$ and $b$, and the noise variance $\sigma^2$.
    """
    return (md_figure,)


@app.cell
def _():
    md_roc = r"""
    ## ROC

    This figure shows the receiver operating characteristic (ROC) of the detector that uses the orthodox approach.
    """
    return (md_roc,)


@app.cell
def _():
    md_bayes = r"""
    ## Bayesian Detection

    This figure shows the decision regions for a detector that uses the Bayesian approach.
    The prior probability of $\mathcal{H}_0$ can be adjusted through the slider below.
    """
    return (md_bayes,)


if __name__ == "__main__":
    app.run()
