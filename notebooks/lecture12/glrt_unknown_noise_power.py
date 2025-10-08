import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Detecting an Unknown Signal in Gaussian Noise

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    In this notebook, we consider the detection of an (unknown) signal in noise.
    In particular, we want to detect a DC level $A$ in Gaussian noise ${w\sim\mathcal{N}(0, \sigma^2I)}$.
    Thus, our system model is given as

    \begin{equation*}
    \begin{cases}
    \mathcal{H}_0: & y[n] = w[n]\\
    \mathcal{H}_1: & y[n] = A + w[n]
    \end{cases}
    \end{equation*}

    where we observe $N$ samples $y[n]$, ${n=0, 1, \dots, N-1}$.

    The detector that we use is based on the (generalized) likelihood ratio test (GLRT) for linear models with Gaussian noise.


    ## Assumptions

    In this notebook, we compare three different assumptions on knowledge that we have about the parameters in the system model.
    In particular, we assume:

    1. Full knowledge
    2. Only knowledge about the noise statistics
    3. No knowledge about the parameters
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Full Knowledge

    The first case is the case that we know both the DC level $A$ and the noise power $\sigma^2$.
    This corresponds to the assumptions we made in the first few lectures of the course.
    The resulting receiver operating characteristic (ROC) is given as

    \begin{equation*}
    P_{\text{D,full}} = Q\left(Q^{-1}\left(P_{\text{FA}}\right) - \sqrt{\frac{NA^2}{\sigma^2}}\right)
    \end{equation*}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### Partial Knowledge

    Next, we drop the assumption that we have knowledge about the DC level $A$.
    However, we still know $\sigma^2$.

    With this, we can write our problem in matrix-vector form where the parameter vector $\theta$ is equal to the scalar $A$, and matrix $H$ is a vector of $N$ ones, i.e., ${H=\mathbf{1}}$.
    Evaluating the GLRT yields the following ROC

    \begin{equation*}
    P_{\text{D}} = Q_{\chi^{'2}_1({NA^2}/{\sigma^2})}\left(Q_{\chi^2_1}^{-1}\left(P_{\text{FA}}\right)\right)
    \end{equation*}
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ### No Knowledge

    Finally, we assume that we do not have any knowledge about $A$ and $\sigma^2$.
    In this case, the ROC resulting from the GLRT is given by

    \begin{equation*}
    Q_{\mathrm{F}'_{1,N-1}(NA^2/\sigma^2)}\left({Q^{-1}_{F_{1,N-1}}}\left(P_{\text{FA}}\right)\right)
    \end{equation*}
    """
    )
    return


@app.cell
def _(
    md_results,
    mo,
    plt,
    prob_detect_known,
    prob_detect_unknown_a,
    prob_detect_unknown_a_sigma,
    prob_fa,
    slider_dc_level,
    slider_num_samples,
    slider_var_noise,
):
    _fig, _axs = plt.subplots()
    _fig.set_tight_layout(True)
    _axs.plot(
        prob_fa,
        prob_detect_unknown_a_sigma,
        label=r"Unknown $A$ and unknown $\sigma^2$",
    )
    _axs.plot(
        prob_fa, prob_detect_unknown_a, label=r"Unknown $A$ and known $\sigma^2$"
    )
    _axs.plot(prob_fa, prob_detect_known, label=r"Known $A$ and $\sigma^2$")
    _axs.plot([0, 1], [0, 1], "--", c="gray")
    _axs.set_xlabel(r"Probability of False Alarm $P_{\text{FA}}$")
    _axs.set_ylabel(r"Probability of Detection $P_{\text{D}}$")
    _axs.grid()
    _axs.legend()

    mo.hstack(
        [
            mo.vstack(
                [
                    slider_num_samples,
                    slider_dc_level,
                    slider_var_noise,
                    mo.mpl.interactive(_fig),
                ]
            ),
            mo.md(md_results),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _():
    md_results = rf"""
    ## Results

    The plot shows the ROCs under the three different assumptions.
    All curves show the same general behavior with respect to the system parameters.

    1. If the number of samples $N$ increases, the performance improves.
    2. If the SNR ($A^2/\sigma^2$) increases, the performance improves.

    As expected, having more knowledge about the system parameters, i.e., having less unknowns, improves the performance of the detection.
    While having full knowledge about $A$ and $\sigma^2$ always leads to significantly better detection probabilities, having knowledge about $\sigma^2$ (while not knowing $A$) is especially beneficial at medium SNRs.
    (At very low SNRs, the performance is bad in both cases, and at very high SNRs, the performance is good in both cases.)
    """
    return (md_results,)


@app.cell
def _(mo, np):
    prob_fa = np.logspace(0, -7, 150)[::-1]  # np.linspace(0, 1, 150)
    prob_fa = np.concat(([0], prob_fa))

    slider_num_samples = mo.ui.slider(1, 20, 1, 10, label=r"Number of samples $N$")
    slider_dc_level = mo.ui.slider(-3, 3, 0.1, 1, label=r"DC level $A$")
    slider_var_noise = mo.ui.slider(
        0.01, 10, 0.1, 1, label=r"Variance $\sigma^2$ of the noise $w$"
    )
    return prob_fa, slider_dc_level, slider_num_samples, slider_var_noise


@app.cell
def _(dc_level, np, num_samples, prob_fa, stats, var_noise):
    prob_detect_unknown_a = stats.ncx2(
        df=1, nc=num_samples * dc_level**2 / var_noise
    ).sf(stats.chi2(df=1).isf(prob_fa))

    prob_detect_known = stats.norm.sf(
        stats.norm.isf(prob_fa) - np.sqrt(num_samples * dc_level**2 / var_noise)
    )

    prob_detect_unknown_a_sigma = stats.ncf(
        dfn=1, dfd=num_samples - 1, nc=num_samples * dc_level**2 / var_noise
    ).sf(stats.f(dfn=1, dfd=num_samples - 1).isf(prob_fa))
    return (
        prob_detect_known,
        prob_detect_unknown_a,
        prob_detect_unknown_a_sigma,
    )


@app.cell
def _(slider_dc_level):
    dc_level = slider_dc_level.value
    return (dc_level,)


@app.cell
def _(slider_num_samples):
    num_samples = int(slider_num_samples.value)
    return (num_samples,)


@app.cell
def _(slider_var_noise):
    var_noise = slider_var_noise.value
    return (var_noise,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


if __name__ == "__main__":
    app.run()
