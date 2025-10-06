import marimo

__generated_with = "0.16.5"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Detecting an Unknown Signal in Gaussian Noise

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    In this notebook, we consider the detection of an unknown signal in noise.
    In particular, we want to detect an unknown DC level $A$ in Gaussian noise ${w\sim\mathcal{N}(0, I)}$.
    Thus, our system model is given as

    \begin{equation*}
    \begin{cases}
    \mathcal{H}_0: & y[n] = w[n]\\
    \mathcal{H}_1: & y[n] = A + w[n]
    \end{cases}
    \end{equation*}

    where we observe $N$ samples $y[n]$, ${n=0, 1, \dots, N-1}$.

    In matrix-vector form, the parameter vector $\theta$ is equal to the scalar $A$, and matrix $H$ is a vector of $N$ ones, i.e., ${H=\mathbf{1}}$.

    The detector that we use is based on the generalized likelihood ratio test (GLRT) for linear models with Gaussian noise.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Orthodox Approach

    In the orthodox approach, the GLRT can be simplified to

    \begin{align*}
    y^T P y &\gtrless \gamma\\
    \Leftrightarrow\quad \frac{1}{N} y^T \mathbf{1} \mathbf{1}^T y &\gtrless \gamma\\
    \Leftrightarrow\quad \left|\frac{1}{N} \sum_{n=0}^{N-1} y[n]\right| &\gtrless \gamma'
    \end{align*}

    i.e., we need to compare the absolute value of the sample average with a threshold $\gamma'$.
    The resulting detection probability is

    \begin{equation*}
    P_{\text{D}} = Q_{\chi^{'2}_1(NA^2)}\left(Q_{\chi^2_1}^{-1}\left(P_{\text{FA}}\right)\right)
    \end{equation*}

    While the above expressions appear complicated, they can easily be evaluated using popular libraries.
    In scipy, the [$Q$-function (survival function)](https://en.wikipedia.org/wiki/Survival_function) of the non-central chi-square distribution ${Q_{\chi^{'2}_k(\lambda)}}$ can be computed by simply calling the [`scipy.stats.ncx2.sf` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ncx2.html).

    ### Comparison with Known $A$
    For comparison, if we know the DC level $A$, we can apply the Neyman-Pearson theorem from the first lectures, which yields a detection probability of

    \begin{equation*}
    P_{\text{D}} = Q\left(Q^{-1}\left(P_{\text{FA}}\right) - \sqrt{NA^2}\right)
    \end{equation*}

    where $Q$ denotes the "regular" [$Q$-function](https://en.wikipedia.org/wiki/Q-function), i.e., the survival function of the standard normal distribution.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## (Semi-)Bayesian Approach

    Alternatively, we can use the semi-Bayesian approach where the estimation of the unknown parameter is done through the Bayesian framework while the detection is following the orthodox approach.
    As the prior distribution of $A$, we assume a normal distribution with zero mean and variance $\sigma^2$, i.e., ${p(A)=\mathcal{N}(0, \sigma^2)}$.

    The resulting detection probability can be evaluated to

    \begin{equation*}
    P_{\text{D}} = Q_{\chi^{2}_1}\left(\frac{Q_{\chi^2_1}^{-1}\left(P_{\text{FA}}\right)}{1 + N \sigma^2}\right)
    \end{equation*}
    """
    )
    return


@app.cell
def _(
    md_roc_orthodox,
    mo,
    plt,
    prob_detect_known,
    prob_detect_semi_bayes,
    prob_detect_unknown,
    prob_fa,
    slider_dc_level,
    slider_num_samples,
    slider_var_prior,
):
    _fig, _axs = plt.subplots()
    _fig.set_tight_layout(True)
    _axs.plot(prob_fa, prob_detect_unknown, label=r"Unknown $A$")
    _axs.plot(prob_fa, prob_detect_known, label=r"Known $A$")
    _axs.plot(prob_fa, prob_detect_semi_bayes, label=r"Semi-Bayesian Approach")
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
                    slider_var_prior,
                    mo.mpl.interactive(_fig),
                ]
            ),
            mo.md(md_roc_orthodox),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _():
    md_roc_orthodox = r"""
    ## Receiver Operating Characteristic

    The plot shows the receiver operating characteristic (ROC) for the three different detectors based on the GLRT.
    The first two curves present the results for the orthodox approach, both for unknown $A$ and known $A$ (with the optimal detector based on the Neyman-Pearson theorem).

    As expected, the performance is better if $A$ is known.
    However, both curves show the same behavior with respect to the parameters $A$ and $N$.
    If $|A|$ increases, the performance gets better (as the SNR increases).
    Similarly, if we collect more samples, i.e., increase $N$, the performance also improves.

    For the semi-Bayesian approach, the performance depends on the number of samples $N$ and the variance of the prior distribution $\sigma^2$.
    As expected, the performance improves if $N$ increases.
    However, if we have a sharper prior distribution, i.e., if $\sigma^2$ is small, the performance decreases.
    The reason behind this is that, while we have better prior information about $A$, it will almost always be close to zero (since the prior distribution has zero mean). This makes the detection problem very difficult as the SNR is very low.
    """
    return (md_roc_orthodox,)


@app.cell
def _(slider_num_samples):
    num_samples = int(slider_num_samples.value)
    return (num_samples,)


@app.cell
def _(slider_dc_level):
    dc_level = slider_dc_level.value
    return (dc_level,)


@app.cell
def _(slider_var_prior):
    var_prior = slider_var_prior.value
    return (var_prior,)


@app.cell
def _(dc_level, np, num_samples, prob_fa, stats, var_prior):
    prob_detect_unknown = stats.ncx2(df=1, nc=num_samples * dc_level**2).sf(
        stats.chi2(df=1).isf(prob_fa)
    )

    prob_detect_known = stats.norm.sf(
        stats.norm.isf(prob_fa) - np.sqrt(num_samples * dc_level**2)
    )

    prob_detect_semi_bayes = stats.chi2(df=1).sf(
        stats.chi2(df=1).isf(prob_fa) / (1 + num_samples * var_prior)
    )
    return prob_detect_known, prob_detect_semi_bayes, prob_detect_unknown


@app.cell
def _(mo, np):
    prob_fa = np.logspace(0, -7, 150)[::-1]  # np.linspace(0, 1, 150)
    prob_fa = np.concat(([0], prob_fa))

    slider_num_samples = mo.ui.slider(1, 30, 1, 10, label=r"Number of samples $N$")
    slider_dc_level = mo.ui.slider(-3, 3, 0.1, 1, label=r"DC level $A$")
    slider_var_prior = mo.ui.slider(
        0.01, 10, 0.1, 1, label=r"Variance $\sigma^2$ of the prior of $A$"
    )
    return prob_fa, slider_dc_level, slider_num_samples, slider_var_prior


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


if __name__ == "__main__":
    app.run()
