import marimo

__generated_with = "0.16.1"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Maximum Likelihood Estimation

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebooks illustrates the maximum likelihood (ML) estimation of an unknown frequency.
    In particular, it is meant to visualize the influence of the number of samples $N$ and the noise variance $\sigma^2$ on the variance of the ML estimator.

    We observe $N$ measurements $$y[n]=\cos\left(2\pi f \frac{n}{N}\right) + w[n]$$ which are corrupted by iid Gaussian noise ${w[n]\sim\mathcal{N}(0, \sigma^2)}$. (Note that this is slightly different to the problem in the lecture as we introduce the normalization by $N$ inside the cosine to simplify the computations.)

    The goal of ML estimation is to find the frequency $\hat{f}_{\text{ML}}$ that maximizes the likelihood function $p(y;f)$.
    Due to the nonlinear nature of the estimation problem, we use the numerical optimization routine [`scipy.optimize.minimize`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.minimize.html) from the scipy library.
    """
    )
    return


@app.cell
def _(
    freq,
    likelihoods,
    md_illustration,
    md_params,
    mo,
    np,
    opt_results,
    param_line,
    plt,
    slider_freq,
    slider_num_samples,
    slider_var_noise,
):
    _params = param_line.ravel()
    _fig, _axs = plt.subplots()
    _fig.set_tight_layout(True)
    _axs.set_xlabel("Frequency")
    _axs.set_ylabel(r"Log Likelihood $\log p(y; f)$")

    for _idx, _likelihood in enumerate(likelihoods):
        _axs.plot(_params, _likelihood, label=f"Realization {_idx + 1:d}")
    _axs.vlines(
        freq,
        np.min(likelihoods),
        np.max(likelihoods),
        ls="--",
        color="r",
        label="True Value $f$",
    )
    _axs.vlines(
        opt_results[:2],
        np.min(likelihoods),
        np.max(likelihoods),
        ls="--",
        color="k",
        label=r"ML Estimates $\hat{f}_{\text{ML}}$",
    )
    _axs.legend()

    mo.hstack(
        [
            mo.vstack(
                [
                    mo.hstack(
                        [
                            mo.vstack(
                                [
                                    slider_freq,
                                    slider_num_samples,
                                    slider_var_noise,
                                ]
                            ),
                            mo.md(md_params),
                        ],
                        widths=[1, 1],
                    ),
                    mo.mpl.interactive(_fig),
                ]
            ),
            mo.md(md_illustration),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(freq, num_samples, var_noise):
    md_illustration = rf"""
    ## Illustration

    The plot shows the (log)-likelihood function for two different vectors $y$, i.e., we run the experiment twice (with $N$ samples each) and compute the ML estimate $\hat{{f}}_{{\text{{ML}}}}$.
    You should be able to make the following observations when varying the different parameters:

    1. For a large number of samples $N$, the runs are very consistent and the likelihood functions (and their maxima) should be very similar.  
    $\rightarrow \hat{{f}}_{{\text{{ML}}}}$ has a low variance
    2. For a small noise variance $\sigma^2$, the likelihood functions (and their maxima) should be very similar as the SNR is large and the received signal is not strongly corrupted by noise.  
    $\rightarrow \hat{{f}}_{{\text{{ML}}}}$ has a low variance
    3. If you set $N$ to a very small value and $\sigma^2$ to a large value, the curves may look vastly different. (You can vary $\sigma^2$ a little bit to rerun the experiment.)  
    $\rightarrow \hat{{f}}_{{\text{{ML}}}}$ has a high variance
    """

    md_params = rf"""
    **Selected Parameters**

    \begin{{align*}}
    f &= {freq:.2f}\\
    N &= {num_samples:d}\\
    \sigma^2 &= {var_noise:.2f}
    \end{{align*}}
    """
    return md_illustration, md_params


@app.cell
def _(n, np, num_samples):
    def ml_opt_function(f, y):
        _part1 = np.sum(np.cos(2 * np.pi * f / num_samples * n) ** 2)
        _part2 = 2 * np.sum(y * np.cos(2 * np.pi * f / num_samples * n))
        return _part1 - _part2
    return (ml_opt_function,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import optimize
    import matplotlib.pyplot as plt
    return mo, np, optimize, plt


@app.cell
def _(mo, np):
    # slider_num_samples = mo.ui.slider(5, 50, 1, 10, label=r"Number of samples $N$")
    slider_num_samples = mo.ui.slider(
        steps=np.logspace(1, 5, 5), value=100, label=r"Number of samples $N$"
    )
    slider_var_noise = mo.ui.slider(
        0.01, 10, 0.1, 1, label=r"Noise variance $\sigma^2$"
    )
    return slider_num_samples, slider_var_noise


@app.cell
def _(np, slider_num_samples):
    num_samples = int(slider_num_samples.value)
    n = np.arange(num_samples)
    return n, num_samples


@app.cell
def _(np, slider_var_noise):
    var_noise = slider_var_noise.value
    std_noise = np.sqrt(var_noise)
    return std_noise, var_noise


@app.cell
def _(mo):
    slider_freq = mo.ui.slider(0.1, 1, 0.1, 0.5, label=r"Frequency $f$")
    return (slider_freq,)


@app.cell
def _(n, np, num_samples, slider_freq):
    freq = slider_freq.value
    signal = np.cos(2 * np.pi * freq / num_samples * n)
    return freq, signal


@app.cell
def _(np, num_mc_runs, num_samples, std_noise):
    noise = std_noise * np.random.randn(num_mc_runs, num_samples)
    return (noise,)


@app.cell
def _(noise, signal):
    measurements = signal + noise
    return (measurements,)


@app.cell
def _(measurements, ml_opt_function, optimize):
    opt_results = [
        optimize.minimize(ml_opt_function, x0=0.5, args=(_y,), bounds=[(0, 1)]).x
        for _y in measurements
    ]
    return (opt_results,)


@app.cell
def _(measurements, n, np, num_samples, var_noise):
    param_line = np.linspace(0, 1, 150)
    param_line = np.expand_dims(param_line, (1,))
    likelihoods = (
        -1
        / (2 * var_noise)
        * np.sum(
            (
                measurements.T
                - np.tile(
                    np.cos(2 * np.pi * param_line / num_samples * n).T, (2, 1, 1)
                ).T
            )
            ** 2,
            axis=1,
        ).T
    )
    return likelihoods, param_line


@app.cell
def _():
    num_mc_runs = 2
    return (num_mc_runs,)


if __name__ == "__main__":
    app.run()
