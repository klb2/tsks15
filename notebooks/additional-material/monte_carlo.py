import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Monte Carlo Simulations for Detection and Estimation

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    Monte Carlo (MC) simulations are a numerical method to evaluate the performance of an estimator.
    The basic idea is to generate $M$ independent trials of your experiment and average the performance over all $M$ runs.

    Some basic rules should be kept in mind:

    - More runs (higher $M$) = better accuracy of the results
    - For each MC trial, generate new independent realiziations of all random variables that are part of the averaging
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Example: DC Level in AWGN

    First, we look at the simple model of an unknown (but constant) DC level $A$ in additive white Gaussian noise (AWGN).
    The $N$ measurements are therefore given as $$y[n] = A + w[n], \quad n=0, \dots, N-1$$ with independent $w[n]\sim\mathcal{N}(0, 1)$.
    As the estimator, we use the sample mean $$\hat{A} = \frac{1}{N} \sum_{n=0}^{N-1} y[n].$$

    The following results demonstrate the performance of this estimator evaluated through MC simulations.

    /// attention | Note

    In this example, we have two different quantities that describe "a number of runs". $N$ is the number of measurement sample we take in each run. In contrast, $M$ is the number of MC runs.
    ///
    """
    )
    return


@app.cell
def _(dc_level, np, num_mc_trials, num_measurement):
    measurements = dc_level + np.random.randn(num_mc_trials, num_measurement)
    estimates = np.mean(measurements, axis=1)
    mean_estimate_varying_M = np.cumsum(estimates) / (np.arange(num_mc_trials) + 1)
    return (mean_estimate_varying_M,)


@app.cell
def _(
    dc_level,
    md_emp_mean,
    mean_estimate_varying_M,
    mo,
    np,
    num_mc_trials,
    plt,
    slider_n,
):
    _fig, _axs = plt.subplots(1, 1, squeeze=True)
    _fig.set_tight_layout(True)
    _ax_mean = _axs

    _ax_mean.plot(np.arange(num_mc_trials) + 1, mean_estimate_varying_M)
    _ax_mean.hlines(dc_level, 0, num_mc_trials, ls="--", color="gray", alpha=0.5)
    _ax_mean.set_ylim([0.8 * dc_level, 1.2 * dc_level])
    _ax_mean.set_xlabel("Number of MC Trials $M$")
    _ax_mean.set_ylabel(r"Empirical Mean of $\hat{A}$")


    mo.hstack(
        [mo.vstack([slider_n, mo.mpl.interactive(_fig)]), mo.md(md_emp_mean)],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(md_emp_var, mo, plt, snr_db, var_a_est):
    _fig, _axs = plt.subplots()
    _fig.set_tight_layout(True)

    for _M, _var in var_a_est.items():
        _axs.semilogy(snr_db, _var, ".-", label=f"$M={_M:d}$")
    _axs.legend()
    _axs.set_xlabel("SNR [dB]")
    _axs.set_ylabel(r"Empirical Variance of $\hat{A}$")


    mo.hstack(
        [mo.mpl.interactive(_fig), mo.md(md_emp_var)],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(dc_level, np):
    _N = 10  # fix the number of samples per run to N=10

    snr_db = np.linspace(0, 20, 20)
    snr = 10 ** (-snr_db / 10.0)

    var_a_est = {}
    for _M in [10, 100, 1000, 10000]:
        _M = int(_M)
        var_a_est[_M] = []
        for _snr in snr:
            A_est = np.zeros(_M)
            for _m in range(_M):
                y = dc_level + np.sqrt(_snr) * np.random.randn(_N)
                A_est[_m] = np.mean(y)
            var_a_est[_M].append(np.var(A_est))
    return snr, snr_db, var_a_est


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Reusing Samples between MC Runs

    It is important that you generate new and independent realizations of all random variables in each MC run.
    Otherwise, the results may appear too smoth and are potentially offset from their actual value.

    /// details | :white_check_mark: Do this
    ```python
    y = A + np.sqrt(snr) * np.random.randn(N, M, len(snr))
    A_est = np.mean(y, axis=0)
    var_a_est = np.var(A_est, axis=0)
    ```
    ///

    /// details | :x: Do _NOT_ do this
    ```python
    noise = np.random.randn(N, M)
    for _snr in snr:
        y = A + np.sqrt(_snr) * noise  # this reuses the same noise samples for each SNR point
        A_est_wrong = np.mean(y, axis=0)
        var_wrong.append(np.var(A_est_wrong))
    ```
    ///

    The following example shows the results and differences between both implementations.
    For this, we fix ${N=10}$ and ${M=100}$.
    """
    )
    return


@app.cell
def _(dc_level, np, snr):
    _N = 10
    _M = 100

    y_correct = dc_level + np.sqrt(snr) * np.random.randn(_N, _M, len(snr))
    A_est_correct = np.mean(y_correct, axis=0)
    var_a_est_correct = np.var(A_est_correct, axis=0)


    var_wrong = []
    noise = np.random.randn(_N, _M)
    for _snr in snr:
        y_wrong = dc_level + np.sqrt(_snr) * noise
        A_est_wrong = np.mean(y_wrong, axis=0)
        var_wrong.append(np.var(A_est_wrong))
    return var_a_est_correct, var_wrong


@app.cell
def _(md_correct_vs_wrong, mo, plt, snr_db, var_a_est_correct, var_wrong):
    _fig, _axs = plt.subplots()
    _fig.set_tight_layout(True)

    _axs.semilogy(snr_db, var_a_est_correct, label=f"Correct")
    _axs.semilogy(snr_db, var_wrong, label="Wrong")
    _axs.legend()
    _axs.set_xlabel("SNR [dB]")
    _axs.set_ylabel(r"Empirical Variance of $\hat{A}$")


    mo.hstack(
        [mo.mpl.interactive(_fig), mo.md(md_correct_vs_wrong)],
        widths=[1.75, 1],
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _():
    md_emp_mean = r"""
    ### Mean of the Estimator

    The figure on the left shows the empirical mean of the estimate $\hat{A}$ over a varying number of MC trials $M$ (for fixed $N$), i.e., $$\frac{1}{M}\sum_{m=1}^{M} \hat{A}[m],$$ where each estimate $\hat{A}[m]$ stems from a different MC run with $N$ independent measurement samples $y[n]$.

    The slider above the figure allows changing the value of $N$.
    In particular, increasing $N$ improves the quality of the estimator (reducing its variance), i.e., even for a small number of MC trials, its value is close to $A$.
    """
    return (md_emp_mean,)


@app.cell
def _():
    md_emp_var = r"""
    ### Variance of the Estimator

    Next, we will take a look at the variance of the estimator for different SNR values and varying numbers of MC runs $M$.
    For this, we fix the number of samples in each run to ${N=10}$.

    First, it can be seen that the general trend is the same for all $M$ (as it should be).
    The variance of the estimator decreases if the SNR increases (i.e., the variance of the noise decreases).

    However, for a small number of MC runs ($M=10$ and $M=100$), the estimated variance fluctuates a lot, whereas it gets closer to the theoretical line for large $M$.
    """
    return (md_emp_var,)


@app.cell
def _():
    md_correct_vs_wrong = r"""
    This figure shows the empirical variance of $\hat{A}$ for the correct implementation and the wrong one that reuses noise samples between MC runs. 

    The correct curve looks similar to the one for the same number of MC runs (${M=100}$) in the figure above.
    In contrast, the one for which noise samples have been reused looks overly smooth.
    """
    return (md_correct_vs_wrong,)


@app.cell
def _():
    num_mc_trials = 25000
    dc_level = 1
    return dc_level, num_mc_trials


@app.cell
def _(slider_n):
    num_measurement = int(slider_n.value)
    return (num_measurement,)


@app.cell
def _(mo, np):
    slider_n = mo.ui.slider(
        steps=np.logspace(0, 4, 5), label="Number of measurements $N$"
    )
    return (slider_n,)


if __name__ == "__main__":
    app.run()
