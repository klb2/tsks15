import marimo

__generated_with = "0.14.17"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Example: FIR Filter (Tapped Delay Line)

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebook illustrates the estimation of the filter parameters in a tapped delay line (FIR filter).
    An illustration of an FIR filter can be found below.

    ![Illustration of an FIR filter](https://upload.wikimedia.org/wikipedia/commons/9/9b/FIR_Filter.svg)


    In this notebook, you can find an estimation of the filter parameters ($b_i$ in the illustration) using the minimum-variance unbiased (MVU) estimator.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Probing Signals

    For the following simulation, you can select a probing signal $x[n]$ and see how it effects the estimate and Cramer-Rao lower bound/variances of the estimates $\hat{b}_i$.
    """
    )
    return


@app.cell
def _(dd_probing_signal):
    dd_probing_signal
    return


@app.cell
def _(
    filter_coeff,
    md_estimation,
    mo,
    output_signal,
    plt,
    probing_signal,
    t,
    var_estimates,
):
    _fig, _axs = plt.subplots(1, 3, squeeze=True)
    _ax_input = _axs[0]
    _ax_input.plot(t, probing_signal, ".-")
    _ax_input.set_title("Probing Signal $x[n]$")
    _ax_input.set_xlabel("Time Index $n$")

    _ax_output = _axs[1]
    _ax_output.plot(t, output_signal, ".-")
    _ax_output.set_title("Output Signal $y[n]$")
    _ax_output.set_xlabel("Time Index $n$")

    _ax_variances = _axs[2]
    _ax_variances.bar(
        [f"$\\hat{{b}}_{{{i}}}$" for i in range(len(filter_coeff))],
        var_estimates,
    )
    _ax_variances.set_title("Variances of the Estimates")
    _ax_variances.set_ylim([0, 1])

    _fig.tight_layout()

    mo.hstack([mo.mpl.interactive(_fig), mo.md(md_estimation)], widths=[2.5, 1])
    return


@app.cell
def _(dd_probing_signal, filter_coeff, linalg, noise, np):
    probing_signal = dd_probing_signal.value

    probing_matrix = linalg.toeplitz(probing_signal, np.zeros(len(filter_coeff)))
    output_signal = probing_matrix @ filter_coeff + noise

    est_filter_coeff = (
        linalg.inv(probing_matrix.T @ probing_matrix)
        @ probing_matrix.T
        @ output_signal
    )
    crb_filter_coeff = linalg.inv(probing_matrix.T @ probing_matrix)
    var_estimates = np.diag(crb_filter_coeff)
    return est_filter_coeff, output_signal, probing_signal, var_estimates


@app.cell
def _(mo, np, num_timeslots, t):
    dd_probing_signal = mo.ui.dropdown(
        options={
            "Sine": 2 * np.sin(t),
            "Sine High Energy": 10 * np.sin(t),
            "Random Binary": np.random.randint(2, size=num_timeslots),
            "Pulse": np.where(np.arange(num_timeslots) < num_timeslots // 2, 1, 0),
        },
        value="Sine",
        label="Select a probing signal $x$",
    )
    return (dd_probing_signal,)


@app.cell
def _(np):
    num_timeslots = 50
    t = np.arange(num_timeslots)
    filter_coeff = [1, 0.5, -0.75, 0.1]
    noise = np.random.randn(num_timeslots)
    return filter_coeff, noise, num_timeslots, t


@app.cell
def _(est_filter_coeff, filter_coeff, var_estimates):
    _table_rows = [
        f"| {h:.2f} | {e:.2f} | {v:.3f} |"
        for h, e, v in zip(filter_coeff, est_filter_coeff, var_estimates)
    ]
    _table_body = "\n".join(_table_rows)
    md_estimation = f"""
    ## Estimation Results

    The true and estimated filter coefficients are shown in the following table together with the variances of the estimates (which are also shown in the third plot).

    | Coefficient $b$ | Estimate $\\hat{{b}}$ | Variance of $\\hat{{b}}$ |
    |----------------:|----------------------:|-------------------------:|
    {_table_body}
    """
    return (md_estimation,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import linalg, signal
    import matplotlib.pyplot as plt
    return linalg, mo, np, plt


if __name__ == "__main__":
    app.run()
