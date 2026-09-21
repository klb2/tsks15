import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Example: FIR Filter (Tapped Delay Line)

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebook illustrates the estimation of the filter parameters in a tapped delay line (FIR filter).
    An illustration of an FIR filter can be found below.

    ![Illustration of an FIR filter](https://upload.wikimedia.org/wikipedia/commons/9/9b/FIR_Filter.svg)


    In this notebook, you can find an estimation of the filter parameters ($b_i$ in the illustration) using the minimum-variance unbiased (MVU) estimator.

    /// note | Different Notation
    In the lecture, we called the probing signal $s[n]$ and the coefficients $h[i]$.
    To be consistent with the illustration above, we call the probing signal $x[n]$ and the coefficients $b_i$ in this notebook.
    ///
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Probing Signals

    For the following simulation, you can select a probing signal $x[n]$ and see how it effects the estimate and Cramer-Rao lower bound/variances of the estimates $\hat{b}_i$.

    Recall from the lecture that there are two desirable properties for a good probing signal $x[n]$:
    1. High energy (large $\sum_{n=0}^{N-1} x^2[n]$)
    2. Orthogonal to itself when shifted ($\sum_{n=0}^{N-1} x[n-i] x[n-j] \approx 0$ for all $i\neq j$)

    In the following, you can select one of five pre-defined probing signals with different properties:

    | Signal | Notable Properties | Observation | Property of Interest from Above |
    |--------|--------------------|-------------|---------------------------------|
    | Sine | Sine wave (simple and deterministic signal) | Baseline (medium variance of estimation, no decoupling) | --- |
    | Sine (High Energy) | Scaled sine wave with larger amplitude | Lower variance due to higher energy. No decoupling | 1 (high energy) |
    | Random Binary | (Pseudo-)Random sequence of 0s and 1s | Orthogonal to itself $\rightarrow$ decoupling. Good performance, even with low signal energy | 2 (orthogonal) |
    | Random Binary (High Energy) | (Pseudo-)Random sequence of 0s and 2s | Orthogonal to itself and higher energy $\rightarrow$ even better performance | 1 (high energy) and 2 (orthogonal) |
    | Pulse | Extremely simple structure. Would be great in the noiseless case | Bad performance | --- |
    """)
    return


@app.cell
def _(dd_probing_signal):
    dd_probing_signal
    return


@app.cell
def _(
    filter_coeff,
    md_estimation,
    md_signal_properties,
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

    mo.hstack(
        [
            mo.vstack([mo.md(md_signal_properties), mo.mpl.interactive(_fig)]),
            mo.md(md_estimation),
        ],
        widths=[2.5, 1],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Observations

    There are a few observations you should be able to make.

    1. Increasing the signal energy reduces the variance.
    2. The random sequences are (almost) orthogonal to themselves when shifted. The CRB matrix is almost diagonal, i.e., the parameters decouple.
      - Related to the decoupling: For the random input, the variances of the individual coefficients are almost the same. For the sine waves, the ones for $\hat{b}_1$ and $\hat{b}_2$ are siginificantly larger.
    3. The random sequences have a very good performance with very little energy.
      - At about the same energy as the pulse, the binary random sequence achieves a siginificantly lower estimation variance.
      - The higher energy random signal has a fraction of the high energy sine wave but a better performance
    """)
    return


@app.cell
def _(dd_probing_signal, filter_coeff, linalg, noise, np):
    probing_signal = dd_probing_signal.value
    probing_signal_name = dd_probing_signal.selected_key

    probing_matrix = linalg.toeplitz(
        probing_signal, np.zeros(len(filter_coeff))
    )
    output_signal = probing_matrix @ filter_coeff + noise

    est_filter_coeff = (
        linalg.inv(probing_matrix.T @ probing_matrix)
        @ probing_matrix.T
        @ output_signal
    )
    crb_filter_coeff = linalg.inv(probing_matrix.T @ probing_matrix)
    var_estimates = np.diag(crb_filter_coeff)
    return (
        crb_filter_coeff,
        est_filter_coeff,
        output_signal,
        probing_signal,
        probing_signal_name,
        var_estimates,
    )


@app.cell
def _(mo, np, num_timeslots, t):
    dd_probing_signal = mo.ui.dropdown(
        options={
            "Sine": 2 * np.sin(t),
            "Sine (High Energy)": 10 * np.sin(t),
            "Random Binary": np.random.randint(2, size=num_timeslots),
            "Random Binary (High Energy)": 2
            * np.random.randint(2, size=num_timeslots),
            # "Random Gaussian": 0.5 * np.random.randn(num_timeslots),
            # "Random Gaussian (High Energy)": np.random.randn(num_timeslots),
            "Pulse": np.where(
                np.arange(num_timeslots) < num_timeslots // 2, 1, 0
            ),
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
def _(
    crb_filter_coeff,
    est_filter_coeff,
    filter_coeff,
    np,
    probing_signal,
    probing_signal_name,
    var_estimates,
):
    _table_rows = [
        f"| {h:.2f} | {e:.2f} | {v:.3f} |"
        for h, e, v in zip(filter_coeff, est_filter_coeff, var_estimates)
    ]
    _table_body = "\n".join(_table_rows)
    _crb_matrix_rows = [
        " & ".join([f"{_k:.2f}" for _k in _row]) for _row in crb_filter_coeff
    ]
    _crb_matrix_body = r"\\".join(_crb_matrix_rows)
    md_estimation = f"""
    ## Estimation Results

    The true and estimated filter coefficients are shown in the following table together with the variances of the estimates (which are also shown in the third plot).

    | Coefficient $b$ | Estimate $\\hat{{b}}$ | Variance of $\\hat{{b}}$ |
    |----------------:|----------------------:|-------------------------:|
    {_table_body}


    Below, you can find the full CRB matrix.  
    The important observation is that it is (almost) a diagonal matrix for the random input signal.

    \\begin{{equation*}}
    \\mathrm{{CRB}} = \\begin{{pmatrix}}
    {_crb_matrix_body}
    \\end{{pmatrix}}
    \\end{{equation*}}
    """

    md_signal_properties = f"""
    Selected probing signal: {probing_signal_name}  
    Signal energy: $\\sum_{{n=0}}^{{N-1}} x[n]^2 = {np.sum(probing_signal**2):.2f}$
    """
    return md_estimation, md_signal_properties


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import linalg, signal
    import matplotlib.pyplot as plt

    return linalg, mo, np, plt


if __name__ == "__main__":
    app.run()
