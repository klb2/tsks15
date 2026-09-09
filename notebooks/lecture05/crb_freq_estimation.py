import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Cramér-Rao Bound of a Simple Frequency Estimation

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebooks illustrates the Cramér-Rao lower bound (CRB) of estimating an unknown frequency.

    We observe $N$ measurements \[y[n]=\cos\left(2\pi f n\right) + w[n],\quad n=0, 1, \dots, N-1\] which are corrupted by iid Gaussian noise ${w[n]\sim\mathcal{N}(0, \sigma^2)}$.
    In the lecture, we used an amplitude $A$ and set $\sigma^2=1$ instead. However, both have the same effect of allowing us to tune the SNR.

    In the following, we illustrate the CRB and the influence of the true frequency $f$.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Illustration

    The upper plot shows the true signal (cosine function) and the measured signal (cosine + noise).
    The lower plot shows the CRB over the frequency and highlights the selected frequency $f$.
    (Note that the frequency slider uses a logarithmic scale, so you can easily change the frequency within the low frequency range.)

    You should be able to make the following observations when varying the different parameters:

    1. For a large number of samples $N$, the CRB decreases for all frequencies.
    $\rightarrow$ Taking more measurements improves the performance
    2. Similarly, reducing the noise variance $\sigma^2$ lowers the CRB.
    $\rightarrow$ Less noise improves the performance
    3. For low frequencies $f$, even the true signal $s[n; f]$ (without noise) looks almost like a flat line and it is difficult to distinguish between low frequencies.
    $\rightarrow$ Estimators will have a high variance when $f$ is small.

    /// admonition | Try the following settings

    1. $N=20$, $\sigma^2=0.1$, hide $s$
    2. Start with the smallest $f$ and then increase it a bit (until around $0.01$). You should essentially not see any difference in the measurements.
    3. Now jump to the highest frequency value and you should see a clear "wave-like" shape.
    4. Repeat without hiding $s$
    ///
    """)
    return


@app.cell
def _(
    checkbox_show_signal,
    crb_freq_estimation,
    freq,
    freq_line,
    md_params,
    measurements,
    mo,
    n,
    num_samples,
    plt,
    signal,
    slider_freq,
    slider_num_samples,
    slider_var_noise,
    var_noise,
):
    _fig, _axs = plt.subplots(2, 1)

    _ax = _axs[0]
    _ax.set_xlabel("Sample $n$")
    _ax.set_ylabel("Signals")
    _ax.set_ylim([-2.5, 2.5])
    _ax.plot(n, measurements, label=r"Measurements $y[n]$")
    if checkbox_show_signal.value:
        _ax.plot(n, signal, label=r"Signal $s[n; f]$")
    _ax.legend()

    _ax = _axs[1]
    _ax.set_xlabel("Frequency")
    _ax.set_ylabel("CRB")
    _crb_line = crb_freq_estimation(freq_line, var_noise, num_samples)
    _crb_selected_freq = crb_freq_estimation(freq, var_noise, num_samples)
    _ax.semilogy(freq_line, _crb_line)
    _ax.plot([freq], [_crb_selected_freq], "ro")

    _fig.set_tight_layout(True)

    mo.vstack(
        [
            mo.hstack(
                [
                    mo.vstack(
                        [
                            slider_freq,
                            slider_num_samples,
                            slider_var_noise,
                            checkbox_show_signal,
                        ],
                    ),
                    mo.md(md_params),
                ],
                widths=[0.5, 1],
            ),
            mo.mpl.interactive(_fig),
        ],
    )
    return


@app.cell
def _(freq, num_samples, var_noise):
    md_params = rf"""
    **Selected Parameters**

    $f = {freq:.2f}$  
    $N = {num_samples:d}$  
    $\sigma^2 = {var_noise:.2f}$
    """
    return (md_params,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt

    return mo, np, plt


@app.cell
def _(np):
    max_freq = 0.25
    freq_line = np.logspace(
        -4, np.log10(max_freq), 100
    )  # np.linspace(0.0005, max_freq, 100)
    return (freq_line,)


@app.cell
def _(freq_line, mo):
    slider_num_samples = mo.ui.slider(
        5, 50, 1, 10, label=r"Number of samples $N$"
    )
    # slider_num_samples = mo.ui.slider(
    #    steps=np.logspace(1, 5, 5), value=100, label=r"Number of samples $N$"
    # )

    slider_var_noise = mo.ui.slider(
        0.01, 1, 0.1, 0.1, label=r"Noise variance $\sigma^2$"
    )

    slider_freq = mo.ui.slider(
        steps=freq_line,
        value=freq_line[50],
        label=r"Frequency $f$",
        # 0.0005, max_freq, 0.005, 0.05, label=r"Frequency $f$"
    )

    checkbox_show_signal = mo.ui.checkbox(value=True, label=r"Show signal $s$")
    return (
        checkbox_show_signal,
        slider_freq,
        slider_num_samples,
        slider_var_noise,
    )


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
def _(n, np, slider_freq):
    freq = slider_freq.value
    signal = np.cos(2 * np.pi * freq * n)
    return freq, signal


@app.cell
def _(np, num_samples, std_noise):
    noise = std_noise * np.random.randn(num_samples)
    return (noise,)


@app.cell
def _(noise, signal):
    measurements = signal + noise
    return (measurements,)


@app.cell
def _(np):
    def crb_freq_estimation(freq, var_noise, num_samples):
        n = np.arange(num_samples)
        n = np.reshape(n, (-1, 1))
        return var_noise / (
            4
            * np.pi**2
            * np.sum((n * np.sin(2 * np.pi * freq * n)) ** 2, axis=0)
        )

    return (crb_freq_estimation,)


if __name__ == "__main__":
    app.run()
