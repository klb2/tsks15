import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Linear Regression

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)

    We consider the linear model $A+Bn$ with unknown parameters $A$ and $B$.
    For this, we obtain $N$ noisy measurements $$y[n] = A + Bn + w[n]$$ with AWGN $w\sim\mathcal{N}(0, I)$ and $n=0, 1, \dots, N-1$.
    Based on these noisy measurements, we aim to estimate both parameters $A$ and $B$.
    """
    )
    return


@app.cell
def _(a, b, n, noise):
    y = a + b * n + noise
    return (y,)


@app.cell
def _(H, np, y):
    param_est = np.linalg.inv(H.T @ H) @ H.T @ y
    y_est = H @ param_est
    crb = np.linalg.inv(H.T @ H)
    return param_est, y_est


@app.cell
def _(mo):
    mo.md(
        r"""Using the interactive sliders below, you can vary the values of the underlying parameters $A$, $B$, and $N$."""
    )
    return


@app.cell
def _(
    md_estimation,
    mo,
    n,
    noise,
    plt,
    slider_a,
    slider_b,
    slider_n,
    y,
    y_est,
):
    _fig, _ax = plt.subplots()
    _fig.set_tight_layout(True)
    _ax.plot(n, y, "o-", label="$y$")
    _ax.plot(n, y_est, "o-", label=r"$\hat{y}$")
    _ax.plot(n, y - noise, "--", c="gray")
    _ax.legend()

    mo.hstack(
        [
            mo.vstack([slider_a, slider_b, slider_n, mo.mpl.interactive(_fig)]),
            mo.md(md_estimation),
        ],
        widths=[2, 1],
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import matplotlib.pyplot as plt
    return mo, np, plt


@app.cell
def _(mo):
    slider_a = mo.ui.slider(0, 5, 0.2, 1, label=r"Offset $A$")
    slider_b = mo.ui.slider(-3, 3, 0.2, 1, label=r"Slope $B$")
    slider_n = mo.ui.slider(2, 30, 1, 10, label=r"Number of Samples $N$")
    return slider_a, slider_b, slider_n


@app.cell
def _(np, slider_a, slider_b, slider_n):
    a = slider_a.value
    b = slider_b.value
    num_steps = slider_n.value
    n = np.arange(num_steps)
    noise = np.random.randn(num_steps)

    H = np.array([[1] * num_steps, np.arange(num_steps)]).T
    return H, a, b, n, noise


@app.cell
def _(a, b, np, param_est):
    md_estimation = rf"""
    ## Estimated Parameters

    For estimating the parameter vector $\theta=\begin{{pmatrix}}A \\B\end{{pmatrix}}$, we use the least squares fit as
    $$\hat{{\theta}} = (H^T H)^{{-1}} H^T y$$


    | Parameter | Estimation | Error |
    |----------:|-----------:|------:|
    | $A={a:.2f}$ | $\hat{{A}} = {param_est[0]:.2f}$ | $`|`\hat{{A}}-A`|` = {np.abs(param_est[0] - a):.3f}$ |
    | $B={b:.2f}$ | $\hat{{B}} = {param_est[1]:.2f}$ | $`|`\hat{{B}}-B`|` = {np.abs(param_est[1] - b):.3f}$ |


    The blue line in the plot shows the noisy measurements $y$.
    The orange line shows the curve with the estimated parameters, i.e., $\hat{{y}} = \hat{{A}} + \hat{{B}} n$.
    The dashed gray line indicates the true function $A+Bn$.
    """
    return (md_estimation,)


if __name__ == "__main__":
    app.run()
