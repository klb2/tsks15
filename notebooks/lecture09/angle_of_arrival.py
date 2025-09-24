import marimo

__generated_with = "0.15.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Example: Angle of Arrival Estimation

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebook illustrates the example of estimating the angle of arrival $\psi$. In particular, we have a uniform linear array of $N$ sensors spaced at half-wavelength distance.

    The received signal at sensor $n$ is given as $$y[n] = A \cos\left(\pi n \sin\psi + \phi\right) + w[n]$$ where $w[n]$ is white Gaussian noise.
    The three parameters $(A, \phi, \psi)$ are unknown, however, only $\psi$ is of interest.
    """
    )
    return


@app.cell
def _(attenuation, mo, phi):
    mo.md(
        rf"""
    ## Parameters

    The attenuation $A$ and the phase offset $\phi$ are drawn randomly in this notebook.
    In the following, their values are:

    \begin{{align*}}
    A &= {attenuation:.3f}\\
    \phi &= {phi:.3f}
    \end{{align*}}
    """
    )
    return


@app.cell
def _(
    attenuation,
    freq,
    md_figure_description,
    mo,
    np,
    num_sensors,
    phi,
    plt,
    psi,
    ratio_x,
    slider_angle_psi,
    slider_num_sensors,
    slider_point_on_line,
    time,
):
    _fig, _axs = plt.subplots(2, 1, squeeze=True)
    _axs_sensors, _axs_signal = _axs

    _axs_sensors.axis("equal")
    _axs_sensors.set_xlim([-1, num_sensors])
    _axs_sensors.set_ylim([0, 1])
    # _axs_sensors.hlines(0, 0, num_sensors - 1, ls="--", color="gray")
    _axs_sensors.scatter(np.arange(num_sensors), np.zeros(num_sensors))
    _axs_sensors.plot([-1, 0], [-np.tan(np.pi / 2 + psi), 0], "--", c="gray")
    _axs_sensors.plot(
        [-1, 2], [np.tan(np.pi / 2 + psi) * (-1 - 2), 0], "--", c="gray"
    )
    _axs_sensors.plot(
        [0, num_sensors - 1], [0, np.tan(psi) * (num_sensors - 1)], c="g"
    )
    _axs_sensors.plot(
        [2, num_sensors - 1],
        [0, np.tan(psi) * (num_sensors - 1 - 2)],
        c="g",
        label="Wavefront",
    )
    _axs_sensors.scatter(
        [
            ratio_x * 2
            + (1 - ratio_x)
            * 2
            * np.tan(np.pi / 2 + psi)
            / (np.tan(np.pi / 2 + psi) - np.tan(psi))
        ],
        [
            (1 - ratio_x)
            * np.tan(psi)
            * 2
            * np.tan(np.pi / 2 + psi)
            / (np.tan(np.pi / 2 + psi) - np.tan(psi))
        ],
        c="k",
    )
    _axs_sensors.set_xlabel("Distance / $\\lambda/2$")
    _axs_sensors.legend()

    _axs_signal.plot(
        time,
        attenuation * np.cos(2 * np.pi * freq * time - phi),
        c="gray",
        label="$n=0$",
    )
    _axs_signal.plot(
        time,
        attenuation
        * np.cos(2 * np.pi * freq * time - 2 * np.pi * np.sin(psi) - phi),
        ls="--",
        c="gray",
        label="$n=2$",
    )
    _axs_signal.plot(
        time,
        attenuation
        * np.cos(
            2 * np.pi * freq * time - ratio_x * 2 * np.pi * np.sin(psi) - phi
        ),
        c="k",
        label="Traveling",
    )
    _axs_signal.set_xlabel("Time $t$")
    _axs_signal.set_ylabel("Received signal")
    _axs_signal.legend()
    _fig.tight_layout()
    mo.hstack(
        (
            mo.vstack([slider_point_on_line, mo.mpl.interactive(_fig)]),
            mo.vstack(
                [
                    mo.md(md_figure_description),
                    slider_angle_psi,
                    slider_num_sensors,
                ]
            ),
        ),
        widths=[1.75, 1],
    )
    return


@app.cell
def _(mo):
    mo.md(
        rf"""
    ## Estimation

    We can estimate the angle of arrival $\psi$ using two different approaches:

    - By jointly estimating the vector $\begin{{pmatrix}}\psi & \phi\end{{pmatrix}}^T$ as a solution to $$\begin{{pmatrix}}\hat{{\psi}} \\ \hat{{\phi}}\end{{pmatrix}} = \argmax_{{\psi, \phi}} \frac{{\left(\sum_{{n=0}}^{{N-1}} y[n] \cos\left(\pi n \sin\psi + \phi\right)\right)^2}}{{\sum_{{n=0}}^{{N-1}} \cos^2\left(\pi n \sin\psi + \phi\right)}}$$
    - By directly optimizing it through the reduced version $$\hat{{\psi}}_{{\text{{red}}}} = \argmax_{{\psi}} y^T H (H^T H)^{{-1}} H^T y$$
    """
    )
    return


@app.cell
def _(
    est_psi_red,
    md_red_optimization,
    md_results,
    mo,
    np,
    opt_func_psi_reduce,
    plt,
    res_est_psi_reduced,
):
    _fig, _axs = plt.subplots()
    _psi_line = np.linspace(0.01, np.pi / 2, 250)
    _psi_nls = np.array([-opt_func_psi_reduce(_p) for _p in _psi_line])
    _axs.plot(_psi_line, _psi_nls)
    _axs.scatter(est_psi_red, -res_est_psi_reduced.fun, c="orange")
    _axs.set_xlabel(r"Angle of arrival $\psi$")
    _fig.tight_layout()
    mo.hstack(
        [
            mo.mpl.interactive(_fig),
            mo.vstack([mo.md(md_red_optimization), mo.md(md_results)]),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(linalg, np, num_sensors, rec_signal):
    def opt_func_psi_reduce(x, y=rec_signal):
        if len(np.shape(y)) == 1:
            y = np.reshape(y, (-1, 1))
        h = np.array(
            [
                np.cos(np.pi * np.arange(num_sensors) * np.sin(x)),
                -np.sin(np.pi * np.arange(num_sensors) * np.sin(x)),
            ]
        ).T
        return np.sum(-y.T @ h @ linalg.pinv(h.T @ h) @ h.T @ y)
    return (opt_func_psi_reduce,)


@app.cell
def _(np, opt_func_psi_reduce, optimize, rec_signal):
    res_est_psi_reduced = optimize.dual_annealing(
        opt_func_psi_reduce,
        args=(np.reshape(rec_signal, (-1, 1)),),
        bounds=((0, np.pi / 2),),
    )
    est_psi_red = res_est_psi_reduced.x[0]
    return est_psi_red, res_est_psi_reduced


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import linalg, optimize
    import matplotlib.pyplot as plt
    return linalg, mo, np, optimize, plt


@app.cell
def _(np):
    time = np.linspace(0, 4, 250)
    freq = 1
    attenuation = 5 * np.random.rand()
    phi = 2 * np.pi * np.random.rand()
    return attenuation, freq, phi, time


@app.cell
def _(mo, np):
    slider_num_sensors = mo.ui.slider(3, 20, 1, 5, label="Number of sensors $N$")
    slider_angle_psi = mo.ui.slider(
        0, np.pi / 2, 0.01, np.pi / 4, label="Angle of arrival $\\psi$"
    )
    slider_point_on_line = mo.ui.slider(
        0, 1, 0.01, 0, label="Ratio of path difference"
    )
    return slider_angle_psi, slider_num_sensors, slider_point_on_line


@app.cell
def _(slider_num_sensors):
    num_sensors = slider_num_sensors.value
    return (num_sensors,)


@app.cell
def _(slider_angle_psi):
    psi = slider_angle_psi.value
    return (psi,)


@app.cell
def _(slider_point_on_line):
    ratio_x = slider_point_on_line.value
    return (ratio_x,)


@app.cell
def _(est_psi, est_psi_red, np, psi):
    md_red_optimization = r"""
    The objective function of the reduced estimation $\hat{{\psi}}_{{\text{{red}}}}$ is shown in this plot together with the (numerically found) maximum.
    Since this optimization problem is one-dimensional (only estimating $\psi$ without estimating $\phi$), it is a simple line that can be nicely illustrated.
    """

    md_results = rf"""
    ## Results

    In the following table, the true and estimated values are compared.

    | | True Value $\psi$ | Joint Estimation $\hat{{\psi}}$ | Reduced Estimation $\hat{{\psi}}_{{\text{{red}}}}$ |
    |:---|---:|---:|---:|
    | Value | ${psi:.3f}$ | ${est_psi:.3f}$ | ${est_psi_red:.3f}$ |
    | Error | ${np.abs(psi - psi):.3f}$ | ${np.abs(psi - est_psi):.3f}$ | ${np.abs(psi - est_psi_red):.3f}$ |
    """
    return md_red_optimization, md_results


@app.cell
def _():
    md_figure_description = """
    ## Illustration

    The figure illustrates the system setup with the $N$ sensors placed along the x-axis.
    Two wavefronts are shown together with their normals (dashed lines).

    The black dot illustrastes a position on the line that a wavefront needs to travel from the first sensor (at $x=0$) to the third sensor (at $x=2$).
    Its position can be moved through the slider above the figure.
    As you move the black dot, the wavefront travels and in the second plot, you can see the corresponding signals (without noise).
    """
    return (md_figure_description,)


@app.cell
def _(attenuation, np, num_sensors, phi, psi):
    noise = np.random.randn(num_sensors)
    rec_signal = (
        attenuation * np.cos(np.pi * np.arange(num_sensors) * np.sin(psi) - phi)
        + noise
    )
    return (rec_signal,)


@app.cell
def _(np, num_sensors, rec_signal):
    def opt_func_psi_phi(x, y=rec_signal):
        _num = (
            np.sum(
                y * np.cos(np.pi * np.arange(num_sensors) * np.sin(x[0]) + x[1])
            )
            ** 2
        )
        _den = np.sum(
            np.cos(np.pi * np.arange(num_sensors) * np.sin(x[0]) + x[1]) ** 2
        )
        return (
            -_num / _den
        )  # minus sign because scipy provides a minimization routine
    return (opt_func_psi_phi,)


@app.cell
def _(np, opt_func_psi_phi, optimize, rec_signal):
    res_est_psi_phi = optimize.minimize(
        opt_func_psi_phi,
        x0=(np.pi / 4, np.pi / 2),
        args=(rec_signal,),
        bounds=((0, np.pi / 2), (0, 2 * np.pi)),
    )
    est_psi, est_phi = res_est_psi_phi.x
    return (est_psi,)


if __name__ == "__main__":
    app.run()
