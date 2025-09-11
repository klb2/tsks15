import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Fisher-Information Matrix and Cramer-Rao Bound: Localization Example

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    In this example, we are interested in localizing a target in a two-dimensional plane, i.e., estimating its $x$ and $y$-coordinates.
    For this, we have two fixed anchors $i$ at positions ${(-1, 0)}$ and ${(0, -1)}$, which can measure their distance $\mu_i$ to the target.
    Therefore, we have the following distance measurements

    \begin{align*}
    \mu_1 &= \sqrt{(x+1)^2 + y^2}\\
    \mu_2 &= \sqrt{x^2 + (y+1)^2}
    \end{align*}

    which we receive distorted by additive white Gaussian noise $w$ with noise power $\sigma^2$.
    The received signal $r$ is therefore given as

    \begin{equation*}
    \vec{r} = \vec{\mu} + \vec{w} = \begin{pmatrix}\mu_1\\ \mu_2\end{pmatrix} + \begin{pmatrix}w_1\\ w_2\end{pmatrix}
    \end{equation*}

    Based on the model, $r$ is distributed according to a jointly normal distribution with means $\mu_i$ and a scaled identity matrix $\sigma^2 I$ as covariance matrix, i.e., $r\sim\mathcal{N}(\vec{\mu}, \sigma^2 I)$.


    A visualization of this setup together with the calculated Fisher information matrix and Cramer-Rao bound can be found below.
    """
    )
    return


@app.cell
def _(
    R1,
    R2,
    md_matrices,
    mo,
    pdf_meas,
    plt,
    slider_coords,
    slider_sigma,
    x,
    y,
):
    _fig, _axs = plt.subplots(2, squeeze=True)
    _fig.set_tight_layout(True)
    _ax, _ax_pdf = _axs

    _ax.scatter([-1, 0], [0, -1], marker="^", c="r", label="Anchors")
    _ax.scatter([x], [y], c="b", label="Target")
    _ax.legend()
    _ax.set_xlim([-2, 1])
    _ax.set_ylim([-2, 1])
    _ax.grid(alpha=0.5)

    _ax_pdf.contourf(R1, R2, pdf_meas)
    _ax_pdf.set_xlabel("Distance from Anchor 1")
    _ax_pdf.set_ylabel("Distance from Anchor 2")

    mo.hstack(
        [
            mo.vstack([slider_coords, mo.mpl.interactive(_fig)]),
            mo.vstack([slider_sigma, mo.md(md_matrices)]),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(np, sigma, x, y):
    i_xx = (
        1
        / sigma
        * ((x + 1) ** 2 / ((x + 1) ** 2 + y**2) + x**2 / (x**2 + (y + 1) ** 2))
    )
    i_xy = (
        1
        / sigma
        * (
            ((x + 1) * y) / ((x + 1) ** 2 + y**2)
            + (x * (y + 1)) / (x**2 + (y + 1) ** 2)
        )
    )
    i_yy = (
        1
        / sigma
        * (y**2 / ((x + 1) ** 2 + y**2) + (y + 1) ** 2 / (x**2 + (y + 1) ** 2))
    )

    fisher_matrix = np.array([[i_xx, i_xy], [i_xy, i_yy]])
    try:
        crb_matrix = np.linalg.inv(fisher_matrix)
    except np.linalg.LinAlgError:
        crb_matrix = None
    return crb_matrix, fisher_matrix


@app.cell
def _(crb_matrix, fisher_matrix):
    md_fim = rf"""
    ## Fisher Information Matrix

    $$I(\theta) = \begin{{pmatrix}}{fisher_matrix[0, 0]:.3f} & {fisher_matrix[0, 1]:.3f}\\{fisher_matrix[1, 0]:.3f} & {fisher_matrix[1, 1]:.3f}\end{{pmatrix}}$$
    """

    if crb_matrix is None:
        md_crb = rf"""
    ## Cramer-Rao Bound

    The FIM is singular, i.e., not invertible. Therefore, the Cramer-Rao bound does not exist and the problem is unidentifiable for this particular constellation.

    Note that this only occurs if the target is on the line $x+y=-1$.
        """
    else:
        md_crb = rf"""
    ## Cramer-Rao Bound

    $$I^{{-1}}(\theta) = \begin{{pmatrix}}{crb_matrix[0, 0]:.3f} & {crb_matrix[0, 1]:.3f}\\{crb_matrix[1, 0]:.3f} & {crb_matrix[1, 1]:.3f}\end{{pmatrix}}$$
        """

    md_matrices = "\n\n".join((md_fim, md_crb))
    return (md_matrices,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


@app.cell
def _(grid, np, slider_coords, slider_sigma, stats):
    x, y = slider_coords.value
    sigma = slider_sigma.value

    mu = [np.sqrt((x + 1) ** 2 + y**2), np.sqrt(x**2 + (y + 1) ** 2)]
    rv_distances = stats.multivariate_normal(mean=mu, cov=sigma * np.identity(2))
    pdf_meas = rv_distances.pdf(grid)
    return pdf_meas, sigma, x, y


@app.cell
def _(mo):
    slider_x = mo.ui.slider(-1.5, 0.5, 0.05, -0.1, label="$x$ position")
    slider_y = mo.ui.slider(-1.5, 0.5, 0.05, -0.1, label="$y$ position")
    slider_coords = mo.ui.array([slider_x, slider_y], label="Target Coordinates")
    slider_sigma = mo.ui.slider(0.1, 2, 0.1, 1, label="Noise variance $\\sigma^2$")
    return slider_coords, slider_sigma


@app.cell
def _(np):
    r = np.linspace(0, 5, 150)
    R1, R2 = np.meshgrid(r, r)
    grid = np.dstack((R1, R2))
    return R1, R2, grid


if __name__ == "__main__":
    app.run()
