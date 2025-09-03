import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(
        r"""
    # Pre-Whitening

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)


    This notebook illustrates the pre-whitening concept to deal with colored noise.
    In particular, we assume that the noise samples follow a jointly Gaussian distribution with known covariance matrix $\Gamma$.

    In the following, we consider a simple two-dimensional example.
    Under hypothesis $\mathcal{H}_1$, the observations $y$ are given by

    \begin{equation*}
    \begin{pmatrix}y_x\\ y_y\end{pmatrix} = \begin{pmatrix}b_x\\ b_y\end{pmatrix} + \begin{pmatrix}w_x\\ w_y\end{pmatrix}
    \end{equation*}

    where $w\sim\mathcal{N}(0, \Gamma)$ and $$\Gamma = \begin{pmatrix}\sigma^2_x & \rho\sigma_x\sigma_y\\ \rho\sigma_x\sigma_y & \sigma^2_y\end{pmatrix}$$
    """
    )
    return


@app.cell
def _(
    X,
    Y,
    b,
    b_transform,
    md_figure,
    mo,
    pdf_orig,
    pdf_transform,
    plt,
    slider_b,
    slider_cov_matrix,
):
    _fig, _axs = plt.subplots(1, 2, squeeze=True)
    _fig.set_tight_layout(True)
    _ax1, _ax2 = _axs
    _ax1.axis("equal")
    _ax2.axis("equal")
    _ax1.set_title("Likelihood under $\\mathcal{H}_1$ with center $b$")
    _ax1.contourf(X, Y, pdf_orig)
    _ax1.scatter(*b, c="k")
    _ax1.set_xlabel("First component $y_x$")
    _ax1.set_ylabel("Second component $y_y$")
    _ax2.set_title("Likelihood under $\\mathcal{H}_1$ with center $b'$")
    _ax2.contourf(X, Y, pdf_transform)
    _ax2.scatter(*b_transform, c="k")
    _ax2.set_xlabel("Transformed first component $y_x'$")
    _ax2.set_ylabel("Transformed second component $y_y'$")

    mo.hstack(
        [
            mo.mpl.interactive(_fig),
            mo.vstack([mo.md(md_figure), slider_b, slider_cov_matrix]),
        ],
        widths=[1.75, 1],
    )
    return


@app.cell
def _(b, b_transform, mo, str_cov_mat, str_whitening_mat):
    mo.md(
        rf"""
    ## Summary of Parameters

    In the original space, we have the center $b$ as

    \begin{{equation*}}
    b = \begin{{pmatrix}} {b[0]:.2f} \\ {b[1]:.2f} \end{{pmatrix}}
    \end{{equation*}}

    and covariance matrix

    \begin{{equation*}}
    \Gamma = \begin{{pmatrix}}{str_cov_mat}\end{{pmatrix}}
    \end{{equation*}}

    The transformation/whitening matrix is given as

    \begin{{equation*}}
    \Gamma^{{-1 / 2}} = \begin{{pmatrix}}{str_whitening_mat}\end{{pmatrix}}
    \end{{equation*}}

    which gives the center $b'$ in the transformed space as

    \begin{{equation*}}
    b' = \Gamma^{{-1 / 2}} b = \begin{{pmatrix}} {b_transform[0]:.2f} \\ {b_transform[1]:.2f} \end{{pmatrix}}
    \end{{equation*}}
    """
    )
    return


@app.cell
def _():
    md_figure = r"""
    In the left figure, you can see the density of the density of $y$.
    Its mean is $b$, and due to the correlation of $w_x$ and $w_y$, the iso-density lines are ellipses.

    The right figure shows the density of the transformed (whitened) variable $y'$ with mean $b'$. As the covariance is the identity, the iso-density lines are circles around $b'.$
    """
    return (md_figure,)


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


@app.cell
def _(np):
    x = np.linspace(-5, 5, 100)
    prob_fa = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, x)
    grid = np.dstack((X, Y))
    return X, Y, grid


@app.cell
def _(mo):
    slider_b = mo.ui.array(
        [
            mo.ui.slider(-3, 3, 0.1, 0, label="$b_x$"),
            mo.ui.slider(-3, 3, 0.1, 0, label="$b_y$"),
        ],
        label="Point $b$",
    )

    slider_cov_matrix = mo.ui.array(
        [
            mo.ui.slider(0.5, 5, 0.1, 1, label="Noise variance $\\sigma_x^2$"),
            mo.ui.slider(0.5, 5, 0.1, 1, label="Noise variance $\\sigma_y^2$"),
            mo.ui.slider(
                -0.9, 0.9, 0.01, 0.5, label="Correlation coefficient $\\rho$"
            ),
        ],
        label="Covariance matrix",
    )
    return slider_b, slider_cov_matrix


@app.cell
def _(np, slider_b):
    b = np.array(slider_b.value)
    return (b,)


@app.cell
def _(b, cov_mat, grid, np, stats):
    _eigval, _eigvec = np.linalg.eigh(cov_mat)
    whitening_mat = _eigvec @ (np.identity(2) / np.sqrt(_eigval)) @ _eigvec.T
    # alternatively, we could do: whitening_mat = np.linalg.inv(scipy.linalg.sqrtm(cov_mat))

    b_transform = whitening_mat @ b[:, np.newaxis]
    b_transform = np.ravel(b_transform)
    rv_h1_transform = stats.multivariate_normal(
        mean=b_transform, cov=np.identity(2)
    )
    pdf_transform = rv_h1_transform.pdf(grid)
    return b_transform, pdf_transform, whitening_mat


@app.cell
def _(b, grid, np, slider_cov_matrix, stats):
    _var_x, _var_y, _corr = slider_cov_matrix.value
    _cov = _corr * np.sqrt(_var_x * _var_y)
    cov_mat = np.array([[_var_x, _cov], [_cov, _var_y]])
    rv_h1_orig = stats.multivariate_normal(mean=b, cov=cov_mat)
    pdf_orig = rv_h1_orig.pdf(grid)
    return cov_mat, pdf_orig


@app.cell
def _(cov_mat, np, whitening_mat):
    str_cov_mat = (
        r"\\".join(
            np.array2string(_x, precision=2, separator=" & ") for _x in cov_mat
        )
        .replace("[", " ")
        .replace("]", " ")
    )

    str_whitening_mat = (
        r"\\".join(
            np.array2string(_x, precision=2, separator=" & ")
            for _x in whitening_mat
        )
        .replace("[", " ")
        .replace("]", " ")
    )
    return str_cov_mat, str_whitening_mat


if __name__ == "__main__":
    app.run()
