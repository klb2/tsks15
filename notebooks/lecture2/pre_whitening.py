import marimo

__generated_with = "0.14.13"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""# Pre-Whitening""")
    return


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
            mo.ui.slider(-3, 3, 0.1, -1, label="$b_x$"),
            mo.ui.slider(-3, 3, 0.1, -2, label="$b_y$"),
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
def _(b, grid, np, slider_cov_matrix, stats):
    _var_x, _var_y, _corr = slider_cov_matrix.value
    _cov = _corr * np.sqrt(_var_x * _var_y)
    cov_mat = np.array([[_var_x, _cov], [_cov, _var_y]])
    rv_h1_orig = stats.multivariate_normal(mean=b, cov=cov_mat)
    pdf_orig = rv_h1_orig.pdf(grid)
    return cov_mat, pdf_orig


@app.cell
def _(b, cov_mat, grid, np, stats):
    _eigval, _eigvec = np.linalg.eigh(cov_mat)
    whitening_mat = _eigvec @ (np.identity(2) / np.sqrt(_eigval)) @ _eigvec.T

    b_transform = whitening_mat @ b[:, np.newaxis]
    b_transform = np.ravel(b_transform)
    rv_h1_transform = stats.multivariate_normal(
        mean=b_transform, cov=np.identity(2)
    )
    pdf_transform = rv_h1_transform.pdf(grid)
    return b_transform, pdf_transform


@app.cell
def _(slider_cov_matrix):
    slider_cov_matrix
    return


@app.cell
def _(X, Y, b, b_transform, mo, pdf_orig, pdf_transform, plt):
    _fig, _axs = plt.subplots(1, 2, squeeze=True)
    _ax1, _ax2 = _axs
    _ax1.set_title("Likelihood under $\\mathcal{H}_1$ with center $b$")
    _ax1.contourf(X, Y, pdf_orig)
    _ax1.scatter(*b, c="k")
    _ax1.set_xlabel("First component $y[1]$")
    _ax1.set_ylabel("Second component $y[2]$")
    _ax2.set_title("Likelihood under $\\mathcal{H}_1$ with center $b'$")
    _ax2.contourf(X, Y, pdf_transform)
    _ax2.scatter(*b_transform, c="k")
    _ax2.set_xlabel("Transformed first component $y'[1]$")
    _ax2.set_ylabel("Transformed second component $y'[2]$")
    _fig.tight_layout()
    mo.mpl.interactive(_fig)
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


if __name__ == "__main__":
    app.run()
