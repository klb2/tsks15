import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _(slider_mean1):
    slider_mean1
    return


@app.cell
def _(slider_mean2):
    slider_mean2
    return


@app.cell
def _(slider_threshold):
    slider_threshold
    return


@app.cell
def _(gamma, np, pdf1, pdf2, plt, prob_fa, prob_tp, rv1, rv2, x):
    _fig, _axs = plt.subplots(1, 2)
    _ax_pdf, _ax_roc = _axs
    _ax_pdf.plot(x, pdf1, label="$\\mathcal{N}(\\mu_1, 1)$", c="b")
    _ax_pdf.plot(x, pdf2, label="$\\mathcal{N}(\\mu_2, 1)$", c="orange")
    _ax_pdf.vlines(
        gamma, 0, 1.05 * np.maximum(np.max(pdf1), np.max(pdf2)), ls="--", color="k"
    )
    _idx_gamma = np.argmax(x > gamma)
    _ax_pdf.fill_between(x[_idx_gamma:], pdf1[_idx_gamma:], color="b", alpha=0.25)
    _ax_pdf.fill_between(
        x[:_idx_gamma], pdf2[:_idx_gamma], color="orange", alpha=0.25
    )

    _ax_roc.plot(prob_fa, prob_tp)
    _ax_roc.plot([rv2.cdf(gamma)], [rv1.cdf(gamma)], "o")
    _ax_roc.plot([0, 1], [0, 1], "k--")

    # mo.mpl.interactive(_fig)
    plt.show()
    return


@app.cell
def _(mo):
    slider_mean1 = mo.ui.slider(-2, -0.5, 0.1, -1, label="Mean $\\mu_1$")
    slider_mean2 = mo.ui.slider(0.5, 2, 0.1, 1, label="Mean $\\mu_2$")
    slider_threshold = mo.ui.slider(-3, 3, 0.1, 0, label="Threshold $\\gamma$")
    return slider_mean1, slider_mean2, slider_threshold


@app.cell
def _(slider_mean1, slider_mean2, slider_threshold):
    mean1 = slider_mean1.value
    mean2 = slider_mean2.value
    gamma = slider_threshold.value
    return gamma, mean1, mean2


@app.cell
def _(np):
    x = np.linspace(-5, 5, 500)
    return (x,)


@app.cell
def _(mean1, mean2, stats, x):
    rv1 = stats.norm(mean1)
    rv2 = stats.norm(mean2)
    pdf1 = rv1.pdf(x)
    pdf2 = rv2.pdf(x)
    prob_fa = rv2.cdf(x)
    prob_tp = rv1.cdf(x)
    return pdf1, pdf2, prob_fa, prob_tp, rv1, rv2


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats
    import matplotlib.pyplot as plt
    return mo, np, plt, stats


if __name__ == "__main__":
    app.run()
