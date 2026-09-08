import marimo

__generated_with = "0.24.0"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # Bayesian Detection: Prior, Likelihood, Posterior

    _Author:_ Karl-Ludwig Besser (Linköping University, Sweden)

    We observe a noisy grayscale image of a six-sided dice roll or a digit.
    The goal is to decide the true value based on the noisy observation.
    This notebook illustrates the roles of the prior distribution, the likelihood, and the posterior in Baysian detection (for maximum likelihood (ML) and maximum a-posteriori (MAP) detector).

    ## Hypotheses and Bayes' Rule

    There are six/ten hypotheses $\mathcal{H}_i$ for dice roll and digits, respectively.
    The prior probabilities $\Pr(\mathcal{H}_i)$ quantify our belief about each outcome (hypothesis) _before_ seeing the image.
    The likelihood $p(y\mid \mathcal{H}_i)$ measures how "compatible" the observed image $y$ is with each hypothesis.
    The posterior $\Pr(\mathcal{H}_i \mid y)$ is our updated belief about each hypothesis _after_ seeing the image.

    The MAP detector chooses the hypothesis with the largest posterior probability.
    The ML detector ignores the prior and chooses the hypothesis with the largest likelihood.
    """)
    return


@app.cell
def _(
    button_run,
    checkbox_reveal,
    clean_image,
    cmap,
    log_likelihoods,
    map_decision,
    map_index,
    ml_decision,
    mo,
    np,
    num_faces,
    observation,
    plt,
    posterior_probs,
    priors,
    radio_example,
    slider_noise_var,
    sliders_priors,
    text_priors,
    x_values,
):
    # Main 2x3 visualization
    fig, _axs = plt.subplots(2, 3)
    fig.tight_layout(pad=2.0)

    # Noisy observation
    _ax = _axs[0, 1]
    _ax.imshow(observation, cmap=cmap)
    _ax.axis("off")
    _ax.set_title("Noisy observation")

    # Clean image
    _ax = _axs[0, 0]
    _ax.axis("off")
    if checkbox_reveal.value:
        _ax.imshow(clean_image, cmap=cmap)
        _ax.set_title("Clean Image")

    # Priors
    _ax = _axs[1, 0]
    _ax.bar(x_values, priors, color="blue")
    _ax.set_title("Prior")
    _ax.set_xlabel("Die face")
    _ax.set_ylabel(r"$\Pr(\mathcal{H}_i)$")
    _ax.set_xticks(x_values)
    _ax.grid(axis="y", alpha=0.3)
    # for _idx, _val in enumerate(priors, start=1):
    #    _ax.text(_idx, _val, f"{_val:.3f}", ha="center", fontsize=8)

    # (Normalized) log-likelihoods
    _ax = _axs[1, 1]
    _ax.bar(x_values, log_likelihoods - np.min(log_likelihoods), color="green")
    _ax.set_title("Log-likelihoods")
    _ax.set_xlabel("Die face")
    _ax.set_ylabel(r"$\log p(y\mid \mathcal{H}_i)$")
    _ax.set_xticks(x_values)
    _ax.grid(axis="y", alpha=0.3)
    # for _idx, _val in enumerate(log_likelihoods, start=1):
    #    _ax.text(_idx, _val + 0.02, f"{_val:.3f}", ha="center", fontsize=8)

    # Posteriors
    _ax = _axs[1, 2]
    _colors = ["orange"] * num_faces
    _colors[map_index] = "red"
    _ax.bar(x_values, posterior_probs, color=_colors, log=True)
    _ax.set_title("Posterior")
    _ax.set_xlabel("Die face")
    _ax.set_ylabel(r"$\Pr(\mathcal{H}_i \mid y)$")
    _ax.set_xticks(x_values)
    _ax.grid(axis="y", alpha=0.3)
    # for _idx, _val in enumerate(posterior_probs, start=1):
    #    _ax.text(_idx, _val + 0.01, f"{_val:.3f}", ha="center", fontsize=8)

    # Detector decisions
    _ax = _axs[0, 2]
    _ax.axis("off")
    _text = f"ML decision: {ml_decision}\nMAP decision: {map_decision}"
    _ax.text(0.5, 0.1, _text, ha="center", va="center", fontsize=12)

    mo.hstack(
        [
            mo.mpl.interactive(fig),
            mo.vstack(
                (
                    radio_example,
                    button_run,
                    sliders_priors,
                    mo.md(text_priors),
                    slider_noise_var,
                    checkbox_reveal,
                )
            ),
        ],
        widths=[2, 1],
    )
    return


@app.cell
def _(
    log_likelihoods,
    map_decision,
    ml_decision,
    mo,
    posterior_probs,
    priors,
    sigma,
    squared_distances,
    x_values,
):
    # Numerical summary table
    _rows = ""
    for _idx, _value in enumerate(x_values):
        _rows += (
            f"| {_value} | {priors[_idx]:.3f} | {squared_distances[_idx]:.1f} | "
            f"{log_likelihoods[_idx]:.3f} | {posterior_probs[_idx]:.3f} |\n"
        )
    _md_table = fr"""
    | Face/Digit | Prior | Squared distance $\|y-a_i\|^2$ | Log-likelihood | Posterior |
    |:---:|:---:|:---:|:---:|:---:|
    {_rows}
    """

    _md_values = f"""
    - Noise variance: **{sigma**2:.3f}**
    - ML decision: **{ml_decision}**
    - MAP decision: **{map_decision}**
    - Sum of priors: **{priors.sum():.4f}**
    - Sum of posterior probabilities: **{posterior_probs.sum():.4f}**
    """

    mo.vstack(
        (
            mo.md("## Numerical Summary"),
            mo.hstack((mo.md(_md_table), mo.md(_md_values)), widths=[1, 1]),
        )
    )
    return


@app.cell
def _():
    import marimo as mo
    import numpy as np
    from scipy import stats, special
    import matplotlib.pyplot as plt

    return mo, np, plt, special


@app.cell
def _(np):
    cmap = "gray"
    base_seed = 42

    rng = np.random.default_rng(base_seed)
    return cmap, rng


@app.cell
def _(create_dice_template, create_digit_template, np, radio_example):
    if radio_example.value == "Dice Roll":
        num_faces = 6
        template_function = create_dice_template
        x_values = np.arange(1, num_faces + 1)
    elif radio_example.value == "Digits":
        num_faces = 10
        template_function = create_digit_template
        x_values = np.arange(num_faces)
    return num_faces, template_function, x_values


@app.cell
def _(np, slider_noise_var):
    noise_var = slider_noise_var.value
    sigma = np.sqrt(noise_var)
    return (sigma,)


@app.cell
def _(np, sliders_priors):
    # Normalize prior weights to a valid probability distribution
    priors = np.array(sliders_priors.value)
    priors = np.concatenate((priors, [1 - np.sum(priors)]))

    if np.any(priors < 0):
        text_priors = f"""
        /// danger | Invalid prior probabilities!

        The selected prior probabilities are not a valid probability distribution: {priors}.
        ///
        """
        print(f"Invalid prior probabilities: {priors}")
        # raise ValueError(f"Invalid prior probabilities: {priors}")
    else:
        _priors = ", ".join([f"{k:.2f}" for k in priors])
        text_priors = f"""
        Selected prior probabilities: ({_priors})
        """
        print(f"Prior probabilities: {priors}")
    return priors, text_priors


@app.cell
def _(mo):
    # Static Sliders and UI Items

    slider_noise_var = mo.ui.slider(
        0.01, 5.0, 0.01, 1, label=r"Noise variance $\sigma^2$"
    )

    checkbox_reveal = mo.ui.checkbox(
        label="Reveal the clean image",
        value=False,
    )

    radio_example = mo.ui.radio(
        options=["Dice Roll", "Digits"],
        value="Dice Roll",
        label="Select the example",
    )
    return checkbox_reveal, radio_example, slider_noise_var


@app.cell
def _(mo, num_faces):
    sliders_priors = mo.ui.array(
        [mo.ui.slider(0, 1, 0.02, 1 / num_faces)] * (num_faces - 1),
        label="Prior Probabilities (the last one is calculated through 1-sum(priors))",
    )
    return (sliders_priors,)


@app.cell
def _(mo):
    button_run = mo.ui.button(label="Draw a new random sample")
    return (button_run,)


@app.cell
def _(np):
    def create_dice_template(face, image_size=31, pip_radius=2, spacing=None):
        """Create a clean grayscale template for a die face."""
        face = face + 1
        if spacing is None:
            spacing = image_size // 4
        template = np.ones((image_size, image_size))
        center = image_size // 2
        # Pip locations relative to center
        pip_offsets = {
            1: [(0, 0)],
            2: [(-1, -1), (1, 1)],
            3: [(-1, -1), (0, 0), (1, 1)],
            4: [(-1, -1), (-1, 1), (1, -1), (1, 1)],
            5: [(-1, -1), (-1, 1), (0, 0), (1, -1), (1, 1)],
            6: [(-1, -1), (0, -1), (1, -1), (-1, 1), (0, 1), (1, 1)],
        }
        for row_off, col_off in pip_offsets[face]:
            row_center = center + row_off * spacing
            col_center = center + col_off * spacing
            rows = np.arange(image_size)
            cols = np.arange(image_size)
            dist = np.sqrt(
                (rows[:, None] - row_center) ** 2
                + (cols[None, :] - col_center) ** 2
            )
            template[dist <= pip_radius] = 0.0
        return template

    return (create_dice_template,)


@app.cell
def _(np):
    def create_digit_template(digit, image_size=31, scale=None):
        """
        Create a clean grayscale template for a handwritten digit (0-9).
        Uses a 5x7 pixel bitmap scaled up to the desired image size.
        Background is white (1.0), digit strokes are black (0.0).
        """
        # 5x7 bitmaps for digits 0-9 (5 columns, 7 rows)
        digit_bitmaps = {
            0: ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
            1: ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
            2: ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
            3: ["11111", "00010", "00100", "00010", "00001", "10001", "01110"],
            4: ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
            5: ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
            6: ["00110", "01000", "10000", "11110", "10001", "10001", "01110"],
            7: ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
            8: ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
            9: ["01110", "10001", "10001", "01111", "00001", "00010", "01100"],
        }

        bitmap = np.array(
            [
                [1.0 if ch == "0" else 0.0 for ch in row]
                for row in digit_bitmaps[digit]
            ]
        )

        # Determine scale factor
        if scale is None:
            scale = max(1, image_size // 7)  # 7 rows in bitmap
        scaled = np.kron(bitmap, np.ones((scale, scale)))

        # Pad or crop to exactly image_size x image_size
        scaled_size = scaled.shape[0]
        if scaled_size < image_size:
            # Pad with white (1.0)
            pad_total = image_size - scaled_size
            pad_before = pad_total // 2
            pad_after = pad_total - pad_before
            template = np.pad(
                scaled,
                ((pad_before, pad_after), (pad_before, pad_after)),
                constant_values=1.0,
            )
        elif scaled_size > image_size:
            # Center crop
            start = (scaled_size - image_size) // 2
            template = scaled[
                start : start + image_size, start : start + image_size
            ]
        else:
            template = scaled

        return template

    return (create_digit_template,)


@app.cell
def _(np, num_faces, template_function):
    templates = np.stack([template_function(i) for i in range(num_faces)])
    return (templates,)


@app.cell
def _(np, observation, priors, sigma, special, templates):
    # Likelihood calculation
    squared_distances = (
        np.linalg.norm(observation - templates, axis=(1, 2)) ** 2
    )
    log_likelihoods = -squared_distances / (2 * sigma**2) - np.size(
        observation
    ) / 2 * np.log(2 * np.pi * sigma**2)
    _unnormal_posteriors = np.log(priors) + log_likelihoods
    _normalization = special.logsumexp(
        _unnormal_posteriors
    )  # np.sum(np.exp(_unnormal_posteriors))
    posterior_probs = np.exp(_unnormal_posteriors - _normalization)
    return log_likelihoods, posterior_probs, squared_distances


@app.cell
def _(clean_image, rng, sigma):
    noise = rng.normal(0, sigma, size=clean_image.shape)
    observation = clean_image + noise
    return (observation,)


@app.cell
def _(log_likelihoods, np, posterior_probs, x_values):
    # Detection decisions
    ml_index = np.argmax(log_likelihoods)
    map_index = np.argmax(posterior_probs)
    ml_decision = x_values[ml_index]
    map_decision = x_values[map_index]
    return map_decision, map_index, ml_decision


@app.cell
def _(button_run, num_faces, rng, templates):
    button_run
    true_face_index = rng.choice(num_faces)  # , p=priors)
    clean_image = templates[true_face_index]
    return (clean_image,)


if __name__ == "__main__":
    app.run()
