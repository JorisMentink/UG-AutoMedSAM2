import re
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets

from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from IPython.display import display, clear_output


def interactive_inference_viewer(
    data,
    seg_handler,
    prompter=None,
    title="UG-AutoMedSAM2",
    window_percentiles=None,
    vmin=None,
    vmax=None,
    point_size=40,
):
    """
    Display segmentations, recontours, uncertainty, and compiled prompts.

    Volumes must have shape (slices, height, width).
    Prompt coordinates must use original image space in (x, y) order.

    MRI display:
        window_percentiles=None -> full volume intensity range.
        window_percentiles=(1, 99) -> percentile contrast windowing.
        vmin / vmax -> explicit limits overriding the calculated limits.

    All controls are displayed together without tabs.
    """

    image = np.asarray(data.img)
    if image.ndim != 3:
        raise ValueError("Expected image shape (slices, height, width).")

    n_slices, height, width = image.shape

    def check_volume(value, name):
        if value is None:
            return None
        value = np.asarray(value)
        if value.shape != image.shape:
            raise ValueError(
                f"{name}: shape {value.shape} differs from image {image.shape}."
            )
        return value

    def checkbox(label, value=False, disabled=False):
        return widgets.Checkbox(
            description=label,
            value=value,
            disabled=disabled,
            indent=False,
            layout=widgets.Layout(width="240px"),
        )

    def row(items):
        return widgets.Box(
            children=list(items),
            layout=widgets.Layout(
                display="flex",
                flex_flow="row wrap",
                align_items="center",
            ),
        )

    def section(label, children):
        return widgets.VBox(
            [widgets.HTML(f"<b>{label}</b>"), *children],
            layout=widgets.Layout(
                border="1px solid #dddddd",
                padding="8px 12px",
                margin="0 0 6px 0",
                width="100%",
            ),
        )

    # ------------------------------------------------------------
    # Segmentation and recontour layers
    # ------------------------------------------------------------
    layers = {}
    segmentation_checks = {}
    recontour_checks = {}

    def add_layer(name, volume, colour, group, visible=False):
        volume = check_volume(volume, name)
        if volume is None:
            return
        layers[name] = (volume, colour)
        group[name] = checkbox(name, visible)

    add_layer(
        "nnU-Net", data.mask, "#ff9900",
        segmentation_checks, True,
    )
    add_layer(
        "UG-AutoMedSAM2",
        getattr(seg_handler, "predicted_seg", None),
        "#3399ff", segmentation_checks, True,
    )
    add_layer(
        "Ground truth", getattr(data, "gt", None),
        "#cc99ff", segmentation_checks,
    )
    add_layer(
        "Consensus", getattr(data, "consensus_recontour", None),
        "#33dd55", recontour_checks, True,
    )

    observers = getattr(data, "observer_recontours", None)
    observer_names = getattr(data, "observer_names", None)

    def observer_label(name):
        text = str(name).strip()
        match = re.fullmatch(
            r"(?:(?:observer|obs)[\s:_-]*)?([a-z])",
            text,
            flags=re.IGNORECASE,
        )
        if match:
            return f"Observer {match.group(1).upper()}"
        return (
            text if text.lower().startswith("observer ")
            else f"Observer {text}"
        )

    if observers is not None:
        if isinstance(observers, dict):
            observer_items = list(observers.items())
        else:
            observer_items = [
                (
                    observer_names[i]
                    if observer_names is not None and i < len(observer_names)
                    else str(i + 1),
                    mask,
                )
                for i, mask in enumerate(observers)
            ]

        # Distinct shades of red, from light to dark.
        colours = plt.cm.Reds(
            np.linspace(0.35, 0.85, max(len(observer_items), 1))
        )

        for (name, mask), colour in zip(observer_items, colours):
            add_layer(
                observer_label(name), mask, colour, recontour_checks
            )

    layer_checks = {**segmentation_checks, **recontour_checks}

    # ------------------------------------------------------------
    # Uncertainty
    # ------------------------------------------------------------
    uncertainty = check_volume(
        getattr(data, "unc_map", None), "Raw uncertainty"
    )
    envelope = check_volume(
        getattr(prompter, "unc_map_bin", None),
        "Uncertainty envelope",
    )

    unc_max = 0.5
    if uncertainty is not None:
        finite = uncertainty[np.isfinite(uncertainty)]
        if finite.size:
            unc_max = max(float(finite.max()), 1e-6)

    uncertainty_check = checkbox(
        "Raw uncertainty", disabled=uncertainty is None
    )
    envelope_check = checkbox(
        "Uncertainty envelope", disabled=envelope is None
    )

    opacity = widgets.FloatSlider(
        value=0.4, min=0, max=1, step=0.05,
        description="Opacity",
        continuous_update=False,
    )
    uncertainty_range = widgets.FloatRangeSlider(
        value=(0, unc_max),
        min=0, max=unc_max, step=unc_max / 100,
        description="Colour range",
        readout_format=".3f",
        continuous_update=False,
        disabled=uncertainty is None,
    )

    # ------------------------------------------------------------
    # Actual compiled prompt sets used by Segmentation
    # ------------------------------------------------------------
    # Prefer compiled inference prompts when available.
    compiled_sets = (
        getattr(seg_handler, "compiled_prompt_sets", None) or {}
    )

    # Otherwise, read the prompts stored by UG_prompter.
    if not compiled_sets:
        generated_prompts = getattr(prompter, "prompts_by_slice", None)

        if generated_prompts:
            compiled_sets = {
                "Generated prompts": (generated_prompts, None)
            }

    prompt_sets = {}
    prompt_set_checks = {}

    for name, (prompt_dict, weight) in compiled_sets.items():
        prompt_sets[name] = {
            int(z): prompt for z, prompt in prompt_dict.items()
        }
        weight_text = (
            "" if weight is None else f" (w={float(weight):g})"
        )
        prompt_set_checks[name] = checkbox(
            f"{name}{weight_text}",
            value=weight is None or weight > 0,
        )

    points_check = checkbox("Point prompts", True)
    boxes_check = checkbox("Bounding boxes", True)
    dense_check = checkbox("Dense-mask prompts", False)

    # ------------------------------------------------------------
    # Navigation and MRI intensity window
    # ------------------------------------------------------------
    baseline = np.asarray(data.mask, dtype=bool)
    occupied = np.flatnonzero(baseline.any(axis=(1, 2)))
    initial_slice = (
        int(occupied[len(occupied) // 2])
        if occupied.size else n_slices // 2
    )

    ys, xs = np.where(baseline.any(axis=0))
    centre_x = (
        float((xs.min() + xs.max()) / 2)
        if xs.size else (width - 1) / 2
    )
    centre_y = (
        float((ys.min() + ys.max()) / 2)
        if ys.size else (height - 1) / 2
    )

    finite_image = image[np.isfinite(image)]
    if not finite_image.size:
        raise ValueError("The image contains no finite values.")

    if window_percentiles is None:
        image_min = float(finite_image.min())
        image_max = float(finite_image.max())
    else:
        percentiles = np.asarray(window_percentiles, dtype=float)
        if (
            percentiles.shape != (2,)
            or not np.all(np.isfinite(percentiles))
            or not 0 <= percentiles[0] < percentiles[1] <= 100
        ):
            raise ValueError(
                "window_percentiles must contain two increasing "
                "percentiles between 0 and 100."
            )
        image_min, image_max = np.percentile(
            finite_image, percentiles
        )

    if vmin is not None:
        image_min = float(vmin)
    if vmax is not None:
        image_max = float(vmax)

    if not np.isfinite(image_min) or not np.isfinite(image_max):
        raise ValueError("vmin and vmax must be finite.")
    if image_max < image_min:
        raise ValueError("vmax must be greater than or equal to vmin.")
    if image_max == image_min:
        image_max = image_min + max(abs(image_min), 1.0) * 1e-6

    slice_slider = widgets.IntSlider(
        value=initial_slice,
        min=0, max=n_slices - 1,
        description="Slice",
        continuous_update=False,
    )
    zoom = widgets.FloatSlider(
        value=1, min=1, max=5, step=0.25,
        description="Zoom",
        continuous_update=False,
    )
    output = widgets.Output()

    # ------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------
    def redraw(change=None):
        z = slice_slider.value

        with output:
            clear_output(wait=True)
            fig, ax = plt.subplots(figsize=(10, 9))
            legend = {}

            try:
                ax.imshow(
                    image[z],
                    cmap="gray",
                    vmin=image_min,
                    vmax=image_max,
                    origin="upper",
                    interpolation="nearest",
                )

                if uncertainty_check.value and uncertainty is not None:
                    low, high = uncertainty_range.value
                    heatmap = np.ma.masked_where(
                        ~np.isfinite(uncertainty[z])
                        | (uncertainty[z] <= 0),
                        uncertainty[z],
                    )
                    artist = ax.imshow(
                        heatmap,
                        cmap="magma",
                        alpha=opacity.value,
                        vmin=low,
                        vmax=max(high, low + 1e-9),
                        origin="upper",
                        interpolation="nearest",
                    )
                    fig.colorbar(
                        artist, ax=ax,
                        fraction=0.035, pad=0.02,
                        label="Uncertainty",
                    )

                if envelope_check.value and envelope is not None:
                    rgba = np.zeros((height, width, 4))
                    rgba[..., :3] = (1.0, 0.85, 0.0)
                    rgba[..., 3] = (
                        envelope[z].astype(bool) * opacity.value
                    )
                    ax.imshow(
                        rgba,
                        origin="upper",
                        interpolation="nearest",
                    )
                    legend["Uncertainty envelope"] = Line2D(
                        [], [], color="gold", lw=5,
                        label="Uncertainty envelope",
                    )

                def draw_contour(mask, name, colour, linestyle="-"):
                    binary = np.asarray(mask) > 0
                    if binary.shape != (height, width):
                        raise ValueError(
                            f"{name}: expected {(height, width)}, "
                            f"got {binary.shape}. "
                            "Use masks in original image space."
                        )
                    if not binary.any():
                        return

                    padded = np.pad(binary.astype(float), 1)
                    ax.contour(
                        np.arange(-1, width + 1),
                        np.arange(-1, height + 1),
                        padded,
                        levels=[0.5],
                        colors=[colour],
                        linewidths=1.5,
                        linestyles=linestyle,
                    )
                    legend[name] = Line2D(
                        [], [], color=colour,
                        linestyle=linestyle, label=name,
                    )

                for name, (volume, colour) in layers.items():
                    if layer_checks[name].value:
                        draw_contour(volume[z], name, colour)

                point_count = 0
                box_count = 0
                dense_count = 0
                present_sets = []

                for set_name, prompt_dict in prompt_sets.items():
                    if not prompt_set_checks[set_name].value:
                        continue

                    prompt = prompt_dict.get(z)
                    if not prompt:
                        continue

                    present_sets.append(str(set_name))

                    if (
                        points_check.value
                        and prompt.get("points") is not None
                    ):
                        points = np.asarray(
                            prompt["points"]
                        ).reshape(-1, 2)
                        labels = np.asarray(
                            prompt["point_labels"]
                        ).reshape(-1)

                        if len(points) != len(labels):
                            raise ValueError(
                                f"{set_name}: point/label counts differ."
                            )

                        for label, colour, marker, name in (
                            (1, "lime", "+", "Positive prompt"),
                            (0, "#ff5555", "x", "Negative prompt"),
                        ):
                            selected = points[labels == label]
                            if selected.size:
                                ax.scatter(
                                    selected[:, 0], selected[:, 1],
                                    c=colour,
                                    marker=marker,
                                    s=point_size,
                                    linewidths=1.4,
                                    zorder=10,
                                )
                                point_count += len(selected)
                                legend[name] = Line2D(
                                    [], [], color=colour,
                                    marker=marker,
                                    markersize=6,
                                    linestyle="None",
                                    label=name,
                                )

                    if (
                        boxes_check.value
                        and prompt.get("bbox") is not None
                    ):
                        boxes = np.asarray(
                            prompt["bbox"]
                        ).reshape(-1, 4)

                        for x0, y0, x1, y1 in boxes:
                            ax.add_patch(Rectangle(
                                (x0, y0), x1 - x0, y1 - y0,
                                fill=False,
                                edgecolor="white",
                                linewidth=1.7,
                                linestyle="--",
                                zorder=9,
                            ))
                            box_count += 1

                        if len(boxes):
                            legend["Bounding box"] = Line2D(
                                [], [], color="white",
                                linestyle="--",
                                label="Bounding box",
                            )

                    if (
                        dense_check.value
                        and prompt.get("mask_input") is not None
                    ):
                        dense = np.asarray(
                            prompt["mask_input"]
                        ).squeeze()

                        # > 0 supports binary masks and signed logits.
                        draw_contour(
                            dense,
                            f"Dense prompt: {set_name}",
                            "white",
                            ":",
                        )
                        dense_count += 1

                spacing = getattr(data, "img_spacing", None)
                if spacing is not None:
                    ax.set_aspect(
                        float(spacing[1]) / float(spacing[2])
                    )

                view_width = width / zoom.value
                view_height = height / zoom.value

                left = np.clip(
                    centre_x - view_width / 2,
                    -0.5, width - 0.5 - view_width,
                )
                top = np.clip(
                    centre_y - view_height / 2,
                    -0.5, height - 0.5 - view_height,
                )

                ax.set_xlim(left, left + view_width)
                ax.set_ylim(top + view_height, top)
                ax.set_title(
                    f"{title} | Slice {z} / {n_slices - 1}"
                )
                ax.axis("off")

                if legend:
                    # Dark background keeps white prompt symbols visible.
                    legend_artist = ax.legend(
                        handles=list(legend.values()),
                        loc="upper left",
                        bbox_to_anchor=(1.15, 1),
                        frameon=True,
                        facecolor="#252525",
                        edgecolor="#555555",
                        framealpha=1.0,
                    )
                    for text in legend_artist.get_texts():
                        text.set_color("white")

                if not prompt_sets:
                    print(
                        "No compiled prompt sets found on seg_handler."
                    )
                elif not present_sets:
                    print(
                        "No prompts from the enabled sets on this slice."
                    )
                else:
                    print(
                        "Prompt sets on this slice: "
                        + ", ".join(present_sets)
                    )
                    print(
                        f"Displayed: {point_count} points, "
                        f"{box_count} boxes, "
                        f"{dense_count} dense-mask prompts."
                    )

                display(fig)

            finally:
                plt.close(fig)

    # ------------------------------------------------------------
    # Grouped controls — all visible, no tabs
    # ------------------------------------------------------------
    prompt_rows = [
        row([points_check, boxes_check, dense_check]),
    ]
    if prompt_set_checks:
        prompt_rows.extend([
            widgets.HTML(
                "<span style='color: #777;'>Compiled prompt sets</span>"
            ),
            row(prompt_set_checks.values()),
        ])

    control_panel = widgets.VBox([
        section("Navigation", [
            row([slice_slider, zoom]),
        ]),
        section("Segmentations", [
            row(segmentation_checks.values()),
        ]),
        section("Recontours and consensus", [
            row(recontour_checks.values()),
        ]),
        section("Uncertainty", [
            row([uncertainty_check, envelope_check]),
            row([opacity, uncertainty_range]),
        ]),
        section("Prompts", prompt_rows),
    ], layout=widgets.Layout(width="100%", max_width="1050px"))

    controls = [
        slice_slider, zoom, opacity, uncertainty_range,
        uncertainty_check, envelope_check,
        points_check, boxes_check, dense_check,
        *layer_checks.values(),
        *prompt_set_checks.values(),
    ]
    for control in controls:
        control.observe(redraw, names="value")

    panel = widgets.VBox([
        control_panel,
        output,
    ])

    display(panel)
    redraw()
    return panel