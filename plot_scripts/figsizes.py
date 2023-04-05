_GOLDEN_RATIO = (5.0**0.5 - 1.0)/1.2
_POINTS_PER_INCH = 72.27

def neurips2022(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=2,
    constrained_layout=False,
    tight_layout=False,
    height_to_width_ratio=_GOLDEN_RATIO,
):
    """Neurips 2022 figure size."""
    return _neurips_common(
        rel_width=rel_width,
        nrows=nrows,
        ncols=ncols,
        constrained_layout=constrained_layout,
        tight_layout=tight_layout,
        height_to_width_ratio=height_to_width_ratio,
    )


def _neurips_common(
    *,
    rel_width=1.0,
    nrows=1,
    ncols=2,
    constrained_layout=True,
    tight_layout=False,
    height_to_width_ratio=_GOLDEN_RATIO,
):
    """Neurips figure size defaults."""

    figsize = _from_base_in(
        base_width_in=5.5 ,
        rel_width=rel_width,
        height_to_width_ratio=height_to_width_ratio,
        nrows=nrows,
        ncols=ncols,
    )
    return _figsize_to_output_dict(
        figsize=figsize,
        constrained_layout=constrained_layout,
        tight_layout=tight_layout,
    )

def _from_base_in(*, base_width_in, rel_width, height_to_width_ratio, nrows, ncols):
    width_in = base_width_in * rel_width
    subplot_width_in = width_in / ncols
    subplot_height_in = height_to_width_ratio * subplot_width_in
    height_in = subplot_height_in * nrows
    print(subplot_width_in)
    print(subplot_height_in)
    return 4, 2.8 #2.76, 2.266#4.125, 3.399#3.5, 3.3#

def _figsize_to_output_dict(*, figsize, constrained_layout, tight_layout):
    return {
        "figure.figsize": figsize,
        "figure.constrained_layout.use": constrained_layout,
        "figure.autolayout": tight_layout,
    }