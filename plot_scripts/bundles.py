import figsizes, fonts, fontsizes


def neurips2022(*,  rel_width=1.0, nrows=1, ncols=1, family="serif", tight_layout = False):
    """Neurips 2022 bundle."""
    font_config = fonts.neurips2022(family=family)
    size = figsizes.neurips2022(rel_width=rel_width, nrows=nrows, ncols=ncols, tight_layout=tight_layout)
    fontsize_config = fontsizes.neurips2022()
    return {**font_config, **size, **fontsize_config}