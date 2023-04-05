

def _neurips_common(*, family="serif"):
    """Default fonts for Neurips."""
    return {
        "font.serif": ["Times New Roman"],
        "mathtext.fontset": "stix",  # free ptmx replacement, for ICML and NeurIPS
        "mathtext.rm": "Times New Roman",
        "mathtext.it": "Times New Roman:italic",
        "mathtext.bf": "Times New Roman:bold",
        "font.family": family,
    }

def neurips2022(*, family="serif"):
    """Fonts for Neurips 2022."""
    return _neurips_common(family=family)
