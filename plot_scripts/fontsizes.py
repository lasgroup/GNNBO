
def _from_base(*, base):
    return {
        "font.size": base - 1,
        "axes.labelsize": base - 1,
        "legend.fontsize": base - 3,
        "xtick.labelsize": base - 5,
        "ytick.labelsize": base - 5,
        "axes.titlesize": base - 2,
    }
def neurips2022():
    return  _from_base(base=13)
