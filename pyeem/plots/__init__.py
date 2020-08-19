import matplotlib.pyplot as plt
from pkg_resources import resource_filename

from .augmentations import (
    mixture_animation,
    plot_prototypical_spectra,
    single_source_animation,
)
from .base import _colorbar, eem_plot
from .preprocessing import plot_calibration_curves, plot_preprocessing

pyeem_base_style = resource_filename("pyeem.plots", "pyeem_base.mplstyle")
plt.style.use(pyeem_base_style)

__all__ = [
    "mixture_animation",
    "plot_prototypical_spectra",
    "single_source_animation",
    "eem_plot",
    "plot_calibration_curves",
    "plot_preprocessing"
]
