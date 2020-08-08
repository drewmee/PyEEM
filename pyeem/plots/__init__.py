import os

import matplotlib.pyplot as plt
from pkg_resources import resource_filename

from .augmentations import (
    mix_animation,
    plot_prototypical_spectra,
    single_source_animation,
)
from .base import combined_surface_contour_plot, contour_plot, surface_plot
from .preprocessing import plot_calibration_curves, plot_preprocessing, raman_area

pyeem_base_style = resource_filename("pyeem.plots", "pyeem_base.mplstyle")
plt.style.use(pyeem_base_style)
