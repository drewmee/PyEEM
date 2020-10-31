import matplotlib.pyplot as plt
from pkg_resources import resource_filename

from .analysis import model_history_plot, prediction_parity_plot
from .augmentations import (
    mixture_animation,
    prototypical_spectra_plot,
    single_source_animation,
)
from .base import _colorbar, eem_plot
from .preprocessing import (
    absorbance_plot,
    calibration_curves_plot,
    preprocessing_routine_plot,
    water_raman_peak_animation,
    water_raman_peak_plot,
    water_raman_timeseries,
)

pyeem_base_style = resource_filename("pyeem.plots", "pyeem_base.mplstyle")
plt.style.use(pyeem_base_style)

__all__ = [
    "eem_plot",
    "absorbance_plot",
    "water_raman_peak_plot",
    "water_raman_peak_animation",
    "water_raman_timeseries",
    "preprocessing_routine_plot",
    "calibration_curves_plot",
    "prototypical_spectra_plot",
    "single_source_animation",
    "mixture_animation",
    "model_history_plot",
    "prediction_parity_plot",
]
