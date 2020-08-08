from pkg_resources import get_distribution

from . import augmentation, datasets, instruments, preprocessing, plots

"""
pyeem_base_style = resource_filename(
    "pyeem.plots.styles", "pyeem_base.mplstyle"
)
STYLE_PATH = realpath(join(*[dirname(__file__), "plots/styles/pyeem.mplstyle"]))
print(os.getcwd())
print(STYLE_PATH)
plt.style.use(pyeem_base_style)
"""

__all__ = [
    "datasets",
    "instruments",
    "preprocessing",
    "augmentation",
    "plots",
]

__version__ = get_distribution("pyeem").version
