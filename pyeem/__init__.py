from pkg_resources import get_distribution

from . import augmentation, datasets, instruments, plots, preprocessing

__all__ = [
    "datasets",
    "instruments",
    "preprocessing",
    "augmentation",
    "plots",
]

__version__ = get_distribution("pyeem").version
