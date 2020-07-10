"""
.. currentmodule:: pyeem.datasets

.. autosummary::
   :toctree: generated/datasets-api

   Load
   load_dreem
   load_rutherford
"""
from ._load import Load
from ._load import load_dreem
from ._load import load_rutherford

__all__ = ["Load", "load_dreem", "load_rutherford"]
