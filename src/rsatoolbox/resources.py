"""Provide access to resource files distributed with rsatoolbox 
"""
from __future__ import annotations
from typing import TYPE_CHECKING
try:
    from importlib.resources import files, as_file
except ImportError:
    from importlib_resources import files, as_file
if TYPE_CHECKING:
    from pathlib import Path


def get_style() -> Path:
    """Returns the location of the mplstyle file for rsatoolbox
    """
    ref = files('rsatoolbox') / 'vis/rdm.mplstyle'
    with as_file(ref) as fpath:
        return fpath
