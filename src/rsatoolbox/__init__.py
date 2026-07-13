#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Top level package: Only imports and organisation
"""

try:
    from importlib.metadata import version

    __version__ = version("mne")
except Exception:
    __version__ = "0.0.0"

from . import cengine
from . import data
from . import inference
from . import model
from . import rdm
from . import simulation
from . import util
from . import vis
