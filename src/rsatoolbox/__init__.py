#!/usr/bin/env python3
# -*- coding: utf-8 -*-
""" Top level package: Only imports and organisation
"""

from importlib.metadata import version
__version__ = version("rsatoolbox")

from . import cengine
from . import data
from . import inference
from . import model
from . import rdm
from . import simulation
from . import util
from . import vis
