"""
saving to and reading from files
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Union, IO


def remove_file(file: Union[str, Path, IO]):
    """ Deletes file from OS if it exists

    Args:
        file (str, Path):
            a filename or opened readable file
    """
    if isinstance(file, (str, Path)) and os.path.exists(file):
        os.remove(file)
    elif hasattr(file, 'name') and os.path.exists(file.name):
        file.truncate(0)
