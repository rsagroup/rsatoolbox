"""Mapping data files in a Brain Imaging Data Structure (BIDS) layout.
"""
from typing import List, Optional
from rsatoolbox.io.optional import import_nibabel


class BidsFile:
    pass


class BidsLayout:

    def __init__(self, path: str):
        pass

    def find_derivative_files(self,
            derivative: str,
            desc: str,
            tasks: Optional[List[str]]=None,
            space: Optional[str]=None,
            ) -> List[BidsFile]:
        return []
