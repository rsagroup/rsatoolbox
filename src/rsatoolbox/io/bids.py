"""Mapping data files in a Brain Imaging Data Structure (BIDS) layout.
"""
from __future__ import annotations
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from rsatoolbox.io.optional import import_nibabel
import numpy, pandas
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame
    import nibabel


class BidsFile:

    def get_events(self) -> DataFrame:
        return pandas.DataFrame()
    
    def get_meta(self) -> Dict:
        return dict()
    
    def get_sibling(self, desc: str) -> BidsFile:
        ## get file with same entities except DESC
        pass


class BidsMriFile(BidsFile):

    def __init__(self, nibabel) -> None:
        self.nibabel = nibabel

    def get_data(self) -> NDArray:
        return numpy.array([])
    
    def get_mri_sibling(self, desc: str) -> BidsMriFile:
        ## get file with same entities except DESC
        pass


class BidsLayout:

    def __init__(self, path: str):
        pass

    def find_mri_derivative_files(self,
            derivative: str,
            desc: str,
            tasks: Optional[List[str]]=None,
            space: Optional[str]=None,
            ) -> List[BidsMriFile]:
        return []
