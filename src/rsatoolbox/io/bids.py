"""Mapping data files in a Brain Imaging Data Structure (BIDS) layout.
"""
from __future__ import annotations
from glob import glob
from os.path import join, isdir, isfile
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

    _path: str

    def __init__(self, path: str):
        self._path = path

    def find_mri_derivative_files(self,
            derivative: str,
            desc: str,
            tasks: Optional[List[str]]=None,
            ) -> List[BidsMriFile]:
        deriv_dir = join(self._path, 'derivatives', derivative)
        if not isdir(deriv_dir):
            raise ValueError(f'Derivative directory not found: {deriv_dir}')
        
        fpaths = glob(join(deriv_dir, '**', 'sub-*'), recursive=True)
        ## filter by DESC
        fpaths = [f for f in fpaths if f'desc-{desc}' in f]
        ## filter out meta files
        fpaths = [f for f in fpaths if not f.endswith('.json')]
        ## filter by TASK
        if tasks is not None:
            subset = []
            for task in tasks:
                subset += [f for f in fpaths if f'task-{task}' in f]
            fpaths = subset
        return fpaths
