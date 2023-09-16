"""Mapping data files in a Brain Imaging Data Structure (BIDS) layout.
"""
from __future__ import annotations
from glob import glob
import json
from os.path import join, isdir, relpath
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from rsatoolbox.io.optional import import_nibabel
import numpy, pandas
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame


class BidsFile:

    _meta: Optional[Dict]

    def __init__(self, relpath: str, layout: BidsLayout) -> None:
        self.relpath = relpath
        self.layout = layout
        self._meta = None

    @property
    def fpath(self) -> str:
        return self.layout.abs_path(self)

    def get_events(self) -> DataFrame:
        return pandas.DataFrame()
    
    def get_meta(self) -> Dict:
        if self._meta is None:
            with open(self.fpath.replace('.nii.gz', '.json')) as fhandle:
                self._meta = json.load(fhandle)
        return self._meta
    
    def get_sibling(self, desc: str) -> BidsFile:
        ## get file with same entities except DESC
        pass


class BidsMriFile(BidsFile):

    def __init__(self, relpath: str, layout: BidsLayout, nibabel) -> None:
        self.nibabel = nibabel
        super().__init__(relpath, layout)

    def get_data(self) -> NDArray:
        return self.nibabel.load(self.fpath).get_fdata()
    
    def get_mri_sibling(self, desc: str) -> BidsMriFile:
        ## get file with same entities except DESC
        pass


class BidsLayout:

    _path: str
    _nibabel: Optional[Any]

    def __init__(self, path: str, nibabel: Optional[Any]=None):
        self._path = path
        self._nibabel = nibabel

    def abs_path(self, file: BidsFile) -> str:
        return join(self._path, file.relpath)

    def find_mri_derivative_files(self,
            derivative: str,
            desc: str,
            tasks: Optional[List[str]]=None
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
        nibabel = import_nibabel(self._nibabel)
        return [BidsMriFile(relpath(f, self._path), self, nibabel) for f in fpaths]
