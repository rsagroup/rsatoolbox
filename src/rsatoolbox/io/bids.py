"""Mapping data files in a Brain Imaging Data Structure (BIDS) layout.
"""
from __future__ import annotations
from glob import glob
import json, os
from os.path import join, isdir, relpath, normpath, basename
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from rsatoolbox.io.optional import import_nibabel
import pandas
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame


class BidsFile:

    _meta: Optional[Dict]
    sub: str
    ses: Optional[str]
    run: Optional[str]
    task: Optional[str]
    modality: str
    derivative: Optional[str]

    def __init__(self, relpath: str, layout: BidsLayout) -> None:
        self.relpath = relpath
        self.layout = layout
        self._meta = None
        self._deconstruct()

    def _deconstruct(self):
        parts = normpath(self.relpath).split(os.sep)
        fname = basename(self.relpath)
        if parts[0] == 'derivatives':
            self.derivative = parts[1]
            parts = parts[2:]
        else:
            self.derivative = None
        sub = self._findEntity('sub', fname)
        if sub is None:
            raise ValueError(f'Missing sub entity in bids filename: {fname}')
        self.sub = sub
        self.ses = self._findEntity('ses', fname)
        self.run = self._findEntity('run', fname)
        self.task = self._findEntity('task', fname)
        if self.ses:
            self.modality = parts[2]
        else:
            self.modality = parts[1]

    def _findEntity(self, entity: str, in_fname: str) -> Optional[str]:
        in_fname.split('_')
        for ent_seg in in_fname.split('_'):
            if ent_seg.startswith(f'{entity}-'):
                return ent_seg.replace(f'{entity}-', '')

    @property
    def fpath(self) -> str:
        return self.layout.abs_path(self)
    
    def get_meta(self) -> Dict:
        if self._meta is None:
            with open(self.fpath.replace('.nii.gz', '.json')) as fhandle:
                self._meta = json.load(fhandle)
        return self._meta
    
    def get_sibling(self, desc: str) -> BidsFile:
        ## get file with same entities except DESC
        pass


class BidsTableFile(BidsFile):

    def get_frame(self) -> DataFrame:
        return pandas.read_csv(self.fpath, sep='\t')


class BidsMriFile(BidsFile):

    def __init__(self, relpath: str, layout: BidsLayout, nibabel) -> None:
        self.nibabel = nibabel
        super().__init__(relpath, layout)

    def get_data(self) -> NDArray:
        return self.nibabel.load(self.fpath).get_fdata()
    
    def get_events(self) -> DataFrame:
        return self.layout.find_events_for(self).get_frame()
    
    def get_mri_sibling(self, desc: str) -> BidsMriFile:
        return


class BidsLayout:

    _path: str
    _nibabel: Optional[Any]

    def __init__(self, path: str, nibabel: Optional[Any]=None):
        self._path = path
        self._nibabel = nibabel

    def abs_path(self, file: BidsFile) -> str:
        return join(self._path, file.relpath)

    def find_events_for(self, base: BidsFile) -> BidsTableFile:
        path_segs = [f'sub-{base.sub}']
        if base.ses is not None:
            path_segs += [f'ses-{base.ses}']
        path_segs += [base.modality]
        fname_segs = [f'sub-{base.sub}']
        if base.ses is not None:
            fname_segs += [f'ses-{base.ses}']
        if base.task is not None:
            fname_segs += [f'task-{base.task}']
        if base.run is not None:
            fname_segs += [f'run-{base.run}']
        fname_segs += ['events.tsv']
        path_segs += ['_'.join(fname_segs)]
        return BidsTableFile(join(*path_segs), self)

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
