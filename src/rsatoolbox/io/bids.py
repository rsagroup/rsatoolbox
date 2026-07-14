"""Mapping data files in a Brain Imaging Data Structure (BIDS) layout.
"""
from __future__ import annotations
from glob import glob
import json
import os
from os.path import join, isdir, relpath, normpath, basename
from typing import TYPE_CHECKING, List, Optional, Dict, Any
from rsatoolbox.io.optional import import_nibabel
import pandas
if TYPE_CHECKING:
    from numpy.typing import NDArray
    from pandas import DataFrame


class BidsFile:

    _meta: Optional[BidsJsonFile]
    sub: Optional[str]
    ses: Optional[str]
    run: Optional[str]
    task: Optional[str]
    space: Optional[str]
    modality: Optional[str]
    derivative: Optional[str]
    desc: Optional[str]
    suffix: str
    ext: str

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
        self.sub = self._findEntity('sub', fname)
        self.ses = self._findEntity('ses', fname)
        self.run = self._findEntity('run', fname)
        self.task = self._findEntity('task', fname)
        self.space = self._findEntity('space', fname)
        if len(parts) > 1:
            if self.ses:
                self.modality = parts[2]
            else:
                self.modality = parts[1]
        self.desc = self._findEntity('desc', fname)
        suffix_ext = fname.split('_')[-1]
        self.suffix = suffix_ext.split('.')[0]
        self.ext = '.'.join(suffix_ext.split('.')[1:])

    def _findEntity(self, entity: str, in_fname: str) -> Optional[str]:
        in_fname.split('_')
        for ent_seg in in_fname.split('_'):
            if ent_seg.startswith(f'{entity}-'):
                return ent_seg.replace(f'{entity}-', '')

    @property
    def fpath(self) -> str:
        return self.layout.abs_path(self)

    def get_meta(self) -> Dict[str, Any]:
        if self._meta is None:
            self._meta = self.layout.find_meta_for(self)
        return self._meta.get_data()

    def get_table_sibling(self, desc: str, suffix: str) -> BidsTableFile:
        return self.layout.find_table_sibling_of(self, desc, suffix)


class BidsTableFile(BidsFile):

    def get_frame(self) -> DataFrame:
        return pandas.read_csv(self.fpath, sep='\t')


class BidsJsonFile(BidsFile):

    _data: Optional[Dict]

    def __init__(self, relpath: str, layout: BidsLayout) -> None:
        super().__init__(relpath, layout)
        self._data = None

    def get_data(self) -> Dict:
        if self._data is None:
            with open(self.fpath) as fhandle:
                self._data = json.load(fhandle)
        return self._data


class BidsMriFile(BidsFile):

    def __init__(self, relpath: str, layout: BidsLayout, nibabel) -> None:
        self.nibabel = nibabel
        super().__init__(relpath, layout)

    def get_data(self) -> NDArray:
        return self.nibabel.load(self.fpath).get_fdata()

    def get_events(self) -> DataFrame:
        return self.layout.find_events_for(self).get_frame()

    def get_mri_sibling(self, desc: str, suffix: str) -> BidsMriFile:
        return self.layout.find_mri_sibling_of(self, desc, suffix)

    def get_key(self) -> BidsTableFile:
        return self.layout.find_table_key_for(self)


class BidsLayout:

    _path: str
    _nibabel: Optional[Any]

    def __init__(self, path: str, nibabel: Optional[Any] = None):
        self._path = path
        self._nibabel = nibabel

    def abs_path(self, file: BidsFile) -> str:
        return join(self._path, file.relpath)

    def _replace(self, base: BidsFile, replace_entities: Dict) -> str:

        def replace_or_inherit(base: BidsFile, entity: str) -> Optional[str]:
            if entity in replace_entities:
                return replace_entities[entity]
            return getattr(base, entity)

        path_segs = []
        derivative = replace_or_inherit(base, 'derivative')
        path_segs += ['derivatives', derivative] if derivative else []
        sub = replace_or_inherit(base, 'sub')
        path_segs += [f'sub-{sub}'] if sub else []
        ses = replace_or_inherit(base, 'ses')
        path_segs += [f'ses-{ses}'] if ses else []
        modality = replace_or_inherit(base, 'modality')
        path_segs += [modality] if modality else []

        fname_segs = [f'sub-{sub}']
        fname_segs += [f'ses-{ses}'] if ses else []
        task = replace_or_inherit(base, 'task')
        fname_segs += [f'task-{task}'] if task else []
        run = replace_or_inherit(base, 'run')
        fname_segs += [f'run-{run}'] if run else []
        space = replace_or_inherit(base, 'space')
        fname_segs += [f'space-{space}'] if space else []
        desc = replace_or_inherit(base, 'desc')
        fname_segs += [f'desc-{desc}'] if desc else []
        suffix = replace_or_inherit(base, 'suffix')
        ext = replace_or_inherit(base, 'ext')
        fname_segs += [f'{suffix}.{ext}']
        path_segs += ['_'.join(fname_segs)]
        return join(*path_segs)

    def find_meta_for(self, base: BidsFile) -> BidsJsonFile:
        fpath = self._replace(base, dict(ext='json'))
        return BidsJsonFile(fpath, self)

    def find_events_for(self, base: BidsFile) -> BidsTableFile:
        fpath = self._replace(
            base,
            dict(
                derivative=None,
                space=None,
                desc=None,
                suffix='events',
                ext='tsv'
            )
        )
        return BidsTableFile(fpath, self)

    def find_table_key_for(self, base: BidsFile) -> BidsTableFile:
        path_segs = ['derivatives', base.derivative] if base.derivative else []
        path_segs += [f'desc-{base.desc}_{base.suffix}.tsv']
        return BidsTableFile(join(*path_segs), self)

    def find_table_sibling_of(self, base: BidsFile, desc: str, suffix: str
                              ) -> BidsTableFile:
        fpath = self._replace(
            base,
            dict(
                desc=desc,
                suffix=suffix,
                ext='tsv',
                space=None
            )
        )
        return BidsTableFile(fpath, self)

    def find_mri_sibling_of(self, base: BidsMriFile, desc: str, suffix: str
                            ) -> BidsMriFile:
        fpath = self._replace(base, dict(desc=desc, suffix=suffix))
        return BidsMriFile(fpath, self, self._nibabel)

    def find_mri_derivative_files(
                self,
                derivative: str,
                desc: str,
                tasks: Optional[List[str]] = None
            ) -> List[BidsMriFile]:
        deriv_dir = join(self._path, 'derivatives', derivative)
        if not isdir(deriv_dir):
            raise ValueError(f'Derivative directory not found: {deriv_dir}')

        fpaths = sorted(glob(join(deriv_dir, '**', 'sub-*'), recursive=True))
        # filter by DESC
        fpaths = [f for f in fpaths if f'desc-{desc}' in f]
        # filter out meta files
        fpaths = [f for f in fpaths if not f.endswith('.json')]
        # filter by TASK
        if tasks is not None:
            subset = []
            for task in tasks:
                subset += [f for f in fpaths if f'task-{task}' in f]
            fpaths = subset
        self._nibabel = import_nibabel(self._nibabel)
        return [BidsMriFile(relpath(f, self._path), self, self._nibabel)
                for f in fpaths]
