"""Operations on multiple Datasets
"""
from __future__ import annotations
from typing import Union, List, overload
from rsatoolbox.data.dataset import Dataset, TemporalDataset


@overload
def merge_datasets(sets: List[TemporalDataset]) -> TemporalDataset:
    ...


@overload
def merge_datasets(sets: List[Dataset]) -> Dataset:
    ...


def merge_datasets(sets: Union[List[Dataset], List[TemporalDataset]]
                   ) -> Union[Dataset, TemporalDataset]:
    if len(set([type(s) for s in sets])) > 1:
        raise ValueError('All datasets must be of the same type')
    if isinstance(sets[0], Dataset):
        return Dataset([])
    elif isinstance(sets[0], TemporalDataset):
        return TemporalDataset([])
    else:
        raise ValueError('Unsupported Dataset type')
