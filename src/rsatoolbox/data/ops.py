"""Operations on multiple Datasets
"""
from __future__ import annotations
from typing import TYPE_CHECKING, Union, List, Set, overload
from copy import deepcopy
from warnings import warn
try:
    from typing import Literal  # pylint: disable=ungrouped-imports
except ImportError:
    from typing_extensions import Literal
from numpy import concatenate, repeat
import rsatoolbox
if TYPE_CHECKING:
    DESC_LEVEL = Union[Literal['obs'], Literal['set']]
    from rsatoolbox.data.dataset import Dataset, TemporalDataset


@overload
def merge_datasets(sets: List[TemporalDataset]) -> TemporalDataset:
    ...


@overload
def merge_datasets(sets: List[Dataset]) -> Dataset:
    ...


def merge_datasets(sets: Union[List[Dataset], List[TemporalDataset]]
                   ) -> Union[Dataset, TemporalDataset]:
    """Concatenate measurements to create one Dataset of the same type

    Only descriptors that exist on all subsets are assigned to the merged
    dataset.
    Dataset-level `descriptors` that are identical across subsets will be
    passed on, those that vary will become `obs_descriptors`.
    Channel and Time descriptors must be identical across subsets.

    Args:
        sets (Union[List[Dataset], List[TemporalDataset]]): List of Dataset
            or TemporalDataset objects. Must all be the same type.

    Returns:
        Union[Dataset, TemporalDataset]: The new dataset combining measurements
            and descriptors from the given subset datasets.
    """
    if len(sets) == 0:
        warn('[merge_datasets] Received empty list, returning empty Dataset')
        return rsatoolbox.data.dataset.Dataset(measurements=[])
    if len({type(s) for s in sets}) > 1:
        raise ValueError('All datasets must be of the same type')
    ds0 = sets[0]
    # numpy pre-allocates so this seems to be a performant solution:
    meas = concatenate([ds.measurements for ds in sets], axis=0)
    obs_descs = dict()
    # loop over obs descriptors that all subsets have in common:
    for k in _shared_descriptors(sets, 'obs'):
        obs_descs[k] = concatenate([ds.obs_descriptors[k] for ds in sets])
    dat_decs = dict()
    for k in _shared_descriptors(sets):
        if len({s.descriptors[k] for s in sets}) == 1:
            # descriptor always has the same value
            dat_decs[k] = ds0.descriptors[k]
        else:
            # descriptor varies across subsets, so repeat it by observation
            obs_descs[k] = repeat(
                [ds.descriptors[k] for ds in sets],
                [ds.n_obs for ds in sets]
            )
    # order is important as long as TemporalDataset inherits from Dataset
    if isinstance(ds0, rsatoolbox.data.dataset.TemporalDataset):
        return rsatoolbox.data.dataset.TemporalDataset(
            measurements=meas,
            descriptors=dat_decs,
            obs_descriptors=obs_descs,
            channel_descriptors=deepcopy(ds0.channel_descriptors),
            time_descriptors=deepcopy(ds0.time_descriptors),
        )
    if isinstance(ds0, rsatoolbox.data.dataset.Dataset):
        return rsatoolbox.data.dataset.Dataset(
            measurements=meas,
            descriptors=dat_decs,
            obs_descriptors=obs_descs,
            channel_descriptors=deepcopy(ds0.channel_descriptors)
        )
    raise ValueError('Unsupported Dataset type')


def _shared_descriptors(
        datasets: Union[List[Dataset], List[TemporalDataset]],
        level: DESC_LEVEL = 'set') -> Set[str]:
    """Find descriptors that all datasets have in common
    """
    if level == 'set':
        each_keys = [set(d.descriptors.keys()) for d in datasets]
    else:
        each_keys = [set(d.obs_descriptors.keys()) for d in datasets]
    return set.intersection(*each_keys)
