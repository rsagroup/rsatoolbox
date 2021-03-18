#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 13:47:59 2021

@author: heiko
"""

from allensdk.core.brain_observatory_cache import BrainObservatoryCache
import allensdk.brain_observatory.natural_scenes as ns
import allensdk.brain_observatory.stimulus_info as stim_info
import numpy as np
import pandas as pd
import os


boc = BrainObservatoryCache(manifest_file='boc/manifest.json')
nwb_filename = 'boc/ophys_experiment_data/%d.nwb'


def download(exp_id, folder='allen_data'):
    """ downloads a specific experiment and extracts the mean cell responses
    from the df/f traces for n frames after the exp_id"""
    if not os.path.isdir(folder):
        os.mkdir(folder)
    filename = os.path.join(folder, 'U_%d.npz' % exp_id)
    if not os.path.isfile(filename):
        exp_data = boc.get_ophys_experiment_data(exp_id)
        exp_ana = ns.NaturalScenes(exp_data)
        stim_table = exp_ana.stim_table
        stimulus = stim_table['frame']
        t_dff, dff = exp_data.get_dff_traces()
        U = np.empty((len(stim_table), exp_data.number_of_cells))
        for i in range(len(stim_table)):
            resp = dff[:, (stim_table['start'][i]+1):(stim_table['end'][i])]
            U[i] = np.mean(resp, 1)
        np.savez(filename, stimulus=stimulus, U=U)
        # remove file after U download to save space
        os.remove(nwb_filename % exp_id)
    else:
        d_dict = np.load(filename)
        stimulus = d_dict['stimulus']
        U = d_dict['U']
    return U.shape


def download_all(folder='allen_data'):
    csv_file = folder + '.csv'
    if not os.path.isfile(csv_file):
        experiments = boc.get_ophys_experiments(
            stimuli=[stim_info.NATURAL_SCENES],
            cre_lines=['Cux2-CreERT2', 'Emx1-IRES-Cre', 'Slc17a7-IRES2-Cre'])
        exp_df = pd.DataFrame(experiments)
        exp_df.to_csv(csv_file)
    else:
        exp_df = pd.read_csv(csv_file)
    order = np.random.permutation(len(exp_df))
    exp_df['n_stim'] = np.nan
    exp_df['n_cell'] = np.nan
    for idx in order:
        n_stim, n_cell = download(exp_df['id'][idx], folder=folder)
        print('downloaded %d: %d' % (idx, exp_df['id'][idx]))
        exp_df.at[idx, 'n_stim'] = n_stim
        exp_df.at[idx, 'n_cell'] = n_cell
    exp_df.to_csv(csv_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--folder', type=str,
                        help='where should the allen data be?',
                        default='allen_data')
    parser.add_argument('action', help='what to do?', type=str,
                        choices=['download', 'nothing'],
                        default='nothing', nargs='?')
    args = parser.parse_args()
    if args.action == 'download':
        download_all(args.folder)
    else:
        print('No action selected, I am done!')
