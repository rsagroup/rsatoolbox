#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test that the operations in the demos are still functional
"""

import unittest


class TestDemos(unittest.TestCase):

    def test_example_dataset(self):
        import numpy as np
        import numpy.matlib as matlib
        import matplotlib.pyplot as plt
        import pyrsa
        import pyrsa.data as rsd  # abbreviation to deal with dataset

        # import the measurements for the dataset
        measurements = {'simTruePatterns': np.random.randn(92, 100)}
        measurements = measurements['simTruePatterns']
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]

        # plot the imported data
        plt.imshow(measurements, cmap='gray')
        plt.xlabel('Voxels')
        plt.ylabel('Conditions')
        plt.title('Measurements')

        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(
            ['cond_' + str(x) for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(
            ['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)
        print(data)

        # create an example dataset with random data, subset some conditions
        nChannel = 50
        nObs = 12
        randomData = np.random.rand(nObs, nChannel)
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5])}
        chn_des = {'voxels': np.array(
            ['voxel_' + str(x) for x in np.arange(nChannel)])}
        data = rsd.Dataset(measurements=randomData,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des
                           )
        # select a subset of the dataset: select data only from conditions 0:4
        sub_data = data.subset_obs(by='conds', value=[0, 1, 2, 3, 4])
        print(sub_data)

        # Split by channels
        nChannel = 3
        nChannelVox = 10  # three ROIs, each with 10 voxels
        nObs = 4
        randomData = np.random.rand(nObs, nChannel*nChannelVox)
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array([0, 1, 2, 3])}
        chn_des = matlib.repmat(['ROI1', 'ROI2', 'ROI3'], 1, nChannelVox)
        chn_des = {'ROIs': np.array(chn_des[0])}
        data = rsd.Dataset(
            measurements=randomData,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des)
        split_data = data.split_channel(by='ROIs')
        print(split_data)

        # create a datasets with random data
        nVox = 50  # 50 voxels/electrodes/measurement channels
        nCond = 10  # 10 conditions
        nSubj = 5  # 5 different subjects
        randomData = np.random.rand(nCond, nVox, nSubj)

        # We can then create a list of dataset objects
        # by appending each dataset for each subject.

        obs_des = {'conds': np.array(
            ['cond_' + str(x) for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(
            ['voxel_' + str(x) for x in np.arange(nVox)])}

        data = []  # list of dataset objects
        for i in np.arange(nSubj):
            des = {'session': 1, 'subj': i+1}
            # append the dataset object to the data list
            data.append(rsd.Dataset(
                measurements=randomData[:, :, 0],
                descriptors=des,
                obs_descriptors=obs_des,
                channel_descriptors=chn_des
                ))

    def test_example_dissimilarities(self):
        # relevant imports
        import numpy as np
        from scipy import io
        import pyrsa
        import pyrsa.data as rsd  # abbreviation to deal with dataset
        import pyrsa.rdm as rsr
        # create a dataset object
        measurements = {'simTruePatterns': np.random.randn(92, 100)}
        measurements = measurements['simTruePatterns']
        nCond = measurements.shape[0]
        nVox = measurements.shape[1]
        # now create a  dataset object
        des = {'session': 1, 'subj': 1}
        obs_des = {'conds': np.array(
            ['cond_' + str(x) for x in np.arange(nCond)])}
        chn_des = {'voxels': np.array(
            ['voxel_' + str(x) for x in np.arange(nVox)])}
        data = rsd.Dataset(measurements=measurements,
                           descriptors=des,
                           obs_descriptors=obs_des,
                           channel_descriptors=chn_des)
        # calculate an RDM
        RDM_euc = rsr.calc_rdm(data)
        RDM_corr = rsr.calc_rdm(data,
                                method='correlation', descriptor='conds')
        # create an RDM object
        rdm_des = {'RDM': np.array(['RDM_1'])}
        RDM_euc2 = rsr.RDMs(
            RDM_euc.dissimilarities,
            dissimilarity_measure=RDM_euc.dissimilarity_measure,
            descriptors=RDM_euc.descriptors,
            rdm_descriptors=rdm_des,
            pattern_descriptors=obs_des)
        print(RDM_euc.dissimilarities)  # here a vector
        dist_matrix = RDM_euc.get_matrices()
        print(dist_matrix)


if __name__ == '__main__':
    unittest.main()
