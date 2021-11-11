#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
test that the operations in the demos are still functional
"""

import unittest


class TestDemos(unittest.TestCase):

    def test_example_dataset(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import rsatoolbox
        import rsatoolbox.data as rsd  # abbreviation to deal with dataset

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
        chn_des = ['ROI1', 'ROI2', 'ROI3'] * nChannelVox
        chn_des = {'ROIs': np.array(chn_des)}
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
        import rsatoolbox
        import rsatoolbox.data as rsd  # abbreviation to deal with dataset
        import rsatoolbox.rdm as rsr
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

    def test_exercise_all(self):
        import numpy as np
        from scipy import io
        import matplotlib.pyplot as plt
        import rsatoolbox
        import os
        path = os.path.dirname(os.path.abspath(__file__))
        matlab_data = io.matlab.loadmat(
            os.path.join(path, '..', 'demos',
                         'rdms_inferring', 'modelRDMs_A2020.mat'))
        matlab_data = matlab_data['modelRDMs']
        n_models = len(matlab_data[0])
        model_names = [matlab_data[0][i][0][0] for i in range(n_models)]
        measurement_model = [matlab_data[0][i][1][0] for i in range(n_models)]
        rdms_array = np.array([matlab_data[0][i][3][0]
                               for i in range(n_models)])
        model_rdms = rsatoolbox.rdm.RDMs(
            rdms_array,
            rdm_descriptors={'brain_computational_model': model_names,
                             'measurement_model': measurement_model},
            dissimilarity_measure='Euclidean'
            )

        conv1_rdms = model_rdms.subset('brain_computational_model', 'conv1')
        plt.figure(figsize=(10, 10))
        rsatoolbox.vis.show_rdm(conv1_rdms,
                           rdm_descriptor='measurement_model')

        conv1_rdms = model_rdms.subset('brain_computational_model', 'conv1')
        print(conv1_rdms)
        matlab_data = io.matlab.loadmat(
            os.path.join(path, '..', 'demos',
                         'rdms_inferring', 'noisyModelRDMs_A2020.mat'))
        repr_names_matlab = matlab_data['reprNames']
        fwhms_matlab = matlab_data['FWHMs']
        noise_std_matlab = matlab_data['relNoiseStds']
        rdms_matlab = matlab_data['noisyModelRDMs']
        repr_names = [repr_names_matlab[i][0][0]
                      for i in range(repr_names_matlab.shape[0])]
        fwhms = fwhms_matlab.squeeze().astype('float')
        noise_std = noise_std_matlab.squeeze().astype('float')
        rdms_matrix = rdms_matlab.squeeze().astype('float')

        i_rep = 2  # np.random.randint(len(repr_names))
        i_noise = 1  # np.random.randint(len(noise_std))
        i_fwhm = 0  # np.random.randint(len(fwhms))

        # print the chosen representation definition
        repr_name = repr_names[i_rep]
        print('The chosen ground truth model is:')
        print(repr_name)
        print('with noise level:')
        print(noise_std[i_noise])
        print('with averaging width (full width at half magnitude):')
        print(fwhms[i_fwhm])

        # put the rdms into an RDMs object and show it
        rdms_data = rsatoolbox.rdm.RDMs(
            rdms_matrix[:, i_rep, i_fwhm, i_noise, :].transpose())

        plt.figure(figsize=(10, 10))
        rsatoolbox.vis.show_rdm(rdms_data)

        models = []
        for i_model in np.unique(model_names):
            rdm_m = model_rdms.subset(
                'brain_computational_model', i_model).subset(
                    'measurement_model', 'complete')
            m = rsatoolbox.model.ModelFixed(i_model, rdm_m)
            models.append(m)

        print('created the following models:')
        for i in range(len(models)):
            print(models[i].name)

        results_1 = rsatoolbox.inference.eval_fixed(
            models, rdms_data, method='corr')
        rsatoolbox.vis.plot_model_comparison(results_1)

        results_2a = rsatoolbox.inference.eval_bootstrap_rdm(
            models, rdms_data, method='corr', N=10)
        rsatoolbox.vis.plot_model_comparison(results_2a)

        results_2b = rsatoolbox.inference.eval_bootstrap_pattern(
            models, rdms_data, method='corr', N=10)
        rsatoolbox.vis.plot_model_comparison(results_2b)

        results_2c = rsatoolbox.inference.eval_bootstrap(
            models, rdms_data, method='corr', N=10)
        rsatoolbox.vis.plot_model_comparison(results_2c)

        models_flex = []
        for i_model in np.unique(model_names):
            models_flex.append(rsatoolbox.model.ModelSelect(i_model,
                model_rdms.subset('brain_computational_model', i_model)))

        print('created the following models:')
        for i in range(len(models_flex)):
            print(models_flex[i].name)

        train_set, test_set, ceil_set = rsatoolbox.inference.sets_k_fold(
            rdms_data, k_pattern=3, k_rdm=2)

        results_3_cv = rsatoolbox.inference.crossval(
            models_flex, rdms_data, train_set, test_set,
            ceil_set=ceil_set, method='corr')
        # plot results
        rsatoolbox.vis.plot_model_comparison(results_3_cv)

        results_3_full = rsatoolbox.inference.bootstrap_crossval(
            models_flex, rdms_data, k_pattern=4, k_rdm=2, method='corr', N=5)
        # plot results
        rsatoolbox.vis.plot_model_comparison(results_3_full)

    def test_temporal_rsa(self):
        import numpy as np
        import matplotlib.pyplot as plt
        import rsatoolbox
        import pickle
        from rsatoolbox.rdm import calc_rdm_movie

        import os
        path = os.path.dirname(os.path.abspath(__file__))
        dat = pickle.load(open(os.path.join(
            path, '..', 'demos',
            "TemporalSampleData", "meg_sample_data.pkl"), "rb"))
        measurements = dat['data']
        cond_names = [x for x in dat['cond_names'].keys()]
        cond_idx = dat['cond_idx']
        channel_names = dat['channel_names']
        times = dat['times']
        print('there are %d observations (trials), %d channels, and %d time-points\n' %
              (measurements.shape))
        print('conditions:')
        print(cond_names)

        fig, ax = plt.subplots(1, 2, figsize=(12, 4))
        ax = ax.flatten()
        for jj, chan in enumerate(channel_names[:2]):
            for ii, cond_ii in enumerate(np.unique(cond_idx)):
                mn = measurements[cond_ii == cond_idx, jj, :].mean(0).squeeze()
                ax[jj].plot(times, mn, label=cond_names[ii])
                ax[jj].set_title(chan)
        ax[jj].legend()
        tim_des = {'time': times}
        des = {'session': 0, 'subj': 0}
        obs_des = {'conds': cond_idx}
        chn_des = {'channels': channel_names}
        data = rsatoolbox.data.TemporalDataset(
            measurements,
            descriptors=des,
            obs_descriptors=obs_des,
            channel_descriptors=chn_des,
            time_descriptors=tim_des)
        data.sort_by('conds')
        print('shape of original measurements')
        print(data.measurements.shape)
        data_split_time = data.split_time('time')
        print('\nafter splitting')
        print(len(data_split_time))
        print(data_split_time[0].measurements.shape)
        print('shape of original measurements')
        print(data.measurements.shape)
        data_subset_time = data.subset_time('time', t_from=-.1, t_to=.5)
        print('\nafter subsetting')
        print(data_subset_time.measurements.shape)
        print(data_subset_time.time_descriptors['time'][0])
        bins = np.reshape(tim_des['time'], [-1, 2])
        print(len(bins))
        print(bins[0])
        print('shape of original measurements')
        print(data.measurements.shape)
        data_binned = data.bin_time('time', bins=bins)
        print('\nafter binning')
        print(data_binned.measurements.shape)
        print(data_binned.time_descriptors['time'][0])
        print('shape of original measurements')
        print(data.measurements.shape)
        data_dataset = data.convert_to_dataset('time')
        print('\nafter binning')
        print(data_dataset.measurements.shape)
        print(data_dataset.obs_descriptors['time'][0])
        rdms_data = calc_rdm_movie(data, method='euclidean',
                                   descriptor='conds')
        print(rdms_data)
        rdms_data_binned = calc_rdm_movie(
            data, method='euclidean',
            descriptor='conds',
            bins=bins)
        print(rdms_data_binned)
        plt.figure(figsize=(10, 15))
        # add formated time as rdm_descriptor
        rdms_data_binned.rdm_descriptors['time_formatted'] = [
            '%0.0f ms' % (np.round(x * 1000, 2))
            for x in rdms_data_binned.rdm_descriptors['time']]

        rsatoolbox.vis.show_rdm(rdms_data_binned,
                           pattern_descriptor='conds',
                           rdm_descriptor='time_formatted')
        from rsatoolbox.rdm import get_categorical_rdm
        rdms_model_in = get_categorical_rdm(['%d' % x for x in range(4)])
        rdms_model_lr = get_categorical_rdm(['l', 'r', 'l', 'r'])
        rdms_model_av = get_categorical_rdm(['a', 'a', 'v', 'v'])
        model_names = ['independent', 'left/right', 'audio/visual']
        # append in one RDMs object
        model_rdms = rdms_model_in
        model_rdms.append(rdms_model_lr)
        model_rdms.append(rdms_model_av)
        model_rdms.rdm_descriptors['model_names'] = model_names
        model_rdms.pattern_descriptors['cond_names'] = cond_names
        plt.figure(figsize=(10, 10))
        rsatoolbox.vis.show_rdm(model_rdms, rdm_descriptor='model_names',
                           pattern_descriptor='cond_names')
        from rsatoolbox.rdm import compare
        r = []
        for mod in model_rdms:
            r.append(compare(mod, rdms_data_binned, method='cosine'))
        for i, r_ in enumerate(r):
            plt.plot(rdms_data_binned.rdm_descriptors['time'], r_.squeeze(),
                     label=model_names[i])
        plt.xlabel('time')
        plt.ylabel('model-data cosine similarity')
        plt.legend()

    def test_demo_rdm_scatterplot(self):
        import os
        from scipy.io import loadmat
        import numpy as np
        from rsatoolbox.rdm import concat, RDMs
        from rsatoolbox.vis import rdm_comparison_scatterplot
        path = os.path.dirname(os.path.abspath(__file__))

        condition_descriptors = {
         'type': (["A"] * 23) + (["B"] * 23) + (["C"] * 46)
        }

        matlab_data = loadmat(
            os.path.join(path, '..', 'demos', '92imageData/92_brainRDMs.mat'))['RDMs']
        n_rdms = len(matlab_data[0])
        rdms_ = RDMs(np.array([matlab_data[0][i][0][0] for i in range(n_rdms)]),
                  pattern_descriptors=condition_descriptors,
                  rdm_descriptors={'name': np.array([f"RDM{i}"
                                                     for i in range(4)])}
                  )

        rdms_a = concat([rdms_[0], rdms_[1]])
        rdms_b = concat([rdms_[2], rdms_[3]])

        rdm_comparison_scatterplot((rdms_a, rdms_b),
            show_marginal_distributions=True,
            show_identity_line=True,
            show_legend=False,
            highlight_selector='type',
            highlight_categories=["A", "B", "C"],
            colors={
                "A": (1, 0, 0),
                "B": (0, 1, 0),
                "C": (0, 0, 1),
            }
            )

if __name__ == '__main__':
    unittest.main()
