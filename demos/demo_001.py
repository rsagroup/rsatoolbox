#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 11:47:14 2020

Analog of DEMO1_RSA_ROI_simulatedAndRealData

% This function computes the results in the main Figures of 
% Nili et al. (PLoS Comp Biol 2013)

@author: heiko
"""

import os
import scipy.io
import numpy as np
import pyrsa

# control variables
n_subjects = 12
subject_pattern_noise_std = 1
n_model_grades = 3
best_model_pattern_deviation_std = 0
worst_model_pattern_deviation_std = 6
method = 'corr'

n_cond = 92 

# load RDMs and category definitions from Kriegeskorte et al. (Neuron 2008)
data_matlab = scipy.io.loadmat(os.path.join('92imageData',
    'Kriegeskorte_Neuron2008_supplementalData.mat'))

category_vectors = data_matlab['categoryVectors']
category_dict = {'animate':category_vectors[:,0],
                 'inanim':category_vectors[:,1],
                 'human':category_vectors[:,2],
                 'nonhumani':category_vectors[:,3],
                 'body':category_vectors[:,4],
                 'face':category_vectors[:,5],
                 'natObj':category_vectors[:,6],
                 'artiObj':category_vectors[:,7],
                 'rand24':category_vectors[:,8],
                 'rand48':category_vectors[:,9],
                 'other48':category_vectors[:,10],
                 'monkeyape':category_vectors[:,11],
                 }

rdms_mit_hit_fig1 = data_matlab['RDMs_mIT_hIT_fig1']
rdm_monkey_vec = rdms_mit_hit_fig1[0][0][2]
rdm_human_vec = rdms_mit_hit_fig1[0][1][2]

rdm_monkey = pyrsa.rdm.RDMs(rdm_monkey_vec, pattern_descriptors=category_dict)
rdm_human = pyrsa.rdm.RDMs(rdm_human_vec,  pattern_descriptors=category_dict)

# show rdm objects
print(rdm_monkey)
print(rdm_human)

# load RDM measurements from data
data_matlab2 = scipy.io.loadmat(os.path.join('92imageData',
    '92_brainRDMs.mat'))

rdms_sess1 = np.array([data_matlab2['RDMs'][0][i][0][0] for i in range(4)])
rdms_sess2 = np.array([data_matlab2['RDMs'][0][i][1][0] for i in range(4)])
rdms_array = np.concatenate((rdms_sess1,rdms_sess2),0)
rdms_human = pyrsa.rdm.RDMs(rdms_array, rdm_descriptors = {
    'session':np.array([1,1,1,1,2,2,2,2]),
    'subject':np.array([1,2,3,4,1,2,3,4])})

# TODO: plot these rdms!

# load reconstructed patterns for simulating models
data_matlab3 = scipy.io.loadmat(os.path.join('92imageData',
    'simTruePatterns.mat'))
sim_true_patterns = data_matlab3['simTruePatterns']
sim_true_patterns2 = data_matlab3['simTruePatterns2']
n_cond, n_dim = sim_true_patterns.shape

# simulate multiple subjects' noisy RDMs

data_list = []
for i_subject in range(n_subjects):
    patterns_subject = sim_true_patterns2 \
        + subject_pattern_noise_std * np.random.randn(n_cond, n_dim)
    dataset = pyrsa.data.Dataset(patterns_subject,
                                 obs_descriptors=category_dict)
    data_list.append(dataset)
subject_rdms = pyrsa.rdm.calc_rdm(data_list)

# TODO: shorten import for this quite handy function
avg_subject_rdm = pyrsa.util.inference_util.pool_rdm(subject_rdms,
                                                     method=method)

# TODO: Again showing RDMs missing!
#rsa.fig.showRDMs(rsa.rdm.concatRDMs_unwrapped(subjectRDMs,avgSubjectRDM),2);
#rsa.fig.handleCurrentFigure([userOptions.rootPath,filesep,'simulatedSubjAndAverage'],userOptions);


# define categorical model RDMs
bin_rdm_animacy = pyrsa.rdm.get_categorical_rdm(category_dict['animate'])

#ITemphasizedCategories=[1 2 5 6] # animate, inanimate, face, body
#[binRDM_cats, nCatCrossingsRDM]=rsa.rdm.categoricalRDM(categoryVectors(:,ITemphasizedCategories),4,true);
#cat_vecs = np.array(
#            [category_dict['animate'],
#            category_dict['face'],
#            category_dict['natObj']]).T
#bin_rdm_cats = pyrsa.rdm.get_categorical_rdm(cat_vecs)

data_matlab4 = scipy.io.loadmat(os.path.join('92imageData',
    'faceAnimateInaniClustersRDM.mat'))
rdm_face_animate_inani = data_matlab4['faceAnimateInaniClustersRDM']

# load behavioural RDM from Mur et al. (Frontiers Perc Sci 2013)
data_matlab5 = scipy.io.loadmat(os.path.join('92imageData',
    '92_behavRDMs.mat'))
rdm_array = np.array([data_matlab5['rdms_behav_92'][0,i][0] for i in range(16)])
rdm_sim_judg = pyrsa.rdm.RDMs(rdm_array,
                              pattern_descriptors=category_dict,
                              rdm_descriptors={'subject':np.arange(16)})

# create modelRDMs of different degrees of noise
pattern_dev_stds = np.linspace(best_model_pattern_deviation_std,
                               worst_model_pattern_deviation_std,
                               n_model_grades);

data_list_graded_model = []
for i_graded_model in range(n_model_grades):
    patterns_c_graded_model = sim_true_patterns2 \
        + pattern_dev_stds[i_graded_model] * np.random.randn(n_cond, n_dim)
    dataset = pyrsa.data.Dataset(patterns_c_graded_model,
                                 obs_descriptors=category_dict)
    data_list_graded_model.append(dataset)
graded_model_rdms = pyrsa.rdm.calc_rdm(data_list_graded_model)
graded_model_rdms.rdm_descriptors['pattern_std'] = pattern_dev_stds



# load RDMs for V1 model and HMAX model with natural image patches 
# from Serre et al. (Computer Vision and Pattern Recognition 2005)
#load([pwd,filesep,'92imageData',filesep,'rdm92_V1model.mat'])
#load([pwd,filesep,'92imageData',filesep,'rdm92_HMAXnatImPatch.mat'])
data_matlab6 = scipy.io.loadmat(os.path.join('92imageData',
    'rdm92_V1model.mat'))
V1_model_rdm = data_matlab6['rdm92_V1model']
hmax_rdm = data_matlab6['rdm92_HMAXnatImPatch']

# load RADON and silhouette models and human early visual RDM
#load(['92imageData',filesep,'92_modelRDMs.mat']);
data_matlab7 = scipy.io.loadmat(os.path.join('92imageData',
    '92_modelRDMs.mat'))
four_cats_rdm = data_matlab7['Models'][0][1][0]
human_early_visual_rdm = data_matlab7['Models'][0][3][0]
silhouette_rdm = data_matlab7['Models'][0][6][0]
radon_rdm = data_matlab7['Models'][0][7][0]

####--------------------------------------------------------------------------
#### from here on the new toolboax works actually differently 
####--------------------------------------------------------------------------
# concatenate and name the modelRDMs
#modelRDMs=cat(3,binRDM_animacy,faceAnimateInaniClustersRDM,FourCatsRDM,rdm_simJudg,humanEarlyVisualRDM,rdm_mIT,silhouetteRDM,rdm92_V1model,rdm92_HMAXnatImPatch,radonRDM,gradedModelRDMs);
#modelRDMs=rsa.rdm.wrapAndNameRDMs(modelRDMs,{'ani./inani.','face/ani./inani.','face/body/nat./artif.','sim. judg.','human early visual','monkey IT','silhouette','V1 model','HMAX-2005 model','RADON','true model','true with noise','true with more noise'});
#modelRDMs=modelRDMs(1:end-2); % leave out the true with noise models

# generate model objects
m1 = pyrsa.model.ModelFixed('ani./inani.', bin_rdm_animacy)
m2 = pyrsa.model.ModelFixed('sim. judg.', rdm_sim_judg)
m3 = pyrsa.model.ModelFixed('human early visual', human_early_visual_rdm)
m4 = pyrsa.model.ModelFixed('face/body/nat/artificial', four_cats_rdm)
m5 = pyrsa.model.ModelFixed('silhouette', silhouette_rdm)
m6 = pyrsa.model.ModelFixed('radon', radon_rdm)
m7 = pyrsa.model.ModelFixed('V1 model', V1_model_rdm)
m8 = pyrsa.model.ModelFixed('HMAX model', hmax_rdm)
m9 = pyrsa.model.ModelFixed('face/ani./inani.', rdm_face_animate_inani)
m10 = pyrsa.model.ModelFixed('monkey IT', rdm_monkey)
# m11 = pyrsa.model.ModelFixed('face/body/nat/artificial', bin_rdm_cats)
# is the same as m4

models = [m1, m2, m3, m4, m5, m6, m7, m8, m9, m10]


# A bunch of plotting and MDS plotting
#
# rsa.fig.showRDMs(modelRDMs,5);
# rsa.fig.handleCurrentFigure([userOptions.rootPath,filesep,'allModels'],userOptions);
# # place the model RDMs in cells in order to pass them to
# # compareRefRDM2candRDMs as candidate RDMs
# for modelRDMI=1:numel(modelRDMs)
#     modelRDMs_cell{modelRDMI}=modelRDMs(modelRDMI);
# end

# %% activity pattern MDS
# categoryIs=[5 6 7 8];
# categoryCols=[0 0 0
#               0 0 0
#               0 0 0
#               0 0 0
#               1 0.5 0
#               1 0 0
#               0 1 0
#               0 0.5 1];



# % MDS plot
# categoryIs=[5 6 7 8];
# categoryCols=[0 0 0
#     0 0 0
#     0 0 0
#     0 0 0
#     1 0.5 0
#     1 0 0
#     0 1 0
#     0 0.5 1];


# for condI = 1:92
#     for catI = 1:numel(categoryIs)
#         if categoryVectors(condI,categoryIs(catI))
#             userOptions.conditionColours(condI,:) = categoryCols(categoryIs(catI),:);
#         end
#     end
# end
# avgRDM.RDM = avgSubjectRDM;
# avgRDM.name = 'subject-averaged RDM';
# avgRDM.color = [0 0 0];
# [blankConditionLabels{1:size(modelRDMs_cell{1}.RDM,1)}] = deal(' ');

# % true-model MDS
# rsa.MDSConditions(modelRDMs_cell{11}, userOptions,struct('titleString','ground-truth MDS',...
#     'fileName','trueRDM_MDS','figureNumber',6));
# % true-model dendrogram

# rsa.dendrogramConditions(modelRDMs_cell{11}, userOptions,...
# struct('titleString', 'Dendrogram of the ground truth RDM', 'useAlternativeConditionLabels', true, 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 7));
# % subject-averaged MDS
# rsa.MDSConditions(avgRDM, userOptions,struct('titleString','subject-averaged MDS',...
#     'fileName','ssMDS','figureNumber',8));
# % subject-averaged Dendrogram
# rsa.dendrogramConditions(avgRDM, userOptions,...
# struct('titleString', 'Dendrogram of the subject-averaged RDM', 'useAlternativeConditionLabels', true, 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 9));

# % one-subject MDS (e.g. simulated subject1), noisier
# rsa.MDSConditions(rsa.rdm.wrapAndNameRDMs(subjectRDMs(:,:,1),{'single-subject RDM'}), userOptions,struct('titleString','sample subject MDS',...
#     'fileName','single-subject RDM','figureNumber',10));

# % one-subject Dendrogram
# rsa.dendrogramConditions(rsa.rdm.wrapAndNameRDMs(subjectRDMs(:,:,3),{'single-subject RDM'}), userOptions,...
# struct('titleString', 'Dendrogram of a single-subject RDM', 'useAlternativeConditionLabels', true, 'alternativeConditionLabels', {blankConditionLabels}, 'figureNumber', 11));



# %% RDM correlation matrix and MDS
# % 2nd order correlation matrix
# userOptions.RDMcorrelationType='Kendall_taua';

# rsa.pairwiseCorrelateRDMs({avgRDM, modelRDMs}, userOptions, struct('figureNumber', 12,'fileName','RDMcorrelationMatrix'));

# % 2nd order MDS
# rsa.MDSRDMs({avgRDM, modelRDMs}, userOptions, struct('titleString', 'MDS of different RDMs', 'figureNumber', 13,'fileName','2ndOrderMDSplot'));

##############################################################################
### statistical inference part
##############################################################################

# userOptions.RDMcorrelationType='Kendall_taua';
# userOptions.RDMrelatednessTest = 'subjectRFXsignedRank';
# userOptions.RDMrelatednessThreshold = 0.05;
# userOptions.RDMrelatednessMultipleTesting = 'FDR';
# userOptions.saveFiguresPDF = 1;
# userOptions.candRDMdifferencesTest = 'subjectRFXsignedRank';
# userOptions.candRDMdifferencesThreshold = 0.05;
# userOptions.candRDMdifferencesMultipleTesting = 'FDR';
# userOptions.plotpValues = '=';
# userOptions.barsOrderedByRDMCorr=true;
# userOptions.resultsPath = userOptions.rootPath;
# userOptions.figureIndex = [14 15];
# userOptions.figure1filename = 'compareRefRDM2candRDMs_barGraph_simulatedITasRef';
#userOptions.figure2filename = 'compareRefRDM2candRDMs_pValues_simulatedITasRef';
#stats_p_r=rsa.compareRefRDM2candRDMs(subjectRDMs, modelRDMs_cell, userOptions);

results_simulation = pyrsa.inference.eval_bootstrap_rdm(models, subject_rdms,
                                                        method='spearman',
                                                        N=1000)
pyrsa.vis.plot_model_comparison(results_simulation)

results_simulation2 = pyrsa.inference.eval_bootstrap(models, subject_rdms,
                                                     method='spearman',
                                                     N=1000)
pyrsa.vis.plot_model_comparison(results_simulation2)

## Finally: real fMRI data (human IT RDM from Kriegeskorte et al. (Neuron 2008) as the reference RDM
# userOptions.RDMcorrelationType = 'Spearman';
# userOptions.RDMrelatednessTest = 'randomisation';
# userOptions.RDMrelatednessThreshold = 0.05;
# userOptions.RDMrelatednessMultipleTesting = 'none';%'FWE'
# userOptions.candRDMdifferencesTest = 'conditionRFXbootstrap';
# userOptions.candRDMdifferencesMultipleTesting = 'FDR';
# userOptions.plotpValues = '*';
# userOptions.nRandomisations = 100;
# userOptions.nBootstrap = 100;
# userOptions.candRDMdifferencesThreshold = 0.05;
# userOptions.candRDMdifferencesMultipleTesting = 'FDR';
# userOptions.figure1filename = 'compareRefRDM2candRDMs_barGraph_hITasRef';
# userOptions.figure2filename = 'compareRefRDM2candRDMs_pValues_hITasRef';
# userOptions.figureIndex = [16 17];
# stats_p_r=rsa.compareRefRDM2candRDMs(RDMs_hIT_bySubject, modelRDMs_cell(1:end-1), userOptions);

results_fmri = pyrsa.inference.eval_bootstrap(models, rdms_human,
                                              rdm_descriptor='subject',
                                              method='spearman',
                                              N=1000)
pyrsa.vis.plot_model_comparison(results_fmri)

results_fmri_tau = pyrsa.inference.eval_bootstrap(models, rdms_human,
                                                  rdm_descriptor='subject',
                                                  method='kendall',
                                                  N=100)
pyrsa.vis.plot_model_comparison(results_fmri_tau)

