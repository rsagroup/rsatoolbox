#!/bin/bash
mkdir paper_example
aws s3 sync s3://natural-scenes-dataset/nsddata/stimuli/nsdsynthetic/nsdsynthetic paper_example/stimuli
aws s3 sync s3://natural-scenes-dataset/nsddata/stimuli/nsdsynthetic/nsdsynthetic_subj01 paper_example/stimuli
aws s3 sync s3://natural-scenes-dataset/nsddata/experiments/nsdsynthetic paper_example/nsdsynthetic
for i in {1..8}; do
  aws s3 cp s3://natural-scenes-dataset/nsddata_betas/ppdata/subj0${i}/func1pt8mm/nsdsyntheticbetas_fithrf/betas_nsdsynthetic.hdf5 paper_example/data/subj0${i}/betas_nsdsynthetic.hdf5;
  aws s3 cp s3://natural-scenes-dataset/nsddata/ppdata/subj0${i}/func1pt8mm/roi/prf-visualrois.nii.gz paper_example/rois/subj0${i}/prf-visualrois.nii.gz;
done
