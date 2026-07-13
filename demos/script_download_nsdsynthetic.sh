#!/bin/bash

# Create directory structure
mkdir paper_example
mkdir -p paper_example/stimuli
mkdir -p paper_example/nsdsynthetic
mkdir -p paper_example/rois

S3_BUCKET="https://natural-scenes-dataset.s3.amazonaws.com"

# Download synthetic stimuli
echo "Downloading nsdsynthetic stimuli..."
for i in {1..220}; do
  img=$(printf "nsdsynthetic%03d" $i)
curl -L -o paper_example/stimuli/$img.png \
  "$S3_BUCKET/nsddata/stimuli/nsdsynthetic/nsdsynthetic/$img.png" 2>/dev/null
  echo "Downloaded $img"
done
for i in {221..284}; do
  img=$(printf "nsdsynthetic%03d" $i)
curl -L -o paper_example/stimuli/$img.png \
  "$S3_BUCKET/nsddata/stimuli/nsdsynthetic/nsdsynthetic_subj01/$img.png" 2>/dev/null
  echo "Downloaded $img"
done

# Download experimental data
echo "Downloading nsdsynthetic experiments data..."
curl -L -o paper_example/nsdsynthetic/nsdsynthetic_expdesign.mat \
  "$S3_BUCKET/nsddata/experiments/nsdsynthetic/nsdsynthetic_expdesign.mat" 2>/dev/null
curl -L -o paper_example/nsdsynthetic/nsdsyntheticimageinformation.csv \
  "$S3_BUCKET/nsddata/experiments/nsdsynthetic/nsdsyntheticimageinformation.csv" 2>/dev/null

# Download betas and ROIs for subjects 01-08
for i in {1..8}; do
  subj=$(printf "subj%02d" $i)
  mkdir -p paper_example/data/$subj
  mkdir -p paper_example/rois/$subj
  
  echo "Downloading betas for $subj..."
  curl -L -o "paper_example/data/$subj/betas_nsdsynthetic.hdf5" \
    "$S3_BUCKET/nsddata_betas/ppdata/$subj/func1pt8mm/nsdsyntheticbetas_fithrf/betas_nsdsynthetic.hdf5"
  
  echo "Downloading ROI data for $subj..."
  curl -L -o "paper_example/rois/$subj/prf-visualrois.nii.gz" \
    "$S3_BUCKET/nsddata/ppdata/$subj/func1pt8mm/roi/prf-visualrois.nii.gz"
done

echo "Download complete!"
