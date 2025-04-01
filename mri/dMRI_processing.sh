#!/bin/bash

# Diffusion MRI Analysis Pipeline
# Written by Payam S. Shabestari, Zurich, 01.2025
# email: payam.sadeghishabestari@uzh.ch
# This script is written mainly for Antinomics project. However It could be used for other purposes.


## chmod +x sMRI_processing.sh
## ./dMRI_processing.sh [args]
## echo sth >> log.txt
## put echos
## good comments
## put look up tables of schaefer here ->

set -e
display_usage() {
	echo "$(basename $0) [subject_id]"
    echo " "
    echo "Usage:"
    echo "[subject_id] -> The subject ID"
    echo " "
    echo " "
    echo " "
    echo "Diffusion MRI analyses, including spherical deconvolution and tractography"
    echo " "
	echo "This script uses MRtrix3 for tractography and creating connectomics"
    echo " "
    echo "1. Segmentation of hippocampal subfields and nuclei of the amygdala using an additional T2 scan"
    echo " "
    echo "2. Segmentation of Brainstem Substructures"
    echo " "
    echo "3. Segmentation of thalamic nuclei using only T1 image"
    echo " "
    echo "4. Segmentations of brainstem nuclei that are part of the Ascending Arousal Network"
    echo " "
    echo "5. Segmentation of the hypothalamus and its associated subunits"
    echo " "
    echo "RUNNING TRACULA TO EXTRACT PROBABLISTIC WHITE MATTER TRACTS"
    echo " "
    echo "1. Pre-processing of the diffusion image data."
    echo " "
    echo "2. Ball-and-stick model fit to reconstruct the pathways from the DWI data."
    echo " "
    echo "3. Generate the probability distributions for all tracts."
    echo " " 
	}


echo "dMRI processing started at $(date '+%Y-%m-%d %H:%M:%S')"
start_time=$(date +%s)

if [[ "$1" == "--h" || $# -lt 1 ]]; then
	display_usage
	exit 1
fi

subject_id=$1

## check if structural analysis have been done!
if [ ! -d "/home/ubuntu/data/subjects_fs_dir/$subject_id" ]; then
    echo "Please run the sMRI_processing.sh script before running this script." >&2
    exit 1
else
    mkdir /home/ubuntu/data/subjects_mrtrix_dir/$subject_id
    subject_dir="/home/ubuntu/data/subjects_mrtrix_dir/$subject_id"
    subject_fs_dir="/home/ubuntu/data/subjects_fs_dir/$subject_id"
    export PATH=/home/ubuntu/data/src_codes/ants-2.5.4/bin:$PATH

## set Paths
spath="/home/ubuntu/data/subjects_raw/$subject_id"
raw_dwi="$spath/dMRI/raw_dwi.rec"
raw_anat="$spath/sMRI/raw_anat_T1.nii"
fs_dir="/usr/local/freesurfer/8.0.0"
luts_dir="/usr/local/mrtrix3/share/mrtrix3/labelconvert"

## Conversion
dcm2niix -f raw_dwi -o $subject_dir $raw_dwi
cd $subject_dir
mrconvert raw_dwi.nii raw_dwi.mif -fslgrad raw_dwi.bvec raw_dwi.bval
echo -e "\e[32mRecording file is converted to nifti and mif formats successfuly!"

### Preprocessing
dwidenoise raw_dwi.mif dwi_den.mif -noise noise.mif
mrcalc raw_dwi.mif dwi_den.mif -subtract residual.mif
mrdegibbs dwi_den.mif dwi_den_gibb.mif
dwifslpreproc dwi_den_gibb.mif dwi_den_preproc.mif -pe_dir ap -rpe_none  # (roger says its okay) (download anf substitute)
dwibiascorrect ants dwi_den_preproc.mif dwi_den_preproc_unbiased.mif -bias bias.mif # check this if it makes things better or worse
echo -e "\e[32mPreprocessing is done successfuly!"

### Constrained Spherical Deconvolution
dwi2mask dwi_den_preproc.mif mask.mif
dwi2response tournier dwi_den_preproc_unbiased.mif wm_response.txt -voxels voxels.mif
dwi2fod csd dwi_den_preproc_unbiased.mif -mask mask.mif wm_response.txt wmfod.mif
mtnormalise wmfod.mif wmfod_norm.mif -mask mask.mif
echo -e "\e[32mCSD is done successfuly!"

### Fixel-based analysis (needs averaging wm_responses to have same for all) -> FD & FA maps
# so will add here when we have enough subjects

### Quantitive Structural Connectivity
mrconvert $raw_anat raw_anat.mif
5ttgen fsl raw_anat.mif 5tt_nocoreg.mif
dwiextract dwi_den_preproc_unbiased.mif mean_b0.mif -bzero
mrconvert mean_b0.mif mean_b0.nii.gz
mrconvert 5tt_nocoreg.mif 5tt_nocoreg.nii.gz
fslroi 5tt_nocoreg.nii.gz 5tt_vol0.nii.gz 0 1 

flirt -in mean_b0.nii.gz -ref 5tt_vol0.nii.gz -interp nearestneighbour \
                                                    -dof 6 \
                                                    -omat diff2struct_fsl.mat

transformconvert diff2struct_fsl.mat mean_b0.nii.gz 5tt_nocoreg.nii.gz flirt_import diff2struct_mrtrix.txt
mrtransform 5tt_nocoreg.mif -linear diff2struct_mrtrix.txt -inverse 5tt_coreg.mif
5tt2gmwmi 5tt_coreg.mif gmwmSeed_coreg.mif

tckgen -act 5tt_coreg.mif -backtrack -seed_gmwmi gmwmSeed_coreg.mif \
                                                    -select 10000000 \
                                                    wmfod_norm.mif \
                                                    tracks_10M.tck

tckedit tracks_10M.tck -number 200k smallerTracks_200k.tck
tcksift2 -act 5tt_coreg.mif -out_mu sift_mu.txt -out_coeffs sift_coeffs.txt \
                                                    tracks_10M.tck \
                                                    wmfod_norm.mif \
                                                    sift_1M.txt

## Create a Connectome for Different Atlases (add schaefer, and histological one)

# aparc atlases (84 & 164)
labelconvert $subject_fs_dir/mri/aparc+aseg.mgz \
                $fs_dir/FreeSurferColorLUT.txt \
                $luts_dir/fs_default.txt \
                aparc_parcels.mif
labelconvert $subject_fs_dir/mri/aparc.a2009s+aseg.mgz \
                $fs_dir/FreeSurferColorLUT.txt \
                $luts_dir/fs_a2009s.txt \
                aparc2009s_parcels.mif

# schaefer atlases (looks like we dont need a labelconvert for schaefer atlas see ->)
# https://community.mrtrix.org/t/whole-brain-connectome-using-schaefer-parcellation/5301/2

## lets see this one later :
tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in sift_1M.txt tracks_10M.tck 0002_parcels_aparc.mif connectome.csv -out_assignment assignments.txt
label2mesh 0002_parcels_aparc.mif parcel_mesh.obj # add also for other atlas
meshfilter parcel_mesh.obj smooth parcel_mesh_smoothed.obj
connectome2tck tracks_10M.tck assignments.txt edge_exemplar.tck -files single -exemplars 0002_parcels_aparc.mif

### TBSS


### Connectome
