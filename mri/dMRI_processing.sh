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
	echo "$(basename $0) [subject_id] [subjects_dir] [saving_dir]"
	echo "This script uses MRtrix to analyze diffusion data.
            as well as extracting probabilistic white matter tracts:
			1) The diffusion image (.rec / .nii format);
			2) The structural T1 image (.rec / .nii format);
			3) The subject ID number;
			4) Path to a directory to save anatomical data of the subject. If not provided, the default directory will be used;" 
	}

if [[ "$1" == "--h" || $# -lt 2 ]]; then
	display_usage
	exit 1
fi

subject_id=$1
subjects_dir=$2
saving_dir=$3

## set Paths
raw_dwi="$subjects_dir/$subject_id/dMRI/raw_dwi.rec"
raw_anat="$subjects_dir/$subject_id/sMRI/raw_anat.nii"
default_saving_dir="/home/ubuntu/data/subjects_mrtrix_dir/$subject"
fs_dir="/usr/local/freesurfer/8.0.0"
luts_dir="/usr/local/mrtrix3/share/mrtrix3/labelconvert" # should be updated




if [ -n "$saving_dir" ]; then
    if [ ! -d "$saving_dir" ]; then
        mkdir -p "$saving_dir"
    fi
    cd "$saving_dir"
else
    if [ ! -d "$default_saving_dir" ]; then
        mkdir -p "$default_saving_dir"
    fi
    cd "$default_saving_dir"
fi

dcm2niix -f raw_dwi -o . $raw_dwi
mrconvert raw_dwi.nii raw_dwi.mif -fslgrad raw_dwi.bvec raw_dwi.bval
echo -e "\e[32mRecording file is converted to nifti and mif formats successfuly!"

### Preprocessing
dwidenoise raw_dwi.mif dwi_den.mif -noise noise.mif
mrcalc raw_dwi.mif dwi_den.mif -subtract residual.mif
mrdegibbs dwi_den.mif dwi_den_gibb.mif
dwifslpreproc dwi_den_gibb.mif dwi_den_preproc.mif -pe_dir ap -rpe_none  # check this (roger says its okay)
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
tcksift2 -act 5tt_coreg.mif -out_mu sift_mu.txt -out_coeffs sift_coeffs.txt
                                                    tracks_10M.tck \
                                                    wmfod_norm.mif \
                                                    sift_1M.txt

## Create a Connectome for Different Atlases

# aparc atlases (84 & 164)
labelconvert $fs_dir/subjects/$subject_id/mri/aparc+aseg.mgz \ 
                                                    $fs_dir/FreeSurferColorLUT.txt \
                                                    $luts_dir/fs_default.txt \
                                                    aparc_parcels.mif
labelconvert $fs_dir/subjects/$subject_id/mri/aparc.a2009s+aseg.mgz \ 
                                                    $fs_dir/FreeSurferColorLUT.txt \
                                                    $luts_dir/fs_a2009s.txt \
                                                    aparc2009s_parcels.mif
# schaefer atlases (looks like we dont need a labelconvert for schaefer atlas see ->)
# https://community.mrtrix.org/t/whole-brain-connectome-using-schaefer-parcellation/5301/2


tck2connectome -symmetric -zero_diagonal -scale_invnodevol -tck_weights_in sift_1M.txt tracks_10M.tck 0002_parcels_aparc.mif connectome.csv -out_assignment assignments.txt
label2mesh 0002_parcels_aparc.mif parcel_mesh.obj # add also for other atlas
meshfilter parcel_mesh.obj smooth parcel_mesh_smoothed.obj
connectome2tck tracks_10M.tck assignments.txt edge_exemplar.tck -files single -exemplars 0002_parcels_aparc.mif

### TBSS


### Connectome



### plots
# mrview of dwibiascorrect
# quality of co-registration
# mrview sub-02_den_preproc_unbiased.mif -overlay.load 5tt_nocoreg.mif -overlay.colourmap 2 -overlay.load 5tt_coreg.mif -overlay.colourmap 1