#!/bin/bash

# Diffusion MRI Analysis Pipeline
# Written by Payam S. Shabestari, Zurich, 01.2025
# email: payam.sadeghishabestari@uzh.ch
# This script is written mainly for Antinomics project. However It could be used for other purposes.


## echo with color
## chmod +x sMRI_processing.sh
## ./dMRI_processing.sh [args]
## echo sth >> log.txt
## copy these files to work with linux : license, buckner atlas, choi atlas, tracula config file

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
default_saving_dir="/home/ubuntu/data/subjects_mrtrix_dir/$subject"
raw_dwi="$subjects_dir/$subject_id/dMRI/raw_dwi.rec"


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
dwifslpreproc dwi_den.mif dwi_den_preproc.mif \
                                    -nocleanup \
                                    -pe_dir AP \
                                    -rpe_none \
                                    -eddy_options \
                                    " --slm=linear --data_is_shelled" # certainly check this




echo -e "\e[32mPreprocessing is done successfuly!"

### Constrained Spherical Deconvolution
dwi2response tournier dwi.mif wm_response.txt