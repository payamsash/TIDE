#!/bin/bash

# T1 + T2 anatomical MRI Analysis Pipeline
# Written by Payam S. Shabestari, Zurich, 01.2025
# email: payam.sadeghishabestari@uzh.ch
# This script is written mainly for Antinomics project. However It could be used for other purposes.

set -e
display_usage() {
	echo "$(basename $0) [subjects]"
	echo "This script uses Freesurfer for cortical and subcortical segmentation:
			1) list of subject ids"
                }

if [[ "$1" == "--h" || $# -lt 1 ]]; then
	display_usage
	exit 1
fi

subjects=("$@")

# Set FreeSurfer environment
export FREESURFER_HOME=/usr/local/freesurfer/8.0.0
export SUBJECTS_DIR=/home/ubuntu/data/subjects_fs_dir
export LD_LIBRARY_PATH=$FREESURFER_HOME/MCRv97/runtime/glnxa64:$FREESURFER_HOME/MCRv97/bin/glnxa64:$FREESURFER_HOME/MCRv97/sys/os/glnxa64:$FREESURFER_HOME/MCRv97/extern/bin/glnxa64
source $FREESURFER_HOME/SetUpFreeSurfer.sh


for subj in "${subjects[@]}"; do
    mri_path="/home/ubuntu/data/subjects_raw/$subj/sMRI/raw_anat_T1.nii"
    subect_path="$SUBJECTS_DIR/$subj"

    if [ ! -d "$subect_path" ]; then
        recon-all -s $subj -i $mri_path
    fi
    echo "Running recon-all for $subj"

    export FS_V8_XOPTS=0 && recon-all -s $subj -all
    if [ $? -eq 0 ]; then
        echo "Recon-all for $subj completed successfully."
    else
        echo "Recon-all for $subj failed. Check logs for details."
    fi
done

echo "All subjects processed."