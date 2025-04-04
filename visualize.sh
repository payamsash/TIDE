#!/bin/bash

# T1 + T2 anatomical MRI visualization
# Written by Payam S. Shabestari, Zurich, 01.2025
# email: payam.sadeghishabestari@uzh.ch
# This script is written mainly for Antinomics project. However It could be used for other purposes.

set -e
display_usage() {
	echo "$(basename $0) [subject_id] [structure]"
    echo " "
    echo "Usage:"
    echo "[subject_id] -> The subject ID"
    echo "[structure] -> The structure to be visualized together with T1 image."
    echo " "
    echo "The options for structure are as following:"
    echo " "
    echo "  1. aparc : aparc segmentation"
    echo "  2. hippo : hippocampal subfields and nuclei of the amygdala"
    echo "  3. brainstem : Brainstem Substructures"
    echo "  4. thalamus : thalamic nuclei"
    echo "  5. aan : brainstem nuclei that are part of the Ascending Arousal Network"
    echo "  6. hypo : hypothalamus and its associated subunits"
    echo "  7. striatum : striatal parcellation estimated by intrinsic functional connectivity"
    echo "  8. cerebellum : cerebellum parcellation estimated by intrinsic functional connectivity"
    echo "  9. tracs : probability distributions for all tracts"
	}


if [[ "$1" == "--h" || $# -lt 1 ]]; then
	display_usage
	exit 1
fi

subject_id=$1
structure=$2

export FREESURFER_HOME=/usr/local/freesurfer/8.0.0
export SUBJECTS_DIR=/home/ubuntu/data/subjects_fs_dir
source $FREESURFER_HOME/SetUpFreeSurfer.sh

lut_dir=$FREESURFER_HOME/average/AAN/atlas/freeview.lut.txt
surf_dir=$SUBJECTS_DIR/$subject_id/surf
dmri_dir=$SUBJECTS_DIR/$subject_id/dmri
tracs_dir=$SUBJECTS_DIR/$subject_id/dpath


cd $SUBJECTS_DIR/$subject_id/mri

if [[ "$structure" == "aparc" ]]; then
    freeview -v T1.mgz -v wm.mgz -v brainmask.mgz -v aseg.auto.mgz:colormap=lut:opacity=0.2 \
                -f $surf_dir/lh.white:edgecolor=blue -f $surf_dir/lh.pial:edgecolor=red \
                -f $surf_dir/rh.white:edgecolor=blue -f $surf_dir/rh.pial:edgecolor=red
fi

if [[ "$structure" == "hippo" ]]; then
    freeview -v nu.mgz -v T1_T2.FSspace.mgz:sample=cubic \
                -v lh.hippoAmygLabels-T1-T1_T2.v22.mgz:colormap=lut \
                -v rh.hippoAmygLabels-T1-T1_T2.v22.mgz:colormap=lut
fi

if [[ "$structure" == "brainstem" ]]; then
    freeview -v nu.mgz -v brainstemSsLabels.v13.mgz:colormap=lut  
fi

if [[ "$structure" == "thalamus" ]]; then
    freeview norm.dwispace.mgz ThalamicNuclei.v13.T1.mgz:colormap=LUT \
                -dti $dmri_dir/dtifit_V1.nii.gz $dmri_dir/dtifit_FA.nii.gz
fi

if [[ "$structure" == "aan" ]]; then
    freeview -v T1.mgz -v arousalNetworkLabels.v10.mgz:colormap=lut:lut=$lut_dir
fi

if [[ "$structure" == "hypo" ]]; then
    freeview -v nu.mgz -v hypothalamic_subunits_seg.v1.mgz:colormap=lut
fi

if [[ "$structure" == "striatum" ]]; then
    freeview -v nu.mgz -v striatum_17_network_Loose_mask.nii.gz:colormap=lut 
fi

if [[ "$structure" == "cerebellum" ]]; then
    freeview -v nu.mgz -v cerebellum_17_network_Loose_mask.nii.gz:colormap=lut
fi

if [[ "$structure" == "tracs" ]]; then
    freeview -v $dmri_dir/dtifit_FA.nii.gz:opacity=.6 \
                -tv $tracs_dir/merged_avg16_syn_bbr.mgz
fi