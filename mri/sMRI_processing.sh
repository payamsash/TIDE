#!/bin/bash

# T1 + T2 anatomical MRI Analysis Pipeline
# Written by Payam S. Shabestari, Zurich, 01.2025
# email: payam.sadeghishabestari@uzh.ch
# This script is written mainly for Antinomics project. However It could be used for other purposes.

set -e
display_usage() {
	echo "$(basename $0) [subject_id] [recon_all]"
    echo " "
    echo "Usage:"
    echo "[subject_id] -> The subject ID"
    echo "[recon-all] -> If false, the cortical segmentation is already done."
    echo "By default, recon-all function from FS will be run."
    echo " "
    echo " "
    echo " "
    echo "CORTICAL AND SUBCORTICAL SEGMENTATION OF T1 IMAGE (+T2 +DWI)"
    echo " "
    echo "This script uses Freesurfer for cortical and subcortical segmentation"
    echo "as well as extracting probabilistic white matter tracts in multiple steps:"
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


if [[ "$1" == "--h" || $# -lt 1 ]]; then
	display_usage
	exit 1
fi

subject_id=$1
recon_all=${2:-false}

## set Paths
export FREESURFER_HOME=/usr/local/freesurfer/8.0.0
export SUBJECTS_DIR=/home/ubuntu/data/subjects_fs_dir
export ANTSPATH=/home/ubuntu/data/src_codes/ants-2.5.4/bin
export LD_LIBRARY_PATH=$FREESURFER_HOME/MCRv97/runtime/glnxa64:$FREESURFER_HOME/MCRv97/bin/glnxa64:$FREESURFER_HOME/MCRv97/sys/os/glnxa64:$FREESURFER_HOME/MCRv97/extern/bin/glnxa64
export PATH=/usr/lib/mrtrix3/bin:$PATH
export PATH=/home/ubuntu/fsl/bin:$PATH
export PATH=/home/ubuntu/data/src_codes/ants-2.5.4/bin:$PATH
export subject_id=$subject_id
source $FREESURFER_HOME/SetUpFreeSurfer.sh

sch_gcs_dir=" $FREESURFER_HOME/gcs"
spath="/home/ubuntu/data/subjects_raw/$subject_id"
raw_t2="$spath/sMRI/raw_anat_T2.nii"
raw_dwi="$spath/dMRI/raw_dwi.rec"
dpath_fs="$SUBJECTS_DIR/$subject_id/DTI"

## cortical and subcortical segmentation
if [[ "$recon_all" == "true" ]]; then
	recon-all -s $subject_id -i $raw_t1
	recon-all -all -subjid $subject_id  
fi

## first round (hippocampus + amygdala, brainstem, AAN, hypothalamus, thalamic nuclei, cerebellum)
echo -e "\e[32mSegmentation of hippocampal subfields and nuclei of the amygdala!"
segmentHA_T2.sh $subject_id $raw_t2 T1_T2 1 

echo -e "\e[32mSegmentation of Brainstem Substructures!"
segmentBS.sh $subject_id

echo -e "\e[32mSegmentation of thalamic nuclei using only T1 image!"
segmentThalamicNuclei.sh $subject_id # for now

echo -e "\e[32mSegmentations of brainstem nuclei that are part of the Ascending Arousal Network!"
sudo chmod +x $FREESURFER_HOME/bin/segmentNuclei
segmentAAN.sh $subject_id

echo -e "\e[32mSegmentation of the hypothalamus and its associated subunits"
mri_segment_hypothalamic_subunits --s $subject_id


## convert DTI to nii, then denoise it with mrtrix, then tracula
mkdir $dpath_fs
dcm2niix -o $dpath_fs -f raw_dwi $raw_dwi
mrconvert $dpath_fs/raw_dwi.nii $dpath_fs/raw_dwi.mif -fslgrad $dpath_fs/raw_dwi.bvec $dpath_fs/raw_dwi.bval
dwidenoise $dpath_fs/raw_dwi.mif $dpath_fs/dwi_den.mif
mrconvert $dpath_fs/dwi_den.mif $dpath_fs/dwi_den.nii -fslgrad $dpath_fs/raw_dwi.bvec $dpath_fs/raw_dwi.bval


echo -e "\e[32mPre-processing of the diffusion image data!"
trac-all -prep -c /home/ubuntu/data/src_codes/tracula_config.txt

echo -e "\e[32mBall-and-stick model fit to reconstruct the pathways from the DWI data!"
trac-all -bedp -c /home/ubuntu/data/src_codes/tracula_config.txt

echo -e "\e[32mGenerate the probability distributions for all tracts!"
trac-all -path -c /home/ubuntu/data/src_codes/tracula_config.txt



## 3. now fix segmentThalamicNuclei_DTI.sh
## 4. now fix histological atlas
## 5. cerebellum (I need to ask this, but they are already in the folder in FS)
## 6. Striatal atlas (I need to ask this)
## 7. schaefer atlas (maybe open a new folder for it) (I think I will only need it in fmri)


# mri_histo_atlas_segment_fireants INPUT_SCAN OUTPUT_DIRECTORY GPU THREADS [BF_MODE]

mri_vol2vol --mov 0002/mri/norm.mgz --o 0002/mri/norm.dwispace.mgz --lta 0002/dmri/xfms/anatorig2diff.bbr.lta  --no-resample --targ 0002/dmri/dtifit_FA.nii.gz
mri_vol2vol --mov 0002/mri/aseg.mgz --o 0002/mri/aseg.dwispace.mgz --lta 0002/dmri/xfms/anatorig2diff.bbr.lta  --no-resample --targ 0002/dmri/dtifit_FA.nii.gz
mri_segment_thalamic_nuclei_dti_cnn --t1 0002/mri/norm.dwispace.mgz --aseg 0002/mri/aseg.dwispace.mgz --fa 0002/dmri/dtifit_FA.nii.gz --v1 0002/dmri/dtifit_V1.nii.gz --o 0002/mri/thalamic_dti.nii.gz --vol 0002/tables/thalamic_dti_volumes.csv

'''
so now error is again illegal hardware instruction which is probably due to TF. -> probably linux will fix

'''




#### cerebellum parcelation

# Step 1: Upsample Buckner atlas from 2mm to 1mm resolution (matching MNI152 1mm template)
mri_vol2vol --mov Buckner_atlas.nii.gz \
            --targ MNI152/mri/norm.mgz \
            --regheader \
            --o Buckner_atlas1mm.nii.gz \
            --no-save-reg \
            --interp nearest

# Step 2: Warp the upsampled atlas to FreeSurfer's nonlinear volumetric space
mri_vol2vol_used --mov Buckner_atlas1mm.nii.gz \
                 --s MNI152_FS \
                 --targ $FREESURFER_HOME/average/mni305.cor.mgz \
                 --m3z talairach.m3z \
                 --o Buckner_atlas_freesurfer_internal_space.nii.gz \
                 --interp nearest

# Step 3: Warp the atlas from FreeSurfer internal space to subject’s native space
mri_vol2vol --mov $SUBJECTS_DIR/SUBJECT_FS/mri/norm.mgz \
            --s SUBJECT_FS \
            --targ Buckner_atlas_freesurfer_internal_space.nii.gz \
            --m3ztalairach.m3z \
            --o Buckner_atlas_subject.nii.gz \
            --interp nearest \
            --inv-morph

# freeview -v ${SUBJECTS_DIR}/${SUBJECT_ID}/mri/orig.mgz \
#            ${OUTPUT_DIR}/Buckner_atlas_subject.nii.gz:colormap=lut


#### Striatal Parcellation
# same as cerebellum

# freeview -v FSL_MNI152_FreeSurferConformed_1mm.nii.gz Choi2012_7Networks_MNI152_FreeSurferConformed1mm_TightMask.nii.gz:colormap=lut:lut=Choi2012_7Networks_ColorLUT.txt

# freeview -v FSL_MNI152_FreeSurferConformed_1mm.nii.gz Choi2012_7Networks_MNI152_FreeSurferConformed1mm_TightMask.nii.gz:colormap=lut:lut=Choi2012_7Networks_ColorLUT.txt Choi2012_7NetworksConfidence_MNI152_FreeSurferConformed1mm_TightMask.nii.gz:colormap=heat:heatscale=0,0.5,1



## schafer atlas (we might need it for fMRI or dMRI)
hemis=("lh" "rh")
for hemi in "${hemis[@]}"; do
	for n in 100 200 300 400; do
		for net_option in 7 17; do
			mris_ca_label -l $SUBJECTS_DIR/$subject_id/label/${hemi}.cortex.label \
			$subject_id ${hemi} $SUBJECTS_DIR/$subject_id/surf/${hemi}.sphere.reg \
			$gcs_dir/${hemi}.Schaefer2018_${n}Parcels_${net_option}Networks.gcs \
			$SUBJECTS_DIR/$subject_id/label/${hemi}.Schaefer2018_${n}Parcels_${net_option}Networks_order.annot
		done
	done
done

## extracting probabilistic white matter tracts



## bem watershed

mne watershed_bem -s 0002 -d /Applications/freesurfer/dev/subjects

##### tables

## extract DK atlas information
measures=("area" "volume" "thickness")
parcels=("aparc" "aparc.a2009s")
for hemi in "${hemis[@]}"; do
    for meas in "${measures[@]}"; do
        for parc in "${parcels[@]}"; do
            # Construct the table file name
            if [[ $parc == "aparc" ]]; then
                tablefile="tables/${parc}_${meas}_${hemi}.txt"
            else
                tablefile="tables/${parc}_${meas}_${hemi}.txt"
            fi
            aparcstats2table --subjects "$subject_id" --hemi "$hemi" --meas "$meas" --parc="$parc" --tablefile "$tablefile"
        done
    done
done

# extract segmentation information
asegstats2table --subjects $subject_id --common-segs --meas volume --stats=aseg.stats --table=tables/segstats.txt
asegstats2table --subjects $subject_id --statsfile=brainstem.v13.stats --tablefile=tables/brainstem.txt

for hemi in "${hemispheres[@]}"; do
	statsfile="thalamic-nuclei.${hemi}.v13.T1.stats"
	tablefile="tables/thalamic-nuclei_${hemi}.txt"
	asegstats2table --subjects $SUB_ID --statsfile="$statsfile" --tablefile="$tablefile"
	
	statsfile="hipposubfields.${hemi}.T2.v22.T2.stats"
	tablefile="tables/hipposubfields_${hemi}.txt"
	asegstats2table --subjects $SUB_ID --statsfile=hipposubfields.lh.T2.v22.T2.stats --tablefile=tables/hipposubfields_lh.txt 
	
	statsfile="amygdalar-nuclei.${hemi}.T2.v22.T2.stats"
	tablefile="tables/amygdalar_${hemi}.txt"
	asegstats2table --subjects $SUB_ID --statsfile=amygdalar-nuclei.lh.T2.v22.T2.stats --tablefile=tables/amygdalar_lh.txt 
done










## for report 

## AAN
freeview -v $SUBJECTS_DIR/0002/mri/T1.mgz -v  $SUBJECTS_DIR/0002/mri/arousalNetworkLabels.v10.mgz:colormap=lut:lut=$FREESURFER_HOME/average/AAN/atlas/freeview.lut.txt

