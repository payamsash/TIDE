# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import ipdb
import os
from pathlib import Path
import shutil
import warnings
from datetime import datetime

def create_subjects_dir(
        subject_id,
        raws_dir,
        subjects_dir,
        on_missing="warn",
        skip_list=None
        ):
    """
    Creates a folder structure with the following hierarchy:
    
    base_path/
    ├── raws/
    ├── sMRI/
    ├── fMRI/
    │   ├── session_1/
    │   ├── session_2/
    ├── dMRI/
    
    Parameters
    ----------
    subject_id : str
        Subject ID used in Antinomics or TIDE projects.
    raws_dir: str │ Path
        The root directory, where all raw MR recordings of the subject is stored.
    subjects_dir : str │ Path
        The root directory where all raw files should be kept.
    """

    ## create the folder structure for subjects
    subjects_dir = Path(subjects_dir)
    if not os.path.exists(subjects_dir): 
        os.makedirs(subjects_dir)
    subject_dir = Path(subjects_dir) / f"{subject_id}"
    os.makedirs(subject_dir, exist_ok=True)

    ## check if data are complete and fill subfolders
    raws_dir = Path(raws_dir)
    mri_dir = raws_dir / "mri" / f"{subject_id}_antinomics"
    eeg_dir = raws_dir / "eeg"
    captrack_dir = raws_dir / "captrack"

    for dir, title in zip([mri_dir, eeg_dir, captrack_dir], ["MRI", "EEG", "Captrack"]):
        if os.path.exists(dir):
            if title == "MRI":
                _create_mri_folders(subject_id, mri_dir, subject_dir)
            if title == "EEG":
                _create_eeg_folders(subject_id, eeg_dir, captrack_dir, subject_dir, skip_list)
        else:
            if on_missing == "warn":
                warnings.warn(f"Subject {subject_id} not found in the {title} directory!", UserWarning)
            if on_missing == "raise":
                raise ValueError(f"Subject {subject_id} not found in the {title} directory!")     


def _create_mri_folders(subject_id, mri_dir, subject_dir):
    """
    Creates MRI folder structure
    """

    mri_subfolders = ["sMRI", "fMRI", "dMRI", "EEG", "reports", "logs"]
    [os.makedirs(subject_dir / subfolder, exist_ok=True) for subfolder in mri_subfolders]  
    fnames = os.listdir(mri_dir)

    dest_paths = {"_t1w_": subject_dir / "sMRI" / "raw_anat_T1.nii",
                "_3dt2_": subject_dir / "sMRI" / "raw_anat_T2.nii",
                "_3_1_fmri": subject_dir / "fMRI" / "raw_func_s1.nii",
                "_4_1_fmri": subject_dir / "fMRI" / "raw_func_s2.nii",
                "_dti_32.rec": subject_dir / "dMRI" / f"raw_dwi.rec",
                "_dti_32.par": subject_dir / "dMRI" / f"raw_dwi.par"
                }   

    for fname in fnames:
        if fname.endswith(".nii"):
            ## extract subject information
            if "smartbrain" in fname:
                short_id = fname[:2]
                assert short_id == subject_id[:2], "Subject ID does not match with MRI ID."
                meas_date = datetime.strptime(fname[3:11], '%Y%m%d').date()

            ## extract structural images
            if "_t1w_" in fname and not dest_paths["_t1w_"].exists():
                print("Moving sMRI (T1) data ...")
                shutil.copy(mri_dir / fname, subject_dir / "sMRI")
                shutil.move(subject_dir / "sMRI" / fname, subject_dir / "sMRI" / "raw_anat_T1.nii")

            if "_3dt2_" in fname and not dest_paths["_3dt2_"].exists():
                print("Moving sMRI (T2) data ...")
                shutil.copy(mri_dir / fname, subject_dir / "sMRI")
                shutil.move(subject_dir / "sMRI" / fname, subject_dir / "sMRI" / "raw_anat_T2.nii")
                
            ## extract functional images
            if "_3_1_fmri.nii" in fname and not dest_paths["_3_1_fmri"].exists():
                print("Moving fMRI data (session 1) ...")
                shutil.copy(mri_dir / fname, subject_dir / "fMRI")
                shutil.move(subject_dir / "fMRI" / fname, subject_dir / "fMRI" / "raw_func_s1.nii")
            if "_4_1_fmri.nii" in fname and not dest_paths["_4_1_fmri"].exists():
                print("Moving fMRI data (session 2) ...")
                shutil.copy(mri_dir / fname, subject_dir / "fMRI")
                shutil.move(subject_dir / "fMRI" / fname, subject_dir / "fMRI" / "raw_func_s2.nii")

        ## extract diffusion images
        if fname.endswith("_dti_32.rec") and not dest_paths["_dti_32.rec"].exists():
            print("Moving dMRI data ...")
            shutil.copy(mri_dir / fname, subject_dir / "dMRI")
            shutil.move(subject_dir / "dMRI" / fname, subject_dir / "dMRI" / "raw_dwi.rec")
        
        if fname.endswith("_dti_32.par") and not dest_paths["_dti_32.par"].exists():
            shutil.copy(mri_dir / fname, subject_dir / "dMRI")
            shutil.move(subject_dir / "dMRI" / fname, subject_dir / "dMRI" / "raw_dwi.par")

        if fname.endswith(".log") and not (subject_dir / "logs" / f"{fname[:-18]}.log").exists():
            print("Moving log files ...")
            shutil.copy(mri_dir / fname, subject_dir / "logs")
            shutil.move(subject_dir / "logs" / fname, subject_dir / "logs" / f"{fname[:-18]}.log")

## eeg data
def _create_eeg_folders(subject_id, eeg_dir, captrack_dir, subject_dir, skip_list):
    """
    Creates EEG folder structure
    """

    eeg_subfolders = [
                    "gpias",
                    "rest.",
                    "rest_v2",
                    "xxxxx",
                    "xxxxy",
                    "omi",
                    "regularity",
                    "audiobook",
                    "movie",
                    "reports",
                    "captrack"
                    ]
    
    [os.makedirs(subject_dir / "EEG" / subfolder, exist_ok=True) for subfolder in eeg_subfolders]
    dest_paths = {paradigm: subject_dir / "EEG" / paradigm for paradigm in eeg_subfolders[:-2]}

    fnames = os.listdir(eeg_dir)

    if skip_list is None:
        subfolders = eeg_subfolders[:-2]
    else:
        subfolders = [i for i in eeg_subfolders if i not in skip_list][:-2]


    for fname in fnames:    
        if fname.startswith(subject_id):
            for paradigm in subfolders:
                if f"_{paradigm}" in fname or f"-{paradigm}" in fname:
                    if not (dest_paths[paradigm] / paradigm).exists():
                        print(f"Moving EEG paradigm {paradigm} ...")
                        print(f"copying from {eeg_dir / fname} to {subject_dir / 'EEG' / paradigm}")
                        ipdb.set_trace()
                        shutil.copy(eeg_dir / fname, subject_dir / "EEG" / paradigm)

    ## only for rest. -> rest_v1
    try:
        shutil.move(subject_dir / "EEG" / "rest.", subject_dir / "EEG" / "rest_v1")
    except:
        print("rest_v1 already exist.")
        print(subject_dir / "EEG" / "rest_v1")

    ## captrack data
    captrack_dir = Path(captrack_dir)
    if not captrack_dir.exists():
        captrack_dir.mkdir(exist_ok=True)
    fnames = os.listdir(captrack_dir)
    for fname in fnames:
        if f"_{subject_id}." in fname:
            shutil.copy(captrack_dir / fname, subject_dir / "EEG" / "captrack")
