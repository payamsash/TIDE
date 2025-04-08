import matplotlib
matplotlib.use('QtAgg')
import matplotlib.pyplot as plt
import pandas as pd
import mne
from mne_icalabel.gui import label_ica_components
from eeg.eeg_preprocessing import preprocessing
from eeg.eeg_processing import run_rs_analysis

mode='beat'
if mode=="payam":
	subject_id = "zfdy"
	subjects_dir="/Users/payamsadeghishabestari/antinomics_clean_codes/subjects"
	site="Zuerich"
	paradigm='rest'
elif mode=="beat":
	subject_id = "50001"
	subjects_dir="/tmp/data_in/"
	site="Regensburg"
	paradigm='rest'

preprocessing(subject_id=subject_id,
                subjects_dir=subjects_dir,
                paradigm=paradigm,
                site=site,
                psd_check=True,
                manual_data_scroll=True,
                run_ica=False,
                manual_ica_removal=False,
                ssp_eog=False,
                ssp_ecg=False,
                create_report=True,
                saving_dir=None,
                verbose="ERROR")
run_rs_analysis(
        subject_id=subject_id,
        subjects_dir=subjects_dir,
        visit='',
        event_ids=None,
        source_analysis=True,
        mri=False,
        subjects_fs_dir=None,
        manual_data_scroll=True,
        automatic_epoch_rejection=False,
        create_report=True,
        saving_dir=None,
        verbose="ERROR",
        overwrite=True
        )
