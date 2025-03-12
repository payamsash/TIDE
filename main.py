from shutil import copyfile
from pathlib import Path
from eeg.eeg_preprocessing import preprocessing
import logging
from tqdm import tqdm
from eeg.eeg_processing import run_rs_analysis

logging.basicConfig(level=logging.INFO)

def copy_flat_to_preprocessing_file_structure(outputpath):
    inputpath1 = tidepath/'project_n'
    inputpath2 = tidepath/'project_t'
    for inputpath in [inputpath1,inputpath2]:
        for source in inputpath.glob('*.eeg'):
            for ext in ['.eeg','.vhdr','.vmrk']:
                patient_name = source.stem
#                dest = inputpath.parent/'project'/'EEG'/Path(patient_name)/Path('rest')/Path(patient_name+'_rest'+ext)
                dest = outputpath/Path(patient_name)/'EEG'/Path('rest')/Path(patient_name+ext)
                src = source.parent/(source.stem+ext)
                dest.parent.mkdir(parents=True,exist_ok=True)
                logging.info(f"{src},{dest}")
                try:
                    copyfile(src,dest)
                except FileNotFoundError as err:
                    print(f'not found:{src}')

def main_preproc(subjects_dir):
    from eeg.eeg_preprocessing import preprocessing
    from eeg.eeg_processing import run_rs_analysis#, update_progress
    from pathlib import Path
    create_report=True
    subjects = [x.name for x in subjects_dir.glob('*')]
    for subject in tqdm(subjects):
        preprocessing(subject_id=subject,
                        subjects_dir=subjects_dir,
                        paradigm="rest",
                        #paradigm="xxxxx",
                        manual_data_scroll=False,
                        run_ica=False, #needs a montage
                        manual_ica_removal=False,
                        eog_correct=False, #gives a nice plot of ICA components that I don't understand
                        pulse_correct=False, #We don't have pulse data, Error: Pulse not in channel list
                        resp_correct=False,
                        create_report=create_report,
                        saving_dir=None,##'/home/toedtli/Tinnitus/Data/tide/project',
                        overwrite=True,
                        #progress=progress,
                        verbose="ERROR")

if __name__=='__main__':
    tidepath = Path('/home/toedtli/Tinnitus/Data/tide/')
    subjects_dir = tidepath/'project'
    #copy_flat_to_preprocessing_file_structure(subjects_dir)
#    main_preproc(subjects_dir)

    subjects_ids = [x.name for x in subjects_dir.glob('*')][-10:]
    for subject_id in tqdm(subjects_ids):
        run_rs_analysis(subject_id,subjects_dir,overwrite=True)#,paradigm='rest'
