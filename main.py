from shutil import copyfile
import warnings
import pandas as pd
from pathlib import Path
from eeg.eeg_preprocessing import preprocessing
import logging
from tqdm import tqdm
from eeg.eeg_processing import run_rs_analysis
import ipdb

logging.basicConfig(level=logging.INFO)

def copy_flat_to_preprocessing_file_structure(inputpath, outputpath):
    """
    Given a file structure like this:
    $ ls 'inputpath'
    con10.eeg   con12.vmrk  con2.vhdr  con5.vhdr  con9.vhdr  
    etc., and an empty outputpath

    copy_flat_to_preprocessing_file_structure(inputpath,outputpath)
    will produce

    $ ls subjects_dir/
    con1   con11  con13  con3  con5  pat1   pat11  pat13  pat15  pat17  pat19  pat20  pat22  pat24  pat26  pat28
    $ ls subjects_dir/con1/EEG/rest/
    con1.eeg  con1.vhdr  con1.vmrk

    this format is what is needed for the preprocessing routine in

    from eeg.eeg_preprocessing import preprocessing

    (see also the doc on the main_preproc routine)

    """
    inputpath=Path(inputpath).expanduser() #expanduser: allow for Tilde ~ in path names
    outputpath=Path(outputpath).expanduser()
    if len([x for x in inputpath.glob('*.eeg')])==0:
        print(f'No .eeg files in {inputpath}')

    for source in inputpath.glob('*.eeg'):
        for ext in ['.eeg','.vhdr','.vmrk']:
            patient_name = source.stem
            dest = outputpath/Path(patient_name)/'EEG'/Path('rest')/Path(patient_name+ext)
#           dest = inputpath.parent/'project'/'EEG'/Path(patient_name)/Path('rest')/Path(patient_name+'_rest'+ext)
            src = source.parent/(source.stem+ext)
            dest.parent.mkdir(parents=True,exist_ok=True)
            logging.info(f"{src},{dest}")
            try:
                copyfile(src,dest)
            except FileNotFoundError as err:
                print(f'not found:{src}')

def main_preproc(subjects_dir):
    """
    The eeg-data contined in subjects_dir (produced with copy_flat_to_preprocessing_file_structure) is preprocessed.
    """
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
                        #saving_dir='/home/toedtli/Tinnitus/Antinomics_Github/testout',#/home/toedtli/Tinnitus/Data/tide/project',
                        overwrite=True,
                        #progress=progress,
                        verbose="ERROR")

def main():
    tidepath = Path('/home/toedtli/Tinnitus/Data/tide/')
    subjects_dir = tidepath/'project'
    inputpath1 = tidepath/'project_n'
    outputpath = subjects_dir 
    copy_flat_to_preprocessing_file_structure(inputpath1,outputpath)

#    inputpath2 = tidepath/'project_t'
#    outputpath = subjects_dir 
#    copy_flat_to_preprocessing_file_structure(inputpath2,outputpath)
    main_preproc(subjects_dir)

    subjects_ids = [x.name for x in subjects_dir.glob('*')]
    for subject_id in tqdm(subjects_ids):
        run_rs_analysis(subject_id,subjects_dir,overwrite=True,write_feather=True)#,paradigm='rest'

def test_preproc(inputpath='.',outputpath='/tmp/data_out/'):
    if len(list(inputpath.glob('*.eeg')))<2:
        logging.warning(f'Expected lots of .eeg, .vhdr and .vmrk-files in the folder {inputpath}...?')
    outputpath = Path(basepath).expanduser()/'data_out'/'subjects'
    if outputpath.exists():
        logging.warning(f'Directory {outputpath} already exists. Output will overwrite.')
    logging.info(f'inputpath:{inputpath},\n{outputpath}:outputpath')
    outputpath.mkdir(exist_ok=True,parents=True)
    copy_flat_to_preprocessing_file_structure(inputpath,outputpath)

    main_preproc(outputpath)

    subjects_ids = [x.name for x in outputpath.glob('*')]
    for subject_id in tqdm(subjects_ids):
        run_rs_analysis(subject_id,outputpath,overwrite=True)#,paradigm='rest'


if __name__=='__main__':
    #this code runs in two versions: a test/demo version (alternative 1), and with a larger dataset project_n
    config_alternative = 1
    if config_alternative==1:
        basepath = '/tmp/' 
        #the following test setup is suggested: 
        # copy the folder data_in/ to basepath:: cp -r data_in/ /tmp/
        # copy the con1.eeg-file into /tmp/data_in/ and remove con1_replace_this_file.eeg 
        #then run python main.py
        inputpath = Path(basepath).expanduser()/'data_in'
        outputpath = Path(basepath).expanduser()/'data_out'
        test_preproc(inputpath,outputpath)
    elif config_alternative==2:
        main()
    print('Main execution is done.')
