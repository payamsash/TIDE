import yaml
import logging
from pathlib import Path
from mne import sys_info
from mne.io import read_raw

def load_config(site,
                config_file):
    """
    Loads config file.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config.get(site, {})


def initiate_logging(logfile, config, analysis_type="preprocessing"):
    """
    Start logging file with system and code information.
    """

    ## add system and config information
    yaml_str = yaml.dump(config, default_flow_style=False)
    with open(logfile, 'w') as f:
        f.write('*' * 100 + '\n')
        f.write('{:^100}\n'.format(f'System Information'))
        f.write('*' * 100 + '\n\n')
        sys_info(fid=f)
        f.write('\n')

        f.write('*' * 100 + '\n')
        f.write('{:^100}\n'.format(f'Config Information'))
        f.write('*' * 100 + '\n\n')
        f.write(yaml_str)
        f.write('\n')

        f.write('*' * 100 + '\n')
        f.write('{:^100}\n'.format(f'{analysis_type.upper()}'))
        f.write('*' * 100 + '\n\n')

    logging.basicConfig(
                    filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
    return logging

def _check_preprocessing_inputs(fname,
                                subject_id,
                                subjects_dir,
                                site,
                                paradigm,
                                psd_check,
                                manual_data_scroll,
                                run_ica,
                                manual_ica_removal,
                                ssp_eog,
                                ssp_ecg,
                                create_report,
                                verbose,
                                overwrite
                                ):
    """
    Checks input variables, raise or warn some messages.
    """

    ## initial checks
    if not isinstance(fname, (str, Path)): raise TypeError(f"fname must be str or Path object, got type {type(fname).__name} instead.")
    if not isinstance(subject_id, str): raise TypeError(f"subject_id must be str, got type {type(subject_id).__name} instead.")
    if not isinstance(subjects_dir, (str, Path)): raise TypeError(f"subjects_dir must be str or Path object, got type {type(subjects_dir).__name} instead.")
    
    sites = ["Austin", "Dublin", "Ghent", "Illinois", "Regensburg", "Tuebingen", "Zuerich"] 
    if not site in sites: raise ValueError(f"site must be one of the {sites}.")
    
    paradigms = ["gpias", "xxxxx", "xxxxy", "omi", "regularity"]
    if not (paradigm in paradigms or paradigm.startswith("rest")):
        raise ValueError(f"paradigm must be one of the {paradigms} or should start with 'rest'.")

    for var_name, var_value in {
                                "psd_check": psd_check,
                                "manual_data_scroll": manual_data_scroll,
                                "run_ica": run_ica,
                                "manual_ica_removal": manual_ica_removal,
                                "ssp_eog": ssp_eog,
                                "ssp_ecg": ssp_ecg,
                                "create_report": create_report
                                }.items():
        if not isinstance(var_value, bool): raise TypeError(f"{var_name} must be boolean, got type {type(var_value).__name__} instead.")

    verboses = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if not (verbose in verboses or isinstance(verbose, bool)): raise ValueError(f"verbose must be one of the {verboses} or boolean.") 

    overwrite_options = ["warn", "ignore", "raise"]
    if not overwrite in overwrite_options: raise ValueError(f"overwrite must be one of the {overwrite_options}.")
        
    # site and data format checks
    ext_site_map = {
                    ".mff": "Austin",
                    ".bdf": "Dublin",
                    ".cdt": "Illinois",
                    ".cnt": "Ghent",
                    ".vhdr": ["Zuerich", "Regensburg", "Tuebingen"],
                    }

    for ext, expected in ext_site_map.items():
        if Path(fname).suffix == ext:
            if isinstance(expected, list):
                assert site in expected, f"site is not selected correctly for {ext}."
            else:
                assert site == expected, f"site is not selected correctly for {ext}."
            break

def _check_processing_inputs(manual_data_scroll,
                                automatic_epoch_rejection,
                                source_analysis,
                                subjects_fs_dir,
                                create_report,
                                overwrite,
                                verbose,
                                ):
    """
    Checks input variables, raise or warn some messages.
    """

    ## initial checks
    for var_name, var_value in {
                                "manual_data_scroll": manual_data_scroll,
                                "source_analysis": source_analysis,
                                "create_report": create_report
                                }.items():
        if not isinstance(var_value, bool): raise TypeError(f"{var_name} must be boolean, got type {type(var_value).__name__} instead.")

    automatic_epoch_rejection_options = ["ptp", "autoreject", "pyriemann"]
    if not (automatic_epoch_rejection in automatic_epoch_rejection_options or automatic_epoch_rejection is None):
        raise ValueError(f"automatic_epoch_rejection must be None or one of the {automatic_epoch_rejection_options}.")
    
    if not (subjects_fs_dir is None or isinstance(subjects_fs_dir, (str, Path))): 
        raise TypeError("subjects_fs_dir must be None or a path to FS subjects directory.")
    
    verboses = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if not (verbose in verboses or isinstance(verbose, bool)): raise ValueError(f"verbose must be one of the {verboses} or boolean.") 

    overwrite_options = ["warn", "ignore", "raise"]
    if not overwrite in overwrite_options: raise ValueError(f"overwrite must be one of the {overwrite_options}.")

def _check_feature_extraction_inputs(
                                    sensor_space_features,
                                    source_space_power,
                                    sensor_space_connectivity,
                                    source_space_connectivity,
                                    connectivity_method,
                                    subjects_fs_dir,
                                    atlas,
                                    freq_bands,
                                    overwrite,
                                    verbose
                                    ):
    """
    Checks input variables, raise or warn some messages.
    """

    ## initial checks
    if not isinstance(sensor_space_features, list):
        raise TypeError(f"{sensor_space_features} must be list, got type {type(sensor_space_features).__name__} instead.")
    for var_name, var_value in {
                                "source_space_power": source_space_power,
                                "sensor_space_connectivity": sensor_space_connectivity,
                                "source_space_connectivity": source_space_connectivity,
                                }.items():
        if not isinstance(var_value, bool):
            raise TypeError(f"{var_name} must be boolean, got type {type(var_value).__name__} instead.")

    connectivity_method_options = ["coh", "pli", "wpli", "cacoh", "mic", "mim", "gc", "gc_tr"]
    if not connectivity_method in connectivity_method_options:
        raise ValueError(f"method must be one of the {connectivity_method_options}.")
    
    if not (subjects_fs_dir is None or isinstance(subjects_fs_dir, (str, Path))):
        raise TypeError(f"subjects_fs_dir must be None or path to subjects FS directory.")
    if not atlas in ["aparc", "aparc.a2009s"]: raise ValueError(f"atlas must be one of the ['aparc', 'aparc.a2009s'].")
    if not isinstance(freq_bands, dict): raise TypeError(f"freq_bands must be dict, got type {type(freq_bands).__name__} instead.")
    
    verboses = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if not (verbose in verboses or isinstance(verbose, bool)): raise ValueError(f"verbose must be one of the {verboses} or boolean.") 
    overwrite_options = ["warn", "ignore", "raise"]
    if not overwrite in overwrite_options: raise ValueError(f"overwrite must be one of the {overwrite_options}.")
    

def create_subject_dir(subject_id, subjects_dir, site):
    """
    Create a structured directory for a subject under the given subjects directory.
    
    Structure:
    subjects_dir/subject_id/
        ├── orig/
        ├── preprocessed/
        ├── mri/
        ├── inv/
        ├── epochs/
        ├── reports/
        ├── logs/
    """
    base_path = Path(subjects_dir) / subject_id
    subdirs = [
                "orig", "preprocessed", "features",
                "inv", "epochs", "reports", "logs",
                ]

    for subdir in subdirs:
        path = base_path / subdir
        path.mkdir(parents=True, exist_ok=False)

    if site == "Zuerich":
        path = base_path / "captrak"
        path.mkdir(parents=True, exist_ok=False)

def read_vhdr_input_fname(fname):
    """
    Checks .vhdr and .vmrk data to have same names, otherwise fix them.
    """
    try:
        raw = read_raw(fname)
    except:
        with open(fname, "r") as file:
            lines = file.readlines()
        
        lines[5] = f'DataFile={fname.stem}.eeg\n'
        lines[6] = f'MarkerFile={fname.stem}.vmrk\n'

        with open(fname, "w") as file:
            file.writelines(lines)
        with open(f"{fname.with_suffix('')}.vmrk", "r") as file:
            lines = file.readlines()
        lines[4] = f'DataFile={fname.stem}.eeg\n'
        with open(f"{fname.with_suffix('')}.vmrk", "w") as file:
            file.writelines(lines)
        raw = read_raw(fname)
    return raw

def gpias_constants():
    events_dict = { 
                "pre": {
                        "PO70": 11,
                        "PO75": 12,
                        "PO80": 13,
                        "PO85": 14,
                        "PO90": 15,
                        },
                "bbn": {
                        "PO": 21,
                        "GO": 22,
                        "GP": 23,
                        },
                "3kHz": {
                        "PO": 31,
                        "GO": 32,
                        "GP": 33,
                        },
                "8kHz": {
                        "PO": 41,
                        "GO": 42,
                        "GP": 43,
                        },
                "post": {
                        "PO70": 51,
                        "PO75": 52,
                        "PO80": 53,
                        "PO85": 54,
                        "PO90": 55,
                        }
                }
    
    default_thrs = {
                    "Zuerich":
                                {
                                "pre": [850, 1500, 2000, 2300],
                                "bbn": [1150, 1700, 2250],
                                "3kHz": [1150, 1700, 2250],
                                "8kHz": [1150, 1700, 2250],
                                "post": [850, 1500, 2000, 2300]
                                },
                    "Regensburg":
                                {
                                "pre": [0.01, 0.017, 0.025, 0.032],
                                "bbn": [0.014, 0.023, 0.029],
                                "3kHz": [0.016, 0.026, 0.035],
                                "8kHz": [0.016, 0.026, 0.035],
                                "post": [0.01, 0.017, 0.025, 0.032]
                                },
                    "Tuebingen":
                                {
                                "pre": [0.44, 0.74, 1, 1.25],
                                "bbn": [0.61, 0.88, 1.18],
                                "3kHz": [0.61, 0.88, 1.18],
                                "8kHz": [0.61, 0.88, 1.18],
                                "post": [0.44, 0.74, 1, 1.25]
                                }
                    }
    distance = 25
    return events_dict, default_thrs, distance