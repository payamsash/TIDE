import yaml
import logging
from pathlib import Path
from mne import sys_info, read_info
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
    if site == "Ghent": raise NotImplementedError
    
    paradigms = ["rest", "rest_v1", "rest_v2", "gpias", "xxxxx", "xxxxy", "omi", "regularity"]
    if not paradigm in paradigms: raise ValueError(f"paradigm must be one of the {paradigms}.")

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
                    ".cdt": "Illinoise",
                    ".ds": "Tuebingen",
                    ".vhdr": ["Zuerich", "Regensburg"],
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
                                verbose,
                                overwrite
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
    

def create_subject_dir(subject_id, subjects_dir):
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
                "orig", "preprocessed", "mri",
                "inv", "epochs", "reports", "logs",
                "captrak"
                ]

    for subdir in subdirs:
        path = base_path / subdir
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


def _detect_peaks(pk_heights,
                peak_idxs,
                stim_ch,
                times,
                height_limits,
                events_dict,
                stim_key,
                plot_peaks=True
                ):
    events = []
    if plot_peaks:
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        ax.plot(times, stim_ch)
        ax.set_title(stim_key)
        
    for key, (lower, upper) in height_limits.items():
        pk_idxs = np.where((pk_heights >= lower) & (pk_heights < upper))[0]
        sub_events = np.zeros(shape=(len(pk_idxs), 3), dtype=int)
        sub_events[:, 0] = times[peak_idxs[pk_idxs]] * 250
        sub_events[:, 1] = 0
        sub_events[:, 2] = events_dict[f"{key[:4]}_{stim_key[3:]}"]
        events.append(sub_events) 

        if plot_peaks:
            ax.scatter(times[peak_idxs[pk_idxs]], pk_heights[pk_idxs])

        if not len(pk_idxs) in [25, 100]:
            warnings.warn(f"\033[91mThe number of detected triggeres for {key} in {stim_key} is " \
                            "not as expected (25 or 100), its {len(pk_idxs)}, try adjusting the threshold!", UserWarning) 
            
    return np.concatenate(np.array(events), axis=0) 