# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import os
from pathlib import Path
import zipfile
import numpy as np
import pandas as pd
from scipy.signal import welch
from mne_connectivity import spectral_connectivity_time
from mne_features.feature_extraction import extract_features

from mne import (set_log_level,
                    read_epochs,
                    read_labels_from_annot,
                    extract_label_time_course)
from mne.minimum_norm import (read_inverse_operator,
                                apply_inverse_epochs)
from .tools import (_check_feature_extraction_inputs,
                    load_config,
                    initiate_logging)

def extract_eeg_features(
        subject_id,
        subjects_dir,
        config_file=None,
        overwrite="warn",
        verbose="ERROR",
        **kwargs
        ):
    
    ## get site and paradigm
    if not isinstance(subject_id, str): raise TypeError(f"subject_id must be str, got type {type(subject_id).__name} instead.")
    if not isinstance(subjects_dir, (str, Path)): raise TypeError(f"subjects_dir must be str or Path object, got type {type(subjects_dir).__name} instead.")
    subject_dir = Path(subjects_dir) / subject_id
    ep_fname = subject_dir / "epochs" / f"epochs-rest-eo.fif"
    epochs = read_epochs(ep_fname, preload=True)
    epochs.pick(picks="eeg")
    info = epochs.info
    paradigm = info["description"]

    assert subject_id == info["subject_info"]["first_name"], \
        f"Subject ID mismatch ({subject_id} != {info['subject_info']['first_name']}) between preprocess and processing sections."
    assert paradigm == "rest", f"paradigm mismatch: {paradigm} != {info['description']}"
    
    ## get values from config file
    if config_file is None:
        yaml_file = os.path.join(os.path.dirname(__file__), 'features-config.yaml')
        config = load_config("general", yaml_file)
    else:
        config = load_config("general", config_file)
    
    config.update(kwargs)
    sensor_space_features = config.get("sensor_space_features", None)
    source_space_power = config.get("source_space_power", True)
    sensor_space_connectivity = config.get("sensor_space_connectivity", True)
    source_space_connectivity = config.get("source_space_connectivity", True)
    connectivity_method = config.get("connectivity_method", "coh")
    subjects_fs_dir = config.get("subjects_fs_dir", None)
    atlas = config.get("atlas", "aparc")
    freq_bands = config.get("freq_bands", {"delta": [0.5, 4],
                                            "theta": [4, 8],
                                            "alpha": [8, 13],
                                            "beta": [13, 30],
                                            "lower_gamma": [30, 45],
                                            "upper_gamma": [45, 80]
                                            }
                                            )

    ## only check inputs
    _check_feature_extraction_inputs(
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
                                    )
    ## start logging
    logging = initiate_logging(
                                subject_dir / "logs" / f"{paradigm}_features.log",
                                config,
                                analysis_type="feature extraction"
                                )
    logging.info(f"Feature extraction script initiated on subject {subject_id}, {paradigm} paradigm.")
    set_log_level(verbose=verbose)
    
    if ep_fname.stem[-2:] == "eo": eye_label = "open"
    if ep_fname.stem[-2:] == "ec": eye_label = "close"

    ## initialize feature extraction
    ch_names = epochs.info["ch_names"]
    kwargs = {
            "X": epochs.get_data(),
            "sfreq": epochs.info["sfreq"],
            "ch_names": ch_names,
            "return_as_df": False
            }
    
    logging.info(f"{sensor_space_features} are selected to be extracted from epochs in sensor level.")
    for feature in sensor_space_features: 
        match feature:
            case "pow_freq_bands":
                funcs_params = {
                                f"{feature}__freq_bands": freq_bands,
                                f"{feature}__psd_method": "multitaper",
                                f"{feature}__normalize": True
                                } 
                y = extract_features(selected_funcs=[feature],
                                funcs_params=funcs_params,
                                **kwargs
                                )
                columns = [f"power_ch_{j}_freq_{k}" for j in ch_names for k in freq_bands]
                df = pd.DataFrame(y, columns=columns)
                logging.info(f"{feature} is computed per epoch, the data shape is {df.shape}")
                add_and_save_df(df, feature, subject_id, subject_dir, paradigm, info, eye_label)

            case "spect_slope":
                funcs_params = {
                                f"{feature}__fmin": 0.1,
                                f"{feature}__fmax": 100,
                                f"{feature}__psd_method": "multitaper"
                                } 
                y = extract_features(selected_funcs=[feature],
                                    funcs_params=funcs_params,
                                    **kwargs
                                    )
                y = y.reshape(len(epochs), len(ch_names), 4)[:,:,:2] # intercept and slope
                y = y.reshape(len(epochs), len(ch_names) * 2)
                columns_1 = [f"intercept_ch_{j}" for j in ch_names]
                columns_2 = [f"slope_ch_{j}" for j in ch_names]
                columns = columns_1 + columns_2
                df = pd.DataFrame(y, columns=columns)
                logging.info(f"{feature} is computed per epoch, the data shape is {df.shape}")
                add_and_save_df(df, feature, subject_id, subject_dir, paradigm, info, eye_label)

            case "hjorth_complexity_spect" | "hjorth_mobility_spect":
                funcs_params = {
                                f"{feature}__psd_method": "multitaper",
                                f"{feature}__normalize": True
                                }
                y = extract_features(selected_funcs=[feature],
                                funcs_params=funcs_params,
                                **kwargs
                                )
                columns = [f"{feature}_ch_{j}" for j in ch_names]
                df = pd.DataFrame(y, columns=columns)
                logging.info(f"{feature} is computed per epoch, the data shape is {df.shape}")
                add_and_save_df(df, feature, subject_id, subject_dir, paradigm, info, eye_label)

            case "hurst_exp" | "app_entropy" | "samp_entropy" | "decorr_time" | "hjorth_mobility" | "hjorth_complexity" | "higuchi_fd" | "katz_fd" | "spect_entropy" | "svd_entropy" | "svd_fisher_info":
                y = extract_features(selected_funcs=[feature],
                                funcs_params=None,
                                **kwargs
                                )
                columns = [f"{feature}_ch_{j}" for j in ch_names]
                df = pd.DataFrame(y, columns=columns)
                logging.info(f"{feature} is computed per epoch, the data shape is {df.shape}")
                add_and_save_df(df, feature, subject_id, subject_dir, paradigm, info, eye_label)
                
    
    ## add source stuff here
    label_ts_computed = False
    if source_space_power:
        inv = read_inverse_operator(fname=subject_dir / "inv" / f"{paradigm}-inv.fif")
        logging.info(f"Inverse operator found for the subject and loaded to the memory.")
        
        if subjects_fs_dir is None:
            if atlas == "aparc":
                labels = read_labels_from_annot(subject="fsaverage", subjects_dir=None, parc=atlas)[:-1]
            if atlas == "aparc.a2009s":
                labels = read_labels_from_annot(subject="fsaverage", subjects_dir=None, parc=atlas)[:-2]
        else:
            labels = read_labels_from_annot(subject=subject_id, subjects_dir=subjects_fs_dir, parc=atlas)
        
        logging.info(f"Brain labels are extracted from source space data.")
        stcs = apply_inverse_epochs(
                                    epochs,
                                    inverse_operator=inv,
                                    lambda2=1.0 / (3.0 ** 2),
                                    method="dSPM",
                                    label=None,
                                    pick_ori="normal",
                                    return_generator=False
                                    )
        logging.info(f"Inverse operator is applied to each single epoch.")

        label_ts = extract_label_time_course(
                                            stcs,
                                            labels,
                                            inv["src"],
                                            mode="mean_flip",
                                            return_generator=False,
                                            )
        label_ts = np.array(label_ts)
        label_ts_computed = True
        logging.info(f"Label time courses are extracted per brain label, the shape of the extracted array is {label_ts.shape}.")
        n_epochs, n_channels, n_times = label_ts.shape
        reshaped_data = label_ts.reshape(-1, n_times)
        freqs, psd = welch(reshaped_data, info["sfreq"], axis=-1, nperseg=min(256, n_times))

        columns = []
        labels_power = []
        for key, value in freq_bands.items():
            min_freq, max_freq = list(value)
            band_mask = (freqs >= min_freq) & (freqs <= max_freq)
            band_powers = np.trapz(psd[:, band_mask], freqs[band_mask], axis=-1)
            labels_power.append(band_powers.reshape(n_epochs, n_channels))
            columns += [f"{lb.name}_{key}" for lb in labels]

        labels_power = np.concatenate(labels_power, axis=1)
        df = pd.DataFrame(labels_power, columns=columns)
        logging.info(f"Power values are computed per epoch in the {list(freq_bands.keys())} frequency bands in the all brain labels.")
        add_and_save_df(df, "source_space_power", subject_id, subject_dir, paradigm, info, eye_label)

    ## connectivity stuff
    if sensor_space_connectivity:
        chs = info["ch_names"]
        i_lower, j_lower = np.tril_indices_from(np.zeros(shape=(len(chs), len(chs))), k=-1)
        columns = []
        freq_cons = []
        for key, value in freq_bands.items(): 
            con = spectral_connectivity_time(
                                            epochs,
                                            freqs=np.arange(value[0], value[1], 5),
                                            method=connectivity_method,
                                            average=False,
                                            sfreq=info["sfreq"],
                                            mode="cwt_morlet",
                                            fmin=value[0],
                                            fmax=value[1],
                                            faverage=True,
                                            n_cycles=value[1] / 6
                                            )
            con_matrix = np.squeeze(con.get_data(output="dense")) # n_epochs * n_chs * n_chs

            cons = []
            for ep_con in con_matrix:
                ep_con_value = ep_con[i_lower, j_lower]
                cons.append(ep_con_value)
            cons = np.array(cons)
            freq_cons.append(cons)

            con_labels = [f"{chs[i]}_vs_{chs[j]}_{key}" for i, j in zip(i_lower, j_lower)]
            columns += con_labels
        
        freq_cons = np.concatenate(freq_cons, axis=-1)
        df = pd.DataFrame(freq_cons, columns=columns)
        logging.info(f"Connectivity values are computed per epoch in the {list(freq_bands.keys())} frequency bands in all EEG channels.")
        add_and_save_df(df, "sensor_space_connectivity", subject_id, subject_dir, paradigm, info, eye_label)
    
    if source_space_connectivity:
        if label_ts_computed:
            lb_names = [lb.name for lb in labels]
            i_lower, j_lower = np.tril_indices_from(np.zeros(shape=(label_ts.shape[1], label_ts.shape[1])), k=-1)
            columns = []
            freq_cons = []
            for key, value in freq_bands.items(): 
                con = spectral_connectivity_time(
                                                label_ts,
                                                freqs=np.arange(value[0], value[1], 5),
                                                method=connectivity_method,
                                                average=False,
                                                sfreq=info["sfreq"],
                                                mode="cwt_morlet",
                                                fmin=value[0],
                                                fmax=value[1],
                                                faverage=True,
                                                n_cycles=value[1] / 6
                                                )
                con_matrix = np.squeeze(con.get_data(output="dense")) # n_epochs * n_chs * n_chs

                cons = []
                for ep_con in con_matrix:
                    ep_con_value = ep_con[i_lower, j_lower]
                    cons.append(ep_con_value)
                cons = np.array(cons)
                freq_cons.append(cons)

                con_labels = [f"{lb_names[i]}_vs_{lb_names[j]}_{key}" for i, j in zip(i_lower, j_lower)]
                columns += con_labels
            
            freq_cons = np.concatenate(freq_cons, axis=-1)
            df = pd.DataFrame(freq_cons, columns=columns)
            logging.info(f"Connectivity values are computed per epoch in the {list(freq_bands.keys())} frequency bands in the all brain labels.")
            add_and_save_df(df, "source_space_connectivity", subject_id, subject_dir, paradigm, info, eye_label)
        
        else:
            raise ValueError("set source_space_power=True to avoid double computation of source estimate values.")

    logging.info(f"Feature extraction finished without an error!")

def add_and_save_df(df, feature, subject_id, subject_dir, paradigm, info, eye_label):
    df["eyes"] = eye_label
    df["site"] = info["experimenter"] 
    df["subject_id"] = subject_id
    csv_fname = subject_dir / "features" / f"features_{feature}_{paradigm}.csv"
    zip_fname = subject_dir / "features" / f"features_{feature}_{paradigm}.zip"
    df.T.to_csv(csv_fname)

    with zipfile.ZipFile(zip_fname, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_fname, os.path.basename(csv_fname))

    os.remove(csv_fname)

def get_full_sensor_features_list():
    full_list = ["pow_freq_bands", "spect_slope", "hjorth_complexity_spect",
                "hjorth_mobility_spect", "hurst_exp", "app_entropy",
                "samp_entropy", "decorr_time", "hjorth_mobility",
                "hjorth_complexity", "higuchi_fd", "katz_fd",
                "spect_entropy", "svd_entropy", "svd_fisher_info"]
    print('\n'.join(full_list))