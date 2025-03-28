# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

## source stuff
## sensor connectivity via coherence 
## source connectivity via coherence


import warnings
from pathlib import Path
from tqdm import tqdm
import pandas as pd
from mne_features.feature_extraction import extract_features
from mne import (set_log_level,
                    read_epochs,
                    concatenate_epochs)


def extract_eeg_features(
        subject_id,
        subjects_dir=None,
        features=["pow_freq_bands", "spect_entropy"],
        paradigm="rest_v1",
        atlas="aparc",
        mri=False,
        saving_dir=None,
        verbose="ERROR"
        ):
    
    set_log_level(verbose=verbose)
    
    ## will update this once I add source and connectivity
    progress = tqdm(total=len(features),
                    desc="",
                    ncols=50,
                    colour="cyan",
                    bar_format='{l_bar}{bar}'
                    )
    
    ## check inputs
    assert isinstance(features, list), "features must be a list object."
    
    if not paradigm.startswith("rest"):
        raise ValueError("paradigm should be either 'both' or starts with 'rest'")
    
    atlases = ["aparc", "aparc.a2009s"]
    if atlas not in atlases:
        raise ValueError(f"atlas should be one of {atlases}")
    
    assert isinstance(mri, bool), "mri must be boolean."

    if subjects_dir == None:
        subjects_dir = Path.cwd().parent / "subjects"
    else:
        subjects_dir = Path(subjects_dir)

    ## read the epochs
    epochs_fnames = []
    epochs_eo_fname = subjects_dir / subject_id / "EEG" / paradigm / "epochs-eo-epo.fif"
    epochs_ec_fname = subjects_dir / subject_id / "EEG" / paradigm / "epochs-ec-epo.fif"
    
    if not epochs_eo_fname.exists():
        raise ValueError(f"Eyes open epochs don't exist in the directory")
    else:
        epochs_fnames.append(epochs_eo_fname)

    if not epochs_ec_fname.exists():
        warnings.warn(f"Eyes closed epochs dont exist in this directory")
    else:
        epochs_fnames.append(epochs_ec_fname)
    
    epochs, eye_labels = [], []
    for ep_fname, title in zip(epochs_fnames, ["open", "close"]):
        eps = read_epochs(ep_fname, preload=True)
        eps.pick("eeg")
        eps.drop_bad()
        epochs.append(eps)
        eye_labels += len(eps) * [title]

    epochs = concatenate_epochs(epochs)
    site = epochs.info["subject_info"]["his_id"]

    ## initialize feature extraction
    ch_names = epochs.info["ch_names"]
    kwargs = {
            "X": epochs.get_data(),
            "sfreq": epochs.info["sfreq"],
            "ch_names": ch_names,
            "return_as_df": False
            }
    freq_bands = {
            "delta": (0.5, 4),
            "theta": (4, 8),
            "alpha": (8, 13),
            "beta": (13, 30),
            "lower_gamma": (30, 45),
            "upper_gamma": (45, 80)
            }
    
    dfs = []
    for feature in features: 
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
                y = y.reshape(len(epochs), len(ch_names), len(freq_bands.keys()))
                columns = [f"power_ch_{j}_freq_{k}" for j in ch_names for k in freq_bands]
                df = pd.DataFrame(y, columns=columns)
                dfs.append(df)

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
                dfs.append(df)

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
                dfs.append(df)

            case "hurst_exp" | "app_entropy" | "samp_entropy" | "decorr_time" | "hjorth_mobility" | "hjorth_complexity" | "higuchi_fd" | "katz_fd" | "spect_entropy" | "svd_entropy" | "svd_fisher_info":
                y = extract_features(selected_funcs=[feature],
                                funcs_params=None,
                                **kwargs
                                )
                columns = [f"{feature}_ch_{j}" for j in ch_names]
                df = pd.DataFrame(y, columns=columns)
                dfs.append(df)

    
    ## add source stuff here


    ## add connectivity stuff here

    ## finalize df and save
    df = pd.concat(dfs)
    df["eyes"] = eye_labels
    df["site"] = len(df) * site # we will need this one for domain adaptation
    df["subject_id"] = len(df) * subject_id

    if saving_dir is None:
        saving_dir = subjects_dir / subject_id / "EEG" / f"{paradigm}" / "features.csv"
    df.to_csv(saving_dir)