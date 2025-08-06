# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import os
from pathlib import Path
import time
import random
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from autoreject import AutoReject
from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances

from mne.io import read_raw_fif, read_info
from mne.coreg import Coregistration
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from mne import (set_log_level,
                events_from_annotations,
                annotations_from_events,
                find_events,
                Epochs,
                concatenate_raws,
                make_fixed_length_epochs,
                setup_source_space,
                make_bem_model,
                make_bem_solution,
                make_forward_solution,
                make_ad_hoc_cov,
                compute_covariance,
                open_report,
                concatenate_epochs
                )
from .tools import (load_config,
                    _check_processing_inputs,
                    initiate_logging)

def process(
        subject_id,
        subjects_dir,
        paradigm,
        config_file=None,
        overwrite="warn",
        verbose="ERROR",
        **kwargs
        ):
    
    """ Sensor and source space analysis of the preprocessed resting-state eeg recordings.
        The process could be fully or semi automatic based on user choice.

        Parameters
        ----------
        subject_id : str
            The subject name, if subject has MRI data as well, should be FreeSurfer subject name, 
            then data from both modality can be analyzed at once.
        subjects_dir : path-like | None
            The path to the directory containing the EEG subjects. 
        paradigm : str
            Name of the EEG paradigm. Name of the EEG paradigm, must be one of the: ['rest', 'rest_v1', 'rest_v2', 'gpias', 'xxxxx', 'xxxxy', 'omi', 'regularity']
        config_file: str | Path
            Path to the .yaml config file. If None, the default 'processing-config.yaml' will be used.
        overwrite :  str
            must be one of the ['ignore', 'warn', 'raise'].
        verbose : bool | str | int | None
            Control verbosity of the logging output. If None, use the default verbosity level.
        event_ids: [4, 6]
            The ids used for resting-state eyes close and open, respectively. If only eyes open give the single id.
        manual_data_scroll : bool
            If True, user can interactively select epochs of the recording to be removed.
            If not, this step will be skipped.
        automatic_epoch_rejection: str | None
            If None, no automatic epoch rejection will be performed, the possible options are "ptp", "pyriemann" and "autorreject".
        source_analysis: bool
            If yes, source analysis will be performed, if False only epochs will be saved.
        subjects_fs_dir: str | None
            If None, default fsaverage will be used, otherwise it should be the path to subjects_dir in FS.
        create_report : bool
            If True, a report will be created per recordinng.
        
        Notes
        -----
        .. This script is mainly designed for Antinomics / TIDE projects, however could be 
            used for other purposes.
    """
    
    ## get site
    if not isinstance(subject_id, str): raise TypeError(f"subject_id must be str, got type {type(subject_id).__name} instead.")
    if not isinstance(subjects_dir, (str, Path)): raise TypeError(f"subjects_dir must be str or Path object, got type {type(subjects_dir).__name} instead.")
    subject_dir = Path(subjects_dir) / subject_id
    prep_fname = subject_dir / "preprocessed" / f"raw_{paradigm}.fif"
    info = read_info(prep_fname)
    site = info["experimenter"]
    assert subject_id == info["subject_info"]["first_name"], \
        f"Subject ID mismatch ({subject_id} != {info['subject_info']['first_name']}) between preprocess and processing sections."
    assert paradigm == info['description'], f"paradigm mismatch: {paradigm} != {info['description']}"
    
    ## get values from config file
    if config_file is None:
        yaml_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'processing-config.yaml')
        config = load_config(site, yaml_file)
    else:
        config = load_config(site, config_file)
    
    config.update(kwargs)
    event_ids = config.get("event_ids", [6, 4])
    manual_data_scroll = config.get("manual_data_scroll", True)
    automatic_epoch_rejection = config.get("automatic_epoch_rejection", None)
    source_analysis = config.get("source_analysis", True)
    subjects_fs_dir = config.get("subjects_fs_dir", None)
    create_report = config.get("create_report", True)

    ## only check inputs
    _check_processing_inputs(manual_data_scroll,
                                automatic_epoch_rejection,
                                source_analysis,
                                subjects_fs_dir,
                                create_report,
                                overwrite,
                                verbose
                                )
    ## start logging
    logging = initiate_logging(
                                subject_dir / "logs" / f"{paradigm}_processing.log",
                                config,
                                analysis_type="processing"
                                )
    logging.info(f"Processing script initiated on subject {subject_id}, {paradigm} paradigm.")
    set_log_level(verbose=verbose)

    raw = read_raw_fif(prep_fname, preload=True)
    logging.info(f"Preprocessed raw loaded to memory.")
    
    ## the real part
    epochs_dir = subject_dir / "epochs"
    if paradigm.startswith("rest"):
        epochs_list = run_rs_processing(raw, event_ids, logging)
        if len(epochs_list) == 2:
            epochs_eo, epochs_ec = epochs_list
            logging.info(f"Eyes open and close events are detected.")
            if manual_data_scroll:
                epochs_eo.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
                epochs_ec.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
                if len(epochs_eo.info["bads"]) > 0:
                    logging.info(f"Channel(s) {epochs_eo.info['bads']} were marked as bad and interpolated.")
                    epochs_eo.interpolate_bads()
                if len(epochs_ec.info["bads"]) > 0:
                    logging.info(f"Channel(s) {epochs_ec.info['bads']} were marked as bad and interpolated.")
                    epochs_ec.interpolate_bads()

            epochs_eo = reject_epochs(epochs_eo, automatic_epoch_rejection)
            epochs_ec = reject_epochs(epochs_ec, automatic_epoch_rejection)
            
            if not automatic_epoch_rejection is None:
                logging.info(f"Bad epoch rejection is done based on {automatic_epoch_rejection} method.")

            epochs_eo.save(fname=epochs_dir / f"epochs-{paradigm}-eo.fif", overwrite=True)
            epochs_ec.save(fname=epochs_dir / f"epochs-{paradigm}-ec.fif", overwrite=True)
            logging.info(f"Eyes open and close epochs are saved.")

        else:
            epochs_eo = epochs_list[0]
            if manual_data_scroll:
                epochs_eo.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
                if len(epochs_eo.info["bads"]) > 0:
                    logging.info(f"Channel(s) {epochs_eo.info['bads']} were marked as bad and interpolated.")
                    epochs_eo.interpolate_bads()
            epochs_eo = reject_epochs(epochs_eo, automatic_epoch_rejection)
            if not automatic_epoch_rejection is None:
                logging.info(f"Bad epoch rejection is done based on {automatic_epoch_rejection} method.")
            
            epochs_eo.save(fname=epochs_dir / f"epochs-{paradigm}-eo.fif", overwrite=True)
            logging.info(f"Eyes open epochs are saved.")

    else:
        epochs = run_erp_processing(raw, logging)
        if manual_data_scroll:
            epochs.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
            if len(epochs.info["bads"]) > 0:
                logging.info(f"Channel(s) {epochs.info['bads']} were marked as bad and interpolated.") 
                epochs.interpolate_bads()
            
        epochs = reject_epochs(epochs, automatic_epoch_rejection)
        if not automatic_epoch_rejection is None:
            logging.info(f"Bad epoch rejection is done based on {automatic_epoch_rejection} method.")
        epochs.save(fname=epochs_dir / f"epochs-{paradigm}.fif", overwrite=True)
        logging.info(f"Epochs are saved.")

        ## save epochs and evokeds
        evokeds = epochs.average(by_event_type=True)
    
    ## source analysis 
    if len(raw.info["projs"]) == 0:
        raw.set_eeg_reference("average", projection=True)

    if paradigm.startswith("rest") and len(epochs_list) == 2:
        epochs_concat = concatenate_epochs(epochs_list)
    elif paradigm.startswith("rest") and len(epochs_list) == 1:
        epochs_concat = epochs_eo
    else:
        epochs_concat = epochs
    
    if source_analysis:
        inv = run_source_analysis(epochs_concat, subjects_fs_dir, logging)
        write_inverse_operator(
                                fname=subject_dir / "inv" / f"{paradigm}-inv.fif",
                                inv=inv,
                                overwrite=True
                                )
        logging.info("Inverse operator saved in subjects directory.")
    
    ## create a report
    if create_report:
        print("Creating report...\n")
        logging.info(f"creating report...")
        fname_report = subject_dir / "reports" / f"{paradigm}.h5"
        report = open_report(fname_report)
        report.add_epochs(epochs=epochs_concat, psd=False, title='Epochs')

        if raw.info["description"] in ["gpias", "omi", "xxxxx", "xxxxy"]:
            report.add_evokeds(evokeds)
        
        ## saving
        report.save(fname=fname_report.with_suffix('.html'), open_browser=False, overwrite=True)
    
    print("\033[32mEEG data were processed sucessfully!\n")
    logging.info(f"Analysis finished without an error!")


def run_rs_processing(raw, event_ids, logging):

    print("Creating epochs...\n")
    logging.info(f"Creating epochs...")

    try:
        events, events_dict = events_from_annotations(raw)
    except:
        events = find_events(raw) # probably for dublin

    print(events)

    if len(events) == 0:
        logging.info("This recording is only eyes open or eyes closed.")
        both_conditions = False
        tmin = 5
        raw.crop(tmin=tmin)
        logging.info(f"{tmin} seconds are cropped from beginning of the data.")
        epochs_eo = make_fixed_length_epochs(raw, duration=2) 
        epochs_eo.load_data()
        logging.info(f"{len(epochs_eo)} fixed length (2s) eyes-open epochs are created.")

    elif len(events) < 4:
        logging.info("This recording is only eyes open or eyes closed.")
        both_conditions = False
        tmin = max(np.squeeze(events)[-1] / 250 + 3, 5) # 3 seconds skip
        raw.crop(tmin=tmin)
        logging.info(f"{tmin} seconds are cropped from beginning of the data.")
        epochs_eo = make_fixed_length_epochs(raw, duration=2)
        epochs_eo.load_data()
        logging.info(f"{len(epochs_eo)} fixed length (2s) eyes-open epochs are created.")

    elif len(events) > 3:
        both_conditions = True
        events_ec = events[:, 0][events[:, 2] == event_ids[0]]  ## eyes closed
        events_eo = events[:, 0][events[:, 2] == event_ids[1]]  ## eyes open

    # add skip couple of seconds
    if both_conditions:
        if len(events_ec) != len(events_eo):
            raise ValueError(f"Number of eyes close and eyes open events don't match. {len(events_ec)} != {len(events_eo)}")

        logging.info(f"{len(events_eo)} events for eyes open and {len(events_ec)} events for eyes closed are detected.")
        raws_ec, raws_eo = [], []
        if events_ec[0] < events_eo[0]:
            mean_dist = int(2 * np.mean(events_eo - events_ec))
            events_ec = np.append(events_ec, events_ec[-1] + mean_dist)

            for ec_s, eo_s in zip(events_ec[:-1], events_eo):
                tmin = ec_s / raw.info["sfreq"] + 3 # skip few seconds
                tmax = eo_s / raw.info["sfreq"] 
                raws_ec.append(raw.copy().crop(tmin=tmin, tmax=tmax))
            for ec_o, ec_s in zip(events_eo, events_ec[1:]):
                tmin = ec_o / 250 + 3 # skip few seconds
                tmax = ec_s / 250
                raws_eo.append(raw.copy().crop(tmin=tmin, tmax=tmax))

        if events_ec[0] > events_eo[0]:
            mean_dist = int(2 * np.mean(events_ec - events_eo))
            events_eo = np.append(events_eo, events_eo[-1] + mean_dist)

            for ec_o, eo_c in zip(events_eo[:-1], events_ec):
                tmin = ec_o / raw.info["sfreq"] + 3 # skip few seconds
                tmax = eo_c / raw.info["sfreq"] 
                raws_eo.append(raw.copy().crop(tmin=tmin, tmax=tmax))
            for ec_c, ec_o in zip(events_ec, events_eo[1:]):
                tmin = ec_c / 250 + 3 # skip few seconds
                tmax = ec_o / 250
                raws_ec.append(raw.copy().crop(tmin=tmin, tmax=tmax))

        epochs_ec, epochs_eo = [make_fixed_length_epochs(
                                                        concatenate_raws(raw_e),
                                                        duration=2
                                                        ) for raw_e in [raws_ec, raws_eo]]
        return [epochs_eo, epochs_ec]
    else:
        return [epochs_eo]


def run_erp_processing(raw, logging):
    
    match raw.info["description"]:
        case "gpias":
            baseline = None
        case "omi" | "xxxxx" | "xxxxy":
            baseline = (None, 0)

    # if raw.info["experimenter"] == "Austin":
    #     events = find_events(raw)
    #     event_annots = annotations_from_events(events, raw.info["sfreq"])
    #     raw.set_annotations(event_annots)

    events, event_ids = events_from_annotations(raw)
    logging.info("Creating epochs...")
    epochs = Epochs(raw=raw,
                    events=events,
                    event_id=event_ids,
                    tmin=-0.2,
                    tmax=0.5,
                    reject_by_annotation=True,
                    baseline=baseline
                    )
    del raw
    return epochs
    

def reject_epochs(epochs, automatic_epoch_rejection):    

    ## drop and save epochs
    if automatic_epoch_rejection is None:
        pass
    if automatic_epoch_rejection == "ptp":
        reject = dict(eeg=40e-6)
        flat = dict(eeg=1e-7)
        epochs.drop_bad(reject=reject, flat=flat)
    if automatic_epoch_rejection == "autoreject":
        ar = AutoReject()
        ar.fit(epochs.load_data())
        epochs = ar.transform(epochs)
    if automatic_epoch_rejection == "pyriemann":
        train_covs = int(0.7 * len(epochs))  # nb of matrices to train the potato (70%)
        train_set = [random.randint(0, len(epochs)) for _ in range(train_covs)]
        covs = Covariances(estimator="lwf").transform(epochs.get_data())
        potato = Potato(metric="riemann", threshold=3, n_iter_max=100).fit(covs[np.array(train_set)])
        p_labels = potato.predict(covs)
        bad_idxs = np.where(p_labels == 0)[0]
        epochs.drop(bad_idxs)
    return epochs
    
            
def run_source_analysis(epochs, subjects_fs_dir, logging):

    if epochs.info["description"].startswith("rest"):
        epochs = epochs[0]
        print("Using ad hoc noise covariance for the recording ...\n")
        logging.info("Using ad hoc noise covariance for the recording...")
        noise_cov = make_ad_hoc_cov(epochs.info)
    else:
        noise_cov = compute_covariance(epochs)

    if subjects_fs_dir is None:
        kwargs = {
                    "subject": "fsaverage",
                    "subjects_dir": subjects_fs_dir
                }
        
        print("Loading MRI information of Freesurfer template subject ...\n")
        logging.info("Loading MRI information of Freesurfer template subject...")
        fs_dir = fetch_fsaverage()
        trans = fs_dir / "bem" / "fsaverage-trans.fif"
        src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
        bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    else:
        kwargs = {
                    "subject": epochs.info["subject_info"]["first_name"],
                    "subjects_dir": subjects_fs_dir
                }

        print("Setting up bilateral hemisphere surface-based source space with subsampling ...\n")
        logging.info("Setting up bilateral hemisphere surface-based source space with subsampling ...")
        src = setup_source_space(**kwargs)
        print("Creating a BEM model for subject ...\n")
        logging.info("Creating a BEM model for subject ...")
        bem_model = make_bem_model(**kwargs)  
        bem = make_bem_solution(bem_model)
        print("Coregistering MRI with a subjects head shape ...\n")
        logging.info("Coregistering MRI with a subjects head shape ...")
        coreg = Coregistration(epochs.info, fiducials='auto', **kwargs)
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations=40, nasion_weight=2.0) 
        coreg.omit_head_shape_points(distance=5.0 / 1000)
        coreg.fit_icp(n_iterations=40, nasion_weight=10)
        trans = coreg.trans

    print("Computing forward solution ...\n")
    logging.info("Computing forward solution ...")
    fwd = make_forward_solution(epochs.info,
                                trans=trans,
                                src=src,
                                bem=bem,
                                meg=False,
                                eeg=True
                                )
    print("Computing the minimum-norm inverse solution ...\n")
    logging.info("Computing the minimum-norm inverse solution ...")
    inverse_operator = make_inverse_operator(epochs.info,
                                            fwd,
                                            noise_cov
                                            )
    return inverse_operator
