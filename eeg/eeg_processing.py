# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch
import os
from tools import load_config, _check_processing_inputs, initiate_logging
import warnings
from pathlib import Path
import random
from tqdm import tqdm
import time

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from autoreject import AutoReject
from pyriemann.clustering import Potato
from pyriemann.estimation import Covariances

from mne.io import read_raw_fif
from mne.coreg import Coregistration
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from mne import (set_log_level,
                events_from_annotations,
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
                concatenate_epochs,
                read_info
                )

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
            The path to the directory containing the EEG subjects. The folder structure should be
            as following which can be created by running "file_preparation" function:
            subjects_dir/
            ├── sMRI/
            ├── fMRI/
            │   ├── session_1/
            │   ├── session_2/
            ├── dMRI/
            ├── EEG/
                ├── paradigm_1/
                ├── paradigm_2/
                ├── ...
        event_ids: dict | None
            If dict, the keys should be the eyes_close and eyes_open (respectively) and the values should be integar.
            If None, the first event (not new segment) will be assumed to be eyes closed trigger.
        source_analysis: bool
            If yes, source analysis will be performed, if False only epochs will be saved.
        mri: bool
            If True, subject has MRI data which is surface reconstructed via Freesurfer.
        subjects_fs_dir: str │ path-like │ None
            The path for the subects_dir of the FS application. If None, the path will be:
            "/Applications/freesurfer/7.4.1/subjects"
        manual_data_scroll : bool
            If True, user can interactively select epochs of the recording to be removed.
            If not, this step will be skipped.
        create_report : bool
            If True, a report will be created per recordinng.
        saving_dir : path-like | None | bool
            The path to the directory where the preprocessed EEG will be saved, If None, it will be saved 
            in the same path as the raw files. If False, preprocessed data will not be saved.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If None, use the default verbosity level.

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
        f"Subject ID mismatch ({subject_id} != {info["subject_info"]["first_name"]}) between preprocess and processing sections."
    assert paradigm == info["description"], f"paradigm mismatch: {paradigm} != {info["description"]}"
    
    ## get values from config file
    if config_file is None:
        yaml_file = os.path.join(os.path.dirname(__file__), 'processing-config.yaml')
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
    progress = tqdm(total=10,
                    desc="",
                    ncols=50,
                    colour="cyan",
                    bar_format='{l_bar}{bar}'
                    )
    time.sleep(1)
    tqdm.write("Loading preprocessed EEG data ...\n")
    progress.update(1)
    raw = read_raw_fif(prep_fname, preload=True)
    logging.info(f"Preprocessed raw loaded to memory.")
    
    ## the real part
    epochs_dir = subject_dir / "epochs"
    if paradigm.startswith("rest"):
        epochs_list = run_rs_processing(raw, event_ids, progress, logging)
        if len(epochs_list) == 2:
            epochs_eo, epochs_ec = epochs_list
            if manual_data_scroll:
                epochs_eo.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
                epochs_ec.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)

            epochs_eo = reject_epochs(epochs_eo, automatic_epoch_rejection)
            epochs_ec = reject_epochs(epochs_ec, automatic_epoch_rejection)

            epochs_eo.save(fname=epochs_dir / f"epochs-{paradigm}-eo.fif", overwrite=True)
            epochs_ec.save(fname=epochs_dir / f"epochs-{paradigm}-ec.fif", overwrite=True)
            logging.info(f"epochs are saved.")

        else:
            epochs_eo = epochs_list[0]
            if manual_data_scroll:
                epochs_eo.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
            epochs_eo = reject_epochs(epochs_eo, automatic_epoch_rejection)
            epochs_eo.save(fname=epochs_dir / f"epochs-{paradigm}-eo.fif", overwrite=True)
            logging.info(f"epochs are saved.")

    else:
        epochs = run_erp_processing(raw)
        if manual_data_scroll:
            epochs.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
        epochs = reject_epochs(epochs, automatic_epoch_rejection)
        epochs.save(fname=epochs_dir / f"epochs-{paradigm}.fif", overwrite=True)
        logging.info(f"epochs are saved.")

    ## source analysis 
    if len(raw.info["projs"]) == 0:
        raw.set_eeg_reference("average", projection=True)
    if source_analysis:
        inv = run_source_analysis(epochs)
        write_inverse_operator(
                                fname=subject_dir / "inv" / "{paradigm}-inv.fif",
                                inv=inv,
                                overwrite=True
                                )
            
    ## create a report
    if create_report:
        tqdm.write("Creating report...\n")
        logging.info(f"creating report...")
        progress.update(1)
        fname_report = subject_dir / "reports" / f"{paradigm}.h5"
        report = open_report(fname_report)

        if paradigm.startswith("rest") and len(epochs_list) == 2:
            epochs_concat = concatenate_epochs(epochs_list)
        elif paradigm.startswith("rest") and len(epochs_list) == 1:
            epochs_concat = epochs_eo
        else:
            epochs_concat = epochs
        
        fig_drop = epochs_concat.plot_drop_log()
        report.add_figure(fig=fig_drop, title="Epochs drop log", image_format="PNG")

        ## saving
        report.save(fname=f"{fname_report.stem}.html", open_browser=False, overwrite=True)
    
    tqdm.write("\033[32mAnalysis finished successfully!\n")
    progress.update(1)
    progress.close()


def run_rs_processing(raw, event_ids, progress, logging):

    tqdm.write("Creating epochs...\n")
    progress.update(1)

    events, events_dict = events_from_annotations(raw)

    if len(events) == 0:
        logging.info("This recording is only eyes open or eyes closed.")
        logging.info(f"events_dict : {events_dict}")
        both_conditions = False
        tmin = 5
        raw.crop(tmin=tmin)
        logging.info(f"{tmin} seconds are cropped from beginning of the data.")
        epochs_eo = make_fixed_length_epochs(raw, duration=2) 
        logging.info(f"{len(epochs_eo)} fixed length (2s) eyes-open epochs are created.")

    elif len(events) < 4:
        logging.info("This recording is only eyes open or eyes closed.")
        logging.info(f"events_dict : {events_dict}")
        both_conditions = False
        tmin = max(np.squeeze(events)[-1] / 250 + 3, 5) # 3 seconds skip
        raw.crop(tmin=tmin)
        logging.info(f"{tmin} seconds are cropped from beginning of the data.")
        epochs_eo = make_fixed_length_epochs(raw, duration=2)
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
            mean_dist = np.mean(events_eo - events_ec)
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
            mean_dist = np.mean(events_ec - events_eo)
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


def run_erp_processing(raw, events, event_ids, progress, logging):
    
    match raw.info["description"]:
        case "gpias":
            baseline = None
            raise NotImplementedError
        case "omi" | "xxxxx" | "xxxxy":
            baseline = (None, 0)

    tqdm.write("Creating epochs...\n")
    progress.update(1)
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
        ar.fit(epochs)
        epochs = ar.transform(epochs)
    if automatic_epoch_rejection == "pyriemann":
        train_covs = int(0.7 * len(epochs))  # nb of matrices to train the potato (70%)
        train_set = [random.randint(0, len(epochs)) for _ in range(train_covs)]
        covs = Covariances(estimator="lwf").transform(epochs.get_data())
        potato = Potato(metric="riemann", threshold=3, n_iter_max=100).fit(covs[train_set])
        p_labels = potato.predict(covs)
        bad_idxs = np.where(p_labels == 0)[0]
        epochs.drop(bad_idxs)

    return epochs
    
            
def run_source_analysis(epochs, subjects_fs_dir, progress, logging):

    if epochs.info["description"].startswith("rest"):
        epochs = epochs[0]
        tqdm.write("Using ad hoc noise covariance for the recording ...\n")
        progress.update(1)
        noise_cov = make_ad_hoc_cov(epochs.info)
    else:
        noise_cov = compute_covariance(epochs)

    if subjects_fs_dir is None:
        kwargs = {
                    "subject": "fsaverage",
                    "subjects_dir": subjects_fs_dir
                }
        
        tqdm.write("Loading MRI information of Freesurfer template subject ...\n")
        progress.update(1)
        fs_dir = fetch_fsaverage()
        trans = fs_dir / "bem" / "fsaverage-trans.fif"
        src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
        bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

    else:
        kwargs = {
                    "subject": epochs.info["subject_info"]["first_name"],
                    "subjects_dir": subjects_fs_dir
                }

        tqdm.write("Setting up bilateral hemisphere surface-based source space with subsampling ...\n")
        progress.update(1)
        src = setup_source_space(**kwargs)
        tqdm.write("Creating a BEM model for subject ...\n")
        progress.update(1)
        bem_model = make_bem_model(**kwargs)  
        bem = make_bem_solution(bem_model)
        tqdm.write("Coregistering MRI with a subjects head shape ...\n")
        progress.update(1)
        coreg = Coregistration(epochs.info, fiducials='auto', **kwargs)
        coreg.fit_fiducials()
        coreg.fit_icp(n_iterations=40, nasion_weight=2.0) 
        coreg.omit_head_shape_points(distance=5.0 / 1000)
        coreg.fit_icp(n_iterations=40, nasion_weight=10)
        trans = coreg.trans

    tqdm.write("Computing forward solution ...\n")
    progress.update(1)
    fwd = make_forward_solution(epochs.info,
                                trans=trans,
                                src=src,
                                bem=bem,
                                meg=False,
                                eeg=True
                                )
    tqdm.write("Computing the minimum-norm inverse solution ...\n")
    progress.update(1)
    inverse_operator = make_inverse_operator(epochs.info,
                                            fwd,
                                            noise_cov
                                            )

    return inverse_operator



def run_erp_analysis(
        subject_id,
        subjects_dir=None,
        paradigm="gpias",
        source_analysis=True,
        events=None,
        mri=False,
        subjects_fs_dir=None,
        manual_data_scroll=False,
        automatic_epoch_rejection=False,
        create_report=True,
        saving_dir=None,
        verbose="ERROR"
        ):
    
    """ Preprocessing of the raw eeg recordings from BrainVision device.
        The process could be fully or semi automatic based on user choice.

        Parameters
        ----------
        subject_id : str
            The subject name, if subject has MRI data as well, should be FreeSurfer subject name, 
            then data from both modality can be analyzed at once.
        subjects_dir : path-like | None
            The path to the directory containing the EEG subjects. The folder structure should be
            as following which can be created by running "file_preparation" function:
            subjects_dir/
            ├── sMRI/
            ├── fMRI/
            │   ├── session_1/
            │   ├── session_2/
            ├── dMRI/
            ├── EEG/
                ├── paradigm_1/
                ├── paradigm_2/
                ├── ...
        paradigm : str
            Name of the EEG paradigm. should be a subfolder in the subjects_dir / subject_id containing
            raw EEG data.
        analysis_type: str
            Should be either "sensor", "source" which will perform sensor-level and source-level analysis, respecively.
        events: ndarray of int, shape (n_events, 3) | None
            If None, the events will be extracted from trigger channel.
        mri: bool
            If True, subject has MRI data which is surface reconstructed via Freesurfer.
        subjects_fs_dir: str │ path-like │ None
            The path for the subects_dir of the FS application. If None, the path will be:
            "/Applications/freesurfer/7.4.1/subjects"
        manual_data_scroll : bool
            If True, user can interactively select epochs of the recording to be removed.
            If not, this step will be skipped.
        create_report : bool
            If True, a report will be created per recordinng.
        saving_dir : path-like | None | bool
            The path to the directory where the preprocessed EEG will be saved, If None, it will be saved 
            in the same path as the raw files. If False, preprocessed data will not be saved.
        verbose : bool | str | int | None
            Control verbosity of the logging output. If None, use the default verbosity level.

        Notes
        -----
        .. This script is mainly designed for Antinomics / TIDE projects, however could be 
            used for other purposes.
        """
    
    set_log_level(verbose=verbose)
    progress = tqdm(total=9,
                    desc="",
                    ncols=50,
                    colour="cyan",
                    bar_format='{l_bar}{bar}'
                    )
    
    ## reading files and montaging 
    time.sleep(1)
    tqdm.write("Loading preprocessed EEG data ...\n")
    progress.update(1)

    if subjects_dir == None:
        subjects_dir = Path.cwd().parent / "subjects"
    else:
        subjects_dir = Path(subjects_dir)

    if subjects_fs_dir == None:
        subjects_fs_dir = "/Applications/freesurfer/7.4.1/subjects"

    fname = subjects_dir / subject_id / "EEG" / paradigm / "raw_prep.fif"
    raw = read_raw_fif(fname, preload=True)
    info = raw.info
    
    ## check paradigms
    if events is None:
        events, event_ids = events_from_annotations(raw)
        if "New Segment/" in event_ids: 
            events = events[events[:, 2] != event_ids["New Segment/"]]
            event_ids.pop("New Segment/")

        match paradigm:
            case "gpias":
                shift = 0.012 # 0.016
                stims, stim_ids = events, event_ids
                stim_ids = {
                            'gappre': 1,
                            'gapbbn': 2,
                            'gap3': 3,
                            'gap8': 4,
                            'gappost': 5,
                            }
                events = []
                for stim_idx, stim in enumerate(stims):
                    ## split recording into blocks
                    start = stim[0]
                    if stim_idx == len(stims) - 1:
                        stop = None
                    else: 
                        stop = stims[stim_idx + 1][0]
                    stim_ch, times = raw.get_data(picks="Audio",
                                                    start=start,
                                                    stop=stop,
                                                    return_times=True)
                    stim_ch = np.squeeze(stim_ch)
                    stim_id = stim[2]
                    stim_key = [key for key, value in stim_ids.items() if value == stim_id - 1][0] # maybe later better write it
                    evs, event_ids = _detect_gpias_events(stim_ch, times, stim_key)
                    events.append(evs)
                
                events = np.concatenate(events, axis=0)
                events[:, 0] = events[:, 0] - shift * info["sfreq"]
                baseline = None
            
            case "omi" | "xxxxx" | "xxxxy":
                baseline = (None, 0)

            case "regularity" | "teas":
                raise NotImplementedError
    
    tqdm.write("Creating epochs...\n")
    progress.update(1)
    epochs = Epochs(raw=raw,
                    events=events,
                    event_id=event_ids,
                    tmin=-0.2,
                    tmax=0.5,
                    reject_by_annotation=True,
                    baseline=baseline
                    )
    del raw

    ## check manual_data_scroll
    if manual_data_scroll:
        epochs.plot(n_channels=80, picks="eeg", events=events, scalings=dict(eeg=50e-6), block=True)

    ## save epochs
    tqdm.write("Computing Evoked objects and saving it...\n")
    progress.update(1)

    if saving_dir == None:
        saving_dir = subjects_dir / subject_id / "EEG" / f"{paradigm}"

    ## drop and save epochs
    if not automatic_epoch_rejection == False:
        if automatic_epoch_rejection == "ptp":
            reject = dict(eeg=40e-6)
            flat = dict(eeg=1e-7)
            epochs.drop_bad(reject=reject, flat=flat)

        if automatic_epoch_rejection == "autoreject":
            ar = AutoReject()
            ar.fit(epochs)
            epochs = ar.transform(epochs)

        if automatic_epoch_rejection == "pyriemann":
            raise NotImplementedError

    epochs.save(fname=saving_dir / "epochs-epo.fif", overwrite=True)

    ## compute evokeds
    evs = epochs.average(by_event_type=True)
    [ev.save(saving_dir / f"{ev.comment}-evo.fif", overwrite=True) for ev in evs]

    ## source analysis
    if source_analysis:
        if len(info["projs"]) == 0:
            raw.set_eeg_reference("average", projection=True)

        if mri:
            kwargs = {"subject": subject_id,
                    "subjects_dir": subjects_fs_dir
                    }
            tqdm.write("Setting up bilateral hemisphere surface-based source space with subsampling ...\n")
            progress.update(1)
            src = setup_source_space(**kwargs)

            tqdm.write("Creating a BEM model for subject ...\n")
            progress.update(1)
            bem_model = make_bem_model(**kwargs)  
            bem = make_bem_solution(bem_model)

            tqdm.write("Coregistering MRI with a subjects head shape ...\n")
            progress.update(1)
            coreg = Coregistration(info, subject_id, subjects_fs_dir, fiducials='auto')
            coreg.fit_fiducials()
            coreg.fit_icp(n_iterations=40, nasion_weight=2.0) 
            coreg.omit_head_shape_points(distance=5.0 / 1000)
            coreg.fit_icp(n_iterations=40, nasion_weight=10)
            trans = coreg.trans

        else:    
            tqdm.write("Loading MRI information of Freesurfer template subject ...\n")
            progress.update(1)
            kwargs = {"subject": "fsaverage",
                    "subjects_dir": None,
                    }
            fs_dir = fetch_fsaverage()
            trans = fs_dir / "bem" / "fsaverage-trans.fif"
            src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
            bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

        tqdm.write("Computing forward solution ...\n")
        progress.update(1)
        fwd = make_forward_solution(info,
                                    trans=trans,
                                    src=src,
                                    bem=bem,
                                    meg=False,
                                    eeg=True
                                    )

        tqdm.write("Estimating the noise covariance of the recording ...\n")
        progress.update(1)
        if paradigm == "gpias":
            po_stims = [key for key in event_ids.keys() if key.startswith("PO")]
            noise_cov = compute_covariance(epochs[po_stims])
        else:
            noise_cov = compute_covariance(epochs)

        tqdm.write("Computing the minimum-norm inverse solution ...\n")
        progress.update(1)
        inverse_operator = make_inverse_operator(info,
                                                fwd,
                                                noise_cov
                                                )
        write_inverse_operator(fname=saving_dir / "operator-inv.fif",
                                inv=inverse_operator)
        ## create a report
        if create_report:
            tqdm.write("Creating report...\n")
            progress.update(1)
            fname_report = subjects_dir / subject_id / "EEG" / "reports" / f"{paradigm}.h5"
            report = open_report(fname_report)
            # for ev in evs:
            #     fig_ev, ax = plt.subplots(1, 1, figsize=(7.5, 3))
            #     ev.plot(time_unit="ms", titles="", axes=ax)
            #     ax.set_title(ev.comment)
            #     ax.spines[["right", "top"]].set_visible(False)
            #     ax.axvspan(xmin=-200, xmax=0, ymin=-20, ymax=20, color="grey", alpha=0.2)
            #     report.add_figure(fig=fig_ev, title="Evoked Response", image_format="PNG")

            fig_drop = epochs.plot_drop_log()
            report.add_figure(fig=fig_drop, title="Epochs drop log", image_format="PNG")

            ## source space
            if source_analysis:
                report.add_bem(title="MRI & BEM",
                                decim=10,
                                width=512,
                                **kwargs
                                )
                report.add_trans(trans=trans,
                                info=info,
                                title="Co-registration",
                                **kwargs
                                )
            ## saving
            report.save(fname=f"{fname_report.as_posix()[:-3]}.html", open_browser=False, overwrite=True)

    tqdm.write("\033[32mAnalysis finished successfully!\n")
    progress.update(1)
    progress.close()


def _detect_gpias_events(
                        stim_ch,
                        times,
                        stim_key,
                        plot_peaks=True
                        ):
    
    
    peak_idxs, peaks_dict = find_peaks(stim_ch, height=[300, 3000], distance=100)
    pk_heights = peaks_dict["peak_heights"]
    events = []
    events_dict = { 
                    "PO70_pre": 11,
                    "PO75_pre": 12,
                    "PO80_pre": 13,
                    "PO85_pre": 14,
                    "PO90_pre": 15,               
                    "PO_bbn": 21,
                    "GO_bbn": 22,
                    "GP_bbn": 23,
                    "PO_3": 31,
                    "GO_3": 32,
                    "GP_3": 33,
                    "PO_8": 41,
                    "GO_8": 42,
                    "GP_8": 43,
                    "PO70_post": 51,
                    "PO75_post": 52,
                    "PO80_post": 53,
                    "PO85_post": 54,
                    "PO90_post": 55,
                    }
    match stim_key:
        case "gappre" | "gappost":
            height_limits = {
                            "PO70": [300, 650],
                            "PO75": [650, 1100],
                            "PO80": [1100, 1500],
                            "PO85": [1500, 1900],
                            "PO90": [1900, 3000],
                            }
        case "gapbbn" | "gap3" | "gap8":
            height_limits = {
                            "PO": [1700, 3000],
                            "GO": [300, 900],
                            "GP": [1200, 1700],
                            }   
    
    events = _detect_peaks(pk_heights, peak_idxs, stim_ch, times, height_limits, events_dict, stim_key, plot_peaks=plot_peaks)
    return events, events_dict


