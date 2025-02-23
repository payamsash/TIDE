# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import warnings
import os
from pathlib import Path
from ast import literal_eval
from tqdm import tqdm
import time
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import customtkinter as ctk

from mne_icalabel import label_components
from mne.coreg import Coregistration
from mne.datasets import fetch_fsaverage
from mne.io import read_raw_brainvision, read_raw_fif
from mne.channels import read_dig_captrak, make_standard_montage
from mne.minimum_norm import make_inverse_operator, apply_inverse
from mne.viz import plot_projs_joint
from mne.preprocessing import (annotate_muscle_zscore,
                                ICA,
                                create_eog_epochs,
                                create_ecg_epochs,
                                compute_proj_ecg,
                                compute_proj_eog,
                                find_bad_channels_lof
                                )
from mne import (set_log_level,
                events_from_annotations,
                concatenate_raws,
                make_fixed_length_epochs,
                Epochs,
                setup_source_space,
                make_bem_model,
                make_bem_solution,
                make_forward_solution,
                compute_covariance,
                read_labels_from_annot,
                extract_label_time_course,
                open_report,
                Report
                )


def run_erp_analysis(
        subject_id,
        subjects_dir=None,
        paradigm="rest",
        analysis_type="sensor",
        events=None,
        mri=False,
        subjects_fs_dir=None,
        manual_data_scroll=False,
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
    progress = tqdm(total=5,
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
    

    ## check paradigms
    if events == None:

        events, event_ids = events_from_annotations(raw)
        if "New Segment/" in event_ids: 
            events = events[events[:, 2] != event_ids["New Segment/"]]
            event_ids.pop("New Segment/")

        match paradigm:
            case "rest":
                thr = 30 * 250 # 30 seconds
                events = events[np.where(np.diff(events[:, 0], prepend=-np.inf) < thr)[0]]
                unique_ids, counts = np.unique(events[:, -1], return_counts=True)
                keep_ids = unique_ids[counts > 1]
                events = events[np.isin(events[:, -1], keep_ids)]
                
                raws_eo, raws_ec = [], []
                for event_idx, event in enumerate(events[::2, 0]):
                    if not event == events[-1][0]:
                        tmin, tmax = event / 250, events[event_idx + 1][0] / 250
                        raws_eo.append(raw.copy().crop(tmin=tmin, tmax=tmax))
                for event_idx, event in enumerate(events[1::2, 0]):
                    if not event == events[-1][0]:
                        tmin, tmax = event / 250, events[event_idx + 1][0] / 250
                        raws_ec.append(raw.copy().crop(tmin=tmin, tmax=tmax))
                
                epochs_eo, epochs_ec = [make_fixed_length_epochs(
                                                                concatenate_raws(raw_e),
                                                                duration=2
                                                                ) for raw_e in [raws_eo, raws_ec]]

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
                    stim_key = [key for key, value in stim_ids.items() if value == stim_id][0]
                    evs, event_ids = _detect_gpias_events(stim_ch, times, stim_key)
                    events.append(evs)
                
                events = np.concatenate(events, axis=0)
                events[:, 0] = events[:, 0] - shift * raw.info["sfreq"]
                baseline = None
            
            case "omi" | "xxxxx" | "xxxxy":
                baseline = (None, 0)

            case "regularity" | "teas":
                raise NotImplementedError
    
    tqdm.write("Creating epochs...\n")
    progress.update(1)
    
    if not paradigm == "rest":
        epochs = Epochs(raw=raw,
                            events=events,
                            event_id=event_ids,
                            tmin=-0.2,
                            tmax=0.5,
                            reject_by_annotation=True,
                            baseline=baseline
                            )

    
    ## check manual_data_scroll
    if manual_data_scroll:
        if paradigm == "rest":
            epochs_eo.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
            epochs_ec.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
        else:
            epochs.plot(n_channels=80, picks="eeg", events=events, scalings=dict(eeg=50e-6), block=True)

    ## save epochs
    tqdm.write("Computing Evoked objects and saving it...\n")
    progress.update(1)

    if paradigm == "rest":
        if saving_dir == None:
            [epochs.save(fname=subjects_dir / subject_id / "EEG" / f"{paradigm}" / f"epochs-{title}-epo.fif") \
                                                for epochs, title in zip([epochs_eo, epochs_ec], ["eo", "ec"])] 
        else:
            epochs.save(fname=saving_dir / "epochs-epo.fiff")
            [epochs.save(fname=saving_dir / f"epochs-{title}-epo.fif") \
                                                for epochs, title in zip([epochs_eo, epochs_ec], ["eo", "ec"])] 
            
        ## add source for rest

    else:
        if saving_dir == None:
            epochs.save(fname=subjects_dir / subject_id / "EEG" / f"{paradigm}" / "epochs-epo.fif")
        else:
            epochs.save(fname=saving_dir / "epochs-epo.fiff")

    if not paradigm == "rest":
        ## compute evokeds
        evs = epochs.average(by_event_type=True)

        ## check type
        if analysis_type == "sensor":
            [ev.save(subjects_dir / subject_id / "EEG" / f"{paradigm}" / f"{ev.comment}-evo.fif") for ev in evs]

        if analysis_type == "source":
            if len(raw.info["projs"]) == 0:
                raw.set_eeg_reference("average", projection=True)

            if mri:
                kwargs = {"subject": subject_id,
                        "subjects_dir": subjects_fs_dir
                        }
                
                brain_labels = read_labels_from_annot(parc="aparc", **kwargs)
                tqdm.write("Setting up bilateral hemisphere surface-based source space with subsampling ...\n")
                progress.update(1)
                src = setup_source_space(**kwargs)

                tqdm.write("Creating a BEM model for subject ...\n")
                progress.update(1)
                bem_model = make_bem_model(**kwargs)  
                bem = make_bem_solution(bem_model)

                tqdm.write("Coregistering MRI with a subjects head shape ...\n")
                progress.update(1)
                coreg = Coregistration(evs.info, subject_id, subjects_fs_dir, fiducials='auto')
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
                trans = subject_id
                fname_src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
                bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"
                brain_labels = read_labels_from_annot(parc="aparc", **kwargs)

            tqdm.write("Computing forward solution ...\n")
            progress.update(1)
            fwd = make_forward_solution(evs.info,
                                        trans=trans,
                                        src=fname_src,
                                        bem=bem,
                                        meg=False,
                                        eeg=True
                                        )

            tqdm.write("Estimating the noise covariance of the recording ...\n")
            progress.update(1)
            noise_cov = compute_covariance(epochs)

            ## add a condition for GPIAS to use ad hoc noise covariance
            
            tqdm.write("Computing the minimum-norm inverse solution ...\n")
            progress.update(1)
            inverse_operator = make_inverse_operator(evs.info,
                                                    fwd,
                                                    noise_cov
                                                    )
            
            tqdm.write("Apply inverse solution and extract label time courses...\n")
            progress.update(1)
            stcs = [apply_inverse(ev, inverse_operator) for ev in evs]
            label_ts = extract_label_time_course(stcs,
                                                brain_labels,
                                                inverse_operator["src"],
                                                mode="mean", # be cautious
                                                allow_empty=True
                                                )
            
            fname_label_ts = subjects_dir / subject_id / "EEG" / f"{paradigm}" / "labels_ts.npy"
            np.save(fname_label_ts, np.array(label_ts)) # shape: (n_evs, n_labels, n_times)

        ## create a report
        if create_report:
            tqdm.write("Creating report...\n")
            progress.update(1)
            fname_report = subjects_dir / subject_id / "EEG" / "reports" / f"{paradigm}.h5"
            report = open_report(fname_report)
            for ev in evs:
                fig_ev, ax = plt.subplots(1, 1, figsize=(7.5, 3))
                ev.plot(time_unit="ms", titles="", axes=ax)
                ax.set_title(ev.comment)
                ax.spines[["right", "top"]].set_visible(False)
                ax.axvspan(xmin=-200, xmax=0, ymin=-20, ymax=20, color="grey", alpha=0.2)
                report.add_figure(fig=fig_ev, title="Evoked Response", image_format="PNG")

            ## source space
            if analysis_type == "source":
                report.add_bem(subject=subject_id,
                            subjects_dir=subjects_dir,
                            title="MRI & BEM",
                            decim=10,
                            width=512)
                report.add_trans(trans=coreg.trans,
                                info=raw.info,
                                subject=subject_id,
                                subjects_dir=subjects_dir,
                                title="Co-registration")
        
        ## saving
        if saving_dir == None:
            report.save(fname=f"{fname_report[:-3]}.html", open_browser=False, overwrite=True)
        else:
            report.save(fname=saving_dir / f"{paradigm}.html", open_browser=False, overwrite=True)

    tqdm.write("Analysis finished successfully!\n")
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
    
    return events


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





