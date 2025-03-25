# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import ipdb
import pandas as pd
import warnings
from pathlib import Path
from tqdm import tqdm
import time

import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

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
                open_report
                )

def run_rs_analysis(
        subject_id,
        subjects_dir=None,
        paradigm='rest',
        event_ids=None,
        source_analysis=True,
        mri=False,
        subjects_fs_dir=None,
        manual_data_scroll=False,
        create_report=True,
        saving_dir=None,
        verbose="ERROR",
        overwrite=False,
        write_feather = False
        ):
    
    """ Sensor and source space analysis of the preprocessed resting-state eeg recordings from BrainVision device.
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
        visit : int
            The visit number of the resting state paradigm. 
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
    
    set_log_level(verbose=verbose)
    total = 10 if mri else 8
    progress = tqdm(total=total,
                    desc="",
                    ncols=50,
                    colour="cyan",
                    bar_format='{l_bar}{bar}'
                    )
    
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
    
    tqdm.write("Creating epochs...\n")
    progress.update(1)

    ## be cautious
    events, _ = events_from_annotations(raw)
    skip=False
    if events.shape[0]==2 or events.shape[0]==4:
        skip=True
        both_conditions = False
        events = events[0,:].reshape(1,-1)
        print('Warning- please check this')
        #ipdb.set_trace()
    if event_ids is None : # zurich device
        if len(events) == 0:
            print("This recording is only eyes open or eyes closed.")
            both_conditions = False
            tmin = 5
            raw.crop(tmin=tmin)
            epochs_eo = make_fixed_length_epochs(raw, duration=2) 
            skip=True
        elif len(events) == 1:
            print("This recording is only eyes open or eyes closed.")
            both_conditions = False
            tmin = max(events[0,0] / info["sfreq"] + 3, 5) # 3 seconds skip
            raw.crop(tmin=tmin)
            epochs_eo = make_fixed_length_epochs(raw, duration=2)
            skip=True
        elif len(events) > 2:
            both_conditions = True
            print('Warning- please check eyes open and eyes closed events are assumptions!')
            #ipdb.set_trace() # At this point, the code stops because event numbers 4 and 6 might not be eyes closed and eyes open, respectively. If they are, just press 'c<ENTER>' to continue.
            events_ec = events[:, 0][events[:, 2] == 6]  ## assume 6 is eyes closed
            events_eo = events[:, 0][events[:, 2] == 4]  ## assume 4 is eyes open
            skip=False # use this if the above events are properly assigned
            skip=True #for now
    
    else: # add other sites here
        both_conditions = True
        events_ec = events[:, 0][events[:, 2] == list(event_ids.keys())[0]]
        events_eo = events[:, 0][events[:, 2] == list(event_ids.keys())[1]]

    # add skip couple of seconds
    if not skip:
        if both_conditions and events_ec.shape[0] == events_eo.shape[0]:
            #if len(events_ec) != len(events_eo):
            #    raise ValueError("Number of eyes open and eyes close events don't not match.")

            raws_ec, raws_eo = [], []
            for ec_s, eo_s in zip(events_ec, events_eo):
                tmin = ec_s / info["sfreq"] + 3 # skip few seconds
                tmax = eo_s / info["sfreq"] 
                raws_ec.append(raw.copy().crop(tmin=tmin, tmax=tmax))

            for ec_o, ec_s in zip(events_eo[:-1], events_ec[1:]):
                tmin = ec_o / 250 + 3 # skip few seconds
                tmax = ec_s / 250
                raws_eo.append(raw.copy().crop(tmin=tmin, tmax=tmax))

            epochs_ec, epochs_eo = [make_fixed_length_epochs(
                                                            concatenate_raws(raw_e),
                                                            duration=2
                                                            ) for raw_e in [raws_ec, raws_eo]]
    del raw
    
    ## check manual_data_scroll
    if manual_data_scroll and not skip:
        epochs_eo.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
        if both_conditions:
            epochs_ec.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
    
    if saving_dir is None:
        saving_dir = subjects_dir / subject_id / "EEG" / f"{paradigm}"

    ## save epochs
    if not skip:
        epochs_eo.save(fname=saving_dir / f"epochs-eo-epo.fif", overwrite=True)
        if both_conditions:
            epochs_ec.save(fname=saving_dir / f"epochs-ec-epo.fif", overwrite=True)
                
    if source_analysis:
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

        else:   # mri==False 
            kwargs = {"subject": "fsaverage",
                    "subjects_dir": None
                    }
#            tqdm.write("Loading MRI information of Freesurfer template subject ...\n")
            progress.update(1)
            fs_dir = fetch_fsaverage()
            trans = fs_dir / "bem" / "fsaverage-trans.fif"
            src = fs_dir / "bem" / "fsaverage-ico-5-src.fif"
            bem = fs_dir / "bem" / "fsaverage-5120-5120-5120-bem-sol.fif"

#        tqdm.write("Computing forward solution ...\n")
        progress.update(1)
        fwd = make_forward_solution(info,
                                    trans=trans,
                                    src=src,
                                    bem=bem,
                                    meg=False,
                                    eeg=True
                                    )
#        tqdm.write("Using ad hoc noise covariance for the recording ...\n")
        progress.update(1)
        noise_cov = make_ad_hoc_cov(info)

        ## add a condition for GPIAS to use ad hoc noise covariance
#        tqdm.write("Computing the minimum-norm inverse solution ...\n")
        progress.update(1)
        inverse_operator = make_inverse_operator(info,
                                                fwd,
                                                noise_cov
                                                )
        
        write_inverse_operator(fname=saving_dir / "operator-inv.fif",
                                inv=inverse_operator,overwrite=overwrite)
        

    ## create a report
    if create_report:
        tqdm.write("Creating report...\n")
        progress.update(1)
        fname_report = subjects_dir / subject_id / "EEG" / "reports" / f"{paradigm}.h5"
        report = open_report(fname_report)

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

    if write_feather and not skip:
        import mne
        evoked = epochs_eo.average()
    #    info2 = evoked.info  # Now, info contains averaging details
        snr = 3.0  # Signal-to-noise ratio
        lambda2 = 1.0 / snr**2  # Regularization parameter
        stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, method='dSPM')
        data = stc.data  # shape (n_sources, n_times)
        times = stc.times  # Time points corresponding to the data

        stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, method='dSPM')
        # Assuming you have already computed `stc`
        df = pd.DataFrame(data=stc.data.T, index=stc.times, columns=[f"Source_{i}" for i in range(stc.data.shape[0])])
        # Rename index for clarity
        df.index.name = "Time (s)"

        print(df.head())  # Display the first few rows
        feather_file = subjects_dir / subject_id / "EEG" / "reports"/f"{paradigm}_source.feather"
        df.to_feather(feather_file)
        print(f"wrote to {feather_file}")


    progress.update(1)
    progress.close()



def run_erp_analysis(
        subject_id,
        subjects_dir=None,
        paradigm="gpias",
        source_analysis=True,
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
            case _:
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

    epochs.save(fname=saving_dir / "epochs-epo.fiff", overwrite=True)

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
            for ev in evs:
                fig_ev, ax = plt.subplots(1, 1, figsize=(7.5, 3))
                ev.plot(time_unit="ms", titles="", axes=ax)
                ax.set_title(ev.comment)
                ax.spines[["right", "top"]].set_visible(False)
                ax.axvspan(xmin=-200, xmax=0, ymin=-20, ymax=20, color="grey", alpha=0.2)
                report.add_figure(fig=fig_ev, title="Evoked Response", image_format="PNG")

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
#    ipdb.set_trace()
#    #a try from Beat:
#    snr = 3.0  # Signal-to-noise ratio
#    lambda2 = 1.0 / snr**2  # Regularization parameter
#    evoked = epochs_eo.average()
#    info = evoked.info  # Now, info contains averaging details
#    stc = mne.minimum_norm.apply_inverse(evoked, inverse_operator, lambda2, method='dSPM')
#    data = stc.data  # shape (n_sources, n_times)
#    times = stc.times
#    # Assuming you have already computed `stc`
#    df = pd.DataFrame(data=stc.data.T, index=stc.times, columns=[f"Source_{i}" for i in range(stc.data.shape[0])])
#    # Rename index for clarity
#    df.index.name = "Time (s)"
#    pd_data_file = subjects_dir / subject_id / "EEG" / paradigm / f"{paradigm}_sources.feather"
##    df.to_hdf( pd_data_file )
#    df.to_feather( pd_data_file )
#    print(f"wrote to {pd_data_file}.") 
#    print(df.head())  # Display the first few rows
    #ipdb.set_trace()
    ################################################################
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
