# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

from pathlib import Path
from tqdm import tqdm
import time

from mne.io import read_raw_fif
from mne.coreg import Coregistration
from mne.datasets import fetch_fsaverage
from mne.minimum_norm import make_inverse_operator, write_inverse_operator
from mne import (set_log_level,
                events_from_annotations,
                concatenate_raws,
                make_fixed_length_epochs,
                setup_source_space,
                make_bem_model,
                make_bem_solution,
                make_forward_solution,
                make_ad_hoc_cov,
                open_report
                )

def run_rs_analysis(
        subject_id,
        subjects_dir=None,
        visit=1,
        event_ids=None,
        source_analysis=True,
        mri=False,
        subjects_fs_dir=None,
        manual_data_scroll=False,
        create_report=True,
        saving_dir=None,
        verbose="ERROR"
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
            If dict, the keys should be the eyes_close and eyes_open and the values should be integar.
            If None, the first event will be assumed to be eyes closed trigger.
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

    paradigm = f"rest_v{visit}"
    fname = subjects_dir / subject_id / "EEG" / paradigm / "raw_prep.fif"
    raw = read_raw_fif(fname, preload=True)
    info = raw.info
    
    tqdm.write("Creating epochs...\n")
    progress.update(1)

    ## be cautious
    events, _ = events_from_annotations(raw)
    if event_ids == None:
        events_ec = events[:, 0][events[:, 2] == 6]  ## assume 6 is eyes closed
        events_eo = events[:, 0][events[:, 2] == 4]  ## assume 4 is eyes open
    else:
        events_ec = events[:, 0][events[:, 2] == event_ids["eyes_close"]]  
        events_eo = events[:, 0][events[:, 2] == event_ids["eyes_open"]]  

    if len(events_ec) != len(events_eo):
        raise ValueError("Number of eyes open and eyes close events dons not match.")

    raws_ec, raws_eo = [], []
    for ec_s, eo_s in zip(events_ec, events_eo):
        tmin, tmax = ec_s / info["sfreq"], eo_s / info["sfreq"]
        raws_ec.append(raw.copy().crop(tmin=tmin, tmax=tmax))

    for ec_o, ec_s in zip(events_eo[:-1], events_ec[1:]):
        tmin, tmax = ec_o / 250, ec_s / 250
        raws_eo.append(raw.copy().crop(tmin=tmin, tmax=tmax))

    epochs_ec, epochs_eo = [make_fixed_length_epochs(
                                                    concatenate_raws(raw_e),
                                                    duration=2
                                                    ) for raw_e in [raws_ec, raws_eo]]
    del raw
    
    ## check manual_data_scroll
    if manual_data_scroll:
        epochs_eo.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
        epochs_ec.plot(n_channels=80, picks="eeg", scalings=dict(eeg=50e-6), block=True)
    
    if saving_dir is None:
        saving_dir = subjects_dir / subject_id / "EEG" / f"{paradigm}"

    ## save epochs
    [epochs.save(fname=saving_dir / f"epochs-{title}-epo.fif", overwrite=True) \
                                    for epochs, title in zip([epochs_ec, epochs_eo], ["ec", "eo"])] 
            
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

        else:    
            kwargs = {"subject": "fsaverage",
                    "subjects_dir": None
                    }
            tqdm.write("Loading MRI information of Freesurfer template subject ...\n")
            progress.update(1)
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
        tqdm.write("Using ad hoc noise covariance for the recording ...\n")
        progress.update(1)
        noise_cov = make_ad_hoc_cov(info)

        ## add a condition for GPIAS to use ad hoc noise covariance
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