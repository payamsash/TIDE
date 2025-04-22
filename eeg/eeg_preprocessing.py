# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import os
from pathlib import Path
from tqdm import tqdm
import time
from warnings import warn
import matplotlib.pyplot as plt

from mne import set_log_level, Report, concatenate_raws
from mne_icalabel import label_components
from mne_icalabel.gui import label_ica_components
from mne.io import read_raw
from mne.channels import read_dig_captrak, make_standard_montage
from mne.viz import plot_projs_joint
from .tools import (load_config,
                    initiate_logging,
                    _check_preprocessing_inputs,
                    create_subject_dir,
                    read_vhdr_input_fname
                    )
from mne.preprocessing import (ICA,
                                create_eog_epochs,
                                create_ecg_epochs,
                                compute_proj_ecg,
                                compute_proj_eog
                                )

def preprocess(
        fname,
        subject_id,
        subjects_dir,
        site="Zuerich",
        paradigm="gpias",
        config_file=None,
        overwrite="warn",
        verbose="ERROR",
        **kwargs
        ):
    
    """ Preprocessing of the raw eeg recordings.
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
        site : str
            The recording site; must be one of the following: ["Austin", "Dublin", "Ghent", "Illinois", "Regensburg", "Tuebingen"]
        paradigm : str
            Name of the EEG paradigm. should be a subfolder in the subjects_dir / subject_id containing
            raw EEG data.
        psd_check : bool
            if True, the psd will be shown, by clicking on noisy channels, you can see the bad channel names.
        manual_data_scroll : bool
            If True, user can interactively annotate segments of the recording to be removed.
            If not, this step will be skipped.
        run_ica: bool
            If True, ICA will be perfomed to detect and remove eye movement components from data.
            This option is set for the gpias paradigm.
        manual_ica_removal : bool
            If True, a window will pop up asking ICA components to be removed.
            If not, a machine learning model will be used to remove ICA components related to eye movements.
        ssp_eog : bool
            If True, will use EOG channels to regress out blinking from data.
        ssp_ecg : bool
            If True, will use ECG or Pulse channel to regress out ECG artifact from data.
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
    
    ## get values from config file
    if config_file is None:
        yaml_file = os.path.join(os.path.dirname(__file__), 'preprocessing-config.yaml')
        config = load_config(site, yaml_file)
    else:
        config = load_config(site, config_file)
    
    config.update(kwargs)
    psd_check = config.get("psd_check", True)
    manual_data_scroll = config.get("manual_data_scroll", True)
    run_ica = config.get("run_ica", False)
    manual_ica_removal = config.get("manual_ica_removal", False)
    ssp_eog = config.get("ssp_eog", True)
    ssp_ecg = config.get("ssp_ecg", True)
    create_report = config.get("create_report", True)
    verbose = config.get("verbose", "ERROR")

    ## only check inputs
    _check_preprocessing_inputs(fname,
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
                                )

    ## check subject_dir and create if not there
    subjects_dir = Path(subjects_dir)
    subject_dir = subjects_dir / subject_id

    created = False
    if not Path.is_dir(subjects_dir / subject_id):
        create_subject_dir(subject_id, subjects_dir)
        created = True

    logging = initiate_logging(
                                subject_dir / "logs" / f"{paradigm}_preprocessing.log",
                                config,
                                type="preprocessing"
                                )


    if created:
        logging.info("preprocessing script initiated and subject directory has been created.")
    else:
        logging.info("preprocessing script initiated. Subject directory was already created.")

    set_log_level(verbose=verbose)
    progress = tqdm(total=5,
                    desc="",
                    ncols=50,
                    colour="cyan",
                    bar_format='{l_bar}{bar}'
                    )
    
    ## finding files
    time.sleep(1)
    tqdm.write("Finding and reading raw EEG data ...\n")
    progress.update(1)

    fname = Path(fname)
    suffix = fname.suffix
    fnames = []
    if fname.stem.endswith(("_1", "-1")):
        i = 1
        while True:
            fname_1 = Path(f"{fname.stem[:-2]}_{i}{suffix}")
            fname_2 = Path(f"{fname.stem[:-2]}-{i}{suffix}")
            if not any([fname_1.exists(), fname_2.exists()]):
                break
            if fname_1.exists(): fnames.append(fname_1)
            if fname_2.exists(): fnames.append(fname_2)
            i += 1
    else:
        fnames = [fname]

    logging.info(f"Following EEG files are selected to be read: {[str(p) for p in fnames]}")

    ## reading files
    match suffix:
        case ".mff":
            raw = concatenate_raws([read_raw(fname) for fname in fnames])
            raw.drop_channels(ch_names="VREF")
            montage = make_standard_montage("GSN-HydroCel-64_1.0")

        case ".bdf":
            raw = concatenate_raws([read_raw(fname) for fname in fnames])
            raw.pick(["eeg", "stim"])
            montage = make_standard_montage("easycap-M1")

        case ".cdt":
            fnames = [fname.with_suffix(".dpa") for fname in fnames]
            raw = concatenate_raws([read_raw(fname) for fname in fnames])

        case ".ds":
            raw = concatenate_raws([read_raw(fname) for fname in fnames])
            raw.pick(["eeg", "stim"])

        case ".vhdr":   
            if site == "Regensburg":
                raw = concatenate_raws([read_vhdr_input_fname(fname) for fname in fnames])
                raw.pick(["eeg", "stim"]) 
                montage = make_standard_montage("easycap-M1")

            if site == "Zuerich":
                captrak_dir = Path(fname).parent / "captrack"
                try:
                    for file_ck in os.listdir(captrak_dir):
                        if file_ck.endswith(f"_{subject_id}.bvct"): # assume that its same for both visits
                            montage = read_dig_captrak(file_ck)
                except:
                    montage = make_standard_montage("easycap-M1")

                ch_types = {"O1": "eog",
                            "O2": "eog",
                            "PO7": "eog",
                            "PO8": "eog",
                            "Pulse": "ecg",
                            "Resp": "ecg",
                            "Audio": "stim"
                            }
                raw = read_vhdr_input_fname(Path(fname))
                raw.set_channel_types(ch_types)
                raw.pick(["eeg", "eog", "ecg", "stim"])
    
    ## now the real part
    raw.load_data()
    raw.set_montage(montage=montage, match_case=False, on_missing="warn")
    logging.info(f"EEG file(s) are loaded into memory and montaged to standard frame.")

    if raw.info["sfreq"] > 1000.0:
        raw.resample(1000, stim_picks=None)

    ## add information to raw 
    raw.info["experimenter"] = site
    raw.info["subject_info"] = {"first_name": subject_id}
    raw.info["description"] = paradigm
    
    orig_fname = subject_dir / "orig" / f"raw_{paradigm}.fif" 
    if orig_fname.exists():
        if overwrite == "warn":
            warn(f"The preprocessed raw {orig_fname} already exist.")
        if overwrite == "raise":
            raise FileExistsError(f"The preprocessed raw {orig_fname} already exist.")
        
    raw.save(orig_fname, overwrite=True)
    logging.info(f"Raw EEG recording saved in the {str(subject_id)} directory")

    ## resampling, filtering and re-referencing 
    tqdm.write("Resampling, filtering and re-referencing ...\n")
    progress.update(1)
    raw.resample(sfreq=250, stim_picks=None)
    logging.info(f"Raw EEG resampled to 250 Hz.")

    if paradigm.startswith("rest"):
        l_freq, h_freq = 0.1, 100
        if site in ["Illinois", "Austin"]:
            line_freq = 60
        else:
            line_freq = 50
        raw.notch_filter(freqs=line_freq, picks="eeg", notch_widths=1)
        logging.info(f"Raw EEG notch filtered at {line_freq} Hz, with width of 1 Hz.")
    else:
        l_freq, h_freq = 1, 40

    raw.filter(picks="eeg", l_freq=l_freq, h_freq=h_freq)
    logging.info(f"Raw EEG bandpass filtered between {l_freq} and {h_freq} Hz."
                    "for more information on filter type see:"
                    "https://mne.tools/stable/auto_tutorials/preprocessing/25_background_filtering.html"
                )
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()
    logging.info(f"Average reference applied to raw data.")
    
    ## eeg plotting for annotating
    if psd_check:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        raw.plot_psd(picks="eeg", fmin=0.1, fmax=120, ax=ax)
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

    if manual_data_scroll:
        raw.annotations.append(onset=0, duration=0, description="bad_segment")
        raw.plot(duration=20.0, n_channels=80, picks="eeg", scalings=dict(eeg=40e-6), block=True)
    
    if len(raw.info["bads"]):
        logging.info(f"{raw.info['bads']} are interpolated.")
        raw.interpolate_bads()
    else:
        logging.info(f"No bad channel was selected for interpolation.")
    
    
    ## ICA
    show = False
    if run_ica:
        tqdm.write("Running ICA ...\n")
        progress.update(1)
        ica = ICA(n_components=0.95, max_iter=800, method='infomax', fit_params=dict(extended=True))
        try:
            ica.fit(raw)
        except:
            ica = ICA(n_components=5, max_iter=800, method='infomax', fit_params=dict(extended=True))
            ica.fit(raw)

        if manual_ica_removal:
            gui = label_ica_components(raw, ica, block=True)
            eog_indices = ica.labels_["eog"]

        else:
            ic_dict = label_components(raw, ica, method="iclabel")
            ic_labels = ic_dict["labels"]
            ic_probs = ic_dict["y_pred_proba"]
            eog_indices = [idx for idx, label in enumerate(ic_labels) \
                            if label == "eye blink" and ic_probs[idx] > 0.70]

        if len(eog_indices) > 0:
            eog_components = ica.plot_properties(raw,
                                                picks=eog_indices,
                                                show=show,
                                                )
            eog_indices_fil = [x for x in eog_indices if x <= 10]
        ica.apply(raw, exclude=eog_indices_fil)
        logging.info(f"ICA analysis was performed and {len(eog_indices_fil)} eye related components were dropped.")
    
    if ssp_ecg:
        tqdm.write("Finding and removing ECG related components...\n")
        progress.update(1)
        
        ## find R peaks
        ev_pulse = create_ecg_epochs(raw,
                                    ch_name="Pulse",
                                    tmin=-0.5,
                                    tmax=0.5,
                                    l_freq=1,
                                    h_freq=20,
                                    ).average(picks="all")
        ## compute and apply projection
        ecg_projs, _ = compute_proj_ecg(raw, n_eeg=2, reject=None)
        raw.add_proj(ecg_projs)
        logging.info(f"ECG projection was computed and applied to data.")

    if ssp_eog:
        tqdm.write("Finding and removing vertical and horizontal EOG components...\n")
        progress.update(1)

        ## vertical
        ev_eog = create_eog_epochs(raw, ch_name=["PO7", "PO8"]).average(picks="all")
        ev_eog.apply_baseline((None, None))
        veog_projs, _ = compute_proj_eog(raw, n_eeg=2, reject=None)
        raw.add_proj(veog_projs)
        logging.info(f"Vertical EOG projection was computed and applied to data.")

        ## horizontal
        ica = ICA(n_components=0.97, max_iter=800, method='infomax', fit_params=dict(extended=True))        
        ica.fit(raw)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=["O1", "O2"], threshold=2)
        eog_indices_fil = [x for x in eog_indices if x <= 10]
        heog_idxs = [eog_idx for eog_idx in eog_indices_fil if eog_scores[0][eog_idx] * eog_scores[1][eog_idx] < 0]
        fig_scores = ica.plot_scores(scores=eog_scores, exclude=eog_indices_fil, show=show)

        if len(heog_idxs) > 0:
            eog_sac_components = ica.plot_properties(raw,
                                                    picks=heog_idxs,
                                                    show=show,
                                                    )
        ica.apply(raw, exclude=heog_idxs)
        logging.info(f"Horizontal EOG component ffrom ICA was detected and dropped from data.")

    raw.apply_proj()

    # creating and saving report
    tqdm.write("Creating report and saving...\n")
    progress.update(1)
    if create_report:
        logging.info(f"Report file initiated.")
        report = Report(title=f"report_subject_{subject_id}")
        report.add_raw(raw=raw, title="Recording Info", butterfly=False, psd=True)
        logging.info(f"General information was added to report.")

        if run_ica:
            if len(eog_indices_fil) > 0:
                report.add_figure(fig=eog_components, title="EOG Components", image_format="PNG")
                logging.info(f"EOG components from ICA were added to report.")
        
        if ssp_ecg:
            fig_ev_pulse, ax = plt.subplots(1, 1, figsize=(7.5, 3))
            ev_pulse.plot(picks="Pulse", time_unit="ms", titles="", axes=ax)
            ax.set_title("Pulse oximetry")
            ax.spines[["right", "top"]].set_visible(False)
            ax.lines[0].set_linewidth(2)
            ax.lines[0].set_color("blue")
            ev_pulse.apply_baseline((None, None))

            fig_ecg = ev_pulse.plot_joint(picks="eeg", ts_args={"time_unit": "ms"})
            fig_proj = plot_projs_joint(ecg_projs, ev_pulse, picks_trace="TP9")

            for fig, title in zip([fig_ev_pulse, fig_ecg, fig_proj], ["Pulse Oximetry Response", "ECG", "ECG Projections"]):
                report.add_figure(fig=fig, title=title, image_format="PNG")
                logging.info(f"ECG projection added to report.")

        if ssp_eog:
            fig_ev_eog, ax = plt.subplots(1, 1, figsize=(7.5, 3))
            ev_eog.plot(picks="PO7", time_unit="ms", titles="", axes=ax)
            ax.set_title("Vertical EOG")
            ax.spines[["right", "top"]].set_visible(False)
            ax.lines[0].set_linewidth(2)
            ax.lines[0].set_color("magenta")
            ev_eog.apply_baseline((None, None))

            fig_eog = ev_eog.plot_joint(picks="eeg", ts_args={"time_unit": "ms"})
            fig_proj = plot_projs_joint(veog_projs, ev_eog, picks_trace="Fp1")

            for fig, title in zip([fig_ev_eog, fig_eog, fig_proj, fig_scores], ["Vertical EOG", "EOG", "EOG Projections", "Scores"]):
                report.add_figure(fig=fig, title=title, image_format="PNG")
            if len(heog_idxs) > 0:
                report.add_figure(fig=eog_sac_components, title="EOG Saccade Components", image_format="PNG")
            logging.info(f"Vertical EOG plots added to report.")
            logging.info(f"Horizontal EOG plots added to report.")

        ## saving stuff     
        prep_fname = subject_dir / "preprocessed" / f"raw_{paradigm}.fif"
        if prep_fname.exists():
            if overwrite == "warn":
                warn(f"The preprocessed raw {prep_fname} already exist.")
            if overwrite == "raise":
                raise FileExistsError(f"The preprocessed raw {prep_fname} already exist.")

        raw.save(prep_fname, overwrite=True)   
        logging.info(f"Preprocessed eeg recording was saved in {subject_id} directory.")
        report.save(fname=subject_dir / "reports" / f"{paradigm}.h5", open_browser=False, overwrite=True)
        logging.info(f"Report was saved in {subject_id} directory.")
        
    tqdm.write("\033[32mEEG data were preprocessed sucessfully!\n")
    progress.update(1)
    progress.close()
    logging.info(f"Preprocessing finished without an error.")
