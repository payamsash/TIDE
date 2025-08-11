# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import os
from pathlib import Path
import time
from warnings import warn
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from mne_icalabel import label_components
from mne_icalabel.gui import label_ica_components
from mne.io import read_raw, RawArray, read_raw_ant
from mne.channels import read_dig_captrak, make_standard_montage
from mne.viz import plot_projs_joint
from mne import (set_log_level,
                        Report,
                        concatenate_raws,
                        events_from_annotations,
                        create_info,
                        find_events,
                        annotations_from_events,
                        read_trans
                        )
from mne.preprocessing import (ICA,
                                create_eog_epochs,
                                create_ecg_epochs,
                                compute_proj_ecg,
                                compute_proj_eog,
                                find_bad_channels_lof
                                )
from .tools import (load_config,
                    initiate_logging,
                    _check_preprocessing_inputs,
                    create_subject_dir,
                    read_vhdr_input_fname, 
                    gpias_constants
                    )

def preprocess(
        fname,
        subject_id,
        subjects_dir,
        site,
        paradigm,
        config_file=None,
        overwrite="warn",
        verbose="ERROR",
        **kwargs
        ):
    
    """ Preprocessing of the raw eeg recordings.
        The process could be fully or semi automatic based on user choice.

        Parameters
        ----------
        fname : str | Path
            eeg filename.
        subject_id : str
            The subject name, if subject has MRI data as well, should be FreeSurfer subject name, 
            then data from both modality can be analyzed at once.
        subjects_dir : path-like | None
            The path to the directory containing the EEG subjects. 
        site : str
            The recording site; must be one of the following: ['Austin', 'Dublin', 'Ghent', 'Illinois', 'Regensburg', 'Tuebingen']
        paradigm : str
            Name of the EEG paradigm. Name of the EEG paradigm, must be one of the: ['gpias', 'xxxxx', 'xxxxy', 'omi'] or should start with 'rest'.
        config_file: str | Path
            Path to the .yaml config file. If None, the default 'pre-processing-config.yaml' will be used.
        overwrite :  str
            must be one of the ['ignore', 'warn', 'raise'].
        verbose : bool | str | int | None
            Control verbosity of the logging output. If None, use the default verbosity level.
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
            psd_check: true
        
        Notes
        -----
        .. This script is mainly designed for Antinomics / TIDE projects, however could be 
            used for other purposes.
    """
    
    ## get values from config file
    if config_file is None:
        yaml_file = os.path.join(os.path.dirname(__file__), '..', 'config', 'preprocessing-config.yaml')
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
        create_subject_dir(subject_id, subjects_dir, site)
        created = True

    logging = initiate_logging(
                                subject_dir / "logs" / f"{paradigm}_preprocessing.log",
                                config,
                                analysis_type="preprocessing"
                                )
    if created:
        logging.info("preprocessing script initiated and subject directory has been created.")
    else:
        logging.info("preprocessing script initiated. Subject directory was already created.")
    set_log_level(verbose=verbose)
    
    ## finding files
    time.sleep(1)
    print("Finding and reading raw EEG data ...\n")

    fname = Path(fname)
    suffix = fname.suffix
    fnames = []
    if fname.stem.endswith(("_1", "-1", "1")):
        fname_3 = Path("._.")
        i = 1
        while True:
            fname_1 = Path(fname.parent / f"{fname.stem[:-2]}_{i}{suffix}")
            fname_2 = Path(fname.parent / f"{fname.stem[:-2]}-{i}{suffix}")
            if fname.stem[-2] not in ["-", "_"]:
                fname_3 = Path(fname.parent / f"{fname.stem[:-1]}{i}{suffix}")
            if not any([fname_1.exists(), fname_2.exists(), fname_3.exists()]):
                break
            if fname_1.exists(): fnames.append(fname_1)
            if fname_2.exists(): fnames.append(fname_2)
            if fname_3.exists(): fnames.append(fname_3)
            i += 1
    else:
        fnames = [fname]
    logging.info(f"Following EEG files are selected to be read: {[str(p) for p in fnames]}")
    
    ## reading files
    match suffix:
        case ".cnt":
            raw = concatenate_raws([read_raw_ant(fname) for fname in fnames])
            montage = make_standard_montage("easycap-M1")
            raw.drop_channels(['M1', 'M2', 'PO5', 'PO6'])
            ch_types = {"EOG": "eog"}
            raw.set_channel_types(ch_types)
            raw.annotations.delete(idx=-1) ## lets check gpias if it works

        case ".mff":
            raw = concatenate_raws([read_raw(fname) for fname in fnames])
            raw.drop_channels(ch_names="VREF")
            montage = make_standard_montage("GSN-HydroCel-64_1.0")

        case ".fif":
            if site == "Austin":
                raw = concatenate_raws([read_raw(fname) for fname in fnames])
                raw.drop_channels(ch_names="VREF")
                montage = make_standard_montage("GSN-HydroCel-64_1.0")

            if site == "Zuerich":
                captrak_dir = Path(fname).parent / "captrack"
                try:
                    for file_ck in os.listdir(captrak_dir):
                        if file_ck.endswith(f"_{subject_id}.bvct"): # assume that its same for both visits
                            montage = read_dig_captrak(file_ck)
                except:
                    montage = make_standard_montage("easycap-M1")

                ch_types = {
                            "O1": "eog",
                            "O2": "eog",
                            "PO7": "eog",
                            "PO8": "eog",
                            "Pulse": "ecg",
                            "Resp": "ecg",
                            "Audio": "stim"
                            }
                raw = read_raw(Path(fname))
                raw.set_channel_types(ch_types)
                raw.pick(["eeg", "eog", "ecg", "stim"])
                eog_chs_1 = ["PO7", "PO8"]
                eog_chs_2 = ["O1", "O2"]

        case ".bdf":
            raw = concatenate_raws([read_raw(fname, exclude=("EXG")) for fname in fnames])
            raw.pick(["eeg", "stim"])
            montage = make_standard_montage("easycap-M1")

        case ".cdt":
            for fname in fnames:
                dpo_file = fname.with_suffix('.cdt.dpo')
                dpa_file = fname.with_suffix('.cdt.dpa')
                if dpo_file.exists():
                    dpo_file.rename(dpa_file)

            raw = concatenate_raws([read_raw(fname) for fname in fnames])
            ch_types = {"VEOG": "eog",
                        "HEOG": "eog",
                        "Trigger": "stim"}
            raw.set_channel_types(ch_types)
            montage = make_standard_montage("easycap-M1")
            raw.drop_channels(["F11", "F12", "FT11", "FT12", "M1", "M2", "Cb1", "Cb2"])
            eog_chs_1 = ["VEOG"]
            eog_chs_2 = ["F7", "F8"]

            transform = read_trans("./eeg/Illinois-trans.fif")
            raw.info["dev_head_t"] = transform
            raw.pick(["eeg", "eog", "stim"])

        case ".vhdr":   
            if site in ["Regensburg", "Tuebingen"]:
                if len(fnames) == 1:
                    raw = read_vhdr_input_fname(fnames[0])
                else:
                    raws = []
                    for f_idx, fname in enumerate(fnames):
                        raw = read_vhdr_input_fname(fname)
                        raw.annotations.append(onset=0, duration=0, description=f_idx+1)
                        raws.append(raw)
                
                    raw = concatenate_raws(raws)
                montage = make_standard_montage("easycap-M1")
                raw.drop_channels(["HRli", "HRre"], on_missing="warn")
                ch_types = {"audio": "stim"}
                try:
                    raw.set_channel_types(ch_types)
                except: 
                    print("probably this data is old recording from Regensburg")
                raw.pick(["eeg", "stim"])

    
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
        
    ## if paradigm gpias, we need to extract trig times from audio and save in annotation
    if paradigm == "gpias":
        if site in ["Zuerich", "Regensburg", "Dublin", "Tuebingen"]:
            events_dict, default_thrs, distance = gpias_constants()
            raw = create_stim_channel_from_audio(raw,
                                                subject_dir,
                                                events_dict,
                                                default_thrs,
                                                distance,
                                                logging)

    if paradigm in ["omi", "xxxxx", "xxxxy"]:
        raw = add_dublin_annotation(raw, paradigm, site)

    ## resampling, filtering and re-referencing 
    print("Resampling, filtering and re-referencing ...\n")
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

    ## automatic detection of bad channels
    noisy_chs, lof_scores = find_bad_channels_lof(raw, threshold=3, return_scores=True)
    raw.info["bads"] = noisy_chs
    logging.info(f"{noisy_chs} were selected as bad channels via LOF method for interpolation, with following scores: {lof_scores}")

    if manual_data_scroll:
        raw.annotations.append(onset=0, duration=0, description="bad_segment")
        raw.plot(duration=20.0, n_channels=80, picks="eeg", scalings=dict(eeg=40e-6), block=True)
    
    if len(raw.info["bads"]):
        logging.info(f"{raw.info['bads']} are interpolated.")
        raw.interpolate_bads()
    
    ## ICA
    show = False
    if run_ica:
        print("Running ICA ...\n")
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
        print("Finding and removing ECG related components...\n")
        
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
        print("Finding and removing vertical and horizontal EOG components...\n")
        
        ## vertical
        ev_eog = create_eog_epochs(raw, ch_name=eog_chs_1).average(picks="all")
        ev_eog.apply_baseline((None, None))
        veog_projs, _ = compute_proj_eog(raw, n_eeg=2, reject=None)
        raw.add_proj(veog_projs)
        logging.info(f"Vertical EOG projection was computed and applied to data.")

        
        ## horizontal
        ica = ICA(n_components=0.97, max_iter=800, method='infomax', fit_params=dict(extended=True))        
        ica.fit(raw)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=eog_chs_2, threshold=1.2)
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
    print("Creating report and saving...\n")
    if create_report:
        logging.info(f"Report file initiated.")
        report = Report(title=f"report_subject_{subject_id}")
        report.add_raw(raw=raw, title="Recording Info", butterfly=False, psd=True)
        logging.info(f"General information was added to report.")

        if run_ica:
            if len(eog_indices) > 0:
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
        
    print("\033[32mEEG data were preprocessed sucessfully!\n")
    logging.info(f"Preprocessing finished without an error.")



def create_stim_channel_from_audio(raw, subject_dir, events_dict, default_thrs, distance, logging):
    
    site = raw.info["experimenter"]
    order = ["pre", "bbn", "3kHz", "8kHz", "post"]
    
    if site == "Dublin":
        events = find_events(raw, initial_event=True)
        split_indices = np.where(events[:, 2] == 65536)[0]
        if len(split_indices) == 5:
            logging.info(f"Five blocks of {order} are detected.")

        for idx, (i, blck) in enumerate(zip(split_indices, order)):
            if not idx == len(split_indices) - 1:
                sub_evs = events[i:split_indices[idx+1]]
            else:
                sub_evs = events[i:]
            
            if blck in ["pre", "post"]:
                mapping = dict(zip([2, 6, 7, 10, 11], [f'PO70_{blck}', f'PO75_{blck}', f'PO80_{blck}', f'PO85_{blck}', f'PO90_{blck}'])) 
            if blck in ["bbn", "3kHz", "8kHz"]:
                mapping = dict(zip([5, 1, 3], [f'PO_{blck}', f'GO_{blck}', f'GP_{blck}']))
    
            annot_from_events = annotations_from_events(    
                                                        sub_evs[1:],
                                                        sfreq=raw.info["sfreq"],
                                                        event_desc=mapping,
                                                        orig_time=raw.info["meas_date"]
                                                        )
            raw.set_annotations(raw.annotations + annot_from_events)

        return raw

    events_orig = events_from_annotations(raw)[0]
    if site == "Zuerich":
        data = raw.get_data(picks="Audio")[0]
        key_dict = {0: "init", 1: "pre", 2: "bbn", 3: "3kHz", 4: "8kHz", 5: "post"}
        split_indices = events_orig[np.isin(events_orig[:, 2], [1, 2, 3, 4, 5])]
        scale = 1
        x_thr = 100

    if site == "Regensburg":
        data = raw.get_data(picks="audio")[0]
        key_dict = {10000: "init", 10001: "pre", 10002: "bbn", 10003: "3kHz", 10004: "8kHz", 10005: "post"}
        split_indices = events_orig[np.isin(events_orig[:, 2], [10001, 10002, 10003, 10004, 10005])]
        scale = 1e-5
        x_thr = 0.001
        
    if site == "Tuebingen":
        data = raw.get_data(picks="audio")[0]
        key_dict = {10000: "init", 10001: "pre", 10002: "bbn", 10003: "3kHz", 10004: "8kHz", 10005: "post"}
        split_indices = events_orig[np.isin(events_orig[:, 2], [10001, 10002, 10003, 10004, 10005])]
        scale = 1e-5
        x_thr = 0.01

    segments = np.split(data, split_indices[:, 0])
    blocks =[key_dict[i] for i in split_indices[:, 2]]
    blocks_dict = dict(zip(["init"] + blocks, segments))
    logging.info(f"Five blocks of {order} are detected.")

    ## get stim levels
    fig, axs = plt.subplots(1, 5, figsize=(17, 3.5), layout="tight")
    lines = []
    for key in blocks_dict:
        if key == "init": 
            continue
        
        ## get the peaks
        peaks, _ = find_peaks(blocks_dict[key], height=100*scale, distance=100, prominence=100*scale)
        peak_count, bin_edges = np.histogram(blocks_dict[key][peaks], bins=100)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        idx = order.index(key)
        axs[idx].bar(bin_centers, peak_count, width=np.diff(bin_edges), color="lightsteelblue", align='center', edgecolor='black')
        axs[idx].set_ylim(0, max(peak_count) + 5)
        axs[idx].set_title(f"{key}", fontstyle="italic")
        axs[idx].spines[["right", "top"]].set_visible(False)

        ## thrsholds
        init_thrs = list(default_thrs[site].values())
        thrs = init_thrs[idx]
        colors = ['red', 'green', 'blue', 'magenta']
        for thresh, col in zip(thrs, colors):
            line = axs[idx].axvline(thresh, color=col, linestyle='--', linewidth=2)
            lines.append(line)
        
        draggers = [DraggableVLine(line, x_thr) for line in lines]
        def on_close(event):
            final_positions = [line.get_xdata()[0] for line in lines]
            np.save(subject_dir / "thrs.npy", np.array(final_positions))
        fig.canvas.mpl_connect('close_event', on_close)
    plt.show(block=True)

    thrs = np.load(subject_dir / "thrs.npy", allow_pickle=True)
    height_limits = {
                    "PO70": [thrs[0] * 0.5, thrs[0]],
                    "PO75": [thrs[0], thrs[1]],
                    "PO80": [thrs[1], thrs[2]],
                    "PO85": [thrs[2], thrs[3]],
                    "PO90": [thrs[3], thrs[3] * 2],
                    "GO": [thrs[4] * 0.5, thrs[4]],
                    "GP": [thrs[4], thrs[6]],
                    "PO": [thrs[6], thrs[6] * 2],
                    }
    logging.info(f"Threshold values are selected to distinguish events as following: {height_limits}")
    
    ## get triggers from blocks
    for main_key, signal in blocks_dict.items():
        categorized = np.zeros_like(signal, dtype=int)
        if  main_key in ["pre", "post"]:
            for sub_key in list(height_limits.keys())[:5]:
                height = height_limits[sub_key]
                peaks, _ = find_peaks(blocks_dict[main_key], height=height, distance=distance, prominence=100*scale)
                categorized[peaks] = events_dict[main_key][sub_key]
                if not len(peaks) == 25:
                    raise ValueError(f"number of events for event id {sub_key} in {main_key} must be 25, got {len(peaks)} instead.")
                logging.info(f"{len(peaks)} {sub_key} events found in {main_key} part.")

        if main_key in ["bbn", "3kHz", "8kHz"]:
            for sub_key in list(height_limits.keys())[5:]:
                height = height_limits[sub_key]
                peaks, _ = find_peaks(blocks_dict[main_key], height=height, distance=distance, prominence=100*scale)
                categorized[peaks] = events_dict[main_key][sub_key]
                if not (len(peaks) == 100 or len(peaks) == 50): # must fix with correct vals
                    raise ValueError(f"number of events for event id {sub_key} in {main_key} part must be 100, got {len(peaks)} instead.")
                logging.info(f"{len(peaks)} {sub_key} events found in {main_key} part.")
        
        if main_key == "init":
            pass

        blocks_dict[main_key] = categorized
    
    ## make them stim channel
    stim_data = np.concatenate(list(blocks_dict.values()))
    info = create_info(["STI1"], raw.info['sfreq'], ['stim'])
    stim_raw = RawArray(stim_data[np.newaxis,], info)
    raw.add_channels([stim_raw], force_update_info=True)
    events = find_events(raw, stim_channel="STI1", output="onset", min_duration=0, shortest_event=1)
    mapping = {value: f"{sub_key}_{key}"
                for key, subdict in events_dict.items()
                for sub_key, value in subdict.items()
                } 
    annot_from_events = annotations_from_events(events,
                                                sfreq=raw.info["sfreq"],
                                                event_desc=mapping,
                                                orig_time=raw.info["meas_date"]
                                                )
    raw.set_annotations(annot_from_events)
    raw.drop_channels(ch_names="STI1")
    os.remove(subject_dir / "thrs.npy")
    logging.info(f"Events are created and saved as annotation in raw data.")

    return raw

class DraggableVLine:
    def __init__(self, line, x_thr):
        self.line = line
        self.x_thr = x_thr
        self.press = False
        self.cid_press   = line.figure.canvas.mpl_connect('button_press_event',   self.on_press)
        self.cid_release = line.figure.canvas.mpl_connect('button_release_event', self.on_release)
        self.cid_motion  = line.figure.canvas.mpl_connect('motion_notify_event',  self.on_motion)

    def on_press(self, event):
        if event.inaxes != self.line.axes:
            return
        x0 = self.line.get_xdata()[0]
        if abs(event.xdata - x0) < self.x_thr:
            self.press = True

    def on_motion(self, event):
        if not self.press or event.inaxes != self.line.axes:
            return
        self.line.set_xdata([event.xdata, event.xdata])
        self.line.figure.canvas.draw_idle()

    def on_release(self, event):
        self.press = False


def add_dublin_annotation(raw, paradigm, site):

    event_config = {
        "omi": {
            "Austin":    ([2], {2: "Stimulus 4"}, False),
            "Dublin":    ([4], {4: "Stimulus 4"}, False),
            "Ghent":     ([1], {1: "Stimulus 4"}, True),
            "Illinois":  ([1], {1: "Stimulus 4"}, True),
            "Regensburg":([11], {11: "Stimulus 4"}, True),
            "Tuebingen": ([215], {215: "Stimulus 4"}, True),
            "Zuerich":   ([4], {4: "Stimulus 4"}, True),
        },
        "xxxxx": {
            "Austin":    ([1, 2, 3], {1: "Stimulus 1", 2: "Stimulus 2", 3: "Stimulus 3"}, False),
            "Dublin":    ([1, 2, 3], {1: "Stimulus 1", 2: "Stimulus 2", 3: "Stimulus 3"}, False),
            "Ghent":     ([1, 2, 3], {1: "Stimulus 1", 2: "Stimulus 2", 3: "Stimulus 3"}, True),
            "Illinois":  ([1, 2, 3], {1: "Stimulus 1", 2: "Stimulus 2", 3: "Stimulus 3"}, True),
            "Zuerich":   ([1, 2, 3], {1: "Stimulus 1", 2: "Stimulus 2", 3: "Stimulus 3"}, True),
            "Regensburg":([14, 13, 12], {14: "Stimulus 1", 13: "Stimulus 2", 12: "Stimulus 3"}, True),
            "Tuebingen": ([245, 183, 119], {245: "Stimulus 1", 183: "Stimulus 2", 119: "Stimulus 3"}, True),
        },
        "xxxxy": {
            "Austin":    ([1, 2, 3], {1: "Stimulus 11", 2: "Stimulus 12", 3: "Stimulus 13"}, False),
            "Dublin":    ([11, 12, 13], {11: "Stimulus 11", 12: "Stimulus 12", 13: "Stimulus 13"}, False),
            "Ghent":     ([1, 2, 3], {1: "Stimulus 11", 2: "Stimulus 12", 3: "Stimulus 13"}, True),
            "Illinois":  ([11, 12, 13], {11: "Stimulus 11", 12: "Stimulus 12", 13: "Stimulus 13"}, True),
            "Zuerich":   ([11, 12, 13], {11: "Stimulus 11", 12: "Stimulus 12", 13: "Stimulus 13"}, True),
            "Regensburg":([14, 13, 12], {14: "Stimulus 1", 13: "Stimulus 2", 12: "Stimulus 3"}, True),
            "Tuebingen": ([231, 246, 230], {231: "Stimulus 1", 246: "Stimulus 2", 230: "Stimulus 3"}, True),
        }
    }

    
    if site == "Regensburg": 
        
        height = 0.001
        distance = 10
        threshold = 0.15
        threshold1 = 0.15
        threshold2 = 1

        audio_data, times = raw.get_data("audio", return_times=True)
        peak_idxs, _ = find_peaks(audio_data[0], height=height, distance=distance)
        stimuli = times[peak_idxs].astype(float)

        split_indices = np.where(np.diff(stimuli) > threshold1)[0] + 1 
        segments = np.split(stimuli, split_indices)
        first_elements = np.array([seg[0] for seg in segments if len(seg) > 0])
        split_indices = np.where(np.diff(first_elements) > threshold2)[0] + 1 
        segments = np.split(first_elements, split_indices)
        first_elements = np.array([seg[0] * raw.info["sfreq"] for seg in segments if len(seg) > 0]) # change 0 to -1 to get last trigger

        sub_evs = np.zeros(shape=(len(first_elements), 3), dtype=int)
        try:
            sub_evs[:, 0] = first_elements
        except:
            warn("There is small mismatch between trigger and stimulus number, so we proceed with triggers ...")

        event_ids, mapping, _ = event_config[paradigm][site]
        events = events_from_annotations(raw)[0]
        mask = np.isin(events[:, 2], event_ids)
        events_sub = events[mask]

        try:
            sub_evs[:, 2] = events_sub[:, 2]
        except: # sometimes there are extra triggers (if you see problem here we need to change the ID)
            threshold = 10
            time = events_sub[:, 0]
            ID = events_sub[:, 2]
            keep = np.ones(len(events_sub), dtype=bool)

            for i in range(1, len(events_sub)):
                if time[i] - time[i - 1] <= threshold:
                    if ID[i] == 14:
                        keep[i] = False 
                    elif ID[i - 1] == 14:
                        keep[i - 1] = False
            events_sub = events_sub[keep]
            sub_evs[:, 2] = events_sub[:, 2]
        
    else:
        try:
            event_ids, mapping, use_annotations = event_config[paradigm][site]
        except KeyError:
            raise ValueError(f"Unsupported paradigm/site combination: {paradigm}/{site}")

        if use_annotations:
            events = events_from_annotations(raw)[0]
        else:
            events = find_events(raw)

        mask = np.isin(events[:, 2], event_ids)
        sub_evs = events[mask]

    if paradigm == "omi" and np.count_nonzero(sub_evs[:, 2] == event_ids[0]) != 125:
        raise ValueError(f"omission paradigm must have 125 events got {len(sub_evs)} instead.")

    if paradigm == "xxxxx":
        if not np.count_nonzero(sub_evs[:, 2] == event_ids[0]) == 500:
            raise ValueError(f"{paradigm} paradigm must have 500 events with ID 1 got {np.count_nonzero(sub_evs[:, 2] == event_ids[0])} instead.")
        if not np.count_nonzero(sub_evs[:, 2] == event_ids[1]) == 75:
            raise ValueError(f"{paradigm} paradigm must have 500 events with ID 2 got {np.count_nonzero(sub_evs[:, 2] == event_ids[1])} instead.")
        if not np.count_nonzero(sub_evs[:, 2] == event_ids[2]) == 50:
            raise ValueError(f"{paradigm} paradigm must have 500 events with ID 3 got {np.count_nonzero(sub_evs[:, 2] == event_ids[2])} instead.")
        
    if paradigm == "xxxxy":
        if not np.count_nonzero(sub_evs[:, 2] == event_ids[0]) == 500:
            raise ValueError(f"{paradigm} paradigm must have 500 events with ID 11 got {np.count_nonzero(sub_evs[:, 2] == event_ids[0])} instead.")
        if not np.count_nonzero(sub_evs[:, 2] == event_ids[1]) == 75:
            raise ValueError(f"{paradigm} paradigm must have 500 events with ID 12 got {np.count_nonzero(sub_evs[:, 2] == event_ids[1])} instead.")
        if not np.count_nonzero(sub_evs[:, 2] == event_ids[2]) == 50:
            raise ValueError(f"{paradigm} paradigm must have 500 events with ID 13 got {np.count_nonzero(sub_evs[:, 2] == event_ids[2])} instead.")    

    if site in ["Illinois"]:
        sub_evs[:, 0] = sub_evs[:, 0] + 0.5 * raw.info["sfreq"]
    
    if site in ["Zuerich"]:
        sub_evs[:, 0] = sub_evs[:, 0] + 0.1 * raw.info["sfreq"]

    annot = annotations_from_events(
                                    events=sub_evs,
                                    sfreq=raw.info["sfreq"],
                                    event_desc=mapping,
                                    orig_time=raw.info["meas_date"]
                                )
    raw.set_annotations(annot)

    return raw
