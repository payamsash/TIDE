# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import os
from pathlib import Path
from ast import literal_eval
from tqdm import tqdm
import time
import matplotlib.pyplot as plt

from mne import set_log_level, Report, concatenate_raws
from mne_icalabel import label_components
from mne_icalabel.gui import label_ica_components
from mne.io import read_raw
from mne.channels import read_dig_captrak, make_standard_montage
from mne.viz import plot_projs_joint
from mne.preprocessing import (ICA,
                                create_eog_epochs,
                                create_ecg_epochs,
                                compute_proj_ecg,
                                compute_proj_eog
                                )

def preprocessing(
        subject_id,
        subjects_dir=None,
        site="Zuerich",
        paradigm="gpias",
        psd_check=True,
        manual_data_scroll=True,
        run_ica=False,
        manual_ica_removal=False,
        ssp_eog=True,
        ssp_ecg=True,
        create_report=True,
        saving_dir=None,
        verbose="ERROR"
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
        manual_data_scroll : bool
            If True, user can interactively annotate segments of the recording to be removed.
            If not, this step will be skipped.
        run_ica: bool
            If True, ICA will be perfomed to detect and remove eye movement components from data.
            This option is set for the gpias paradigm.
        manual_ica_removal : bool
            If True, a window will pop up asking ICA components to be removed.
            If not, a machine learning model will be used to remove ICA components related to eye movements.
        respiratory_correct : bool
            Not Implemented yet ...
        pulse_correct : bool
            Not Implemented yet ...
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
    tqdm.write("Loading raw EEG data ...\n")
    progress.update(1)

    if subjects_dir == None:
        subjects_dir = Path.cwd().parent / "subjects"
    else:
        subjects_dir = Path(subjects_dir)
    
    sites = ["Austin", "Dublin", "Ghent", "Illinois", "Regensburg", "Tuebingen", "Zuerich"]
    if site not in sites:
        raise ValueError(f"site option must be one of {sites}, got {site} instead.")

    fname_paradigm = subjects_dir / subject_id / "EEG" / paradigm 
    fnames = [f for f in sorted(os.listdir(fname_paradigm)) if not f.startswith(".")]
    assert len(fnames) > 0, f"EEG data not found in this directory: {fname_paradigm}!"

    match site:
        case "Austin":
            raws = [read_raw(fname_paradigm / fname) for fname in fnames if fname.endswith(".edf")]
            raw = concatenate_raws(raws)
            raw.rename_channels(mapping=lambda s: s[:1] + s[4:])
            montage = make_standard_montage("GSN-HydroCel-64_1.0")

        case "Dublin":
            raws = [read_raw(fname_paradigm / fname) for fname in fnames if fname.endswith(".bdf")]
            raw = concatenate_raws(raws)
            raw.pick(["eeg", "stim"])
            montage = make_standard_montage("easycap-M1")

        case "Ghent":
            raise NotImplementedError

        case "Illinois": # curry ** last thing to check **
            raise NotImplementedError
            
        case "Regensburg":
            raws = []
            for fname in fnames:
                if fname.endswith(".vhdr"):
                    raws.append(_read_vhdr_input_fname(fname_paradigm / fname, subject_id, paradigm))
            raw = concatenate_raws(raws)
            montage = make_standard_montage("easycap-M1")
            raw.pick(["eeg", "stim"])

        case "Tuebingen":
            raws = [read_raw(fname_paradigm / fname) for fname in fnames if fname.endswith(".ds")]
            raw = concatenate_raws(raws)
            raw.pick(["eeg", "stim"])

        case "Zuerich": 
            fname = fname_paradigm / f"{subject_id}_{paradigm}.vhdr"
            if not fname.exists():
                raise ValueError(f"Subject {subject_id}_{paradigm}.vhdr not found in the EEG directory!")
            
            captrak_dir = subjects_dir / subject_id / "EEG" / "captrack"
            try:
                for file_ck in os.listdir(captrak_dir):
                    if file_ck.endswith(".bvct"): 
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
            
            raw = _read_vhdr_input_fname(fname, subject_id, paradigm)
            raw.set_channel_types(ch_types)
            raw.pick(["eeg", "eog", "ecg", "stim"])
    
    raw.load_data()
    raw.set_montage(montage=montage, match_case=False, on_missing="warn")
    raw.info["subject_info"] = {"his_id": site}

    ## resampling, filtering and re-referencing 
    tqdm.write("Resampling, filtering and re-referencing ...\n")
    progress.update(1)
    raw.resample(sfreq=250, stim_picks=None)

    if paradigm.startswith("rest"):
        l_freq, h_freq = 0.1, 100
        if site in ["Illinois", "Austin"]:
            line_freq = 60
        else:
            line_freq = 50
        raw.notch_filter(freqs=line_freq, picks="eeg", notch_widths=1)
    else:
        l_freq, h_freq = 1, 40

    raw.filter(picks="eeg", l_freq=l_freq, h_freq=h_freq)
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()
    
    ## eeg plotting for annotating
    if psd_check:
        fig, ax = plt.subplots(1, 1, figsize=(10, 4))
        raw.plot_psd(picks="eeg", fmin=0.1, fmax=120, ax=ax)
        while plt.fignum_exists(fig.number):
            plt.pause(0.1)

    if manual_data_scroll:
        raw.annotations.append(onset=0, duration=0, description="bad_segment")
        raw.plot(duration=20.0, n_channels=80, picks="eeg", scalings=dict(eeg=40e-6), block=True)
    raw.interpolate_bads()
    
    ## ICA
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
                                                show=False,
                                                )
            eog_indices_fil = [x for x in eog_indices if x <= 10]
        ica.apply(raw, exclude=eog_indices_fil)
    
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

    if ssp_eog:
        tqdm.write("Finding and removing vertical and horizontal EOG components...\n")
        progress.update(1)

        ## vertical
        ev_eog = create_eog_epochs(raw, ch_name=["PO7", "PO8"]).average(picks="all")
        ev_eog.apply_baseline((None, None))
        veog_projs, _ = compute_proj_eog(raw, n_eeg=2, reject=None)
        raw.add_proj(veog_projs)

        ## horizontal
        ica = ICA(n_components=0.97, max_iter=800, method='infomax', fit_params=dict(extended=True))        
        ica.fit(raw)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=["O1", "O2"], threshold=2)
        eog_indices_fil = [x for x in eog_indices if x <= 10]
        heog_idxs = [eog_idx for eog_idx in eog_indices_fil if eog_scores[0][eog_idx] * eog_scores[1][eog_idx] < 0]
        fig_scores = ica.plot_scores(scores=eog_scores, exclude=eog_indices_fil)

        if len(heog_idxs) > 0:
            eog_sac_components = ica.plot_properties(raw,
                                                picks=heog_idxs,
                                                show=False,
                                                )
        ica.apply(raw, exclude=heog_idxs)

    raw.apply_proj()

    # creating and saving report
    tqdm.write("Creating report and saving...\n")
    progress.update(1)
    if create_report:
        report = Report(title=f"report_subject_{subject_id}")
        report.add_raw(raw=raw, title="Recording Info", butterfly=False, psd=True)

        if run_ica:
            if len(eog_indices_fil) > 0:
                report.add_figure(fig=eog_components, title="EOG Components", image_format="PNG")
        
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

        if saving_dir == None:
            saving_dir = subjects_dir / subject_id / "EEG" / f"{paradigm}"
        report.save(fname=saving_dir.parent / "reports" / f"{paradigm}.h5", open_browser=False, overwrite=True)

    if not saving_dir is False:
        raw.save(fname=saving_dir / "raw_prep.fif", overwrite=True)
        
    tqdm.write("\033[32mEEG data were preprocessed sucessfully!\n")
    progress.update(1)
    progress.close()


def _read_vhdr_input_fname(fname, subject_id, paradigm):
    """
    Checks .vhdr and .vmrk data to have same names, otherwise fix them.
    """
    try:
        raw = read_raw(fname)
    except:
        with open(fname, "r") as file:
            lines = file.readlines()

        lines[5] = f'DataFile={subject_id}_{paradigm}.eeg\n'
        lines[6] = f'MarkerFile={subject_id}_{paradigm}.vmrk\n'

        with open(fname, "w") as file:
            file.writelines(lines)
        with open(f"{str(fname)[:-4]}vmrk", "r") as file:
            lines = file.readlines()

        lines[4] = f'DataFile={subject_id}_{paradigm}.eeg\n'
        with open(f"{str(fname)[:-4]}vmrk", "w") as file:
            file.writelines(lines)

        raw = read_raw(fname)
    return raw