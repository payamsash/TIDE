# Written by Payam S. Shabestari, Zurich, 01.2025 
# Email: payam.sadeghishabestari@uzh.ch

import os
from pathlib import Path
from ast import literal_eval
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import customtkinter as ctk

from mne import set_log_level, Report
from mne_icalabel import label_components
from mne.io import read_raw_brainvision
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
        paradigm="gpias",
        manual_data_scroll=True,
        run_ica=False,
        manual_ica_removal=False,
        eog_correct=True,
        resp_correct=True,
        pulse_correct=True,
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

    fname = subjects_dir / subject_id / "EEG" / paradigm / f"{subject_id}_{paradigm}.vhdr"
    captrak_dir = subjects_dir / subject_id / "EEG" / "captrack"
    if not fname.exists():
        raise ValueError(f"Subject {subject_id}_{paradigm}.vhdr not found in the EEG directory!")

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
    
    raw = read_raw_brainvision(vhdr_fname=fname)
    raw.set_channel_types(ch_types)
    raw.pick(["eeg", "eog", "ecg", "stim"])
    raw.load_data()
    raw.set_montage(montage=montage)

    ## resampling, filtering and re-referencing 
    tqdm.write("Resampling, filtering and re-referencing ...\n")
    progress.update(1)
    raw.resample(sfreq=250, stim_picks=None)

    if paradigm.startswith("rest"):
        l_freq, h_freq = 0.1, 100
        raw.notch_filter(freqs=50, picks="eeg")
    else:
        l_freq, h_freq = 1, 40

    raw.filter(picks="eeg", l_freq=l_freq, h_freq=h_freq)
    raw.set_eeg_reference("average", projection=True)
    raw.apply_proj()
    
    ## eeg plotting for annotating
    if manual_data_scroll:
        raw.annotations.append(onset=0, duration=0, description="bad_segment")
        raw.plot(duration=20.0, n_channels=80, picks="eeg", scalings=dict(eeg=40e-6), block=True)

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
            ica.plot_properties(raw)
            ctk.set_appearance_mode("Dark")
            ctk.set_default_color_theme("blue")  
            app = ctk.CTk()
            app.title("Remove ICA Components")
            app.geometry("500x300")
            eog_indices = None
            label = ctk.CTkLabel(app, text="Enter a list of ICA component indices to remove from the data\n\nfor example: [0, 2]",
                                font=("Arial", 14, "bold"))
            label.pack(pady=20)
            entry = ctk.CTkEntry(app, placeholder_text="Enter a list")
            entry.pack(pady=20)

            def on_button_click():
                global eog_indices
                try:
                    eog_indices = literal_eval(entry.get())
                    label_result.configure(text=f"Selected components are removed from data")
                except ValueError:
                    label_result.configure(text="Please enter a valid list of numbers")
                # app.quit() # fails in mac, but might work in windows
                
            button = ctk.CTkButton(app, text="Apply", command=on_button_click)
            button.pack()
            label_result = ctk.CTkLabel(app, text="")
            label_result.pack(pady=20)
            app.mainloop()

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
        ica.apply(raw, exclude=eog_indices)
    
    if pulse_correct:
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

    if eog_correct:
        tqdm.write("Finding and removing vertical and horizontal EOG components...\n")
        progress.update(1)

        ## vertical
        ev_eog = create_eog_epochs(raw, ch_name=["PO7", "PO8"]).average(picks="all")
        ev_eog.apply_baseline((None, None))
        veog_projs, _ = compute_proj_eog(raw, n_eeg=2, reject=None)

        ## horizontal
        ica = ICA(n_components=0.97, max_iter=800, method='infomax', fit_params=dict(extended=True))        
        ica.fit(raw)
        eog_indices, eog_scores = ica.find_bads_eog(raw, ch_name=["O1", "O2"], threshold=2)
        heog_idxs = [eog_idx for eog_idx in eog_indices if eog_scores[0][eog_idx] * eog_scores[1][eog_idx] < 0]
        fig_scores = ica.plot_scores(scores=eog_scores, exclude=eog_indices)

        if len(heog_idxs) > 0:
            eog_sac_components = ica.plot_properties(raw,
                                                picks=heog_idxs,
                                                show=False,
                                                )
        ica.apply(raw, exclude=heog_idxs)

    if resp_correct:
        raise NotImplementedError

    raw.apply_proj()

    # creating and saving report
    tqdm.write("Creating report and saving...\n")
    progress.update(1)
    if create_report:
        report = Report(title=f"report_subject_{subject_id}")
        report.add_raw(raw=raw, title="Recording Info", butterfly=False, psd=False)

        if run_ica:
            if len(eog_indices) > 0:
                report.add_figure(fig=eog_components, title="EOG Components", image_format="PNG")
        
        if pulse_correct:
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

        if eog_correct:
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
        raw.save(fname=saving_dir / "raw_prep.fif")
        
    tqdm.write("\033[32mEEG data were preprocessed sucessfully!\n")
    progress.update(1)
    progress.close()

