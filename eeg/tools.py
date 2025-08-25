import yaml
import logging
from pathlib import Path
import mne
from mne import sys_info
from mne.io import read_raw
import customtkinter as ctk

__all__ = ["load_config", "initiate_logging", "_check_preprocessing_inputs",
            "_check_processing_inputs", "_check_feature_extraction_inputs",
            "create_subject_dir", "read_vhdr_input_fname", "create_multi_level_ttl", 
            "run_multi_threshold_gui"]

def load_config(site,
                config_file):
    """
    Loads config file.
    """
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)
    return config.get(site, {})


def initiate_logging(logfile, config, analysis_type="preprocessing"):
    """
    Start logging file with system and code information.
    """

    ## add system and config information
    yaml_str = yaml.dump(config, default_flow_style=False)
    with open(logfile, 'w') as f:
        f.write('*' * 100 + '\n')
        f.write('{:^100}\n'.format(f'System Information'))
        f.write('*' * 100 + '\n\n')
        sys_info(fid=f)
        f.write('\n')

        f.write('*' * 100 + '\n')
        f.write('{:^100}\n'.format(f'Config Information'))
        f.write('*' * 100 + '\n\n')
        f.write(yaml_str)
        f.write('\n')

        f.write('*' * 100 + '\n')
        f.write('{:^100}\n'.format(f'{analysis_type.upper()}'))
        f.write('*' * 100 + '\n\n')

    logging.basicConfig(
                    filename=logfile,
                    level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S'
                    )
    return logging

def _check_preprocessing_inputs(fname,
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
                                ):
    """
    Checks input variables, raise or warn some messages.
    """

    ## initial checks
    if not isinstance(fname, (str, Path)): raise TypeError(f"fname must be str or Path object, got type {type(fname).__name} instead.")
    if not isinstance(subject_id, str): raise TypeError(f"subject_id must be str, got type {type(subject_id).__name} instead.")
    if not isinstance(subjects_dir, (str, Path)): raise TypeError(f"subjects_dir must be str or Path object, got type {type(subjects_dir).__name} instead.")
    
    sites = ["Austin", "Dublin", "Ghent", "Illinois", "Regensburg", "Tuebingen", "Zuerich"] 
    if not site in sites: raise ValueError(f"site must be one of the {sites}.")
    
    paradigms = ["gpias", "xxxxx", "xxxxy", "omi", "regularity"]
    if not (paradigm in paradigms or paradigm.startswith("rest")):
        raise ValueError(f"paradigm must be one of the {paradigms} or should start with 'rest'.")

    for var_name, var_value in {
                                "psd_check": psd_check,
                                "manual_data_scroll": manual_data_scroll,
                                "run_ica": run_ica,
                                "manual_ica_removal": manual_ica_removal,
                                "ssp_eog": ssp_eog,
                                "ssp_ecg": ssp_ecg,
                                "create_report": create_report
                                }.items():
        if not isinstance(var_value, bool): raise TypeError(f"{var_name} must be boolean, got type {type(var_value).__name__} instead.")

    verboses = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if not (verbose in verboses or isinstance(verbose, bool)): raise ValueError(f"verbose must be one of the {verboses} or boolean.") 

    overwrite_options = ["warn", "ignore", "raise"]
    if not overwrite in overwrite_options: raise ValueError(f"overwrite must be one of the {overwrite_options}.")
        
    # site and data format checks
    ext_site_map = {
                    ".mff": "Austin",
                    ".bdf": "Dublin",
                    ".cdt": "Illinois",
                    ".cnt": "Ghent",
                    ".vhdr": ["Zuerich", "Regensburg", "Tuebingen"],
                    }

    for ext, expected in ext_site_map.items():
        if Path(fname).suffix == ext:
            if isinstance(expected, list):
                assert site in expected, f"site is not selected correctly for {ext}."
            else:
                assert site == expected, f"site is not selected correctly for {ext}."
            break

def _check_processing_inputs(manual_data_scroll,
                                automatic_epoch_rejection,
                                source_analysis,
                                subjects_fs_dir,
                                create_report,
                                overwrite,
                                verbose,
                                ):
    """
    Checks input variables, raise or warn some messages.
    """

    ## initial checks
    for var_name, var_value in {
                                "manual_data_scroll": manual_data_scroll,
                                "source_analysis": source_analysis,
                                "create_report": create_report
                                }.items():
        if not isinstance(var_value, bool): raise TypeError(f"{var_name} must be boolean, got type {type(var_value).__name__} instead.")

    automatic_epoch_rejection_options = ["ptp", "autoreject", "pyriemann"]
    if not (automatic_epoch_rejection in automatic_epoch_rejection_options or automatic_epoch_rejection is None):
        raise ValueError(f"automatic_epoch_rejection must be None or one of the {automatic_epoch_rejection_options}.")
    
    if not (subjects_fs_dir is None or isinstance(subjects_fs_dir, (str, Path))): 
        raise TypeError("subjects_fs_dir must be None or a path to FS subjects directory.")
    
    verboses = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if not (verbose in verboses or isinstance(verbose, bool)): raise ValueError(f"verbose must be one of the {verboses} or boolean.") 

    overwrite_options = ["warn", "ignore", "raise"]
    if not overwrite in overwrite_options: raise ValueError(f"overwrite must be one of the {overwrite_options}.")

def _check_feature_extraction_inputs(
                                    sensor_space_features,
                                    source_space_power,
                                    sensor_space_connectivity,
                                    source_space_connectivity,
                                    connectivity_method,
                                    subjects_fs_dir,
                                    atlas,
                                    freq_bands,
                                    overwrite,
                                    verbose
                                    ):
    """
    Checks input variables, raise or warn some messages.
    """

    ## initial checks
    if not isinstance(sensor_space_features, list):
        raise TypeError(f"{sensor_space_features} must be list, got type {type(sensor_space_features).__name__} instead.")
    for var_name, var_value in {
                                "source_space_power": source_space_power,
                                "sensor_space_connectivity": sensor_space_connectivity,
                                "source_space_connectivity": source_space_connectivity,
                                }.items():
        if not isinstance(var_value, bool):
            raise TypeError(f"{var_name} must be boolean, got type {type(var_value).__name__} instead.")

    connectivity_method_options = ["coh", "pli", "wpli", "cacoh", "mic", "mim", "gc", "gc_tr"]
    if not connectivity_method in connectivity_method_options:
        raise ValueError(f"method must be one of the {connectivity_method_options}.")
    
    if not (subjects_fs_dir is None or isinstance(subjects_fs_dir, (str, Path))):
        raise TypeError(f"subjects_fs_dir must be None or path to subjects FS directory.")
    if not atlas in ["aparc", "aparc.a2009s"]: raise ValueError(f"atlas must be one of the ['aparc', 'aparc.a2009s'].")
    if not isinstance(freq_bands, dict): raise TypeError(f"freq_bands must be dict, got type {type(freq_bands).__name__} instead.")
    
    verboses = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
    if not (verbose in verboses or isinstance(verbose, bool)): raise ValueError(f"verbose must be one of the {verboses} or boolean.") 
    overwrite_options = ["warn", "ignore", "raise"]
    if not overwrite in overwrite_options: raise ValueError(f"overwrite must be one of the {overwrite_options}.")
    

def create_subject_dir(subject_id, subjects_dir, site):
    """
    Create a structured directory for a subject under the given subjects directory.
    
    Structure:
    subjects_dir/subject_id/
        ├── orig/
        ├── preprocessed/
        ├── mri/
        ├── inv/
        ├── epochs/
        ├── reports/
        ├── logs/
    """
    base_path = Path(subjects_dir) / subject_id
    subdirs = [
                "orig", "preprocessed", "features",
                "inv", "epochs", "reports", "logs",
                ]

    for subdir in subdirs:
        path = base_path / subdir
        path.mkdir(parents=True, exist_ok=False)

    if site == "Zuerich":
        path = base_path / "captrak"
        path.mkdir(parents=True, exist_ok=False)

def read_vhdr_input_fname(fname):
    """
    Checks .vhdr and .vmrk data to have same names, otherwise fix them.
    """
    try:
        raw = read_raw(fname)
    except:
        with open(fname, "r") as file:
            lines = file.readlines()
        
        lines[5] = f'DataFile={fname.stem}.eeg\n'
        lines[6] = f'MarkerFile={fname.stem}.vmrk\n'

        with open(fname, "w") as file:
            file.writelines(lines)
        with open(f"{fname.with_suffix('')}.vmrk", "r") as file:
            lines = file.readlines()
        lines[4] = f'DataFile={fname.stem}.eeg\n'
        with open(f"{fname.with_suffix('')}.vmrk", "w") as file:
            file.writelines(lines)
        raw = read_raw(fname)
    return raw

def create_multi_level_ttl(signal, thresholds, hold=25):
    levels = np.digitize(signal, thresholds)
    stretched = levels.copy()

    i = 0
    while i < len(stretched):
        if stretched[i] != 0:
            end = min(i + hold, len(stretched))
            stretched[i:end] = stretched[i + 1]
            i = end   
        else:
            i += 1
    return stretched

class ThresholdPanel(ctk.CTkFrame):
    def __init__(self, parent, raw, title, init_thresholds):
        super().__init__(parent, corner_radius=8, border_width=1)

        self.raw = raw
        self.title_text = title
        self.sliders = []

        ctk.CTkLabel(self, text=title, font=("Arial", 13, "bold")).pack(pady=2)

        # pick slider range dynamically (to match your scale)
        tmin = 0
        tmax = max(init_thresholds) * 1.2 if max(init_thresholds) > 0 else 10

        for i, val in enumerate(init_thresholds):
            frame = ctk.CTkFrame(self)
            frame.pack(fill="x", pady=1, padx=3)

            ctk.CTkLabel(frame, text=f"T{i+1}").pack(side="left", padx=2)

            slider = ctk.CTkSlider(
                frame,
                from_=tmin,
                to=tmax,
                width=200,              
                number_of_steps=1000,   
                button_length=0.2        
            )
            slider.set(val)
            slider.pack(side="left", padx=2)

            lbl = ctk.CTkLabel(frame, text=f"{val:.1f}")   # live value only
            lbl.pack(side="left", padx=2)

            slider.configure(command=lambda v, l=lbl: l.configure(text=f"{float(v):.1f}"))
            self.sliders.append(slider)

        # make textbox taller so no scrolling needed
        self.textbox = ctk.CTkTextbox(self, height=150, width=180)
        self.textbox.pack(pady=3)

        self.update_btn = ctk.CTkButton(self, text="Update", width=100, command=self.update)
        self.update_btn.pack(pady=2)


    def get_thresholds(self):
        return np.sort([s.get() for s in self.sliders])

    def update(self):
        thresholds = self.get_thresholds()
        ttl = create_multi_level_ttl(
            self.raw.get_data(picks="audio")[0],
            thresholds
        )

        stim_data = np.array([ttl])
        info = mne.create_info(["STI1"], self.raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)

        raw = self.raw.drop_channels(["STI1"], on_missing="warn")
        raw.add_channels([stim_raw], force_update_info=True)

        events = mne.find_events(raw, stim_channel="STI1",
                                output="onset", verbose=False)

        self.textbox.delete("1.0", "end")
        self.textbox.insert("end", f"Thresholds: {thresholds}\n")
        for i in range(1, len(thresholds) + 1):
            count = np.count_nonzero(events[:, 2] == i)
            self.textbox.insert("end", f"Code {i}: {count} events\n")

    def apply_and_save(self):
        """Inject STI1 channel + save events into self.raw"""
        thresholds = self.get_thresholds()
        ttl = create_multi_level_ttl(
            self.raw.get_data(picks="audio")[0],
            thresholds
        )
        self.raw.drop_channels("STI1", on_missing="ignore")
        stim_data = np.array([ttl])
        info = mne.create_info(["STI1"], self.raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        self.raw.add_channels([stim_raw], force_update_info=True)

        events = mne.find_events(self.raw, stim_channel="STI1",
                                output="onset", verbose=False)
        return events


class MultiThresholdGUI(ctk.CTk):
    def __init__(self, raws, titles, thresholds_list):
        super().__init__()
        self.title("GPIAS Threshold GUI")
        self.geometry("1330x450")  # smaller height, wider layout

        container = ctk.CTkFrame(self)
        container.pack(fill="both", expand=True)

        self.panels = []
        for col, (raw, title, thrs) in enumerate(zip(raws, titles, thresholds_list)):
            panel = ThresholdPanel(container, raw, title, thrs)
            panel.grid(row=0, column=col, padx=5, pady=5, sticky="n")
            self.panels.append(panel)

        self.apply_btn = ctk.CTkButton(self, text="Apply All", command=self.apply_all)
        self.apply_btn.pack(pady=5)

        self.saved_events = {}

    def apply_all(self):
        results = {}
        for panel in self.panels:
            events = panel.apply_and_save()
            results[panel.title_text] = events
        self.saved_events = results
        self.destroy()  # close window


def run_multi_threshold_gui(raws, titles, thresholds_list):
    ctk.set_appearance_mode("dark")
    app = MultiThresholdGUI(raws, titles, thresholds_list)
    app.mainloop()
    return app.saved_events