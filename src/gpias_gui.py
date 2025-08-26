import sys
import os
from pathlib import Path
import argparse
import numpy as np
import mne
import customtkinter as ctk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eeg.tools import read_vhdr_input_fname


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
            self.raw.get_data(picks="audio")[0], # regensburg
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


def main():
    parser = argparse.ArgumentParser(description=("""
    ************************************
    Gui for event detection of the gpias recordings.



    Notes
    -----
    This script is mainly designed for Antinomics / TIDE projects, however could be used for other purposes.
    """
    ),
    formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("fname", help="file path to the eeg recording.")
    parser.add_argument("subject_id", help="The subject name, if subject has MRI data as well, should be FreeSurfer subject name, then data from both modality can be analyzed at once.")
    parser.add_argument("site", help="The recording site; must be one of the following: ['Austin', 'Dublin', 'Ghent', 'Illinois', 'Regensburg', 'Tuebingen', 'Zuerich']")
    args = parser.parse_args()
    fname, subject_id, site = list(vars(args).values())

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
                montage = make_standard_montage("easycap-M1")

                if subject_id in [70001, 70002, 70003, 70007, 70009, 70030, 70042, 70052]: # old recordings
                    ssp_eog = True # will overwrite the option selected by user
                    ch_types = {
                            "O1": "eog",
                            "O2": "eog",
                            "PO7": "eog",
                            "PO8": "eog",
                            "Pulse": "ecg",
                            "Resp": "ecg",
                            "Audio": "stim"
                            }
                    eog_chs_1 = ["PO7", "PO8"]
                    eog_chs_2 = ["O1", "O2"]
                
                else:
                    ch_types = {
                                "Pulse": "ecg",
                                "Resp": "ecg",
                                "Audio": "stim"
                                }
                
                raw = read_raw(Path(fname))
                raw.set_channel_types(ch_types)
                raw.pick(["eeg", "eog", "ecg", "stim"])
                
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
                        
                        raw.set_annotations(None)
                        raw.load_data()
                        raw.annotations.append(onset=0, duration=0, description=f"s_{f_idx+1}")
                        raws.append(raw)
                
                ch_types = {"audio": "stim"}
                try:
                    raw.set_channel_types(ch_types)
                except: 
                    print("probably this data is old recording from Regensburg")
                raw.pick(["eeg", "stim"])

    titles = ["pre", "bbn", "3khz", "8khz", "post"]
    my_dict = {
                                "pre": [0.003, 0.01, 0.017, 0.02185, 0.0285],
                                "bbn": [0.005, 0.015, 0.021, 0.026],
                                "3kHz": [0.005, 0.015, 0.021, 0.026],
                                "8kHz": [0.005, 0.015, 0.021, 0.026],
                                "post": [0.003, 0.01, 0.017, 0.02185, 0.0285]
                                }
    thresholds_list = list(my_dict.values())

    events_dict = run_multi_threshold_gui(raws, titles, thresholds_list)
    updated_raws = []
    mapping = {
                "pre": {1: "PO70", 2: "PO75", 3: "PO80", 4: "PO85", 5: "PO90"},
                "bbn": {1: "GO_bbn", 2: "G_bbn", 3: "GP_bbn", 4: "PO_bbn"},
                "3khz": {1: "GO_3", 2: "G_3", 3: "GP_3", 4: "PO_3"},
                "8khz": {1: "GO_8", 2: "G_8", 3: "GP_8", 4: "PO_8"},
                "post": {1: "PO70", 2: "PO75", 3: "PO80", 4: "PO85", 5: "PO90"}
            }
    for raw, title in zip(raws, titles):
        annot_from_events = mne.annotations_from_events(    
                                                        events_dict[title],
                                                        sfreq=raw.info["sfreq"],
                                                        event_desc=mapping[title],
                                                        orig_time=raw.info["meas_date"]
                                                        )
        raw.set_annotations(annot_from_events)
        updated_raws.append(raw)

    final_raw = mne.concatenate_raws(updated_raws)
    fname_save = fname.parent / "raw_gpias.fif"
    final_raw.save(fname_save, overwrite=True)

if __name__ == "__main__":
    main()


