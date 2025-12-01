import sys
import os
from pathlib import Path
import argparse
import numpy as np
import mne

import csv
import copy
import itertools
from matplotlib.widgets import Button
from scipy.signal import find_peaks

import customtkinter as ctk
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from eeg.tools import read_vhdr_input_fname

import matplotlib
matplotlib.use("TkAgg")  # or "MacOSX"

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def pick_threshold_with_slider(raw, session_name, init_val):
    audio = raw.get_data(picks="audio")[0]
    times = raw.times

    fig, ax = plt.subplots(figsize=(10, 4))
    plt.subplots_adjust(bottom=0.25)  # make space for slider

    start = int(10000)
    stop = int(round(len(audio)/2))

    ax.plot(times[start:stop], audio[start:stop], lw=0.6, color="black")    
    hline = ax.axhline(y=init_val, color="red", lw=1.5, linestyle="--")
    ax.set_title(f"Set threshold for '{session_name}'")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")

    # Slider axis: [left, bottom, width, height] in figure coordinates
    ax_slider = plt.axes([0.15, 0.1, 0.7, 0.03])
    slider = Slider(ax_slider, 'Threshold', audio.min(), audio.max(), valinit=init_val)

    def update(val):
        hline.set_ydata([val, val])
        fig.canvas.draw_idle()

    slider.on_changed(update)

    plt.show(block=True)
    plt.close(fig)
    
    print(f"Selected threshold for {session_name}: {slider.val:.5f}")
    return slider.val


def create_multi_level_ttl(signal, thresholds, hold=25):
    # Step 1: Digitize the signal into discrete levels
    levels = np.digitize(signal, thresholds)  # 0 = below first threshold, 1 = between first and second, etc.
    #print("Levels preview:", levels[19000:20000])

    # Step 2: Filter out short runs before stretching
    filtered = np.zeros_like(levels)
    i = 0
    while i < len(levels):
        val = levels[i]
        if val == 0:
            i += 1
            continue
        #min_run = 2 if val == 1 else 5
        min_run = 1

        start = i
        while i < len(levels) and levels[i] == val:
            i += 1
        end = i
        run_length = end - start
        if run_length >= min_run:
            filtered[start:end] = val

    # Step 3: find peaks inside the filtered runs

    raw_peak_indices, _ = find_peaks(signal, distance=200, prominence = 0.05)    # Regensburg: prominence = 0.001
    print(f"Found {len(raw_peak_indices)} raw peaks")


    # Keep only peaks that fall inside valid filtered areas
    valid_peak_indices = [p for p in raw_peak_indices if filtered[p] > 0]
    print(f"Peaks inside valid runs: {len(valid_peak_indices)}")

    rising_peaks_indices = [p for p in valid_peak_indices 
                    if p > 0 and signal[p] > signal[p - 1]]
    
    # Debug: print peak values
    for p in rising_peaks_indices[:10]:
        print(f"Peak at {p}, level={filtered[p]}, amplitude={signal[p]:.5f}")

    # Step 4: Create TTL based on peak anchor
    ttl = np.zeros_like(levels)
    for p in rising_peaks_indices:
        level = filtered[p]              # the correct level at the peak
        start = max(0, p - 20)    # clear before peak
        end = min(len(ttl), p + hold)

        # Zero before peak (cleanup)
        ttl[start:p] = 0

        # Apply TTL pulse
        ttl[p:end] = level
    return ttl 


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
                number_of_steps=1000000,
                button_length=0.2
            )
            slider.set(val)
            slider.pack(side="left", padx=2)

            # lbl = ctk.CTkLabel(frame, text=f"{val:.1f}")  # live value only
            # lbl.pack(side="left", padx=2)
            # slider.configure(command=lambda v, l=lbl: l.configure(text=f"{float(v):.1f}"))

            # self.sliders.append(slider)
            # entry = ctk.CTkEntry(frame, width=50)
            # entry.insert(0, f"{val:.3f}")
            # entry.pack(side="left", padx=2)
            # def on_slider_change(v, e=entry):
            #     e.delete(0, "end")
            #     e.insert(0, f"{float(v):.5f}")
            # def on_entry_change(event, s=slider):
            #     try:
            #         val = float(event.widget.get())
            #         s.set(val)
            #     except ValueError:
            #         pass
            # slider.configure(command=lambda v: on_slider_change(v))
            # entry.bind("<Return>", on_entry_change)

            entry = ctk.CTkEntry(frame, width=50)
            entry.insert(0, f"{val:.5f}")
            entry.pack(side="left", padx=2)

            # bind slider → entry
            slider.configure(command=lambda v, e=entry: e.delete(0, "end") or e.insert(0, f"{float(v):.5f}"))
            # bind entry → slider
            entry.bind("<Return>", lambda event, s=slider: s.set(float(event.widget.get()) if event.widget.get() else 0))

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
            self.raw.get_data(picks="audio")[0],  # regensburg
            thresholds
        )

        stim_data = np.array([ttl])
        info = mne.create_info(["STI1"], self.raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)

        raw = self.raw.drop_channels(["STI1"], on_missing="warn")
        raw.add_channels([stim_raw], force_update_info=True)

        events = mne.find_events(raw, stim_channel="STI1", output="step", 
                                 consecutive="increasing",shortest_event=20, verbose=False)
        self.textbox.delete("1.0", "end")
        self.textbox.insert("end", f"Thresholds: {thresholds}\n")
        for i in range(1, len(thresholds) + 1):
            count = np.count_nonzero(events[:, 2] == i)
            self.textbox.insert("end", f"Code {i}: {count} events\n")

    def apply_and_save(self):
        thresholds = self.get_thresholds()
        ttl = create_multi_level_ttl(
        self.raw.get_data(picks="audio")[0],
        thresholds
        )

        self.raw.drop_channels("STI1", on_missing="ignore")
        stim_data = np.array([ttl])
    
        #print("=== TTL array preview ===")
        #print(ttl[16000:17000])  # first 200 samples for quick inspection
        #print("Nonzero TTL values and counts:", {val: np.count_nonzero(ttl == val) for val in np.unique(ttl)})

        info = mne.create_info(["STI1"], self.raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)
        self.raw.add_channels([stim_raw], force_update_info=True)

        events = mne.find_events(
            self.raw, stim_channel="STI1", output="step",
            consecutive="increasing", shortest_event=20, verbose=False
        )

        #print("=== Detected events ===")
        #print(events[:20])  # first 20 events
        #print("Unique TTL codes in events:", np.unique(events[:, 2]))
    
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

        self.saved_events = {}  # <-- store thresholds here
        self.final_thresholds = {}

    def apply_all(self):
        results = {}
        result_thresholds = {}
        for panel in self.panels:
            events = panel.apply_and_save()
            results[panel.title_text] = events
            result_thresholds[panel.title_text] = panel.get_thresholds()  

        self.saved_events = results
        self.final_thresholds = result_thresholds
        self.destroy()  # close window


def run_multi_threshold_gui(raws, titles, thresholds_list):
    ctk.set_appearance_mode("dark")
    app = MultiThresholdGUI(raws, titles, thresholds_list)
    app.mainloop()
    return app.saved_events, app.final_thresholds

    
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
    parser.add_argument("fname", help="file path to the eeg raw recording.")
    parser.add_argument("subject_id", help="The subject name, if subject has MRI data as well, should be FreeSurfer subject name, then data from both modality can be analyzed at once.")
    
    parser.add_argument("subjects_dir", help="Path to the subjects_dir, will be created if its not existing")
    
    parser.add_argument("site", help="The recording site; must be one of the following: ['Austin', 'Dublin', 'Ghent', 'Illinois', 'Regensburg', 'Tuebingen', 'Zuerich']")
    args = parser.parse_args()
    fname, subject_id, subjects_dir, site = list(vars(args).values())

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
    print(f"Number of files found: {len(fnames)}")

        
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
                print(f"Number of files found: {len(fnames)}")
                
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
                for raw in raws:
                    try:
                        raw.set_channel_types(ch_types)
                    except: 
                        print("probably this data is old recording from Regensburg")
                    raw.pick(["eeg", "stim"])
                
                #raw.pick(["eeg", "stim"])
                #raw.filter(l_freq=None, h_freq=50, picks="stim")

    titles = ["pre", "bbn", "3kHz", "8kHz", "post"]
    ## this dictionary is just a default for regensburg site (majority of subjects)
    ## you should adjust in based on your need (only affects Tuebingen, Regensburg and Zuerich)
    my_dict = {
                                # Threshold for Regensburg
                                # "pre": [0.005, 0.01, 0.017, 0.02185, 0.03026], 
                                # "bbn": [0.005, 0.015, 0.021, 0.026],
                                # "3kHz": [0.005, 0.015, 0.0248, 0.0312],
                                # "8kHz": [0.005, 0.015, 0.023, 0.0312],
                                # "post": [0.0068, 0.01, 0.017, 0.02185, 0.0285]
                                "pre": [0.2, 0.4, 0.6, 0.9, 1.2],
                                "bbn": [0.24, 0.5832, 0.7, 1.06],
                                "3kHz": [0.24, 0.5832, 0.7, 1.06],
                                "8kHz": [0.24, 0.5832, 0.7, 1.06],
                                "post": [0.2, 0.4, 0.56, 0.76, 1.2]
                                }
                                
    for key in my_dict.keys():
        #new_thresh = pick_threshold_with_slider(raw, key, my_dict[key][0])

        session_to_index = {"pre": 0, "bbn": 1, "3kHz": 2, "8kHz": 3, "post": 4}
        new_thresh = pick_threshold_with_slider(raws[session_to_index[key]], key, my_dict[key][0])  

        my_dict[key][0] = new_thresh

    #thresholds_list = list(my_dict.values())
    #events_dict = run_multi_threshold_gui(raws, titles, thresholds_list)
    
    thresholds_list = [copy.deepcopy(my_dict[key]) for key in titles]

    events_dict, final_thresholds = run_multi_threshold_gui(raws, titles, thresholds_list)
    
    updated_raws = []
    mapping = {
                "pre": {1: "PO70_pre", 2: "PO75_pre", 3: "PO80_pre", 4: "PO85_pre", 5: "PO90_pre"},
                "bbn": {1: "GO_bbn", 2: "G_bbn", 3: "GP_bbn", 4: "PO_bbn"},
                "3kHz": {1: "GO_3kHz", 2: "G_3kHz", 3: "GP_3kHz", 4: "PO_3kHz"},
                "8kHz": {1: "GO_8kHz", 2: "G_8kHz", 3: "GP_8kHz", 4: "PO_8kHz"},
                "post": {1: "PO70_post", 2: "PO75_post", 3: "PO80_post", 4: "PO85_post", 5: "PO90_post"}
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
    #fname_save = fname.parent / "raw_gpias.fif"

    # Create a new folder for trigger-defined data
    subjects_dir = Path(subjects_dir)
    trigger_defined_dir = subjects_dir / subject_id / "trigger_defined"
    trigger_defined_dir.mkdir(parents=True, exist_ok=True)

    # Filenames
    fname_save = trigger_defined_dir / f"{subject_id}_gpias_trigger_defined.fif"
    thresholds_save = trigger_defined_dir / f"{subject_id}_gpias_thresholds.csv"

    # Plot audio vs STI1 for each session
    for raw, title in zip(updated_raws, ["pre", "bbn", "3kHz", "8kHz", "post"]):
        audio_data = raw.get_data(picks="audio")[0]
        sti_data = raw.get_data(picks="STI1")[0]

        # Only show a small window (e.g., 200 samples)
        start = 0
        stop = len(audio_data)-1

        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(np.arange(start, stop), audio_data[start:stop], color="black", lw=0.7, label="Audio")
        ax.plot(np.arange(start, stop), sti_data[start:stop] / sti_data.max(), color="yellow", lw=2, alpha=0.7, label="STI1 Trigger (normalized)")
        ax.set_xlabel("Samples")
        ax.set_ylabel("Amplitude")
        ax.set_title(f"{title.upper()} Session: Audio vs STI1 Trigger")
        ax.legend()
        plt.show()
        plt.close(fig)
    # Save trigger-defined raw file
    print(f"Saving trigger-defined data to: {fname_save}")
    final_raw.save(fname_save, overwrite=True)

    with open(thresholds_save, "w", newline="") as f:
        writer = csv.writer(f)
        # Write header
        writer.writerow(["Session"] + [f"T{i+1}" for i in range(max(len(t) for t in final_thresholds.values()))])
        # Write each session
        for session, thrs in final_thresholds.items():
            writer.writerow([session] + list(thrs))

    print(f"Final thresholds saved to: {thresholds_save}")

if __name__ == "__main__":
    main()


