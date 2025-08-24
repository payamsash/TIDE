import numpy as np
import mne
import customtkinter as ctk


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

if __name__ == "__main__":
    
    # fname = '/Users/payamsadeghishabestari/temp_folder/ynkf_gpias.vhdr'
    # raw = mne.io.read_raw_brainvision(fname, preload=True)
    # data = raw.get_data(picks="Audio")[0]
    # run_threshold_gui(raw, data)


    fname1 = "/Users/payamsadeghishabestari/Downloads/OneDrive_1_24-08-2025/50078_gapdetection_session1.vhdr"
    fname2 = "/Users/payamsadeghishabestari/Downloads/OneDrive_1_24-08-2025/50078_gapdetection_bbn.vhdr"
    fname3 = "/Users/payamsadeghishabestari/Downloads/OneDrive_1_24-08-2025/50078_gapdetection_nbn3kHz.vhdr"
    fname4 = "/Users/payamsadeghishabestari/Downloads/OneDrive_1_24-08-2025/50078_gapdetection_nbn8kHz.vhdr"
    fname5 = "/Users/payamsadeghishabestari/Downloads/OneDrive_1_24-08-2025/50078_gapdetection_session5.vhdr"

    raws = []
    ch_types = {"audio": "stim"}
    montage = mne.channels.make_standard_montage("easycap-M1")
    for f_idx, fname in enumerate([fname1, fname2, fname3, fname4, fname5]):
        raw = mne.io.read_raw_brainvision(fname)
        raw.drop_channels(["HRli", "HRre"], on_missing="warn")
        try:
            raw.set_channel_types(ch_types)
        except: 
            print("probably this data is old recording from Regensburg")
        raw.pick(["eeg", "stim"])
        raw.load_data()

        raws.append(raw)

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
    print(events_dict)