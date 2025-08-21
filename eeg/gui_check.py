import numpy as np
import mne
import customtkinter as ctk

# Example TTL function (yours may differ)
def multi_level_ttl(signal, thresholds):
    ttl = np.zeros_like(signal, dtype=int)
    for i, thr in enumerate(thresholds, start=1):
        ttl[signal >= thr] = i
    return ttl

class ThresholdGUI(ctk.CTk):
    def __init__(self, raw, data):
        super().__init__()

        self.raw = raw
        self.data = data
        self.title("GPIAS event detector")
        self.geometry("500x400")

        # default thresholds
        self.thresholds = [300, 700, 1250, 1700, 2155]
        self.sliders = []

        # UI
        ctk.CTkLabel(self, text="Adjust thresholds:").pack(pady=5)

        for i, val in enumerate(self.thresholds):
            frame = ctk.CTkFrame(self)
            frame.pack(fill="x", pady=2, padx=5)

            ctk.CTkLabel(frame, text=f"Thr{i+1}").pack(side="left", padx=5)
            slider = ctk.CTkSlider(frame, from_=0, to=3000, number_of_steps=300, width=300)
            slider.set(val)
            slider.pack(side="left", fill="x", expand=True, padx=5)
            self.sliders.append(slider)

        self.textbox = ctk.CTkTextbox(self, height=150, width=400)
        self.textbox.pack(pady=10)

        btn_frame = ctk.CTkFrame(self)
        btn_frame.pack(pady=5)

        self.update_btn = ctk.CTkButton(btn_frame, text="Update", command=self.update)
        self.update_btn.pack(side="left", padx=10)

        self.ok_btn = ctk.CTkButton(btn_frame, text="OK", command=self.ok)
        self.ok_btn.pack(side="left", padx=10)

        self.final_thresholds = None

    def update(self):
        thresholds = np.sort([s.get() for s in self.sliders])
        ttl = multi_level_ttl(abs(self.data), thresholds)

        stim_data = np.array([ttl])
        info = mne.create_info(["STI1"], self.raw.info['sfreq'], ['stim'])
        stim_raw = mne.io.RawArray(stim_data, info)

        raw_copy = self.raw.copy().drop_channels(["STI1"], on_missing="ignore")
        raw_copy.add_channels([stim_raw], force_update_info=True)

        events = mne.find_events(raw_copy, stim_channel="STI1",
                                 output="onset", min_duration=0,
                                 shortest_event=1, verbose=False)

        self.textbox.delete("1.0", "end")
        self.textbox.insert("end", f"Thresholds: {thresholds}\n")
        for i in range(1, len(thresholds)+1):
            count = np.count_nonzero(events[:, 2] == i)
            self.textbox.insert("end", f"Code {i}: {count} events\n")

    def ok(self):
        self.final_thresholds = np.sort([s.get() for s in self.sliders])
        self.destroy()  # close window

# ---------------- Run ----------------
def run_threshold_gui(raw, data):
    ctk.set_appearance_mode("dark")  # or "light"
    app = ThresholdGUI(raw, data)
    app.mainloop()
    return app.final_thresholds

if __name__ == "__main__":
    
    fname = '/Users/payamsadeghishabestari/temp_folder/ynkf_gpias.vhdr'
    raw = mne.io.read_raw_brainvision(fname, preload=True)
    data = raw.get_data(picks="Audio")[0]
    run_threshold_gui(raw, data)