from pathlib import Path
import numpy as np
import mne

def correct_xy_triggers(fname):

    fname = Path(fname)
    dpo_file = fname.with_suffix('.cdt.dpo')
    dpa_file = fname.with_suffix('.cdt.dpa')
    if dpo_file.exists():
        dpo_file.rename(dpa_file)

    raw = mne.io.read_raw(fname)
    events, events_id = mne.events_from_annotations(raw)
    events = np.where(events == events_id["15"], events_id["12"],
                    np.where(events == events_id["17"], events_id["13"], events))
    
    keys_to_select = [np.str_('11'), np.str_('12'), np.str_('13')]
    events_id = {k: events_id[k] for k in keys_to_select if k in events_id}
    annots = mne.annotations_from_events(
                                        events, 
                                        sfreq=raw.info["sfreq"],
                                        event_desc={v: k for k, v in events_id.items()}
                                        )
    raw.set_annotations(annots)
    raw.save(fname.with_suffix(".fif.gz"))

if __name__ == "__main__":
    fname = "/Users/payamsadeghishabestari/Downloads/40324/eeg/40324_ses-1_oddball_xy.cdt"
    correct_xy_triggers(fname)