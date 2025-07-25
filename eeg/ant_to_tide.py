import os
import shutil
from pathlib import Path
import pandas as pd
import argparse
from tqdm import tqdm
import mne
mne.set_log_level("ERROR")

def ant_to_tide():

    parser = argparse.ArgumentParser(description=("""
    ************************************
    Provide the input directory containing the .vhdr files and 
    the output directory where the downsampled .fif files should be saved.
    Example usage:
        python ant_to_tide.py /path/to/input /path/to/output
    ************************************
    """),
    formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("input_dir")
    parser.add_argument("output_dir")

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sfreq = 1000.0
    paradigms = ("omi", "gpias", "dublin", "rest", "rest_2", "xxxxx", "xxxxy")
    mapping = {
                "antinomics_id": [],
                "paradigm": [],
                "tide_id": []
                }

    files = os.listdir(input_dir)
    subjects_dict = {}
    for fname in sorted(files, key=lambda f: os.path.getctime(input_dir / f)):
        cond1 = fname.endswith(".vhdr")
        cond2 = any(sub in fname for sub in paradigms)

        if cond1 and cond2:
            file_path = input_dir / fname
            sub_id = file_path.stem[:4]
            paradigm = file_path.stem[5:]

            if not sub_id in subjects_dict:
                subjects_dict[sub_id] = []
            
            subjects_dict[sub_id].append(paradigm)

    # print(subjects_dict)
    
    
    tide_id = 70047
    for sub_id in tqdm(list(subjects_dict.keys())[tide_id-70001:]):
        pars = subjects_dict[sub_id]
        for par in pars:
            input_fname = input_dir / f"{sub_id}_{par}.vhdr"
            print(f"working on {sub_id}_{par}.vhdr ...")

            os.makedirs(output_dir / str(tide_id) / "eeg", exist_ok=True)
            os.makedirs(output_dir / str(tide_id) / "audiometry", exist_ok=True)
            fname_save = output_dir / str(tide_id) / "eeg" / f"{tide_id}_{par}.fif"

            raw = mne.io.read_raw_brainvision(input_fname, preload=False)
            if not raw.info["sfreq"] == sfreq:
                print(f"{sub_id} in {par} has sfreq of {raw.info['sfreq']} ...")
                raw.load_data()
                raw.resample(sfreq)
            raw.save(fname_save)
            del raw

            mapping["antinomics_id"].append(sub_id)
            mapping["paradigm"].append(paradigm)
            mapping["tide_id"].append(tide_id)
        
        tide_id += 1
    
    df = pd.DataFrame(mapping)
    df.to_csv(Path.cwd() / "antinomics_to_tide.csv")
    
if __name__ == "__main__":
    ant_to_tide()