import argparse
from eeg.eeg_feature_extraction import extract_eeg_features

def main():
    parser = argparse.ArgumentParser(description=("""
    ************************************
    Extracting features from clean epochs.
    The process is fully automatic.



    Notes
    -----
    This script is mainly designed for Antinomics / TIDE projects, however could be used for other purposes.
    """
    ),
    formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("subject_id", help="The subject name, if subject has MRI data as well, should be FreeSurfer subject name, then data from both modality can be analyzed at once.")
    parser.add_argument("subjects_dir", help="Path to the subjects_dir, will be created if its not existing")
    parser.add_argument("--config_file", default=None, help="The yaml file containing specific options.")
    parser.add_argument("--overwrite", default="warn", help="must be one of the ['ignore', 'warn', 'raise']")
    parser.add_argument("--verbose", default="ERROR", help="Control verbosity of the logging output. If None, use the default verbosity level.")

    args = parser.parse_args()
    extract_eeg_features(**vars(args))

if __name__ == "__main__":
    main()
