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
    parser.add_argument("--config_file", help="The yaml file containing specific options.")
    parser.add_argument("--sensor_space_features", default=['pow_freq_bands'], help="List of the features you want to extract per epoch, you can see the full list of features name by running get_full_features_list()")
    parser.add_argument("--source_space_power", default=True, help="If True, power in brain labels will be computed.")
    parser.add_argument("--sensor_space_connectivity", default=True, help="If True, connectivity features between EEG channels will be computed.")
    parser.add_argument("--source_space_connectivity", default=False, help="If True, connectivity features between brain labels will be computed.")
    parser.add_argument("--connectivity_method", default="coh", help="method to compute connectivity, possible options are: ['coh', 'cohy', 'imcoh', 'cacoh', 'mic', 'mim', 'plv', 'ciplv', 'ppc', 'pli', 'dpli', 'wpli', 'wpli2_debiased', 'gc', 'gc_tr']")
    parser.add_argument("--subjects_fs_dir", default=None, help="The path to the directory containing the FreeSurfer subjects reconstructions.")
    parser.add_argument("--atlas", default="aparc", help="The parcellation to use, e.g., 'aparc' or 'aparc.a2009s'.")
    parser.add_argument("--freq_bands", default={"delta": [1, 4],
                                                "theta": [4, 8],
                                                "alpha": [8, 13],
                                                "beta": [13, 30],
                                                "lower_gamma": [30, 45],
                                                "upper_gamma": [45, 80]
                                                }, help="Path to the subjects_dir, will be created if its not existing")
    parser.add_argument("--overwrite", default="warn", help="must be one of the ['ignore', 'warn', 'raise']")
    parser.add_argument("--verbose", default="ERROR", help="Control verbosity of the logging output. If None, use the default verbosity level.")

    args = parser.parse_args()
    extract_eeg_features(**vars(args))

if __name__ == "__main__":
    main()
