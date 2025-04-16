# TIDE EEG Processing Pipeline
### Identification and validation of a biomarker for tinnitus: an objective data-driven personalized approach to diagnosis of chronic tinnitus 

## Abstract
Tinnitus diagnosis remains fundamentally reliant on subjective self-reports, as an unequivocal
objective biomarker has yet to be identified. While numerous studies have revealed measurable
neural activity differences between individuals with and without tinnitus, translating these
group-level findings into a reliable single-subject diagnostic tool has proven elusive. Among
key paradigms, resting-state EEG, the auditory oddball paradigm, and the Gap Prepulse
Inhibition of the Acoustic Startle paradigm have consistently shown promise in detecting
group-level discrepancies associated with tinnitus.

The TIDE consortium has embarked to identify and validate robust biomarkers for tinnitus
perception. We aim to develop an objective diagnostic biomarker capable of reliably detecting
tinnitus at the individual level, marking a transformative step in the field.
The consortium currently records EEG activity from 560 participants (280 tinnitus, 280
controls) to reach this goal. Neural activity is recorded using electroencephalography across
three   conditions:   resting   state,   auditory   oddball   paradigm,   and   GPIAS   paradigm.
Complementing these recordings are comprehensive behavioural and clinical assessments,
including:

- **Questionnaires**: European School for Interdisciplinary Tinnitus Research Screening
Questionnaire, Tinnitus Handicap Inventory, Tinnitus Functional Index, Hospital
Anxiety   and   Depression   Scale,   Perceived   Stress   Questionnaire,   Hyperacusis
Questionnaire.
- **Auditory Evaluations**:   Pure-tone   audiometry,   tinnitus   pitch   matching,   tinnitus
loudness matching.

**Data analysis** employs a rigorous, blinded protocol involving inferential statistics, logistic
regression, and cutting-edge machine learning techniques. To develop and validate classifiers
capable of identifying tinnitus perception, an 80%-20% training-to-test dataset split is used.
The study’s commitment to scientific integrity and transparency is underscored by its pledge to
openly publish anonymized data, analysis scripts, and stimulation paradigm protocols.
This multicenter study is being conducted across sites in Austin-US, Dublin-Ireland, Ghent
-Belgium, Illinois-US, Regensburg-Germany, Tübingen-Germany, and Zürich/St. Gallen
-Switzerland.
The TIDE consortium aims to develop objective, reliable, and clinically applicable biomarkers for tinnitus by combining large-scale data collection with sophisticated analytical 
techniques.

*Author: Winfried Schlee*

## Usage
- Install `virtualenv` (if not already installed), create a virtual environment and navigate to the directory where you want to create the virtual environment, activate the virtual environment and install dependencies from `requirements.txt` after activating the virtual environment:
```
# go to directory
cd .../Antinomics/eeg

# create myenv
python -m venv myenv

# On Windows, use:
myenv\Scripts\activate

# On Mac/Linux, use:
source myenv/bin/activate

# install the packages
pip install -r requirements.txt
```

- Prepare your eeg recordings in the following format:
    -    First create a folder e.g *"subjects_dir"* which will include subfolders with your *subject IDs* e.g `subjects_dir / <subject_id>` each subfolder must have at least a folder named *EEG*. 
    -    The *EEG* subfolder must have paradigm folders inside as well as *reports* folder. The paradigm names must be one of the followings: *"rest", "omi", "gpias", "regularity", "xxxxy", "xxxxx".*
    -    Each paradigm folder must have raw eeg file(s) inside, the name of the eeg files must be in the following format: `<subject_id>_<paradigm>` e.g for the brainvision devices and the `bert` subject you should have the following files inside the `rest` folder: `bert_rest.vmrk`, `bert_rest.vhdr` and `bert_rest.eeg`.
    -    In case you have multiple eeg files, put them with numbering in the correct *paradigm* folder e.g: `bert_gpias_1.bdf`, `bert_gpias_2.bdf`, ...

- Open the `main.ipynb` and run the `preprocessing` cell with your selected parameters. Later you can run `run_erp_analysis` or `run_rs_analysis` based on your needs.

---

## Contact
The ideal place to ask questions or discuss the contribution process is the [issue tracker](https://github.com/payamsash/Antinomics/issues) on our GitHub repository. If you encounter a bug or have an idea for a new feature, please report it using the issue tracker. To get the most effective support, you can share your data via a Google Drive link, along with a screenshot of your code and the error message you received when running it.
