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


## Project Structure
```plaintext
├── config                      # Config files to customize 
├── eeg
│   ├── eeg_preprocessing.py           
│   ├── eeg_processing.py
│   ├── eeg_features_extraction.py  
├── src
│   ├── preprocess.py           
│   ├── process.py
│   ├── extract_features.py     
├── venv                        # Virtual environment for the project
```

## Installation and Basic Usage
1. Clone the repo:
    ```
    git clone https://github.com/payamsash/TIDE.git
    ```
### with venv
2. Set up a virtual environment (optional but recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .\env\Scripts\activate
    ```

3. Navigate to the project directory and install dependencies:
    ```bash
    cd TIDE
    pip install -r requirements.txt
    ```

### with Conda
3. Create a Conda environment:
   ```bash
   conda env create -f environment.yml
    ```
4. Activate Conda
    ```bash
    conda activate tide_venv
    ```
5. Run the main script to start processing signals and rendering visuals:
    ```bash
    python src/preprocess.py --help
    ```
---

## Usage
**Preprocessing**

The default preprocessing options for each site are specified in the `preprocessing-config.yaml` file. Before proceeding, it's recommended to inspect the Power Spectral Density (PSD) plot to identify any channels that exhibit unusual patterns compared to others. You can view channel names by clicking on the plot. Take note of any suspicious channels and examine them further using the scrollable plot view (`manual_scroll_data=True`). If you decide a channel is bad, simply click on its signal—it will be marked as bad and automatically interpolated during preprocessing. The script will create a folder for each subject within the subjects_dir directory and save the following files: the original raw data (in `fif` format), the preprocessed data, logs, and reports.

For GPIAS paradigm:

-   EEG files must be ordered as follows: `["pre", "bbn", "3kHz", "8kHz", "post"]`.
-   Each file should end with one of these suffixes: `["1", "_1", "-1"]`.
For example, you should rename the files as: `gpias_1.bdf`, `gpias_2.bdf`, `...`, `gpias_5.bdf`.

-   A plot displaying five subplots—each corresponding to one of the five blocks—will appear. You can manually adjust the vertical dashed lines to separate the histograms. Once satisfied, close the window. It’s best to position the lines near the leftmost bars (indicating lower values).

For resting-state paradigm:

-   Consider using more descriptive names for the paradigm, such as rest_closed, rest_open, or rest_both, to reflect whether the subject had their eyes closed, open, or alternated between the two states during the recording.

**Processing**

The default processing options for each site are defined in the `processing-config.yaml` file. Before running the pipeline, it is recommended to review the epochs plot (`manual_scroll_data=True`) to identify any channels or epochs that show abnormal patterns. You can simply click on an epoch or a channel name to mark it for exclusion. Additionally, three automatic epoch rejection algorithms are available based on user preference: `"ptp", "pyriemann", and "autoreject"`. By default, no automatic rejection is applied unless explicitly specified. If you have a structural MRI scan of the subject, you should provide the corresponding `FreeSurfer` subject directory via the `subjects_fs_dir` argument. After processing, the extracted epochs will be saved in the subject's epochs directory, along with logs and HTML-formatted reports.

**Feature extraction**

You can run this script on your resting-state epochs and select the specific features you wish to extract from the data. To view the full list of available sensor-level features, run the following:
```
from eeg_feature_extraction import get_full_sensor_features_list
get_full_sensor_features_list()
```
You can toggle options (True/False) to specify whether to extract source label power, sensor connectivity, or source connectivity features. Additionally, you can choose the method used for connectivity estimation (see `features-config.yaml`).

## Contact
The ideal place to ask questions or discuss the contribution process is the [issue tracker](https://github.com/payamsash/Antinomics/issues) on our GitHub repository. If you encounter a bug or have an idea for a new feature, please report it using the issue tracker. To get the most effective support, you can share your data via a Google Drive link, along with a screenshot of your code and the error message you received when running it.
