# TIDE EEG Processing Pipeline
### Identification and validation of a biomarker for tinnitus: an objective data-driven personalized approach to diagnosis of chronic tinnitus 


## Usage
- Install `virtualenv` (if not already installed), create a virtual environment and navigate to the directory where you want to create the virtual environment, activate the virtual environment and install dependencies from `requirements.txt` after activating the virtual environment:
```
# if not installed
pip install virtualenv

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
