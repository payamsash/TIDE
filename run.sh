#!/bin/bash
rm -rf venv
./create_env.sh
rm -rf /tmp/data_out/
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
conda activate ./venv
cp -r data_in/ /tmp/
cp ~/Tinnitus/Data/tide/project_n/con1.eeg /tmp/data_in/con1.eeg
rm /tmp/data_in/con1/EEG/rest/con1_replace_this_with_con1_eeg

rm -f /tmp/data_out/*
cp ~/Tinnitus/Data/tide/project_n/con1.eeg /tmp/data_in/con1.eeg

python main.py
