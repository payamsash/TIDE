#!/bin/sh
rm -rf venv
./create_env.sh
rm -rf /tmp/data_out/
conda init bash
conda activate ./venv
python main.py
