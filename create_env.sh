mamba env create  -f tide_detailed.yml -p ./venv -y #if this fails, try with -f tide.yml
#conda env create -p ./venv
conda activate ./venv
mamba env update -p ./venv #does that really work?
