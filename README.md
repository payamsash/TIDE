# Python

Python established itself as **the** open source language for scientific computing and can be used for all of our purposes. There are still some advantages of using R/studio and advanced packages for statistics and plotting. Using open source languages (like Python and R) is not only helpful given its thriving development community and good documentation (always think of AI code assistants), but also mandatory for output code in many research projects (e.g. enforced by SNF rules). Please note that we will use the terms library/package interchangeably, as they refer to the same thing but are used differently in the various programming languages. 

In this section we'll present 
- how to set up python library management with conda/mamba
- create a conda/mamba Python virtual environment


## Conda/Mamba

For coding and/or data analysis with Python, additional packages are usually needed. Often, these packages depend on further packages. This can escalate quickly and with dependencies several hundred packages would be needed for your code. Furthermore, Python and/or library versions, so the time dimension, also has to be considered. Libraries might change, become incompatible with Python and/or other libraries etc.. Here, conda/mamba comes in handy to manage your packages, so dealing with dependencies of your code, and helping with versions by creating fixed environments pertaining to one or several code projects. A conda environment is essentially a directory that houses a specific version of Python and all the packages your project requires. Please note that both the conda and mamba are used/presented in the following, as they complement each other for our purposes (always run your commands with the either conda or mamba depending on the step, exactly as indicated in the instructions below).

Install conda with your package/app manager or manually:

https://conda.io/projects/conda/en/latest/user-guide/install/index.html

Theoretically, Python packages/libraries can be installed using pip. It is strongly discouraged and will not be supported in this knowledge base.

![Loser](https://imgflip.com/i/8jmfe2)

Conda can be slow or even unresponsive. Therefore, it needs to be paired with mamba (a good opportunity to test your conda installation):

    conda install -c conda-forge mamba

## Virtual environment, venv


A Python environment is a directory hosting its own Python interpreter along with Python packages. This setup is distinct and operates independently from the system's default Python interpreter and its packages. Moreover, each environment is isolated, ensuring no cross-interference between them.

Now it's time to build your first venv with conda/mamba. Conda/mamba creates virtual environments for scientific computing, which is useful as the global venv module https://docs.python.org/3/library/venv.html does simply not contain/support (all) scientific Python packages. 
Given that your venv might be used for different (code) projects it's advisable to dedicate a folder on your local machine to save your venvs. Another approach is to create a venv per project. This is certainly best practice when it comes to finalizing, publishing, and archiving a project/analysis of yours (e.g. in containers, see cluster computing section). Try to navigate yourself through this options and to identify what fits best to your current needs. Note that many (scientific) Python libraries may need specific (not most current) versions of Python itself or other libraries. which favours (many) venvs. I personally cultivate a dual strategy where I maintain global venvs for my main workflows (e.g. one for Python MNE M/EEG analysis, one for MNE-LSL real-time development, one for R statistics and plotting, one for Psychopy experimental paradigms etc.) and specific venvs on the project level (e.g. for experimental study projects which will be published).

A created venv can then be easily selected or defaulted in VS code. Note that it's not good to git your venvs except the `environment.yml`, which will help (others) build their local venvs. So create your venv in a folder on your local machine and/or add the .venv folder to `.gitignore` (`.` as a prefix for folders indicate hidden folders.`Environment.yml `and `.gitignore` will be explained in the following).

### Add configuration files

In order to have your individual git repositories and your venvs under control, it's advisable to set up respective (standard) files.

First, create a `.gitignore` files using VS code, by hand or by putting any file to `.gitignore` via right-click. This file will define what git should not track and thus not push/pull to/from local and/or remote repositories. This is critical especially with regards to a possible venv residing in your repository (see above). Always make sure that no venvs or large files are tracked by git (e.g. M/EEG raw data). The size limit of a Gitlab repo is usually about 200 MB. See a template `.gitignore` below with some common ignored files and folders. Please refer to the git documentation for more information on `.gitignore` formatting https://git-scm.com/docs/gitignore. `*` stands for "wildcard" as introduced before. **Always make sure that no venvs or large files are tracked by Git**, as it bloats the repository and makes it impossible to further work with the repository. This is why ".venv" always **must** be part of the `.gitignore` list.


    .DS_Store   
    .venv
    *.fif
    *.eeg
    tmp_data
    data

Now it's time to (finally) create your first environment. For this, you need to create an `environment.yml` file. It follows the following formatting standard and is here cultivated with the most critical libraries for M/EEG analysis:

    name: <nameofyourenv>
    channels:
    - conda-forge
    dependencies:
    - mne
    - pandas
    - ipython
    - ipykernel
    - pip:
        - pygame

1) Channels refers to the source of libraries, here conda-forge, a community-led collection of recipes, build infrastructure, and distributions for the conda package manager.

2) Dependencies then list the actual needed libraries

3) Under pip libraries are listed which can not be provided by conda, but will be still integrated as part of the venv. Note that this is the only exception from the rule not to use pip

It may be that your setup/IDE has issues with "ipykernel" and needs to installed it separately by following the instructions in the IDE.

You can now commit and push the changes in your git repository, if needed.

### Create and activate the environment

To create your venv after defining its content in the `environment.yml`, type in the (VS code) terminal the following

    mamba env create -p ./.venv

The activation is then performed with the following command

    conda activate ./.venv

 Typing `which python` should now return a path of your created `.venv` folder.

What are all these snakes, especially conda and mamba? Well, unfortunately, it's a bit messy but following the guidelines here should help you making the right choices. Make sure you use the appropriate mamba and conda commands. In any case, you should never do conda/mamba install commands and always use the mamba update function to install or update libraries in your venvs. Simply add the libraries to the `environment.yml`files and update the venv:

    mamba env update -p ./.venv

Finally, if needed, venvs can be removed by simply deleting its folder

    rm -r .venv

where `-r` is the flag for recursive deletion (deleting all the folders and subfolders).
