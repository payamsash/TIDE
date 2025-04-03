#!/bin/bash

set -e  # Exit immediately if a command fails
set -o pipefail  # Catch errors in piped commands

ENV_DIR="./venv"
ENV_FILE="environment.yml"

error_exit() {
    echo -e "\e[31mError: $1\e[0m" >&2
    exit 1
}

# Ensure mamba is available
if ! command -v mamba &>/dev/null; then
    error_exit "Mamba is not installed or not in PATH. Please install it first."
fi

# Initialize mamba if needed
if ! conda info --envs &>/dev/null; then
    eval "$(mamba shell.bash hook)" || error_exit "Failed to initialize Mamba."
    export PATH="$HOME/miniconda3/bin:$PATH"  # Adjust path if needed
fi

# Remove existing environment if it exists
if [ -d "$ENV_DIR" ]; then
    echo "Removing existing environment..."
    rm -rf "$ENV_DIR" || error_exit "Failed to remove existing environment."
fi

# Create the environment
echo "Creating the virtual environment from $ENV_FILE..."
mamba env create -p "$ENV_DIR" -f "$ENV_FILE" -y || error_exit "Failed to create environment from $ENV_FILE."


