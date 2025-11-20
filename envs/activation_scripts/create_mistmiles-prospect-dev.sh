#!/usr/bin/env bash
cd /Users/Jonah/PhD/Research/quiescent_galaxies/envs

# Path variables
BASE_DIR="/Users/Jonah/PhD/Research/quiescent_galaxies/envs"
YAML="${BASE_DIR}/environment-nofsps-prospect-dev.yml"
PROSPECTOR_LOCAL="/Users/Jonah/PhD/Research/quiescent_galaxies/dev/prospector_v2"
ENV_NAME="mistmiles-prospect-dev"

# Create environment shell (no fsps or prospector)
mamba env create -n "$ENV_NAME" -f "$YAML"

# Activate environment
eval "$(mamba shell hook --shell bash)"
mamba activate "$ENV_NAME"

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"

# Install FSPS (stable version)
python -m pip install fsps

# Install prospector from local repo
echo "Installing local prospector (editable): ${PROSPECTOR_LOCAL}"
python -m pip install -e "${PROSPECTOR_LOCAL}"