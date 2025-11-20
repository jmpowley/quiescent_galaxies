#!/usr/bin/env bash
cd /Users/Jonah/PhD/Research/quiescent_galaxies/envs

# Path variables (edit if required)
BASE_DIR="/Users/Jonah/PhD/Research/quiescent_galaxies/envs"
YAML="${BASE_DIR}/environment-nofsps-prospect-dev.yml"
PROSPECTOR_LOCAL="/Users/Jonah/PhD/Research/quiescent_galaxies/dev/prospector_v2"
ENV_NAME="mistc3k-prospect-dev"

# Create environment shell (no fsps or prospector)
mamba env create -n "$ENV_NAME" -f "$YAML"

# Activate environment
eval "$(mamba shell hook --shell bash)"
mamba activate mistc3k-prospect-dev

echo "Active environment: $CONDA_DEFAULT_ENV"
echo "Python location: $(which python)"

# Install FSPS with flags
export FFLAGS="-DMIST=1 -DMILES=0 -DC3K=1"
echo "Installing fsps from source with FFLAGS=$FFLAGS ..."
python -m pip install --no-binary fsps --no-build-isolation fsps -v
unset FFLAGS

# Install prospector from local repo
echo "Installing local prospector (editable): ${PROSPECTOR_LOCAL}"
python -m pip install -e "${PROSPECTOR_LOCAL}"