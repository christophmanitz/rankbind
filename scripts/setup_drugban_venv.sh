#!/bin/bash
# scripts/setup_drugban_venv.sh — Create separate venv for DrugBAN (DGL + torch 2.4)
#
# DGL 2.4.x requires torch 2.4.x and cannot coexist with torch 2.8.
# This creates a dedicated venv with all DrugBAN dependencies.
#
# Run on login node (not compute node):
#   bash scripts/setup_drugban_venv.sh

set -e

VENV_DIR="$HOME/venvs/drugban"

if [ -d "$VENV_DIR" ]; then
    echo "Venv already exists: $VENV_DIR"
    echo "Delete it first if you want to recreate: rm -rf $VENV_DIR"
    exit 1
fi

module purge
module load GCC/11.3.0
module load CUDA/12.4.0
module load Python/3.10.4-GCCcore-11.3.0

echo "Creating DrugBAN venv at: $VENV_DIR"
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# Install torch 2.4.x (compatible with DGL 2.4.x)
pip install --upgrade pip
pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124

# Install DGL and DGLLife
pip install dgl==2.4.0 -f https://data.dgl.ai/wheels/torch-2.4/cu124/repo.html
pip install dgllife

# Other dependencies for DrugBAN
# NOTE: numpy<2 required — rdkit-pypi 2022.x is compiled against NumPy 1.x
pip install 'numpy<2' pandas scikit-learn rdkit-pypi yacs

echo ""
echo "DrugBAN venv ready: $VENV_DIR"
echo "Activate with: source $VENV_DIR/bin/activate"
echo "Test with: python -c 'import dgl; import dgllife; print(\"OK\")'"
