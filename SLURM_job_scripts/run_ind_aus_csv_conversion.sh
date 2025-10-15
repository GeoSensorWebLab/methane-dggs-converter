#!/bin/bash
#SBATCH --job-name=ind_aus_csv_conversion
#SBATCH --output=log/ind_aus_csv_conversion_%j.out
#SBATCH --error=log/ind_aus_csv_conversion_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2:00:0
#SBATCH --mem=12G
#SBATCH --partition=cpu2019

# Set script directory
SCRIPTDIR=/home/mingke.li/methane_grid_calculation_ARC
cd $SCRIPTDIR || { echo "Directory $SCRIPTDIR not found"; exit 1; }

echo "Job starting at:" $(date)

# Load conda environment
export PATH=/home/mingke.li/miniconda3/bin:$PATH
source /home/mingke.li/miniconda3/etc/profile.d/conda.sh
conda activate netcdf_dggs_converter

# Set Python path and environment variables
export PYTHON_PATH="/home/mingke.li/miniconda3/envs/netcdf_dggs_converter/bin/python"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_CORES=8

# Create log directory
mkdir -p log

echo "Starting India/Australia CSV to DGGS conversion process..."
echo "Using $NUM_CORES CPU cores for processing"

# Run India/Australia CSV to DGGS conversion
echo "=========================================="
echo "Running India/Australia CSV to DGGS conversion"
echo "=========================================="
echo "Start time: $(date)"
$PYTHON_PATH scripts/csv_conversion/ind_aus_csv_to_dggs_converter.py
EXIT_CODE=$?
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "India/Australia CSV conversion completed successfully"
else
    echo "India/Australia CSV conversion failed with exit code $EXIT_CODE"
    exit 1
fi

echo ""
echo "India/Australia CSV to DGGS conversion completed successfully!"
echo "Job finished at:" $(date)


