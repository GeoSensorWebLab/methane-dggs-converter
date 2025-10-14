#!/bin/bash
#SBATCH --job-name=mexico_netcdf_conversion
#SBATCH --output=log/mexico_netcdf_conversion_%j.out
#SBATCH --error=log/mexico_netcdf_conversion_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=5:00:0
#SBATCH --mem=16G
#SBATCH --partition=cpu2019
#SBATCH --mail-user=mingke.li@ucalgary.ca
#SBATCH --mail-type=END,FAIL

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
export OMP_NUM_THREADS=1  # Prevent OpenMP from using all cores
export NUM_CORES=24       # Match the number of CPUs requested

# Create log directory
mkdir -p log

echo "Starting Mexico NetCDF to DGGS conversion process..."
echo "Using $NUM_CORES CPU cores for parallel processing"

# Run Mexico NetCDF to DGGS conversion
echo "=========================================="
echo "Running Mexico NetCDF to DGGS conversion"
echo "=========================================="
echo "Start time: $(date)"
$PYTHON_PATH scripts/netcdf_conversion/mexico_netcdf_to_dggs_converter.py
EXIT_CODE=$?
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Mexico NetCDF conversion completed successfully"
else
    echo "Mexico NetCDF conversion failed with exit code $EXIT_CODE"
    exit 1
fi

echo ""
echo "Mexico NetCDF to DGGS conversion completed successfully!"
echo "Job finished at:" $(date)
