#!/bin/bash
#SBATCH --job-name=china_sacms_netcdf_conversion
#SBATCH --output=log/china_sacms_netcdf_conversion_%j.out
#SBATCH --error=log/china_sacms_netcdf_conversion_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --time=2:00:00
#SBATCH --mem=24G
#SBATCH --partition=cpu2021

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

echo "Starting China SACMS NetCDF to DGGS conversion process..."
echo "Using $NUM_CORES CPU cores for parallel processing"

# Run China SACMS NetCDF to DGGS conversion
echo "=========================================="
echo "Running China SACMS NetCDF to DGGS conversion"
echo "=========================================="
echo "Start time: $(date)"
$PYTHON_PATH scripts/netcdf_conversion/china_SACMS_netcdf_to_dggs_converter.py
EXIT_CODE=$?
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "China SACMS NetCDF conversion completed successfully"
else
    echo "China SACMS NetCDF conversion failed with exit code $EXIT_CODE"
    exit 1
fi

echo ""
echo "China SACMS NetCDF to DGGS conversion completed successfully!"
echo "Job finished at:" $(date)
