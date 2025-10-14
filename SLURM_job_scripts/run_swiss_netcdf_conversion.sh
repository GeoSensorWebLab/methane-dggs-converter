#!/bin/bash
#SBATCH --job-name=swiss_netcdf_conversion
#SBATCH --output=log/swiss_netcdf_conversion_%j.out
#SBATCH --error=log/swiss_netcdf_conversion_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --time=5:00:0
#SBATCH --mem=24G
#SBATCH --partition=cpu2021
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
export NUM_CORES=12       # Match the number of CPUs requested

# Create log directory
mkdir -p log

echo "Starting Swiss NetCDF to DGGS conversion process..."
echo "Using $NUM_CORES CPU cores for parallel processing"

# Run Swiss NetCDF to DGGS conversion
echo "=========================================="
echo "Running Swiss NetCDF to DGGS conversion"
echo "=========================================="
echo "Start time: $(date)"
$PYTHON_PATH scripts/netcdf_conversion/swiss_netcdf_to_dggs_converter.py
EXIT_CODE=$?
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "Swiss NetCDF conversion completed successfully"
else
    echo "Swiss NetCDF conversion failed with exit code $EXIT_CODE"
    exit 1
fi

echo ""
echo "Swiss NetCDF to DGGS conversion completed successfully!"
echo "Job finished at:" $(date)


