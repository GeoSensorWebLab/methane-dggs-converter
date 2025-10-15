#!/bin/bash
#SBATCH --job-name=china_tiff_conversion
#SBATCH --output=log/china_tiff_conversion_%j.out
#SBATCH --error=log/china_tiff_conversion_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --time=10:00:0
#SBATCH --mem=30G
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
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NUM_CORES=16

# Create log directory
mkdir -p log

echo "Starting China GeoTIFF to DGGS conversion process..."
echo "Using $NUM_CORES CPU cores for parallel processing"

# Run China GeoTIFF to DGGS conversion
echo "=========================================="
echo "Running China GeoTIFF to DGGS conversion"
echo "=========================================="
echo "Start time: $(date)"
$PYTHON_PATH scripts/geotiff_conversion/china_tiff_to_dggs_converter.py
EXIT_CODE=$?
echo "End time: $(date)"

if [ $EXIT_CODE -eq 0 ]; then
    echo "China GeoTIFF conversion completed successfully"
else
    echo "China GeoTIFF conversion failed with exit code $EXIT_CODE"
    exit 1
fi

echo ""
echo "China GeoTIFF to DGGS conversion completed successfully!"
echo "Job finished at:" $(date)


