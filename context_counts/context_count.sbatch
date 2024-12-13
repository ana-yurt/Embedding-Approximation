#!/bin/bash
#SBATCH --job-name=context_array
#SBATCH --output=context_%A_%a.log
#SBATCH --error=context_%A_%a.err
#SBATCH --account=macs40123
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=1
#SBATCH --mem=25G
#SBATCH --time=03:00:00          # Max time for job
#SBATCH --array=0-3   # Array job with indices from 0 to 4 (adjust based on number of words)

# Define an array of target words
TARGET_WORDS=("stupid" "idiot" "foolish" "stupidity")

# Select the target word based on the array index
TARGET=${TARGET_WORDS[$SLURM_ARRAY_TASK_ID]}

# Load required modules or activate the environment
module load python/anaconda-2022.05
source activate myenv

# Run the Python script with the target word as an argument
/home/maxzhuyt/.conda/envs/myenv/bin/python context_count.py "$TARGET" 1830 2005
