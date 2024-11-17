#!/bin/bash
#SBATCH --job-name=alacarte_words
#SBATCH --output=alacarte_%A_%a.log
#SBATCH --error=alacarte_%A_%a.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --account=macs40123
#SBATCH --partition=caslake
#SBATCH --mem=15G
#SBATCH --array=0-70
#SBATCH --time=09:00:00 

# Move into the directory you submitted from
cd $SLURM_SUBMIT_DIR

# Load Miniconda
source activate myenv

# Define an array to map SLURM_ARRAY_TASK_ID to the specific years
START_YEAR=1930
YEAR=$((START_YEAR + SLURM_ARRAY_TASK_ID))
SAMPLES=20
MAX_THREADS=10
ANCHOR_WORDS_POSITIVE=("she" "her" "woman" "women" "mother" "daughter" "sister" "female")
ANCHOR_WORDS_NEGATIVE=("he" "his" "him" "man" "men" "father" "son" "brother" "male")

# Run the Python script with the specified arguments
python alacarte_words.py \
    --year $YEAR \
    --samples $SAMPLES \
    --max_threads $MAX_THREADS \
    --target_word "idiot" \
    --anchor_words_positive "${ANCHOR_WORDS_POSITIVE[@]}" \
    --anchor_words_negative "${ANCHOR_WORDS_NEGATIVE[@]}"