#!/bin/bash
#SBATCH --job-name=context_array
#SBATCH --output=context_%A_%a.log
#SBATCH --error=context_%A_%a.err
#SBATCH --account=macs40123
#SBATCH --partition=caslake
#SBATCH --ntasks-per-node=3
#SBATCH --mem=35G
#SBATCH --time=04:00:00          # Max time for job
#SBATCH --array=0   # Array job with indices from 0 to 3 (adjust based on number of word lists)

# Define an array of target word lists, each as a comma-separated string
# ("studies,study,studying,studied,textbook,textbooks")
TARGET_WORD_LISTS=("stupid,foolish,fool,idiot,idiotic,dummy,stupidity,fools,dumb")
#("impulsive,stubborn,uncontrollable,violence,interrupt,interrupting,violent,angry,destroy,rebellious,anger,quarrel,disrupted,violently,aggressive,destructive,conflict,attack,fight,fights,fighting,attacks,aggression,bully,rebel,rebels")
# ("impulsive,stubborn,uncontrollable,violence,interrupt,interrupting,violent,angry,destroy,rebellious,anger,quarrel,disrupted,violently,aggressive,destructive,conflict,attack,fight,fights,fighting,attacks,aggression,bully,rebel,rebels")
#("attentive,carefully,polite,politely,listens,attention,politeness,meticulous,careful,listened,listen,listening,courteous,conscientious")
# ("intelligent,intellect,intelligence,smart,clever,genius,brilliant,brilliance,intellectually,intellectual,ingenious,cleverly,smarter")# Select the target words list based on the array index
TARGET_WORDS=${TARGET_WORD_LISTS[$SLURM_ARRAY_TASK_ID]}

# Load required modules or activate the environment
module load python/anaconda-2022.05
source activate myenv

# Run the Python script with the list of target words as an argument
/home/maxzhuyt/.conda/envs/myenv/bin/python context_count_multiple.py "$TARGET_WORDS" 1830 2005
