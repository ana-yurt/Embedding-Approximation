import os
import sys
import re
import pandas as pd
from collections import Counter
from tqdm import tqdm
import logging

def process_year(targets, year, directory, window_size=10, top_n=2000, output_dir="/home/maxzhuyt/Desktop/coha/context_counts"):
    year_pattern = re.compile(rf'.*_{year}_\d+\.txt')
    context_words = []

    # Loop over files for the given year
    for filename in os.listdir(directory):
        if filename.startswith('._'):
            continue
        if year_pattern.match(filename):
            logging.info(f"Processing {filename}...")
            file_path = os.path.join(directory, filename)
            with open(file_path, 'r', encoding='utf-8') as file:
                words = file.read().split()

            # Find all contexts for any target word within the specified window
            for i, word in enumerate(words):
                if word in targets:
                    start = max(i - window_size, 0)
                    end = min(i + window_size + 1, len(words))
                    context_words.extend(words[start:i] + words[i+1:end])
    
    # Count context words and normalize by total context length
    logging.info(f"Processed {year} with {len(context_words)} context words")
    context_counts = Counter(context_words)
    total_contexts = sum(context_counts.values())
    normalized_counts = {word: count / total_contexts for word, count in context_counts.items()}
    
    # Keep only the top N context words
    top_contexts = dict(Counter(normalized_counts).most_common(top_n))
    
    # Convert to DataFrame with the year as the index
    year_df = pd.DataFrame(top_contexts, index=[year])

    # Save the DataFrame for this year to a CSV file
    output_file = os.path.join(output_dir, f"context_counts_{'_'.join(targets)}_{year}.csv")
    year_df.to_csv(output_file)
    logging.info(f"Saved context counts for {year} to {output_file}")

def count_context_words_sequential(targets, start_year, end_year, directory, window_size=10, top_n=2000, output_dir="/home/maxzhuyt/Desktop/coha/context_counts"):
    # Process each year sequentially, saving each result immediately
    for year in range(start_year, end_year + 1):
        process_year(targets, year, directory, window_size, top_n, output_dir)


def combine_yearly_csvs(targets, start_year, end_year, output_dir, combined_output_file):
    # Initialize an empty DataFrame to store all years' data
    combined_df = pd.DataFrame()
    
    # Loop through each year's file and append it to the combined DataFrame
    for year in range(start_year, end_year + 1):
        yearly_file = os.path.join(output_dir, f"context_counts_{'_'.join(targets)}_{year}.csv")
        if os.path.exists(yearly_file):
            year_df = pd.read_csv(yearly_file, index_col=0)
            combined_df = pd.concat([combined_df, year_df], axis=0)
        else:
            print(f"Warning: File for {year} not found in {output_dir}")

    # Save the combined DataFrame as a single CSV file
    combined_df.index.name = 'Year'
    combined_df.to_csv(combined_output_file)
    print(f"Combined CSV saved to {combined_output_file}")



if __name__ == "__main__":
    # Arguments: comma-separated target words, start year, end year
    if len(sys.argv) != 4:
        print("Usage: python count_context_words.py <target_words_comma_separated> <start_year> <end_year>")
        sys.exit(1)

    target_words = sys.argv[1].split(",")
    start_year = int(sys.argv[2])
    end_year = int(sys.argv[3])
    directory = "/project/macs40123/maxzhuyt/tokenized_coha"
    output_dir = f"/scratch/midway3/maxzhuyt/context_counts/{'_'.join(target_words)}"
    combined_output_file = f"/scratch/midway3/maxzhuyt/context_counts/{'_'.join(target_words[:3])}_combined_{start_year}_{end_year}.csv"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Counting context words for '{', '.join(target_words)}' from {start_year} to {end_year}...")

    # Generate and save the normalized counts for each year sequentially
    count_context_words_sequential(target_words, start_year, end_year, directory, output_dir=output_dir)
    combine_yearly_csvs(target_words, start_year, end_year, output_dir, combined_output_file)

    print(f"Context word counts saved to {output_dir}")
