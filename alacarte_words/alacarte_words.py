import os
import numpy as np
import pandas as pd
import re
import logging
from gensim.models import KeyedVectors
from scipy.spatial.distance import cosine
from sklearn.linear_model import LinearRegression
from concurrent.futures import ThreadPoolExecutor, as_completed
from random import sample
import argparse

# Set up logging
logging.basicConfig(filename="alacarte_words.log", level=logging.INFO)
logger = logging.getLogger(__name__)

# Function to extract contexts for a given word from the text
def extract_contexts(word, text, max_contexts=20, window_size=10):
    # Split text into lines/sentences
    sentences = text.splitlines()
    contexts = []

    # Process each sentence
    for sentence in sentences:
        # Tokenize the sentence
        words = sentence.split()
        # Find indices of the target word in the sentence
        word_indices = [i for i, w in enumerate(words) if w.lower() == word.lower()]

        # Extract context for each occurrence of the target word
        for index in word_indices:
            # Define context window, ensuring it doesn’t exceed sentence bounds
            start = max(0, index - window_size)
            end = min(len(words), index + window_size + 1)
            context_words = words[start:end]
            contexts.append(context_words)
            # Stop if we've reached the maximum number of contexts
            if len(contexts) >= max_contexts:
                return contexts
    return contexts

def get_year_files(data_path, year):
    year_files = []
    for file in os.listdir(data_path):
        match = re.search(r'\d{4}', file)
        if match:
            file_year = int(match.group())
            if abs(file_year - year) <= 2:
                year_files.append(file)
    return year_files

# Function to calculate the cosine similarity between a word and the direction vector
def calculate_similarity(word_vector, direction_vector):
    return 1 - cosine(word_vector, direction_vector)

# Function to calculate z-scored similarity
def calculate_z_scored_similarity(model, word_vector, direction_vector, sample_size=5000):
    all_vocab = list(model.key_to_index.keys())
    sample_vocab = np.random.choice(all_vocab, sample_size, replace=False)
    similarities = [calculate_similarity(model[w], direction_vector) for w in sample_vocab]
    word_similarity = calculate_similarity(word_vector, direction_vector)
    word_z_score = (word_similarity - np.mean(similarities)) / np.std(similarities)
    return word_z_score

# Evaluate the quality of transformation matrix
def evaluate_transformation_quality(A, X, Y, model, common_words):
    recovered_vectors = np.dot(X, A.T)
    original_vectors = np.array(Y)
    similarities = [1 - cosine(recovered_vec, original_vec) for recovered_vec, original_vec in zip(recovered_vectors, original_vectors)]
    avg_similarity = np.mean(similarities)
    logger.info(f"Average cosine similarity between recovered and original vectors: {avg_similarity:.4f}")

# Learn transformation matrix
def learn_transformation_matrix(model, common_words, data_path, year, target_word):
    X, Y = [], []
    year_files = get_year_files(data_path, year)
    all_texts = [open(os.path.join(data_path, file), 'r', encoding='utf-8', errors='ignore').read() for file in year_files]
    for word in common_words:
        if word not in model:
            continue
        context_vectors = []
        for text in all_texts:
            contexts = extract_contexts(word, text, max_contexts=3)
            context_vectors.extend([model[w] for context in contexts for w in context if w in model])
        if context_vectors:
            final_average_context_vector = np.mean(context_vectors, axis=0)
            X.append(final_average_context_vector)
            Y.append(model[word])
    logger.info(f"Found {len(X)} valid data points for common words in the regression.")

    target_context_vectors = []
    for text in all_texts:
        contexts = extract_contexts(target_word, text, max_contexts=10)
        target_context_vectors.extend([model[w] for context in contexts for w in context if w in model])
    
    sample_num = len(target_context_vectors) // 5 
    for _ in range(30):
        sampled_context_vectors = np.mean(np.array(sample(target_context_vectors, sample_num)), axis=0)
        X.append(sampled_context_vectors)
        Y.append(model[target_word])

    X, Y = np.array(X), np.array(Y)
    if len(X) > 0 and len(Y) > 0:
        lr = LinearRegression(fit_intercept=False)
        lr.fit(X, Y)
        A = lr.coef_
        evaluate_transformation_quality(A, X, Y, model, common_words + [target_word])
        return A
    else:
        logger.warning("No valid data points for regression.")
        return None

# Process each sample
def process_sample_id(sample_id, year, embedding_year, data_path, word_frequencies, target_word, anchor_words_positive, anchor_words_negative):
    try:
        model_file = f'/project/macs40123/maxzhuyt/embeddings_5/word_embeddings_{sample_id}/word_vectors_{embedding_year}.kv'
        model = KeyedVectors.load(model_file, mmap='r')
        A = learn_transformation_matrix(model, list(word_frequencies.keys()), data_path, year, target_word)
        male_vector = np.mean([model[word] for word in anchor_words_negative if word in model], axis=0)
        female_vector = np.mean([model[word] for word in anchor_words_positive if word in model], axis=0)
        gender_direction_vector = female_vector - male_vector

        # Collect data for CSV output
        data = []
        for word, freq in word_frequencies.items():
            if word in model:
                original_loading = calculate_z_scored_similarity(model, model[word], gender_direction_vector)
                transformed_vector = np.dot(A, model[word]) if A is not None else model[word]
                transformed_loading = calculate_z_scored_similarity(model, transformed_vector, gender_direction_vector)
                data.append([word, freq, original_loading, transformed_loading])
        logger.info(f"Processed sample_id {sample_id}")
        return data
    except Exception as e:
        logger.error(f"Error processing sample_id {sample_id}: {e}")
        return []


def main():
    parser = argparse.ArgumentParser(description="Process year data for target word analysis.")
    parser.add_argument("--year", type=int, required=True, help="Year to process")
    parser.add_argument("--samples", type=int, default=20, help="Number of samples")
    parser.add_argument("--max_threads", type=int, default=8, help="Maximum number of threads")
    parser.add_argument("--target_word", type=str, required=True, help="Target word")
    parser.add_argument("--anchor_words_positive", nargs='+', required=True, help="List of positive anchor words")
    parser.add_argument("--anchor_words_negative", nargs='+', required=True, help="List of negative anchor words")
    args = parser.parse_args()

    # Load data file and filter rows for the specified year
    csv_path = "/scratch/midway3/maxzhuyt/context_counts/stupid_combined_1830_2005.csv"
    df = pd.read_csv(csv_path, index_col=0)
    year_data = df.loc[args.year].to_dict()
    word_frequencies = {word: freq for word, freq in year_data.items() if freq > 0}

    logger.info(f"Processing year {args.year} with {len(word_frequencies)} non-zero frequency words.")

    data_path = '/project/macs40123/maxzhuyt/tokenized_coha'
    # Process samples concurrently
    with ThreadPoolExecutor(max_workers=args.max_threads) as executor:
        futures = [
            executor.submit(
                process_sample_id, sample_id, args.year, args.year, data_path,
                word_frequencies, args.target_word, args.anchor_words_positive, args.anchor_words_negative
            )
            for sample_id in range(1, args.samples)
        ]
        results = [future.result() for future in as_completed(futures)]

    # Average the loading scores across samples
    averaged_results = {}
    for data in results:
        for word, freq, orig_load, trans_load in data:
            if word not in averaged_results:
                averaged_results[word] = [0, 0, 0]  # freq_sum, orig_sum, trans_sum
            averaged_results[word][0] += freq
            averaged_results[word][1] += orig_load
            averaged_results[word][2] += trans_load

    averaged_data = [
        [word, values[0] / args.samples, values[1] / args.samples, values[2] / args.samples]
        for word, values in averaged_results.items()
    ]

    # Save results to CSV
    output_file = f"output_{args.year}_{args.target_word}.csv"
    df_output = pd.DataFrame(averaged_data, columns=["Word", "Frequency", "Original Loading", "Transformed Loading"])
    df_output.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")

if __name__ == "__main__":
    main()
