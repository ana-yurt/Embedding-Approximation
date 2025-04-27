import spacy
import os
import json
import pandas as pd
from collections import defaultdict, Counter
from multiprocessing import Pool
import logging

import re

# === Constants ===
words = ['stupid', 'foolish', 'dumb', 'idiotic']
regex_pattern = re.compile(r'\b(?:' + '|'.join(words) + r')\b', flags=re.IGNORECASE)

DATA_DIR = '/scratch/midway3/maxzhuyt/cleaned_coha_lower'
OUTPUT_JSON = "files_by_year.json"
OUTPUT_CSV = f"{'_'.join(words)}_targets_by_year_sbatch.csv"
LOG_PATH = f"{'_'.join(words)}_extraction.log"


# === Setup logging ===
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(f"multiword_logger")
file_handler = logging.FileHandler(LOG_PATH)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# === Build file map by year ===
files_by_year = defaultdict(list)
for fname in os.listdir(DATA_DIR):
    if fname.endswith(".txt"):
        parts = fname.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            year = int(parts[1])
            if year >= 1900:
                files_by_year[year].append(os.path.join(DATA_DIR, fname))

with open(OUTPUT_JSON, "w") as f:
    json.dump(files_by_year, f, indent=2)

# === Multiprocessing task ===
def process_year(year_files_tuple):
    year, files = year_files_tuple
    logger.info(f"Processing year {year} with {len(files)} files")

    # Re-initialize spaCy inside subprocess
    nlp_sent = spacy.load("en_core_web_sm", disable=["ner", "tagger", "parser"])
    nlp_sent.add_pipe("sentencizer")
    nlp_sent.max_length = 2012100  # Set max length for large documents
    nlp_full = spacy.load("en_core_web_lg")
    nlp_full.max_length = 2012100  # Set max length for large documents
    def extract_stupid_sentences(text):
        doc = nlp_sent(text)
        return [sent.text for sent in doc.sents if regex_pattern.search(sent.text)]

    def parse_targets(sentences):
        counter = Counter()
        for doc in nlp_full.pipe(sentences, batch_size=8):
            for token in doc:
                if token.text.lower() in words:
                    dep = token.dep_
                    head = token.head

                    # "something1 is a stupid something2"
                    if dep == "amod" and head.pos_ in {"NOUN", "PROPN"}:
                        counter[head.lemma_.lower()] += 1
                        copula_verb = head.head
                        if copula_verb.lemma_ == "be":
                            subj = [tok for tok in copula_verb.children if tok.dep_ == "nsubj"]
                            if subj:
                                counter[subj[0].lemma_.lower()] += 1

                    # "something looks/appears/seems stupid"
                    elif dep == "acomp":
                        verb = head
                        subj = [tok for tok in verb.children if tok.dep_ == "nsubj"]
                        if subj:
                            counter[subj[0].lemma_.lower()] += 1

                    # "something1 and something2 are stupid"
                    elif dep == "conj" and token.head.text.lower() in words:
                        conj_targets = [child for child in token.subtree if child.dep_ == "nsubj"]
                        for subj in conj_targets:
                            counter[subj.lemma_.lower()] += 1
            

                    # "somebody tried to be stupid" or "be a stupid person"
                    elif token.dep_ == "acomp" and head.lemma_ == "be":
                        for ancestor in token.ancestors:
                            if ancestor.dep_ == "ROOT" and ancestor.pos_ == "VERB":
                                subject = [child for child in ancestor.children if child.dep_ == "nsubj"]
                                if subject:
                                    counter[subject[0].lemma_.lower()] += 1
                                break
        return counter


    year_counter = Counter()
    for path in files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                text = f.read()
            sents = extract_stupid_sentences(text)
            count = parse_targets(sents)
            year_counter.update(count)
        except Exception as e:
            logger.error(f"Error processing {path}: {e}")
            return {"year": int(year), "error": str(e)}
    logger.info(f"Finished processing year {year}, found {len(year_counter)} targets")
    return [{"year": int(year), "target": target, "count": count} for target, count in year_counter.items()]

# === Run with multiprocessing ===
year_file_pairs = sorted(files_by_year.items())
with Pool(processes=46) as pool:
    all_results = pool.map(process_year, year_file_pairs)

# === Flatten and save ===
flattened = [item for sublist in all_results if isinstance(sublist, list) for item in sublist]
df = pd.DataFrame(flattened)
df.sort_values(["year", "count"], ascending=[True, False], inplace=True)
df.to_csv(OUTPUT_CSV, index=False)
logger.info(f"Finished writing to {OUTPUT_CSV}")
