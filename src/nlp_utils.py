
import numpy as np
from tqdm import tqdm
import difflib
from itertools import combinations
from nltk.translate.bleu_score import sentence_bleu, corpus_bleu, SmoothingFunction

import re


def differs_only_by_color(s1, s2, asset_dir):
    with open(f"{asset_dir}/colors.txt", 'r') as f:
        COLORS = [line.strip() for line in f if line.strip()]
    pattern = r'\b(' + '|'.join(re.escape(color) for color in COLORS) + r')\b'
    
    # Replace color names with a placeholder
    s1_clean = re.sub(pattern, '<COLOR>', s1)
    s2_clean = re.sub(pattern, '<COLOR>', s2)
    
    return s1_clean == s2_clean


def differs_only_by_number(s1, s2):
    # Remove all digits from both strings
    s1_non_digits = re.sub(r'\d+', '', s1)
    s2_non_digits = re.sub(r'\d+', '', s2)
    return s1_non_digits == s2_non_digits


def get_diff(s1, s2):
    # Convert strings to lists of words
    list1 = s1.splitlines()
    list2 = s2.splitlines()
    diffs = []
    for i in range(len(list2)):
        if(i < len(list1) and list1[i] != list2[i]):
            diffs.append((list1[i], list2[i]))
        elif(i >= len(list1)):
            diffs.append((None, list2[i]))
    return diffs



def quality_bleu(texts, refs):
    smooth_fn = SmoothingFunction().method0
    references = []
    hypotheses = []
    for candidate in texts:
        references.append([text.split() for text in refs])
        candidate_tokens = candidate.split()
        hypotheses.append(candidate_tokens)
    try:
        score = corpus_bleu(references, hypotheses, weights=[tuple(1/1 for _ in range(1)), 
                                                             tuple(1/2 for _ in range(2)),
                                                            tuple(1/3 for _ in range(3)),
                                                            tuple(1/4 for _ in range(4))], smoothing_function=smooth_fn)
    except:
        print("Issue getting score!")
    return np.mean(score)

def self_bleu(texts):
    if not texts or len(texts) < 2:
        raise ValueError("At least two texts are required to compute Self-BLEU.")
    smooth_fn = SmoothingFunction().method0
    references = []
    hypotheses = []
    for i, candidate in enumerate(texts):
        references.append([text.split() for j, text in enumerate(texts) if j != i])
        candidate_tokens = candidate.split()
        hypotheses.append(candidate_tokens)
    score = corpus_bleu(references, hypotheses, weights=[tuple(1/1 for _ in range(1)), 
                                                         tuple(1/2 for _ in range(2)),
                                                         tuple(1/3 for _ in range(3)),
                                                         tuple(1/4 for _ in range(4))], smoothing_function=smooth_fn)
    return np.mean(score)

def jaccard_similarity(set1, set2):
    """Compute Jaccard similarity between two sets."""
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union != 0 else 0

def self_jaccard(texts, n_gram=4):
    if not texts or len(texts) < 2:
        raise ValueError("At least two texts are required to compute Self-BLEU.")
    token_texts = [set(t.split()) for t in texts]
    pairs = list(combinations(token_texts, 2))  # Get all unique pairs
    if not pairs:  # Handle edge case if there's only one or zero lists
        return 0
    similarities = [jaccard_similarity(s1, s2) for s1, s2 in pairs]
    return similarities

