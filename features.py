# --- Set up environment ---
import pandas as pd
import spacy
import numpy as np
import re

from collections import Counter
from collections import defaultdict
from tqdm import tqdm
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.nn.functional import softmax
from torch import argmax

tqdm.pandas()
nlp = spacy.load("en_core_web_sm")

# ----- Baseline features -----
# --- Sentence and word length ---
def extract_length_features(doc):
    # Sentences (filtered for valid tokens)
    sent_lengths = [
        len([t for t in sent if not t.is_punct])
        for sent in doc.sents
        if len(sent) > 0
    ]

    # Words (exclude punctuation)
    words = [t for t in doc if not t.is_punct]
    word_lengths = [len(t.text) for t in words]

    # Compute stats
    return pd.Series({
        "mean_sent_len": np.nanmean(sent_lengths) if sent_lengths else 0,
        "std_sent_len":  np.nanstd(sent_lengths) if len(sent_lengths) > 1 else 0,
        "mean_word_len": np.nanmean(word_lengths) if word_lengths else 0,
        "std_word_len":  np.nanstd(word_lengths) if len(word_lengths) > 1 else 0,
    })

# --- Stop words ---
def extract_stopword_rate(doc):
    tokens = [t for t in doc if not t.is_punct]
    num_tokens = len(tokens)
    num_stopwords = sum(t.is_stop for t in tokens)

    stopword_ratio = num_stopwords / num_tokens if num_tokens > 0 else 0
    return pd.Series({
        "stopword_ratio": stopword_ratio
    })

# --- Punctuation ---
# Define punctuation marks
punct_marks = [",", ".", ";", ":", "?", "!", "'", '"', "(", ")", "-", "…"]

def extract_punct_rates(text):
    total_chars = len(text)
    counts = {f"punct_{p}": text.count(p) / total_chars * 1000 if total_chars > 0 else 0
              for p in punct_marks}
    return pd.Series(counts)

# --- Casing ratios ---
def extract_char_class_ratios(text):
    total_chars = len(text)
    total_tokens = len(text.split())

    # Avoid division by zero
    if total_chars == 0:
        return pd.Series({k: 0 for k in ["upper_ratio", "title_ratio", "digit_ratio", "space_ratio"]})

    upper_ratio = sum(c.isupper() for c in text) / total_chars
    digit_ratio = sum(c.isdigit() for c in text) / total_chars
    space_ratio = sum(c.isspace() for c in text) / total_chars

    # Titlecase = first letter uppercase, rest lowercase
    title_ratio = sum(w.istitle() for w in text.split()) / total_tokens if total_tokens > 0 else 0

    return pd.Series({
        "upper_ratio": upper_ratio,
        "title_ratio": title_ratio,
        "digit_ratio": digit_ratio,
        "space_ratio": space_ratio
    })

# --- Type–Token ratio ---
def extract_ttr(doc):
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    num_tokens = len(tokens)
    num_types = len(set(tokens))

    ttr = num_types / num_tokens if num_tokens > 0 else 0
    return pd.Series({"ttr": ttr})

# --- UPOS distribution --
upos_tags = ['NOUN', 'VERB', 'ADJ', 'ADV', 'PRON', 'ADP', 'DET', 'CCONJ', 'SCONJ', 'NUM', 'AUX', 'INTJ', 'PART', 'PROPN']

def extract_upos_freq(doc):
    tags = [t.pos_ for t in doc if not t.is_punct and not t.is_space]
    total = len(tags)

    tag_counts = Counter(tags)
    freqs = {
        f"upos_{tag}": tag_counts.get(tag, 0) / total if total > 0 else 0
        for tag in upos_tags
    }
    return pd.Series(freqs)

# --- Max 5-gram repetition rate ---
def compute_5gram_repetition(doc):
    # Use alpha-only tokens (optional: can include all if you prefer)
    tokens = [t.text.lower() for t in doc if not t.is_space]
    n = 5
    total_tokens = len(tokens)

    if total_tokens < n:
        return pd.Series({"rep_5gram_ratio": 0.0})

    # Count all 5-grams
    counts = defaultdict(int)
    for i in range(total_tokens - n + 1):
        fivegram = tuple(tokens[i:i+n])
        counts[fivegram] += 1

    # Find all repeated 5-grams
    repeated_spans = set()
    for i in range(total_tokens - n + 1):
        fivegram = tuple(tokens[i:i+n])
        if counts[fivegram] > 1:
            repeated_spans.update(range(i, i+n))

    # Compute repetition ratio
    rep_ratio = len(repeated_spans) / total_tokens if total_tokens > 0 else 0
    return pd.Series({"rep_5gram_ratio": rep_ratio})

# --- Self-similarity ---
def compute_self_similarity(doc, k=200):
    # Use clean lowercase tokens
    tokens = [t.text.lower() for t in doc if t.is_alpha]
    total = len(tokens)

    half = total // 2
    first_half = tokens[:half]
    second_half = tokens[half:]

    # Count top-k unigrams in each half
    first_top = set([w for w, _ in Counter(first_half).most_common(k)])
    second_top = set([w for w, _ in Counter(second_half).most_common(k)])

    # Compute Jaccard similarity
    intersection = first_top & second_top
    union = first_top | second_top

    sim = len(intersection) / len(union) if union else 0.0
    return pd.Series({"self_sim_jaccard": sim})

# ----- Bias features -----
# --- Sentiment analysis ---

vader = SentimentIntensityAnalyzer()
def compute_sentiment_features(text):
    sentences = sent_tokenize(text)
    if not sentences:
        return pd.Series({"sentiment_mean": 0.0, "sentiment_var": 0.0})

    scores = [vader.polarity_scores(s)["compound"] for s in sentences]
    return pd.Series({
        "sentiment_mean": np.mean(scores),
        "sentiment_var": np.var(scores)
    })

# --- Profanity and insults ---
# Load profanity / insult lexicon (one word per line)
def load_profanity_lexicon(path):
    with open(path, "r", encoding="cp1252") as f:
        return {line.strip().lower() for line in f if line.strip()}

def compute_profanity_rate(text, lexicon):
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return 0.0
    return sum(t in lexicon for t in tokens) / len(tokens)

# --- Positivity and negativity --

# Load lexicons

def load_lexicon(path):
    """Loads a lexicon file (one word per line, allows comment lines)."""
    with open(path, "r") as f:
        return {line.strip().lower() for line in f
                if line.strip() and not line.startswith(";")}
                    
# Compute rates
def compute_emotional_tone_from_lexicons(text, pos_lex, neg_lex):
    tokens = re.findall(r"\b\w+\b", text.lower())
    if not tokens:
        return pd.Series({"pos_rate": 0.0, "neg_rate": 0.0, "polarity_score": 0.0})
    pos_rate = sum(t in pos_lex for t in tokens) / len(tokens)
    neg_rate = sum(t in neg_lex for t in tokens) / len(tokens)
    return pd.Series({
        "pos_rate": pos_rate,
        "neg_rate": neg_rate,
        "polarity_score": pos_rate - neg_rate
    })

# --- Identity terms ---
# Define lexicons
identity_lexicons = {
    "gender": {
        "he", "him", "his", "she", "her", "hers", "woman", "man",
        "male", "female", "boy", "girl", "mother", "father",
        "sister", "brother"
    },

    "race_ethnicity": {
        "white", "black", "asian", "latino", "hispanic",
        "african", "european", "indian", "arab"
    },

    "religion": {
        "christian", "muslim", "jewish", "hindu", "buddhist",
        "islam", "christianity", "judaism"
    },

    "nationality": {
        "american", "british", "german", "french", "spanish",
        "russian", "chinese", "japanese", "italian", "canadian"
    },

    "orientation": {
        "gay", "lesbian", "bisexual", "transgender", "queer", "lgbt"
    },

    "disability": {
        "blind", "deaf", "autistic", "disabled", "mental",
        "handicapped", "wheelchair"
    },

    "age": {
        "child", "kid", "youth", "teen", "adult", "elderly",
        "senior", "old", "young"
    }
}

# Compute rates
def compute_identity_term_rates(text, lexicons=identity_lexicons):
    tokens = re.findall(r"\b\w+\b", text.lower())
    total = len(tokens)

    if total == 0:
        return pd.Series({f"{k}_identity_rate": 0.0 for k in lexicons})

    features = {}
    for category, words in lexicons.items():
        count = sum(t in words for t in tokens)
        features[f"{category}_identity_rate"] = count / total

    return pd.Series(features)

# --- Hedging and modality ---
# Define lexicons
hedge_lexicon = {
    "modals": [
        "may", "might", "could", "should", "would",
        "can", "shall", "ought",
        "possibly", "conceivably"
    ],

    "adverbs": [
        "perhaps", "probably", "possibly", "apparently", "likely",
        "evidently", "presumably", "seemingly", "maybe",
        "hopefully", "arguably", "theoretically",
        "hypothetically", "roughly", "approximately",
        "almost", "virtually", "essentially", "generally",
        "typically", "usually", "frequently", "often",
        "mostly", "largely", "somewhat", "partially"
    ],

    "verbs": [
        "seem", "appear", "suggest", "imply", "assume", "believe",
        "think", "guess", "suspect", "estimate", "suppose",
        "consider", "imagine", "indicate", "hint", "propose",
        "speculate", "predict", "infer", "presume",
    ],

    "phrases": [
        "it seems", "it appears", "it is possible", "it is likely",
        "there is a chance", "it could be", "it might be",
        "to some extent", "in a way", "in some cases",
        "in certain respects", "it may be that",
        "one could argue", "one might think",
        "it could appear", "it could suggest"
    ]
}

# Compute hedging frequency
def compute_hedge_rate(text, lexicon):
    text_lower = text.lower()
    tokens = re.findall(r"\b\w+\b", text_lower)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    # Single-word matches
    single_words = set(lexicon["modals"] + lexicon["adverbs"] + lexicon["verbs"])
    count_single = sum(t in single_words for t in tokens)

    # Multi-word phrase matches
    count_phrases = sum(p in text_lower for p in lexicon["phrases"])

    return (count_single + count_phrases) / total_tokens

# --- Categorical statements ---
# Define lexicons
categorical_lexicon = [
    # Universal quantifiers
    "always", "never", "ever", "all", "none",
    "every", "everyone", "everything", "everybody",
    "nobody", "nothing", "noone", "no-one",

    # Absolutes / extremes
    "completely", "entirely", "absolutely", "totally",
    "utterly", "purely", "perfectly", "fully",
    "definitely", "certainly", "inevitably", "unquestionably",
    "undeniably", "unconditionally",

    # Extremeness / maximality
    "forever", "eternal", "eternally", "infinite", "infinitely",
    "permanent", "permanently", "final", "finally",
    "ultimate", "ultimately", "guaranteed", "guarantee",

    # Strong modal certainty
    "must", "cannot", "can't", "will", "won't",
    "always", "never",

    # Strong evaluative absolutes
    "best", "worst", "only", "everyone", "nobody",
    "universal", "universally", "all-time"
]

# Compute rate of categorical statements
def compute_categorical_rate(text, lexicon=categorical_lexicon):
    text_lower = text.lower()
    tokens = re.findall(r"\b\w+\b", text_lower)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    return sum(t in lexicon for t in tokens) / total_tokens

# --- Assertiveness ---
# Define lexicons
assertive_lexicon = [
    # Strong factual/claim verbs
    "prove", "proves", "proved",
    "demonstrate", "demonstrates", "demonstrated",
    "show", "shows", "shown",
    "confirm", "confirms", "confirmed",
    "establish", "establishes", "established",
    "verify", "verifies", "verified",

    # Strong certainty adverbs
    "clearly", "certainly", "definitely", "undeniably",
    "obviously", "evidently", "unquestionably",
    "undoubtedly", "incontrovertibly", "irrefutably",
    "surely", "decisively",

    # Strong necessity / obligation signals
    "must", "cannot", "can't", "will", "won't",
    "have to", "has to", "need to",

    # Strong evaluative certainty
    "without a doubt", "beyond doubt", "beyond question",
    "it is clear", "it is certain",
    "everyone knows", "it is obvious",

    # Rhetorical emphasis
    "in fact", "the truth is", "the reality is"
]

# Compute rate of assertive words
def compute_assertive_rate(text, lexicon=assertive_lexicon):
    text_lower = text.lower()
    tokens = re.findall(r"\b\w+\b", text_lower)
    total_tokens = len(tokens)
    if total_tokens == 0:
        return 0.0

    return sum(t in lexicon for t in tokens) / total_tokens

# --- Subjectivity ---
# Define lexicons
subjective_lexicon = [
    # subjective adjectives
    "good", "bad", "terrible", "wonderful", "awful", "amazing",
    "horrible", "excellent", "disgusting", "fantastic", "stupid",
    "brilliant", "ridiculous", "shocking", "tragic", "beautiful",
    "ugly", "unfair", "unjust", "biased", "corrupt",

    # subjective verbs (opinions, evaluations)
    "believe", "think", "feel", "guess", "suspect", "argue",
    "claim", "insist", "support", "oppose", "criticize",
    "praise", "endorse", "blame", "doubt", "prefer", "hate",
    "love", "enjoy", "fear", "worry"
]

# Compute rate of subjective words
def compute_subjectivity_score(text, lexicon=subjective_lexicon):
    text_lower = text.lower()
    tokens = re.findall(r"\b\w+\b", text_lower)
    total = len(tokens)
    if total == 0:
        return 0.0
    return sum(t in lexicon for t in tokens) / total