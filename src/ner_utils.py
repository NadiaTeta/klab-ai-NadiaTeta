"""Reusable helpers for token-level Named Entity Recognition (NER) with word embeddings.

Nothing here is tied to CoNLL-2003 or to one embedding: a dataset is just a list of
token lists plus a list of BIO tag lists, and an embedding is any function that
maps a word to a vector (or None when the word is unknown). That makes it easy to
swap in another dataset (e.g. MasakhaNER for Kinyarwanda) or other vectors.
"""
import re
import time
from collections import Counter

import numpy as np
from seqeval.metrics import classification_report, f1_score
from seqeval.metrics.sequence_labeling import get_entities
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

ENTITY_TYPES = ["PER", "ORG", "LOC", "MISC"]


# ---------------------------------------------------------------- data loading

def load_hf_ner(name="eriktks/conll2003", revision="refs/convert/parquet"):
    """Load a Hugging Face NER dataset with `tokens` and `ner_tags` columns.

    Returns ({split: (token_lists, tag_lists)}, tag_names), with tag names instead of numbers.
    """
    from datasets import load_dataset

    dataset = load_dataset(name, revision=revision)
    tag_names = dataset["train"].features["ner_tags"].feature.names
    splits = {split: ([list(t) for t in data["tokens"]],
                      [[tag_names[i] for i in seq] for seq in data["ner_tags"]])
              for split, data in dataset.items()}
    return splits, tag_names


def read_conll_file(path):
    """Read a 'word TAG' per line file; blank lines separate sentences, '#' lines are comments."""
    sentences, tags, words, labels = [], [], [], []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if not line:
                if words:
                    sentences.append(words)
                    tags.append(labels)
                    words, labels = [], []
                continue
            word, label = line.rsplit(maxsplit=1)
            words.append(word)
            labels.append(label)
    if words:
        sentences.append(words)
        tags.append(labels)
    return sentences, tags


def count_entities(tag_seqs, entity_types=ENTITY_TYPES):
    """Count entities by counting B- tags (each entity starts with exactly one)."""
    counts = Counter(t[2:] for seq in tag_seqs for t in seq if t.startswith("B-"))
    return {etype: counts[etype] for etype in entity_types}


def flatten(tag_seqs):
    return [t for seq in tag_seqs for t in seq]


def regroup(flat_tags, sentences):
    """Cut a flat list of predicted tags back into one list per sentence."""
    out, pos = [], 0
    for sent in sentences:
        out.append(list(flat_tags[pos:pos + len(sent)]))
        pos += len(sent)
    return out


# ---------------------------------------------------------------- features

def spelling_features(sentence, i):
    """Simple spelling (shape) features for the word at position i and its neighbours."""
    word = sentence[i]
    feats = {
        "is_title": word[:1].isupper(),        # starts with a capital letter
        "is_upper": word.isupper(),            # ALL CAPS, e.g. "WASAC"
        "is_lower": word.islower(),
        "has_digit": any(ch.isdigit() for ch in word),
        "has_hyphen": "-" in word,
        "is_punct": not any(ch.isalnum() for ch in word),
        "is_first": i == 0,                    # capitals at the start of a sentence mean less
        "is_last": i == len(sentence) - 1,
    }
    if i > 0:
        feats["prev_is_title"] = sentence[i - 1][:1].isupper()
        feats["prev_is_upper"] = sentence[i - 1].isupper()
    if i < len(sentence) - 1:
        feats["next_is_title"] = sentence[i + 1][:1].isupper()
        feats["next_is_upper"] = sentence[i + 1].isupper()
    return {name: float(value) for name, value in feats.items()}


def baseline_features(sentence, i):
    """Baseline: who the word is (word, neighbours, suffix) + spelling features."""
    feats = spelling_features(sentence, i)
    feats["word=" + sentence[i].lower()] = 1.0
    feats["suffix3=" + sentence[i][-3:].lower()] = 1.0
    feats["prev=" + (sentence[i - 1].lower() if i > 0 else "<START>")] = 1.0
    feats["next=" + (sentence[i + 1].lower() if i < len(sentence) - 1 else "<END>")] = 1.0
    return feats


def sentences_to_dicts(sentences, feature_fn):
    return [feature_fn(sent, i) for sent in sentences for i in range(len(sent))]


class BaselineFeaturizer:
    """Sparse one-hot features without embeddings (DictVectorizer over baseline_features)."""

    def __init__(self, feature_fn=baseline_features):
        self.feature_fn = feature_fn
        self.vectorizer = DictVectorizer()

    def fit_transform(self, sentences):
        return self.vectorizer.fit_transform(sentences_to_dicts(sentences, self.feature_fn))

    def transform(self, sentences):
        return self.vectorizer.transform(sentences_to_dicts(sentences, self.feature_fn))


def make_lookup(keyed_vectors, try_original_case=True):
    """Turn gensim KeyedVectors into a word -> vector-or-None function.

    try_original_case=False looks words up in lowercase only (right for GloVe);
    True tries the word as written first, then lowercase (right for cased vectors like FastText).
    """
    index = keyed_vectors.key_to_index

    def lookup(word):
        forms = (word, word.lower()) if try_original_case else (word.lower(),)
        for form in forms:
            if form in index:
                return keyed_vectors[form]
        return None

    return lookup


class EmbeddingFeaturizer:
    """Dense features: [previous, current, next word vectors] + spelling features, standardised.

    Unknown words (and the slots before the first / after the last word) get a zero vector.
    """

    def __init__(self, lookup, dim, spelling_fn=spelling_features):
        self.lookup, self.dim, self.spelling_fn = lookup, dim, spelling_fn
        self.spelling_vectorizer = DictVectorizer()
        self.scaler = StandardScaler()

    def _raw(self, sentences):
        n_words = sum(len(s) for s in sentences)
        vectors = np.zeros((n_words, 3 * self.dim), dtype=np.float32)
        row = 0
        for sent in sentences:
            sent_vecs = [self.lookup(w) for w in sent]
            for i in range(len(sent)):
                for slot, j in enumerate([i - 1, i, i + 1]):          # previous, current, next
                    if 0 <= j < len(sent) and sent_vecs[j] is not None:
                        vectors[row, slot * self.dim:(slot + 1) * self.dim] = sent_vecs[j]
                row += 1
        spelling = self.spelling_vectorizer.transform(sentences_to_dicts(sentences, self.spelling_fn))
        return np.hstack([vectors, spelling.toarray().astype(np.float32)])

    def fit_transform(self, sentences):
        self.spelling_vectorizer.fit(sentences_to_dicts(sentences, self.spelling_fn))
        return self.scaler.fit_transform(self._raw(sentences))

    def transform(self, sentences):
        return self.scaler.transform(self._raw(sentences))


def featurizer_state(featurizer):
    """Everything needed to rebuild a fitted featurizer later, except the embedding lookup
    (a lookup wraps the vectors, which are too big to save with the model)."""
    if isinstance(featurizer, BaselineFeaturizer):
        return {"kind": "baseline", "vectorizer": featurizer.vectorizer}
    return {"kind": "embedding", "dim": featurizer.dim,
            "spelling_vectorizer": featurizer.spelling_vectorizer, "scaler": featurizer.scaler}


def featurizer_from_state(state, lookup=None):
    """Rebuild a featurizer saved with featurizer_state(); embedding featurizers need a lookup."""
    if state["kind"] == "baseline":
        featurizer = BaselineFeaturizer()
        featurizer.vectorizer = state["vectorizer"]
        return featurizer
    featurizer = EmbeddingFeaturizer(lookup, state["dim"])
    featurizer.spelling_vectorizer = state["spelling_vectorizer"]
    featurizer.scaler = state["scaler"]
    return featurizer


def oov_stats(sentences, lookup):
    """How many tokens / distinct words the embedding doesn't know."""
    all_words = [w for s in sentences for w in s]
    unique = set(all_words)
    oov_tokens = sum(lookup(w) is None for w in all_words)
    oov_unique = sorted(w for w in unique if lookup(w) is None)
    return {"OOV tokens": oov_tokens,
            "OOV % of tokens": round(100 * oov_tokens / len(all_words), 2),
            "OOV distinct words": len(oov_unique),
            "OOV % of distinct words": round(100 * len(oov_unique) / len(unique), 2),
            "examples": ", ".join(oov_unique[:8])}


# ---------------------------------------------------------------- models

def make_logreg(C=1.0):
    return LogisticRegression(C=C, max_iter=1000)


def make_random_forest():
    return RandomForestClassifier(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=42)


def make_mlp():
    return MLPClassifier(hidden_layer_sizes=(256,), early_stopping=True, max_iter=50, random_state=42)


def train_logreg_with_validation(X_train, y_train, X_val, val_sentences, val_tags, Cs=(0.001, 0.01, 0.1, 1, 10, 100)):
    """Train one Logistic Regression per C, print validation F1, return the best model."""
    best_model, best_f1 = None, -1
    for C in Cs:
        start = time.time()
        model = make_logreg(C).fit(X_train, y_train)
        val_f1 = f1_score(val_tags, regroup(model.predict(X_val), val_sentences))
        print(f"C={C:<6} validation F1 = {val_f1:.4f}   ({time.time() - start:.0f}s)")
        if val_f1 > best_f1:
            best_model, best_f1 = model, val_f1
    print("Best C:", best_model.C)
    return best_model


def evaluate(model, X, sentences, tags):
    """Predict, then return (predicted tag lists, seqeval report dict, overall micro F1)."""
    pred = regroup(model.predict(X), sentences)
    return pred, classification_report(tags, pred, output_dict=True, zero_division=0), f1_score(tags, pred)


def print_report(tags, pred):
    print(classification_report(tags, pred, digits=3, zero_division=0))


def entity_recall_by_seen(sentences, true_tags, pred_tags, train_vocab):
    """Recall on entities whose words appear in training ('seen') vs entities with no known word ('unseen')."""
    found = {"seen": [0, 0], "unseen": [0, 0]}
    for words, true_seq, pred_seq in zip(sentences, true_tags, pred_tags):
        predicted = set(get_entities(pred_seq))
        for etype, start, end in get_entities(true_seq):
            group = "seen" if any(w.lower() in train_vocab for w in words[start:end + 1]) else "unseen"
            found[group][0] += (etype, start, end) in predicted
            found[group][1] += 1
    return found


# ---------------------------------------------------------------- prediction

def tokenize(sentence):
    """Split into words and punctuation, e.g. 'Kigali.' -> ['Kigali', '.']"""
    return re.findall(r"\w+(?:[-']\w+)*|[^\w\s]", sentence)


def predict_entities(sentence, model, featurizer):
    """Return [(word, predicted tag), ...] for a plain-text sentence."""
    words = tokenize(sentence)
    return list(zip(words, model.predict(featurizer.transform([words]))))


def entities_from_pairs(pairs):
    """[(word, tag), ...] -> [(entity type, entity text), ...]"""
    words, tags = [w for w, _ in pairs], [t for _, t in pairs]
    return [(etype, " ".join(words[s:e + 1])) for etype, s, e in get_entities(tags)]
