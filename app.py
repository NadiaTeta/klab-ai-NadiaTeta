"""Gradio demo: Named Entity Recognition with word embeddings.

Compares the baseline (no embeddings) with the best model from notebooks/ner_embeddings.ipynb.
Run the notebook once first: its last cell saves the models to models/, and its FastText step
caches the vectors in data/embeddings/.

    python app.py      # then open http://127.0.0.1:7860
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import gradio as gr
import joblib
from gensim.models import KeyedVectors
from seqeval.metrics.sequence_labeling import get_entities

import ner_utils as nu

COLORS = {"PER": "#F4A582", "ORG": "#92C5DE", "LOC": "#A6DBA0", "MISC": "#C2A5CF"}
TYPE_NAMES = {"PER": "person", "ORG": "organisation", "LOC": "location", "MISC": "other name"}
EXAMPLES = [
    "WASAC announced that water will be cut in Kicukiro and Gikondo on Friday.",
    "Residents of Nyamirambo said WASAC has not restored water since Monday.",
    "Eric Niyonzima met the president of Kenya in Nairobi.",
    "Nadia studies software engineering at the African Leadership University in Kigali.",
]


def load_embedding(info):
    """Rebuild the word -> vector lookup for a saved embedding model."""
    path = os.path.join(ROOT, info["path"]) if info.get("path") else None
    if path and os.path.exists(path):
        vectors = KeyedVectors.load(path, mmap="r")  # stays on disk, read only when needed
    else:
        import gensim.downloader as api
        print(f"Downloading {info['gensim_name']} (first run only)...")
        vectors = api.load(info["gensim_name"])
    return nu.make_lookup(vectors, try_original_case=info["try_original_case"])


def load_bundle(role):
    path = os.path.join(ROOT, "models", f"{role}.joblib")
    if not os.path.exists(path):
        sys.exit(f"{path} not found. Run notebooks/ner_embeddings.ipynb once (its last cell saves the models).")
    bundle = joblib.load(path)
    lookup = load_embedding(bundle["embedding"]) if bundle["embedding"] else None
    bundle["featurizer"] = nu.featurizer_from_state(bundle["featurizer"], lookup)
    return bundle


print("Loading models...")
BASELINE = load_bundle("baseline")
BEST = load_bundle("best")
print(f"Loaded: {BASELINE['name']} (test F1 {BASELINE['test_f1']:.3f}) and {BEST['name']} (test F1 {BEST['test_f1']:.3f})")


def detokenize(words):
    """Join tokens back into readable text: no space before punctuation like . , ) ?"""
    return re.sub(r" ([.,!?;:)\]'%])", r"\1", " ".join(words)).replace("( ", "(")


def tag(sentence, bundle):
    """Return (highlighted spans, table rows) for one model."""
    pairs = nu.predict_entities(sentence, bundle["model"], bundle["featurizer"])
    words, tags = [w for w, _ in pairs], [t for _, t in pairs]
    entities = get_entities(tags)
    owner = [None] * len(words)  # which entity each word belongs to
    for k, (_, start, end) in enumerate(entities):
        for j in range(start, end + 1):
            owner[j] = k
    spans, i = [], 0
    while i < len(words):
        j = i
        while j + 1 < len(words) and owner[j + 1] == owner[i]:
            j += 1
        text = detokenize(words[i:j + 1])
        next_is_punct = j + 1 < len(words) and re.fullmatch(r"[.,!?;:)\]'%]", words[j + 1])
        if j + 1 < len(words) and not next_is_punct:
            text += " " if owner[i] is None else ""
        spans.append((text, None if owner[i] is None else entities[owner[i]][0]))
        if owner[i] is not None and j + 1 < len(words) and not next_is_punct:
            spans.append((" ", None))
        i = j + 1
    rows = [[detokenize(words[s:e + 1]), f"{t} ({TYPE_NAMES[t]})"] for t, s, e in entities]
    return spans, rows or [["(no names found)", ""]]


def analyze(sentence):
    sentence = (sentence or "").strip()
    if not sentence:
        empty = [["(type a sentence first)", ""]]
        return [], empty, [], empty
    base_spans, base_rows = tag(sentence, BASELINE)
    best_spans, best_rows = tag(sentence, BEST)
    return base_spans, base_rows, best_spans, best_rows


with gr.Blocks(title="NER with Word Embeddings") as demo:
    gr.Markdown("# Named Entity Recognition with Word Embeddings")
    gr.Markdown("Colours: **PER** = person (orange) · **ORG** = organisation (blue) · "
                "**LOC** = location (green) · **MISC** = other names, e.g. nationalities (purple).")
    sentence = gr.Textbox(label="English sentence", placeholder="Type or paste a sentence...", lines=2)
    analyze_btn = gr.Button("Analyze", variant="primary")
    with gr.Row():
        with gr.Column():
            gr.Markdown(f"### Before: {BASELINE['name']}\nTest F1 on CoNLL-2003: {BASELINE['test_f1']:.3f}")
            base_text = gr.HighlightedText(label="Entities", color_map=COLORS, show_legend=True)
            base_table = gr.Dataframe(headers=["Entity", "Type"], label="Entities found", interactive=False)
        with gr.Column():
            gr.Markdown(f"### After: {BEST['name']}\nTest F1 on CoNLL-2003: {BEST['test_f1']:.3f}")
            best_text = gr.HighlightedText(label="Entities", color_map=COLORS, show_legend=True)
            best_table = gr.Dataframe(headers=["Entity", "Type"], label="Entities found", interactive=False)
    outputs = [base_text, base_table, best_text, best_table]
    gr.Examples(examples=[[s] for s in EXAMPLES], inputs=sentence, outputs=outputs, fn=analyze,
                run_on_click=True, cache_examples=False)
    analyze_btn.click(analyze, inputs=sentence, outputs=outputs, api_name="analyze")
    sentence.submit(analyze, inputs=sentence, outputs=outputs)

# Always show the light theme, even when the computer is in dark mode (easier to read on a projector)
FORCE_LIGHT = """
() => {
    const url = new URL(window.location.href);
    if (url.searchParams.get("__theme") !== "light") {
        url.searchParams.set("__theme", "light");
        window.location.replace(url.href);
    }
}
"""

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft(primary_hue="blue"), js=FORCE_LIGHT)
