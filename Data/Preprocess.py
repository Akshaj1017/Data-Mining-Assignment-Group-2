import argparse
import re
import zipfile
from io import TextIOWrapper
from pathlib import Path

import pandas as pd
import nltk
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# lazy/downloads
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

def clean_text(text: str) -> str:
    text = re.sub(r"n't", " not", text)
    text = re.sub(r"'ve", " have", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r"'m", " am", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub('<[^<]+?>', '', text)   # HTML
    text = re.sub(r'\d+', ' ', text)      # numbers -> space
    text = re.sub(r'[^\w\s]', ' ', text)  # punctuation -> space
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)      # collapse spaces
    return text.strip()

def preprocess_texts_spacy(texts, hotel_names):
    # Only tagger/ner for speed
    nlp = spacy.load("en_core_web_sm", disable=["parser"])
    docs = nlp.pipe(texts, batch_size=100, n_process=1)

    stop_words = set(stopwords.words('english'))
    stop_words.update(spacy.lang.en.stop_words.STOP_WORDS)
    # keep negations
    stop_words -= {'not', 'no', 'nor', 'neither', 'never', 'none'}

    lemmatizer = WordNetLemmatizer()
    out = []
    for doc, hotel_name in zip(docs, hotel_names):
        hotel_name_l = (hotel_name or "").lower()
        toks = []
        for tok in doc:
            if tok.pos_ == "PROPN" or tok.ent_type_ in {"PERSON", "GPE"}:
                continue
            w = tok.text.lower()
            if hotel_name_l and w == hotel_name_l:
                continue
            if w in stop_words:
                continue
            if (len(w) <= 1) or w.isdigit() or re.search(r'\d', w) or re.fullmatch(r'th', w):
                continue
            lem = lemmatizer.lemmatize(w)
            if len(lem) > 1 and lem.isalpha():
                toks.append(lem)
        out.append(" ".join(toks))
    return out

def read_zip_build_df(zip_path: Path, include_subset=None) -> pd.DataFrame:
    texts, labels, hotel_names = [], [], []

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for info in zf.infolist():
            if not info.filename.endswith('.txt'):
                continue
            if include_subset and include_subset not in info.filename:
                continue

            label = "DECEPTIVE" if "deceptive" in info.filename.lower() else (
                    "TRUTHFUL" if "truthful" in info.filename.lower() else None)
            if label is None:
                continue

            # Hotel (heuristic)
            name_parts = Path(info.filename).name.split('_')
            hotel = name_parts[1] if len(name_parts) > 1 else ""

            with zf.open(info.filename) as fh:
                text = TextIOWrapper(fh, encoding='utf-8').read()

            texts.append(clean_text(text))
            labels.append(label)
            hotel_names.append(hotel)

    processed = preprocess_texts_spacy(texts, hotel_names)
    return pd.DataFrame({"Review": processed, "Label": labels})

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--zip", type=str, help="Path to op_spam_v1.4.zip (or similar).")
    p.add_argument("--out_csv", type=str, default="Data/preprocessed_df.csv")
    p.add_argument("--subset", type=str, default=None, help="e.g. 'negative' to match prior setup")
    args = p.parse_args()

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.zip:
        df = read_zip_build_df(Path(args.zip), include_subset=args.subset)
        df.to_csv(out, index=False)
        print(f"Saved preprocessed CSV to: {out.resolve()} (rows={len(df)})")
    else:
        raise SystemExit("No --zip provided and CSV missing. Provide a dataset ZIP.")

if __name__ == "__main__":
    main()
