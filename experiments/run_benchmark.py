"""Reproducible sentiment benchmark on the scraped ExpressShop reviews.

Labels come from the star rating (1-2 negative, 3 neutral, 4-5 positive).
The original `sentiment` column was produced by an Arabic sentiment model
applied to French text, so it is only used to measure its agreement with
the ratings, not as ground truth.

Usage: python experiments/run_benchmark.py [--skip-bert]
"""

import argparse
import json
import os
import random
import string
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

SEED = 42
LABELS = ["negative", "neutral", "positive"]
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results")


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def rating_to_label(rating):
    return "negative" if rating <= 2 else "neutral" if rating == 3 else "positive"


def load_data():
    df = pd.read_csv(os.path.join(ROOT, "final_dataset.csv"))
    df = df.dropna(subset=["content", "rating"])
    before = len(df)
    df = df.drop_duplicates(subset="content").reset_index(drop=True)
    df["label"] = df["rating"].astype(int).map(rating_to_label)
    return df, before - len(df)


STOP = set(stopwords.words("french"))


def preprocess(text):
    text = str(text).lower()
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(w for w in text.split() if w not in STOP)


def scores(y_true, y_pred):
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 3),
        "macro_f1": round(f1_score(y_true, y_pred, average="macro", labels=LABELS), 3),
        "f1_per_class": {
            l: round(s, 3)
            for l, s in zip(LABELS, f1_score(y_true, y_pred, average=None, labels=LABELS))
        },
    }


def classic_models():
    tfidf = lambda: TfidfVectorizer(preprocessor=preprocess, ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    return {
        "Majority class (baseline)": DummyClassifier(strategy="most_frequent"),
        "TF-IDF + Naive Bayes": make_pipeline(tfidf(), MultinomialNB()),
        "TF-IDF + Logistic Regression": make_pipeline(
            tfidf(), LogisticRegression(max_iter=2000, class_weight="balanced", C=5)
        ),
    }


def zero_shot_star_model(texts):
    """Pretrained multilingual review model (predicts 1-5 stars), no fine-tuning."""
    from transformers import pipeline

    clf = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment",
                   truncation=True, max_length=256)
    preds = clf(list(texts), batch_size=32)
    return [rating_to_label(int(p["label"][0])) for p in preds]


def fine_tune_camembert(train_texts, train_labels, test_texts, epochs=4):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    name = "almanach/camembert-base"
    tok = AutoTokenizer.from_pretrained(name)
    model = AutoModelForSequenceClassification.from_pretrained(name, num_labels=len(LABELS))
    to_id = {l: i for i, l in enumerate(LABELS)}

    def encode(texts):
        return tok(list(texts), truncation=True, max_length=160, padding="max_length", return_tensors="pt")

    enc = encode(train_texts)
    y = torch.tensor([to_id[l] for l in train_labels])
    # Class weights counter the imbalance (positives dominate).
    counts = torch.bincount(y, minlength=len(LABELS)).float()
    weights = counts.sum() / (len(LABELS) * counts)
    loader = DataLoader(TensorDataset(enc["input_ids"], enc["attention_mask"], y), batch_size=16, shuffle=True)

    opt = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * epochs * len(loader)), epochs * len(loader))
    loss_fn = torch.nn.CrossEntropyLoss(weight=weights)
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for ids, mask, labels in loader:
            opt.zero_grad()
            loss = loss_fn(model(input_ids=ids, attention_mask=mask).logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            total += loss.item()
        print(f"  CamemBERT epoch {epoch + 1}/{epochs} - loss {total / len(loader):.3f}", flush=True)

    model.eval()
    enc = encode(test_texts)
    preds = []
    with torch.no_grad():
        for i in range(0, len(test_texts), 32):
            logits = model(input_ids=enc["input_ids"][i:i + 32], attention_mask=enc["attention_mask"][i:i + 32]).logits
            preds += logits.argmax(-1).tolist()
    return [LABELS[p] for p in preds]


def plot_confusion(y_true, y_pred, title, path):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS)
    plt.figure(figsize=(4.6, 3.8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=LABELS, yticklabels=LABELS, cbar=False)
    plt.xlabel("Predicted")
    plt.ylabel("Actual (from star rating)")
    plt.title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-bert", action="store_true")
    args = parser.parse_args()
    seed_everything()
    os.makedirs(OUT, exist_ok=True)

    df, n_dup = load_data()
    results = {"dataset": {
        "reviews_after_dedup": len(df), "duplicates_removed": int(n_dup),
        "label_distribution": df["label"].value_counts().to_dict(),
    }}

    # How well did the original model-generated labels match the star ratings?
    high = df[df["rating"] >= 4]
    results["original_labels"] = {
        "share_of_4_5_star_reviews_labelled_negative": round((high["sentiment"] == "negative").mean(), 3),
        "agreement_with_rating_labels": round((df["sentiment"] == df["label"]).mean(), 3),
    }

    X_train, X_test, y_train, y_test = train_test_split(
        df["content"], df["label"], test_size=0.2, stratify=df["label"], random_state=SEED)
    results["split"] = {"train": len(X_train), "test": len(X_test)}

    models = {}
    cv = StratifiedKFold(5, shuffle=True, random_state=SEED)
    for name, model in classic_models().items():
        t = time.time()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        cv_f1 = cross_validate(model, df["content"], df["label"], cv=cv, scoring="f1_macro")["test_score"]
        models[name] = {**scores(y_test, pred), "cv_macro_f1": f"{cv_f1.mean():.3f} ± {cv_f1.std():.3f}",
                        "seconds": round(time.time() - t, 1)}
        print(name, models[name], flush=True)
        if name == "TF-IDF + Logistic Regression":
            plot_confusion(y_test, pred, name, os.path.join(OUT, "confusion_logreg.png"))

    if not args.skip_bert:
        t = time.time()
        pred = zero_shot_star_model(X_test)
        models["Multilingual BERT, zero-shot (nlptown)"] = {**scores(y_test, pred), "seconds": round(time.time() - t, 1)}
        print("zero-shot", models["Multilingual BERT, zero-shot (nlptown)"], flush=True)

        t = time.time()
        pred = fine_tune_camembert(X_train, y_train, X_test)
        models["CamemBERT, fine-tuned"] = {**scores(y_test, pred), "seconds": round(time.time() - t, 1)}
        print("camembert", models["CamemBERT, fine-tuned"], flush=True)
        plot_confusion(y_test, pred, "CamemBERT, fine-tuned", os.path.join(OUT, "confusion_camembert.png"))

    results["models"] = models
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
