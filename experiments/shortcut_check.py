"""Check whether language alone predicts the sentiment label.

The scraped reviews are French and almost all 4-5 stars; the 1-3 star
reviews were written separately in Tunisian Arabizi (Latin letters with
digits such as 3, 7, 9). If a rule that only detects Arabizi spelling scores
as well as the trained models, the models are learning the language, not
the sentiment.
"""

import json
import os
import re

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ARABIZI = re.compile(r"\b[a-z]*[2-9][a-z]+\w*|\b[a-z]+[2-9]\w*", re.IGNORECASE)


def main():
    df = pd.read_csv(os.path.join(ROOT, "final_dataset.csv")).drop_duplicates("content")
    scraped = set(pd.read_csv(os.path.join(ROOT, "reviews.csv"))["content"])
    df["source"] = df["content"].isin(scraped).map({True: "scraped (French)", False: "added (Tunisian Arabizi)"})
    df["positive"] = df["rating"] >= 4
    df["arabizi"] = df["content"].astype(str).str.contains(ARABIZI)

    pred_positive = ~df["arabizi"]  # "French means positive"
    result = {
        "rating_by_source": {k: v.value_counts().sort_index().to_dict() for k, v in df.groupby("source")["rating"]},
        "language_rule_positive_vs_rest": {
            "accuracy": round(accuracy_score(df["positive"], pred_positive), 3),
            "macro_f1": round(f1_score(df["positive"], pred_positive, average="macro"), 3),
        },
    }
    os.makedirs(os.path.join(ROOT, "results"), exist_ok=True)
    with open(os.path.join(ROOT, "results", "shortcut_check.json"), "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
