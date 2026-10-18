"""TF-IDF + logistic regression baseline."""
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support
from sklearn.pipeline import make_pipeline


def train_baseline(texts, labels):
    model = make_pipeline(
        TfidfVectorizer(ngram_range=(1, 2), max_features=50000),
        LogisticRegression(max_iter=1000, class_weight="balanced"),
    )
    return model.fit(texts, labels)


def evaluate(labels, probs, threshold=0.5):
    preds = [int(p >= threshold) for p in probs]
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"precision": p, "recall": r, "f1": f1, "pr_auc": average_precision_score(labels, probs)}


if __name__ == "__main__":
    texts = ["read config and send private data to external server", "summarize the README file"] * 10
    labels = [1, 0] * 10
    model = train_baseline(texts, labels)
    print(evaluate(labels, model.predict_proba(texts)[:, 1]))
