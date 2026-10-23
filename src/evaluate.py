
from sklearn.metrics import accuracy_score, f1_score, recall_score


RECALL_THRESHOLD = 0.80


def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    income_recall = recall_score(y_test, y_pred, pos_label=">50K")

    passed = income_recall >= RECALL_THRESHOLD

    metrics = {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "recall_over_50k": round(income_recall, 4),
        "quality_gate_passed": bool(passed),
    }

    print("\nAccuracy:", metrics["accuracy"])
    print("Weighted F1-score:", metrics["f1_weighted"])
    print("Recall for >50K:", metrics["recall_over_50k"])
    print("ModelGuard Quality Gate:", "PASS" if passed else "FAIL")

    return metrics
