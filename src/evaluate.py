from sklearn.metrics import accuracy_score , f1_score, recall_score
import json 
from pathlib import Path 


import json
from pathlib import Path
from sklearn.metrics import accuracy_score, f1_score, recall_score

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted")
    income_recall = recall_score(y_test, y_pred, pos_label=">50K")

    recall_threshold = 0.80
    passed = income_recall >= recall_threshold

    report = {
        "accuracy": round(accuracy, 4),
        "f1_weighted": round(f1, 4),
        "recall_over_50k": round(income_recall, 4),
        "quality_gate_passed": bool(passed),
    }

    Path("reports").mkdir(parents=True, exist_ok=True)

    with open("reports/metrics.json", "w") as file:
        json.dump(report, file, indent=4)

    print("Metrics saved to reports/metrics.json")

    return report


