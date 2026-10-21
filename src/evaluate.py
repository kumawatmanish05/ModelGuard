from sklearn.metrics import accuracy_score , f1_score, recall_score

def evaluate_model(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)

    f1 = f1_score(
        y_test,
        y_pred,
        average = "weighted"
    )

    income_recall = recall_score(
        y_test,
        y_pred,
        pos_label = ">50K"
    )

    print("\nAccuracy:", round(accuracy, 4))
    print("\nF1 Score:", round(f1, 4))
    print("\nRecall for >50K:", round(income_recall, 4))

    return{
        "accuracy": accuracy,
        "f1_score": f1,
        "recall_over_50K": income_recall
    }