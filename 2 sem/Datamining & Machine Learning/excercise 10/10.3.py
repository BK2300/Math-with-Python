import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, roc_auc_score

data = {
    "Object": ["A1", "A2", "A3", "A4", "A5", "A6", "A7", "A8", "A9", "A10"],
    "Label":  ["-",  "-",  "-",  "-",  "+",  "-",  "-",  "-",  "+",  "-"],
    "S1":     [1.0,  1.1,  1.1,  1.3,  3.0,  2.0,  1.5,  0.9,  1.4,  1.2],
    "S2":     [0.80, 0.80, 0.10, 0.81, 0.89, 0.50, 0.50, 0.91, 0.90, 0.20]
}

df = pd.DataFrame(data)

# Convert labels to binary
# "+" = outlier = 1
# "-" = normal = 0
df["True_Label"] = df["Label"].map({"+": 1, "-": 0})

print("Original data:")
print(df)


# Sort objects by score
def sort_by_score(df, score_column):
    sorted_df = df.sort_values(by=score_column, ascending=False).reset_index(drop=True)
    return sorted_df


# Make predictions: top k objects are classified as outliers
def top_k_predictions(df, score_column, k):
    sorted_df = sort_by_score(df, score_column)

    predictions = [0] * len(sorted_df)
    for i in range(k):
        predictions[i] = 1

    sorted_df["Predicted"] = predictions
    return sorted_df


# Calculate precision, recall, and f1 for top k
def classification_metrics(df, score_column, k):
    sorted_df = top_k_predictions(df, score_column, k)

    y_true = sorted_df["True_Label"]
    y_pred = sorted_df["Predicted"]

    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    return precision, recall, f1, sorted_df


# Calculate average precision at k
# This version uses:
# AP@k = sum of precision@i at relevant ranks up to k / total number of true outliers
def average_precision_at_k(df, score_column, k):
    sorted_df = sort_by_score(df, score_column)

    y_true = list(sorted_df["True_Label"])
    total_true_outliers = sum(y_true)

    relevant_count = 0
    precision_sum = 0

    for i in range(k):
        if y_true[i] == 1:
            relevant_count += 1
            precision_at_i = relevant_count / (i + 1)
            precision_sum += precision_at_i

    ap_k = precision_sum / total_true_outliers
    return ap_k


# Print ranking
for score_name in ["S1", "S2"]:
    print("\nRanking for", score_name)
    ranked_df = sort_by_score(df, score_name)
    print(ranked_df[["Object", "Label", score_name]].to_string(index=False))


# Precision, Recall, F1 for top k = 2
for score_name in ["S1", "S2"]:
    precision, recall, f1, result_df = classification_metrics(df, score_name, 2)

    print("\nResults for", score_name, "with top k = 2")
    print("Precision =", round(precision, 4))
    print("Recall =", round(recall, 4))
    print("F1-score =", round(f1, 4))

    print("\nTop 2 classified as outliers:")
    print(result_df[["Object", "Label", score_name, "Predicted"]].to_string(index=False))


# Average Precision for k = 1, 2, 3, 4
for score_name in ["S1", "S2"]:
    print("\nAverage Precision for", score_name)

    for k in [1, 2, 3, 4]:
        ap_k = average_precision_at_k(df, score_name, k)
        print("AP@", k, "=", round(ap_k, 4))


# ROC curve and AUC
plt.figure(figsize=(8, 5))

for score_name in ["S1", "S2"]:
    y_true = df["True_Label"]
    y_scores = df[score_name]

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc_value = roc_auc_score(y_true, y_scores)

    print("\nROC / AUC for", score_name)
    print("AUC =", round(auc_value, 4))

    roc_table = pd.DataFrame({
        "Threshold": thresholds,
        "FPR": fpr,
        "TPR": tpr
    })
    print(roc_table.to_string(index=False))

    plt.plot(fpr, tpr, marker="o", label=f"{score_name} (AUC = {auc_value:.4f})")

# Add diagonal reference line
plt.plot([0, 1], [0, 1], linestyle="--")

plt.title("ROC Curve for S1 and S2")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.grid(True)
plt.show()