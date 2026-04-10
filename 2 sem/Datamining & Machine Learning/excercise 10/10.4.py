import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors, LocalOutlierFactor
from sklearn.cluster import OPTICS

file_path = r"C:\Users\BK230\OneDrive\Desktop\python pycharm folder\Math-with-Python\2 sem\Datamining & Machine Learning\excercise 10\3clusters-and-noise-2d.csv"

df = pd.read_csv(file_path, sep=r"\s+|,", engine="python", header=None, comment="#")
df.columns = ["x", "y", "label"]

print("First 5 rows:")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nLabel counts:")
print(df["label"].value_counts())

X = df[["x", "y"]].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# kNN outlier score
def knn_outlier_score(X, k):
    nn = NearestNeighbors(n_neighbors=k)
    nn.fit(X)
    distances, indices = nn.kneighbors(X)
    return distances[:, k - 1]

# Plot helper
def plot_scores(X_original, scores, title):
    plt.figure(figsize=(7, 5))
    plt.scatter(X_original[:, 0], X_original[:, 1], c=scores)
    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.colorbar(label="score")
    plt.show()

# kNN scores for different neighborhood sizes
for k in [3, 5, 10]:
    scores = knn_outlier_score(X_scaled, k)

    print("\nkNN outlier scores with k =", k)
    print("Minimum score:", round(scores.min(), 4))
    print("Maximum score:", round(scores.max(), 4))
    print("Average score:", round(scores.mean(), 4))

    plot_scores(X, scores, f"kNN outlier scores (k = {k})")

# LOF scores for different neighborhood sizes
for k in [3, 5, 10]:
    lof = LocalOutlierFactor(n_neighbors=k)
    lof_labels = lof.fit_predict(X_scaled)
    lof_scores = -lof.negative_outlier_factor_

    print("\nLOF scores with k =", k)
    print("Minimum score:", round(lof_scores.min(), 4))
    print("Maximum score:", round(lof_scores.max(), 4))
    print("Average score:", round(lof_scores.mean(), 4))

    plot_scores(X, lof_scores, f"LOF scores (k = {k})")

# Show the strongest LOF outliers for one chosen k
chosen_k = 5
lof = LocalOutlierFactor(n_neighbors=chosen_k)
lof_labels = lof.fit_predict(X_scaled)
lof_scores = -lof.negative_outlier_factor_

result_df = df.copy()
result_df["LOF_score"] = lof_scores
result_df["LOF_label"] = lof_labels

result_df = result_df.sort_values("LOF_score", ascending=False)

print("\nTop 10 LOF outliers with k = 5")
print(result_df[["x", "y", "label", "LOF_score", "LOF_label"]].head(10).to_string(index=False))

# OPTICS
optics = OPTICS(min_samples=5)
optics.fit(X_scaled)

optics_labels = optics.labels_

plt.figure(figsize=(7, 5))
plt.scatter(X[:, 0], X[:, 1], c=optics_labels)
plt.title("OPTICS clustering")
plt.xlabel("x")
plt.ylabel("y")
plt.show()

reachability = optics.reachability_[optics.ordering_]
ordering = np.arange(len(reachability))

plt.figure(figsize=(10, 4))
plt.plot(ordering, reachability, marker="o", linestyle="")
plt.title("OPTICS reachability plot")
plt.xlabel("Point order")
plt.ylabel("Reachability distance")
plt.show()

n_clusters = len(set(optics_labels)) - (1 if -1 in optics_labels else 0)
n_noise = np.sum(optics_labels == -1)

print("\nOPTICS summary:")
print("Number of clusters:", n_clusters)
print("Number of noise points:", n_noise)