# ============================================================
# EXERCISE 10.1 - SEEDS DATASET
# Beginner-friendly full version
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors


# ============================================================
# PART 1 - DATA PREPROCESSING
# Exercise 10.1(a)
# ============================================================

# Change the file name if needed
file_path = "seeds_dataset.txt"

# Column names for the Seeds dataset
columns = [
    "area",
    "perimeter",
    "compactness",
    "kernel_length",
    "kernel_width",
    "asymmetry_coefficient",
    "kernel_groove_length",
    "class"
]

# Read the dataset
df = pd.read_csv(file_path, sep=r"\s+", header=None, names=columns)

print("=== PART 1: DATA PREPROCESSING ===")
print("Dataset shape:")
print(df.shape)

print("\nFirst 5 rows:")
print(df.head())

print("\nMissing values:")
print(df.isnull().sum())

print("\nClass distribution:")
print(df["class"].value_counts().sort_index())

# Split into features and labels
X = df.drop("class", axis=1)
y = df["class"]

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

print("\nThe data has been standardized.")

# Use PCA to make 2D plots later
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

print("PCA has been created for visualization.")


# ============================================================
# PART 2 - K-MEANS CLUSTERING
# Exercise 10.1(b)
# ============================================================

print("\n=== PART 2: K-MEANS CLUSTERING ===")

k_values = [2, 3, 4, 5, 6]
kmeans_results = []

for k in k_values:
    model = KMeans(n_clusters=k, init="k-means++", random_state=42, n_init=10)
    labels = model.fit_predict(X_scaled)
    score = silhouette_score(X_scaled, labels)

    kmeans_results.append([k, score])

# Show the scores
print("\nK-means silhouette scores:")
for result in kmeans_results:
    print("k =", result[0], "| silhouette score =", round(result[1], 4))

# Find the best k
best_k = max(kmeans_results, key=lambda x: x[1])[0]
print("\nBest k for K-means:", best_k)

# Train the best K-means model again
best_kmeans = KMeans(n_clusters=best_k, init="k-means++", random_state=42, n_init=10)
kmeans_labels = best_kmeans.fit_predict(X_scaled)

# Plot silhouette scores
scores_only = [result[1] for result in kmeans_results]

plt.figure(figsize=(8, 4))
plt.plot(k_values, scores_only, marker="o")
plt.title("K-means silhouette scores")
plt.xlabel("k")
plt.ylabel("Silhouette score")
plt.grid(True)
plt.show()

# Plot the best K-means clustering result
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels)
plt.title(f"K-means clustering with k = {best_k}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

print("\nCluster sizes for K-means:")
print(pd.Series(kmeans_labels).value_counts().sort_index())


# ============================================================
# PART 3 - EM CLUSTERING / GAUSSIAN MIXTURE MODEL
# Exercise 10.1(c)
# ============================================================

print("\n=== PART 3: EM CLUSTERING / GAUSSIAN MIXTURE MODEL ===")

gmm_results = []

for k in k_values:
    gmm = GaussianMixture(n_components=k, random_state=42)
    gmm.fit(X_scaled)
    labels = gmm.predict(X_scaled)
    bic = gmm.bic(X_scaled)

    gmm_results.append([k, bic])

# Show the BIC scores
print("\nGMM BIC scores:")
for result in gmm_results:
    print("k =", result[0], "| BIC =", round(result[1], 2))

# Find the best k (lowest BIC)
best_gmm_k = min(gmm_results, key=lambda x: x[1])[0]
print("\nBest k for EM / GMM:", best_gmm_k)

# Train the best GMM again
best_gmm = GaussianMixture(n_components=best_gmm_k, random_state=42)
best_gmm.fit(X_scaled)
gmm_labels = best_gmm.predict(X_scaled)

# Plot BIC values
bic_values = [result[1] for result in gmm_results]

plt.figure(figsize=(8, 4))
plt.plot(k_values, bic_values, marker="o")
plt.title("GMM BIC values")
plt.xlabel("k")
plt.ylabel("BIC")
plt.grid(True)
plt.show()

# Plot the best GMM clustering result
plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=gmm_labels)
plt.title(f"EM / GMM clustering with k = {best_gmm_k}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

print("\nCluster sizes for EM / GMM:")
print(pd.Series(gmm_labels).value_counts().sort_index())


# ============================================================
# PART 4 - DBSCAN
# Exercise 10.1(d)
# ============================================================

print("\n=== PART 4: DBSCAN ===")

# Step 1: Make a k-distance plot to help choose eps
neighbors = NearestNeighbors(n_neighbors=4)
neighbors_fit = neighbors.fit(X_scaled)
distances, indices = neighbors_fit.kneighbors(X_scaled)

# Use the distance to the 4th nearest neighbor
k_distances = np.sort(distances[:, 3])

plt.figure(figsize=(8, 4))
plt.plot(k_distances)
plt.title("4-nearest neighbor distance plot")
plt.xlabel("Points sorted by distance")
plt.ylabel("Distance")
plt.grid(True)
plt.show()

print("\nLook at the plot and choose an eps value near the bend in the curve.")

# Step 2: Try a few simple DBSCAN settings
dbscan_settings = [
    [0.8, 4],
    [0.9, 4],
    [1.0, 4],
    [1.1, 4],
    [1.2, 4]
]

dbscan_results = []

for setting in dbscan_settings:
    eps = setting[0]
    min_samples = setting[1]

    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Count clusters (ignore noise = -1)
    unique_labels = set(labels)
    cluster_count = len(unique_labels - {-1})
    noise_count = list(labels).count(-1)

    # Calculate silhouette score only if possible
    if cluster_count >= 2:
        score = silhouette_score(X_scaled, labels)
    else:
        score = None

    dbscan_results.append([eps, min_samples, cluster_count, noise_count, score])

# Show DBSCAN results
print("\nDBSCAN results:")
for result in dbscan_results:
    eps = result[0]
    min_samples = result[1]
    clusters = result[2]
    noise = result[3]
    score = result[4]

    print("eps =", eps,
          "| min_samples =", min_samples,
          "| clusters =", clusters,
          "| noise points =", noise,
          "| silhouette =", score)

# Choose one DBSCAN model manually
chosen_eps = 1.0
chosen_min_samples = 4

best_dbscan = DBSCAN(eps=chosen_eps, min_samples=chosen_min_samples)
dbscan_labels = best_dbscan.fit_predict(X_scaled)

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=dbscan_labels)
plt.title(f"DBSCAN with eps = {chosen_eps}, min_samples = {chosen_min_samples}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

print("\nCluster sizes for DBSCAN:")
print(pd.Series(dbscan_labels).value_counts().sort_index())


# ============================================================
# PART 5 - HIERARCHICAL CLUSTERING
# Exercise 10.1(e)
# ============================================================

print("\n=== PART 5: HIERARCHICAL CLUSTERING ===")

hierarchical = AgglomerativeClustering(n_clusters=best_k)
hier_labels = hierarchical.fit_predict(X_scaled)

hier_score = silhouette_score(X_scaled, hier_labels)

print("\nHierarchical clustering silhouette score:", round(hier_score, 4))

plt.figure(figsize=(7, 5))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=hier_labels)
plt.title(f"Hierarchical clustering with k = {best_k}")
plt.xlabel("PCA 1")
plt.ylabel("PCA 2")
plt.show()

print("\nCluster sizes for Hierarchical clustering:")
print(pd.Series(hier_labels).value_counts().sort_index())


# ============================================================
# PART 6 - FINAL COMPARISON
# Exercise 10.1(f)
# ============================================================

print("\n=== PART 6: FINAL COMPARISON ===")

# Best K-means silhouette score
best_kmeans_score = max(kmeans_results, key=lambda x: x[1])[1]

# Best GMM BIC
best_gmm_bic = min(gmm_results, key=lambda x: x[1])[1]

print("\nSummary:")
print("Best K-means k =", best_k)
print("Best K-means silhouette score =", round(best_kmeans_score, 4))

print("\nBest EM / GMM k =", best_gmm_k)
print("Best EM / GMM BIC =", round(best_gmm_bic, 2))

print("\nChosen DBSCAN eps =", chosen_eps)
print("Chosen DBSCAN min_samples =", chosen_min_samples)

print("\nHierarchical clustering k =", best_k)
print("Hierarchical silhouette score =", round(hier_score, 4))

print("\nFinal note:")
print("Use these results to discuss which clustering method fits the Seeds dataset best.")