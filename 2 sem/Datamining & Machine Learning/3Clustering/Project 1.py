import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

# =========================
# SETTINGS - you can change these
# =========================
DATA_FILE = "D31.txt"
OUTPUT_DIR = "clustering_output"

# If True, clustering is done on standardized data.
# Good for many datasets. For D31 you can try both True and False.
USE_STANDARD_SCALER = False

# If Plotly is installed, interactive HTML plots will also be saved.
MAKE_INTERACTIVE_HTML = True

# Algorithms / parameter grids
KMEANS_KS = [25, 31, 35]

DBSCAN_CONFIGS = [
    {"eps": 0.30, "min_samples": 5},
    {"eps": 0.40, "min_samples": 5},
    {"eps": 0.50, "min_samples": 5},
    {"eps": 0.60, "min_samples": 5},
    {"eps": 0.80, "min_samples": 5},
    {"eps": 1.00, "min_samples": 5},
]

AGGLO_CONFIGS = [
    {"linkage": "ward", "n_clusters": 31},
    {"linkage": "complete", "n_clusters": 31},
    {"linkage": "average", "n_clusters": 31},
    {"linkage": "single", "n_clusters": 31},
    {"linkage": "complete", "n_clusters": 25},
    {"linkage": "complete", "n_clusters": 35},
]


# =========================
# HELPERS
# =========================
def ensure_output_dir(path: str):
    os.makedirs(path, exist_ok=True)


def load_dataset(file_path: str):
    """
    Expected format:
    col1 = x
    col2 = y
    col3 = true label
    """
    data = np.loadtxt(file_path)
    if data.shape[1] < 3:
        raise ValueError("Dataset must have at least 3 columns: x, y, label")

    X = data[:, :2]
    y_true = data[:, 2].astype(int)
    return X, y_true


def get_model_data(X: np.ndarray, use_scaler: bool):
    if not use_scaler:
        return X.copy(), None
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler


def count_clusters(labels: np.ndarray) -> int:
    unique = set(labels)
    if -1 in unique:
        unique.remove(-1)
    return len(unique)


def count_noise(labels: np.ndarray) -> int:
    return int(np.sum(labels == -1))


def safe_silhouette(X_model: np.ndarray, labels: np.ndarray):
    """
    Silhouette needs at least 2 non-noise clusters.
    For DBSCAN, silhouette is computed on non-noise points only.
    """
    mask = labels != -1
    labels_no_noise = labels[mask]
    X_no_noise = X_model[mask]

    unique_clusters = np.unique(labels_no_noise)

    if len(unique_clusters) < 2:
        return np.nan

    if len(X_no_noise) <= len(unique_clusters):
        return np.nan

    try:
        return float(silhouette_score(X_no_noise, labels_no_noise))
    except Exception:
        return np.nan


def safe_ari(y_true: np.ndarray, labels: np.ndarray):
    """
    ARI can compare found labels with ground truth labels.
    """
    try:
        return float(adjusted_rand_score(y_true, labels))
    except Exception:
        return np.nan


def plot_clusters_static(X_plot: np.ndarray, labels: np.ndarray, title: str, save_path: str):
    plt.figure(figsize=(7, 6))

    noise_mask = labels == -1
    cluster_mask = ~noise_mask

    # Plot normal clusters
    if np.any(cluster_mask):
        plt.scatter(
            X_plot[cluster_mask, 0],
            X_plot[cluster_mask, 1],
            c=labels[cluster_mask],
            cmap="nipy_spectral",
            s=12,
            alpha=0.9
        )

    # Plot noise as black X
    if np.any(noise_mask):
        plt.scatter(
            X_plot[noise_mask, 0],
            X_plot[noise_mask, 1],
            c="black",
            marker="x",
            s=20,
            alpha=0.9,
            label="noise"
        )
        plt.legend()

    plt.title(title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def plot_ground_truth_static(X_plot: np.ndarray, y_true: np.ndarray, save_path: str):
    plt.figure(figsize=(7, 6))
    plt.scatter(
        X_plot[:, 0],
        X_plot[:, 1],
        c=y_true,
        cmap="nipy_spectral",
        s=12,
        alpha=0.9
    )
    plt.title("Ground Truth Labels")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180)
    plt.close()


def save_interactive_plot(X_plot: np.ndarray, labels: np.ndarray, title: str, save_path: str):
    if not MAKE_INTERACTIVE_HTML:
        return

    try:
        import plotly.express as px
    except ImportError:
        return

    label_text = []
    for v in labels:
        if v == -1:
            label_text.append("noise")
        else:
            label_text.append(str(int(v)))

    df = pd.DataFrame({
        "x": X_plot[:, 0],
        "y": X_plot[:, 1],
        "cluster": label_text
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="cluster",
        title=title,
        opacity=0.85
    )
    fig.update_traces(marker={"size": 6})
    fig.write_html(save_path)


def save_ground_truth_interactive(X_plot: np.ndarray, y_true: np.ndarray, save_path: str):
    if not MAKE_INTERACTIVE_HTML:
        return

    try:
        import plotly.express as px
    except ImportError:
        return

    df = pd.DataFrame({
        "x": X_plot[:, 0],
        "y": X_plot[:, 1],
        "label": y_true.astype(str)
    })

    fig = px.scatter(
        df,
        x="x",
        y="y",
        color="label",
        title="Ground Truth Labels",
        opacity=0.85
    )
    fig.update_traces(marker={"size": 6})
    fig.write_html(save_path)


def evaluate_solution(
    X_plot: np.ndarray,
    X_model: np.ndarray,
    y_true: np.ndarray,
    labels: np.ndarray,
    algorithm: str,
    params: dict,
    output_dir: str
):
    n_clusters_found = count_clusters(labels)
    noise_points = count_noise(labels)
    silhouette = safe_silhouette(X_model, labels)
    ari = safe_ari(y_true, labels)

    # File-safe name
    file_name = algorithm + "_" + "_".join([f"{k}-{v}" for k, v in params.items()])
    file_name = file_name.replace(".", "_").replace(" ", "_").replace("/", "-")

    title = f"{algorithm} | {params} | clusters={n_clusters_found} | sil={silhouette:.3f} | ari={ari:.3f}"

    plot_clusters_static(
        X_plot,
        labels,
        title,
        os.path.join(output_dir, f"{file_name}.png")
    )

    save_interactive_plot(
        X_plot,
        labels,
        title,
        os.path.join(output_dir, f"{file_name}.html")
    )

    return {
        "algorithm": algorithm,
        "params": json.dumps(params),
        "n_clusters_found": n_clusters_found,
        "noise_points": noise_points,
        "silhouette": silhouette,
        "ari_vs_ground_truth": ari,
        "plot_file": f"{file_name}.png",
        "html_file": f"{file_name}.html" if MAKE_INTERACTIVE_HTML else ""
    }


# =========================
# MAIN
# =========================
def main():
    ensure_output_dir(OUTPUT_DIR)

    # Load data
    X_plot, y_true = load_dataset(DATA_FILE)
    X_model, scaler = get_model_data(X_plot, USE_STANDARD_SCALER)

    print("Data loaded successfully")
    print("Shape:", X_plot.shape)
    print("Unique true labels:", len(np.unique(y_true)))
    print("Using StandardScaler:", USE_STANDARD_SCALER)

    # Save ground truth plots
    gt_png = os.path.join(OUTPUT_DIR, "ground_truth.png")
    gt_html = os.path.join(OUTPUT_DIR, "ground_truth.html")
    plot_ground_truth_static(X_plot, y_true, gt_png)
    save_ground_truth_interactive(X_plot, y_true, gt_html)

    results = []

    # -------------------------
    # 1) KMeans
    # -------------------------
    for k in KMEANS_KS:
        model = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = model.fit_predict(X_model)

        result = evaluate_solution(
            X_plot=X_plot,
            X_model=X_model,
            y_true=y_true,
            labels=labels,
            algorithm="KMeans",
            params={"k": k},
            output_dir=OUTPUT_DIR
        )
        results.append(result)
        print(f"Finished KMeans with k={k}")

    # -------------------------
    # 2) DBSCAN
    # -------------------------
    for cfg in DBSCAN_CONFIGS:
        model = DBSCAN(eps=cfg["eps"], min_samples=cfg["min_samples"])
        labels = model.fit_predict(X_model)

        result = evaluate_solution(
            X_plot=X_plot,
            X_model=X_model,
            y_true=y_true,
            labels=labels,
            algorithm="DBSCAN",
            params=cfg,
            output_dir=OUTPUT_DIR
        )
        results.append(result)
        print(f"Finished DBSCAN with eps={cfg['eps']}, min_samples={cfg['min_samples']}")

    # -------------------------
    # 3) Agglomerative / Hierarchical
    # -------------------------
    for cfg in AGGLO_CONFIGS:
        model = AgglomerativeClustering(
            n_clusters=cfg["n_clusters"],
            linkage=cfg["linkage"]
        )
        labels = model.fit_predict(X_model)

        result = evaluate_solution(
            X_plot=X_plot,
            X_model=X_model,
            y_true=y_true,
            labels=labels,
            algorithm="Agglomerative",
            params=cfg,
            output_dir=OUTPUT_DIR
        )
        results.append(result)
        print(f"Finished Agglomerative with linkage={cfg['linkage']}, n_clusters={cfg['n_clusters']}")

    # -------------------------
    # Save summary
    # -------------------------
    df_results = pd.DataFrame(results)

    # Sort: best ARI first, then silhouette
    df_sorted = df_results.sort_values(
        by=["ari_vs_ground_truth", "silhouette"],
        ascending=False,
        na_position="last"
    ).reset_index(drop=True)

    csv_path = os.path.join(OUTPUT_DIR, "results_summary.csv")
    df_sorted.to_csv(csv_path, index=False)

    print("\nDone.")
    print(f"Saved results to: {OUTPUT_DIR}")
    print(f"Summary CSV: {csv_path}")

    print("\nTop 10 results:")
    print(df_sorted.head(10).to_string(index=False))

    # Best per algorithm
    print("\nBest result per algorithm:")
    for algo in df_sorted["algorithm"].unique():
        best_row = df_sorted[df_sorted["algorithm"] == algo].iloc[0]
        print("-" * 60)
        print(f"Algorithm: {algo}")
        print(best_row.to_string())


if __name__ == "__main__":
    main()