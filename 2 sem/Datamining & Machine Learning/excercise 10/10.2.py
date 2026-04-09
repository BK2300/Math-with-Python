import pandas as pd

# Points from the figure
points = {
    "A": (1, 1),
    "B": (2, 1),
    "C": (1, 2),
    "D": (2, 2),
    "E": (3, 5),
    "F": (3, 9),
    "G": (3, 10),
    "H": (4, 10),
    "I": (4, 11),
    "J": (5, 10),
    "K": (7, 10),
    "L": (10, 9),
    "M": (10, 6),
    "N": (9, 5),
    "O": (10, 5),
    "P": (11, 5),
    "Q": (9, 4),
    "R": (10, 4),
    "S": (11, 4),
    "T": (10, 3)
}

# Manhattan distance
def manhattan_distance(p1, p2):
    x1, y1 = points[p1]
    x2, y2 = points[p2]
    return abs(x1 - x2) + abs(y1 - y2)

# Get all distances from one point to all other points
def get_distances(point_name):
    distances = []

    for other_point in points:
        if other_point != point_name:
            d = manhattan_distance(point_name, other_point)
            distances.append((other_point, d))

    distances.sort(key=lambda x: (x[1], x[0]))
    return distances

# Distance to the k-th nearest neighbor
def k_distance(point_name, k):
    distances = get_distances(point_name)
    return distances[k - 1][1]

# All neighbors with distance <= k-distance
def k_neighborhood(point_name, k):
    kd = k_distance(point_name, k)
    distances = get_distances(point_name)

    neighbors = []
    for other_point, d in distances:
        if d <= kd:
            neighbors.append((other_point, d))

    return neighbors

# Average distance to all neighbors in the k-neighborhood
def aggregated_knn_distance(point_name, k):
    neighbors = k_neighborhood(point_name, k)
    total = 0

    for other_point, d in neighbors:
        total += d

    return total / len(neighbors)

# Reachability distance
def reachability_distance(p, o, k):
    dist_po = manhattan_distance(p, o)
    k_dist_o = k_distance(o, k)
    return max(k_dist_o, dist_po)

# Local reachability density
def local_reachability_density(point_name, k):
    neighbors = k_neighborhood(point_name, k)
    total = 0

    for other_point, d in neighbors:
        total += reachability_distance(point_name, other_point, k)

    return len(neighbors) / total

# LOF
def lof(point_name, k):
    neighbors = k_neighborhood(point_name, k)
    lrd_p = local_reachability_density(point_name, k)
    total = 0

    for other_point, d in neighbors:
        lrd_o = local_reachability_density(other_point, k)
        total += lrd_o / lrd_p

    return total / len(neighbors)

# LOF for E, K, O with k = 2
print("\nLOF with k = 2")
for point_name in ["E", "K", "O"]:
    print(point_name, "->", round(lof(point_name, 2), 4))

# LOF for E, K, O with k = 4
print("\nLOF with k = 4")
for point_name in ["E", "K", "O"]:
    print(point_name, "->", round(lof(point_name, 4), 4))

# kNN distance for all points with k = 2
print("\nkNN distance for all points with k = 2")
rows_k2 = []

for point_name in points:
    rows_k2.append({
        "Point": point_name,
        "kNN_distance_k2": k_distance(point_name, 2)
    })

df_k2 = pd.DataFrame(rows_k2)
df_k2 = df_k2.sort_values("Point")
print(df_k2.to_string(index=False))

# kNN distance for all points with k = 4
print("\nkNN distance for all points with k = 4")
rows_k4 = []

for point_name in points:
    rows_k4.append({
        "Point": point_name,
        "kNN_distance_k4": k_distance(point_name, 4)
    })

df_k4 = pd.DataFrame(rows_k4)
df_k4 = df_k4.sort_values("Point")
print(df_k4.to_string(index=False))

# Aggregated kNN distance for all points with k = 2
print("\nAggregated kNN distance for all points with k = 2")
agg_rows_k2 = []

for point_name in points:
    agg_rows_k2.append({
        "Point": point_name,
        "aggregated_kNN_k2": round(aggregated_knn_distance(point_name, 2), 4)
    })

agg_df_k2 = pd.DataFrame(agg_rows_k2)
agg_df_k2 = agg_df_k2.sort_values("Point")
print(agg_df_k2.to_string(index=False))

# Aggregated kNN distance for all points with k = 4
print("\nAggregated kNN distance for all points with k = 4")
agg_rows_k4 = []

for point_name in points:
    agg_rows_k4.append({
        "Point": point_name,
        "aggregated_kNN_k4": round(aggregated_knn_distance(point_name, 4), 4)
    })

agg_df_k4 = pd.DataFrame(agg_rows_k4)
agg_df_k4 = agg_df_k4.sort_values("Point")
print(agg_df_k4.to_string(index=False))