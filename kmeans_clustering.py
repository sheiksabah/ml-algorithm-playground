import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from scipy.optimize import linear_sum_assignment

iris = load_iris()
X = iris.data
y = iris.target

print("Dataset shape:", X.shape)
print("Features:", iris.feature_names)
print("Actual classes:", iris.target_names)

kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X)
centroids = kmeans.cluster_centers_

print("\nK-means clustering results")
print("Number of clusters:", kmeans.n_clusters)

print("\nRMSE for each cluster:")

for cluster_id in range(3):
    cluster_points = X[cluster_labels == cluster_id]
    centroid = centroids[cluster_id]

    squared_distances = np.sum((cluster_points - centroid) ** 2, axis=1)
    rmse = np.sqrt(np.mean(squared_distances))

    print(f"Cluster {cluster_id}: {rmse:.4f} "
          f"({len(cluster_points)} samples)")

print("\nCluster composition by actual class:")
for cluster_id in range(3):
    actual_classes = y[cluster_labels == cluster_id]
    counts = np.bincount(actual_classes, minlength=3)

    print(
        f"Cluster {cluster_id}: "
        f"setosa={counts[0]}, "
        f"versicolor={counts[1]}, "
        f"virginica={counts[2]}"
    )

contingency = np.zeros((3, 3), dtype=int)

for cluster_id in range(3):
    for class_id in range(3):
        contingency[cluster_id, class_id] = np.sum(
            (cluster_labels == cluster_id) & (y == class_id)
        )

rows, cols = linear_sum_assignment(-contingency)
matched_correct = contingency[rows, cols].sum()
matched_accuracy = matched_correct / len(y)

print("\nCluster quality comparison")
print("Samples matched to their majority class:", matched_correct)
print(f"Matched clustering accuracy: {matched_accuracy:.2%}")
