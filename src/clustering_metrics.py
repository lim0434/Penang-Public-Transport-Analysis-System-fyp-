from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


SCORE_COLUMNS = (
    "stop_density_score",
    "population_coverage_score",
    "walking_distance_score",
    "destination_access_score",
)


def _silhouette_from_labels(features, labels):
    """Calculate mean silhouette using Euclidean distances and NumPy."""
    labels = np.asarray(labels)
    groups = np.unique(labels)
    sample_count = len(labels)
    if len(groups) < 2 or len(groups) >= sample_count:
        return float("nan")

    differences = features[:, None, :] - features[None, :, :]
    distances = np.sqrt(np.einsum("ijk,ijk->ij", differences, differences))
    sample_scores = np.zeros(sample_count, dtype=float)

    for index, label in enumerate(labels):
        in_group = labels == label
        in_group[index] = False
        if not in_group.any():
            sample_scores[index] = 0.0
            continue

        mean_own_distance = float(distances[index, in_group].mean())
        other_group_distances = [
            float(distances[index, labels == other].mean())
            for other in groups
            if other != label
        ]
        nearest_other_distance = min(other_group_distances)
        scale = max(mean_own_distance, nearest_other_distance)
        sample_scores[index] = (
            (nearest_other_distance - mean_own_distance) / scale
            if scale > 0
            else 0.0
        )

    return float(sample_scores.mean())


def calculate_silhouette_scores(frame, group_counts=(2, 3, 4)):
    """Score candidate K-means counts using the current complete TAI records."""
    if not set(SCORE_COLUMNS).issubset(frame.columns):
        return {}

    features = (
        frame.loc[:, SCORE_COLUMNS]
        .apply(pd.to_numeric, errors="coerce")
        .dropna()
        .to_numpy(dtype=float)
    )
    if len(features) < 3:
        return {}

    results = {}
    for group_count in group_counts:
        if group_count < 2 or group_count >= len(features):
            continue
        model = KMeans(n_clusters=group_count, random_state=42, n_init=20)
        labels = model.fit_predict(features)
        score = _silhouette_from_labels(features, labels)
        if np.isfinite(score):
            results[int(group_count)] = score
    return results
