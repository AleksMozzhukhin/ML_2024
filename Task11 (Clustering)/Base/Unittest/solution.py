import numpy as np

import sklearn
import sklearn.metrics


def silhouette_score(x, labels):
    '''
    Вычисляет коэффициент силуэта для выборки.

    :param np.ndarray x: Непустой двумерный массив векторов-признаков (n_samples, n_features)
    :param np.ndarray labels: Непустой одномерный массив меток объектов (n_samples,)
    :return float: Коэффициент силуэта для выборки x с метками labels
    '''
    n_samples = len(labels)
    distances = sklearn.metrics.pairwise_distances(x)
    unique_labels, labels_indices, cluster_sizes = np.unique(labels, return_inverse=True, return_counts=True)
    n_labels = len(unique_labels)

    if n_labels <= 1:
        return 0.0

    sum_dists_to_clusters = np.zeros((n_samples, n_labels), dtype=np.float64)
    cluster_sizes_per_sample = np.zeros(n_samples, dtype=np.float64)
    masks_matrix = np.zeros((n_samples, n_labels), dtype=bool)

    for k in range(n_labels):
        mask_k = (labels_indices == k)
        masks_matrix[:, k] = mask_k
        sum_dists_to_clusters[:, k] = np.sum(distances[:, mask_k], axis=1, dtype=np.float64)
        cluster_sizes_per_sample[mask_k] = cluster_sizes[k]
    s_all = sum_dists_to_clusters[masks_matrix]
    multi_point_cluster_mask = (cluster_sizes_per_sample > 1)
    s_all[~multi_point_cluster_mask] = 0
    s_all[multi_point_cluster_mask] /= (cluster_sizes_per_sample[multi_point_cluster_mask] - 1)
    A = sum_dists_to_clusters / cluster_sizes  # cluster_sizes > 0
    other_cluster_avg_dists = A[~masks_matrix]
    d_all = np.min(other_cluster_avg_dists.reshape(n_samples, n_labels - 1), axis=1)
    d_all[~multi_point_cluster_mask] = 0
    max_sd = np.maximum(s_all, d_all)
    sil_i = np.zeros(n_samples, dtype=np.float64)
    np.divide(d_all - s_all, max_sd, out=sil_i, where=(max_sd != 0))

    sil_score = np.mean(sil_i)

    return sil_score


def bcubed_score(true_labels, predicted_labels):
    '''
    :param np.ndarray true_labels: Непустой одномерный массив меток объектов
    :param np.ndarray predicted_labels: Непустой одномерный массив меток объектов
    :return float: B-Cubed для объектов с истинными метками true_labels и предсказанными метками predicted_labels
    '''
    n_samples = len(true_labels)
    if n_samples == 0:
        return 0.0
    C_eq = (predicted_labels[:, None] == predicted_labels)
    L_eq = (true_labels[:, None] == true_labels)
    Correctness = C_eq & L_eq
    sum_correct_precision = np.sum(Correctness * C_eq, axis=1, dtype=np.float64)
    count_precision_pairs = np.sum(C_eq, axis=1, dtype=np.float64)
    precision_per_sample = sum_correct_precision / count_precision_pairs
    avg_precision = np.mean(precision_per_sample)
    sum_correct_recall = np.sum(Correctness * L_eq, axis=1, dtype=np.float64)
    count_recall_pairs = np.sum(L_eq, axis=1, dtype=np.float64)
    recall_per_sample = sum_correct_recall / count_recall_pairs
    avg_recall = np.mean(recall_per_sample)
    numerator = 2 * avg_precision * avg_recall
    denominator = avg_precision + avg_recall
    if denominator == 0:
        bcubed_f1 = 0.0
    else:
        bcubed_f1 = numerator / denominator

    return bcubed_f1
