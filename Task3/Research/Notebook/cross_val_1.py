import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    folds = []
    n = [i for i in range(num_objects)]
    fold_size = num_objects // num_folds
    if num_objects % num_folds != 0:
        split_r = [fold_size for _ in range(0, (num_folds-1)*fold_size, fold_size)]
        split_r.append(num_objects-fold_size)
    else:
        split_r = [fold_size for _ in range(num_folds)]
    index = 0
    for x in split_r:
        tmp = set(n[index:index + x])
        folds.append((np.array([x for x in n if x not in tmp]), np.array(n[index:index + x])))
        index += x
    return folds


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    res = {}
    for normalizer in parameters['normalizers']:
        for n_neighbors in parameters['n_neighbors']:
            for metric in parameters['metrics']:
                for weight in parameters['weights']:
                    fold_scores = []
                    for train_indices, val_indices in folds:
                        X_train = X[train_indices]
                        X_val = X[val_indices]
                        if normalizer[0] is not None:
                            normalizer[0].fit(X_train)
                            y_train = y[train_indices]
                            y_val = y[val_indices]
                            X_train = normalizer[0].transform(X_train)
                            X_val = normalizer[0].transform(X_val)
                        knn = knn_class(n_neighbors=n_neighbors, weights=weight, metric=metric)
                        knn.fit(X_train, y_train)
                        fold_scores.append(score_function(y_val, knn.predict(X_val)))
                    res[(normalizer[1], n_neighbors, metric, weight)] = np.mean(np.array(fold_scores))
    return res
