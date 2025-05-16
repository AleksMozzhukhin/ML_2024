import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> (list)[tuple[np.ndarray, np.ndarray]]:
    count_fold = num_objects // num_folds
    res_list = []
    i_indexes = np.arange(num_objects)
    for i in range(num_folds):
        end_index = count_fold * (i + 1)
        if i == num_folds - 1:
            end_index = num_objects
        i_fold = i_indexes[i * count_fold:end_index]
        i_rest = np.concatenate(
            [i_indexes[:i * count_fold], i_indexes[end_index:]])
        res_list.append((i_rest, i_fold))

    return res_list


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    res = {}
    for neigh in parameters['n_neighbors']:
        for metr in parameters['metrics']:
            for weigh in parameters['weights']:
                for normal in parameters['normalizers']:
                    score = 0
                    for i_train, i_fold in folds:
                        X_train, X_test = X[i_train], X[i_fold]
                        y_train, y_test = y[i_train], y[i_fold]
                        if normal[0] is not None:
                            scaler = normal[0]
                            scaler.fit(X_train)
                            X_train = scaler.transform(X_train)
                            X_test = scaler.transform(X_test)
                        knn = knn_class(n_neighbors=neigh,
                                        metric=metr, weights=weigh)
                        knn.fit(X_train, y_train)
                        y_predict = knn.predict(X_test)
                        score += score_function(y_test, y_predict)
                    score /= len(folds)
                    res[(normal[1], neigh, metr, weigh)] = score

    return res
