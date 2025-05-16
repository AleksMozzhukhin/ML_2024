import numpy as np
import typing
from collections import defaultdict


def kfold_split(num_objects: int,
                num_folds: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Split [0, 1, ..., num_objects - 1] into equal num_folds folds
       (last fold can be longer) and returns num_folds train-val
       pairs of indexes.

    Parameters:
    num_objects: number of objects in train set
    num_folds: number of folds for cross-validation split

    Returns:
    list of length num_folds, where i-th element of list
    contains tuple of 2 numpy arrays, he 1st numpy array
    contains all indexes without i-th fold while the 2nd
    one contains i-th fold
    """
    answer = []
    a = [i for i in range(num_objects)]
    fld = num_objects // num_folds
    if num_objects % num_folds != 0:
        split_ratio = [fld for _ in range(0, (num_folds-1)*fld, fld)]
        split_ratio.append(num_objects-fld)
    else:
        split_ratio = [fld for _ in range(num_folds)]
    index = 0
    for x in split_ratio:
        tmp = set(a[index:index + x])
        answer.append((np.array([x for x in a if x not in tmp]), np.array(a[index:index + x])))
        index += x
    return answer


def knn_cv_score(X: np.ndarray, y: np.ndarray, parameters: dict[str, list],
                 score_function: callable,
                 folds: list[tuple[np.ndarray, np.ndarray]],
                 knn_class: object) -> dict[str, float]:
    """Takes train data, counts cross-validation score over
    grid of parameters (all possible parameters combinations)

    Parameters:
    X: train set
    y: train labels
    parameters: dict with keys from
        {n_neighbors, metrics, weights, normalizers}, values of type list,
        parameters['normalizers'] contains tuples (normalizer, normalizer_name)
        see parameters example in your jupyter notebook

    score_function: function with input (y_true, y_predict)
        which outputs score metric
    folds: output of kfold_split
    knn_class: class of knn model to fit

    Returns:
    dict: key - tuple of (normalizer_name, n_neighbors, metric, weight),
    value - mean score over all folds
    """
    ans = {}
    for i in parameters['normalizers']:
        # if i[0] == None:
        #     X_norm = X
        # else:
        #     i[0].fit(X)
        #     X_norm = i[0].transform(X)
        for j in parameters['n_neighbors']:
            for k in parameters['metrics']:
                for p in parameters['weights']:
                    cur_iteration = []
                    for f, g in folds:
                        X_train, X_test = X[f], X[g]
                        if i[0] is not None:
                            i[0].fit(X_train)
                            X_train = i[0].transform(X_train)
                            X_test = i[0].transform(X_test)
                        knn = knn_class(n_neighbors=j, weights=p, metric=k)
                        knn.fit(X_train, y[f])
                        cur_iteration.append(score_function(y[g], knn.predict(X_test)))
                    ans[(i[1], j, k, p)] = np.mean(np.array(cur_iteration))
    return ans
