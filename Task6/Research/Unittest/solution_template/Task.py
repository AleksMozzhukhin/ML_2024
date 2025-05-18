import numpy as np


class Preprocessor:

    def __init__(self):
        pass

    def fit(self, X, Y=None):
        pass

    def transform(self, X):
        pass

    def fit_transform(self, X, Y=None):
        pass


class MyOneHotEncoder(Preprocessor):

    def __init__(self, dtype=np.float64):
        super(Preprocessor).__init__()
        self.dtype = dtype
        self.types = {}

    def fit(self, X, Y=None):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: unused
        """
        for col in X.columns:
            self.types[col] = sorted(X[col].unique())

    def transform(self, X):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        returns: transformed objects, numpy-array, shape [n_objects, |f1| + |f2| + ...]
        """
        n_objects = X.shape[0]
        n_features = sum(len(categories) for categories in self.types.values())
        result = np.zeros((n_objects, n_features))

        shift_indexes = 0
        for column, categories in self.types.items():
            for i, category in enumerate(categories):
                indices = np.where(X[column] == category)
                result[indices, shift_indexes + i] = 1
            shift_indexes += len(categories)

        return result

    def fit_transform(self, X, Y=None):
        self.fit(X)
        return self.transform(X)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


class SimpleCounterEncoder:

    def __init__(self, dtype=np.float64):
        self.dtype = dtype
        self.count = {}

    def fit(self, X, Y):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        """
        for col in X.columns:
            unique_values = X[col].unique()
            self.count[col] = {}
            for value in unique_values:
                indexes = X[col] == value
                self.count[col][value] = [Y[indexes].mean(), np.mean(indexes)]

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        n_obj, n_feach = X.shape
        result = np.zeros((n_obj, 3 * n_feach))

        for i, col in enumerate(X.columns):
            for j in range(n_obj):
                value = X.iloc[j, i]
                mean_expected, frac = self.count[col][value]
                result[j, 3 * i] = mean_expected
                result[j, 3 * i + 1] = frac
                result[j, 3 * i + 2] = (mean_expected + a) / (frac + b)

        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)

    def get_params(self, deep=True):
        return {"dtype": self.dtype}


def group_k_fold(size, n_splits=3, seed=1):
    idx = np.arange(size)
    np.random.seed(seed)
    idx = np.random.permutation(idx)
    n_ = size // n_splits
    for i in range(n_splits - 1):
        yield idx[i * n_: (i + 1) * n_], np.hstack((idx[:i * n_], idx[(i + 1) * n_:]))
    yield idx[(n_splits - 1) * n_:], idx[:(n_splits - 1) * n_]


class FoldCounters:

    def __init__(self, n_folds=3, dtype=np.float64):
        self.dtype = dtype
        self.n_folds = n_folds
        self.fold_count = []

    def fit(self, X, Y, seed=1):
        """
        param X: training objects, pandas-dataframe, shape [n_objects, n_features]
        param Y: target for training objects, pandas-series, shape [n_objects,]
        param seed: random seed, int
        """
        for fold_idx, rest_idx in group_k_fold(X.shape[0], self.n_folds, seed):
            fold_counter = {}
            X_fold, Y_fold = X.iloc[rest_idx], Y.iloc[rest_idx]
            for column in X.columns:
                unique_val = X_fold[column].unique()
                fold_counter[column] = {}
                for value in unique_val:
                    fold_counter[column][value] = [Y_fold[X_fold[column] == value].mean(),
                                                   np.mean(X_fold[column] == value)]
            self.fold_count.append((fold_idx, fold_counter))

    def transform(self, X, a=1e-5, b=1e-5):
        """
        param X: objects to transform, pandas-dataframe, shape [n_objects, n_features]
        param a: constant for counters, float
        param b: constant for counters, float
        returns: transformed objects, numpy-array, shape [n_objects, 3 * n_features]
        """
        n_obj, n_feach = X.shape
        result = np.zeros((n_obj, 3 * n_feach))
        for fold_idx, fold_count in self.fold_count:
            for i, column in enumerate(X.columns):
                for j in fold_idx:
                    value = X.iloc[j, i]
                    mean_expected, fraction = fold_count[column][value]
                    result[j, 3 * i] = mean_expected
                    result[j, 3 * i + 1] = fraction
                    result[j, 3 * i + 2] = (mean_expected + a) / (fraction + b)
        return result

    def fit_transform(self, X, Y, a=1e-5, b=1e-5):
        self.fit(X, Y)
        return self.transform(X, a, b)


def weights(x, y):
    """
    param x: training set of one feature, numpy-array, shape [n_objects,]
    param y: target for training objects, numpy-array, shape [n_objects,]
    returns: optimal weights, numpy-array, shape [|x unique values|,]
    """
    unique_values = np.unique(x)
    enc_x = np.eye(unique_values.shape[0])[x]
    weight = np.zeros(enc_x.shape[1])
    lr = 1e-2

    for i in range(1000):
        p = np.dot(enc_x, weight)
        grad = np.dot(enc_x.T, (p - y))
        weight -= grad * lr

    return weight
