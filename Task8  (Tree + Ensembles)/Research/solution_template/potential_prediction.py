import os
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.pipeline import Pipeline
from scipy.sparse import csr_matrix


def centring_data(arr):
    non_zero_coords = np.argwhere(arr != 0)
    if non_zero_coords.size == 0:
        return arr

    min_coords = non_zero_coords.min(axis=0)
    max_coords = non_zero_coords.max(axis=0)
    center_non_zero = (min_coords + max_coords) // 2
    center_arr = np.array(arr.shape) // 2
    shift = center_arr - center_non_zero
    result = np.zeros_like(arr)
    new_coords = non_zero_coords + shift

    for coord, new_coord in zip(non_zero_coords, new_coords):
        if all(0 <= n < dim for n, dim in zip(new_coord, arr.shape)):
            result[tuple(new_coord)] = arr[tuple(coord)]
    return result


class PotentialTransformer:

    def fit(self, x, y):
        return self

    def fit_transform(self, x, y):
        return self.transform(x)

    def transform(self, x):
        X = np.array(x)
        n_samples = X.shape[0]
        X_transformed = []
        for matrix in X:
            matrix_adj = matrix - 20
            centered = centring_data(matrix_adj)
            X_transformed.append(centered.flatten())
        X_transformed = np.array(X_transformed)
        return X_transformed


def load_dataset(data_dir):
    files, X, Y = [], [], []
    for file in sorted(os.listdir(data_dir)):
        potential = np.load(os.path.join(data_dir, file))
        files.append(file)
        X.append(potential["data"])
        Y.append(potential["target"])
    return files, np.array(X), np.array(Y)


def train_model_and_predict(train_dir, test_dir):
    _, X_train, Y_train = load_dataset(train_dir)
    test_files, X_test, _ = load_dataset(test_dir)
    # it's suggested to modify only the following line of this function
    X_train = csr_matrix(PotentialTransformer().fit_transform(X_train, 0))
    X_test = csr_matrix(PotentialTransformer().fit_transform(X_test, 0))
    regressor = Pipeline([
        ('pca', TruncatedSVD(n_components=10)),
        ('decision_tree', ExtraTreesRegressor(n_estimators=900, criterion="friedman_mse", max_depth=20, n_jobs=-1))
    ])
    regressor.fit(X_train, Y_train)
    predictions = regressor.predict(X_test)
    return {file: value for file, value in zip(test_files, predictions)}
