import numpy as np
from sklearn.svm import SVC


def train_svm_and_predict(train_features, train_target, test_features):
    """
    train_features: np.array, (num_elements_train x num_features) - train data description, the same features and the same order as in train data
    train_target: np.array, (num_elements_train) - train data target
    test_features: np.array, (num_elements_test x num_features) -- some test data, features are in the same order as train features

    return: np.array, (num_elements_test) - test data predicted target, 1d array
    """
    model = SVC(kernel='rbf', C=10, gamma=0.1, class_weight='balanced')
    model.fit(train_features[:, [3, 4]], train_target)
    test_target = model.predict(test_features[:, [3, 4]])
    return test_target
