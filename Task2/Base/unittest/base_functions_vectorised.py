import numpy as np

def get_part_of_array(X: np.ndarray) -> np.ndarray:
    return X[::4, 120:500:5]


def sum_non_neg_diag(X: np.ndarray) -> int:
    answer=X.diagonal()[X.diagonal()>=0]
    return answer.sum() if (np.size(answer) != 0) else -1



def replace_values(X: np.ndarray) -> np.ndarray:
    m = np.mean(X, 0)
    new_x=np.where((X<0.25*m) | (X>1.5*m), -1, X)
    return new_x

