from typing import List
from copy import deepcopy


def get_part_of_array(X: List[List[float]]) -> List[List[float]]:
    ans = []
    for i in range(0, len(X), 4):
        ans.append([])
        for j in range(120, 500, 5):
            ans[len(ans) - 1].append(X[i][j])
    return ans


def sum_non_neg_diag(X: List[List[int]]) -> int:
    ans = 0
    flag = True
    for i in range(min(len(X), len(X[0]))):
        if X[i][i] >= 0:
            flag = False
            ans += X[i][i]
    if flag:
        return -1
    else:
        return ans


def replace_values(X: List[List[float]]) -> List[List[float]]:
    ans = deepcopy(X)
    for i in range(len(X[0])):
        m = 0
        for j in range(len(X)):
            m += X[j][i]
        m /= len(X)
        for j in range(len(X)):
            if X[j][i] < 0.25 * m or X[j][i] > 1.5 * m:
                ans[j][i] = -1
    return ans
