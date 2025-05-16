from collections import Counter
from typing import List


def are_multisets_equal(x: List[int], y: List[int]) -> bool:
    x.sort()
    y.sort()
    if x==y:
        return True
    else:
        return False


def max_prod_mod_3(x: List[int]) -> int:
    max_prod=-1
    for i in range(len(x)-1):
        if x[i]%3==0 or x[i+1]%3==0:
            if max_prod<x[i]*x[i+1]:
                max_prod=x[i]*x[i+1]
    return max_prod


def convert_image(image: List[List[List[float]]], weights: List[float]) -> List[List[float]]:
    for j in range(len(weights)):
        for i in range(len(image)):
                for k in range(len(image[0])):
                    image[i][k][j]*=weights[j]

    ans=[]
    for i in range(len(image)):
        ans.append([])
        for j in range(len(image[0])):
            ans[i].append(sum(image[i][j]))
    return ans


def rle_scalar(x: List[List[int]], y:  List[List[int]]) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_1 = []
    y_1 = []
    for i in range(len(x)):
        x_1.extend([x[i][0]] * x[i][1])
    for i in range(len(y)):
        y_1.extend([y[i][0]] * y[i][1])
    if len(x_1) != len(y_1):
        return -1
    else:
        return sum(x_1[i] * y_1[i] for i in range(len(x_1)))


def cosine_distance(X: List[List[float]], Y: List[List[float]]) -> List[List[float]]:
    """
    Вычислить матрицу косинусных расстояний между объектами X и Y. 
    В случае равенства хотя бы одно из двух векторов 0, косинусное расстояние считать равным 1.
    """
    ans = []
    for i in range(len(X)):
        ans.append([])
        x_normalized=sum(tmp**2 for tmp in X[i])**0.5
        for j in range(len(Y)):
            y_normalized = sum(tmp**2 for tmp in Y[j])**0.5
            if x_normalized==0 or y_normalized==0:
                ans[i].append(float(1))
            else:
                ans[i].append(sum(X[i][k] * Y[j][k] for k in range(len(X[0])))/(x_normalized*y_normalized))
    return ans