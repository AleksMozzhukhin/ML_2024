import numpy as np


def are_multisets_equal(x: np.ndarray, y: np.ndarray) -> bool:
    return np.array_equal(np.sort(x), np.sort(y))


def max_prod_mod_3(x: np.ndarray) -> int:
    """
    Вернуть максимальное прозведение соседних элементов в массиве x, 
    таких что хотя бы один множитель в произведении делится на 3.
    Если таких произведений нет, то вернуть -1.
    """
    answer = (x[:-1] * x[1:])[(x[:-1] % 3 == 0) | (x[1:] % 3 == 0)]
    return np.max(answer) if (answer.size > 0) else -1


def convert_image(image: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Сложить каналы изображения с указанными весами.
    """
    image_times_whights = [image[:, :, i] * weights[i] for i in range(weights.shape[0])]
    return np.sum(image_times_whights, axis=0)


def rle_scalar(x: np.ndarray, y: np.ndarray) -> int:
    """
    Найти скалярное произведение между векторами x и y, заданными в формате RLE.
    В случае несовпадения длин векторов вернуть -1.
    """
    x_1 = np.repeat(x[:, 0], x[:, 1])
    y_1 = np.repeat(y[:, 0], y[:, 1])
    if x_1.shape != y_1.shape:
        return -1
    else:
        return np.dot(x_1, y_1)


def cosine_distance(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    with np.errstate(divide='ignore', invalid='ignore'):
        ans = np.dot(X, Y.T) / np.outer(np.linalg.norm(X, axis=1), np.linalg.norm(Y, axis=1))
    return np.nan_to_num(ans, nan=1)
