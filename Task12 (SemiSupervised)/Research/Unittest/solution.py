import numpy as np

import sklearn
from sklearn.cluster import KMeans


class KMeansClassifier(sklearn.base.BaseEstimator):
    def __init__(self, n_clusters):
        """
        :param int n_clusters: Число кластеров которых нужно выделить в обучающей выборке с помощью алгоритма кластеризации
        """
        super().__init__()
        if n_clusters < 1:
            raise ValueError("n_clusters must be positive")
        self.n_clusters = int(n_clusters)
        self.kmeans = None
        self.cluster_to_class_ = None

    def fit(self, data, labels):
        """
            Функция обучает кластеризатор KMeans с заданным числом кластеров, а затем с помощью
        self._best_fit_classification восстанавливает разметку объектов

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов обучающей выборки
        :param np.ndarray labels: Непустой одномерный массив. Разметка обучающей выборки. Неразмеченные объекты имеют метку -1.
            Размеченные объекты могут иметь произвольную неотрицательную метку. Существует хотя бы один размеченный объект
        :return KMeansClassifier
        """
        data = np.asarray(data)
        labels = np.asarray(labels)

        self.kmeans = KMeans(n_clusters=self.n_clusters, n_init="auto", random_state=0)
        self.kmeans.fit(data)

        self.cluster_to_class_, _ = self._best_fit_classification(
            self.kmeans.labels_, labels
        )
        return self

    def predict(self, data):
        """
        Функция выполняет предсказание меток класса для объектов, поданных на вход. Предсказание происходит в два этапа
            1. Определение меток кластеров для новых объектов
            2. Преобразование меток кластеров в метки классов с помощью выученного преобразования

        :param np.ndarray data: Непустой двумерный массив векторов-признаков объектов
        :return np.ndarray: Предсказанные метки класса
        """
        if self.kmeans is None or self.cluster_to_class_ is None:
            raise RuntimeError("Classifier has not been fitted yet")
        clusters = self.kmeans.predict(np.asarray(data))
        return self.cluster_to_class_[clusters]

    def _best_fit_classification(self, cluster_labels, true_labels):
        """
        :param np.ndarray cluster_labels: Непустой одномерный массив. Предсказанные метки кластеров.
            Содержит элементы в диапазоне [0, ..., n_clusters - 1]
        :param np.ndarray true_labels: Непустой одномерный массив. Частичная разметка выборки.
            Неразмеченные объекты имеют метку -1. Размеченные объекты могут иметь произвольную неотрицательную метку.
            Существует хотя бы один размеченный объект
        :return
            np.ndarray mapping: Соответствие между номерами кластеров и номерами классов в выборке,
                то есть mapping[idx] -- номер класса для кластера idx
            np.ndarray predicted_labels: Предсказанные в соответствии с mapping метки объектов

            Соответствие между номером кластера и меткой класса определяется как номер класса с максимальным числом объектов
        внутри этого кластера.
            * Если есть несколько классов с числом объектов, равным максимальному, то выбирается метка с наименьшим номером.
            * Если кластер не содержит размеченных объектов, то выбирается номер класса с максимальным числом элементов в выборке.
            * Если же и таких классов несколько, то также выбирается класс с наименьшим номером
        """
        clust = np.asarray(cluster_labels, dtype=int)
        true = np.asarray(true_labels, dtype=int)
        if clust.shape[0] != true.shape[0]:
            raise ValueError("`cluster_labels` and `true_labels` size mismatch")

        mask_lbl = true >= 0
        unique_cls, total_cnt = np.unique(true[mask_lbl], return_counts=True)
        g_major = unique_cls[total_cnt == total_cnt.max()].min()

        cont = np.zeros((self.n_clusters, unique_cls.size), dtype=np.int64)
        if mask_lbl.any():
            rows = clust[mask_lbl]
            cols = np.searchsorted(unique_cls, true[mask_lbl])
            np.add.at(cont, (rows, cols), 1)

        mapping = np.full(self.n_clusters, g_major, dtype=int)
        if unique_cls.size:
            max_per_row = cont.max(axis=1)
            has_labeled = max_per_row > 0
            mapping[has_labeled] = unique_cls[np.argmax(cont[has_labeled], axis=1)]

        predicted = mapping[clust]
        return mapping, predicted
