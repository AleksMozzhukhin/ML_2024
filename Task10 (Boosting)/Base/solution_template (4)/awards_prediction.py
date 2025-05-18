from xgboost import XGBRegressor

import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from typing import Dict, List, Set


class OptimizedTransformer(BaseEstimator, TransformerMixin):
    """
    Трансформер для обработки категориальных признаков фильмов.
    Оптимизирован для избежания фрагментации DataFrame.
    """

    def __init__(self, keyword_frequency_threshold: int = 58):
        """
        Инициализация трансформера с пороговым значением частоты ключевых слов.

        Args:
            keyword_frequency_threshold: Минимальная частота для включения ключевого слова в модель
        """
        self.keyword_frequency_threshold = keyword_frequency_threshold
        self.keywords_dict: Dict[str, int] = {}
        self.unique_genres: Set[str] = set()
        self.unique_directors: Set[str] = set()
        self.unique_locations: Set[str] = set()
        self.frequent_keywords: Set[str] = set()
        self.categorical_features: List[str] = ['actor_0_gender', 'actor_1_gender', 'actor_2_gender']

    def fit(self, X: pd.DataFrame, y=None) -> 'OptimizedTransformer':
        """
        Обучение трансформера на данных.

        Args:
            X: DataFrame с признаками
            y: Целевые значения (не используются в трансформере)

        Returns:
            self: Экземпляр трансформера
        """
        self._collect_unique_values(X)

        self._count_keyword_frequencies(X)

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Преобразование данных фильмов в формат, подходящий для моделирования.

        Args:
            X: DataFrame с исходными признаками

        Returns:
            DataFrame с преобразованными признаками
        """
        X_transformed = X.copy()

        for cat_feature in self.categorical_features:
            if cat_feature in X_transformed.columns:
                X_transformed[cat_feature] = X_transformed[cat_feature].astype('category')

        all_features = {}

        self._transform_list_features(X_transformed, all_features, 'genres', self.unique_genres)
        self._transform_list_features(X_transformed, all_features, 'directors', self.unique_directors)
        self._transform_list_features(X_transformed, all_features, 'filming_locations', self.unique_locations)
        self._transform_list_features(X_transformed, all_features, 'keywords', self.frequent_keywords)

        one_hot_df = pd.DataFrame(all_features, index=X_transformed.index)

        X_transformed = X_transformed.drop(columns=['genres', 'directors', 'filming_locations', 'keywords'],
                                           errors='ignore')

        result = pd.concat([X_transformed, one_hot_df], axis=1)

        return result

    def _collect_unique_values(self, X: pd.DataFrame) -> None:
        """
        Собирает уникальные значения из списковых полей.

        Args:
            X: DataFrame с признаками
        """
        if 'genres' in X.columns:
            self.unique_genres = set(genre for genres in X['genres'] for genre in genres)

        if 'directors' in X.columns:
            self.unique_directors = set(director for directors in X['directors'] for director in directors)

        if 'filming_locations' in X.columns:
            self.unique_locations = set(location for locations in X['filming_locations'] for location in locations)

    def _count_keyword_frequencies(self, X: pd.DataFrame) -> None:
        """
        Подсчитывает частоту ключевых слов и сохраняет те, что превышают порог.

        Args:
            X: DataFrame с признаками
        """
        if 'keywords' in X.columns:
            all_keywords = [word for words in X['keywords'] for word in words]

            from collections import Counter
            keyword_counter = Counter(all_keywords)

            self.keywords_dict = {k: v for k, v in sorted(keyword_counter.items(),
                                                          key=lambda item: item[1],
                                                          reverse=True)}

            self.frequent_keywords = {k for k, v in self.keywords_dict.items()
                                      if v >= self.keyword_frequency_threshold}

    def _transform_list_features(self, X: pd.DataFrame, features_dict: Dict[str, List[int]],
                                 feature_name: str, unique_values: Set[str]) -> None:
        """
        Преобразует списковый признак в one-hot кодирование.

        Args:
            X: DataFrame с признаками
            features_dict: Словарь для сохранения one-hot признаков
            feature_name: Имя признака для преобразования
            unique_values: Множество уникальных значений данного признака
        """
        if feature_name not in X.columns:
            return

        for value in unique_values:
            features_dict[value] = [1 if value in row[feature_name] else 0
                                    for _, row in X.iterrows()]


def train_model_and_predict(train_file: str, test_file: str) -> np.ndarray:
    df_train = pd.read_json(train_file, lines=True)
    df_test = pd.read_json(test_file, lines=True)

    y_train = df_train["awards"]
    X_train = df_train.drop(columns=["awards"])

    model_params = {
        "n_estimators": 399,
        "learning_rate": 0.07556948073537184,
        "max_depth": 4,
        "min_child_weight": 4,
        "gamma": 0.0006927836739806663,
        "subsample": 0.7194027797168907,
        "colsample_bytree": 0.8702295178289116,
        "reg_alpha": 0.009270676701797624,
        "enable_categorical": True,
    }
    pipeline = Pipeline([
        ('transformer', OptimizedTransformer(keyword_frequency_threshold=58)),
        ('model', XGBRegressor(**model_params))
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(df_test).astype(np.float64)

    return predictions
