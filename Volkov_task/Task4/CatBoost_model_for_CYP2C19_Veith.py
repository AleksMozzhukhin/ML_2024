#%%
import numpy as np
import pandas as pd
from pandas import DataFrame
#%%
from tdc.single_pred import ADME

data = ADME(name='CYP2C19_Veith')
split = data.get_split()

#%%
train_df = pd.DataFrame(split['train'])
valid_df = pd.DataFrame(split['valid'])
test_df = pd.DataFrame(split['test'])
full_df = pd.concat([train_df, valid_df, test_df], ignore_index=True)
#%%
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import rdFingerprintGenerator  # Функция для проверки валидности SMILES


def validate_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol is not None
    except:
        return False


# Функция для генерации Morgan фингерпринтов
def generate_morgan_fp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return list(morgan_gen.GetFingerprint(mol))  # Генерация вектора как списка
    return None


train_df['valid_smiles'] = train_df['Drug'].apply(validate_smiles)
valid_df['valid_smiles'] = valid_df['Drug'].apply(validate_smiles)
test_df['valid_smiles'] = test_df['Drug'].apply(validate_smiles)

train_df = train_df[train_df['valid_smiles']].drop(columns=['valid_smiles'])
valid_df = valid_df[valid_df['valid_smiles']].drop(columns=['valid_smiles'])
test_df = test_df[test_df['valid_smiles']].drop(columns=['valid_smiles'])

# 3. Преобразование SMILES в Morgan фингерпринты
# Создание генератора Morgan фингерпринтов
morgan_gen = rdFingerprintGenerator.GetMorganGenerator(radius=4, fpSize=2048)
train_df['morgan_fp'] = train_df['Drug'].apply(generate_morgan_fp)
valid_df['morgan_fp'] = valid_df['Drug'].apply(generate_morgan_fp)
test_df['morgan_fp'] = test_df['Drug'].apply(generate_morgan_fp)
#%%
X_train = np.array(train_df['morgan_fp'].tolist())
X_valid = np.array(valid_df['morgan_fp'].tolist())
X_test = np.array(test_df['morgan_fp'].tolist())
y_train = train_df['Y'].values
y_valid = valid_df['Y'].values
y_test = test_df['Y'].values
#%%
print(len(X_test))
print(len(X_train))
print(len(X_valid))
print(len(y_test))
print(len(y_train))
print(len(y_valid))
#%%
import optuna
from sklearn.metrics import f1_score, roc_auc_score
from catboost import CatBoostClassifier
import numpy as np


# Пользовательская метрика F1-Score для CatBoost
def f1_eval_metric(y_true, y_pred):
    y_pred_class = (y_pred > 0.5).astype(int)
    return "F1-Score", f1_score(y_true, y_pred_class), True  # True: метрика максимизируется


# Функция для оптимизации с использованием ROC-AUC
def objective(trial):
    # Гиперпараметры для поиска
    params = {
        'iterations': trial.suggest_int('iterations', 100, 1000),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_loguniform('l2_leaf_reg', 1e-3, 10),
        'bagging_temperature': trial.suggest_uniform('bagging_temperature', 0, 1),
        'random_strength': trial.suggest_uniform('random_strength', 0, 10),
        'loss_function': 'Logloss',
        'task_type': 'GPU',  # Использование GPU
        'verbose': 0,
        'thread_count': -1
    }

    # Обучение модели
    model = CatBoostClassifier(**params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid), verbose=0)

    # ROC-AUC для оптимизации
    preds = model.predict_proba(X_valid)[:, 1]
    roc_auc = roc_auc_score(y_valid, preds)
    return roc_auc


# Оптимизация гиперпараметров
study = optuna.create_study(direction="maximize")  # Максимизация ROC-AUC
study.optimize(objective, n_trials=50)
#%%
# Лучшие параметры
best_params = study.best_params

# Обучение модели с лучшими параметрами
final_model = CatBoostClassifier(**best_params, task_type='GPU')
final_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
#%%
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, accuracy_score

# Обучение модели с записью истории метрик
final_model = CatBoostClassifier(
    **best_params,
    eval_metric='AUC',  # Указываем ROC-AUC для оценки
    task_type="GPU",  # Используем GPU
    verbose=20  # Показываем результат каждые 20 итераций
)

# Обучение модели
final_model.fit(X_train, y_train, eval_set=(X_valid, y_valid), plot=False)

# Получение истории метрик
evals_result = final_model.get_evals_result()

# Извлечение метрик
roc_auc_history = evals_result['validation']['AUC']  # ROC-AUC на валидационной выборке
iterations = range(len(roc_auc_history))  # Итерации

#%%
# Расчёт Accuracy на каждой итерации
predictions_proba = final_model.predict_proba(X_test)
threshold = 0.5  # Порог классификации
predictions_class = (predictions_proba[:, 1] > threshold).astype(int)
accuracy = accuracy_score(y_test, predictions_class)
rog_auc = roc_auc_score(y_test, predictions_class)
# Построение графика ROC-AUC
plt.figure(figsize=(10, 6))
plt.plot(iterations, roc_auc_history, label='Validation ROC-AUC', color='blue')
plt.title('ROC-AUC During Training', fontsize=16)
plt.xlabel('Iterations', fontsize=14)
plt.ylabel('ROC-AUC', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()
plt.show()
#%%
print(f"ROC-AUC: {roc_auc}")
print(f"Accuracy: {accuracy}")
#%%
import matplotlib.pyplot as plt
plt.style.use('default')  # Устанавливаем светлый стиль

feature_importances = final_model.get_feature_importance()

plt.figure(figsize=(10, 6))
plt.bar(range(len(feature_importances)), feature_importances, color='green')
plt.title('Важность признаков')
plt.xlabel('Индекс признака')
plt.ylabel('Важность')
plt.show()
# Индексы признаков
feature_indices = np.arange(len(feature_importances))

nonzero_indices = np.where(feature_importances > 0)[0]
nonzero_importances = feature_importances[nonzero_indices]

sorted_indices = np.argsort(nonzero_importances)
# 1. Для 15 самых значимых признаков
top_15_indices = nonzero_indices[sorted_indices[-15:]]  # Индексы 15 самых значимых признаков
top_15_importances = nonzero_importances[sorted_indices[-15:]]

plt.figure(figsize=(12, 6))
plt.bar(range(len(top_15_importances)), top_15_importances, color='lime')
plt.title('15 наиболее полезных признаков', fontsize=16)
plt.xlabel('Индекс признака', fontsize=14)
plt.ylabel('Важность', fontsize=14)
plt.xticks(range(len(top_15_importances)), top_15_indices, rotation=45)  # Подписи индексов признаков
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()

# 2. Для 15 наименее значимых признаков
bottom_15_indices = nonzero_indices[sorted_indices[:15]]  # Индексы 15 наименее значимых признаков
bottom_15_importances = nonzero_importances[sorted_indices[:15]]


plt.figure(figsize=(12, 6))
plt.bar(range(len(bottom_15_importances)), bottom_15_importances, color='red')
plt.title('15 наименее полезных признаков', fontsize=16)
plt.xlabel('Индекс признака', fontsize=14)
plt.ylabel('Важность', fontsize=14)
plt.xticks(range(len(bottom_15_importances)), bottom_15_indices, rotation=45)  # Подписи индексов признаков
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
#%%
