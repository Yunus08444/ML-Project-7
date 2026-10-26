
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

warnings.filterwarnings("ignore")

# --------------------------------------------------
# 1. Загрузка данных
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "train.csv"
RESULTS_DIR = BASE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"Файл не найден: {DATA_PATH}\n"
        "Помести train.csv в папку data."
    )

df = pd.read_csv(DATA_PATH)

print("=" * 65)
print("ПРАКТИЧЕСКАЯ РАБОТА № 7 — СЛУЧАЙНЫЙ ЛЕС")
print("=" * 65)
print(f"Размер датасета: {df.shape}")
print(f"Количество пропусков: {df.isnull().sum().sum()}")

# SalePrice — целевая переменная.
# Id — идентификатор, поэтому исключаем его из признаков.
X = df.drop(columns=["SalePrice", "Id"], errors="ignore")
y = df["SalePrice"]

# --------------------------------------------------
# 2. Разделение данных
# 60% обучение, 20% валидация, 20% тест
# --------------------------------------------------

X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42
)

print(f"\nОбучающая выборка: {X_train.shape}")
print(f"Валидационная выборка: {X_val.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# --------------------------------------------------
# 3. Предобработка признаков
# --------------------------------------------------

def make_preprocessor(data):
    numeric_features = data.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = data.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipeline, numeric_features),
        ("cat", categorical_pipeline, categorical_features)
    ])


def make_pipeline(data, model):
    return Pipeline([
        ("preprocessor", make_preprocessor(data)),
        ("model", model)
    ])


# --------------------------------------------------
# 4. Функция оценки моделей
# --------------------------------------------------

def evaluate_model(name, model, X_eval, y_eval):
    predictions = model.predict(X_eval)

    return {
        "Модель": name,
        "MAE": mean_absolute_error(y_eval, predictions),
        "RMSE": np.sqrt(mean_squared_error(y_eval, predictions)),
        "R2": r2_score(y_eval, predictions)
    }


# --------------------------------------------------
# 5. Дерево решений и случайный лес по умолчанию
# --------------------------------------------------

print("\nОбучение базовых моделей...")

tree_model = make_pipeline(
    X_train_val,
    DecisionTreeRegressor(random_state=42)
)
tree_model.fit(X_train_val, y_train_val)

default_rf = make_pipeline(
    X_train_val,
    RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )
)
default_rf.fit(X_train_val, y_train_val)

results = [
    evaluate_model("Decision Tree", tree_model, X_test, y_test),
    evaluate_model("Random Forest (default)", default_rf, X_test, y_test)
]

# --------------------------------------------------
# 6. Подбор гиперпараметров с кросс-валидацией
# --------------------------------------------------

print("\nПодбор гиперпараметров случайного леса...")

tuning_pipeline = make_pipeline(
    X_train,
    RandomForestRegressor(random_state=42, n_jobs=-1)
)

param_grid = {
    "model__n_estimators": [100, 200],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_leaf": [1, 2, 4]
}

cv = KFold(n_splits=3, shuffle=True, random_state=42)

search = GridSearchCV(
    estimator=tuning_pipeline,
    param_grid=param_grid,
    scoring="neg_root_mean_squared_error",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

search.fit(X_train, y_train)

best_params = {
    key.replace("model__", ""): value
    for key, value in search.best_params_.items()
}

print("\nЛучшие параметры:")
print(best_params)
print(f"Лучший CV RMSE: {-search.best_score_:.2f}")

# --------------------------------------------------
# 7. Обучение настроенного леса на всех признаках
# --------------------------------------------------

best_rf = make_pipeline(
    X_train_val,
    RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
)
best_rf.fit(X_train_val, y_train_val)

results.append(
    evaluate_model("Random Forest (tuned)", best_rf, X_test, y_test)
)

# --------------------------------------------------
# 8. Встроенная важность признаков
# --------------------------------------------------

preprocessor = best_rf.named_steps["preprocessor"]
forest = best_rf.named_steps["model"]

feature_names = preprocessor.get_feature_names_out()
importances = forest.feature_importances_

importance_df = pd.DataFrame({
    "Признак": feature_names,
    "Важность": importances
}).sort_values("Важность", ascending=False)

importance_df.to_csv(
    RESULTS_DIR / "feature_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nТоп-10 признаков по встроенной важности:")
print(importance_df.head(10).to_string(index=False))

top_features = importance_df.head(20).sort_values("Важность")

plt.figure(figsize=(11, 8))
plt.barh(top_features["Признак"], top_features["Важность"])
plt.xlabel("Важность признака")
plt.ylabel("Признак")
plt.title("Топ-20 признаков случайного леса")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "feature_importance.png", dpi=150)
plt.show()

# --------------------------------------------------
# 9. Эксперимент с удалением наименее важных признаков
# --------------------------------------------------
# Используем permutation importance на валидационной
# выборке, которая не участвовала в обучении best_rf.

print("\nВычисление permutation importance...")

perm = permutation_importance(
    best_rf,
    X_val,
    y_val,
    scoring="neg_root_mean_squared_error",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "Признак": X_val.columns,
    "Важность": perm.importances_mean
}).sort_values("Важность", ascending=False)

permutation_df.to_csv(
    RESULTS_DIR / "permutation_importance.csv",
    index=False,
    encoding="utf-8-sig"
)

print("\nНаименее важные исходные признаки:")
print(permutation_df.tail(10).to_string(index=False))

# Убираем 20% исходных признаков с наименьшей
# permutation importance.
n_remove = max(1, int(len(permutation_df) * 0.20))
removed_features = permutation_df.tail(n_remove)["Признак"].tolist()
kept_features = [col for col in X.columns if col not in removed_features]

print(f"\nУдаляем признаков: {len(removed_features)} из {len(X.columns)}")
print("Удалённые признаки:", removed_features)
print("Осталось признаков:", len(kept_features))

reduced_rf = make_pipeline(
    X_train_val[kept_features],
    RandomForestRegressor(
        **best_params,
        random_state=42,
        n_jobs=-1
    )
)

reduced_rf.fit(
    X_train_val[kept_features],
    y_train_val
)

results.append(
    evaluate_model(
        "Random Forest (reduced features)",
        reduced_rf,
        X_test[kept_features],
        y_test
    )
)

# --------------------------------------------------
# 10. Сравнение метрик
# --------------------------------------------------

results_df = pd.DataFrame(results)

print("\n" + "=" * 75)
print("СРАВНЕНИЕ МОДЕЛЕЙ НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 75)
print(results_df.round(4).to_string(index=False))

results_df.to_csv(
    RESULTS_DIR / "model_comparison.csv",
    index=False,
    encoding="utf-8-sig"
)

# График сравнения RMSE
plt.figure(figsize=(10, 5))
plt.bar(results_df["Модель"], results_df["RMSE"])
plt.ylabel("RMSE (меньше — лучше)")
plt.title("Сравнение моделей по RMSE")
plt.xticks(rotation=15, ha="right")
plt.tight_layout()
plt.savefig(RESULTS_DIR / "model_comparison.png", dpi=150)
plt.show()

# --------------------------------------------------
# 11. Сохранение настроек и итогов
# --------------------------------------------------

with open(RESULTS_DIR / "best_params.txt", "w", encoding="utf-8") as file:
    file.write("Лучшие параметры случайного леса:\n")
    file.write(str(best_params) + "\n")
    file.write(f"Лучший CV RMSE: {-search.best_score_:.2f}\n")
    file.write(f"Удалённые признаки: {removed_features}\n")
    file.write(f"Количество деревьев: {best_params['n_estimators']}\n")

print("\nРабота завершена.")
print(f"Все результаты сохранены в: {RESULTS_DIR}")