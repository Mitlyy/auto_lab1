# Lab 1

## Что выбрано

- Алгоритм с учителем: `RandomForestClassifier`.
- Датасет: OpenML `diabetes` (`data_id=37`).
- Целевая функция: средняя `accuracy` на `4-fold CV`.
- Оптимизируемые гиперпараметры (7 шт.):
  `n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`,
  `max_features`, `bootstrap`, `criterion`.


## Выполнено

1. Выбран алгоритм с большим числом гиперпараметров (`RandomForestClassifier`).
2. Выбран датасет и метрика (OpenML `diabetes`, `4-fold CV accuracy`).
3. Реализованы вручную:
   - случайный поиск;
   - байесовская оптимизация (`GaussianProcessRegressor` + `Expected Improvement`).
4. Проведено сравнение ручных `Bayesian` vs `Random`.
5. Построен график значения целевой функции от шага.
6. Построена визуализация пространства гиперпараметров (PCA-проекция в 2D, цвет = score).
7. Оценена важность гиперпараметров (`RandomForestRegressor.feature_importances_` на истории трайлов).
8. Повторены шаги 4-7 через `Optuna` (`TPESampler` vs `RandomSampler`).

## Структура проекта

- `auto_lab1/run_auto_lab1.py` - CLI-точка входа.
- `auto_lab1/src/auto_lab1/config.py` - конфиг эксперимента.
- `auto_lab1/src/auto_lab1/search_space.py` - пространство гиперпараметров, кодирование/декодирование.
- `auto_lab1/src/auto_lab1/objective.py` - датасет и целевая функция.
- `auto_lab1/src/auto_lab1/manual_search.py` - ручный random search и BO.
- `auto_lab1/src/auto_lab1/optuna_search.py` - Optuna random/TPE.
- `auto_lab1/src/auto_lab1/plotting.py` - построение графиков.
- `auto_lab1/src/auto_lab1/reporting.py` - summary и importance.
- `auto_lab1/src/auto_lab1/pipeline.py` - оркестрация всего пайплайна.


## Результаты

### Ручная реализация

| method | best_score | best_step | mean_score |
| --- | --- | --- | --- |
| manual_bo | 0.77734 | 26 | 0.76660 |
| manual_random | 0.76953 | 16 | 0.76257 |

Итог: ручной BO лучше случайного поиска по лучшему найденному значению метрики.

### Optuna

| method | best_score | best_step | mean_score |
| --- | --- | --- | --- |
| optuna_tpe | 0.78776 | 17 | 0.77047 |
| optuna_random | 0.78516 | 9 | 0.76497 |

Итог: `Optuna TPE` тоже лучше `Optuna Random`.

## Графики

### Manual: best score vs step

![manual best vs step](outputs/figures/manual_best_vs_step.png)

### Manual: space projection

![manual space projection](outputs/figures/manual_space_projection.png)

### Manual: hyperparameter importance

![manual importance](outputs/figures/manual_importance.png)

### Optuna: best score vs step

![optuna best vs step](outputs/figures/optuna_best_vs_step.png)

### Optuna: space projection

![optuna space projection](outputs/figures/optuna_space_projection.png)

### Optuna: hyperparameter importance

![optuna importance](outputs/figures/optuna_importance.png)

## Краткий вывод

- В обоих сценариях (ручной и Optuna) байесовский подход дал лучший итог, чем random search.
- На OpenML `diabetes` чаще всего важны: `max_depth`, `min_samples_leaf`, `max_features`, иногда `bootstrap`.
