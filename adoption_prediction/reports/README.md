# Результаты первого запуска

[Разбор результатов и ошибок](2026-11-08_group_cv_seed42/analysis.md).
Запуск предоставлен в [выполненном notebook](../notebooks/01-kaggle-baseline_1.ipynb).

В `2026-11-08_group_cv_seed42/` сохранены:

- `experiments/` — конфиг, окружение, фолды, метрики, OOF, срезы и графики пяти моделей;
- `eda/` — аудит, таблицы и графики, восстановленные из outputs notebook;
- `refit/` — метаданные CatBoostClassifier и submission на 3 972 объявления;
- `derived_age_slices.csv` — описательные возрастные срезы, рассчитанные из OOF;
- `requirements-kaggle.txt` — наблюдавшиеся версии основных пакетов Kaggle.

Основной текст отчета — `analysis.md`.

Метрики можно пересчитать командой из основного README: `scripts/verify_run.py` проверяет OOF, сводные/per-fold метрики, матрицы ошибок и структуру submission.