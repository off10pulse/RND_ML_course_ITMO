# Сохранённые результаты

## ДЗ2: веса классов и метаданные первого фото

Запуск 16.11.2026 находится в `2026-11-16_hw2_group_cv_seed42/`. Выводы и описание экспериментов — в [основном README](../README.md#дз2-веса-классов-и-метаданные-фото).

- [experiments.csv](2026-11-16_hw2_group_cv_seed42/experiments.csv) — четыре варианта, гипотезы, seed, метрики и время;
- [comparisons.csv](2026-11-16_hw2_group_cv_seed42/comparisons.csv) — разницы метрик, bootstrap-интервалы и две абляции;
- [class_metrics.csv](2026-11-16_hw2_group_cv_seed42/class_metrics.csv) — precision, recall и F1 по классам;
- `baseline/`, `weights/`, `metadata/`, `full/` — конфиг, окружение, фолды, метрики и OOF каждого варианта;
- `photo_features/` — рассчитанные признаки train и аудит их наличия;
- `comparison.png`, `confusion_matrices.png` — сравнение метрик и матрицы ошибок;
- `plan.json`, `hypothesis_checks.json` — критерии до запуска и результаты их проверки.

Лучший mean QWK — у варианта `metadata`: 0.3382 +- 0.0218. Критерии обеих гипотез не достигнуты.

## ДЗ1: первый запуск

[Разбор результатов и ошибок](2026-11-08_group_cv_seed42/analysis.md).
Запуск предоставлен в [выполненном notebook](../notebooks/adoption_prediction_notebook.ipynb).

В `2026-11-08_group_cv_seed42/` сохранены:

- `experiments/` — конфиг, окружение, фолды, метрики, OOF, срезы и графики пяти моделей;
- `eda/` — аудит, таблицы и графики, восстановленные из outputs notebook;
- `refit/` — метаданные CatBoostClassifier и submission на 3 972 объявления;
- `derived_age_slices.csv` — описательные возрастные срезы, рассчитанные из OOF;
- `requirements-kaggle.txt` — наблюдавшиеся версии основных пакетов Kaggle.

Основной текст — `analysis.md`.

Метрики можно пересчитать командой из основного README: `scripts/verify_run.py` проверяет OOF, сводные/per-fold метрики, матрицы ошибок и структуру submission.
