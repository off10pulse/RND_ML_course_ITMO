# Сохраненные результаты

## ДЗ3: ошибки и устойчивость

Запуск 26.11.2026 находится в `2026-11-26_hw3_diagnostics/`. Основные результаты и ограничения — в [README проекта](../README.md#дз3-ошибки-устойчивость-и-ограничения).

- [error_examples.csv](2026-11-26_hw3_diagnostics/error_examples.csv), [case_notes.json](2026-11-26_hw3_diagnostics/case_notes.json), `case_photos/` — четыре карточки с описаниями, фото и гипотезами причин ошибок;
- [error_patterns.csv](2026-11-26_hw3_diagnostics/error_patterns.csv) — частоты типовых ошибок;
- [slices.csv](2026-11-26_hw3_diagnostics/slices.csv), [class_metrics.csv](2026-11-26_hw3_diagnostics/class_metrics.csv) — качество по группам и классам;
- [split_robustness.csv](2026-11-26_hw3_diagnostics/split_robustness.csv), `splits/` — два новых разбиения и исходная оценка ДЗ2, конфиги, фолды и OOF;
- [age_noise/summary.csv](2026-11-26_hw3_diagnostics/age_noise/summary.csv), `age_noise/` — проверка возраста, семь условий с OOF, метрики по фолдам и параметры шума;
- `slices.png`, `classes.png`, `slice_target_distribution.png`, `split_robustness.png`, `age_noise.png` — визуальные артефакты;
- `model_config.yaml`, `checks_config.yaml`, `input.json` — выбранная модель, условия проверок и окружение.

QWK при смене разбиения — 0.3319–0.3382. При шуме возраста sigma=1 и 3 месяца — 0.3220 и 0.2772 соответственно. Гиперпараметры по результатам этих проверок не подбирались.

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
