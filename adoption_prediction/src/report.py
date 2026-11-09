"""Репорт в md"""
import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def markdown_table(frame):
    header = "| " + " | ".join(str(c) for c in frame.columns) + " |\n"
    divider = "| " + " | ".join("---" for _ in frame.columns) + " |\n"
    rows = ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return header + divider + "\n".join(rows)


def build_report(output):
    output = Path(output)
    summary = pd.read_csv(output / "summary.csv")
    display = summary[["model", "qwk_mean", "qwk_std", "oof_macro_f1", "oof_mae"]].round(4)
    text = ["# Локальная валидация\n", "Метрики рассчитаны по сохраненным OOF-предсказаниям.\n",
            markdown_table(display), "\nСреднее и std относятся к фолдам; это не независимые запуски. "
            "Pooled OOF QWK в summary.csv отличается от среднего по фолдам.\n"]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(summary.model, summary.qwk_mean, yerr=summary.qwk_std.fillna(0), capsize=4)
    ax.tick_params(axis="x", rotation=20)
    ax.set(ylabel="QWK (mean +- fold std)", title="Same folds for every model")
    fig.tight_layout()
    fig.savefig(output / "model_comparison.png", dpi=140)
    plt.close(fig)
    text.append("![Сравнение моделей](model_comparison.png)\n")
    for name in summary.model:
        matrix = pd.read_csv(output / name / "confusion_matrix.csv", index_col=0)
        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(matrix.values, cmap="Blues")
        for (i, j), value in np.ndenumerate(matrix.values):
            ax.text(j, i, str(value), ha="center", va="center")
        ax.set(xlabel="Predicted", ylabel="True", title=name, xticks=range(5), yticks=range(5))
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        fig.savefig(output / name / "confusion_matrix.png", dpi=140)
        plt.close(fig)
        errors = pd.read_csv(output / name / "errors.csv").head(3)
        text.extend([f"\n## {name}\n", f"![Матрица ошибок]({name}/confusion_matrix.png)\n",
                     "Примепы трех наибольших ошибок:\n",
                     markdown_table(errors[["PetID", "Type", "Age", "y_true", "prediction", "absolute_error"]])])
    (output / "report.md").write_text("\n".join(text), encoding="utf-8")
    return output / "report.md"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    print(build_report(args.run_dir))


if __name__ == "__main__":
    main()
