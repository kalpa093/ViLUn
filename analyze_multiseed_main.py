#!/usr/bin/env python3
"""Aggregate main-run accuracy and perform paired non-parametric tests."""

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats


METRICS = ("retain_acc", "test_acc", "forget_acc")
PRIVACY_METRICS = ("mia_asr", "auc_correctness", "tpr1fpr", "tpr01fpr")
METHOD_ORDER = ("Retrain", "GA", "SISA", "SalUn", "PS", "DELETE", "ViLUn", "ViLUn_f")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="*", type=int)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    value = row.get(key, "")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {key}={value!r} in row {row}") from exc
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {key}={value!r} in row {row}")
    return result


def as_float_any(row, keys):
    if isinstance(keys, str):
        keys = (keys,)
    for key in keys:
        if row.get(key, "") not in ("", None):
            return as_float(row, key)
    raise ValueError(f"None of the fields {keys} are available in row {row}")


def latest_matching(rows, **conditions):
    matches = []
    for row in rows:
        if all(str(row.get(key, "")).lower() == str(value).lower() for key, value in conditions.items()):
            matches.append(row)
    if not matches:
        raise KeyError(f"No row matches {conditions}")
    return matches[-1]


def normalized_row(method, dataset, seed, row, retain_key, test_key, forget_key):
    return {
        "method": method,
        "dataset": dataset,
        "seed": seed,
        "retain_acc": as_float_any(row, retain_key),
        "test_acc": as_float_any(row, test_key),
        "forget_acc": as_float_any(row, forget_key),
    }


def to_percent(value):
    return value * 100.0 if abs(value) <= 1.5 else value


def mean_fields(row, keys):
    values = [as_float(row, key) for key in keys if row.get(key, "") not in ("", None)]
    if not values:
        raise ValueError(f"None of the fields {keys} are available in row {row}")
    return float(np.mean(values))


def collect_mia_seed(seed_root, seed):
    path = seed_root / "unlearned_mia_metrics" / "summary_baseline_with_vilun_mia_metrics.csv"
    rows = read_rows(path)
    method_map = {
        "retrain": "Retrain",
        "gradient_ascent": "GA",
        "sisa": "SISA",
        "salun": "SalUn",
        "ps": "PS",
        "delete": "DELETE",
        "vilun": "ViLUn",
        "vilun_heldout": "ViLUn",
        "vilun_f": "ViLUn_f",
    }
    # Normalize aliases before aggregation. If a merged CSV contains both an
    # older and a current name for the same method, the last row wins.
    normalized = {}
    for row in rows:
        source_method = row.get("Method", "").lower()
        dataset = row.get("Dataset", "").lower()
        method = method_map.get(source_method)
        if method is None or dataset not in ("cifar10", "cifar100", "tinyimagenet"):
            continue
        normalized[(method, dataset)] = row

    output = []
    for dataset in ("cifar10", "cifar100", "tinyimagenet"):
        for method in METHOD_ORDER:
            row = normalized.get((method, dataset))
            if row is None:
                continue
            output.append({
                "method": method,
                "dataset": dataset,
                "seed": seed,
                "mia_asr": to_percent(as_float(row, "ASR_Average")),
                "auc_correctness": to_percent(as_float(row, "AUC_Correctness")),
                "tpr1fpr": to_percent(mean_fields(row, (
                    "TPR1FPR_Confidence", "TPR1FPR_Entropy", "TPR1FPR_ModEntropy",
                ))),
                "tpr01fpr": to_percent(mean_fields(row, (
                    "TPR01FPR_Confidence", "TPR01FPR_Entropy", "TPR01FPR_ModEntropy",
                ))),
            })
    expected = len(METHOD_ORDER) * 3
    if len(output) != expected:
        found = {(row["method"], row["dataset"]) for row in output}
        missing = [
            (method, dataset)
            for method in METHOD_ORDER
            for dataset in ("cifar10", "cifar100", "tinyimagenet")
            if (method, dataset) not in found
        ]
        raise RuntimeError(f"Incomplete MIA summary for seed {seed}: {missing}")
    return output


def collect_seed(seed_root, seed):
    history = seed_root / "history"
    records = []

    retrain_ga = read_rows(history / "summary_retrain_ga.csv")
    heldout = read_rows(history / "summary_heldout.csv")
    vilun_f = read_rows(history / "summary_vilun.csv")
    salun = read_rows(history / "summary_salun.csv")
    sisa = read_rows(history / "summary_sisa.csv")
    ps = read_rows(history / "summary_ps.csv")
    delete = read_rows(history / "summary_delete.csv")

    datasets = ("cifar10", "cifar100", "tinyimagenet")
    for dataset in datasets:
        for method, token in (("Retrain", "retrain"), ("GA", "gradient_ascent")):
            row = latest_matching(retrain_ga, unlearning=token, dataset=dataset, seed=seed)
            records.append(normalized_row(
                method, dataset, seed, row,
                ("selected_train_retain_acc", "best_train_retain_acc"),
                ("selected_test_retain_acc", "best_test_retain_acc"),
                ("selected_train_forget_acc", "best_train_forget_acc"),
            ))

        row = latest_matching(heldout, Dataset=dataset, Seed=seed)
        records.append(normalized_row(
            "ViLUn", dataset, seed, row,
            ("Final_Train_Retain_Acc", "Best_Train_Retain_Acc"),
            ("Final_Test_Retain_Acc", "Best_Test_Retain_Acc"),
            ("Final_Train_Forget_Acc", "Best_Train_Forget_Acc"),
        ))

        row = latest_matching(vilun_f, Dataset=dataset, Seed=seed)
        records.append(normalized_row(
            "ViLUn_f", dataset, seed, row,
            ("Final_Train_Retain_Acc", "Best_Train_Retain_Acc"),
            ("Final_Test_Retain_Acc", "Best_Test_Retain_Acc"),
            ("Final_Train_Forget_Acc", "Best_Train_Forget_Acc"),
        ))

        for method, rows in (("SalUn", salun), ("SISA", sisa), ("PS", ps)):
            row = latest_matching(rows, Dataset=dataset, Seed=seed)
            records.append(normalized_row(
                method, dataset, seed, row,
                ("Final_Train_Retain_Acc", "Best_Train_Retain_Acc"),
                ("Final_Test_Retain_Acc", "Best_Test_Retain_Acc"),
                ("Final_Train_Forget_Acc", "Best_Train_Forget_Acc"),
            ))

        row = latest_matching(delete, Dataset=dataset, Seed=seed)
        records.append(normalized_row(
            "DELETE", dataset, seed, row,
            ("Final_Train_Retain_Acc", "Best_Train_Retain_Acc"),
            ("Final_Test_Retain_Acc", "Best_Test_Retain_Acc"),
            ("Final_TrainForget_Acc", "Best_TrainForget_Acc"),
        ))

    return records


def write_csv(path, rows, fieldnames):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def mean_std_rows(records, metrics=METRICS):
    grouped = defaultdict(list)
    for row in records:
        for metric in metrics:
            grouped[(row["dataset"], row["method"], metric)].append(row[metric])

    output = []
    for (dataset, method, metric), values in sorted(
        grouped.items(), key=lambda item: (
            item[0][0], METHOD_ORDER.index(item[0][1]), metrics.index(item[0][2])
        )
    ):
        array = np.asarray(values, dtype=float)
        n = len(array)
        mean = float(array.mean())
        std = float(array.std(ddof=1)) if n > 1 else float("nan")
        variance = float(array.var(ddof=1)) if n > 1 else float("nan")
        if n > 1:
            half_width = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
            ci_low, ci_high = mean - half_width, mean + half_width
        else:
            ci_low = ci_high = float("nan")
        output.append({
            "dataset": dataset,
            "method": method,
            "metric": metric,
            "n": n,
            "mean": f"{mean:.4f}",
            "std": f"{std:.4f}",
            "variance": f"{variance:.4f}",
            "ci95_low": f"{ci_low:.4f}",
            "ci95_high": f"{ci_high:.4f}",
            "paper_value": f"{mean:.2f} $\\pm$ {std:.2f}",
        })
    return output


def add_retrain_gaps(records):
    lookup = {(row["dataset"], row["seed"], row["method"]): row for row in records}
    output = []
    for row in records:
        retrain = lookup[(row["dataset"], row["seed"], "Retrain")]
        result = dict(row)
        for metric, prefix in zip(METRICS, ("dr", "dt", "df")):
            result[f"{prefix}_gap"] = abs(row[metric] - retrain[metric])
        result["total_mae"] = np.mean([result["dr_gap"], result["dt_gap"], result["df_gap"]])
        output.append(result)
    return output


def aggregate_total_mae_by_seed(gap_rows):
    grouped = defaultdict(list)
    for row in gap_rows:
        grouped[(row["method"], row["seed"])].append(row["total_mae"])

    per_seed = []
    for (method, seed), values in sorted(
        grouped.items(), key=lambda item: (METHOD_ORDER.index(item[0][0]), item[0][1])
    ):
        per_seed.append({
            "method": method,
            "seed": seed,
            "datasets": len(values),
            "total_mae": float(np.mean(values)),
        })

    by_method = defaultdict(list)
    for row in per_seed:
        by_method[row["method"]].append(row["total_mae"])

    summaries = []
    for method in METHOD_ORDER:
        values = np.asarray(by_method[method], dtype=float)
        n = len(values)
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if n > 1 else float("nan")
        variance = float(values.var(ddof=1)) if n > 1 else float("nan")
        if n > 1:
            half_width = float(stats.t.ppf(0.975, n - 1) * std / math.sqrt(n))
            ci_low, ci_high = mean - half_width, mean + half_width
        else:
            ci_low = ci_high = float("nan")
        summaries.append({
            "method": method,
            "n_seeds": n,
            "mean": f"{mean:.4f}",
            "std": f"{std:.4f}",
            "variance": f"{variance:.4f}",
            "ci95_low": f"{ci_low:.4f}",
            "ci95_high": f"{ci_high:.4f}",
            "paper_value": f"{mean:.2f} $\\pm$ {std:.2f}",
        })
    return per_seed, summaries


def rank_biserial(reference, comparison):
    # Positive values mean the reference has lower Total MAE.
    differences = np.asarray(comparison) - np.asarray(reference)
    differences = differences[differences != 0]
    if len(differences) == 0:
        return 0.0
    ranks = stats.rankdata(np.abs(differences))
    positive = float(ranks[differences > 0].sum())
    negative = float(ranks[differences < 0].sum())
    return (positive - negative) / (positive + negative)


def holm_adjust(p_values):
    count = len(p_values)
    order = sorted(range(count), key=lambda index: p_values[index])
    adjusted = [0.0] * count
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (count - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted


def statistical_rows(gap_rows):
    methods = tuple(method for method in METHOD_ORDER if method != "Retrain")
    lookup = {(row["dataset"], row["seed"], row["method"]): row for row in gap_rows}
    blocks = sorted({(row["dataset"], row["seed"]) for row in gap_rows})
    missing = [
        (dataset, seed, method)
        for dataset, seed in blocks
        for method in methods
        if (dataset, seed, method) not in lookup
    ]
    if missing:
        raise RuntimeError(f"Incomplete paired blocks, first missing entries: {missing[:10]}")

    arrays = {
        method: np.asarray([lookup[(dataset, seed, method)]["total_mae"] for dataset, seed in blocks])
        for method in methods
    }
    statistic, p_value = stats.friedmanchisquare(*(arrays[method] for method in methods))
    friedman = [{
        "blocks": len(blocks),
        "methods": len(methods),
        "statistic": f"{statistic:.6g}",
        "p_value": f"{p_value:.6g}",
    }]

    reference = arrays["ViLUn"]
    comparisons = [method for method in methods if method != "ViLUn"]
    raw_p_values = []
    posthoc = []
    for method in comparisons:
        compared = arrays[method]
        if np.allclose(reference, compared):
            wilcoxon_stat, raw_p = 0.0, 1.0
        else:
            result = stats.wilcoxon(reference, compared, alternative="two-sided", method="auto")
            wilcoxon_stat, raw_p = float(result.statistic), float(result.pvalue)
        raw_p_values.append(raw_p)
        posthoc.append({
            "reference": "ViLUn",
            "comparison": method,
            "blocks": len(blocks),
            "mean_total_mae_reference": float(reference.mean()),
            "mean_total_mae_comparison": float(compared.mean()),
            "mean_difference_comparison_minus_reference": float((compared - reference).mean()),
            "median_difference_comparison_minus_reference": float(np.median(compared - reference)),
            "wilcoxon_statistic": wilcoxon_stat,
            "p_raw": raw_p,
            "rank_biserial": rank_biserial(reference, compared),
        })

    adjusted = holm_adjust(raw_p_values)
    for row, p_adjusted in zip(posthoc, adjusted):
        row["p_holm"] = p_adjusted
        row["significant_0.05"] = "yes" if p_adjusted < 0.05 else "no"
        for key, value in list(row.items()):
            if isinstance(value, float):
                row[key] = f"{value:.6g}"
    return friedman, posthoc


def main():
    args = parse_args()
    seeds = args.seeds
    if not seeds:
        seeds = sorted(
            int(path.name.removeprefix("seed"))
            for path in args.root.glob("seed*")
            if path.is_dir() and path.name.removeprefix("seed").isdigit()
        )
    if not seeds:
        raise SystemExit("No seed directories found")

    records = []
    privacy_records = []
    for seed in seeds:
        seed_root = args.root / f"seed{seed}"
        records.extend(collect_seed(seed_root, seed))
        mia_summary = seed_root / "unlearned_mia_metrics" / "summary_baseline_with_vilun_mia_metrics.csv"
        if mia_summary.is_file():
            privacy_records.extend(collect_mia_seed(seed_root, seed))

    output_dir = args.output_dir or args.root / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "main_accuracy_long.csv", records, ["method", "dataset", "seed", *METRICS])

    summaries = mean_std_rows(records)
    write_csv(
        output_dir / "main_accuracy_mean_std.csv",
        summaries,
        ["dataset", "method", "metric", "n", "mean", "std", "variance", "ci95_low", "ci95_high", "paper_value"],
    )

    gap_rows = add_retrain_gaps(records)
    gap_fields = ["method", "dataset", "seed", *METRICS, "dr_gap", "dt_gap", "df_gap", "total_mae"]
    write_csv(output_dir / "main_total_mae_by_seed.csv", gap_rows, gap_fields)

    overall_by_seed, overall_summaries = aggregate_total_mae_by_seed(gap_rows)
    write_csv(
        output_dir / "main_overall_total_mae_by_seed.csv",
        overall_by_seed,
        ["method", "seed", "datasets", "total_mae"],
    )
    write_csv(
        output_dir / "main_overall_total_mae_mean_std.csv",
        overall_summaries,
        ["method", "n_seeds", "mean", "std", "variance", "ci95_low", "ci95_high", "paper_value"],
    )

    friedman, posthoc = statistical_rows(gap_rows)
    write_csv(output_dir / "friedman_total_mae.csv", friedman, list(friedman[0]))
    write_csv(output_dir / "wilcoxon_holm_total_mae.csv", posthoc, list(posthoc[0]))

    if privacy_records:
        expected_privacy_rows = len(seeds) * len(METHOD_ORDER) * 3
        if len(privacy_records) != expected_privacy_rows:
            raise RuntimeError(
                f"MIA results exist for only part of the requested seeds: "
                f"expected {expected_privacy_rows}, found {len(privacy_records)}"
            )
        write_csv(
            output_dir / "main_privacy_long.csv",
            privacy_records,
            ["method", "dataset", "seed", *PRIVACY_METRICS],
        )
        privacy_summaries = mean_std_rows(privacy_records, PRIVACY_METRICS)
        write_csv(
            output_dir / "main_privacy_mean_std.csv",
            privacy_summaries,
            ["dataset", "method", "metric", "n", "mean", "std", "variance", "ci95_low", "ci95_high", "paper_value"],
        )

    print(f"[SAVE] {len(records)} run rows from seeds {seeds} -> {output_dir}")
    print(f"[TEST] Friedman p={friedman[0]['p_value']} over {friedman[0]['blocks']} paired blocks")
    for row in posthoc:
        print(
            f"[TEST] ViLUn vs {row['comparison']}: "
            f"p_holm={row['p_holm']}, r_rb={row['rank_biserial']}"
        )


if __name__ == "__main__":
    main()
