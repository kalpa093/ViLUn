import argparse
import csv
from pathlib import Path


def read_latest_summary(path: Path, model_path=None):
    if not path.exists():
        return None
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return None
    if model_path is None:
        return rows[-1]
    model_path_str = str(model_path)
    for row in reversed(rows):
        if row.get("Model_Path") == model_path_str:
            return row
    return rows[-1]


def main():
    parser = argparse.ArgumentParser(
        description="Collect full-head/headless villain MIA summaries into one CSV."
    )
    parser.add_argument("--output_root", default="./villain_mia_runs_rnn")
    parser.add_argument("--expert_model", default="rnn")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    out_path = Path(args.out) if args.out else output_root / "summary_villain_mia_metrics.csv"
    rows = []
    for run_dir in sorted(output_root.glob(f"*_{args.expert_model}_seed{args.seed}")):
        suffix = f"_{args.expert_model}_seed{args.seed}"
        dataset = run_dir.name[:-len(suffix)] if run_dir.name.endswith(suffix) else run_dir.name
        for variant in ("full_head", "headless"):
            model_path = run_dir / "saved_models" / variant / f"expert_{dataset}_{args.expert_model}_seed{args.seed}.pth"
            summary_path = run_dir / "mia_history" / variant / "summary_mia.csv"
            row = read_latest_summary(summary_path, model_path)
            if row is None:
                continue
            rows.append({
                "variant": variant,
                "dataset": dataset,
                "expert_model": args.expert_model,
                **row,
            })

    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Summary] Wrote {len(rows)} rows -> {out_path}")


if __name__ == "__main__":
    main()
