import argparse
import csv
import shutil
import subprocess
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset

from pretrain_models import (
    evaluate_model,
    get_model,
    load_dataset,
    set_seed,
    strip_classifier_head,
    train_model,
)


def ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)
    return path


def copy_forget_indices(src: Path, dst_dir: Path):
    ensure_dir(dst_dir)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    return dst


def run_mia(eval_script: Path, dataset: str, expert_model: str, data_dir: str, history_dir: Path,
            model_path: Path, forget_indices_path: Path, seed: int,
            repr_attack_epochs: int, repr_attack_batch_size: int,
            repr_attack_lr: float, repr_attack_train_fraction: float,
            repr_attack_nonmember_source: str, include_gradient: bool):
    cmd = [
        sys.executable,
        str(eval_script),
        "--method", "expert",
        "--dataset", dataset,
        "--model", expert_model,
        "--data_dir", data_dir,
        "--history_dir", str(history_dir),
        "--model_path", str(model_path),
        "--forget_indices_path", str(forget_indices_path),
        "--seed", str(seed),
        "--representation_mia",
        "--repr_attack_epochs", str(repr_attack_epochs),
        "--repr_attack_batch_size", str(repr_attack_batch_size),
        "--repr_attack_lr", str(repr_attack_lr),
        "--repr_attack_train_fraction", str(repr_attack_train_fraction),
        "--repr_attack_nonmember_source", repr_attack_nonmember_source,
    ]
    if include_gradient:
        cmd.append("--gradient_mia")
    subprocess.run(cmd, check=True)

    summary_path = history_dir / "summary_mia.csv"
    with open(summary_path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for row in reversed(rows):
        if row.get("Model_Path") == str(model_path):
            return row
    raise RuntimeError(f"No matching MIA row found for {model_path}")


def append_combined_summary(path: Path, row: dict):
    ensure_dir(path.parent)
    fieldnames = [
        "variant",
        "dataset",
        "expert_model",
        "seed",
        "train_acc",
        "test_acc",
        "forget_acc",
        "time_s",
        "asr_correctness",
        "asr_confidence",
        "asr_entropy",
        "asr_modentropy",
        "asr_average",
        "auc_correctness",
        "auc_confidence",
        "auc_entropy",
        "auc_modentropy",
        "tpr1fpr_correctness",
        "tpr1fpr_confidence",
        "tpr1fpr_entropy",
        "tpr1fpr_modentropy",
        "tpr01fpr_correctness",
        "tpr01fpr_confidence",
        "tpr01fpr_entropy",
        "tpr01fpr_modentropy",
        "repr_auc_l2_centroid",
        "repr_tpr1fpr_l2_centroid",
        "repr_tpr01fpr_l2_centroid",
        "repr_auc_cosine_centroid",
        "repr_tpr1fpr_cosine_centroid",
        "repr_tpr01fpr_cosine_centroid",
        "repr_auc_norm",
        "repr_tpr1fpr_norm",
        "repr_tpr01fpr_norm",
        "repr_auc_attack_mlp",
        "repr_tpr1fpr_attack_mlp",
        "repr_tpr01fpr_attack_mlp",
        "grad_auc_loss",
        "grad_tpr1fpr_loss",
        "grad_tpr01fpr_loss",
        "grad_auc_head_norm",
        "grad_tpr1fpr_head_norm",
        "grad_tpr01fpr_head_norm",
        "grad_auc_weight_norm",
        "grad_tpr1fpr_weight_norm",
        "grad_tpr01fpr_weight_norm",
        "grad_auc_logit_norm",
        "grad_tpr1fpr_logit_norm",
        "grad_tpr01fpr_logit_norm",
        "repr_attack_protocol",
        "repr_attack_nonmember_source",
        "repr_attack_train_fraction",
        "repr_attack_epochs",
        "model_path",
        "forget_indices_path",
    ]
    existing_rows = []
    if path.exists() and path.stat().st_size > 0:
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            old_fields = reader.fieldnames or []
            existing_rows = list(reader)
        fieldnames = list(old_fields)
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for old_row in existing_rows:
            writer.writerow(old_row)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser(
        description="Train a fresh villain model from forget indices, then run MIA on both full-head and headless checkpoints."
    )
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "tinyimagenet", "mnist", "yale"])
    parser.add_argument("--expert_model", required=True, choices=["cnn", "rnn", "mlp", "resnet18", "resnet50", "lenet", "vit"])
    parser.add_argument("--forget_indices_path", required=True, help="Path to forget_indices_*.pt")
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--output_root", default="./villain_mia_runs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--expert_epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--repr_attack_epochs", type=int, default=30)
    parser.add_argument("--repr_attack_batch_size", type=int, default=256)
    parser.add_argument("--repr_attack_lr", type=float, default=1e-3)
    parser.add_argument("--repr_attack_train_fraction", type=float, default=0.5)
    parser.add_argument("--repr_attack_nonmember_source", choices=["test", "retain"], default="test")
    args = parser.parse_args()

    eval_script = Path(__file__).resolve().parent / "evaluate_mia.py"
    forget_indices_path = Path(args.forget_indices_path).resolve()
    output_root = Path(args.output_root).resolve() / f"{args.dataset}_{args.expert_model}_seed{args.seed}"
    if not eval_script.exists():
        raise FileNotFoundError(f"evaluate_mia.py not found: {eval_script}")
    if not forget_indices_path.exists():
        raise FileNotFoundError(f"forget indices not found: {forget_indices_path}")

    os_env_gpu = str(args.gpu)
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = os_env_gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(args.seed)

    print(f"[Villain MIA] dataset={args.dataset}, expert_model={args.expert_model}, seed={args.seed}")
    print(f"[Villain MIA] forget_indices={forget_indices_path}")
    print(f"[Villain MIA] output_root={output_root}")

    train_set, test_set, num_classes, in_channels, img_size = load_dataset(args.dataset, args.data_dir, train_augment=False)
    forget_indices = torch.load(forget_indices_path, map_location="cpu")
    if isinstance(forget_indices, torch.Tensor):
        forget_indices = forget_indices.tolist()

    forget_loader = DataLoader(Subset(train_set, forget_indices), batch_size=args.batch_size, shuffle=True)
    full_loader = DataLoader(train_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=128, shuffle=False)
    forget_eval_loader = DataLoader(Subset(train_set, forget_indices), batch_size=128, shuffle=False)

    model = get_model(args.expert_model, in_channels, num_classes, img_size).to(device)

    start = time.time()
    model = train_model(
        model,
        forget_loader,
        device,
        epochs=args.expert_epochs,
        lr=args.lr,
        patience=args.patience,
        desc=f"[Villain MIA] {args.dataset}/{args.expert_model}",
        verbose=True,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start

    train_acc, train_loss = evaluate_model(model, full_loader, device)
    test_acc, test_loss = evaluate_model(model, test_loader, device)
    forget_acc, forget_loss = evaluate_model(model, forget_eval_loader, device)

    saved_full = ensure_dir(output_root / "saved_models" / "full_head") / f"expert_{args.dataset}_{args.expert_model}_seed{args.seed}.pth"
    saved_headless = ensure_dir(output_root / "saved_models" / "headless") / f"expert_{args.dataset}_{args.expert_model}_seed{args.seed}.pth"
    metrics_dir = ensure_dir(output_root / "history")
    mia_full_dir = ensure_dir(output_root / "mia_history" / "full_head")
    mia_headless_dir = ensure_dir(output_root / "mia_history" / "headless")

    torch.save(model.state_dict(), saved_full)
    stripped_state, removed_head = strip_classifier_head(model.state_dict())
    torch.save(stripped_state, saved_headless)

    local_forget_copy = copy_forget_indices(forget_indices_path, metrics_dir)

    with open(metrics_dir / "training_metrics.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "dataset", "expert_model", "seed",
                "train_acc", "train_loss",
                "test_acc", "test_loss",
                "forget_acc", "forget_loss",
                "time_s", "removed_head_keys", "forget_indices_path",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "dataset": args.dataset,
            "expert_model": args.expert_model,
            "seed": args.seed,
            "train_acc": round(train_acc, 4),
            "train_loss": round(train_loss, 6),
            "test_acc": round(test_acc, 4),
            "test_loss": round(test_loss, 6),
            "forget_acc": round(forget_acc, 4),
            "forget_loss": round(forget_loss, 6),
            "time_s": round(elapsed, 2),
            "removed_head_keys": "|".join(removed_head),
            "forget_indices_path": str(local_forget_copy),
        })

    full_row = run_mia(
        eval_script,
        args.dataset,
        args.expert_model,
        args.data_dir,
        mia_full_dir,
        saved_full,
        forget_indices_path,
        args.seed,
        args.repr_attack_epochs,
        args.repr_attack_batch_size,
        args.repr_attack_lr,
        args.repr_attack_train_fraction,
        args.repr_attack_nonmember_source,
        True,
    )
    headless_row = run_mia(
        eval_script,
        args.dataset,
        args.expert_model,
        args.data_dir,
        mia_headless_dir,
        saved_headless,
        forget_indices_path,
        args.seed,
        args.repr_attack_epochs,
        args.repr_attack_batch_size,
        args.repr_attack_lr,
        args.repr_attack_train_fraction,
        args.repr_attack_nonmember_source,
        False,
    )

    combined = output_root / "summary_villain_mia.csv"
    common = {
        "dataset": args.dataset,
        "expert_model": args.expert_model,
        "seed": args.seed,
        "train_acc": round(train_acc, 4),
        "test_acc": round(test_acc, 4),
        "forget_acc": round(forget_acc, 4),
        "time_s": round(elapsed, 2),
        "forget_indices_path": str(forget_indices_path),
        "repr_attack_protocol": "forget_split",
        "repr_attack_nonmember_source": args.repr_attack_nonmember_source,
        "repr_attack_train_fraction": args.repr_attack_train_fraction,
        "repr_attack_epochs": args.repr_attack_epochs,
    }
    append_combined_summary(combined, {
        **common,
        "variant": "full_head",
        "asr_correctness": full_row["ASR_Correctness"],
        "asr_confidence": full_row["ASR_Confidence"],
        "asr_entropy": full_row["ASR_Entropy"],
        "asr_modentropy": full_row["ASR_ModEntropy"],
        "asr_average": full_row["ASR_Average"],
        "auc_correctness": full_row.get("AUC_Correctness", ""),
        "auc_confidence": full_row.get("AUC_Confidence", ""),
        "auc_entropy": full_row.get("AUC_Entropy", ""),
        "auc_modentropy": full_row.get("AUC_ModEntropy", ""),
        "tpr1fpr_correctness": full_row.get("TPR1FPR_Correctness", ""),
        "tpr1fpr_confidence": full_row.get("TPR1FPR_Confidence", ""),
        "tpr1fpr_entropy": full_row.get("TPR1FPR_Entropy", ""),
        "tpr1fpr_modentropy": full_row.get("TPR1FPR_ModEntropy", ""),
        "tpr01fpr_correctness": full_row.get("TPR01FPR_Correctness", ""),
        "tpr01fpr_confidence": full_row.get("TPR01FPR_Confidence", ""),
        "tpr01fpr_entropy": full_row.get("TPR01FPR_Entropy", ""),
        "tpr01fpr_modentropy": full_row.get("TPR01FPR_ModEntropy", ""),
        "repr_auc_l2_centroid": full_row.get("AUC_ReprL2Centroid", ""),
        "repr_tpr1fpr_l2_centroid": full_row.get("TPR1FPR_ReprL2Centroid", ""),
        "repr_tpr01fpr_l2_centroid": full_row.get("TPR01FPR_ReprL2Centroid", ""),
        "repr_auc_cosine_centroid": full_row.get("AUC_ReprCosineCentroid", ""),
        "repr_tpr1fpr_cosine_centroid": full_row.get("TPR1FPR_ReprCosineCentroid", ""),
        "repr_tpr01fpr_cosine_centroid": full_row.get("TPR01FPR_ReprCosineCentroid", ""),
        "repr_auc_norm": full_row.get("AUC_ReprNorm", ""),
        "repr_tpr1fpr_norm": full_row.get("TPR1FPR_ReprNorm", ""),
        "repr_tpr01fpr_norm": full_row.get("TPR01FPR_ReprNorm", ""),
        "repr_auc_attack_mlp": full_row.get("AUC_ReprAttackMLP", ""),
        "repr_tpr1fpr_attack_mlp": full_row.get("TPR1FPR_ReprAttackMLP", ""),
        "repr_tpr01fpr_attack_mlp": full_row.get("TPR01FPR_ReprAttackMLP", ""),
        "grad_auc_loss": full_row.get("AUC_GradLoss", ""),
        "grad_tpr1fpr_loss": full_row.get("TPR1FPR_GradLoss", ""),
        "grad_tpr01fpr_loss": full_row.get("TPR01FPR_GradLoss", ""),
        "grad_auc_head_norm": full_row.get("AUC_GradHeadNorm", ""),
        "grad_tpr1fpr_head_norm": full_row.get("TPR1FPR_GradHeadNorm", ""),
        "grad_tpr01fpr_head_norm": full_row.get("TPR01FPR_GradHeadNorm", ""),
        "grad_auc_weight_norm": full_row.get("AUC_GradWeightNorm", ""),
        "grad_tpr1fpr_weight_norm": full_row.get("TPR1FPR_GradWeightNorm", ""),
        "grad_tpr01fpr_weight_norm": full_row.get("TPR01FPR_GradWeightNorm", ""),
        "grad_auc_logit_norm": full_row.get("AUC_GradLogitNorm", ""),
        "grad_tpr1fpr_logit_norm": full_row.get("TPR1FPR_GradLogitNorm", ""),
        "grad_tpr01fpr_logit_norm": full_row.get("TPR01FPR_GradLogitNorm", ""),
        "model_path": str(saved_full),
    })
    append_combined_summary(combined, {
        **common,
        "variant": "headless",
        "asr_correctness": headless_row["ASR_Correctness"],
        "asr_confidence": headless_row["ASR_Confidence"],
        "asr_entropy": headless_row["ASR_Entropy"],
        "asr_modentropy": headless_row["ASR_ModEntropy"],
        "asr_average": headless_row["ASR_Average"],
        "auc_correctness": headless_row.get("AUC_Correctness", ""),
        "auc_confidence": headless_row.get("AUC_Confidence", ""),
        "auc_entropy": headless_row.get("AUC_Entropy", ""),
        "auc_modentropy": headless_row.get("AUC_ModEntropy", ""),
        "tpr1fpr_correctness": headless_row.get("TPR1FPR_Correctness", ""),
        "tpr1fpr_confidence": headless_row.get("TPR1FPR_Confidence", ""),
        "tpr1fpr_entropy": headless_row.get("TPR1FPR_Entropy", ""),
        "tpr1fpr_modentropy": headless_row.get("TPR1FPR_ModEntropy", ""),
        "tpr01fpr_correctness": headless_row.get("TPR01FPR_Correctness", ""),
        "tpr01fpr_confidence": headless_row.get("TPR01FPR_Confidence", ""),
        "tpr01fpr_entropy": headless_row.get("TPR01FPR_Entropy", ""),
        "tpr01fpr_modentropy": headless_row.get("TPR01FPR_ModEntropy", ""),
        "repr_auc_l2_centroid": headless_row.get("AUC_ReprL2Centroid", ""),
        "repr_tpr1fpr_l2_centroid": headless_row.get("TPR1FPR_ReprL2Centroid", ""),
        "repr_tpr01fpr_l2_centroid": headless_row.get("TPR01FPR_ReprL2Centroid", ""),
        "repr_auc_cosine_centroid": headless_row.get("AUC_ReprCosineCentroid", ""),
        "repr_tpr1fpr_cosine_centroid": headless_row.get("TPR1FPR_ReprCosineCentroid", ""),
        "repr_tpr01fpr_cosine_centroid": headless_row.get("TPR01FPR_ReprCosineCentroid", ""),
        "repr_auc_norm": headless_row.get("AUC_ReprNorm", ""),
        "repr_tpr1fpr_norm": headless_row.get("TPR1FPR_ReprNorm", ""),
        "repr_tpr01fpr_norm": headless_row.get("TPR01FPR_ReprNorm", ""),
        "repr_auc_attack_mlp": headless_row.get("AUC_ReprAttackMLP", ""),
        "repr_tpr1fpr_attack_mlp": headless_row.get("TPR1FPR_ReprAttackMLP", ""),
        "repr_tpr01fpr_attack_mlp": headless_row.get("TPR01FPR_ReprAttackMLP", ""),
        "grad_auc_loss": headless_row.get("AUC_GradLoss", ""),
        "grad_tpr1fpr_loss": headless_row.get("TPR1FPR_GradLoss", ""),
        "grad_tpr01fpr_loss": headless_row.get("TPR01FPR_GradLoss", ""),
        "grad_auc_head_norm": headless_row.get("AUC_GradHeadNorm", ""),
        "grad_tpr1fpr_head_norm": headless_row.get("TPR1FPR_GradHeadNorm", ""),
        "grad_tpr01fpr_head_norm": headless_row.get("TPR01FPR_GradHeadNorm", ""),
        "grad_auc_weight_norm": headless_row.get("AUC_GradWeightNorm", ""),
        "grad_tpr1fpr_weight_norm": headless_row.get("TPR1FPR_GradWeightNorm", ""),
        "grad_tpr01fpr_weight_norm": headless_row.get("TPR01FPR_GradWeightNorm", ""),
        "grad_auc_logit_norm": headless_row.get("AUC_GradLogitNorm", ""),
        "grad_tpr1fpr_logit_norm": headless_row.get("TPR1FPR_GradLogitNorm", ""),
        "grad_tpr01fpr_logit_norm": headless_row.get("TPR01FPR_GradLogitNorm", ""),
        "model_path": str(saved_headless),
    })

    print("\n[Villain MIA] Done.")
    print(f"  Full-head ASR average : {full_row['ASR_Average']}")
    print(f"  Headless ASR average  : {headless_row['ASR_Average']}")
    print(f"  Combined summary      : {combined}")


if __name__ == "__main__":
    main()
