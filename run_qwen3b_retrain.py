import argparse
import csv
import glob
import os
import time

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from vilun_llm import (
    LLMVisionClassifier,
    evaluate_full,
    load_cifar10,
    sanitize_tag,
    set_seed,
)


def resolve_forget_indices_path(args):
    if args.forget_indices_path:
        return args.forget_indices_path
    orig_tag = sanitize_tag(args.orig_model.split("/")[-1] if "/" in args.orig_model else args.orig_model)
    pattern = os.path.join(args.history_dir, f"forget_indices_vilun_llm_cifar10_{orig_tag}_*_seed{args.seed}.pt")
    candidates = sorted(glob.glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No forget-indices file found. Pass --forget_indices_path explicitly or place a file matching {pattern}"
        )
    return candidates[0]


def make_loader(dataset, indices, batch_size, shuffle):
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=shuffle)


def save_summary(summary_path, row):
    header = [
        "Method", "Dataset", "Orig_Model", "Seed", "Save_Tag", "Final_Epoch",
        "Final_Train_Retain_Acc", "Final_Test_Retain_Acc", "Final_Forget_Acc",
        "Time_Retrain(s)",
        "Model_Path", "Forget_Indices_Path"
    ]
    write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    with open(summary_path, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow(row)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_model", type=str, default="Qwen/Qwen2.5-3B")
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--history_dir", type=str, default="./history")
    parser.add_argument("--save_dir", type=str, default="./saved_models")
    parser.add_argument("--save_tag", type=str, default="retrain_llm_qwen3b_seed42")
    parser.add_argument("--forget_indices_path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--llm_layers", type=int, default=8)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    os.makedirs(args.history_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    print("[Step 1] Loading CIFAR-10 for LLM retrain...")
    train_llm, test_llm, _, _ = load_cifar10(args.data_dir)

    forget_indices_path = resolve_forget_indices_path(args)
    forget_indices = torch.load(forget_indices_path, map_location="cpu").long()
    forget_idx_set = set(forget_indices.tolist())
    retain_indices = [i for i in range(len(train_llm)) if i not in forget_idx_set]

    print(f"[Load] Forget indices <- {forget_indices_path} ({len(forget_indices)} samples)")
    print(f"[Info] Retain subset size: {len(retain_indices)}")

    retain_loader_train = make_loader(train_llm, retain_indices, args.batch_size, True)

    print(f"[Step 2] Initializing retrain oracle on {args.orig_model}...")
    model = LLMVisionClassifier(
        model_name=args.orig_model,
        num_classes=10,
        num_layers=args.llm_layers,
        device=args.device,
    )
    model = model.to(args.device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    history_rows = []

    print("[Step 3] Retraining on retain subset only...")
    start = time.time()
    for epoch in range(1, args.train_epochs + 1):
        model.train()
        epoch_loss = 0.0
        seen = 0
        for imgs, labels in retain_loader_train:
            imgs = imgs.to(args.device)
            labels = labels.to(args.device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = F.cross_entropy(out.float(), labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_loss += loss.item() * labels.size(0)
            seen += labels.size(0)

        res = evaluate_full(model, train_llm, test_llm, forget_indices, args.device, verbose=False)

        row = {
            "epoch": epoch,
            "train_forget_acc": res["Train_Forget"]["acc"],
            "train_forget_loss": res["Train_Forget"]["loss"],
            "train_retain_acc": res["Train_Retain"]["acc"],
            "train_retain_loss": res["Train_Retain"]["loss"],
            "test_retain_acc": res["Test_Retain "]["acc"],
            "test_retain_loss": res["Test_Retain "]["loss"],
        }
        history_rows.append(row)

        print(
            f"  Epoch [{epoch:>3}/{args.train_epochs}]"
            f" | Train_Retain: {row['train_retain_acc']:6.2f}%"
            f" | Test_Retain: {row['test_retain_acc']:6.2f}%"
            f" | Train_Forget: {row['train_forget_acc']:6.2f}%"
        )

    total_time = time.time() - start

    history_csv = os.path.join(args.history_dir, f"{args.save_tag}.csv")
    with open(history_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(history_rows[0].keys()))
        writer.writeheader()
        writer.writerows(history_rows)

    model_path = os.path.join(args.save_dir, f"{args.save_tag}.pth")
    torch.save(model.state_dict(), model_path)

    final_row = history_rows[-1]
    summary_path = os.path.join(args.history_dir, "summary_retrain_llm.csv")
    save_summary(summary_path, [
        "Retrain-LLM", "cifar10", args.orig_model, args.seed, args.save_tag, final_row['epoch'],
        f"{final_row['train_retain_acc']:.2f}",
        f"{final_row['test_retain_acc']:.2f}",
        f"{final_row['train_forget_acc']:.2f}",
        f"{total_time:.1f}",
        model_path,
        forget_indices_path,
    ])

    print(f"[Save] History -> {history_csv}")
    print(f"[Save] Model   -> {model_path}")
    print(f"[Save] Summary -> {summary_path}")
    print(
        f"[Final] Epoch {final_row['epoch']}"
        f" | D_r={final_row['train_retain_acc']:.2f}"
        f" | D_t={final_row['test_retain_acc']:.2f}"
        f" | D_f={final_row['train_forget_acc']:.2f}"
    )


if __name__ == '__main__':
    main()
