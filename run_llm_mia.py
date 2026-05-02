import argparse
import csv
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from evaluate_mia import MIABenchmarks
from vilun_llm import LLMVisionClassifier, load_cifar10, sanitize_tag, set_seed


@torch.no_grad()
def collect_probs(model, loader, device):
    model.eval()
    probs_all = []
    labels_all = []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        logits = model(imgs).float()
        probs_all.append(F.softmax(logits, dim=-1).cpu())
        labels_all.append(labels.cpu())
    return torch.cat(probs_all).numpy(), torch.cat(labels_all).numpy()


def resolve_paths(args):
    orig_tag = sanitize_tag(args.orig_model.split("/")[-1] if "/" in args.orig_model else args.orig_model)
    save_tag = args.save_tag or f"llm_vilun_cifar10_{sanitize_tag(args.expert_model)}_seed{args.seed}"
    model_path = args.model_path or os.path.join(
        args.save_path,
        f"unlearned_vilun_llm_heldout_{save_tag}.pt",
    )
    forget_indices_path = args.forget_indices_path or os.path.join(
        args.history_dir,
        f"forget_indices_vilun_llm_cifar10_{orig_tag}_{args.expert_model}_seed{args.seed}.pt",
    )
    return save_tag, model_path, forget_indices_path


def append_summary(args, save_tag, model_path, forget_indices_path, results, n_forget):
    os.makedirs(args.history_dir, exist_ok=True)
    path = os.path.join(args.history_dir, "summary_mia_llm.csv")
    exists = os.path.exists(path) and os.path.getsize(path) > 0
    header = [
        "Method", "Dataset", "Orig_Model", "Expert_Model", "Save_Tag",
        "N_Forget", "Forget_Indices_Path", "Seed",
        "ASR_Correctness", "ASR_Confidence", "ASR_Entropy",
        "ASR_ModEntropy", "ASR_Average", "Model_Path",
    ]
    with open(path, "a", newline="") as f:
        writer = csv.writer(f)
        if not exists:
            writer.writerow(header)
        writer.writerow([
            "ViLUN-LLM-HeldOut", "cifar10", args.orig_model, args.expert_model,
            save_tag, n_forget, forget_indices_path, args.seed,
            round(results["correctness"]["asr"], 4),
            round(results["confidence"]["asr"], 4),
            round(results["entropy"]["asr"], 4),
            round(results["modified_entropy"]["asr"], 4),
            round(results["average_asr"], 4),
            model_path,
        ])
    print(f"[Save] LLM MIA summary -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--orig_model", default="Qwen/Qwen2.5-3B")
    parser.add_argument("--expert_model", required=True)
    parser.add_argument("--model_path", default=None)
    parser.add_argument("--forget_indices_path", default=None)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--save_path", default="./saved_models")
    parser.add_argument("--history_dir", default="./history")
    parser.add_argument("--save_tag", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--llm_layers", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    set_seed(args.seed)
    save_tag, model_path, forget_indices_path = resolve_paths(args)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"LLM unlearned model not found: {model_path}")
    if not os.path.exists(forget_indices_path):
        raise FileNotFoundError(f"LLM forget indices not found: {forget_indices_path}")

    train_llm, test_llm, _, _ = load_cifar10(args.data_dir)
    forget_idx = torch.load(forget_indices_path, map_location="cpu").long()
    forget_set = set(forget_idx.tolist())
    retain_idx = [i for i in range(len(train_llm)) if i not in forget_set]

    retain_loader = DataLoader(Subset(train_llm, retain_idx), batch_size=args.batch_size, shuffle=False)
    forget_loader = DataLoader(Subset(train_llm, forget_idx.tolist()), batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(test_llm, batch_size=args.batch_size, shuffle=False)

    model = LLMVisionClassifier(
        model_name=args.orig_model,
        num_classes=10,
        num_layers=args.llm_layers,
        device=args.device,
    ).to(args.device)
    state = torch.load(model_path, map_location=args.device)
    model.load_state_dict(state, strict=True)

    retain_perf = collect_probs(model, retain_loader, args.device)
    test_perf = collect_probs(model, test_loader, args.device)
    forget_perf = collect_probs(model, forget_loader, args.device)

    mia = MIABenchmarks(retain_perf, test_perf, forget_perf, 10, correctness_mode="threshold")
    results = mia.run_attacks()
    append_summary(args, save_tag, model_path, forget_indices_path, results, len(forget_idx))


if __name__ == "__main__":
    main()
