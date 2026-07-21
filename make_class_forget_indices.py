import argparse
import os
import torch

from pretrain_models import load_dataset


def labels_from_dataset(dataset):
    if hasattr(dataset, "targets"):
        targets = dataset.targets
        return targets.detach().cpu().long() if isinstance(targets, torch.Tensor) else torch.tensor(targets).long()
    if hasattr(dataset, "tensors"):
        return dataset.tensors[1].detach().cpu().long()
    return torch.tensor([int(y) for _, y in dataset]).long()


def save_indices(path, indices):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(indices.cpu().long(), path)
    print(f"[Save] {path} ({len(indices)} samples)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, choices=["cifar10", "cifar100", "tinyimagenet", "mnist", "yale"])
    parser.add_argument("--model", required=True)
    parser.add_argument("--data_dir", default="./data")
    parser.add_argument("--history_dir", default="./history")
    parser.add_argument("--target_class", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--also_write_baseline_names", action="store_true")
    args = parser.parse_args()

    train_set, _, _, _, _ = load_dataset(args.dataset, args.data_dir, train_augment=False)
    labels = labels_from_dataset(train_set)
    indices = (labels == args.target_class).nonzero(as_tuple=True)[0]
    if len(indices) == 0:
        raise ValueError(f"No samples found for class {args.target_class} in {args.dataset}")

    names = [f"forget_indices_{args.dataset}_{args.model}_seed{args.seed}.pt"]
    if args.also_write_baseline_names:
        names.extend([
            f"forget_indices_ps_{args.dataset}_{args.model}_seed{args.seed}.pt",
            f"forget_indices_delete_{args.dataset}_{args.model}_seed{args.seed}.pt",
            f"forget_indices_salun_{args.dataset}_{args.model}_seed{args.seed}.pt",
            f"forget_indices_sisa_{args.dataset}_{args.model}_seed{args.seed}.pt",
        ])

    for name in names:
        path = os.path.join(args.history_dir, name)
        save_indices(path, indices)


if __name__ == "__main__":
    main()
