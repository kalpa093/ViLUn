"""
evaluate_mia.py
---------------
Membership-inference evaluation for ViLUn and the comparison methods.

Supports the following model types (via --method):
  vilun / vilun_heldout : ViLUn_f / ViLUn checkpoints
  original / retrain    : reference checkpoints
  gradient_ascent       : Gradient Ascent checkpoint
  delete / ps / salun   : approximate-unlearning checkpoints
  sisa                  : SISA shard container
  expert                : full-head or headless villain checkpoint

MIA shadow setup:
  Shadow train : retain set of train split  (known members)
  Shadow test  : test set                   (known non-members)
  Target       : forget set of train split  (test residual membership)

Results are appended to the requested history directory as summary_mia.csv.
"""

import argparse
import os
import csv
import copy
try:
    import fcntl
except ImportError:                                                
    fcntl = None
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from PIL import Image
import glob
import random

                                                                
                                 
                                                                

class ResNet18Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x):
        return self.model(x)

class ResNet50Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=None)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(2048, num_classes)
    def forward(self, x):
        return self.model(x)

class CNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))

class LeNetNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(400, 120); self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)
    def forward(self, x):
        x = self.features(x); x = self.pool(x); x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
        return self.classifier(x)

class ViTNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.vit_b_16(weights=None)
        if in_channels != 3:
            self.model.conv_proj = nn.Conv2d(in_channels, 768, 16, 16)
        self.model.heads.head = nn.Linear(768, num_classes)
    def forward(self, x):
        if x.size(-1) < 224:
            x = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
        return self.model(x)

class MLPNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Linear(in_channels * img_size * img_size, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU()
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x):
        return self.classifier(self.mixer(torch.flatten(x, 1)))

class RNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.lstm = nn.LSTM(in_channels * img_size, 128, 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x):
        B, C, H, W = x.size()
        if H != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size))
            B, C, H, W = x.size()
        _, (h_n, _) = self.lstm(x.permute(0,2,3,1).reshape(B, H, W*C))
        return self.classifier(torch.cat((h_n[-2], h_n[-1]), dim=1))

def get_model(model_name, in_channels, num_classes, img_size=32):
    m = {
        'cnn':      CNNNetwork(in_channels, num_classes),
        'resnet18': ResNet18Network(in_channels, num_classes),
        'resnet50': ResNet50Network(in_channels, num_classes),
        'lenet':    LeNetNetwork(in_channels, num_classes),
        'vit':      ViTNetwork(in_channels, num_classes),
        'mlp':      MLPNetwork(in_channels, num_classes, img_size),
        'rnn':      RNNNetwork(in_channels, num_classes, img_size),
    }
    if model_name not in m:
        raise ValueError(f"Unknown model: {model_name}")
    return m[model_name]

                                                                
                                          
                                                                

def load_yale_custom(root_dir, img_size=64):
    for pattern in [os.path.join(root_dir, "subject*"),
                    os.path.join(root_dir, "yalefaces", "subject*")]:
        file_list = glob.glob(pattern)
        if file_list: break
    if not file_list:
        X = torch.randn(100, 1, img_size, img_size)
        y = torch.randint(0, 15, (100,))
        return TensorDataset(X, y), 15, 1
    tf = transforms.Compose([transforms.Grayscale(1),
                              transforms.Resize((img_size, img_size)),
                              transforms.ToTensor()])
    X_list, y_list = [], []
    for f in file_list:
        if not os.path.isfile(f): continue
        try:
            num = int(os.path.basename(f).split(".")[0].replace("subject","")) - 1
            X_list.append(tf(Image.open(f)))
            y_list.append(torch.tensor(num, dtype=torch.long))
        except: continue
    ds = TensorDataset(torch.stack(X_list), torch.stack(y_list))
    return ds, len(torch.unique(torch.stack(y_list))), 1


def find_tinyimagenet_root(data_dir):
    project_dir = os.path.dirname(os.path.abspath(data_dir))
    candidates = [
        os.path.join(data_dir, 'tiny-imagenet-200'),
        os.path.join(data_dir, 'tinyimagenet', 'tiny-imagenet-200'),
        os.path.join(data_dir, 'TinyImageNet', 'tiny-imagenet-200'),
        os.path.join(data_dir, 'tinyimagenet'),
        os.path.join(project_dir, 'tiny-imagenet-200'),
        os.path.join(project_dir, 'tinyimagenet', 'tiny-imagenet-200'),
    ]
    for path in candidates:
        if os.path.isdir(os.path.join(path, 'train')) and os.path.isdir(os.path.join(path, 'val')):
            return path
    raise FileNotFoundError("TinyImageNet not found under data_dir; expected tiny-imagenet-200/train and val.")


class TinyImageNetValDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.transform = transform
        ann_path = os.path.join(root, 'val', 'val_annotations.txt')
        img_dir = os.path.join(root, 'val', 'images')
        self.samples = []
        with open(ann_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) >= 2 and parts[1] in class_to_idx:
                    self.samples.append((os.path.join(img_dir, parts[0]), class_to_idx[parts[1]]))
        self.targets = [y for _, y in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target

def load_datasets(dataset, data_dir, seed=42):
    """Returns (train_set, test_set, num_classes, in_channels, img_size)."""
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if dataset == 'yale':
        full, nc, ic = load_yale_custom(os.path.join(data_dir, 'yale'))
        return full, full, nc, ic, 64

    if dataset == 'tinyimagenet':
        root = find_tinyimagenet_root(data_dir)
        tf = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        tr = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform=tf)
        te = TinyImageNetValDataset(root, tr.class_to_idx, transform=tf)
        return tr, te, 200, 3, 64

    configs = {
        'cifar10':  (torchvision.datasets.CIFAR10,
                     transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010)),
                     10, 3, 32),
        'cifar100': (torchvision.datasets.CIFAR100,
                     transforms.Normalize((0.5071,0.4867,0.4408),(0.2675,0.2565,0.2761)),
                     100, 3, 32),
        'mnist':    (torchvision.datasets.MNIST,
                     transforms.Normalize((0.1307,),(0.3081,)),
                     10, 1, 32),
    }
    cls, norm, nc, ic, sz = configs[dataset]
    tf = transforms.Compose([transforms.Resize((sz,sz)), transforms.ToTensor(), norm])
    tr = cls(root=data_dir, train=True,  download=True, transform=tf)
    te = cls(root=data_dir, train=False, download=True, transform=tf)
    return tr, te, nc, ic, sz

def get_labels(dataset):
    if hasattr(dataset, 'targets'):
        t = dataset.targets
        return t.clone() if isinstance(t, torch.Tensor) else torch.tensor(t)
    if hasattr(dataset, 'tensors'):
        return dataset.tensors[1]
    return torch.tensor([y for _, y in dataset])

                                                                
                       
                                                                

def is_classifier_head_key(key):
    return (
        key in {'classifier.weight', 'classifier.bias'}
        or key in {'model.fc.weight', 'model.fc.bias'}
        or key in {'model.heads.head.weight', 'model.heads.head.bias'}
    )


def load_single_model(path, model_name, in_channels, num_classes, img_size, device,
                      allow_missing_head=False):
    """Load a single .pth state-dict (plain dict or wrapped)."""
    model = get_model(model_name, in_channels, num_classes, img_size).to(device)
    ckpt = torch.load(path, map_location=device)
                                                        
    if isinstance(ckpt, dict):
        if 'model_state_dict' in ckpt:
            sd = ckpt['model_state_dict']
        elif 'state_dict' in ckpt:
            sd = ckpt['state_dict']
        else:
            sd = ckpt
    else:
        sd = ckpt
    result = model.load_state_dict(sd, strict=not allow_missing_head)
    if allow_missing_head:
        missing = list(getattr(result, 'missing_keys', []))
        unexpected = list(getattr(result, 'unexpected_keys', []))
        non_head_missing = [key for key in missing if not is_classifier_head_key(key)]
        if non_head_missing or unexpected:
            raise RuntimeError(
                f"Could not load checkpoint {path}. "
                f"Missing non-head keys: {non_head_missing}; unexpected keys: {unexpected}"
            )
        head_missing = [key for key in missing if is_classifier_head_key(key)]
        if head_missing:
            print(f"  [Expert MIA] Checkpoint has no logit head; evaluating with random unused head: {head_missing}")
        model._loaded_missing_head = bool(head_missing)
    else:
        model._loaded_missing_head = False
    model.eval()
    return model

def load_sisa_ensemble(container_dir, dataset, model_name,
                       in_channels, num_classes, img_size,
                       num_shards, num_slices, device):
    """
    Load SISA ensemble: last slice of every shard.
    File pattern: <dataset>_<model>_<seed>_shard<S>_slice<L>.pth
    """
    models = []
    for shard_id in range(num_shards):
        pattern = os.path.join(
            container_dir,
            f"*shard{shard_id}_slice{num_slices-1}.pth"
        )
        matches = glob.glob(pattern)
        if not matches:
            print(f"  [WARN] No checkpoint for shard {shard_id} in {container_dir}")
            continue
        m = get_model(model_name, in_channels, num_classes, img_size).to(device)
        ckpt = torch.load(matches[0], map_location=device)
        sd = ckpt.get('model_state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
        m.load_state_dict(sd)
        m.eval()
        models.append(m)
    if not models:
        raise RuntimeError(f"No shard models found in {container_dir}")
    print(f"  Loaded {len(models)}/{num_shards} SISA shards.")
    return models

                                                                
                   
                                                                

@torch.no_grad()
def collect_probs(models_or_model, loader, device):
    """
    Collect softmax probabilities and labels.
    Accepts either a single model or a list of models (SISA ensemble).
    """
    is_list = isinstance(models_or_model, list)
    if not is_list:
        models_or_model.eval()

    all_probs, all_labels = [], []
    for batch in loader:
        imgs, labels = batch[0], batch[1]
        imgs = imgs.to(device)

        if is_list:
            logits_sum = sum(m(imgs) for m in models_or_model)
            probs = F.softmax(logits_sum, dim=-1)
        else:
            probs = F.softmax(models_or_model(imgs), dim=-1)

        all_probs.append(probs.cpu())
        all_labels.append(labels)

    return torch.cat(all_probs).numpy(), torch.cat(all_labels).numpy()


def _average_ranks(values):
    """Return 1-based average ranks with tie handling."""
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    i = 0
    while i < len(values):
        j = i + 1
        while j < len(values) and values[order[j]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + 1 + j)
        ranks[order[i:j]] = avg_rank
        i = j
    return ranks


def binary_score_metrics(member_scores, nonmember_scores, fprs=(0.01, 0.001)):
    """
    ROC-style metrics for scores where larger means "more likely member".
    Members are the target forget samples; non-members are held-out test samples.

    Low-FPR TPR is selected only from attainable ROC operating points whose
    realized FPR does not exceed the requested budget. Equal scores are handled
    as one threshold group, so ties cannot silently inflate the actual FPR.
    """
    pos = np.asarray(member_scores, dtype=np.float64)
    neg = np.asarray(nonmember_scores, dtype=np.float64)
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        out = {"auc": float("nan")}
        for fpr in fprs:
            out[f"tpr_at_{fpr:g}_fpr"] = float("nan")
            out[f"actual_fpr_at_{fpr:g}_fpr"] = float("nan")
        return out

    scores = np.concatenate([pos, neg])
    ranks = _average_ranks(scores)
    pos_ranks = ranks[:len(pos)]
    auc = (pos_ranks.sum() - len(pos) * (len(pos) + 1) / 2.0) / (len(pos) * len(neg))
    out = {"auc": float(auc)}

    labels = np.concatenate([
        np.ones(len(pos), dtype=np.int64),
        np.zeros(len(neg), dtype=np.int64),
    ])
    order = np.argsort(-scores, kind="mergesort")
    sorted_scores = scores[order]
    sorted_labels = labels[order]

    # A deterministic threshold includes every sample tied at that score.
    # Evaluate the ROC only after each complete tie group has been included.
    group_ends = np.flatnonzero(np.r_[sorted_scores[1:] != sorted_scores[:-1], True])
    cumulative_tp = np.cumsum(sorted_labels)[group_ends]
    cumulative_fp = np.cumsum(1 - sorted_labels)[group_ends]
    roc_tpr = np.r_[0.0, cumulative_tp / len(pos)]
    roc_fpr = np.r_[0.0, cumulative_fp / len(neg)]

    for target_fpr in fprs:
        if target_fpr < 0 or target_fpr > 1:
            raise ValueError(f"FPR target must be in [0, 1], got {target_fpr}")
        valid = roc_fpr <= float(target_fpr) + 1e-15
        best_tpr = float(np.max(roc_tpr[valid]))
        best_candidates = np.flatnonzero(valid & np.isclose(roc_tpr, best_tpr, rtol=0.0, atol=1e-15))
        best_idx = best_candidates[np.argmin(roc_fpr[best_candidates])]
        key = f"{target_fpr:g}"
        out[f"tpr_at_{key}_fpr"] = best_tpr
        out[f"actual_fpr_at_{key}_fpr"] = float(roc_fpr[best_idx])
    return out


def flatten_metric_dict(metrics, prefix):
    flat = {}
    for key, vals in metrics.items():
        if not isinstance(vals, dict):
            continue
        for metric_name, metric_value in vals.items():
            flat[f"{prefix}_{key}_{metric_name}"] = metric_value
    return flat


def get_classifier_module(model):
    """Find the final classification layer so we can hook its input features."""
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        return model.classifier
    if hasattr(model, "model"):
        inner = model.model
        if hasattr(inner, "fc") and isinstance(inner.fc, nn.Module):
            return inner.fc
        if hasattr(inner, "heads") and hasattr(inner.heads, "head"):
            return inner.heads.head
    raise ValueError(f"Could not locate classifier module for {model.__class__.__name__}")


@torch.no_grad()
def collect_head_features(model, loader, device):
    """
    Collect the representation immediately before the classification head.
    This works for headless checkpoints too: the randomly initialized head is
    only used as a hook target, while the recorded feature is produced by the
    transferred backbone.
    """
    model.eval()
    classifier = get_classifier_module(model)
    all_features, all_labels = [], []
    captured = []

    def pre_hook(_module, inputs):
        captured.append(inputs[0].detach())

    handle = classifier.register_forward_pre_hook(pre_hook)
    try:
        for batch in loader:
            imgs, labels = batch[0], batch[1]
            imgs = imgs.to(device)
            captured.clear()
            _ = model(imgs)
            if not captured:
                raise RuntimeError("Feature hook did not capture classifier input.")
            feats = captured[-1]
            all_features.append(feats.reshape(feats.size(0), -1).cpu())
            all_labels.append(labels)
    finally:
        handle.remove()

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


@torch.no_grad()
def collect_ensemble_head_features(models, loader, device):
    """Collect the mean pre-head representation of a same-architecture ensemble."""
    if not models:
        raise ValueError("Cannot collect representations from an empty ensemble.")

    captures = [[] for _ in models]
    handles = []
    for model_idx, model in enumerate(models):
        model.eval()
        classifier = get_classifier_module(model)

        def pre_hook(_module, inputs, idx=model_idx):
            captures[idx].append(inputs[0].detach())

        handles.append(classifier.register_forward_pre_hook(pre_hook))

    all_features, all_labels = [], []
    try:
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1]
            per_model_features = []
            for model_idx, model in enumerate(models):
                captures[model_idx].clear()
                _ = model(imgs)
                if not captures[model_idx]:
                    raise RuntimeError(
                        f"Feature hook did not capture classifier input for ensemble model {model_idx}."
                    )
                feats = captures[model_idx][-1]
                per_model_features.append(feats.reshape(feats.size(0), -1))

            feature_dims = {features.size(1) for features in per_model_features}
            if len(feature_dims) != 1:
                raise ValueError(
                    "Ensemble representation MIA requires all shard models to use the same feature dimension."
                )
            mean_features = torch.stack(per_model_features, dim=0).mean(dim=0)
            all_features.append(mean_features.cpu())
            all_labels.append(labels)
    finally:
        for handle in handles:
            handle.remove()

    return torch.cat(all_features).numpy(), torch.cat(all_labels).numpy()


def l2_normalize_np(x, eps=1e-12):
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.maximum(norm, eps)


def class_centroid_scores(ref_feats, ref_labels, query_feats, query_labels):
    """
    Representation-level MIA scores from distance/similarity to class centroids
    estimated on shadow member data. Larger scores mean more member-like.
    """
    ref_feats = np.asarray(ref_feats, dtype=np.float64)
    query_feats = np.asarray(query_feats, dtype=np.float64)
    ref_labels = np.asarray(ref_labels)
    query_labels = np.asarray(query_labels)

    global_centroid = ref_feats.mean(axis=0)
    centroids = {}
    for cls in np.unique(ref_labels):
        mask = ref_labels == cls
        if mask.any():
            centroids[int(cls)] = ref_feats[mask].mean(axis=0)

    query_centroids = np.stack([
        centroids.get(int(lbl), global_centroid) for lbl in query_labels
    ])
    l2_scores = -np.linalg.norm(query_feats - query_centroids, axis=1)

    q_norm = l2_normalize_np(query_feats)
    c_norm = l2_normalize_np(query_centroids)
    cosine_scores = np.sum(q_norm * c_norm, axis=1)
    norm_scores = np.linalg.norm(query_feats, axis=1)
    return {
        "repr_l2_centroid": l2_scores,
        "repr_cosine_centroid": cosine_scores,
        "repr_norm": norm_scores,
    }


def representation_mia_metrics(retain_feats, retain_labels, test_feats, test_labels,
                               forget_feats, forget_labels):
    test_scores = class_centroid_scores(retain_feats, retain_labels, test_feats, test_labels)
    forget_scores = class_centroid_scores(retain_feats, retain_labels, forget_feats, forget_labels)
    metrics = {}
    for key in forget_scores:
        metrics[key] = binary_score_metrics(forget_scores[key], test_scores[key])
    return metrics


@torch.no_grad()
def collect_head_gradient_scores(model, loader, device, num_classes):
    """
    Compute per-sample CE-gradient summaries for the trained classification head.
    Larger returned scores mean "more member-like"; since members usually have
    lower loss and smaller gradients, we negate loss/gradient norms.
    """
    model.eval()
    classifier = get_classifier_module(model)
    all_scores = {
        "grad_loss": [],
        "grad_head_norm": [],
        "grad_weight_norm": [],
        "grad_logit_norm": [],
    }
    captured = []

    def pre_hook(_module, inputs):
        captured.append(inputs[0].detach())

    handle = classifier.register_forward_pre_hook(pre_hook)
    try:
        for batch in loader:
            imgs, labels = batch[0].to(device), batch[1].to(device)
            captured.clear()
            logits = model(imgs)
            if not captured:
                raise RuntimeError("Feature hook did not capture classifier input.")
            feats = captured[-1].reshape(logits.size(0), -1)
            labels = labels.long()

            losses = F.cross_entropy(logits, labels, reduction="none")
            probs = F.softmax(logits, dim=-1)
            one_hot = F.one_hot(labels, num_classes=num_classes).to(probs.dtype)
            delta = probs - one_hot

            grad_logit_norm = torch.linalg.vector_norm(delta, dim=1)
            feat_norm = torch.linalg.vector_norm(feats, dim=1)
            grad_weight_norm = grad_logit_norm * feat_norm
            grad_head_norm = torch.sqrt(grad_weight_norm.pow(2) + grad_logit_norm.pow(2))

            all_scores["grad_loss"].append((-losses).cpu())
            all_scores["grad_head_norm"].append((-grad_head_norm).cpu())
            all_scores["grad_weight_norm"].append((-grad_weight_norm).cpu())
            all_scores["grad_logit_norm"].append((-grad_logit_norm).cpu())
    finally:
        handle.remove()

    return {key: torch.cat(vals).numpy() for key, vals in all_scores.items()}


def gradient_mia_metrics(test_scores, forget_scores):
    """
    Gradient-based white-box MIA scores for full-head models.
    Members are target forget samples and non-members are held-out test samples.
    """
    metrics = {}
    for key in forget_scores:
        metrics[key] = binary_score_metrics(forget_scores[key], test_scores[key])
    return metrics


class RepresentationAttackMLP(nn.Module):
    """Attack classifier over backbone embeddings."""
    def __init__(self, input_dim):
        super().__init__()
        hidden = min(256, max(64, input_dim // 2))
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),
            nn.Linear(hidden, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _split_permutation(n, train_fraction, seed):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    split = int(round(train_fraction * n))
    split = min(max(split, 1), n - 1) if n > 1 else n
    return perm[:split], perm[split:]


def _balanced_attack_arrays(member_feats, member_labels, nonmember_feats, nonmember_labels,
                            train_fraction, seed):
    mem_train_idx, mem_eval_idx = _split_permutation(len(member_feats), train_fraction, seed)
    non_train_idx, non_eval_idx = _split_permutation(len(nonmember_feats), train_fraction, seed + 17)

    n_train = min(len(mem_train_idx), len(non_train_idx))
    if n_train == 0:
        raise ValueError("Not enough member/non-member features to train representation attack.")
    n_eval = min(len(mem_eval_idx), len(non_eval_idx))
    if n_eval == 0:
        raise ValueError("Not enough held-out member/non-member features to evaluate representation attack.")

    rng = np.random.default_rng(seed + 31)
    mem_train_idx = rng.choice(mem_train_idx, size=n_train, replace=False)
    non_train_idx = rng.choice(non_train_idx, size=n_train, replace=False)
    mem_eval_idx = rng.choice(mem_eval_idx, size=n_eval, replace=False)
    non_eval_idx = rng.choice(non_eval_idx, size=n_eval, replace=False)

    train_feats = np.concatenate([member_feats[mem_train_idx], nonmember_feats[non_train_idx]], axis=0)
    train_labels = np.concatenate([member_labels[mem_train_idx], nonmember_labels[non_train_idx]], axis=0)
    train_membership = np.concatenate([np.ones(n_train), np.zeros(n_train)], axis=0)
    order = rng.permutation(len(train_membership))

    return (
        train_feats[order],
        train_labels[order],
        train_membership[order],
        member_feats[mem_eval_idx],
        member_labels[mem_eval_idx],
        nonmember_feats[non_eval_idx],
        nonmember_labels[non_eval_idx],
    )


def _attack_feature_matrix(feats, labels, num_classes, mean=None, std=None):
    feats = np.asarray(feats, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if mean is None:
        mean = feats.mean(axis=0, keepdims=True)
    if std is None:
        std = feats.std(axis=0, keepdims=True)
    feats = (feats - mean) / np.maximum(std, 1e-6)
    feats = np.clip(feats, -10.0, 10.0)

    one_hot = np.zeros((len(labels), num_classes), dtype=np.float32)
    valid = (labels >= 0) & (labels < num_classes)
    one_hot[np.arange(len(labels))[valid], labels[valid]] = 1.0
    return np.concatenate([feats, one_hot], axis=1).astype(np.float32), mean, std


def train_embedding_attack_mlp(member_feats, member_labels, nonmember_feats, nonmember_labels,
                               num_classes, device,
                               seed=42, epochs=30, batch_size=256, lr=1e-3,
                               train_fraction=0.5):
    """
    Embedding-based NN MIA over backbone features.
    Split known members and non-members into attack-train/eval subsets, train a
    binary attack classifier, then evaluate held-out members against held-out
    non-members. For villain artifacts, members should be the forget samples.
    """
    tr_feats, tr_labels, tr_membership, eval_member_feats, eval_member_labels, eval_non_feats, eval_non_labels = _balanced_attack_arrays(
        np.asarray(member_feats),
        np.asarray(member_labels),
        np.asarray(nonmember_feats),
        np.asarray(nonmember_labels),
        train_fraction=train_fraction,
        seed=seed,
    )
    x_train, mean, std = _attack_feature_matrix(tr_feats, tr_labels, num_classes)
    x_member, _, _ = _attack_feature_matrix(eval_member_feats, eval_member_labels, num_classes, mean, std)
    x_non, _, _ = _attack_feature_matrix(eval_non_feats, eval_non_labels, num_classes, mean, std)

    attack_model = RepresentationAttackMLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(attack_model.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.BCEWithLogitsLoss()

    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(tr_membership.astype(np.float32))),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    attack_model.train()
    for _ in range(epochs):
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(attack_model(xb), yb)
            loss.backward()
            optimizer.step()

    def score_array(x):
        scores = []
        attack_model.eval()
        with torch.no_grad():
            for start in range(0, len(x), batch_size):
                xb = torch.from_numpy(x[start:start + batch_size]).to(device)
                scores.append(torch.sigmoid(attack_model(xb)).cpu().numpy())
        return np.concatenate(scores, axis=0)

    member_scores = score_array(x_member)
    non_scores = score_array(x_non)
    return {
        "repr_attack_mlp": binary_score_metrics(member_scores, non_scores)
    }

                                                                
                                                    
                                                                

class MIABenchmarks:
    """
    Four threshold-based MIA attacks.

    Shadow train : retain set  (known members)
    Shadow test  : test set    (known non-members)
    Target       : forget set  (assess residual membership)

    ASR close to 0.5 → good forgetting (indistinguishable from non-members).
    """
    def __init__(self, shadow_train_perf, shadow_test_perf,
                 target_train_perf, num_classes, correctness_mode="sisa"):
        self.num_classes = num_classes
        self.correctness_mode = correctness_mode
        self.s_tr_out, self.s_tr_lbl = shadow_train_perf
        self.s_te_out, self.s_te_lbl = shadow_test_perf
        self.t_tr_out, self.t_tr_lbl = target_train_perf

        def _corr(out, lbl): return (np.argmax(out, 1) == lbl).astype(int)
        def _conf(out, lbl): return np.take_along_axis(out, lbl[:,None], 1).squeeze(1)
        def _entr(out):
            return np.sum(out * (-np.log(np.maximum(out, 1e-30))), axis=1)
        def _m_entr(out, lbl):
            lp  = -np.log(np.maximum(out, 1e-30))
            rv  = 1 - out
            lrp = -np.log(np.maximum(rv, 1e-30))
            mo  = out.copy();  mo[range(len(lbl)), lbl] = rv[range(len(lbl)), lbl]
            mlp = lrp.copy(); mlp[range(len(lbl)), lbl] = lp[range(len(lbl)), lbl]
            return np.sum(mo * mlp, axis=1)

        self.s_tr_corr = _corr(self.s_tr_out, self.s_tr_lbl)
        self.s_te_corr = _corr(self.s_te_out, self.s_te_lbl)
        self.t_tr_corr = _corr(self.t_tr_out, self.t_tr_lbl)

        self.s_tr_conf = _conf(self.s_tr_out, self.s_tr_lbl)
        self.s_te_conf = _conf(self.s_te_out, self.s_te_lbl)
        self.t_tr_conf = _conf(self.t_tr_out, self.t_tr_lbl)

        self.s_tr_entr = _entr(self.s_tr_out)
        self.s_te_entr = _entr(self.s_te_out)
        self.t_tr_entr = _entr(self.t_tr_out)

        self.s_tr_m_entr = _m_entr(self.s_tr_out, self.s_tr_lbl)
        self.s_te_m_entr = _m_entr(self.s_te_out, self.s_te_lbl)
        self.t_tr_m_entr = _m_entr(self.t_tr_out, self.t_tr_lbl)

    def _thre_setting(self, tr_vals, te_vals):
        candidates = np.concatenate([tr_vals, te_vals])
        best_thre, best_acc = -np.inf, 0.0
        for t in candidates:
            acc = 0.5 * (np.mean(tr_vals >= t) + np.mean(te_vals < t))
            if acc > best_acc:
                best_thre, best_acc = t, acc
        return best_thre

    def _mem_inf_corr(self):
        """Correctness-based attack."""
                                                         
        if self.correctness_mode == "threshold":
            return self._mem_inf_thre("Correctness", self.s_tr_corr, self.s_te_corr, self.t_tr_corr)
        asr = np.mean(self.t_tr_corr == 1)
        print(f"  {'Correctness':<20}: ASR = {asr:.4f}")
        return float(asr)

    def _mem_inf_thre(self, name, s_tr, s_te, t_tr):
        """Per-class threshold attack."""
        t_mem = 0
        for c in range(self.num_classes):
            mask_tr = self.s_tr_lbl == c
            mask_te = self.s_te_lbl == c
            mask_t  = self.t_tr_lbl == c
            if mask_tr.sum() == 0 or mask_te.sum() == 0: continue
            thre = self._thre_setting(s_tr[mask_tr], s_te[mask_te])
            t_mem += int(np.sum(t_tr[mask_t] >= thre))
        asr = t_mem / (len(self.t_tr_lbl) + 1e-9)
        print(f"  {name:<20}: ASR = {asr:.4f}  ({t_mem}/{len(self.t_tr_lbl)})")
        return float(asr)

    def _target_metrics(self, member_scores, nonmember_scores):
        return binary_score_metrics(member_scores, nonmember_scores)

    def run_attacks(self):
        print("\n[MIA] Running attacks on forget set:")
        results = {}
        results['correctness'] = {
            'asr': self._mem_inf_corr(),
            **self._target_metrics(self.t_tr_corr, self.s_te_corr),
        }
        results['confidence'] = {
            'asr': self._mem_inf_thre('Confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf),
            **self._target_metrics(self.t_tr_conf, self.s_te_conf),
        }
        results['entropy'] = {
            'asr': self._mem_inf_thre('Entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr),
            **self._target_metrics(-self.t_tr_entr, -self.s_te_entr),
        }
        results['modified_entropy'] = {
            'asr': self._mem_inf_thre(
                'Modified Entropy',
                -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr),
            **self._target_metrics(-self.t_tr_m_entr, -self.s_te_m_entr),
        }

        avg = float(np.mean([v['asr'] for v in results.values()]))
        results['average_asr'] = avg
        print(f"  {'Average':<20}: ASR = {avg:.4f}")
        print("\n[MIA] Target forget-vs-test ROC metrics (higher score = more member-like):")
        for name, vals in results.items():
            if not isinstance(vals, dict):
                continue
            print(
                f"  {name:<20}: AUC = {vals['auc']:.4f} | "
                f"TPR@1%FPR = {vals['tpr_at_0.01_fpr']:.4f} "
                f"(actual {vals['actual_fpr_at_0.01_fpr']:.4f}) | "
                f"TPR@0.1%FPR = {vals['tpr_at_0.001_fpr']:.4f} "
                f"(actual {vals['actual_fpr_at_0.001_fpr']:.4f})"
            )
        return results

                                                                
      
                                                                


def resolve_model(args):
    """
    Return the effective model name for the given dataset.
    Dataset-to-model mapping (used when --model is left at default 'resnet18'):
      CIFAR10   → resnet18
      CIFAR100  → resnet50
      MNIST     → resnet18
      Yale      → cnn
    If the user explicitly passes a different --model, that takes priority.
    """
    default_map = {
        'cifar10':  'resnet18',
        'cifar100': 'resnet50',
        'tinyimagenet': 'resnet50',
        'mnist':    'resnet18',
        'yale':     'mlp',
    }
    if args.model == 'resnet18':                                       
        return default_map.get(args.dataset, 'resnet18')
    return args.model                                                      


def append_summary_row(summary_csv, row):
    """
    Append a row while upgrading older summary_mia.csv headers in-place.
    Existing ASR-only rows remain readable, with blanks for new columns.
    """
    os.makedirs(os.path.dirname(summary_csv), exist_ok=True)
    existing_rows = []
    fieldnames = list(row.keys())
    if os.path.exists(summary_csv) and os.path.getsize(summary_csv) > 0:
        with open(summary_csv, newline='') as f:
            reader = csv.DictReader(f)
            fieldnames = list(reader.fieldnames or [])
            existing_rows = list(reader)
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)

    with open(summary_csv, 'w', newline='') as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
        writer.writeheader()
        for old_row in existing_rows:
            writer.writerow(old_row)
        writer.writerow(row)
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_UN)


def rounded_or_blank(value, ndigits=4):
    try:
        if value is None or not np.isfinite(value):
            return ""
        return round(float(value), ndigits)
    except Exception:
        return ""


def run_mia(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = resolve_model(args)
    print(f"\n{'='*60}")
    print(f"  MIA Evaluation")
    print(f"  Method  : {args.method}")
    print(f"  Dataset : {args.dataset}")
    print(f"  Model   : {args.model}")
    print(f"  Forget indices : {args.forget_indices_path}")
    print(f"{'='*60}")

                                                                
    train_set, test_set, num_classes, in_channels, img_size =\
        load_datasets(args.dataset, args.data_dir, seed=args.seed)

                                                                   
    if not args.forget_indices_path or not os.path.exists(args.forget_indices_path):
        raise ValueError(
            "--forget_indices_path is required and must point to the canonical .pt file "
            "(history/forget_indices_<dataset>_<model>_seed<seed>.pt)"
        )
    forget_idx = torch.load(args.forget_indices_path, map_location='cpu')
    if isinstance(forget_idx, torch.Tensor):
        forget_idx = forget_idx.tolist()
    print(f"[Load] Forget indices loaded from '{args.forget_indices_path}' "
          f"({len(forget_idx)} samples)")

    forget_idx_set = set(forget_idx)
    retain_idx     = torch.tensor([i for i in range(len(train_set)) if i not in forget_idx_set])

    retain_loader = DataLoader(Subset(train_set, retain_idx),
                               batch_size=128, shuffle=False)
    forget_loader = DataLoader(Subset(train_set, forget_idx),
                               batch_size=128, shuffle=False)
    test_loader   = DataLoader(test_set, batch_size=128, shuffle=False)

    print(f"  Retain : {len(retain_idx)} | Forget : {len(forget_idx)} "
          f"| Test : {len(test_set)}")

                                                                
    if args.method == 'sisa':
        container = os.path.join(
            args.sisa_dir,
            f"{args.dataset}_{args.model}_{args.seed}"
        )
        print(f"  SISA container : {container}")
        inference_model = load_sisa_ensemble(
            container, args.dataset, args.model,
            in_channels, num_classes, img_size,
            args.num_shards, args.num_slices, device
        )
    else:
        if not args.model_path:
            raise ValueError("--model_path is required for non-SISA methods")
        print(f"  Model path : {args.model_path}")
        inference_model = load_single_model(
            args.model_path, args.model,
            in_channels, num_classes, img_size, device,
            allow_missing_head=(args.method == 'expert')
        )

                                                                
    print("\n  Collecting probabilities...")
    retain_perf = collect_probs(inference_model, retain_loader, device)
    test_perf   = collect_probs(inference_model, test_loader,   device)
    forget_perf = collect_probs(inference_model, forget_loader, device)

                                                                
    correctness_mode = "sisa" if args.method == "sisa" else "threshold"
    mia = MIABenchmarks(retain_perf, test_perf, forget_perf, num_classes,
                        correctness_mode=correctness_mode)
    results = mia.run_attacks()

    repr_results = {}
    if args.representation_mia:
        print("\n[Representation MIA] Collecting pre-head features...")
        if isinstance(inference_model, list):
            print("[Representation MIA] Using the mean pre-head representation across SISA shards.")
            retain_feats = collect_ensemble_head_features(inference_model, retain_loader, device)
            test_feats = collect_ensemble_head_features(inference_model, test_loader, device)
            forget_feats = collect_ensemble_head_features(inference_model, forget_loader, device)
        else:
            retain_feats = collect_head_features(inference_model, retain_loader, device)
            test_feats = collect_head_features(inference_model, test_loader, device)
            forget_feats = collect_head_features(inference_model, forget_loader, device)
        repr_results = representation_mia_metrics(
            retain_feats[0], retain_feats[1],
            test_feats[0], test_feats[1],
            forget_feats[0], forget_feats[1],
        )
        if args.repr_attack_nonmember_source == "retain":
            nonmember_feats, nonmember_labels = retain_feats
        else:
            nonmember_feats, nonmember_labels = test_feats
        print(
            "[Representation MIA] Training embedding attack MLP with "
            f"forget-split members vs {args.repr_attack_nonmember_source} non-members."
        )
        repr_results.update(train_embedding_attack_mlp(
            forget_feats[0], forget_feats[1],
            nonmember_feats, nonmember_labels,
            num_classes=num_classes,
            device=device,
            seed=args.seed,
            epochs=args.repr_attack_epochs,
            batch_size=args.repr_attack_batch_size,
            lr=args.repr_attack_lr,
            train_fraction=args.repr_attack_train_fraction,
        ))
        print("[Representation MIA] Target forget-vs-test ROC metrics:")
        for name, vals in repr_results.items():
            print(
                f"  {name:<24}: AUC = {vals['auc']:.4f} | "
                f"TPR@1%FPR = {vals['tpr_at_0.01_fpr']:.4f} "
                f"(actual {vals['actual_fpr_at_0.01_fpr']:.4f}) | "
                f"TPR@0.1%FPR = {vals['tpr_at_0.001_fpr']:.4f} "
                f"(actual {vals['actual_fpr_at_0.001_fpr']:.4f})"
            )

    grad_results = {}
    if args.gradient_mia:
        if isinstance(inference_model, list):
            print("\n[Gradient MIA] Skipped for SISA ensembles.")
        elif getattr(inference_model, "_loaded_missing_head", False):
            print("\n[Gradient MIA] Skipped because checkpoint is headless; no trained logit head/loss gradient is available.")
        else:
            print("\n[Gradient MIA] Computing full-head CE-gradient scores...")
            test_grad_scores = collect_head_gradient_scores(inference_model, test_loader, device, num_classes)
            forget_grad_scores = collect_head_gradient_scores(inference_model, forget_loader, device, num_classes)
            grad_results = gradient_mia_metrics(test_grad_scores, forget_grad_scores)
            print("[Gradient MIA] Target forget-vs-test ROC metrics:")
            for name, vals in grad_results.items():
                print(
                    f"  {name:<24}: AUC = {vals['auc']:.4f} | "
                    f"TPR@1%FPR = {vals['tpr_at_0.01_fpr']:.4f} "
                    f"(actual {vals['actual_fpr_at_0.01_fpr']:.4f}) | "
                    f"TPR@0.1%FPR = {vals['tpr_at_0.001_fpr']:.4f} "
                    f"(actual {vals['actual_fpr_at_0.001_fpr']:.4f})"
                )

                                                                
    os.makedirs(args.history_dir, exist_ok=True)
    summary_csv = os.path.join(args.history_dir, "summary_mia.csv")

    model_path_str = args.model_path if args.model_path else\
        f"sisa:{args.dataset}_{args.model}_{args.seed}"

    summary_row = {
        'Method': args.method,
        'Dataset': args.dataset,
        'Model': args.model,
        'LowFPRProtocol': 'exact_roc_fpr_leq_tie_grouped_v2',
        'N_Forget': len(forget_idx),
        'Forget_Indices_Path': args.forget_indices_path,
        'Seed': args.seed,
        'ASR_Correctness': rounded_or_blank(results['correctness']['asr']),
        'ASR_Confidence': rounded_or_blank(results['confidence']['asr']),
        'ASR_Entropy': rounded_or_blank(results['entropy']['asr']),
        'ASR_ModEntropy': rounded_or_blank(results['modified_entropy']['asr']),
        'ASR_Average': rounded_or_blank(results['average_asr']),
        'AUC_Correctness': rounded_or_blank(results['correctness']['auc']),
        'AUC_Confidence': rounded_or_blank(results['confidence']['auc']),
        'AUC_Entropy': rounded_or_blank(results['entropy']['auc']),
        'AUC_ModEntropy': rounded_or_blank(results['modified_entropy']['auc']),
        'TPR1FPR_Correctness': rounded_or_blank(results['correctness']['tpr_at_0.01_fpr']),
        'TPR1FPR_Confidence': rounded_or_blank(results['confidence']['tpr_at_0.01_fpr']),
        'TPR1FPR_Entropy': rounded_or_blank(results['entropy']['tpr_at_0.01_fpr']),
        'TPR1FPR_ModEntropy': rounded_or_blank(results['modified_entropy']['tpr_at_0.01_fpr']),
        'TPR01FPR_Correctness': rounded_or_blank(results['correctness']['tpr_at_0.001_fpr']),
        'TPR01FPR_Confidence': rounded_or_blank(results['confidence']['tpr_at_0.001_fpr']),
        'TPR01FPR_Entropy': rounded_or_blank(results['entropy']['tpr_at_0.001_fpr']),
        'TPR01FPR_ModEntropy': rounded_or_blank(results['modified_entropy']['tpr_at_0.001_fpr']),
        'ActualFPR1_Correctness': rounded_or_blank(results['correctness']['actual_fpr_at_0.01_fpr']),
        'ActualFPR1_Confidence': rounded_or_blank(results['confidence']['actual_fpr_at_0.01_fpr']),
        'ActualFPR1_Entropy': rounded_or_blank(results['entropy']['actual_fpr_at_0.01_fpr']),
        'ActualFPR1_ModEntropy': rounded_or_blank(results['modified_entropy']['actual_fpr_at_0.01_fpr']),
        'ActualFPR01_Correctness': rounded_or_blank(results['correctness']['actual_fpr_at_0.001_fpr']),
        'ActualFPR01_Confidence': rounded_or_blank(results['confidence']['actual_fpr_at_0.001_fpr']),
        'ActualFPR01_Entropy': rounded_or_blank(results['entropy']['actual_fpr_at_0.001_fpr']),
        'ActualFPR01_ModEntropy': rounded_or_blank(results['modified_entropy']['actual_fpr_at_0.001_fpr']),
        'Model_Path': model_path_str,
        'ReprAttackProtocol': "forget_split" if args.representation_mia else "",
        'ReprAttackNonmemberSource': args.repr_attack_nonmember_source if args.representation_mia else "",
        'ReprAttackTrainFraction': args.repr_attack_train_fraction if args.representation_mia else "",
        'ReprAttackEpochs': args.repr_attack_epochs if args.representation_mia else "",
    }
    repr_column_names = {
        "repr_l2_centroid": "ReprL2Centroid",
        "repr_cosine_centroid": "ReprCosineCentroid",
        "repr_norm": "ReprNorm",
        "repr_attack_mlp": "ReprAttackMLP",
    }
    for key, prefix in repr_column_names.items():
        vals = repr_results.get(key, {})
        summary_row[f"AUC_{prefix}"] = rounded_or_blank(vals.get('auc'))
        summary_row[f"TPR1FPR_{prefix}"] = rounded_or_blank(vals.get('tpr_at_0.01_fpr'))
        summary_row[f"TPR01FPR_{prefix}"] = rounded_or_blank(vals.get('tpr_at_0.001_fpr'))
        summary_row[f"ActualFPR1_{prefix}"] = rounded_or_blank(vals.get('actual_fpr_at_0.01_fpr'))
        summary_row[f"ActualFPR01_{prefix}"] = rounded_or_blank(vals.get('actual_fpr_at_0.001_fpr'))

    grad_column_names = {
        "grad_loss": "GradLoss",
        "grad_head_norm": "GradHeadNorm",
        "grad_weight_norm": "GradWeightNorm",
        "grad_logit_norm": "GradLogitNorm",
    }
    for key, prefix in grad_column_names.items():
        vals = grad_results.get(key, {})
        summary_row[f"AUC_{prefix}"] = rounded_or_blank(vals.get('auc'))
        summary_row[f"TPR1FPR_{prefix}"] = rounded_or_blank(vals.get('tpr_at_0.01_fpr'))
        summary_row[f"TPR01FPR_{prefix}"] = rounded_or_blank(vals.get('tpr_at_0.001_fpr'))
        summary_row[f"ActualFPR1_{prefix}"] = rounded_or_blank(vals.get('actual_fpr_at_0.01_fpr'))
        summary_row[f"ActualFPR01_{prefix}"] = rounded_or_blank(vals.get('actual_fpr_at_0.001_fpr'))

    append_summary_row(summary_csv, summary_row)

    print(f"\n[Save] MIA result appended → {summary_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIA evaluation for ViLUN / DELETE / PS / SalUn / SISA / Original"
    )

                                                                
    parser.add_argument("--method", type=str, required=True,
                        help="Method label written to the summary. Only 'sisa' and 'expert' use special loading logic.")
    parser.add_argument("--model_path", type=str, default=None,
                        help="Path to .pth model file (not needed for --method sisa)")

                                                                
    parser.add_argument("--dataset",   type=str, required=True,
                        choices=["cifar10", "cifar100", "mnist", "yale", "tinyimagenet"])
    parser.add_argument("--model",     type=str, required=True,
                        choices=["cnn", "resnet18", "resnet50", "lenet", "vit", "mlp", "rnn"])
    parser.add_argument("--target_id", type=int, default=0,
                        help="Yale only: subject index to unlearn (0-14). Ignored for CIFAR/MNIST.")
    parser.add_argument("--forget_indices_path", type=str, default=None,
                        help="Path to canonical forget_indices_<dataset>_<model>_seed<seed>.pt. "
                             "Required — ensures MIA uses the exact same forget set as unlearning.")
    parser.add_argument("--seed",      type=int, default=42)
    parser.add_argument("--data_dir",  type=str, default="./data")
    parser.add_argument("--history_dir", type=str, default="./history")
    parser.add_argument("--representation_mia", action="store_true",
                        help="Also run representation-level MIA using pre-head backbone features.")
    parser.add_argument("--gradient_mia", action="store_true",
                        help="Also run full-head gradient-based MIA using CE-gradient summaries.")
    parser.add_argument("--repr_attack_epochs", type=int, default=30,
                        help="Epochs for the embedding-based representation attack MLP.")
    parser.add_argument("--repr_attack_batch_size", type=int, default=256,
                        help="Batch size for the embedding-based representation attack MLP.")
    parser.add_argument("--repr_attack_lr", type=float, default=1e-3,
                        help="Learning rate for the embedding-based representation attack MLP.")
    parser.add_argument("--repr_attack_train_fraction", type=float, default=0.5,
                        help="Fraction of member/non-member features used to train the representation attack MLP.")
    parser.add_argument("--repr_attack_nonmember_source", type=str, default="test",
                        choices=["test", "retain"],
                        help="Non-member pool for the forget-split representation attack MLP.")

                                                                
    parser.add_argument("--sisa_dir",   type=str, default="./sisa_models/containers",
                        help="Root dir of SISA containers")
    parser.add_argument("--num_shards", type=int, default=5)
    parser.add_argument("--num_slices", type=int, default=3)

    args = parser.parse_args()
    run_mia(args)
