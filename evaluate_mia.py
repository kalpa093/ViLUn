"""
evaluate_mia.py
---------------
Membership Inference Attack (MIA) evaluation for all ViLUN-related models.

Supports the following model types (via --method):
  vilun        : saved_models/unlearned_<dataset>_*_<orig>_<expert>.pth
  vilun_heldout: saved_models/unlearned_heldout_<dataset>_<orig>_expert<expert>_target<id>_seed<seed>.pth
  delete       : saved_models/delete_<dataset>_<model>_target<id>_seed<seed>.pth
  ps           : ps_models/ps_<dataset>_<model>_<seed>.pth
  salun        : salun_models/salun_<dataset>_<model>_<seed>.pth
  sisa         : sisa_models/containers/<dataset>_<model>_<seed>/  (ensemble of shards)
  original     : saved_models/original_<dataset>_<model>_<seed>.pth  (baseline)
  expert       : saved_models/expert_<dataset>_<model>_<seed>.pth     (villain model)

MIA shadow setup:
  Shadow train : retain set of train split  (known members)
  Shadow test  : test set                   (known non-members)
  Target       : forget set of train split  (test residual membership)

All results are appended to:  history/summary_mia.csv
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

    def run_attacks(self):
        print("\n[MIA] Running attacks on forget set:")
        results = {}
        results['correctness']     = {'asr': self._mem_inf_corr()}
        results['confidence']      = {'asr': self._mem_inf_thre(
            'Confidence', self.s_tr_conf, self.s_te_conf, self.t_tr_conf)}
        results['entropy']         = {'asr': self._mem_inf_thre(
            'Entropy', -self.s_tr_entr, -self.s_te_entr, -self.t_tr_entr)}
        results['modified_entropy']= {'asr': self._mem_inf_thre(
            'Modified Entropy',
            -self.s_tr_m_entr, -self.s_te_m_entr, -self.t_tr_m_entr)}

        avg = float(np.mean([v['asr'] for v in results.values()]))
        results['average_asr'] = avg
        print(f"  {'Average':<20}: ASR = {avg:.4f}")
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
    print(f"[Load] Forget indices loaded from '{args.forget_indices_path}' "
          f"({len(forget_idx)} samples)")

    forget_idx_set = set(forget_idx.tolist())
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

                                                                
    os.makedirs(args.history_dir, exist_ok=True)
    summary_csv = os.path.join(args.history_dir, "summary_mia.csv")
    exists = os.path.exists(summary_csv)

    model_path_str = args.model_path if args.model_path else\
        f"sisa:{args.dataset}_{args.model}_{args.seed}"

    with open(summary_csv, 'a', newline='') as f:
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_EX)
        writer = csv.writer(f)
        exists = os.path.getsize(summary_csv) > 0
        if not exists:
            writer.writerow([
                'Method', 'Dataset', 'Model', 'N_Forget', 'Forget_Indices_Path', 'Seed',
                'ASR_Correctness', 'ASR_Confidence',
                'ASR_Entropy', 'ASR_ModEntropy', 'ASR_Average',
                'Model_Path'
            ])
        writer.writerow([
            args.method, args.dataset, args.model, len(forget_idx), args.forget_indices_path, args.seed,
            round(results['correctness']['asr'],      4),
            round(results['confidence']['asr'],       4),
            round(results['entropy']['asr'],          4),
            round(results['modified_entropy']['asr'], 4),
            round(results['average_asr'],             4),
            model_path_str
        ])
        if fcntl is not None:
            fcntl.flock(f, fcntl.LOCK_UN)

    print(f"\n[Save] MIA result appended → {summary_csv}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="MIA evaluation for ViLUN / DELETE / PS / SalUn / SISA / Original"
    )

                                                                
    parser.add_argument("--method", type=str, required=True,
                        choices=["vilun", "vilun_heldout", "delete",
                                 "ps", "salun", "sisa", "original", "expert"],
                        help="Which unlearning method's model to evaluate")
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

                                                                
    parser.add_argument("--sisa_dir",   type=str, default="./sisa_models/containers",
                        help="Root dir of SISA containers")
    parser.add_argument("--num_shards", type=int, default=5)
    parser.add_argument("--num_slices", type=int, default=3)

    args = parser.parse_args()
    run_mia(args)
