"""
pretrain_models.py
─────────────────────────────────────────────────────────────────────────────
  1. Forget indices → history/forget_indices_{dataset}_{model}_seed{seed}.pt

  2. Original models → saved_models/original_{dataset}_{model}_seed{seed}.pth

  3. Expert (villain) models → saved_models/expert_{dataset}_{expert_model}_seed{seed}.pth

  python pretrain_models.py --data_dir ./data --save_dir ./saved_models --history_dir ./history --seed 42 --gpus 0 1 2
"""

import os
import sys
import time
import csv
import copy
import glob
import random
import argparse
import shutil
import subprocess

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset, Dataset
from PIL import Image


                                                                               
          
                                                                               
DS_MODEL = {
    'cifar10':  'resnet18',
    'cifar100': 'resnet50',
    'tinyimagenet': 'resnet50',
    'yale':     'mlp',
}
DATASETS = ['cifar10', 'cifar100', 'tinyimagenet']
ABLATION_DS = 'cifar10'
ALL_MODELS   = ['cnn', 'rnn', 'mlp', 'resnet18', 'resnet50']
FORGET_RATIO = 0.1        
REFERENCE_PRETRAIN_EPOCHS = 10
REFERENCE_PRETRAIN_LR = 1e-4
REFERENCE_VAL_SPLIT = 0.1
CIFAR10_PRETRAIN_EPOCHS = 20
CIFAR100_PRETRAIN_EPOCHS = 100
TINYIMAGENET_PRETRAIN_EPOCHS = 100
MNIST_PRETRAIN_EPOCHS = 10
YALE_PRETRAIN_EPOCHS = 15
CIFAR100_PRETRAIN_LR = 1e-3
TINYIMAGENET_PRETRAIN_LR = 1e-3
CIFAR100_WEIGHT_DECAY = 5e-4
TINYIMAGENET_WEIGHT_DECAY = 5e-4
YALE_SUBJECT_ID = 0                                      


def get_original_pretrain_epochs(dataset):
    if dataset == 'cifar10':
        return CIFAR10_PRETRAIN_EPOCHS
    if dataset == 'cifar100':
        return CIFAR100_PRETRAIN_EPOCHS
    if dataset == 'tinyimagenet':
        return TINYIMAGENET_PRETRAIN_EPOCHS
    if dataset == 'mnist':
        return MNIST_PRETRAIN_EPOCHS
    if dataset == 'yale':
        return YALE_PRETRAIN_EPOCHS
    return REFERENCE_PRETRAIN_EPOCHS


def get_original_pretrain_lr(dataset):
    if dataset == 'cifar100':
        return CIFAR100_PRETRAIN_LR
    if dataset == 'tinyimagenet':
        return TINYIMAGENET_PRETRAIN_LR
    return REFERENCE_PRETRAIN_LR


                                                                               
         
                                                                               
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


                                                                               
                 
                                                                               
def load_yale_custom(root_dir, img_size=64):
    for pattern in [os.path.join(root_dir, 'subject*'),
                    os.path.join(root_dir, 'yalefaces', 'subject*')]:
        file_list = glob.glob(pattern)
        if file_list:
            break
    if not file_list:
        X = torch.randn(165, 1, img_size, img_size)
        y = torch.repeat_interleave(torch.arange(15), 11)
        return TensorDataset(X, y), 15, 1

    tf = transforms.Compose([transforms.Grayscale(1),
                              transforms.Resize((img_size, img_size)),
                              transforms.ToTensor()])
    X_list, y_list = [], []
    for fp in sorted(file_list):
        if not os.path.isfile(fp): continue
        try:
            num = int(os.path.basename(fp).split('.')[0].replace('subject', '')) - 1
            X_list.append(tf(Image.open(fp)))
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
    raise FileNotFoundError(
        "TinyImageNet not found. Expected data/tiny-imagenet-200 with train/ and val/ directories."
    )


class TinyImageNetValDataset(Dataset):
    def __init__(self, root, class_to_idx, transform=None):
        self.transform = transform
        ann_path = os.path.join(root, 'val', 'val_annotations.txt')
        img_dir = os.path.join(root, 'val', 'images')
        self.samples = []
        with open(ann_path, encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split('\t')
                if len(parts) < 2:
                    continue
                filename, class_id = parts[0], parts[1]
                if class_id in class_to_idx:
                    self.samples.append((os.path.join(img_dir, filename), class_to_idx[class_id]))
        self.targets = [label for _, label in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, target


def get_train_transform(dataset, size, norm, augment=False):
    transforms_list = [transforms.Resize((size, size))]
    if augment and dataset in ('cifar100', 'tinyimagenet'):
        transforms_list.extend([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transforms_list.extend([transforms.ToTensor(), norm])
    return transforms.Compose(transforms_list)


def load_dataset(dataset, data_dir, train_augment=False):
    """Returns (train_set, test_set, num_classes, in_channels, img_size)."""
    if dataset == 'yale':
        full, nc, ic = load_yale_custom(os.path.join(data_dir, 'yale'))
        return full, full, nc, ic, 64
    if dataset == 'tinyimagenet':
        root = find_tinyimagenet_root(data_dir)
        norm = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        train_tf = get_train_transform(dataset, 64, norm, augment=train_augment)
        test_tf = get_train_transform(dataset, 64, norm, augment=False)
        train_root = os.path.join(root, 'train')
        train_set = torchvision.datasets.ImageFolder(train_root, transform=train_tf)
        test_set = TinyImageNetValDataset(root, train_set.class_to_idx, transform=test_tf)
        return train_set, test_set, 200, 3, 64
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
    train_tf = get_train_transform(dataset, sz, norm, augment=train_augment)
    test_tf = get_train_transform(dataset, sz, norm, augment=False)
    tr = cls(root=data_dir, train=True,  download=True, transform=train_tf)
    te = cls(root=data_dir, train=False, download=True, transform=test_tf)
    return tr, te, nc, ic, sz


                                                                               
                   
                                                                               
class ResNet18Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = torch.flatten(x, 1)
        logits = self.model.fc(feature)
        if return_features:
            return logits, feature
        return logits

class ResNet50Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=None)
        self.model.conv1 = nn.Conv2d(in_channels, 64, 3, 1, 1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(2048, num_classes)
    def forward(self, x, return_features=False):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)
        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        x = self.model.avgpool(x)
        feature = torch.flatten(x, 1)
        logits = self.model.fc(feature)
        if return_features:
            return logits, feature
        return logits

class CNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1,1))
        )
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x, return_features=False):
        feature = torch.flatten(self.features(x), 1)
        logits = self.classifier(feature)
        if return_features:
            return logits, feature
        return logits

class MLPNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Linear(in_channels*img_size*img_size, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 128), nn.GELU()
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x, return_features=False):
        feature = self.mixer(torch.flatten(x, 1))
        logits = self.classifier(feature)
        if return_features:
            return logits, feature
        return logits

class RNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.lstm = nn.LSTM(in_channels*img_size, 128, 2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x, return_features=False):
        B, C, H, W = x.size()
        if H != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size))
            B, C, H, W = x.size()
        x = x.permute(0,2,3,1).reshape(B, H, W*C)
        _, (h_n, _) = self.lstm(x)
        feature = torch.cat((h_n[-2], h_n[-1]), dim=1)
        logits = self.classifier(feature)
        if return_features:
            return logits, feature
        return logits

def get_model(model_name, in_channels, num_classes, img_size):
    if model_name == 'cnn':      return CNNNetwork(in_channels, num_classes)
    if model_name == 'resnet18': return ResNet18Network(in_channels, num_classes)
    if model_name == 'resnet50': return ResNet50Network(in_channels, num_classes)
    if model_name == 'mlp':      return MLPNetwork(in_channels, num_classes, img_size)
    if model_name == 'rnn':      return RNNNetwork(in_channels, num_classes, img_size)
    raise ValueError(f"Unknown model: {model_name}")


def get_feature_dim(model_name):
    dims = {
        'cnn': 64,
        'rnn': 256,
        'mlp': 128,
        'resnet18': 512,
        'resnet50': 2048,
    }
    return dims[model_name]


def is_classifier_head_key(key):
    """Return True for logit-head parameters that are not needed by ViLUn."""
    return (
        key in {'classifier.weight', 'classifier.bias'}
        or key in {'model.fc.weight', 'model.fc.bias'}
        or key in {'model.heads.head.weight', 'model.heads.head.bias'}
    )


def strip_classifier_head(state_dict):
    """Remove classifier/logit head weights from a villain checkpoint."""
    stripped = {
        key: value for key, value in state_dict.items()
        if not is_classifier_head_key(key)
    }
    removed = sorted(set(state_dict.keys()) - set(stripped.keys()))
    return stripped, removed


                                                                               
                    
                                                                               
def train_model(model, loader, device, epochs=200, lr=1e-3, patience=50,
                desc='', verbose=True, val_loader=None, test_loader=None,
                epoch_logs=None, optimizer_name='adam', weight_decay=0.0,
                momentum=0.9, cosine_scheduler=False):
    model.train()
    if optimizer_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine_scheduler else None
    criterion = nn.CrossEntropyLoss()
    best_score = None
    best_weights = None
    counter = 0

    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            correct += (out.argmax(1) == target).sum().item()
            total   += target.size(0)

        acc = 100.0 * correct / total
        val_acc = None
        val_loss = None
        monitor_score = -epoch_loss
        val_msg = ""
        if val_loader is not None:
            val_acc, val_loss = evaluate_model(model, val_loader, device)
            monitor_score = val_acc
            val_msg = f" | Val Acc: {val_acc:.2f}% | Val Loss: {val_loss:.4f}"
        if test_loader is not None and epoch_logs is not None:
            test_acc, test_loss = evaluate_model(model, test_loader, device)
            epoch_logs.append({
                'epoch': epoch + 1,
                'test_acc': round(test_acc, 4),
                'test_loss': round(test_loss, 6),
            })
            model.train()
            test_msg = f" | Test Acc: {test_acc:.2f}% | Test Loss: {test_loss:.4f}"
        else:
            test_msg = ""

        if verbose:
            print(f"  {desc} Epoch [{epoch+1:>4}/{epochs}] "
                  f"| Loss: {epoch_loss:.4f} | Acc: {acc:.2f}%{val_msg}{test_msg}", flush=True)

        if best_score is None or monitor_score > best_score:
            best_score = monitor_score
            best_weights = copy.deepcopy(model.state_dict())
            counter = 0
        elif patience is not None:
            counter += 1
            if counter >= patience:
                print(f"  {desc} [Early Stop] at epoch {epoch+1}", flush=True)
                break
        if scheduler is not None:
            scheduler.step()

    if best_weights:
        model.load_state_dict(best_weights)
    return model


def train_divergent_expert(expert_model, original_model, loader, device,
                           expert_dim, original_dim, epochs=50, lr=1e-3,
                           diverge_gamma=0.5, desc='', verbose=True):
    expert_model.train()
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    projector = nn.Identity() if expert_dim == original_dim else nn.Linear(expert_dim, original_dim).to(device)
    params = list(expert_model.parameters())
    if not isinstance(projector, nn.Identity):
        params += list(projector.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    best_weights = None
    best_acc = -1.0
    for epoch in range(epochs):
        epoch_loss, correct, total = 0.0, 0, 0
        cosine_sum, batches = 0.0, 0
        expert_model.train()
        if not isinstance(projector, nn.Identity):
            projector.train()

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            expert_logits, expert_features = expert_model(data, return_features=True)
            with torch.no_grad():
                _, original_features = original_model(data, return_features=True)

            projected_expert = projector(expert_features)
            ce_loss = criterion(expert_logits, target)
            cosine = F.cosine_similarity(projected_expert, original_features.detach(), dim=1).mean()
            loss = ce_loss + diverge_gamma * cosine

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (expert_logits.argmax(1) == target).sum().item()
            total += target.size(0)
            cosine_sum += cosine.item()
            batches += 1

        acc = 100.0 * correct / total
        avg_cosine = cosine_sum / max(1, batches)
        if verbose:
            print(f"  {desc} Epoch [{epoch+1:>4}/{epochs}] "
                  f"| Loss: {epoch_loss:.4f} | Forget Acc: {acc:.2f}% "
                  f"| Expert-Original Cos: {avg_cosine:.4f}", flush=True)
        if acc >= best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(expert_model.state_dict())

    if best_weights:
        expert_model.load_state_dict(best_weights)
    return expert_model


def evaluate_model(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_sum += F.cross_entropy(out, target, reduction='sum').item()
            correct  += (out.argmax(1) == target).sum().item()
            total    += target.size(0)
    return 100.0 * correct / total, loss_sum / total


def save_perf_csv(path, entries):
    """entries: list of dicts with keys: split, acc, loss"""
    exists = os.path.exists(path)
    with open(path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['split', 'acc', 'loss'])
        if not exists:
            writer.writeheader()
        writer.writerows(entries)


def save_epoch_test_csv(path, rows):
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'test_acc', 'test_loss'])
        writer.writeheader()
        writer.writerows(rows)


def split_train_val_dataset(dataset, val_split, seed):
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    val_idx = indices[:n_val].tolist()
    train_idx = indices[n_val:].tolist()
    return Subset(dataset, train_idx), Subset(dataset, val_idx)


def split_train_val_indices(dataset, val_split, seed):
    n_total = len(dataset)
    n_val = max(1, int(n_total * val_split))
    rng = np.random.default_rng(seed)
    indices = np.arange(n_total)
    rng.shuffle(indices)
    return indices[n_val:].tolist(), indices[:n_val].tolist()


                                                                               
                           
                                                                               
def generate_and_save_forget_indices(dataset, train_set, seed, history_dir):
    """
    Generate forget_indices once, save to canonical path, then create
    method-specific symlinks (copies) so every unlearning script finds
    the right filename.
    Returns forget_indices tensor.
    """
    set_seed(seed)

    if dataset == 'yale':
        if hasattr(train_set, 'tensors'):
            all_labels = train_set.tensors[1]
        else:
            all_labels = torch.tensor([y for _, y in train_set])
        subject_idx = (all_labels == YALE_SUBJECT_ID).nonzero(as_tuple=True)[0].tolist()
        n_forget = min(11, len(subject_idx))
        forget_indices = torch.tensor(random.sample(subject_idx, n_forget))
        print(f"  [Yale] Subject {YALE_SUBJECT_ID}: {n_forget} images as forget set", flush=True)
    else:
        n_total  = len(train_set)
        n_forget = max(1, int(FORGET_RATIO * n_total))
        forget_indices = torch.tensor(random.sample(range(n_total), n_forget))
        print(f"  [{dataset}] Sampled {n_forget}/{n_total} ({FORGET_RATIO*100:.0f}%) as forget set",
              flush=True)

                                                                
    om = DS_MODEL[dataset]
    fname = f"forget_indices_{dataset}_{om}_seed{seed}.pt"
    torch.save(forget_indices, os.path.join(history_dir, fname))
    print(f"  [Save] Forget indices saved: {fname}", flush=True)
    return forget_indices


def generate_expert_forget_indices(ablation_ds, train_set, seed, history_dir):
    """For Exp2 ablation: cifar10 with all 5 orig models share the same forget indices."""
    set_seed(seed)
    n_total  = len(train_set)
    n_forget = max(1, int(FORGET_RATIO * n_total))
    forget_indices = torch.tensor(random.sample(range(n_total), n_forget))
    print(f"  [{ablation_ds} ablation] Forget set: {n_forget}/{n_total}", flush=True)

                                                                                           
    for om in ALL_MODELS:
        fname = f"forget_indices_{ablation_ds}_{om}_seed{seed}.pt"
        torch.save(forget_indices, os.path.join(history_dir, fname))
    return forget_indices


                                                                               
                                                           
                                                                               
import multiprocessing as mp

def _train_worker(job):
    """Worker function for multiprocessing pool."""
    (gpu_id, job_type, dataset, model_name, in_channels, num_classes,
     img_size, data_dir, save_dir, history_dir, seed,
     forget_indices, lr, epochs, patience, desc, force_set,
     strip_expert_head) = job

    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_seed(seed)

                                                                             
    if job_type == 'original':
        save_path = os.path.join(save_dir,
                                 f"original_{dataset}_{model_name}_seed{seed}.pth")
        perf_path = os.path.join(history_dir,
                                 f"pretrain_original_{dataset}_{model_name}_seed{seed}.csv")
        epoch_test_path = os.path.join(history_dir,
                                       f"pretrain_original_{dataset}_{model_name}_seed{seed}_epoch_test.csv")
    elif job_type == 'expert':
        save_path = os.path.join(save_dir,
                                 f"expert_{dataset}_{model_name}_seed{seed}.pth")
        perf_path = os.path.join(history_dir,
                                 f"pretrain_expert_{dataset}_{model_name}_seed{seed}.csv")
        epoch_test_path = None

    fname = os.path.basename(save_path)
    force_expert = job_type == 'expert' and '__experts__' in force_set
    if os.path.exists(save_path) and '__all__' not in force_set and fname not in force_set and not force_expert:
        print(f"[GPU{gpu_id}] SKIP {desc} — already exists: {save_path}", flush=True)
        return save_path
    if fname in force_set or force_expert:
        print(f"[GPU{gpu_id}] FORCE retrain {desc}", flush=True)

    train_set, test_set, _, _, _ = load_dataset(dataset, data_dir)
    model = get_model(model_name, in_channels, num_classes, img_size).to(device)
    original_for_expert = None
    effective_epochs = epochs
    effective_patience = patience
    effective_lr = lr
    optimizer_name = 'adam'
    weight_decay = 0.0
    cosine_scheduler = False
    val_loader = None

                                                                              
    if job_type == 'original':
        if dataset == 'yale':
            loader = DataLoader(train_set, batch_size=64, shuffle=True)
            effective_epochs = YALE_PRETRAIN_EPOCHS
            effective_patience = None
        else:
            if dataset in ('cifar100', 'tinyimagenet'):
                aug_train_set, _, _, _, _ = load_dataset(dataset, data_dir, train_augment=True)
                train_idx, val_idx = split_train_val_indices(train_set, REFERENCE_VAL_SPLIT, seed)
                loader = DataLoader(Subset(aug_train_set, train_idx), batch_size=64, shuffle=True)
                val_loader = DataLoader(Subset(train_set, val_idx), batch_size=128, shuffle=False)
                optimizer_name = 'sgd'
                weight_decay = TINYIMAGENET_WEIGHT_DECAY if dataset == 'tinyimagenet' else CIFAR100_WEIGHT_DECAY
                cosine_scheduler = True
            else:
                train_subset, val_subset = split_train_val_dataset(train_set, REFERENCE_VAL_SPLIT, seed)
                loader = DataLoader(train_subset, batch_size=64, shuffle=True)
                val_loader = DataLoader(val_subset, batch_size=128, shuffle=False)
            effective_epochs = get_original_pretrain_epochs(dataset)
            effective_lr = get_original_pretrain_lr(dataset)
            effective_patience = None

    elif job_type == 'expert':
                         
        loader = DataLoader(Subset(train_set, forget_indices),
                            batch_size=32, shuffle=True)
        original_model_name = DS_MODEL[dataset]
        original_path = os.path.join(save_dir, f"original_{dataset}_{original_model_name}_seed{seed}.pth")
        waited = 0
        while not os.path.exists(original_path) and waited < 3600:
            if waited == 0:
                print(f"[GPU{gpu_id}] Waiting for original model before divergent expert: {original_path}", flush=True)
            time.sleep(10)
            waited += 10
        if os.path.exists(original_path):
            original_for_expert = get_model(original_model_name, in_channels, num_classes, img_size).to(device)
            original_for_expert.load_state_dict(torch.load(original_path, map_location=device))
        else:
            print(f"[GPU{gpu_id}] WARNING: original model not found; falling back to CE-only expert.", flush=True)
        if dataset == 'yale':
            effective_epochs = YALE_PRETRAIN_EPOCHS
            effective_patience = None

    print(f"\n{'='*60}", flush=True)
    print(f"[GPU{gpu_id}] {desc}", flush=True)
    print(f"  Model: {model_name} | Dataset: {dataset} | "
          f"LR: {effective_lr} | Epochs: {effective_epochs} | Patience: {effective_patience}", flush=True)
    if job_type == 'original':
        print(f"  Optimizer: {optimizer_name} | Weight Decay: {weight_decay} | Cosine: {cosine_scheduler}", flush=True)
    print(f"{'='*60}", flush=True)

                                                                              
    t0 = time.time()
    epoch_logs = []
    train_test_loader = DataLoader(test_set, batch_size=128, shuffle=False) if job_type == 'original' else None
    if job_type == 'expert' and original_for_expert is not None:
        model = train_divergent_expert(
            model, original_for_expert, loader, device,
            expert_dim=get_feature_dim(model_name),
            original_dim=get_feature_dim(DS_MODEL[dataset]),
            epochs=effective_epochs, lr=effective_lr,
            diverge_gamma=0.5, desc=desc, verbose=True
        )
    else:
        model = train_model(model, loader, device,
                            epochs=effective_epochs, lr=effective_lr, patience=effective_patience,
                            desc=desc, verbose=True, val_loader=val_loader,
                            test_loader=train_test_loader, epoch_logs=epoch_logs,
                            optimizer_name=optimizer_name, weight_decay=weight_decay,
                            cosine_scheduler=cosine_scheduler)
    if torch.cuda.is_available(): torch.cuda.synchronize()
    elapsed = time.time() - t0

                                                                              
    full_loader = DataLoader(train_set, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_set,  batch_size=128, shuffle=False)

    train_acc, train_loss = evaluate_model(model, full_loader, device)
    test_acc,  test_loss  = evaluate_model(model, test_loader,  device)

    perf_entries = [
        {'split': 'train', 'acc': round(train_acc, 4), 'loss': round(train_loss, 6)},
        {'split': 'test',  'acc': round(test_acc,  4), 'loss': round(test_loss,  6)},
        {'split': 'time_s','acc': round(elapsed, 2),   'loss': 0},
    ]

                                      
    if job_type == 'expert' and forget_indices is not None:
        forget_eval_loader = DataLoader(
            Subset(train_set, forget_indices), batch_size=128, shuffle=False)
        forget_acc, forget_loss = evaluate_model(model, forget_eval_loader, device)
        perf_entries.append(
            {'split': 'forget', 'acc': round(forget_acc, 4), 'loss': round(forget_loss, 6)}
        )
        print(f"  Forget Acc: {forget_acc:.2f}% (should be high — expert on forget set)",
              flush=True)
    else:
        forget_acc = float('nan')

    save_perf_csv(perf_path, perf_entries)
    if job_type == 'original':
        save_epoch_test_csv(epoch_test_path, epoch_logs)
        print(f"  [Save] Epoch test log: {epoch_test_path}", flush=True)

                                                                              
    os.makedirs(save_dir, exist_ok=True)
    state_to_save = model.state_dict()
    if job_type == 'expert' and strip_expert_head:
        state_to_save, removed_head = strip_classifier_head(state_to_save)
        print(
            f"  [Privacy] Stripped expert logit head before saving: {removed_head}",
            flush=True
        )
    torch.save(state_to_save, save_path)

    print(f"\n[GPU{gpu_id}] {desc} DONE", flush=True)
    print(f"  Train Acc: {train_acc:.2f}% | Test Acc: {test_acc:.2f}% "
          f"| Time: {elapsed:.1f}s", flush=True)
    print(f"  Saved → {save_path}", flush=True)
    return save_path


def _run_gpu_jobs(gpu_jobs):
    for job in gpu_jobs:
        _train_worker(job)


def run_expert_logit_mia(args, expert_jobs):
    """Run logit-based MIA on saved expert checkpoints after head stripping."""
    if not expert_jobs:
        return

    eval_script = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'evaluate_mia.py')
    print("\n[Step 3] Evaluating stripped-head expert logit MIA...", flush=True)
    for job in expert_jobs:
        (_, job_type, dataset, model_name, _in_channels, _num_classes,
         _img_size, data_dir, save_dir, history_dir, seed,
         _forget_indices, _lr, _epochs, _patience, desc, _force_set,
         _strip_expert_head) = job
        if job_type != 'expert':
            continue

        model_path = os.path.join(save_dir, f"expert_{dataset}_{model_name}_seed{seed}.pth")
        forget_model = DS_MODEL[dataset]
        forget_path = os.path.join(history_dir, f"forget_indices_{dataset}_{forget_model}_seed{seed}.pt")
        if not os.path.exists(model_path):
            print(f"  [SKIP] Missing expert checkpoint for MIA: {model_path}", flush=True)
            continue
        if not os.path.exists(forget_path):
            print(f"  [SKIP] Missing forget indices for MIA: {forget_path}", flush=True)
            continue

        cmd = [
            sys.executable, eval_script,
            '--method', 'expert',
            '--dataset', dataset,
            '--model', model_name,
            '--data_dir', data_dir,
            '--history_dir', history_dir,
            '--model_path', model_path,
            '--forget_indices_path', forget_path,
            '--seed', str(seed),
        ]
        print(f"  [MIA] {desc} | checkpoint has no trained logit head", flush=True)
        subprocess.run(cmd, check=True)


                                                                               
      
                                                                               
def main():
    parser = argparse.ArgumentParser(
        description="Pre-train all original and expert models before experiments"
    )
    parser.add_argument('--data_dir',    type=str, default='./data')
    parser.add_argument('--save_dir',    type=str, default='./saved_models')
    parser.add_argument('--history_dir', type=str, default='./history')
    parser.add_argument('--seed',        type=int, default=42)
    parser.add_argument('--gpus',        type=int, nargs='+', default=[0, 1, 2])
    parser.add_argument('--epochs',      type=int, default=200)
    parser.add_argument('--patience',    type=int, default=50)
    parser.add_argument('--lr',          type=float, default=1e-3)
    parser.add_argument('--expert_epochs', type=int, default=50,
                        help='Epochs for expert model training on forget set')
    parser.add_argument('--force_experts', action='store_true',
                        help='Retrain all expert/villain models even if cached checkpoints already exist')
    parser.add_argument('--keep_expert_head', action='store_true',
                        help='Keep classifier/logit head in saved expert checkpoints')
    parser.add_argument('--evaluate_expert_mia', action='store_true',
                        help='After expert pretraining, run logit-based MIA on saved expert checkpoints')
    parser.add_argument('--force', type=str, nargs='*', default=None,
                        metavar='FILENAME',
                        help=('model file'))
    args = parser.parse_args()
    args.force_all = (args.force is not None and len(args.force) == 0)
    args.force_list = set(args.force) if args.force else set()
    if args.force_experts:
        args.force_list.add('__experts__')

    os.makedirs(args.save_dir,    exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)

    set_seed(args.seed)
    print("\n" + "="*70, flush=True)
    print(" PRETRAIN MODELS — Pre-generating all models before experiments", flush=True)
    print("="*70, flush=True)

                                                                           
                                                                 
                                                                           
    print("\n[Step 1] Generating forget indices...", flush=True)
    forget_indices_map = {}                             

    for dataset in DATASETS:
        om = DS_MODEL[dataset]
        train_set, _, _, _, _ = load_dataset(dataset, args.data_dir)
        fi = generate_and_save_forget_indices(dataset, train_set, args.seed, args.history_dir)
        forget_indices_map[(dataset, om)] = fi
                                              
    cifar10_train, _, _, _, _ = load_dataset(ABLATION_DS, args.data_dir)
    ablation_fi = generate_expert_forget_indices(
        ABLATION_DS, cifar10_train, args.seed, args.history_dir)
    for om in ALL_MODELS:
        forget_indices_map[(ABLATION_DS, om)] = ablation_fi

    print("\n[Step 1] Done. All forget indices saved.", flush=True)

                                                                           
                          
                                                                           
    jobs = []                                                                
                                                                                           
    n_gpus = len(args.gpus)
    job_counter = 0

    def add_job(job_type, dataset, model_name, forget_idx=None):
        nonlocal job_counter
        _, _, nc, ic, sz = load_dataset.__wrapped__(dataset, args.data_dir)\
            if hasattr(load_dataset, '__wrapped__') else (None, None, None, None, None)
                                                 
        ds_info = {
            'cifar10':  (10, 3, 32), 'cifar100': (100, 3, 32),
            'tinyimagenet': (200, 3, 64),
            'mnist':    (10, 1, 32), 'yale':     (15,  1, 64),
        }
        nc, ic, sz = ds_info[dataset]
        gpu = args.gpus[job_counter % n_gpus]
        ep  = args.expert_epochs if job_type == 'expert' else args.epochs
        desc = (f"[{job_type.upper()}] {dataset}/{model_name}"
                if job_type == 'original'
                else f"[EXPERT] {dataset}/{model_name}")
        force_set = (set() if not args.force_all
                     else {'__all__'}                          
                     ) if not args.force_list else args.force_list
                                                               
        if args.force_all:
            force_set = {'__all__'}
        else:
            force_set = args.force_list
        job = (gpu, job_type, dataset, model_name, ic, nc, sz,
               args.data_dir, args.save_dir, args.history_dir, args.seed,
               forget_idx, args.lr, ep, args.patience, desc, force_set,
               not args.keep_expert_head)
        jobs.append(job)
        job_counter += 1

                                                                            
                                          
    seen_orig = set()
    for dataset in DATASETS:
        om = DS_MODEL[dataset]
        key = (dataset, om)
        if key not in seen_orig:
            seen_orig.add(key)
            add_job('original', dataset, om, forget_idx=forget_indices_map[(dataset, om)])

                                       
    for om in ALL_MODELS:
        key = (ABLATION_DS, om)
        if key not in seen_orig:
            seen_orig.add(key)
            add_job('original', ABLATION_DS, om, forget_idx=forget_indices_map[(ABLATION_DS, om)])

                                                                            
    seen_expert = set()
                                           
    for dataset in DATASETS:
        om = DS_MODEL[dataset]
        key = (dataset, 'rnn')
        if key not in seen_expert:
            seen_expert.add(key)
            add_job('expert', dataset, 'rnn',
                    forget_idx=forget_indices_map[(dataset, om)])

                                               
    for vm in ALL_MODELS:
        key = (ABLATION_DS, vm)
        if key not in seen_expert:
            seen_expert.add(key)
            add_job('expert', ABLATION_DS, vm,
                    forget_idx=forget_indices_map[(ABLATION_DS, DS_MODEL[ABLATION_DS])])

    print(f"\n[Step 2] Jobs to run: {len(jobs)} "
          f"(orig: {len(seen_orig)}, expert: {len(seen_expert)})", flush=True)
    print(f"  GPUs: {args.gpus}", flush=True)

                                                                           
                                    
                                                                           
    print("\n[Step 3] Training models in parallel...", flush=True)
    total_start = time.time()

                                  
                             
    ctx = mp.get_context('spawn')

    def run_phase(phase_name, phase_jobs):
        print(f"\n[Phase] {phase_name}: {len(phase_jobs)} jobs", flush=True)
        gpu_queues = {g: [] for g in args.gpus}
        for job in phase_jobs:
            gpu_queues[job[0]].append(job)

        procs = []
        for gpu_id, gpu_jobs in gpu_queues.items():
            if not gpu_jobs:
                continue
            p = ctx.Process(target=_run_gpu_jobs, args=(gpu_jobs,))
            p.start()
            procs.append((gpu_id, p))

        failed = False
        for gpu_id, p in procs:
            p.join()
            if p.exitcode != 0:
                print(f"[ERROR] {phase_name} worker on GPU{gpu_id} failed with exit code {p.exitcode}", flush=True)
                failed = True
        if failed:
            sys.exit(1)

    original_jobs = [job for job in jobs if job[1] == 'original']
    expert_jobs = [job for job in jobs if job[1] == 'expert']
    run_phase("Original models", original_jobs)
    run_phase("Divergent expert models", expert_jobs)
    if args.evaluate_expert_mia:
        run_expert_logit_mia(args, expert_jobs)

    elapsed_total = time.time() - total_start
    print(f"\n{'='*70}", flush=True)
    print(f" ALL MODELS PRETRAINED — Total time: {elapsed_total:.1f}s ({elapsed_total/60:.1f} min)",
          flush=True)
    print(f"{'='*70}\n", flush=True)

                                                                           
                      
                                                                           
    print("[Step 4] Summary:", flush=True)
    pth_files = sorted(glob.glob(os.path.join(args.save_dir, '*.pth')))
    print(f"  Models saved ({len(pth_files)}):")
    for p in pth_files:
        print(f"    {os.path.basename(p)}")

    fi_files = sorted(glob.glob(os.path.join(args.history_dir, 'forget_indices_*.pt')))
    print(f"\n  Forget indices saved ({len(fi_files)}):")
    for p in fi_files:
        print(f"    {os.path.basename(p)}")

    print("\n[Done] All pre-training complete. "
          "Run run_experiments.sh to start unlearning experiments.", flush=True)


if __name__ == '__main__':
    main()
