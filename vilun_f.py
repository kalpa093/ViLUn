import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset, Dataset
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import glob
import random
import argparse
import csv
import copy
import json
import optuna
import time

                                            
                                
                                            
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience=50, mode='min'):
        self.patience = patience
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score, model):
        if self.best_score is None:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
        elif (self.mode == 'min' and score >= self.best_score) or\
             (self.mode == 'max' and score <= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0


def get_tuned_params_path(args):
    os.makedirs(args.history_dir, exist_ok=True)
    return os.path.join(
        args.history_dir,
        f"tuned_params_vilun_{args.dataset}_{args.model}_{args.expert_model}_seed{args.seed}.json"
    )


def load_cached_tuned_params(args):
    path = get_tuned_params_path(args)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_tuned_params(args, params, best_score=None):
    payload = {
        "method": "ViLUN",
        "dataset": args.dataset,
        "model": args.model,
        "expert_model": args.expert_model,
        "seed": args.seed,
        "beta": float(params["beta"]),
        "unlearn_lr": float(params["unlearn_lr"]),
        "n_trials": int(args.n_trials),
        "tune_patience": int(args.tune_patience),
        "tune_max_epochs": int(args.tune_max_epochs),
    }
    if best_score is not None:
        payload["best_score"] = float(best_score)
    path = get_tuned_params_path(args)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Save] Tuned hyperparameters -> {path}")

                                            
                
                                            
def load_yale_custom(root_dir, img_size=64):
    search_path = os.path.join(root_dir, "subject*")
    file_list = glob.glob(search_path)
    if len(file_list) == 0:
        search_path = os.path.join(root_dir, "yalefaces", "subject*")
        file_list = glob.glob(search_path)

    if len(file_list) == 0:
        X = torch.randn(100, 1, img_size, img_size)
        y = torch.randint(0, 15, (100,))
        return TensorDataset(X, y), 15, 1

    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])

    X_list, y_list = [], []
    for f_path in file_list:
        if not os.path.isfile(f_path): continue
        filename = os.path.basename(f_path)
        try:
            subject_part = filename.split(".")[0] 
            subject_num = int(subject_part.replace("subject", ""))
            label_idx = subject_num - 1 
            img = Image.open(f_path)
            img_tensor = transform(img)
            X_list.append(img_tensor)
            y_list.append(torch.tensor(label_idx, dtype=torch.long))
        except: continue

    full_dataset = TensorDataset(torch.stack(X_list), torch.stack(y_list))
    unique_labels = torch.unique(torch.stack(y_list))
    return full_dataset, len(unique_labels), 1


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

def load_dataset_factory(args):
    data_dir = args.data_dir
    if args.dataset == 'yale':
        yale_dir = os.path.join(data_dir, 'yale')
        full_dataset, num_classes, in_channels = load_yale_custom(yale_dir)
        return full_dataset, full_dataset, num_classes, in_channels
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set, 10, 3
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set, 100, 3
    elif args.dataset == 'tinyimagenet':
        root = find_tinyimagenet_root(data_dir)
        transform = transforms.Compose([
            transforms.Resize((64, 64)), transforms.ToTensor(),
            transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        ])
        train_set = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform=transform)
        test_set = TinyImageNetValDataset(root, train_set.class_to_idx, transform=transform)
        return train_set, test_set, 200, 3
    elif args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set, 10, 1
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

                                            
           
                                            
class ResNet18Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet18Network, self).__init__()
        self.model = torchvision.models.resnet18(weights=None)
        if in_channels != 3: self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, num_classes)
    def forward(self, x, return_features=False):
        x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x); x = self.model.maxpool(x)
        x = self.model.layer1(x); x = self.model.layer2(x); x = self.model.layer3(x); x = self.model.layer4(x)
        x = self.model.avgpool(x); feature = torch.flatten(x, 1); logits = self.model.fc(feature)
        if return_features: return logits, feature
        return logits

class ResNet50Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ResNet50Network, self).__init__()
        self.model = torchvision.models.resnet50(weights=None)
        self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.model.maxpool = nn.Identity()
        self.model.fc = nn.Linear(2048, num_classes)
    def forward(self, x, return_features=False):
        x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x); x = self.model.maxpool(x)
        x = self.model.layer1(x); x = self.model.layer2(x); x = self.model.layer3(x); x = self.model.layer4(x)
        x = self.model.avgpool(x); feature = torch.flatten(x, 1); logits = self.model.fc(feature)
        if return_features: return logits, feature
        return logits

class CNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(CNNNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
    def forward(self, x, return_features=False):
        x = self.features(x); feature = torch.flatten(x, 1); logits = self.classifier(feature)
        if return_features: return logits, feature
        return logits

class LeNetNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(LeNetNetwork, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120); self.fc2 = nn.Linear(120, 84); self.classifier = nn.Linear(84, num_classes)
    def forward(self, x, return_features=False):
        x = self.features(x); x = self.pool(x); x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); feature = F.relu(self.fc2(x)); logits = self.classifier(feature)
        if return_features: return logits, feature
        return logits

class ViTNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ViTNetwork, self).__init__()
        self.model = torchvision.models.vit_b_16(weights=torchvision.models.ViT_B_16_Weights.DEFAULT)
        if in_channels != 3: self.model.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16)
        self.model.heads.head = nn.Linear(768, num_classes)
    def forward(self, x, return_features=False):
        if x.size(-1) < 224: x = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
        x = self.model._process_input(x); n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1); x = self.model.encoder(x)
        feature = x[:, 0]; logits = self.model.heads(feature)
        if return_features: return logits, feature
        return logits

class MLPNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super(MLPNetwork, self).__init__()
        input_dim = in_channels * img_size * img_size
        self.mixer = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(), nn.Linear(512, 512), nn.GELU(), nn.Linear(512, 128), nn.GELU()
        )
        self.classifier = nn.Linear(128, num_classes)
    def forward(self, x, return_features=False):
        x = torch.flatten(x, 1); feature = self.mixer(x); logits = self.classifier(feature)
        if return_features: return logits, feature
        return logits

class RNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super(RNNNetwork, self).__init__()
        self.img_size = img_size
        input_size = in_channels * img_size 
        hidden_size = 128
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)
    def forward(self, x, return_features=False):
        B, C, H, W = x.size()
        if H != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size))
            B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H, W * C)
        lstm_out, (h_n, c_n) = self.lstm(x)
        feature = torch.cat((h_n[-2,:,:], h_n[-1,:,:]), dim=1); logits = self.classifier(feature)
        if return_features: return logits, feature
        return logits

class FeatureProjector(nn.Module):
    def __init__(self, input_dim=64, output_dim=512):
        super(FeatureProjector, self).__init__()
        self.proj = nn.Sequential(nn.Linear(input_dim, output_dim), nn.ReLU(), nn.Linear(output_dim, output_dim))
    def forward(self, x):
        return self.proj(x)


def feature_repel_loss(current_features, expert_features, margin=0.0):
    cosine = F.cosine_similarity(current_features, expert_features, dim=1)
    return F.relu(cosine - margin).pow(2).mean()


def feature_preserve_loss(current_features, teacher_features):
    return (1.0 - F.cosine_similarity(current_features, teacher_features.detach(), dim=1)).mean()


def heldout_consistency_loss(student_logits, teacher_logits):
    """Preserve original-model behavior on auxiliary held-out samples."""
    teacher_probs = F.softmax(teacher_logits.detach(), dim=1)
    student_log_probs = F.log_softmax(student_logits, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean")


def fit_feature_projector(projector, expert_model, teacher_model, loader, device, epochs=3, lr=1e-3):
    projector.train()
    expert_model.eval()
    teacher_model.eval()
    optimizer = optim.Adam(projector.parameters(), lr=lr)

    for _ in range(max(0, epochs)):
        for data, _ in loader:
            data = data.to(device)
            with torch.no_grad():
                _, expert_features = expert_model(data, return_features=True)
                _, teacher_features = teacher_model(data, return_features=True)
            projected_features = projector(expert_features)
            loss = feature_preserve_loss(projected_features, teacher_features)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    projector.eval()
    for param in projector.parameters():
        param.requires_grad = False

def get_model(model_name, in_channels, num_classes, img_size):
    if model_name == 'cnn': return CNNNetwork(in_channels, num_classes), 64
    elif model_name == 'resnet18': return ResNet18Network(in_channels, num_classes), 512
    elif model_name == 'resnet50': return ResNet50Network(in_channels, num_classes), 2048
    elif model_name == 'lenet': return LeNetNetwork(in_channels, num_classes), 84
    elif model_name == 'vit': return ViTNetwork(in_channels, num_classes), 768
    elif model_name == 'mlp': return MLPNetwork(in_channels, num_classes, img_size), 128
    elif model_name == 'rnn': return RNNNetwork(in_channels, num_classes, img_size), 256
    else: raise ValueError(f"Unknown model: {model_name}")


def is_classifier_head_key(key):
    return (
        key in {'classifier.weight', 'classifier.bias'}
        or key in {'model.fc.weight', 'model.fc.bias'}
        or key in {'model.heads.head.weight', 'model.heads.head.bias'}
    )


def export_headless_state(model):
    return {key: value.detach().cpu() for key, value in model.state_dict().items()
            if not is_classifier_head_key(key)}


def load_expert_state(model, path, device):
    """Load a villain checkpoint, allowing the public checkpoint to omit logits."""
    state = torch.load(path, map_location=device)
    result = model.load_state_dict(state, strict=False)
    missing = list(getattr(result, 'missing_keys', []))
    unexpected = list(getattr(result, 'unexpected_keys', []))
    non_head_missing = [key for key in missing if not is_classifier_head_key(key)]
    if non_head_missing or unexpected:
        raise RuntimeError(
            f"Could not load expert checkpoint {path}. "
            f"Missing non-head keys: {non_head_missing}; unexpected keys: {unexpected}"
        )
    head_missing = [key for key in missing if is_classifier_head_key(key)]
    if head_missing:
        print(f"  >> Expert checkpoint omits logit head; using random unused head: {head_missing}")

                                            
                      
                                            
def evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose=True,
                        test_exclude_idx=None):
    """Sample-target evaluation using forget sample indices."""
    forget_idx_set   = set(forget_indices.tolist() if hasattr(forget_indices, 'tolist') else list(forget_indices))
    all_train_idx    = list(range(len(train_set)))
    retain_train_idx = [i for i in all_train_idx if i not in forget_idx_set]
    forget_train_idx = list(forget_idx_set)

    def make_loader(dataset, indices):
        if len(indices) == 0: return None
        return DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False)

    if test_exclude_idx is not None and len(test_exclude_idx) > 0:
        exclude_set = set(test_exclude_idx.tolist() if hasattr(test_exclude_idx, 'tolist') else list(test_exclude_idx))
        test_keep_idx = [i for i in range(len(test_set)) if i not in exclude_set]
        test_retain_loader = make_loader(test_set, test_keep_idx)
    else:
        test_retain_loader = DataLoader(test_set, batch_size=128, shuffle=False)

    loaders = {
        "Train_Forget": make_loader(train_set, forget_train_idx),
        "Train_Retain": make_loader(train_set, retain_train_idx),
        "Test_Retain ": test_retain_loader,
    }
    
    model.eval()
    if verbose:
        print("  [Detailed Evaluation]")
        print(f"  {'Metric':<15} | {'Acc (%)':<10} | {'Loss':<10}")
        print("  " + "-"*40)
    
    results = {}
    with torch.no_grad():
        for name, loader in loaders.items():
            if loader is None: continue
            correct, total, loss_sum = 0, 0, 0
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                loss_sum += F.cross_entropy(out, target, reduction='sum').item()
                _, pred = torch.max(out, 1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            
            acc = 100 * correct / total
            avg_loss = loss_sum / total
            results[name] = {'acc': acc, 'loss': avg_loss}
            if verbose: print(f"  {name:<15} | {acc:6.2f}     | {avg_loss:.4f}")
    if verbose: print("  " + "-"*40)
    return results

                                            
                                     
                                            
def train_standard(model, loader, device, epochs=50, lr=1e-3, patience=50):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
                      
    early_stopper = EarlyStopping(patience=patience, mode='min') 
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
        early_stopper(epoch_loss, model)
        if early_stopper.early_stop:
            print(f"    [Early Stop] Training stopped at epoch {epoch+1}.")
            break
            
    if early_stopper.best_weights is not None:
        model.load_state_dict(early_stopper.best_weights)
    return model


def train_divergent_expert(expert_model, original_model, loader, device,
                           expert_feat_dim, orig_feat_dim, epochs=50, lr=1e-3,
                           diverge_gamma=0.5, patience=None):
    expert_model.train()
    original_model.eval()
    for param in original_model.parameters():
        param.requires_grad = False

    projector = nn.Identity() if expert_feat_dim == orig_feat_dim else FeatureProjector(expert_feat_dim, orig_feat_dim).to(device)
    params = list(expert_model.parameters())
    if not isinstance(projector, nn.Identity):
        params += list(projector.parameters())
    optimizer = optim.Adam(params, lr=lr)
    criterion = nn.CrossEntropyLoss()

    early_stopper = EarlyStopping(patience=patience, mode='max') if patience is not None else None
    best_weights = None
    best_acc = -1.0
    for epoch in range(epochs):
        expert_model.train()
        if not isinstance(projector, nn.Identity):
            projector.train()
        correct, total, epoch_loss, cosine_sum, batches = 0, 0, 0.0, 0.0, 0

        for data, target in loader:
            data, target = data.to(device), target.to(device)
            expert_logits, expert_features = expert_model(data, return_features=True)
            with torch.no_grad():
                _, original_features = original_model(data, return_features=True)
            projected_features = projector(expert_features)

            loss_ce = criterion(expert_logits, target)
            loss_diverge = F.cosine_similarity(projected_features, original_features.detach(), dim=1).mean()
            loss = loss_ce + diverge_gamma * loss_diverge

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            correct += (expert_logits.argmax(1) == target).sum().item()
            total += target.size(0)
            cosine_sum += loss_diverge.item()
            batches += 1

        acc = 100.0 * correct / total
        avg_cosine = cosine_sum / max(1, batches)
        print(
            f"    [Divergent Expert] Epoch [{epoch+1:>4}/{epochs}]"
            f" | Loss: {epoch_loss:.4f}"
            f" | Forget Acc: {acc:6.2f}%"
            f" | Expert-Original Cos: {avg_cosine:.4f}"
        )
        if acc >= best_acc:
            best_acc = acc
            best_weights = copy.deepcopy(expert_model.state_dict())
        if early_stopper is not None:
            early_stopper(acc, expert_model)
            if early_stopper.early_stop:
                break

    if early_stopper is not None and early_stopper.best_weights is not None:
        expert_model.load_state_dict(early_stopper.best_weights)
    elif best_weights is not None:
        expert_model.load_state_dict(best_weights)
    return expert_model

def build_heldout_loader(test_set, heldout_ratio, batch_size=64, seed=42):
    """Build the auxiliary held-out loader used for feature-space alignment."""
    n_total = len(test_set)
    n_heldout = max(1, int(n_total * heldout_ratio))
    rng = random.Random(seed)
    heldout_idx = rng.sample(range(n_total), n_heldout)
    heldout_loader = DataLoader(Subset(test_set, heldout_idx), batch_size=batch_size, shuffle=True)
    print(f"  [Held-out] {n_heldout}/{n_total} test samples selected (ratio={heldout_ratio})")
    return heldout_loader, heldout_idx


def get_dataset_labels(dataset):
    if hasattr(dataset, 'targets'):
        targets = dataset.targets
        return targets.clone() if isinstance(targets, torch.Tensor) else torch.tensor(targets)
    if hasattr(dataset, 'tensors'):
        return dataset.tensors[1]
    return torch.tensor([y for _, y in dataset])


def build_anchor_loader(train_set, forget_idx, anchor_ratio, batch_size=64, seed=42, class_balanced=True):
    """Build a small non-forget anchor set for KD stability."""
    if anchor_ratio <= 0:
        print("  [Anchor] disabled (ratio=0)")
        return None, []

    forget_set = set(forget_idx.tolist() if hasattr(forget_idx, 'tolist') else list(forget_idx))
    retain_idx = [i for i in range(len(train_set)) if i not in forget_set]
    n_anchor = max(1, int(len(retain_idx) * anchor_ratio))
    rng = random.Random(seed)

    if class_balanced:
        labels = get_dataset_labels(train_set)
        by_class = {}
        for idx in retain_idx:
            by_class.setdefault(int(labels[idx]), []).append(idx)
        classes = sorted(by_class)
        per_class = max(1, n_anchor // max(1, len(classes)))
        anchor_idx = []
        for cls in classes:
            candidates = by_class[cls]
            rng.shuffle(candidates)
            anchor_idx.extend(candidates[:min(per_class, len(candidates))])
        if len(anchor_idx) < n_anchor:
            used = set(anchor_idx)
            remaining = [i for i in retain_idx if i not in used]
            rng.shuffle(remaining)
            anchor_idx.extend(remaining[:n_anchor - len(anchor_idx)])
        anchor_idx = anchor_idx[:n_anchor]
    else:
        anchor_idx = rng.sample(retain_idx, min(n_anchor, len(retain_idx)))

    loader = DataLoader(Subset(train_set, anchor_idx), batch_size=batch_size, shuffle=True)
    mode = "class-balanced" if class_balanced else "random"
    print(f"  [Anchor] {len(anchor_idx)}/{len(retain_idx)} retain samples selected ({mode}, ratio={anchor_ratio})")
    return loader, anchor_idx


def build_retain_loader(train_set, forget_idx, retain_ratio=1.0, batch_size=64, seed=42, class_balanced=True):
    forget_idx_set = set(forget_idx.tolist())
    retain_idx = [i for i in range(len(train_set)) if i not in forget_idx_set]
    if retain_ratio <= 0:
        raise ValueError("retain_ratio must be > 0")

    n_total = len(retain_idx)
    n_keep = max(1, int(round(n_total * min(retain_ratio, 1.0))))
    rng = random.Random(seed)

    if n_keep < n_total and class_balanced:
        if hasattr(train_set, 'targets'):
            labels = train_set.targets
        elif hasattr(train_set, 'tensors'):
            labels = train_set.tensors[1].tolist()
        else:
            labels = [y for _, y in train_set]
        by_class = {}
        for idx in retain_idx:
            by_class.setdefault(int(labels[idx]), []).append(idx)
        classes = sorted(by_class)
        per_class = max(1, n_keep // max(1, len(classes)))
        selected = []
        for cls in classes:
            candidates = by_class[cls]
            rng.shuffle(candidates)
            selected.extend(candidates[:min(per_class, len(candidates))])
        if len(selected) < n_keep:
            used = set(selected)
            remaining = [i for i in retain_idx if i not in used]
            rng.shuffle(remaining)
            selected.extend(remaining[:n_keep - len(selected)])
        retain_idx = selected[:n_keep]
    elif n_keep < n_total:
        retain_idx = rng.sample(retain_idx, n_keep)

    loader = DataLoader(Subset(train_set, retain_idx), batch_size=batch_size, shuffle=True)
    mode = "class-balanced" if class_balanced else "random"
    print(f"  [Retain] {len(retain_idx)}/{n_total} train samples selected for retain CE ({mode}, ratio={retain_ratio})")
    return loader, retain_idx


def load_retrain_target(history_dir, dataset, model, seed):
    """Load retrain reference metrics used to select the closest unlearning epoch."""
    path = os.path.join(history_dir, "summary_retrain_ga.csv")
    if not os.path.exists(path):
        return None
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if (
                row.get("unlearning") == "retrain"
                and row.get("dataset") == dataset
                and row.get("model") == model
                and str(row.get("seed")) == str(seed)
            ):
                try:
                    return {
                        "train_retain_acc": float(row["best_train_retain_acc"]),
                        "test_retain_acc": float(row["best_test_retain_acc"]),
                        "train_forget_acc": float(row["best_train_forget_acc"]),
                    }
                except Exception:
                    return None
    return None


def epoch_selection_score(metrics, retrain_target):
    """Lower is better. Use Total-MAE to retrain when available, otherwise max test retain."""
    if retrain_target is None:
        return -metrics["test_retain_acc"]
    return (
        abs(metrics["train_retain_acc"] - retrain_target["train_retain_acc"])
        + abs(metrics["test_retain_acc"] - retrain_target["test_retain_acc"])
        + abs(metrics["train_forget_acc"] - retrain_target["train_forget_acc"])
    ) / 3.0

                                            
                             
                                            
def objective(trial, args, base_orig_model, expert_model, train_set, test_set, forget_idx,
              forget_loader, train_full_loader, retain_loader, heldout_loader, heldout_idx,
              orig_feat_dim, expert_feat_dim, device):
    beta = trial.suggest_float("beta", 1.0, 50.0)
    unlearn_lr = trial.suggest_float("unlearn_lr", 1e-6, 1e-3, log=True)
    
    model = copy.deepcopy(base_orig_model)
    projector = FeatureProjector(input_dim=expert_feat_dim, output_dim=orig_feat_dim).to(device)
    
    expert_model.eval()
    frozen_teacher = copy.deepcopy(model).to(device)
    frozen_teacher.eval()
    for param in frozen_teacher.parameters():
        param.requires_grad = False
    fit_feature_projector(
        projector, expert_model, frozen_teacher, retain_loader, device,
        epochs=args.projector_epochs, lr=args.projector_lr
    )
    optimizer = optim.Adam(model.parameters(), lr=unlearn_lr)
    retain_iter = iter(retain_loader)
    
                                                 
    tune_epochs = min(args.unlearn_epochs, args.tune_max_epochs)
    early_stopper = EarlyStopping(patience=args.tune_patience, mode='max')
    
    for epoch in range(tune_epochs):
        model.train()
        projector.train()
        
        for data_f, _ in forget_loader:
            try:
                data_r, target_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                data_r, target_r = next(retain_iter)

            data_f = data_f.to(device)
            data_r = data_r.to(device)
            target_r = target_r.to(device)

            _, feat_orig_f = model(data_f, return_features=True)
            student_retain_logits = model(data_r)
            with torch.no_grad():
                _, feat_expert_f = expert_model(data_f, return_features=True)
                teacher_retain_logits = frozen_teacher(data_r)
            feat_expert_aligned = projector(feat_expert_f)
            
            loss_unlearn = feature_repel_loss(feat_orig_f, feat_expert_aligned, args.feature_margin)
            loss_retain = heldout_consistency_loss(student_retain_logits, teacher_retain_logits)
            loss = args.alpha * loss_retain + beta * loss_unlearn
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
        eval_res = evaluate_4_quadrant(model, train_set, test_set, forget_idx, device, verbose=False)
        score = eval_res['Test_Retain ']['acc'] - eval_res.get('Train_Forget', {}).get('acc', 0.0)
        
        early_stopper(score, model)
        if early_stopper.early_stop:
            break
            
        trial.report(score, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return early_stopper.best_score if early_stopper.best_score is not None else score

                                            
                 
                                            

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

def run_pipeline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = resolve_model(args)
    print(f"=== Negative Feature Distillation Unlearning ===  [Model: {args.model}]")
    
    train_set, test_set, num_classes, in_channels = load_dataset_factory(args)
    img_size = 64 if args.dataset in ('yale', 'tinyimagenet') else 32

                                                                  
                                                                     
    _fi_path = os.path.join(args.history_dir,
                            f"forget_indices_{args.dataset}_{args.model}_seed{args.seed}.pt")
    if os.path.exists(_fi_path):
        forget_idx = torch.load(_fi_path, map_location='cpu')
        print(f"[Load] Forget indices ← {_fi_path} ({len(forget_idx)} samples)", flush=True)
    else:
                         
        random.seed(args.seed)                                  
        if args.dataset == 'yale':
            if hasattr(train_set, 'tensors'):
                _all_labels = train_set.tensors[1]
            else:
                _all_labels = torch.tensor([y for _, y in train_set])
            _subj = (_all_labels == args.target_id).nonzero(as_tuple=True)[0].tolist()
            if not _subj:
                raise ValueError(f"No images for Yale subject {args.target_id}")
            forget_idx = torch.tensor(random.sample(_subj, min(11, len(_subj))))
            print(f"[Yale] Subject {args.target_id}: {len(forget_idx)} images as forget set")
        else:
            random.seed(args.seed)                    
            _n = len(train_set)
            _nf = max(1, int(0.1 * _n))
            forget_idx = torch.tensor(random.sample(range(_n), _nf))
            print(f"[{args.dataset}] Sampled {_nf}/{_n} (10%) as forget set")

    forget_idx_set = set(forget_idx.tolist())
    retain_idx     = torch.tensor([i for i in range(len(train_set)) if i not in forget_idx_set])

    forget_loader     = DataLoader(Subset(train_set, forget_idx), batch_size=64, shuffle=True)
    train_full_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    heldout_loader, heldout_idx = build_heldout_loader(test_set, args.heldout_ratio, seed=args.seed)
    retain_loader, retain_idx = build_retain_loader(
        train_set, forget_idx,
        retain_ratio=args.retain_ratio,
        batch_size=64, seed=args.seed,
        class_balanced=not args.no_class_balanced_retain
    )

    print(f"\n[Step 1] Initializing Models...")
    original_model, orig_feat_dim = get_model(args.model, in_channels, num_classes, img_size)
    original_model = original_model.to(device)
    os.makedirs(args.save_path, exist_ok=True)

    time_orig_train = 0.0
    cached_orig = os.path.join(args.save_path, f"original_{args.dataset}_{args.model}_seed{args.seed}.pth")
    if args.original is not None and os.path.exists(args.original):
        print(f"  >> Loading pre-trained Original Model from: {args.original}")
        original_model.load_state_dict(torch.load(args.original, map_location=device))
    elif os.path.exists(cached_orig):
        print(f"  >> Loading cached Original Model from: {cached_orig}")
        original_model.load_state_dict(torch.load(cached_orig, map_location=device))
    else:
        print(f"  >> Training Original Model ({args.model.upper()})...")
        t0 = time.time()
        original_model = train_standard(original_model, train_full_loader, device, epochs=args.train_epochs, lr=args.orig_lr, patience=args.patience)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_orig_train = time.time() - t0
        torch.save(original_model.state_dict(), cached_orig)
        print(f"  [Time] Original Model Training: {time_orig_train:.2f}s → saved to {cached_orig}")
    
    print("\n[Baseline] Evaluating Original Model Performance...")
    baseline_res = evaluate_4_quadrant(original_model, train_set, test_set, forget_idx, device, verbose=True)

    os.makedirs(args.history_dir, exist_ok=True)
    orig_csv_filename = os.path.join(args.history_dir, f"vilun_original_{args.dataset}_{args.model}_{args.expert_model}_seed{args.seed}.csv")
    
    with open(orig_csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([
            'Train_Forget_Acc', 'Train_Forget_Loss', 
            'Train_Retain_Acc', 'Train_Retain_Loss', 
            'Test_Retain_Acc', 'Test_Retain_Loss'
        ])
        
        writer.writerow([
            baseline_res.get('Train_Forget', {}).get('acc', 'N/A'),
            baseline_res.get('Train_Forget', {}).get('loss', 'N/A'),
            baseline_res.get('Train_Retain', {}).get('acc', 'N/A'),
            baseline_res.get('Train_Retain', {}).get('loss', 'N/A'),
            baseline_res.get('Test_Retain ', {}).get('acc', 'N/A'),
            baseline_res.get('Test_Retain ', {}).get('loss', 'N/A')
        ])
    print(f"  [Save] Original baseline performance saved to '{orig_csv_filename}'")
                                                

    expert_model, expert_feat_dim = get_model(args.expert_model, in_channels, num_classes, img_size)
    expert_model = expert_model.to(device)
    time_expert_train = 0.0
                                                                    
    cached_expert = os.path.join(
        args.save_path,
        f"expert_{args.dataset}_{args.expert_model}_seed{args.seed}.pth"
    )
    if args.expert_path is not None and os.path.exists(args.expert_path):
        print(f"  >> Loading pre-trained Expert Model from: {args.expert_path}")
        load_expert_state(expert_model, args.expert_path, device)
    elif os.path.exists(cached_expert):
        print(f"  >> Loading cached Expert Model from: {cached_expert}")
        load_expert_state(expert_model, cached_expert, device)
    else:
        print(f"  >> Training divergent Expert Model ({args.expert_model.upper()}) on forget set...")
        t0 = time.time()
        expert_model = train_divergent_expert(
            expert_model, original_model, forget_loader, device,
            expert_feat_dim=expert_feat_dim,
            orig_feat_dim=orig_feat_dim,
            epochs=args.expert_epochs,
            lr=args.expert_lr,
            diverge_gamma=args.expert_diverge_gamma,
            patience=None
        )
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_expert_train = time.time() - t0
        torch.save(export_headless_state(expert_model), cached_expert)
        print(f"  [Time] Expert Model Training: {time_expert_train:.2f}s → saved to {cached_expert}")

                                                
                                  
                                                
    print(f"\n[Step 3] Final Unlearning Process (Beta: {args.beta:.2f}, LR: {args.unlearn_lr:.6f})...")
    print("  >> Model owner uses: retain KL + forget feature repulsion")
    print(f"  >> Retain set available to owner: {len(retain_idx)} train samples")
    start_time_unlearn = time.time()
    time_eval = 0.0
    projector = FeatureProjector(input_dim=expert_feat_dim, output_dim=orig_feat_dim).to(device)
    expert_model.eval()
    frozen_teacher = copy.deepcopy(original_model).to(device)
    frozen_teacher.eval()
    for param in frozen_teacher.parameters():
        param.requires_grad = False
    fit_feature_projector(
        projector, expert_model, frozen_teacher, retain_loader, device,
        epochs=args.projector_epochs, lr=args.projector_lr
    )
    optimizer = optim.Adam(original_model.parameters(), lr=args.unlearn_lr)
    retain_iter = iter(retain_loader)
    retrain_target = load_retrain_target(args.history_dir, args.dataset, args.model, args.seed)
    best_state = None
    best_idx = 0
    best_score = float("inf")
    
    history = {
        'epoch': [], 'alpha': [], 'beta': [],
        'train_forget_acc': [], 'train_forget_loss': [],
        'train_retain_acc': [], 'train_retain_loss': [],
        'test_retain_acc': [], 'test_retain_loss': []
    }

    for epoch in range(args.unlearn_epochs):
        original_model.train()
        projector.train()
        for data_f, _ in forget_loader:
            try:
                data_r, target_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                data_r, target_r = next(retain_iter)

            data_f = data_f.to(device)
            data_r = data_r.to(device)
            target_r = target_r.to(device)
            _, feat_orig_f = original_model(data_f, return_features=True)
            student_retain_logits = original_model(data_r)
            with torch.no_grad():
                _, feat_expert_f = expert_model(data_f, return_features=True)
                teacher_retain_logits = frozen_teacher(data_r)
            feat_expert_aligned = projector(feat_expert_f)
            
            loss_unlearn = feature_repel_loss(feat_orig_f, feat_expert_aligned, args.feature_margin)
            loss_retain = heldout_consistency_loss(student_retain_logits, teacher_retain_logits)
            loss = args.alpha * loss_retain + args.beta * loss_unlearn
            
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            
        if torch.cuda.is_available(): torch.cuda.synchronize()
        eval_start = time.time()
        eval_res = evaluate_4_quadrant(original_model, train_set, test_set, forget_idx, device)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_eval += time.time() - eval_start
        _tf = eval_res.get("Train_Forget", {}).get("acc", float("nan"))
        _tr_train = eval_res.get("Train_Retain", {}).get("acc", float("nan"))
        _tr_test = eval_res["Test_Retain "]["acc"]
        print(
            f"  Epoch [{epoch+1:>4}/{args.unlearn_epochs}]"
            f" | Forget Acc: {_tf:6.2f}%"
            f" | Retain Acc: {_tr_train:6.2f}%"
            f" | Test Acc: {_tr_test:6.2f}%"
        )
        
        history['epoch'].append(epoch + 1)
        history['alpha'].append(args.alpha)
        history['beta'].append(args.beta)
        history['train_forget_acc'].append(eval_res.get('Train_Forget', {}).get('acc', float('nan')))
        history['train_forget_loss'].append(eval_res.get('Train_Forget', {}).get('loss', float('nan')))
        history['train_retain_acc'].append(eval_res.get('Train_Retain', {}).get('acc', float('nan')))
        history['train_retain_loss'].append(eval_res.get('Train_Retain', {}).get('loss', float('nan')))
        history['test_retain_acc'].append(eval_res['Test_Retain ']['acc'])
        history['test_retain_loss'].append(eval_res['Test_Retain ']['loss'])

        current_metrics = {
            "train_retain_acc": history['train_retain_acc'][-1],
            "test_retain_acc": history['test_retain_acc'][-1],
            "train_forget_acc": history['train_forget_acc'][-1],
        }
        current_score = epoch_selection_score(current_metrics, retrain_target)
        if current_score < best_score:
            best_score = current_score
            best_idx = len(history['epoch']) - 1
            best_state = copy.deepcopy(original_model.state_dict())

    if torch.cuda.is_available(): torch.cuda.synchronize()
    time_unlearn_total = time.time() - start_time_unlearn
    time_unlearn = max(0.0, time_unlearn_total - time_eval)
    print(f"  [Time] Final Unlearning Updates: {time_unlearn:.2f}s")
    print(f"  [Time] Diagnostic Evaluation   : {time_eval:.2f}s")

    save_stem = args.save_tag or f"{args.dataset}_{args.model}_{args.expert_model}_seed{args.seed}"
    unlearned_save_path = os.path.join(args.save_path, f"vilun_{save_stem}.pth")
    if best_state is not None:
        original_model.load_state_dict(best_state)
    torch.save(original_model.state_dict(), unlearned_save_path)
    print(f"\n[Save] Unlearned model saved to '{unlearned_save_path}'")

                                                                 
    forget_indices_path = os.path.join(
        args.history_dir,
        f"forget_indices_{args.dataset}_{args.model}_seed{args.seed}.pt"
    )
    if not os.path.exists(forget_indices_path):
        torch.save(forget_idx, forget_indices_path)
        print(f"[Save] Forget indices saved to '{forget_indices_path}'")
    else:
        print(f"[Save] Forget indices already exist: '{forget_indices_path}'")

    os.makedirs(args.history_dir, exist_ok=True)
    csv_filename = os.path.join(args.history_dir, f"vilun_{save_stem}.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    print(f"[Save] Epoch tracking data saved to '{csv_filename}'")

                                                                   
    summary_csv = os.path.join(args.history_dir, "summary_vilun.csv")
    best_test_retain_acc = history['test_retain_acc'][best_idx]
    best_train_retain_acc = history['train_retain_acc'][best_idx]
    best_forget_acc = history['train_forget_acc'][best_idx]
    write_header = not os.path.exists(summary_csv) or os.path.getsize(summary_csv) == 0
    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow([
                'Method', 'Dataset', 'Orig_Model', 'Expert_Model', 'Alpha', 'Beta', 'Seed', 'Save_Tag',
                'Best_Epoch', 'Best_Train_Retain_Acc', 'Best_Test_Retain_Acc', 'Best_Train_Forget_Acc',
                'Time_Orig_Train(s)', 'Time_Expert_Train(s)', 'Time_Unlearn(s)', 'Time_Eval(s)',
                'Model_Path', 'Forget_Indices_Path'
            ])
        writer.writerow([
            'ViLUN',
            args.dataset, args.model, args.expert_model, args.alpha, args.beta, args.seed, save_stem,
            best_idx + 1,
            f"{best_train_retain_acc:.2f}", f"{best_test_retain_acc:.2f}", f"{best_forget_acc:.2f}",
            f"{time_orig_train:.2f}", f"{time_expert_train:.2f}", f"{time_unlearn:.2f}", f"{time_eval:.2f}",
            unlearned_save_path, forget_indices_path
        ])
    print(f"\n[Save] Summary → '{summary_csv}'")
    print(f"\n[Save] All results & Computation times safely saved!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="yale", choices=["yale", "mnist", "cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--target_id", type=int, default=0,
                        help="Yale only: subject index to unlearn (0-14). Ignored for CIFAR/MNIST.")
    parser.add_argument("--train_epochs", type=int, default=200)
    parser.add_argument("--unlearn_epochs", type=int, default=50)
    
    parser.add_argument("--unlearn_lr", type=float, default=0.00005)
    parser.add_argument("--orig_lr", type=float, default=1e-3)
    parser.add_argument("--expert_lr", type=float, default=1e-3)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--feature_margin", type=float, default=-0.3,
                        help="Cosine margin for feature-space forget repulsion")
    parser.add_argument("--heldout_ratio", type=float, default=0.1,
                        help="Ratio of test set to use as held-out data for feature-space alignment")
    parser.add_argument("--retain_ratio", type=float, default=1.0,
                        help="Fraction of retain data used for retain CE")
    parser.add_argument("--anchor_ratio", type=float, default=0.1,
                        help="Ratio of non-forget train samples used as an optional anchor KD set")
    parser.add_argument("--anchor_alpha", type=float, default=2.0,
                        help="Weight for optional anchor-set KD")
    parser.add_argument("--anchor_batch_size", type=int, default=64,
                        help="Batch size for optional anchor-set KD")
    parser.add_argument("--no_class_balanced_anchor", action="store_true",
                        help="Use random anchors instead of class-balanced anchors")
    parser.add_argument("--no_class_balanced_retain", action="store_true",
                        help="Use random sampling instead of class-balanced sampling for retain CE subset")
    parser.add_argument("--projector_epochs", type=int, default=3,
                        help="Epochs used to align and freeze the expert-to-original feature projector")
    parser.add_argument("--projector_lr", type=float, default=1e-3,
                        help="Learning rate used only for feature projector alignment")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience for unlearning")
    
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="resnet18", choices=["cnn", "resnet18", "resnet50", "lenet", "vit", "mlp", "rnn"])
    parser.add_argument("--expert_model", type=str, default="cnn", choices=["cnn", "resnet18", "resnet50", "lenet", "vit", "mlp", "rnn"])
    parser.add_argument("--original",    type=str, default=None,
                        help="Path to pre-trained original model")
    parser.add_argument("--expert_path", type=str, default=None,
                        help="Path to pre-trained expert(villain) model")
    parser.add_argument("--expert_epochs", type=int, default=50,
                        help="Epochs to train expert model on forget set")
    parser.add_argument("--expert_diverge_gamma", type=float, default=0.5,
                        help="Weight for making the expert feature diverge from the original feature on forget samples")
    parser.add_argument("--save_path", type=str, default="./saved_models")
    parser.add_argument("--history_dir", type=str, default="./history")
    parser.add_argument("--save_tag", type=str, default=None,
                        help="Optional suffix used to make per-experiment outputs unique")

    parser.add_argument("--tune", action="store_true", help="Enable Optuna hyperparameter tuning")
    parser.add_argument("--n_trials", type=int, default=20, help="Number of Optuna trials to run")
    parser.add_argument("--tune_patience", type=int, default=8,
                        help="Early stopping patience used only inside Optuna trials")
    parser.add_argument("--tune_max_epochs", type=int, default=60,
                        help="Maximum epochs used only inside Optuna trials")
    parser.add_argument("--reuse_tuned_params", dest="reuse_tuned_params", action="store_true",
                        help="Reuse cached tuned beta/lr for the same dataset/model/expert setting")
    parser.add_argument("--no_reuse_tuned_params", dest="reuse_tuned_params", action="store_false")
    parser.set_defaults(reuse_tuned_params=True)
    parser.add_argument("--freeze_beta", action="store_true",
                        help="Keep the CLI beta value even when cached tuned parameters are reused")
    
    args = parser.parse_args()
    run_pipeline(args)
