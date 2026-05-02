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
import time                

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
    return CIFAR10_PRETRAIN_EPOCHS


def get_original_pretrain_lr(dataset):
    if dataset == 'cifar100':
        return CIFAR100_PRETRAIN_LR
    if dataset == 'tinyimagenet':
        return TINYIMAGENET_PRETRAIN_LR
    if dataset == 'yale':
        return 1e-3
    return REFERENCE_PRETRAIN_LR

                                            
                                
                                            
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

class EarlyStopping:
    def __init__(self, patience=50, mode='max'):
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
        elif (self.mode == 'min' and score > self.best_score) or\
             (self.mode == 'max' and score < self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_weights = copy.deepcopy(model.state_dict())
            self.counter = 0

                                            
                 
                                            
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

def get_train_transform(dataset, size, norm, augment=False):
    transforms_list = [transforms.Resize((size, size))]
    if augment and dataset in ('cifar100', 'tinyimagenet'):
        transforms_list.extend([
            transforms.RandomCrop(size, padding=4),
            transforms.RandomHorizontalFlip(),
        ])
    transforms_list.extend([transforms.ToTensor(), norm])
    return transforms.Compose(transforms_list)


def load_dataset_factory(args, train_augment=False):
    data_dir = args.data_dir
    if args.dataset == 'yale':
        yale_dir = os.path.join(data_dir, 'yale')
        full_dataset, num_classes, in_channels = load_yale_custom(yale_dir)
        return full_dataset, full_dataset, num_classes, in_channels
    elif args.dataset == 'cifar10':
        norm = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_set = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True,
                                                transform=get_train_transform('cifar10', 32, norm, train_augment))
        test_set = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True,
                                               transform=get_train_transform('cifar10', 32, norm, False))
        return train_set, test_set, 10, 3
    elif args.dataset == 'cifar100':
        norm = transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True,
                                                 transform=get_train_transform('cifar100', 32, norm, train_augment))
        test_set = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True,
                                                transform=get_train_transform('cifar100', 32, norm, False))
        return train_set, test_set, 100, 3
    elif args.dataset == 'tinyimagenet':
        root = find_tinyimagenet_root(data_dir)
        norm = transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))
        train_set = torchvision.datasets.ImageFolder(
            os.path.join(root, 'train'),
            transform=get_train_transform('tinyimagenet', 64, norm, train_augment)
        )
        test_set = TinyImageNetValDataset(
            root, train_set.class_to_idx,
            transform=get_train_transform('tinyimagenet', 64, norm, False)
        )
        return train_set, test_set, 200, 3
    elif args.dataset == 'mnist':
        norm = transforms.Normalize((0.1307,), (0.3081,))
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True,
                                              transform=get_train_transform('mnist', 32, norm, train_augment))
        test_set = torchvision.datasets.MNIST(root=data_dir, train=False, download=True,
                                             transform=get_train_transform('mnist', 32, norm, False))
        return train_set, test_set, 10, 1
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

                                            
           
                                            

class ViTNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super(ViTNetwork, self).__init__()
        self.model = torchvision.models.vit_b_16(weights=None)
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

def get_model(model_name, in_channels, num_classes, img_size):
    if model_name == 'cnn': return CNNNetwork(in_channels, num_classes), 64
    elif model_name == 'resnet18': return ResNet18Network(in_channels, num_classes), 512
    elif model_name == 'resnet50': return ResNet50Network(in_channels, num_classes), 2048
    elif model_name == 'vit': return ViTNetwork(in_channels, num_classes), 768
    elif model_name == 'mlp': return MLPNetwork(in_channels, num_classes, img_size), 128
    elif model_name == 'rnn': return RNNNetwork(in_channels, num_classes, img_size), 256
    else: raise ValueError(f"Unknown model: {model_name}")

                                            
                      
                                            
def evaluate_4_quadrant(model, train_set, test_set, forget_indices, device):
    """
    Sample-target unlearning evaluation.
    forget_indices: set or array of train-set indices that form the forget set.
    Train_Forget  = forget samples
    Train_Retain  = all other train samples
    Test_Forget / Test_Retain are not split here (no ground-truth forget label in test),
    so we evaluate the full test set as Test_Retain and skip Test_Forget.
    """
    forget_idx_set = set(forget_indices.tolist() if hasattr(forget_indices, 'tolist') else list(forget_indices))
    all_train_idx  = list(range(len(train_set)))
    retain_train_idx = [i for i in all_train_idx if i not in forget_idx_set]
    forget_train_idx = list(forget_idx_set)

    def make_loader(dataset, indices):
        if len(indices) == 0: return None
        return DataLoader(Subset(dataset, indices), batch_size=128, shuffle=False)

    loaders = {
        "Train_Forget": make_loader(train_set, forget_train_idx),
        "Train_Retain": make_loader(train_set, retain_train_idx),
        "Test_Retain ": DataLoader(test_set, batch_size=128, shuffle=False),
    }

    model.eval()
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
            break
            
    if early_stopper.best_weights is not None:
        model.load_state_dict(early_stopper.best_weights)
    return model


def split_indices(indices, val_split, seed):
    indices = list(indices.tolist() if hasattr(indices, 'tolist') else indices)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    n_val = max(1, int(len(indices) * val_split))
    return indices[n_val:], indices[:n_val]


def evaluate_loader(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            out = model(data)
            loss_sum += F.cross_entropy(out, target, reduction='sum').item()
            correct += (out.argmax(1) == target).sum().item()
            total += target.size(0)
    return 100.0 * correct / total, loss_sum / total


def build_retrain_optimizer(model, dataset, lr):
    if dataset in ('cifar100', 'tinyimagenet'):
        weight_decay = TINYIMAGENET_WEIGHT_DECAY if dataset == 'tinyimagenet' else CIFAR100_WEIGHT_DECAY
        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=0.9,
            weight_decay=weight_decay,
        )
        return optimizer, True
    return optim.Adam(model.parameters(), lr=lr), False

                                            
                               
                                            
def run_unlearning_epoch(model, loader, optimizer, criterion, device, method):
    model.train()
    for data, target in loader:
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        
        if method == 'retrain':
                              
            loss = criterion(outputs, target)
        elif method == 'gradient_ascent':
                                                       
            loss = -criterion(outputs, target)
            
        loss.backward()
        optimizer.step()

                                            
                                           
                                            
def update_metadata(filename, unlearning, dataset, model_name, seed, best_epoch, best_score, best_time):
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['unlearning', 'dataset', 'model', 'seed', 'best_epoch', 'best_test_retain_acc', 'best_train_forget_acc', 'best_score', 'time_unlearn(s)'])
                                                                                                
        writer.writerow([unlearning, dataset, model_name, seed, best_epoch, 'N/A', 'N/A', f"{best_score:.4f}", f"{best_time:.4f}"])

                                            
                 
                                            

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
        'mnist':    'cnn',
        'yale':     'mlp',
    }
    if args.model == 'resnet18':                                       
        return default_map.get(args.dataset, 'resnet18')
    return args.model                                                      

def run_pipeline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = resolve_model(args)
    print(f"=== {args.unlearning.upper()} Pipeline Started ===  [Model: {args.model}]")
    
    train_set, test_set, num_classes, in_channels = load_dataset_factory(args)
    img_size = 64 if args.dataset in ('yale', 'tinyimagenet') else 32

                                                                  
                                                                    
                                                                              
                                                                     
    _fi_path = os.path.join(args.history_dir,
                            f"forget_indices_{args.dataset}_{args.model}_seed{args.seed}.pt")
    if os.path.exists(_fi_path):
        forget_indices = torch.load(_fi_path, map_location='cpu')
        print(f"[Load] Forget indices ← {_fi_path} ({len(forget_indices)} samples)", flush=True)
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
            forget_indices = torch.tensor(random.sample(_subj, min(11, len(_subj))))
            print(f"[Yale] Subject {args.target_id}: {len(forget_indices)} images as forget set")
        else:
            random.seed(args.seed)                    
            _n = len(train_set)
            _nf = max(1, int(0.1 * _n))
            forget_indices = torch.tensor(random.sample(range(_n), _nf))
            print(f"[{args.dataset}] Sampled {_nf}/{_n} (10%) as forget set")

    forget_idx_set  = set(forget_indices.tolist())
    retain_indices  = torch.tensor([i for i in range(len(train_set)) if i not in forget_idx_set])
    retain_train_indices, retain_val_indices = split_indices(retain_indices, REFERENCE_VAL_SPLIT, args.seed)

    if args.unlearning == 'retrain' and args.dataset in ('cifar100', 'tinyimagenet'):
        aug_train_set, _, _, _ = load_dataset_factory(args, train_augment=True)
        retrain_loader = DataLoader(Subset(aug_train_set, retain_train_indices), batch_size=64, shuffle=True)
    else:
        retrain_loader = DataLoader(Subset(train_set, retain_train_indices), batch_size=64, shuffle=True)

    retain_loader     = DataLoader(Subset(train_set, retain_indices), batch_size=64, shuffle=True)
    retain_val_loader = DataLoader(Subset(train_set, retain_val_indices), batch_size=128, shuffle=False)
    forget_loader     = DataLoader(Subset(train_set, forget_indices), batch_size=64, shuffle=True)
    train_full_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)

    final_model, _ = get_model(args.model, in_channels, num_classes, img_size)
    final_model = final_model.to(device)

    if args.unlearning == 'gradient_ascent':
        if args.original is not None and os.path.exists(args.original):
            print(f"  >> Loading pre-trained Original Model from: {args.original}")
            final_model.load_state_dict(torch.load(args.original, map_location=device))
        else:
            print(f"  >> No original model found. Training from scratch for Gradient Ascent...")
            final_model = train_standard(final_model, train_full_loader, device, epochs=50, lr=args.train_lr, patience=50)

    if args.unlearning == 'retrain':
        args.unlearn_epochs = get_original_pretrain_epochs(args.dataset)
        args.train_lr = get_original_pretrain_lr(args.dataset)
        unlearn_loader = retrain_loader
        optimizer, use_cosine = build_retrain_optimizer(final_model, args.dataset, args.train_lr)
        scheduler = (
            optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.unlearn_epochs)
            if use_cosine else None
        )
        print(
            f"  >> Retrain uses original pretrain recipe on retain data only: "
            f"epochs={args.unlearn_epochs}, lr={args.train_lr}, "
            f"retain_train={len(retain_train_indices)}, retain_val={len(retain_val_indices)}"
        )
    else:
        unlearn_loader = forget_loader
        optimizer = optim.SGD(final_model.parameters(), lr=args.unlearn_lr,
                              momentum=args.ga_momentum, weight_decay=args.ga_weight_decay)
        scheduler = None
    criterion = nn.CrossEntropyLoss()
    early_stopper = EarlyStopping(patience=args.patience, mode='min')
    best_weights = None
    best_score = None
    
    history_data = []
    cumulative_time = 0.0

    active_lr = args.train_lr if args.unlearning == "retrain" else args.unlearn_lr
    print(f"\n[Execution] Running {args.unlearning.upper()} (LR: {active_lr})...")
    for epoch in range(args.unlearn_epochs):
        
        start_time = time.time()
        run_unlearning_epoch(final_model, unlearn_loader, optimizer, criterion, device, args.unlearning)
        if scheduler is not None:
            scheduler.step()
        end_time = time.time()
        
        epoch_time = end_time - start_time
        cumulative_time += epoch_time
        
        eval_res = evaluate_4_quadrant(final_model, train_set, test_set, forget_indices, device)
        
        t_f_acc  = eval_res.get('Train_Forget', {}).get('acc', float('nan'))
        t_f_loss = eval_res.get('Train_Forget', {}).get('loss', float('nan'))
        t_r_acc  = eval_res['Test_Retain ']['acc']
        t_r_loss = eval_res['Test_Retain ']['loss']
        val_acc = float('nan')
        val_loss = float('nan')
        if args.unlearning == 'retrain':
            val_acc, val_loss = evaluate_loader(final_model, retain_val_loader, device)
            score = val_acc
            if best_score is None or score > best_score:
                best_score = score
                best_weights = copy.deepcopy(final_model.state_dict())
        else:
            score = t_f_acc
            early_stopper(score, final_model)
        
        tr_r_acc  = eval_res.get('Train_Retain', {}).get('acc', float('nan'))
        tr_r_loss = eval_res.get('Train_Retain', {}).get('loss', float('nan'))
        history_data.append({
            'epoch': epoch + 1,
            'train_forget_acc':  t_f_acc,
            'train_forget_loss': t_f_loss,
            'train_retain_acc':  tr_r_acc,
            'train_retain_loss': tr_r_loss,
            'retain_val_acc':    val_acc,
            'retain_val_loss':   val_loss,
            'test_retain_acc':   t_r_acc,
            'test_retain_loss':  t_r_loss,
            'score':             score,
            'time':              cumulative_time
        })
        
        val_msg = f" | Val: {val_acc:6.2f}%" if args.unlearning == 'retrain' else ""
        print(f"  Epoch [{epoch+1:>4}/{args.unlearn_epochs}] | Score: {score:6.2f}{val_msg} | Retain: {t_r_acc:6.2f}% | Forget: {t_f_acc:6.2f}% | Time: {cumulative_time:.1f}s")

        if args.unlearning != 'retrain' and early_stopper.early_stop:
            print(f"  [Early Stop] Unlearning stopped at epoch {epoch+1}.")
            break

    csv_filename = os.path.join(args.history_dir, f"{args.unlearning}_{args.dataset}_{args.model}_seed{args.seed}.csv")
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=history_data[0].keys())
        writer.writeheader()
        writer.writerows(history_data)
    print(f"[Save] History saved to '{csv_filename}'")

                     
    if args.unlearning == 'retrain' and best_weights is not None:
        final_model.load_state_dict(best_weights)
    elif args.unlearning != 'retrain' and early_stopper.best_weights is not None:
        final_model.load_state_dict(early_stopper.best_weights)

           
    model_filename = os.path.join(
        args.save_path,
        f"{args.unlearning}_{args.dataset}_{args.model}_seed{args.seed}.pth"
    )
    torch.save(final_model.state_dict(), model_filename)
    print(f"[Save] Unlearned model saved to '{model_filename}'")

                                                                 
    forget_indices_path = os.path.join(
        args.history_dir,
        f"forget_indices_{args.dataset}_{args.model}_seed{args.seed}.pt"
    )
    if not os.path.exists(forget_indices_path):
        torch.save(forget_indices, forget_indices_path)
        print(f"[Save] Forget indices saved to '{forget_indices_path}'")
    else:
        print(f"[Save] Forget indices already exist: '{forget_indices_path}'")

    best_epoch_record = (
        max(history_data, key=lambda x: x['score'])
        if args.unlearning == 'retrain'
        else min(history_data, key=lambda x: x['score'])
    )
    metadata_filename = os.path.join(args.history_dir, "summary_retrain_ga.csv")
    
    summary_exists = os.path.exists(metadata_filename)
    with open(metadata_filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow(['unlearning', 'dataset', 'model', 'seed', 'best_epoch',
                             'best_train_retain_acc', 'best_test_retain_acc', 'best_train_forget_acc',
                             'selection_metric', 'time_unlearn(s)'])
        writer.writerow([
            args.unlearning, args.dataset, args.model, args.seed,
            best_epoch_record['epoch'],
            round(best_epoch_record['train_retain_acc'] if best_epoch_record['train_retain_acc'] == best_epoch_record['train_retain_acc'] else 0.0, 4),
            round(best_epoch_record['test_retain_acc'],  4),
            round(best_epoch_record['train_forget_acc'] if best_epoch_record['train_forget_acc'] == best_epoch_record['train_forget_acc'] else 0.0, 4),
            round(best_epoch_record['score'], 4),
            round(best_epoch_record['time'], 4)
        ])
    print(f"[Save] Metadata updated at '{metadata_filename}' with Best Epoch: {best_epoch_record['epoch']} (Time to best: {best_epoch_record['time']:.2f}s)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unlearning", type=str, required=True, choices=["retrain", "gradient_ascent"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["yale", "mnist", "cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--data_dir", type=str, default="./data")
    parser.add_argument("--target_id", type=int, default=0,
                        help="Yale only: subject index to unlearn (0-14). Ignored for CIFAR/MNIST.")
    parser.add_argument("--unlearn_epochs", type=int, default=100)
    parser.add_argument("--train_lr",   type=float, default=1e-3,
                        help="LR for Retrain (Adam)")
    parser.add_argument("--unlearn_lr", type=float, default=1e-4,
                        help="LR for Gradient Ascent (SGD)")
    parser.add_argument("--retrain_weight_decay", type=float, default=0.0,
                        help="Weight decay for retrain")
    parser.add_argument("--ga_momentum", type=float, default=0.5,
                        help="Momentum for gradient ascent")
    parser.add_argument("--ga_weight_decay", type=float, default=1e-4,
                        help="Weight decay for gradient ascent")
    parser.add_argument("--patience", type=int, default=50, help="Early stopping patience for unlearning")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model", type=str, default="resnet18", choices=["cnn", "resnet18", "resnet50", "lenet", "vit", "mlp", "rnn"])
    parser.add_argument("--original", type=str, default=None, help="Path to original model for gradient ascent")
    parser.add_argument("--save_path", type=str, default="./saved_models")
    parser.add_argument("--history_dir", type=str, default="./history")
    
    args = parser.parse_args()
    run_pipeline(args)
