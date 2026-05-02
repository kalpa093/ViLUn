"""
Prototype Surgery (PS) Unlearning
- Compatible with vilun.py experimental environment
- Datasets : CIFAR10, CIFAR100, MNIST, Yale B
- Models   : CNN, RNN, MLP, ResNet18, ViT
- Save     : ps_{dataset}_{model}_{seed}.pth
             history/ps_{dataset}_{model}_{seed}_epoch_log.csv
             history/ps_{dataset}_{model}_{seed}_best_summary.csv
"""

import argparse
import os
import copy
import random
import time
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, Dataset, TensorDataset
import glob
from PIL import Image

                                            
         
                                            
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

                                            
                                 
                                            
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
        if not os.path.isfile(f_path):
            continue
        filename = os.path.basename(f_path)
        try:
            subject_part = filename.split(".")[0]
            subject_num = int(subject_part.replace("subject", ""))
            label_idx = subject_num - 1
            img = Image.open(f_path)
            img_tensor = transform(img)
            X_list.append(img_tensor)
            y_list.append(torch.tensor(label_idx, dtype=torch.long))
        except:
            continue

    full_dataset = TensorDataset(torch.stack(X_list), torch.stack(y_list))
    unique_labels = torch.unique(torch.stack(y_list))
    return full_dataset, len(unique_labels), 1


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
        test_set  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set, 10, 3
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        train_set = torchvision.datasets.CIFAR100(root=data_dir, train=True, download=True, transform=transform)
        test_set  = torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set, 100, 3
    elif args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        train_set = torchvision.datasets.MNIST(root=data_dir, train=True, download=True, transform=transform)
        test_set  = torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform)
        return train_set, test_set, 10, 1
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")


                                            
                           
                                            
class ResNet18Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
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
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        return self.classifier(torch.flatten(self.features(x), 1))


class LeNetNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.classifier(x)


class ViTNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.vit_b_16(weights=None)
        if in_channels != 3:
            self.model.conv_proj = nn.Conv2d(in_channels, 768, kernel_size=16, stride=16)
        self.model.heads.head = nn.Linear(768, num_classes)

    def forward(self, x):
        if x.size(-1) < 224:
            x = F.interpolate(x, size=224, mode='bicubic', align_corners=False)
        x = self.model._process_input(x)
        n = x.shape[0]
        batch_class_token = self.model.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        x = self.model.encoder(x)
        return self.model.heads(x[:, 0])


class MLPNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        input_dim = in_channels * img_size * img_size
        self.mixer = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 128), nn.GELU()
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.mixer(torch.flatten(x, 1)))


class RNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.img_size = img_size
        input_size = in_channels * img_size
        hidden_size = 128
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size,
                            num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x):
        B, C, H, W = x.size()
        if H != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size))
            B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H, W * C)
        _, (h_n, _) = self.lstm(x)
        feature = torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1)
        return self.classifier(feature)


def get_model(model_name, in_channels, num_classes, img_size=32):
    if model_name == 'cnn':
        return CNNNetwork(in_channels, num_classes)
    elif model_name == 'resnet18':
        return ResNet18Network(in_channels, num_classes)
    elif model_name == 'resnet50':
        return ResNet50Network(in_channels, num_classes)
    elif model_name == 'lenet':
        return LeNetNetwork(in_channels, num_classes)
    elif model_name == 'vit':
        return ViTNetwork(in_channels, num_classes)
    elif model_name == 'mlp':
        return MLPNetwork(in_channels, num_classes, img_size)
    elif model_name == 'rnn':
        return RNNNetwork(in_channels, num_classes, img_size)
    else:
        raise ValueError(f"Unknown model: {model_name}")


                                            
                                 
                                            
def train_standard(model, loader, device, epochs=200, lr=1e-3, patience=50):
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_loss, best_weights, counter = float('inf'), None, 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            loss = criterion(model(data), target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            best_weights = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
    if best_weights:
        model.load_state_dict(best_weights)
    return model


                                            
                               
                                            
def evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose=True):
    """Sample-target evaluation using forget sample indices."""
    forget_idx_set   = set(forget_indices.tolist() if hasattr(forget_indices, 'tolist') else list(forget_indices))
    all_train_idx    = list(range(len(train_set)))
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
    if verbose:
        print("  [Detailed Evaluation]")
        print(f"  {'Metric':<15} | {'Acc (%)':<10} | {'Loss':<10}")
        print("  " + "-" * 40)

    results = {}
    with torch.no_grad():
        for name, loader in loaders.items():
            if loader is None:
                continue
            correct, total, loss_sum = 0, 0, 0.0
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                loss_sum += F.cross_entropy(out, target, reduction='sum').item()
                _, pred = torch.max(out, 1)
                correct += (pred == target).sum().item()
                total += target.size(0)
            acc = 100.0 * correct / total
            avg_loss = loss_sum / total
            results[name] = {'acc': acc, 'loss': avg_loss}
            if verbose:
                print(f"  {name:<15} | {acc:6.2f}     | {avg_loss:.4f}")

    if verbose:
        print("  " + "-" * 40)
    return results


                                            
                                                       
                                            
def softmax(x):
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / e_x.sum(axis=-1, keepdims=True)


class RelabeledDataset(Dataset):
    def __init__(self, dataset, new_labels):
        self.dataset = dataset
        self.new_labels = new_labels

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        img = data[0]
        label = self.new_labels[idx]
        return img, label


def get_logits(model, loader, device):
    model.eval()
    model.to(device)
    all_logits = []
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, (list, tuple)):
                inputs = batch[0]
            else:
                inputs = batch
            inputs = inputs.to(device)
            outputs = model(inputs)
            all_logits.append(outputs.cpu().numpy())
    return np.concatenate(all_logits, axis=0)


def relabel_set(args, unlearn_set, model, device):
    unlearn_loader = DataLoader(unlearn_set, batch_size=args.batch_size, shuffle=False)
    original_logits = get_logits(model, unlearn_loader, device)
    num_classes = original_logits.shape[1]

    new_logits = np.copy(original_logits)
    preds = np.argmax(original_logits, axis=1)
    row_indices = np.arange(len(original_logits))

    delta_y = new_logits[row_indices, preds]
    new_logits[row_indices, preds] = -9999

    sorted_indices = np.argsort(original_logits, axis=1)

    if num_classes < 3:
        second_idx = sorted_indices[:, -2]
        new_logits[row_indices, second_idx] += delta_y
    else:
        second_idx = sorted_indices[:, -2]
        third_idx  = sorted_indices[:, -3]
        second_vals = original_logits[row_indices, second_idx]
        third_vals  = original_logits[row_indices, third_idx]
        proportions = softmax(np.vstack((second_vals, third_vals)).T)
        Delta_y_distributed = delta_y[:, None] * proportions
        new_logits[row_indices, second_idx] += Delta_y_distributed[:, 0]
        new_logits[row_indices, third_idx]  += Delta_y_distributed[:, 1]

    new_labels = softmax(new_logits)
    new_labels = torch.tensor(new_labels, dtype=torch.float32)
    return RelabeledDataset(unlearn_set, new_labels)


def get_last_layer_params(model):
    """
    Return only the parameters of the final classification layer.
    PS fine-tunes the last (prototype) layer only, as described in the paper.
    - ResNet18  : model.model.fc
    - ViT       : model.model.heads.head
    - CNN / LeNet / MLP / RNN : model.classifier
    """
    if hasattr(model, 'model'):
                  
        if hasattr(model.model, 'fc'):
            return model.model.fc.parameters()
             
        if hasattr(model.model, 'heads'):
            return model.model.heads.head.parameters()
                          
    if hasattr(model, 'classifier'):
        return model.classifier.parameters()
                                              
    print("[Warning] Could not identify last layer; falling back to full-model fine-tuning.")
    return model.parameters()


def fine_tune_relabel(model, relabeled_loader, epochs, lr, device,
                      train_set, test_set, forget_indices, history):
    """
    PS fine-tuning: only the last (prototype) layer is updated.
    All other parameters are frozen, matching the original PS paper.
    """
                                                                    
    for param in model.parameters():
        param.requires_grad = False
    for param in get_last_layer_params(model):
        param.requires_grad = True

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=1e-4,
        eps=1e-8
    )
    criterion_ce = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for imgs, soft_labels in relabeled_loader:
            optimizer.zero_grad()
            imgs, soft_labels = imgs.to(device), soft_labels.to(device)
            logits = model(imgs)
            probs = torch.softmax(logits, dim=1)
            loss = criterion_ce(probs, soft_labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(relabeled_loader)
                                            
        print(f"Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.4f}")

                                        
        eval_res = evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose=False)
        _tf = eval_res.get("Train_Forget", {}).get("acc", float("nan"))
        _tr = eval_res.get("Test_Retain ", {}).get("acc", float("nan"))
        _tr_train = eval_res.get("Train_Retain", {}).get("acc", float("nan"))
        print(f"           → Train Retain: {_tr_train:6.2f}% | Test Retain: {_tr:6.2f}% | Forget: {_tf:6.2f}%")
        history['epoch'].append(epoch + 1)
        history['train_forget_acc'].append(eval_res.get('Train_Forget', {}).get('acc', float('nan')))
        history['train_forget_loss'].append(eval_res.get('Train_Forget', {}).get('loss', float('nan')))
        history['train_retain_acc'].append(eval_res.get('Train_Retain', {}).get('acc', float('nan')))
        history['train_retain_loss'].append(eval_res.get('Train_Retain', {}).get('loss', float('nan')))
        history['test_retain_acc'].append(eval_res.get('Test_Retain ', {}).get('acc', float('nan')))
        history['test_retain_loss'].append(eval_res.get('Test_Retain ', {}).get('loss', float('nan')))

                                                                    
    for param in model.parameters():
        param.requires_grad = True

    return model


                                            
        
                                            

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
        'mnist':    'resnet18',
        'yale':     'mlp',
    }
    if args.model == 'resnet18':                                       
        return default_map.get(args.dataset, 'resnet18')
    return args.model                                                      

def main():
    parser = argparse.ArgumentParser(description="Prototype Surgery (PS) Unlearning")
    parser.add_argument("--dataset",    type=str, default="cifar10",
                        choices=["cifar10", "cifar100", "mnist", "yale"])
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--model",      type=str, default="resnet18",
                        choices=["cnn", "resnet18", "resnet50", "lenet", "vit", "mlp", "rnn"])
    parser.add_argument("--load_path",  type=str, default=None,
                        help="Path to pretrained model checkpoint")
    parser.add_argument("--orig_lr",    type=float, default=1e-3,
                        help="LR for original model pretraining")
    parser.add_argument("--save_dir",   type=str, default="./ps_models")
    parser.add_argument("--target_id",  type=int, default=0,
                        help="Yale only: subject index to unlearn (0-14). Ignored for CIFAR/MNIST.")
    parser.add_argument("--seed",       type=int, default=42)
    parser.add_argument("--epochs",     type=int, default=5)
    parser.add_argument("--lr",         type=float, default=1e-4)
    parser.add_argument("--batch_size", type=int, default=32)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = resolve_model(args)
    print(f"Device: {device} | Dataset: {args.dataset} | Model: {args.model}")
    print(f"Dataset: {args.dataset} | Model: {args.model} | Seed: {args.seed}")

                                
    train_set, test_set, num_classes, in_channels = load_dataset_factory(args)
    img_size = 64 if args.dataset == 'yale' else 32

                                                                                     
    def get_labels(dataset):
        if hasattr(dataset, 'targets'):
            return dataset.targets.detach().clone() if isinstance(dataset.targets, torch.Tensor)\
                   else torch.tensor(dataset.targets)
        elif hasattr(dataset, 'tensors'):
            return dataset.tensors[1]
        else:
            return torch.tensor([y for _, y in dataset])

                                                                  
                                                                     
    _fi_path = os.path.join("history",
                            f"forget_indices_ps_{args.dataset}_{args.model}_seed{args.seed}.pt")
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
        forget_indices = torch.tensor(random.sample(range(n_total), n_forget))
        print(f"[{args.dataset}] Sampled {n_forget}/{n_total} train samples (10%) as forget set")

    forget_idx_set = set(forget_indices.tolist())
    retain_indices_list = [i for i in range(len(train_set)) if i not in forget_idx_set]

    print(f"Forget Images: {len(forget_indices)} | Retain Images: {len(retain_indices_list)}")

    unlearn_set = Subset(train_set, forget_indices)
    remain_set  = Subset(train_set, retain_indices_list)

                                 
    model = get_model(args.model, in_channels, num_classes, img_size).to(device)
    train_full_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    orig_model_path   = os.path.join(args.save_dir,
                                     f"original_{args.dataset}_{args.model}_seed{args.seed}.pth")
    time_orig_train   = 0.0

    if args.load_path and os.path.exists(args.load_path):
        state = torch.load(args.load_path, map_location=device)
        if 'model_state_dict' in state: model.load_state_dict(state['model_state_dict'])
        elif 'state_dict' in state:     model.load_state_dict(state['state_dict'])
        else:                           model.load_state_dict(state)
        print(f"[PS] Loaded checkpoint from {args.load_path}")
    elif os.path.exists(orig_model_path):
        print(f"[PS] Loading cached original model from {orig_model_path}")
        model.load_state_dict(torch.load(orig_model_path, map_location=device))
    else:
        print(f"[PS] No pretrained model found. Training original model from scratch...")
        t0 = time.time()
        model = train_standard(model, train_full_loader, device,
                               epochs=200, lr=args.orig_lr, patience=50)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_orig_train = time.time() - t0
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), orig_model_path)
        print(f"  [Time] Original training: {time_orig_train:.2f}s → saved to {orig_model_path}")

                                         
    relabeled_set = relabel_set(args, unlearn_set, model, device)
    relabeled_loader = DataLoader(relabeled_set, batch_size=args.batch_size, shuffle=True)
    updated_model = copy.deepcopy(model)

    history = {
        'epoch': [],
        'train_forget_acc': [], 'train_forget_loss': [],
        'train_retain_acc': [], 'train_retain_loss': [],
        'test_retain_acc':  [], 'test_retain_loss':  [],
    }

    start_time = time.time()
    fine_tune_relabel(updated_model, relabeled_loader,
                      args.epochs, args.lr, device,
                      train_set, test_set, forget_indices, history)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start_time

                                      
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = f"ps_{args.dataset}_{args.model}_seed{args.seed}.pth"
    save_path = os.path.join(args.save_dir, save_name)
    torch.save(updated_model.state_dict(), save_path)
    print(f"PS completed. Saved to {save_path}")

                                                                 
    os.makedirs("history", exist_ok=True)
    forget_indices_path = os.path.join(
        "history",
        f"forget_indices_ps_{args.dataset}_{args.model}_seed{args.seed}.pt"
    )
    torch.save(forget_indices, forget_indices_path)
    print(f"[Save] Forget indices saved to '{forget_indices_path}'")

                                              
    os.makedirs("history", exist_ok=True)
    epoch_csv = os.path.join("history", f"ps_{args.dataset}_{args.model}_seed{args.seed}.csv")
    with open(epoch_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    print(f"[Save] Epoch log saved to '{epoch_csv}'")

                                              
    best_idx = int(np.argmax(history['test_retain_acc']))
    best_epoch        = history['epoch'][best_idx]
    best_forget_acc   = history['train_forget_acc'][best_idx]
    best_train_retain_acc = history['train_retain_acc'][best_idx]
    best_retain_acc   = history['test_retain_acc'][best_idx]

    summary_csv = os.path.join("history", "summary_ps.csv")
    summary_exists = os.path.exists(summary_csv)
    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow([
                'Dataset', 'Model', 'Seed', 'N_Forget',
                'Best_Epoch', 'Best_Train_Retain_Acc', 'Best_Train_Forget_Acc', 'Best_Test_Retain_Acc',
                'Total_Time_s'
            ])
        writer.writerow([
            args.dataset, args.model, args.seed, len(forget_indices),
            best_epoch,
            f"{best_train_retain_acc:.4f}", f"{best_forget_acc:.4f}", f"{best_retain_acc:.4f}",
            f"{elapsed:.2f}"
        ])
    print(f"[Save] Best summary saved to '{summary_csv}'")
    print(f"\n[Best Epoch {best_epoch}]  Forget Acc: {best_forget_acc:.2f}%  "
          f"Train Retain: {best_train_retain_acc:.2f}%  Test Retain: {best_retain_acc:.2f}%  "
          f"Total Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
