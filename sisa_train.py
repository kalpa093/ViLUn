"""
SISA (Sharded, Isolated, Sliced, and Aggregated) Unlearning
- Compatible with vilun.py experimental environment
- Datasets : CIFAR10, CIFAR100, MNIST, Yale B
- Models   : CNN, RNN, MLP, ResNet18, ViT
- Save     : containers/{dataset}_{model}_{seed}/
               {dataset}_{model}_{seed}_shard{i}_slice{j}.pth   (original SISA logic)
             history/sisa_{dataset}_{model}_{seed}_epoch_log.csv
             history/sisa_{dataset}_{model}_{seed}_best_summary.csv
"""

import argparse
import os
import time
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import glob
from PIL import Image

                                            
         
                                            
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True


                                            
                                 
                                            
def load_yale_custom(root_dir, img_size=64):
    search_path = os.path.join(root_dir, "subject*")
    file_list   = glob.glob(search_path)
    if len(file_list) == 0:
        search_path = os.path.join(root_dir, "yalefaces", "subject*")
        file_list   = glob.glob(search_path)

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
            subject_num = int(filename.split(".")[0].replace("subject", ""))
            label_idx   = subject_num - 1
            img_tensor  = transform(Image.open(f_path))
            X_list.append(img_tensor)
            y_list.append(torch.tensor(label_idx, dtype=torch.long))
        except:
            continue

    full_dataset  = TensorDataset(torch.stack(X_list), torch.stack(y_list))
    unique_labels = torch.unique(torch.stack(y_list))
    return full_dataset, len(unique_labels), 1


def load_dataset_factory(args):
    data_dir = args.data_dir
    if args.dataset == 'yale':
        full_dataset, num_classes, in_channels = load_yale_custom(os.path.join(data_dir, 'yale'))
        return full_dataset, full_dataset, num_classes, in_channels
    elif args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        return (torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=transform),
                torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform),
                10, 3)
    elif args.dataset == 'cifar100':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761))
        ])
        return (torchvision.datasets.CIFAR100(root=data_dir, train=True,  download=True, transform=transform),
                torchvision.datasets.CIFAR100(root=data_dir, train=False, download=True, transform=transform),
                100, 3)
    elif args.dataset == 'mnist':
        transform = transforms.Compose([
            transforms.Resize((32, 32)), transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
        return (torchvision.datasets.MNIST(root=data_dir, train=True,  download=True, transform=transform),
                torchvision.datasets.MNIST(root=data_dir, train=False, download=True, transform=transform),
                10, 1)
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
        self.features   = nn.Sequential(
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
        self.features   = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, kernel_size=5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool        = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1         = nn.Linear(16 * 5 * 5, 120)
        self.fc2         = nn.Linear(120, 84)
        self.classifier  = nn.Linear(84, num_classes)

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
        x = torch.cat([self.model.class_token.expand(n, -1, -1), x], dim=1)
        x = self.model.encoder(x)
        return self.model.heads(x[:, 0])


class MLPNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.mixer      = nn.Sequential(
            nn.Linear(in_channels * img_size * img_size, 512), nn.GELU(),
            nn.Linear(512, 512), nn.GELU(),
            nn.Linear(512, 128), nn.GELU()
        )
        self.classifier = nn.Linear(128, num_classes)

    def forward(self, x):
        return self.classifier(self.mixer(torch.flatten(x, 1)))


class RNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.img_size   = img_size
        self.lstm       = nn.LSTM(input_size=in_channels * img_size, hidden_size=128,
                                  num_layers=2, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        B, C, H, W = x.size()
        if H != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size))
            B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H, W * C)
        _, (h_n, _) = self.lstm(x)
        return self.classifier(torch.cat((h_n[-2, :, :], h_n[-1, :, :]), dim=1))


def get_model(model_name, in_channels, num_classes, img_size=32):
    if model_name == 'cnn':       return CNNNetwork(in_channels, num_classes)
    elif model_name == 'resnet18': return ResNet18Network(in_channels, num_classes)
    elif model_name == 'resnet50': return ResNet50Network(in_channels, num_classes)
    elif model_name == 'lenet':    return LeNetNetwork(in_channels, num_classes)
    elif model_name == 'vit':      return ViTNetwork(in_channels, num_classes)
    elif model_name == 'mlp':      return MLPNetwork(in_channels, num_classes, img_size)
    elif model_name == 'rnn':      return RNNNetwork(in_channels, num_classes, img_size)
    else: raise ValueError(f"Unknown model: {model_name}")


                                            
                               
                                            
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
                total   += target.size(0)
            acc      = 100.0 * correct / total
            avg_loss = loss_sum / total
            results[name] = {'acc': acc, 'loss': avg_loss}
            if verbose:
                print(f"  {name:<15} | {acc:6.2f}     | {avg_loss:.4f}")

    if verbose:
        print("  " + "-" * 40)
    return results


@torch.no_grad()
def evaluate_ensemble(manager, model_name, in_channels, num_classes, img_size,
                      num_shards, num_slices, loader, device):
    models = []
    for shard_id in range(num_shards):
        model_path = manager.get_model_path(shard_id, num_slices - 1)
        if not os.path.exists(model_path):
            print(f"  [Warning] Checkpoint missing for shard {shard_id}, skipping.")
            continue
        m = get_model(model_name, in_channels, num_classes, img_size).to(device)
        ckpt = torch.load(model_path, map_location=device)
        m.load_state_dict(ckpt['model_state_dict'])
        m.eval()
        models.append(m)

    if len(models) == 0:
        print("  [Warning] No shard models available for evaluation.")
        return 0.0, 0.0

    criterion = nn.CrossEntropyLoss(reduction='sum')
    correct = 0
    total = 0
    total_loss = 0.0

    for batch in loader:
        inputs, labels = batch[0], batch[1]
        inputs, labels = inputs.to(device), labels.to(device)

        logits_sum = None
        for m in models:
            logits = m(inputs)
            logits_sum = logits if logits_sum is None else logits_sum + logits

        total_loss += criterion(logits_sum, labels).item()
        preds = logits_sum.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    if total == 0:
        return 0.0, 0.0
    return correct / total, total_loss / total


                                            
                                                                 
                                            
def train_one_slice(model, loader, criterion, optimizer, device, epochs):
    model.train()
    final_acc  = 0
    final_loss = 0

    for epoch in range(epochs):
        total_loss = 0
        correct    = 0
        total      = 0

        for batch in loader:
            if isinstance(batch, (list, tuple)):
                if len(batch) >= 2:
                    images, labels = batch[0], batch[1]
                else:
                    images = batch[0]
                    labels = torch.zeros(images.size(0))
            else:
                images = batch
                labels = torch.zeros(images.size(0))

            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            preds       = outputs.argmax(1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)

        final_acc  = correct / total if total > 0 else 0
        final_loss = total_loss / total if total > 0 else 0

    return final_acc, final_loss


def get_cumulative_indices(all_indices, shard_id, slice_id, num_shards, num_slices, unlearn_set=None):
    total_len   = len(all_indices)
    shard_size  = total_len // num_shards

    shard_start = shard_id * shard_size
    shard_end   = (shard_id + 1) * shard_size if shard_id < num_shards - 1 else total_len
    shard_indices = all_indices[shard_start:shard_end]

    slice_chunk_size   = len(shard_indices) // num_slices
    current_slice_end  = (slice_id + 1) * slice_chunk_size

    if slice_id == num_slices - 1:
        current_slice_end = len(shard_indices)

    cumulative_indices = shard_indices[:current_slice_end]

    if unlearn_set is not None and len(unlearn_set) > 0:
        return np.setdiff1d(cumulative_indices, list(unlearn_set))

    return cumulative_indices


                                            
                                                               
                                            
class SISAManager:
    def __init__(self, dataset, model_name, seed, save_dir, num_shards):
        """
        Container path: {save_dir}/containers/{dataset}_{model}_{seed}/
        Checkpoint    : {dataset}_{model}_{seed}_shard{i}_slice{j}.pth
        """
        self.dataset     = dataset
        self.model_name  = model_name
        self.seed        = seed
        self.save_dir    = save_dir
        self.num_shards  = num_shards
                             
        container_name   = f"{dataset}_{model_name}_{seed}"
        self.container_dir = os.path.join(save_dir, "containers", container_name)
        os.makedirs(self.container_dir, exist_ok=True)

    def get_model_path(self, shard_id, slice_id):
        fname = f"{self.dataset}_{self.model_name}_{self.seed}_shard{shard_id}_slice{slice_id}.pth"
        return os.path.join(self.container_dir, fname)


                                            
        
                                            

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
    parser = argparse.ArgumentParser(description="SISA Unlearning")
    parser.add_argument("--dataset",         type=str,   default="cifar10",
                        choices=["cifar10", "cifar100", "mnist", "yale"])
    parser.add_argument("--data_dir",        type=str,   default="./data")
    parser.add_argument("--model",           type=str,   default="resnet18",
                        choices=["cnn", "resnet18", "resnet50", "lenet", "vit", "mlp", "rnn"])
    parser.add_argument("--save_dir",        type=str,   default="./sisa_models")
    parser.add_argument("--epochs_per_slice",type=int,   default=5)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=1e-3)
    parser.add_argument("--shards",          type=int,   default=5,
                        help="Number of shards")
    parser.add_argument("--slices",          type=int,   default=3,
                        help="Number of slices per shard")
    parser.add_argument("--seed",            type=int,   default=42)
    parser.add_argument("--target_id",       type=int,   default=0,
                        help="Yale only: subject index to unlearn (0-14). Ignored for CIFAR/MNIST.")
    parser.add_argument("--unlearn",         action="store_true",
                        help="Run in unlearning mode (remove target_id class)")
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = resolve_model(args)
    print(f"Device: {device}")
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

    manager = SISAManager(
        dataset=args.dataset,
        model_name=args.model,
        seed=args.seed,
        save_dir=args.save_dir,
        num_shards=args.shards,
    )

    full_dataset = train_set
    all_indices  = np.arange(len(full_dataset))
    np.random.shuffle(all_indices)                                               

                                                                                
    unlearn_set    = set()
    affected_shards = {}

    if args.unlearn:
                                                                 
        _fi_path = os.path.join("history",
                                f"forget_indices_sisa_{args.dataset}_{args.model}_seed{args.seed}.pt")
        if os.path.exists(_fi_path):
            forget_indices = torch.load(_fi_path, map_location='cpu')
            forget_indices = list(forget_indices.numpy())
            print(f"[Load] Forget indices ← {_fi_path} ({len(forget_indices)} samples)", flush=True)
        else:
                      
            if args.dataset == 'yale':
                _all_labels = (train_set.tensors[1] if hasattr(train_set, 'tensors')
                               else torch.tensor([y for _, y in train_set]))
                _subj = (_all_labels == args.target_id).nonzero(as_tuple=True)[0].tolist()
                if not _subj: raise ValueError(f"No images for Yale subject {args.target_id}")
                forget_indices = random.sample(_subj, min(11, len(_subj)))
                print(f"[Yale] Subject {args.target_id}: {len(forget_indices)} images as forget set")
            else:
                random.seed(args.seed)                     
                _n = len(full_dataset)
                _nf = max(1, int(0.1 * _n))
                forget_indices = random.sample(range(_n), _nf)
                print(f"[{args.dataset}] Sampled {_nf}/{_n} (10%) as forget set")

        unlearn_set = set(forget_indices)
        forget_indices_tensor = torch.tensor(forget_indices)
        print(f"Total samples to unlearn: {len(unlearn_set)}")

        index_to_pos = {idx: pos for pos, idx in enumerate(all_indices)}
        shard_size   = len(all_indices) // args.shards

        for original_idx in unlearn_set:
            pos      = index_to_pos[original_idx]
            shard_id = min(pos // shard_size, args.shards - 1)

            start_pos_of_shard = shard_id * shard_size
            pos_in_shard       = pos - start_pos_of_shard

            this_shard_len = shard_size if shard_id < args.shards - 1\
                             else len(all_indices) - start_pos_of_shard
            slice_size = this_shard_len // args.slices
            slice_id   = min(pos_in_shard // slice_size, args.slices - 1)

            if shard_id not in affected_shards:
                affected_shards[shard_id] = slice_id
            else:
                affected_shards[shard_id] = min(affected_shards[shard_id], slice_id)

        if not affected_shards:
            print("No shards affected.")
            return
    else:
        for s in range(args.shards):
            affected_shards[s] = 0

    criterion = nn.CrossEntropyLoss()

                                       
                                                                               
                                                                                
    history = {
        'shard': [], 'slice': [],
        'train_forget_acc': [], 'train_forget_loss': [],
        'train_retain_acc': [], 'train_retain_loss': [],
        'test_retain_acc':  [], 'test_retain_loss':  [],
        'slice_time_s':     [],
    }

    total_start = time.time()

    for shard_id in sorted(affected_shards.keys()):
        start_slice = affected_shards[shard_id]
        print(f"\n[{'Unlearning' if args.unlearn else 'Training'} Shard {shard_id + 1}/{args.shards}]"
              f" Start from Slice {start_slice}")

        model     = get_model(args.model, in_channels, num_classes, img_size).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        if start_slice > 0:
            prev_ckpt = manager.get_model_path(shard_id, start_slice - 1)
            if os.path.exists(prev_ckpt):
                print(f"  > Loading checkpoint from Slice {start_slice - 1}")
                checkpoint = torch.load(prev_ckpt, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                if 'optimizer_state_dict' in checkpoint:
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            else:
                print(f"  > Warning: Checkpoint for Slice {start_slice-1} missing. "
                      f"Restarting shard from 0.")
                start_slice = 0

        for slice_id in range(start_slice, args.slices):
            print(f"  > Processing Slice {slice_id + 1}/{args.slices}")

            train_indices = get_cumulative_indices(
                all_indices, shard_id, slice_id, args.shards, args.slices, unlearn_set
            )

            if len(train_indices) == 0:
                print("    Skipping (Empty Slice)")
                continue

            subset = Subset(full_dataset, train_indices)
            loader = DataLoader(subset, batch_size=args.batch_size, shuffle=True, drop_last=True)

            st = time.time()
            acc, loss = train_one_slice(model, loader, criterion, optimizer,
                                        device, args.epochs_per_slice)
            et = time.time()

                                                
            print(f"    Done ({et-st:.1f}s) - Acc: {acc:.4f}, Loss: {loss:.4f}")

                                                                                
            save_path = manager.get_model_path(shard_id, slice_id)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                'shard':               shard_id,
                'slice':               slice_id,
                'model_state_dict':    model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'accuracy':            acc,
            }, save_path)

                                                              
            eval_res = evaluate_4_quadrant(model, train_set, test_set,
                                           forget_indices_tensor, device, verbose=False)
            _tf = eval_res.get("Train_Forget", {}).get("acc", float("nan"))
            _tr = eval_res.get("Test_Retain ", {}).get("acc", float("nan"))
            _tr_train = eval_res.get("Train_Retain", {}).get("acc", float("nan"))
            print(f"    Eval  → Train Retain: {_tr_train:6.2f}% | Test Retain: {_tr:6.2f}% | Forget: {_tf:6.2f}%")
            history['shard'].append(shard_id)
            history['slice'].append(slice_id)
            history['train_forget_acc'].append(eval_res.get('Train_Forget', {}).get('acc', float('nan')))
            history['train_forget_loss'].append(eval_res.get('Train_Forget', {}).get('loss', float('nan')))
            history['train_retain_acc'].append(eval_res.get('Train_Retain', {}).get('acc', float('nan')))
            history['train_retain_loss'].append(eval_res.get('Train_Retain', {}).get('loss', float('nan')))
            history['test_retain_acc'].append(eval_res.get('Test_Retain ', {}).get('acc', float('nan')))
            history['test_retain_loss'].append(eval_res.get('Test_Retain ', {}).get('loss', float('nan')))
            history['slice_time_s'].append(round(et - st, 2))

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - total_start
    print("\nSISA completed.")

                                                                      
    if args.unlearn:
        os.makedirs("history", exist_ok=True)
        forget_indices_path = os.path.join(
            "history",
            f"forget_indices_sisa_{args.dataset}_{args.model}_seed{args.seed}.pt"
        )
        torch.save(forget_indices_tensor, forget_indices_path)
        print(f"[Save] Forget indices saved to '{forget_indices_path}'")

                                                    
    os.makedirs("history", exist_ok=True)
    epoch_csv = os.path.join("history", f"sisa_{args.dataset}_{args.model}_seed{args.seed}.csv")
    with open(epoch_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    print(f"[Save] Epoch log saved to '{epoch_csv}'")

                                                                               
    if args.unlearn and len(unlearn_set) > 0:
        forget_indices_arr = np.array(list(unlearn_set), dtype=int)
        retain_indices_arr = np.setdiff1d(np.arange(len(full_dataset)), forget_indices_arr)
        retain_loader = DataLoader(Subset(full_dataset, retain_indices_arr), batch_size=args.batch_size, shuffle=False)
        forget_loader = DataLoader(Subset(full_dataset, forget_indices_arr), batch_size=args.batch_size, shuffle=False)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False)

        train_retain_acc, _ = evaluate_ensemble(
            manager, args.model, in_channels, num_classes, img_size,
            args.shards, args.slices, retain_loader, device
        )
        train_forget_acc, _ = evaluate_ensemble(
            manager, args.model, in_channels, num_classes, img_size,
            args.shards, args.slices, forget_loader, device
        )
        test_retain_acc, _ = evaluate_ensemble(
            manager, args.model, in_channels, num_classes, img_size,
            args.shards, args.slices, test_loader, device
        )

        best_shard = "ensemble"
        best_slice = args.slices - 1
        best_train_retain_acc = train_retain_acc * 100.0
        best_forget_acc = train_forget_acc * 100.0
        best_retain_acc = test_retain_acc * 100.0
    elif len(history['test_retain_acc']) > 0:
        best_idx  = int(np.argmax(history['test_retain_acc']))
        best_shard = history['shard'][best_idx]
        best_slice = history['slice'][best_idx]
        best_forget_acc = history['train_forget_acc'][best_idx]
        best_train_retain_acc = history['train_retain_acc'][best_idx]
        best_retain_acc = history['test_retain_acc'][best_idx]
    else:
        best_shard = best_slice = -1
        best_forget_acc = best_train_retain_acc = best_retain_acc = float('nan')

    summary_csv = os.path.join("history",
                               "summary_sisa.csv")
    summary_exists = os.path.exists(summary_csv)
    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow([
                'Dataset', 'Model', 'Seed', 'N_Forget',
                'Shards', 'Slices',
                'Best_Shard', 'Best_Slice',
                'Best_Train_Retain_Acc', 'Best_Train_Forget_Acc', 'Best_Test_Retain_Acc',
                'Total_Time_s'
            ])
        writer.writerow([
            args.dataset, args.model, args.seed, len(unlearn_set),
            args.shards, args.slices,
            best_shard, best_slice,
            f"{best_train_retain_acc:.4f}", f"{best_forget_acc:.4f}", f"{best_retain_acc:.4f}",
            f"{elapsed:.2f}"
        ])
    print(f"[Save] Best summary saved to '{summary_csv}'")
    print(f"\n[Best  Shard {best_shard} / Slice {best_slice}]  "
          f"Forget Acc: {best_forget_acc:.2f}%  "
          f"Train Retain: {best_train_retain_acc:.2f}%  "
          f"Test Retain: {best_retain_acc:.2f}%  "
          f"Total Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
