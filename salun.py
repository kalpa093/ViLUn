"""
SalUn: Saliency-Based Unlearning
- Compatible with vilun.py experimental environment
- Datasets : CIFAR10, CIFAR100, MNIST, Yale B
- Models   : CNN, RNN, MLP, ResNet18, ViT
- Save     : salun_{dataset}_{model}_{seed}.pth
             history/salun_{dataset}_{model}_{seed}_epoch_log.csv
             history/salun_{dataset}_{model}_{seed}_best_summary.csv
"""

import argparse
import os
import copy
import time
import csv
import random
import itertools
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset, TensorDataset
import glob
from PIL import Image

                                                    
class SaliencyPruner:
    def __init__(self, model, device='cuda'):
        self.model  = model
        self.device = device

    def compute_gradients(self, dataloader, criterion):
        self.model.eval()
        self.model.zero_grad()

        accumulated_grads = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                accumulated_grads[name] = torch.zeros_like(param.data)

        print("[SalUn] Computing gradients for saliency map...")

        for batch in dataloader:
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs, targets = inputs.to(self.device), targets.to(self.device)

            outputs = self.model(inputs)
            loss    = criterion(outputs, targets)
            loss.backward()

            with torch.no_grad():
                for name, param in self.model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        accumulated_grads[name] += param.grad.abs()

            self.model.zero_grad()

                                        
        return accumulated_grads

    def generate_mask(self, accumulated_grads, ratio=0.1):
        all_grads = []
        for name, grad in accumulated_grads.items():
            all_grads.append(grad.view(-1))

        all_grads = torch.cat(all_grads)
        num_params = all_grads.numel()
        num_keep   = max(int(num_params * ratio), 1)

        threshold, _ = torch.topk(all_grads, num_keep)
        cutoff = threshold[-1]

        print(f"[SalUn] Mask generation: keeping top {ratio*100:.1f}% params "
              f"(Threshold: {cutoff:.6f})")

        mask_dict = {}
        for name, grad in accumulated_grads.items():
            mask_dict[name] = (grad >= cutoff).float().to(self.device)

        return mask_dict


def save_checkpoint(state, filename='checkpoint.pth'):
    torch.save(state, filename)


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = self.avg = self.sum = self.count = 0

    def update(self, val, n=1):
        self.val   = val
        self.sum  += val * n
        self.count += n
        self.avg   = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes precision@k"""
    with torch.no_grad():
        maxk      = max(topk)
        batch_size = target.size(0)
        _, pred   = output.topk(maxk, 1, True, True)
        pred      = pred.t()
        correct   = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


                                            
                                 
                                            
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True


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

    full_dataset   = TensorDataset(torch.stack(X_list), torch.stack(y_list))
    unique_labels  = torch.unique(torch.stack(y_list))
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
    if model_name == 'cnn':      return CNNNetwork(in_channels, num_classes)
    elif model_name == 'resnet18': return ResNet18Network(in_channels, num_classes)
    elif model_name == 'resnet50': return ResNet50Network(in_channels, num_classes)
    elif model_name == 'lenet':   return LeNetNetwork(in_channels, num_classes)
    elif model_name == 'vit':     return ViTNetwork(in_channels, num_classes)
    elif model_name == 'mlp':     return MLPNetwork(in_channels, num_classes, img_size)
    elif model_name == 'rnn':     return RNNNetwork(in_channels, num_classes, img_size)
    else: raise ValueError(f"Unknown model: {model_name}")


                                            
                                 
                                            
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


def set_batchnorm_eval(model):
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module.eval()


def random_labels_excluding(targets, num_classes):
    if num_classes <= 1:
        return targets.clone()
    offsets = torch.randint(1, num_classes, targets.shape, device=targets.device)
    return (targets + offsets) % num_classes


                                            
                               
                                            
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
            acc = 100.0 * correct / total
            avg_loss = loss_sum / total
            results[name] = {'acc': acc, 'loss': avg_loss}
            if verbose:
                print(f"  {name:<15} | {acc:6.2f}     | {avg_loss:.4f}")

    if verbose:
        print("  " + "-" * 40)
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
        'mnist':    'resnet18',
        'yale':     'mlp',
    }
    if args.model == 'resnet18':                                       
        return default_map.get(args.dataset, 'resnet18')
    return args.model                                                      

def main():
    parser = argparse.ArgumentParser(description="SalUn: Saliency Unlearning Framework")
    parser.add_argument("--seed",        type=int,   default=42)
    parser.add_argument("--dataset",     type=str,   default="cifar10",
                        choices=["cifar10", "cifar100", "mnist", "yale"])
    parser.add_argument("--data_dir",    type=str,   default="./data")
    parser.add_argument("--model",       type=str,   default="resnet18",
                        choices=["cnn", "resnet18", "resnet50", "lenet", "vit", "mlp", "rnn"])
    parser.add_argument("--load_path",   type=str,   default=None,
                        help="Path to the pretrained model (original model)")
    parser.add_argument("--orig_lr",      type=float, default=1e-3,
                        help="LR for original model pretraining")
    parser.add_argument("--save_dir",    type=str,   default="./salun_models")
    parser.add_argument("--unlearn",     type=str,   default="RL", choices=["RL", "GA"],
                        help="RL: Random Label, GA: Gradient Ascent")
    parser.add_argument("--target_id",   type=int,   default=0,
                        help="Yale only: subject index to unlearn (0-14). Ignored for CIFAR/MNIST.")
    parser.add_argument("--unlearn_epochs", type=int,   default=10)
    parser.add_argument("--unlearn_lr",     type=float, default=1e-2)
    parser.add_argument("--batch_size",     type=int,   default=32)
    parser.add_argument("--mask_ratio",     type=float, default=0.01,
                        help="Ratio of parameters to update (sparsity of mask)")
    parser.add_argument("--retain_alpha",   type=float, default=0.0,
                        help="Optional retain CE weight for sample-level SalUn stabilization")
    parser.add_argument("--freeze_bn",      action="store_true",
                        help="Freeze BatchNorm running statistics during SalUn updates")
    parser.add_argument("--momentum",       type=float, default=0.9)
    parser.add_argument("--weight_decay",   type=float, default=5e-4)
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

                                                                  
                                                                     
    _fi_path = os.path.join("history",
                            f"forget_indices_salun_{args.dataset}_{args.model}_seed{args.seed}.pt")
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
    retain_indices = torch.tensor([i for i in range(len(train_set)) if i not in forget_idx_set])

    print(f"Forget Set Size: {len(forget_indices)}, Retain Set Size: {len(retain_indices)}")

    forget_set = Subset(train_set, forget_indices)
    retain_set = Subset(train_set, retain_indices)

    forget_loader = DataLoader(forget_set, batch_size=args.batch_size, shuffle=True, num_workers=4)

                                 
    model = get_model(args.model, in_channels, num_classes, img_size).to(device)

    train_full_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    orig_model_path   = os.path.join(args.save_dir,
                                     f"original_{args.dataset}_{args.model}_seed{args.seed}.pth")
    time_orig_train   = 0.0

    if args.load_path and os.path.exists(args.load_path):
        print(f"[SalUn] Loading pretrained model from {args.load_path}")
        checkpoint = torch.load(args.load_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    elif os.path.exists(orig_model_path):
        print(f"[SalUn] Loading cached original model from {orig_model_path}")
        model.load_state_dict(torch.load(orig_model_path, map_location=device))
    else:
        print(f"[SalUn] No pretrained model found. Training original model from scratch...")
        t0 = time.time()
        model = train_standard(model, train_full_loader, device,
                               epochs=200, lr=1e-3, patience=50)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_orig_train = time.time() - t0
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(model.state_dict(), orig_model_path)
        print(f"  [Time] Original training: {time_orig_train:.2f}s → saved to {orig_model_path}")

                                                                             
    print("\n[Salun] Generating Saliency Mask")
    criterion = nn.CrossEntropyLoss()
    pruner    = SaliencyPruner(model, device)

    grads = pruner.compute_gradients(forget_loader, criterion)
    mask  = pruner.generate_mask(grads, ratio=args.mask_ratio)

    print(f"\n[SalUn] Start Unlearning ({args.unlearn})")

    optimizer = optim.SGD(model.parameters(), lr=args.unlearn_lr,
                          momentum=args.momentum, weight_decay=args.weight_decay)
    model.train()
    if args.freeze_bn:
        set_batchnorm_eval(model)
    retain_iter = itertools.cycle(retain_loader) if args.retain_alpha > 0 else None

    history = {
        'epoch': [],
        'train_forget_acc': [], 'train_forget_loss': [],
        'train_retain_acc': [], 'train_retain_loss': [],
        'test_retain_acc':  [], 'test_retain_loss':  [],
    }

    start_time = time.time()

    for epoch in range(args.unlearn_epochs):
                                                    
                                                                  
        model.train()
        if args.freeze_bn:
            set_batchnorm_eval(model)
        losses = AverageMeter()
        top1   = AverageMeter()
        epoch_start = time.time()

        for batch in forget_loader:
            if len(batch) == 3:
                inputs, targets, _ = batch
            else:
                inputs, targets = batch

            inputs, targets = inputs.to(device), targets.to(device)

                                                          
            if args.unlearn == "RL":
                random_targets = random_labels_excluding(targets, num_classes)
                outputs = model(inputs)
                loss    = criterion(outputs, random_targets)
            elif args.unlearn == "GA":
                outputs = model(inputs)
                loss    = -criterion(outputs, targets)

            if retain_iter is not None:
                retain_batch = next(retain_iter)
                if len(retain_batch) == 3:
                    retain_inputs, retain_targets, _ = retain_batch
                else:
                    retain_inputs, retain_targets = retain_batch
                retain_inputs = retain_inputs.to(device)
                retain_targets = retain_targets.to(device)
                retain_outputs = model(retain_inputs)
                loss = loss + args.retain_alpha * criterion(retain_outputs, retain_targets)

            optimizer.zero_grad()
            loss.backward()

            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in mask and param.grad is not None:
                        param.grad.mul_(mask[name])

            optimizer.step()

            acc1 = accuracy(outputs, targets, topk=(1,))                       
            losses.update(loss.item(), inputs.size(0))
            top1.update(acc1[0].item(), inputs.size(0))

                                            
        print(f"Epoch [{epoch+1}/{args.unlearn_epochs}] "
              f"Time: {time.time()-epoch_start:.2f}s | "
              f"Loss: {losses.avg:.4f} | "
              f"Acc: {top1.avg:.2f}%")

                                                   
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

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    elapsed = time.time() - start_time

                                      
    os.makedirs(args.save_dir, exist_ok=True)
    save_name = f"salun_{args.dataset}_{args.model}_seed{args.seed}.pth"
    save_path = os.path.join(args.save_dir, save_name)
    save_checkpoint({
        'epoch':       args.unlearn_epochs,
        'model_state_dict': model.state_dict(),
        'state_dict':  model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'optimizer':   optimizer.state_dict(),
        'mask':        mask,
    }, filename=save_path)
    print(f"\nSalUn Completed. Model saved to {save_path}")

                                                                 
    os.makedirs("history", exist_ok=True)
    forget_indices_path = os.path.join(
        "history",
        f"forget_indices_salun_{args.dataset}_{args.model}_seed{args.seed}.pt"
    )
    torch.save(forget_indices, forget_indices_path)
    print(f"[Save] Forget indices saved to '{forget_indices_path}'")

                                              
    os.makedirs("history", exist_ok=True)
    epoch_csv = os.path.join("history", f"salun_{args.dataset}_{args.model}_seed{args.seed}.csv")
    with open(epoch_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    print(f"[Save] Epoch log saved to '{epoch_csv}'")

                                              
    best_idx  = int(np.argmax(history['test_retain_acc']))
    best_epoch      = history['epoch'][best_idx]
    best_forget_acc = history['train_forget_acc'][best_idx]
    best_train_retain_acc = history['train_retain_acc'][best_idx]
    best_retain_acc = history['test_retain_acc'][best_idx]

    summary_csv = os.path.join("history", "summary_salun.csv")
    summary_exists = os.path.exists(summary_csv)
    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow([
                'Dataset', 'Model', 'Seed', 'N_Forget', 'Unlearn_Mode',
                'Best_Epoch', 'Best_Train_Retain_Acc', 'Best_Train_Forget_Acc', 'Best_Test_Retain_Acc',
                'Total_Time_s'
            ])
        writer.writerow([
            args.dataset, args.model, args.seed, len(forget_indices), args.unlearn,
            best_epoch,
            f"{best_train_retain_acc:.4f}", f"{history['train_forget_acc'][best_idx]:.4f}", f"{best_retain_acc:.4f}",
            f"{elapsed:.2f}"
        ])
    print(f"[Save] Best summary saved to '{summary_csv}'")
    print(f"\n[Best Epoch {best_epoch}]  Forget Acc: {best_forget_acc:.2f}%  "
          f"Train Retain: {best_train_retain_acc:.2f}%  Test Retain: {best_retain_acc:.2f}%  "
          f"Total Time: {elapsed:.2f}s")


if __name__ == "__main__":
    main()
