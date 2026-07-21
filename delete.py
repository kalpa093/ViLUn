import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
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
        self.best_epoch = 0
        self.best_weights = None
        self.early_stop = False

    def __call__(self, score, model, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
            self.best_weights = copy.deepcopy(model.state_dict())
        elif (self.mode == 'min' and score >= self.best_score) or\
             (self.mode == 'max' and score <= self.best_score):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_epoch = epoch
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
        x = self.features(x)
        return self.classifier(torch.flatten(x, 1))

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
        x = self.features(x); x = self.pool(x); x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x)); x = F.relu(self.fc2(x))
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
        return self.model(x)

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
        self.lstm = nn.LSTM(
            input_size=in_channels * img_size, hidden_size=128,
            num_layers=2, batch_first=True, bidirectional=True
        )
        self.classifier = nn.Linear(256, num_classes)
    def forward(self, x):
        B, C, H, W = x.size()
        if H != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size))
            B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H, W * C)
        _, (h_n, _) = self.lstm(x)
        feature = torch.cat((h_n[-2], h_n[-1]), dim=1)
        return self.classifier(feature)

def get_model(model_name, in_channels, num_classes, img_size):
    if model_name == 'cnn':      return CNNNetwork(in_channels, num_classes)
    elif model_name == 'resnet18': return ResNet18Network(in_channels, num_classes)
    elif model_name == 'resnet50': return ResNet50Network(in_channels, num_classes)
    elif model_name == 'lenet':  return LeNetNetwork(in_channels, num_classes)
    elif model_name == 'vit':    return ViTNetwork(in_channels, num_classes)
    elif model_name == 'mlp':    return MLPNetwork(in_channels, num_classes, img_size)
    elif model_name == 'rnn':    return RNNNetwork(in_channels, num_classes, img_size)
    else: raise ValueError(f"Unknown model: {model_name}")

                                            
            
                                            
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
        early_stopper(epoch_loss, model, epoch)
        if early_stopper.early_stop:
            print(f"    [Early Stop] Training stopped at epoch {epoch+1}.")
            break

    if early_stopper.best_weights is not None:
        model.load_state_dict(early_stopper.best_weights)
    return model

                                            
               
                                            
def build_delete_soft_targets(teacher_logits, true_labels):
    """
    Build DELETE-style masked soft targets from a frozen teacher.

    Sample-target adaptation:
    for each forget sample, mask out its own true-class logit in the teacher
    output, then normalize the remaining probabilities.

    Args:
        teacher_logits: (B, C) frozen original-model logits
        true_labels:    (B,)   original class labels of forget samples
    Returns:
        soft_target: (B, C)
    """
    masked_teacher_logits = teacher_logits.clone()
    masked_teacher_logits[torch.arange(teacher_logits.size(0), device=teacher_logits.device), true_labels] = float("-inf")
    return F.softmax(masked_teacher_logits, dim=1)


def delete_loss(student_logits, teacher_logits, true_labels):
    """
    DELETE-style sample-target loss using frozen-teacher masked soft targets.
    """
    soft_target = build_delete_soft_targets(teacher_logits, true_labels)
    log_prob = F.log_softmax(student_logits, dim=1)
    loss = F.kl_div(log_prob, soft_target.detach(), reduction='batchmean')
    return loss

                                            
              
                                            
def evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose=True):
    """
    Sample-target evaluation.
    forget_indices: array/tensor of train-set indices forming the forget set.
    """
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
            if loader is None: continue
            correct, total, loss_sum = 0, 0, 0.0
            for data, target in loader:
                data, target = data.to(device), target.to(device)
                out = model(data)
                loss_sum += F.cross_entropy(out, target, reduction='sum').item()
                pred = out.argmax(dim=1)
                correct += (pred == target).sum().item()
                total   += target.size(0)
            acc = 100.0 * correct / total
            avg_loss = loss_sum / total
            results[name] = {'acc': acc, 'loss': avg_loss}
            if verbose:
                print(f"  {name:<15} | {acc:6.2f}     | {avg_loss:.4f}")

    if verbose: print("  " + "-" * 40)
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

def run_pipeline(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.model = resolve_model(args)
    print(f"=== DELETE Baseline Unlearning ===")
    print(f"  Dataset: {args.dataset} | Model: {args.model} | Seed: {args.seed}")

                                            
    train_set, test_set, num_classes, in_channels = load_dataset_factory(args)
    img_size = 64 if args.dataset == 'yale' else 32

                                                                  
                                                                     
    _fi_path = os.path.join("history",
                            f"forget_indices_delete_{args.dataset}_{args.model}_seed{args.seed}.pt")
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

    forget_idx_set = set(forget_indices.tolist())
    forget_loader     = DataLoader(Subset(train_set, forget_indices), batch_size=64, shuffle=True)
    train_full_loader = DataLoader(train_set, batch_size=64, shuffle=True)

    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs("history", exist_ok=True)

                                            
    print(f"\n[Step 1] Original Model...")
    model = get_model(args.model, in_channels, num_classes, img_size).to(device)
    time_orig_train = 0.0

    if args.original is not None and os.path.exists(args.original):
        print(f"  >> Loading from: {args.original}")
        model.load_state_dict(torch.load(args.original, map_location=device))
    else:
        print(f"  >> Training original model ({args.model.upper()})...")
        t0 = time.time()
        model = train_standard(model, train_full_loader, device,
                               epochs=args.train_epochs, lr=args.orig_lr, patience=args.patience)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        time_orig_train = time.time() - t0
        orig_path = os.path.join(args.save_path,
                                 f"original_{args.dataset}_{args.model}_seed{args.seed}.pth")
        torch.save(model.state_dict(), orig_path)
        print(f"  [Time] Original training: {time_orig_train:.2f}s")

    print("\n[Baseline] Original model performance:")
    evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose=True)

                                                      
    frozen_teacher = copy.deepcopy(model).to(device)
    frozen_teacher.eval()
    for param in frozen_teacher.parameters():
        param.requires_grad = False

                                            
                                                                        
    print(f"\n[Step 2] DELETE Unlearning (frozen-teacher masked distillation on forget data only)...")
    print(f"  lr={args.unlearn_lr}, epochs={args.unlearn_epochs}, patience={args.patience}")

    optimizer    = optim.Adam(model.parameters(), lr=args.unlearn_lr)
    history = {
        'epoch': [], 'delete_loss': [],
        'train_forget_acc': [], 'train_forget_loss': [],
        'train_retain_acc': [], 'train_retain_loss': [],
        'test_retain_acc':  [], 'test_retain_loss':  []
    }

    t0 = time.time()
    for epoch in range(args.unlearn_epochs):
        model.train()
        epoch_loss = 0.0
        for data_f, labels_f in forget_loader:
            data_f, labels_f = data_f.to(device), labels_f.to(device)
            with torch.no_grad():
                teacher_logits = frozen_teacher(data_f)
            student_logits = model(data_f)
            loss = delete_loss(student_logits, teacher_logits, labels_f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(forget_loader)
        eval_res = evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose=False)
        t_f_acc  = eval_res.get("Train_Forget", {}).get("acc", float("nan"))
        t_r_acc  = eval_res["Test_Retain "]["acc"]
        tr_r_acc = eval_res.get('Train_Retain', {}).get('acc', float('nan'))
        print(f"  Epoch [{epoch+1:>4}/{args.unlearn_epochs}] | D-Loss: {avg_loss:.4f} | Train Retain: {tr_r_acc:6.2f}% | Test Retain: {t_r_acc:6.2f}% | Forget: {t_f_acc:6.2f}%")

        history['epoch'].append(epoch + 1)
        history['delete_loss'].append(avg_loss)
        history['train_forget_acc'].append(eval_res.get('Train_Forget', {}).get('acc', float('nan')))
        history['train_forget_loss'].append(eval_res.get('Train_Forget', {}).get('loss', float('nan')))
        history['train_retain_acc'].append(eval_res.get('Train_Retain', {}).get('acc', float('nan')))
        history['train_retain_loss'].append(eval_res.get('Train_Retain', {}).get('loss', float('nan')))
        history['test_retain_acc'].append(eval_res['Test_Retain ']['acc'])
        history['test_retain_loss'].append(eval_res['Test_Retain ']['loss'])
        t_f_acc = eval_res.get('Train_Forget', {}).get('acc', float('nan'))
    if torch.cuda.is_available(): torch.cuda.synchronize()
    time_unlearn = time.time() - t0
    print(f"  [Time] Unlearning: {time_unlearn:.2f}s")

                                            
    print("\n[Final] Unlearned model performance:")
    final_res = evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose=True)

                                            
    unlearned_path = os.path.join(
        args.save_path,
        f"delete_{args.dataset}_{args.model}_seed{args.seed}.pth"
    )
    torch.save(model.state_dict(), unlearned_path)
    print(f"\n[Save] Unlearned model → '{unlearned_path}'")

                                                                 
    forget_indices_path = os.path.join(
        "history",
        f"forget_indices_delete_{args.dataset}_{args.model}_seed{args.seed}.pt"
    )
    torch.save(forget_indices, forget_indices_path)
    print(f"[Save] Forget indices saved to '{forget_indices_path}'")

                                            
    hist_csv = os.path.join("history",
                            f"delete_{args.dataset}_{args.model}_seed{args.seed}.csv")
    with open(hist_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    print(f"[Save] Epoch history → '{hist_csv}'")

                                           
    summary_csv = os.path.join("history", "summary_delete.csv")
    summary_exists = os.path.exists(summary_csv)
    with open(summary_csv, mode='a', newline='') as f:
        writer = csv.writer(f)
        if not summary_exists:
            writer.writerow([
                'Method', 'Dataset', 'Model', 'N_Forget', 'Seed',
                'Unlearn_LR', 'Final_Epoch',
                'Final_Train_Retain_Acc', 'Final_Test_Retain_Acc', 'Final_TrainForget_Acc',
                'Time_Orig_Train', 'Time_Unlearn'
            ])
        writer.writerow([
            'DELETE',
            args.dataset, args.model, len(forget_indices), args.seed,
            args.unlearn_lr,
            history['epoch'][-1],
            round(final_res.get('Train_Retain', {}).get('acc', float('nan')), 4),
            round(final_res['Test_Retain ']['acc'], 4),
            round(final_res.get('Train_Forget', {}).get('acc', float('nan')), 4),
            round(time_orig_train, 2),
            round(time_unlearn, 2)
        ])
    print(f"[Save] Summary appended → '{summary_csv}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DELETE Baseline Unlearning (Zhou et al., CVPR 2025)")

          
    parser.add_argument("--dataset",    type=str, default="cifar10",
                        choices=["yale", "mnist", "cifar10", "cifar100", "tinyimagenet"])
    parser.add_argument("--data_dir",   type=str, default="./data")
    parser.add_argument("--target_id",  type=int, default=0,
                        help="Yale only: subject index to unlearn (0-14). Ignored for CIFAR/MNIST.")

           
    parser.add_argument("--model",      type=str, default="resnet18",
                        choices=["cnn", "resnet18", "resnet50", "lenet", "vit", "mlp", "rnn"])
    parser.add_argument("--original",   type=str, default=None,
                        help="Path to pre-trained original model (.pth)")
    parser.add_argument("--save_path",  type=str, default="./saved_models")

              
    parser.add_argument("--train_epochs",  type=int,   default=200)
    parser.add_argument("--orig_lr",       type=float, default=1e-3)

                
    parser.add_argument("--unlearn_epochs", type=int,   default=200,
                        help="Max unlearning epochs")
    parser.add_argument("--unlearn_lr",     type=float, default=1e-5,
                        help="Learning rate for DELETE unlearning step")
    parser.add_argument("--patience",       type=int,   default=50,
                        help="Early stopping patience for unlearning")

          
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    run_pipeline(args)
