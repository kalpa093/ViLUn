"""
ViLUN for LLM — CIFAR10 Image Classification
=============================================
기존 vilun.py 구조를 그대로 유지하되,
original model만 LLM(ViT 방식 image encoder + classification head)으로 교체.

변경점:
  original model : CNN/ResNet → LLMVisionClassifier (Qwen2.5-7B / Llama-3.2-3B)
                                이미지를 patch embed → LLM encoder → CLS head
  expert  model  : 기존 소형 모델(CNN/LeNet/ResNet18/MLP/RNN) + Qwen2.5-1.5B
  나머지 전체    : 기존 vilun.py와 동일

평가:
  - 기존 4-quadrant (Acc + Loss) : 동일
  - Forget/Retain Perplexity 추가: image→patch token으로 변환 후 LLM loss 계산

실행 예시:
  # Standard ViLUN
  python vilun_llm.py \\
      --mode standard \\
      --orig_model Qwen/Qwen2.5-7B \\
      --expert_model cnn \\
      --device_orig cuda:0 --device_expert cuda:1

  # Held-out ViLUN
  python vilun_llm.py \\
      --mode heldout \\
      --orig_model meta-llama/Llama-3.2-3B \\
      --expert_model qwen1.5b \\
      --device_orig cuda:0 --device_expert cuda:2
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms
from transformers import AutoModel, AutoConfig
import numpy as np
import os
import random
import argparse
import csv
import copy
import time

                                            
         
                                            
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

class EarlyStopping:
    def __init__(self, patience=5, mode='min'):
        self.patience   = patience
        self.mode       = mode
        self.counter    = 0
        self.best_score = None
        self.best_state = None                               
        self.early_stop = False

    def __call__(self, score, model):
        improved = (self.best_score is None) or\
                   (self.mode == 'min' and score < self.best_score) or\
                   (self.mode == 'max' and score > self.best_score)
        if improved:
            self.best_score = score
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter    = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

                                            
                                  
                                            
def load_cifar10(data_dir='./data', img_size=32):
                                                        
    transform_llm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
                          
    transform_small = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    train_llm   = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=transform_llm)
    test_llm    = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_llm)
    train_small = torchvision.datasets.CIFAR10(root=data_dir, train=True,  download=True, transform=transform_small)
    test_small  = torchvision.datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform_small)
    return train_llm, test_llm, train_small, test_small

def sanitize_tag(text):
    return str(text).replace("/", "_").replace("\\", "_").replace(":", "_")


def build_loader(dataset, indices, batch_size=16, shuffle=True):
    return DataLoader(Subset(dataset, list(indices)), batch_size=batch_size, shuffle=shuffle)


def get_or_create_forget_indices(args, train_set):
    os.makedirs(args.history_dir, exist_ok=True)
    orig_tag = sanitize_tag(args.orig_model.split("/")[-1] if "/" in args.orig_model else args.orig_model)
    path = os.path.join(args.history_dir, f"forget_indices_vilun_llm_cifar10_{orig_tag}_{args.expert_model}_seed{args.seed}.pt")
    if os.path.exists(path):
        forget_idx = torch.load(path, map_location="cpu")
        print(f"[Load] Forget indices -> {path} ({len(forget_idx)} samples)")
        return forget_idx.long(), path

    random.seed(args.seed)
    n_total = len(train_set)
    n_forget = max(1, int(0.1 * n_total))
    forget_idx = torch.tensor(sorted(random.sample(range(n_total), n_forget)), dtype=torch.long)
    torch.save(forget_idx, path)
    print(f"[Save] Forget indices -> {path} ({len(forget_idx)}/{n_total} samples)")
    return forget_idx, path


def build_heldout_loader(test_set, heldout_ratio, batch_size=16):
    n_total = len(test_set)
    n_heldout = max(1, int(n_total * heldout_ratio))
    heldout_idx = random.sample(range(n_total), n_heldout)
    heldout_loader = DataLoader(Subset(test_set, heldout_idx), batch_size=batch_size, shuffle=True)
    print(f"  [Held-out] {n_heldout}/{n_total} test samples selected (ratio={heldout_ratio})")
    return heldout_loader

                                            
                         
                                                           
                                                                    
                                            
class LLMVisionClassifier(nn.Module):
    """
    HuggingFace LLM backbone + patch embedding + classification head.

    이미지 처리 흐름:
      (B, 3, 224, 224)
        → PatchEmbed: (B, num_patches, hidden_size)
        → LLM encoder (일부 레이어만 사용, 메모리 절감)
        → CLS pooling: (B, hidden_size)
        → Linear head: (B, num_classes)

    메모리 절감 전략:
      - gradient checkpointing 활성화
      - LLM 레이어 수를 --llm_layers로 제한 (기본 8)
      - bf16으로 로드
    """
    def __init__(self, model_name, num_classes=10, patch_size=16, img_size=224,
                 num_layers=8, device='cuda:0'):
        super().__init__()
        self.device_llm = device

        print(f"    Loading LLM backbone: {model_name}")
        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_hidden_layers = num_layers             

        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        self.backbone.gradient_checkpointing_enable()

        hidden_size = config.hidden_size
        self.feat_dim = hidden_size

                                                   
        num_patches = (img_size // patch_size) ** 2
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),                                     
        ).to(device=device, dtype=torch.bfloat16)

                             
        self.head = nn.Linear(hidden_size, num_classes).to(device=device, dtype=torch.bfloat16)

    def forward(self, x, return_features=False):
                             
        x = x.to(device=self.device_llm, dtype=torch.bfloat16)

                                                         
        patches = self.patch_embed(x)                                          
        patches = patches.permute(0, 2, 1)                                     

                     
        out     = self.backbone(inputs_embeds=patches)
        hidden  = out.last_hidden_state                                        

                                     
        feature = hidden.mean(dim=1)                              
        logits  = self.head(feature)                              

        if return_features:
            return logits, feature
        return logits

    def parameters(self, recurse=True):
        return list(self.backbone.parameters()) +\
               list(self.patch_embed.parameters()) +\
               list(self.head.parameters())

    def state_dict(self, **kwargs):
                                                                 
        sd = {}
        for k, v in self.backbone.state_dict().items():
            sd[f'backbone.{k}'] = v
        for k, v in self.patch_embed.state_dict().items():
            sd[f'patch_embed.{k}'] = v
        for k, v in self.head.state_dict().items():
            sd[f'head.{k}'] = v
        return sd

    def load_state_dict(self, state_dict, strict=True):
        backbone_sd    = {k[len('backbone.'):]:    v for k, v in state_dict.items() if k.startswith('backbone.')}
        patch_embed_sd = {k[len('patch_embed.'):]: v for k, v in state_dict.items() if k.startswith('patch_embed.')}
        head_sd        = {k[len('head.'):]:        v for k, v in state_dict.items() if k.startswith('head.')}
        self.backbone.load_state_dict(backbone_sd, strict=strict)
        self.patch_embed.load_state_dict(patch_embed_sd, strict=strict)
        self.head.load_state_dict(head_sd, strict=strict)

    def train(self, mode=True):
        self.backbone.train(mode)
        self.patch_embed.train(mode)
        self.head.train(mode)
        return self

    def eval(self):
        self.backbone.eval()
        self.patch_embed.eval()
        self.head.eval()
        return self

                                            
                                      
                                            
class CNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Linear(64, num_classes)
        self.feat_dim   = 64

    def forward(self, x, return_features=False):
        feat   = torch.flatten(self.features(x), 1)
        logits = self.classifier(feat)
        if return_features: return logits, feat
        return logits

class LeNetNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, 5, padding=2), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(6, 16, 5), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.pool = nn.AdaptiveAvgPool2d((5, 5))
        self.fc1  = nn.Linear(16*5*5, 120)
        self.fc2  = nn.Linear(120, 84)
        self.classifier = nn.Linear(84, num_classes)
        self.feat_dim   = 84

    def forward(self, x, return_features=False):
        x   = self.pool(self.features(x))
        x   = torch.flatten(x, 1)
        x   = F.relu(self.fc1(x))
        feat = F.relu(self.fc2(x))
        logits = self.classifier(feat)
        if return_features: return logits, feat
        return logits

class ResNet18Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet18(weights=None)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(512, num_classes)
        self.feat_dim = 512

    def forward(self, x, return_features=False):
        x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x); x = self.model.maxpool(x)
        x = self.model.layer1(x); x = self.model.layer2(x); x = self.model.layer3(x); x = self.model.layer4(x)
        x = self.model.avgpool(x); feat = torch.flatten(x, 1); logits = self.model.fc(feat)
        if return_features: return logits, feat
        return logits

class ResNet50Network(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.model = torchvision.models.resnet50(weights=None)
        if in_channels != 3:
            self.model.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3, bias=False)
        self.model.fc = nn.Linear(2048, num_classes)
        self.feat_dim = 2048

    def forward(self, x, return_features=False):
        x = self.model.conv1(x); x = self.model.bn1(x); x = self.model.relu(x); x = self.model.maxpool(x)
        x = self.model.layer1(x); x = self.model.layer2(x); x = self.model.layer3(x); x = self.model.layer4(x)
        x = self.model.avgpool(x); feat = torch.flatten(x, 1); logits = self.model.fc(feat)
        if return_features: return logits, feat
        return logits

class MLPNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        input_dim = in_channels * img_size * img_size
        self.mixer = nn.Sequential(
            nn.Linear(input_dim, 512), nn.GELU(),
            nn.Linear(512, 512),       nn.GELU(),
            nn.Linear(512, 128),       nn.GELU()
        )
        self.classifier = nn.Linear(128, num_classes)
        self.feat_dim   = 128

    def forward(self, x, return_features=False):
        x = torch.flatten(x, 1)
        feat = self.mixer(x); logits = self.classifier(feat)
        if return_features: return logits, feat
        return logits

class RNNNetwork(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, img_size=32):
        super().__init__()
        self.img_size = img_size
        self.lstm = nn.LSTM(in_channels * img_size, 128, num_layers=2,
                            batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(256, num_classes)
        self.feat_dim   = 256

    def forward(self, x, return_features=False):
        B, C, H, W = x.size()
        x = x.permute(0, 2, 3, 1).reshape(B, H, W * C)
        _, (h_n, _) = self.lstm(x)
        feat = torch.cat([h_n[-2], h_n[-1]], dim=1)
        logits = self.classifier(feat)
        if return_features: return logits, feat
        return logits

class LLMExpertClassifier(nn.Module):
    """소형 LLM expert (Qwen2.5-1.5B): 동일한 이미지 분류 구조."""
    def __init__(self, model_name='Qwen/Qwen2.5-1.5B', num_classes=10,
                 patch_size=16, img_size=224, num_layers=4, device='cuda:1'):
        super().__init__()
        self.device_llm = device

        config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
        config.num_hidden_layers = num_layers

        self.backbone = AutoModel.from_pretrained(
            model_name,
            config=config,
            torch_dtype=torch.bfloat16,
            device_map=device,
            trust_remote_code=True,
            ignore_mismatched_sizes=True,
        )
        hidden_size   = config.hidden_size
        self.feat_dim = hidden_size

        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size),
            nn.Flatten(2),
        ).to(device=device, dtype=torch.bfloat16)

        self.head = nn.Linear(hidden_size, num_classes).to(device=device, dtype=torch.bfloat16)

    def forward(self, x, return_features=False):
        x       = x.to(device=self.device_llm, dtype=torch.bfloat16)
        patches = self.patch_embed(x).permute(0, 2, 1)
        out     = self.backbone(inputs_embeds=patches)
        feat    = out.last_hidden_state.mean(dim=1)
        logits  = self.head(feat)
        if return_features: return logits, feat
        return logits

    def parameters(self, recurse=True):
        return list(self.backbone.parameters()) +\
               list(self.patch_embed.parameters()) +\
               list(self.head.parameters())

    def state_dict(self, **kwargs):
        sd = {}
        for k, v in self.backbone.state_dict().items():
            sd[f'backbone.{k}'] = v
        for k, v in self.patch_embed.state_dict().items():
            sd[f'patch_embed.{k}'] = v
        for k, v in self.head.state_dict().items():
            sd[f'head.{k}'] = v
        return sd

    def load_state_dict(self, state_dict, strict=True):
        backbone_sd    = {k[len('backbone.'):]:    v for k, v in state_dict.items() if k.startswith('backbone.')}
        patch_embed_sd = {k[len('patch_embed.'):]: v for k, v in state_dict.items() if k.startswith('patch_embed.')}
        head_sd        = {k[len('head.'):]:        v for k, v in state_dict.items() if k.startswith('head.')}
        self.backbone.load_state_dict(backbone_sd, strict=strict)
        self.patch_embed.load_state_dict(patch_embed_sd, strict=strict)
        self.head.load_state_dict(head_sd, strict=strict)

    def train(self, mode=True):
        self.backbone.train(mode); self.patch_embed.train(mode); self.head.train(mode)
        return self

    def eval(self):
        self.backbone.eval(); self.patch_embed.eval(); self.head.eval()
        return self


def get_expert_model(expert_name, num_classes, device_expert, img_size=32):
    if   expert_name == 'cnn':      m = CNNNetwork(3, num_classes);             dim = 64
    elif expert_name == 'lenet':    m = LeNetNetwork(3, num_classes);           dim = 84
    elif expert_name == 'resnet18': m = ResNet18Network(3, num_classes);        dim = 512
    elif expert_name == 'resnet50': m = ResNet50Network(3, num_classes);        dim = 2048
    elif expert_name == 'mlp':      m = MLPNetwork(3, num_classes, img_size);   dim = 128
    elif expert_name == 'rnn':      m = RNNNetwork(3, num_classes, img_size);   dim = 256
    elif expert_name == 'qwen1.5b':
        m = LLMExpertClassifier(device=device_expert)
        return m.to(device_expert), m.feat_dim
    else:
        raise ValueError(f"Unknown expert: {expert_name}")
    return m.to(device_expert), dim

                                            
                                      
                                            
class FeatureProjector(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(input_dim, output_dim), nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
    def forward(self, x):
        return self.proj(x)

def feature_repel_loss(current_features, expert_features, margin=0.0):
    cosine = F.cosine_similarity(current_features.float(), expert_features.float(), dim=1)
    return F.relu(cosine - margin).pow(2).mean()

def confidence_weighted_feature_repel_loss(current_features, expert_features, expert_logits, margin=0.0):
    cosine = F.cosine_similarity(current_features.float(), expert_features.float(), dim=1)
    per_sample = F.relu(cosine - margin).pow(2)
    with torch.no_grad():
        confidence = F.softmax(expert_logits.float(), dim=1).max(dim=1).values
        confidence = confidence / confidence.mean().clamp_min(1e-6)
    return (confidence * per_sample).mean()

def anti_expert_target_loss(student_logits, expert_logits):
    with torch.no_grad():
        expert_targets = expert_logits.float().argmax(dim=1, keepdim=True)
    return F.log_softmax(student_logits.float(), dim=1).gather(1, expert_targets).mean()

def forget_entropy_loss(logits):
    probs = F.softmax(logits.float(), dim=1)
    log_probs = F.log_softmax(logits.float(), dim=1)
    entropy = -(probs * log_probs).sum(dim=1)
    return -entropy.mean()

def prepare_expert_images(images, device_expert, expert_input_size=32):
    images = images.to(device_expert)
    if images.shape[-1] != expert_input_size or images.shape[-2] != expert_input_size:
        images = F.interpolate(images.float(), size=expert_input_size, mode='bilinear', align_corners=False)
    return images

def fit_feature_projector(projector, expert_model, teacher_model, loader,
                          device_orig, device_expert, expert_input_size=32,
                          epochs=3, lr=1e-3):
    projector.train()
    expert_model.eval()
    teacher_model.eval()
    optimizer = optim.Adam(projector.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss, total_n = 0.0, 0
        for images, _ in loader:
            images_orig = images.to(device_orig)
            images_exp = prepare_expert_images(images_orig, device_expert, expert_input_size)

            with torch.no_grad():
                _, teacher_features = teacher_model(images_orig, return_features=True)
                _, expert_features = expert_model(images_exp, return_features=True)

            projected = projector(expert_features.to(device_orig).float())
            loss = F.mse_loss(projected, teacher_features.float())

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * images.size(0)
            total_n += images.size(0)

        print(f"  [Projector] Epoch {epoch+1}/{epochs} | Loss: {total_loss / max(total_n, 1):.6f}")

                                            
                                    
                                            
def train_standard(model, loader, device, epochs=10, lr=1e-3, patience=5, is_llm=False):
    """기존 vilun.py train_standard와 동일. LLM은 lr/grad_clip 조정."""
    model.train()
    lr_actual  = 2e-5 if is_llm else lr
    optimizer  = optim.AdamW(model.parameters(), lr=lr_actual)
    stopper    = EarlyStopping(patience=patience, mode='min')

    for epoch in range(epochs):
        ep_loss = 0.0
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            out    = model(imgs)
                                                             
            loss   = F.cross_entropy(out.float(), labels)
            loss.backward()
            if is_llm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss += loss.item()

        improved = stopper(ep_loss, model)
        if stopper.early_stop:
            print(f"    [Early Stop] Training stopped at epoch {epoch+1}.")
            break

    if stopper.best_state is not None:
        model.load_state_dict(stopper.best_state)
    return model

                                            
                                  
                                            
@torch.no_grad()
def evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose=True):
    """Sample-target evaluation aligned with vilun.py."""
    forget_idx_set = set(forget_indices.tolist() if hasattr(forget_indices, "tolist") else list(forget_indices))
    all_train_idx = list(range(len(train_set)))
    retain_train_idx = [i for i in all_train_idx if i not in forget_idx_set]
    forget_train_idx = list(forget_idx_set)

    def get_loader(dataset, indices):
        if len(indices) == 0:
            return None
        return DataLoader(Subset(dataset, indices), batch_size=64, shuffle=False)

    loaders = {
        'Train_Forget': get_loader(train_set, forget_train_idx),
        'Train_Retain': get_loader(train_set, retain_train_idx),
        'Test_Retain ': DataLoader(test_set, batch_size=64, shuffle=False),
    }

    model.eval()
    if verbose:
        print("  [Evaluation]")
        print(f"  {'Metric':<15} | {'Acc (%)':>8} | {'Loss':>8}")
        print("  " + "-"*38)

    results = {}
    for name, loader in loaders.items():
        if loader is None: continue
        correct, total, loss_sum = 0, 0, 0.0
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            out    = model(imgs).float()
            loss_sum += F.cross_entropy(out, labels, reduction='sum').item()
            correct  += (out.argmax(1) == labels).sum().item()
            total    += labels.size(0)
        acc = 100 * correct / total
        avg_loss = loss_sum / total
        results[name] = {'acc': acc, 'loss': avg_loss}
        if verbose: print(f"  {name:<15} | {acc:>8.2f} | {avg_loss:>8.4f}")

    if verbose: print("  " + "-"*38)
    return results


@torch.no_grad()
def evaluate_ppl(model, loader, device):
    """
    LLM 추가 지표: 이미지 배치에 대한 cross-entropy를 perplexity로 변환.
    분류 loss exp()로 PPL 근사 — LLM 특성 반영.
    """
    model.eval()
    total_loss, total_n = 0.0, 0
    for imgs, labels in loader:
        imgs   = imgs.to(device)
        labels = labels.to(device)
        out    = model(imgs).float()
        loss   = F.cross_entropy(out, labels, reduction='sum').item()
        total_loss += loss
        total_n    += labels.size(0)
    avg_loss = total_loss / max(total_n, 1)
    return float(np.exp(avg_loss))                  


def evaluate_full(model, train_set, test_set, forget_indices, device,
                  forget_loader, retain_loader, verbose=True):
    """4-quadrant Acc + Forget/Retain PPL 통합 평가."""
    res = evaluate_4_quadrant(model, train_set, test_set, forget_indices, device, verbose)

    forget_ppl = evaluate_ppl(model, forget_loader, device)
    retain_ppl = evaluate_ppl(model, retain_loader, device)

    if verbose:
        print(f"  {'Forget PPL (↑)':>15} | {forget_ppl:>8.4f}")
        print(f"  {'Retain PPL (↓)':>15} | {retain_ppl:>8.4f}")

    res['forget_ppl'] = forget_ppl
    res['retain_ppl'] = retain_ppl
    return res

                                            
                   
                                            
def run_unlearning(args, orig_model, expert_model,
                   forget_indices,
                   forget_loader, retain_loader,
                   train_set, test_set,
                   eval_forget_loader, eval_retain_loader,
                   heldout_loader=None):
    """
    mode='standard' : forget data 기반 (기존 vilun.py)
    mode='heldout'  : heldout data 기반 (vilun.py)
    loss 부호만 다르고 나머지는 동일.
    """
    is_heldout    = (args.mode == 'heldout')
    active_loader = heldout_loader if is_heldout else forget_loader
    device_orig   = args.device_orig
    device_expert = args.device_expert

    orig_feat_dim   = orig_model.feat_dim
    expert_feat_dim = expert_model.feat_dim

    expert_model.eval()
    projector = FeatureProjector(expert_feat_dim, orig_feat_dim).to(device_orig)
    frozen_teacher = copy.deepcopy(orig_model).to(device_orig)
    frozen_teacher.eval()
    for param in frozen_teacher.parameters():
        param.requires_grad = False

    fit_loader = heldout_loader if is_heldout else retain_loader
    expert_input_size = 224 if args.expert_model == 'qwen1.5b' else 32
    fit_feature_projector(
        projector, expert_model, frozen_teacher, fit_loader,
        device_orig, device_expert, expert_input_size,
        epochs=args.projector_epochs, lr=args.projector_lr
    )
    projector.eval()
    for param in projector.parameters():
        param.requires_grad = False

    optimizer = optim.Adam(orig_model.parameters(), lr=args.unlearn_lr)
    retain_iter   = iter(retain_loader)
    history = {
        'epoch': [], 'train_forget_acc': [], 'train_forget_loss': [],
        'train_retain_acc': [], 'train_retain_loss': [],
        'test_retain_acc': [], 'test_retain_loss': [],
        'forget_ppl': [], 'retain_ppl': []
    }
    best_epoch, best_retain_acc, best_forget_acc = 1, 0.0, 0.0
    best_forget_ppl, best_retain_ppl             = 0.0, 0.0

    for epoch in range(args.unlearn_epochs):
        orig_model.train(); projector.eval()

        for imgs_a, _ in active_loader:
            try:    imgs_r, labels_r = next(retain_iter)
            except StopIteration:
                retain_iter = iter(retain_loader)
                imgs_r, labels_r = next(retain_iter)

            imgs_a   = imgs_a.to(device_orig)
            imgs_r   = imgs_r.to(device_orig)
            labels_r = labels_r.to(device_orig)

            student_active_logits, feat_orig = orig_model(imgs_a, return_features=True)

            with torch.no_grad():
                imgs_a_exp = prepare_expert_images(imgs_a, device_expert, expert_input_size)
                expert_active_logits, feat_exp = expert_model(imgs_a_exp, return_features=True)
                feat_exp = feat_exp.to(device_orig).float()
                expert_active_logits = expert_active_logits.to(device_orig).float()

            feat_orig = feat_orig.float()
            feat_aligned = projector(feat_exp)

            out_r = orig_model(imgs_r).float()
            loss_retain = F.cross_entropy(out_r, labels_r)

            if is_heldout:
                loss_unlearn = confidence_weighted_feature_repel_loss(
                    feat_orig, feat_aligned, expert_active_logits, args.feature_margin
                )
                loss_anti_expert = anti_expert_target_loss(student_active_logits, expert_active_logits)
                loss = (
                    args.alpha * loss_retain
                    + args.beta * loss_unlearn
                    + args.heldout_anti_gamma * loss_anti_expert
                )
            else:
                loss_unlearn = feature_repel_loss(feat_orig, feat_aligned, args.feature_margin)
                loss_forget_entropy = forget_entropy_loss(student_active_logits)
                loss = (
                    args.alpha * loss_retain
                    + args.beta * loss_unlearn
                    + args.forget_entropy_gamma * loss_forget_entropy
                )

            optimizer.zero_grad()
            loss.backward()
            if args.orig_model != 'none':                
                torch.nn.utils.clip_grad_norm_(orig_model.parameters(), max_norm=1.0)
            optimizer.step()

        res = evaluate_full(orig_model, train_set, test_set, forget_indices,
                            device_orig, eval_forget_loader, eval_retain_loader, verbose=False)

        retain_acc = res['Test_Retain ']['acc']
        train_retain_acc = res['Train_Retain']['acc']
        forget_acc = res['Train_Forget']['acc']
        print(
            f"  Epoch [{epoch+1:>4}/{args.unlearn_epochs}]"
            f" | Forget Acc: {forget_acc:6.2f}%"
            f" | Retain Acc: {train_retain_acc:6.2f}%"
            f" | Test Acc: {retain_acc:6.2f}%"
            f" | Forget PPL: {res['forget_ppl']:8.4f}"
            f" | Retain PPL: {res['retain_ppl']:8.4f}"
        )
        history['epoch'].append(epoch + 1)
        history['train_forget_acc'].append(forget_acc)
        history['train_forget_loss'].append(res['Train_Forget']['loss'])
        history['train_retain_acc'].append(train_retain_acc)
        history['train_retain_loss'].append(res['Train_Retain']['loss'])
        history['test_retain_acc'].append(retain_acc)
        history['test_retain_loss'].append(res['Test_Retain ']['loss'])
        history['forget_ppl'].append(res['forget_ppl'])
        history['retain_ppl'].append(res['retain_ppl'])

    best_idx = int(np.argmax(history['test_retain_acc']))
    best_epoch = history['epoch'][best_idx]
    best_retain_acc = history['test_retain_acc'][best_idx]
    best_forget_acc = history['train_forget_acc'][best_idx]
    best_forget_ppl = history['forget_ppl'][best_idx]
    best_retain_ppl = history['retain_ppl'][best_idx]

    return history, best_epoch, best_retain_acc, best_forget_acc, best_forget_ppl, best_retain_ppl

                                            
        
                                            
def save_results(args, history, best_epoch,
                 best_retain_acc, best_forget_acc,
                 best_forget_ppl, best_retain_ppl,
                 time_orig, time_expert, time_unlearn,
                 forget_indices_path):
    os.makedirs(args.save_path, exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)

    orig_tag = sanitize_tag(args.orig_model.split('/')[-1] if '/' in args.orig_model else args.orig_model)
    save_stem = args.save_tag or f"cifar10_{orig_tag}_{args.expert_model}_seed{args.seed}"

           
    if args.mode == 'heldout':
        file_tag = f"vilun_llm_heldout_{save_stem}"
        model_path = os.path.join(args.save_path, f"unlearned_{file_tag}.pt")
        method_name = "ViLUN-LLM-HeldOut"
    else:
        file_tag = f"vilun_llm_{save_stem}"
        model_path = os.path.join(args.save_path, f"{file_tag}.pt")
        method_name = "ViLUN-LLM"
    torch.save(args._model_state, model_path)
    print(f"[Save] Model → {model_path}")

                   
    hist_path = os.path.join(args.history_dir, f"{file_tag}.csv")
    with open(hist_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(history.keys())
        writer.writerows(zip(*history.values()))
    print(f"[Save] Epoch history → {hist_path}")

                  
    summary_path = os.path.join(args.history_dir, 'summary_vilun_llm.csv')
    header = [
        'Method', 'Dataset', 'Orig_Model', 'Expert_Model', 'Seed', 'Save_Tag',
        'Alpha', 'Beta', 'Best_Epoch',
        'Best_Train_Retain_Acc', 'Best_Test_Retain_Acc', 'Best_Forget_Acc',
        'Best_Forget_PPL', 'Best_Retain_PPL',
        'Time_Orig(s)', 'Time_Expert(s)', 'Time_Unlearn(s)',
        'Model_Path', 'Forget_Indices_Path'
    ]
    write_header = not os.path.exists(summary_path) or os.path.getsize(summary_path) == 0
    with open(summary_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(header)
        writer.writerow([
            method_name,
            'cifar10', args.orig_model, args.expert_model, args.seed, save_stem,
            args.alpha, args.beta, best_epoch,
            f"{history['train_retain_acc'][best_epoch-1]:.2f}", f"{best_retain_acc:.2f}", f"{best_forget_acc:.2f}",
            f"{best_forget_ppl:.4f}", f"{best_retain_ppl:.4f}",
            f"{time_orig:.1f}", f"{time_expert:.1f}", f"{time_unlearn:.1f}",
            model_path, forget_indices_path
        ])
    print(f"[Save] Summary → {summary_path}")

    print(f"\n{'='*55}")
    print(f"  [Result] Mode           : {method_name}")
    print(f"  [Result] Orig Model     : {args.orig_model}")
    print(f"  [Result] Expert Model   : {args.expert_model}")
    print(f"  [Result] Best Epoch     : {best_epoch}")
    print(f"  [Result] Train Retain   : {history['train_retain_acc'][best_epoch-1]:.2f}%")
    print(f"  [Result] Test Retain    : {best_retain_acc:.2f}%")
    print(f"  [Result] Forget Acc     : {best_forget_acc:.2f}%  (↓ good)")
    print(f"  [Result] Forget PPL     : {best_forget_ppl:.4f}")
    print(f"  [Result] Retain PPL     : {best_retain_ppl:.4f}")
    print(f"  [Time]   Unlearn        : {time_unlearn:.1f}s")
    print(f"{'='*55}")

                                            
                 
                                            
def run_pipeline(args):
    set_seed(args.seed)
    print(f"\n=== ViLUN-LLM-CIFAR10 [{args.mode.upper()}] ===")
    print(f"  Orig: {args.orig_model}  |  Expert: {args.expert_model}")
    print(f"  Sample-target forget setting  |  Seed: {args.seed}\n")

    device_orig   = args.device_orig
    device_expert = args.device_expert

                                              
                   
                                              
    print("[Step 1] Loading CIFAR10 ...")
    train_llm, test_llm, train_small, test_small = load_cifar10(args.data_dir)
    forget_idx, forget_indices_path = get_or_create_forget_indices(args, train_llm)
    forget_idx_set = set(forget_idx.tolist())
    retain_idx = [i for i in range(len(train_llm)) if i not in forget_idx_set]

    forget_loader_llm = build_loader(train_llm, forget_idx.tolist(), batch_size=args.batch_size, shuffle=True)
    retain_loader_llm = build_loader(train_llm, retain_idx, batch_size=args.batch_size, shuffle=True)
    eval_forget_llm = build_loader(train_llm, forget_idx.tolist(), batch_size=args.batch_size, shuffle=False)
    eval_retain_llm = build_loader(train_llm, retain_idx, batch_size=args.batch_size, shuffle=False)
    forget_loader_small = build_loader(train_small, forget_idx.tolist(), batch_size=args.batch_size, shuffle=True)

    full_loader_llm = DataLoader(train_llm, batch_size=args.batch_size, shuffle=True)

    heldout_loader = None
    if args.mode == 'heldout':
        heldout_loader = build_heldout_loader(test_llm, args.heldout_ratio, batch_size=args.batch_size)

                                              
                                          
                                              
    print(f"\n[Step 2] Original LLM ({args.orig_model}) ...")
    orig_model = LLMVisionClassifier(
        model_name=args.orig_model,
        num_classes=10,
        num_layers=args.llm_layers,
        device=device_orig
    )
    time_orig = 0.0

    if args.original and os.path.exists(args.original):
        print(f"  Loading checkpoint: {args.original}")
        orig_model.load_state_dict(torch.load(args.original, map_location='cpu'))
    else:
        print("  Fine-tuning on CIFAR10 (full train set) ...")
        t0 = time.time()
        orig_model = train_standard(
            orig_model, full_loader_llm, device_orig,
            epochs=args.train_epochs, lr=args.orig_lr,
            patience=args.patience, is_llm=True
        )
        time_orig = time.time() - t0

        orig_tag  = sanitize_tag(args.orig_model.split('/')[-1] if '/' in args.orig_model else args.orig_model)
        ft_path   = os.path.join(args.save_path,
                    f"original_llm_{orig_tag}_cifar10_seed{args.seed}.pt")
        os.makedirs(args.save_path, exist_ok=True)
        torch.save(orig_model.state_dict(), ft_path)
        print(f"  Saved: {ft_path}  [{time_orig:.1f}s]")

    print("\n[Baseline] Original model:")
    evaluate_full(orig_model, train_llm, test_llm, forget_idx,
                  device_orig, eval_forget_llm, eval_retain_llm)

    if args.unlearn_epochs == 0:
        print("  [Skip] unlearn_epochs=0: original LLM checkpoint prepared only.")
        return

                                              
                             
                                              
    print(f"\n[Step 3] Expert ({args.expert_model}) — forget data only ...")
    t1 = time.time()
    expert_model, expert_feat_dim = get_expert_model(
        args.expert_model, num_classes=10, device_expert=device_expert, img_size=32
    )
    expert_model.feat_dim = expert_feat_dim

                                    
    expert_forget_loader = forget_loader_llm if args.expert_model == 'qwen1.5b'\
                           else forget_loader_small

    expert_model = train_standard(
        expert_model, expert_forget_loader, device_expert,
        epochs=10, lr=args.expert_lr,
        patience=5, is_llm=(args.expert_model == 'qwen1.5b')
    )
    time_expert = time.time() - t1
    print(f"  [Time] {time_expert:.1f}s")

    if args.mode == 'heldout':
        print("  >> Expert params handed to model owner. Forget data stays with requester. ✓")

                                              
                         
                                              
    print(f"\n[Step 4] Unlearning [{args.mode}] ...")
    t2 = time.time()
    (history, best_epoch,
     best_retain_acc, best_forget_acc,
     best_forget_ppl, best_retain_ppl) = run_unlearning(
        args, orig_model, expert_model,
        forget_idx,
        forget_loader_llm, retain_loader_llm,
        train_llm, test_llm,
        eval_forget_llm, eval_retain_llm,
        heldout_loader
    )
    time_unlearn = time.time() - t2

                                                         
    if args.unlearn_epochs == 0:
        print("  [Skip] unlearn_epochs=0: summary not saved (fine-tune only run).")
        return

                  
    args._model_state = orig_model.state_dict()

    save_results(args, history, best_epoch,
                 best_retain_acc, best_forget_acc,
                 best_forget_ppl, best_retain_ppl,
                 time_orig, time_expert, time_unlearn,
                 forget_indices_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',         type=str, default='standard',
                        choices=['standard', 'heldout'])
    parser.add_argument('--orig_model',   type=str, default='Qwen/Qwen2.5-7B',
                        help='HuggingFace LLM ID (encoder backbone)')
    parser.add_argument('--expert_model', type=str, default='cnn',
                        choices=['cnn', 'lenet', 'resnet18', 'resnet50', 'mlp', 'rnn', 'qwen1.5b'])
    parser.add_argument('--original',     type=str, default=None,
                        help='Fine-tuned checkpoint .pt. 없으면 새로 학습.')
    parser.add_argument('--llm_layers',   type=int, default=8,
                        help='LLM에서 사용할 레이어 수 (메모리 절감, default=8)')

    parser.add_argument('--data_dir',     type=str,   default='./data')
    parser.add_argument('--target_id',    type=int,   default=0)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--heldout_ratio',type=float, default=0.1)

    parser.add_argument('--train_epochs',  type=int,   default=10)
    parser.add_argument('--unlearn_epochs',type=int,   default=100)
    parser.add_argument('--orig_lr',       type=float, default=2e-5)
    parser.add_argument('--expert_lr',     type=float, default=1e-3)
    parser.add_argument('--unlearn_lr',    type=float, default=1e-5)
    parser.add_argument('--alpha',         type=float, default=1.0)
    parser.add_argument('--beta',          type=float, default=10.0)
    parser.add_argument('--feature_margin', type=float, default=-0.2,
                        help='Cosine margin for ViLUn/ViLUn_f feature repulsion')
    parser.add_argument('--forget_entropy_gamma', type=float, default=0.5,
                        help='ViLUn_f forget entropy loss weight')
    parser.add_argument('--heldout_anti_gamma', type=float, default=0.1,
                        help='ViLUn anti-villain target loss weight')
    parser.add_argument('--projector_epochs', type=int, default=3,
                        help='Feature projector fitting epochs before unlearning')
    parser.add_argument('--projector_lr', type=float, default=1e-3,
                        help='Feature projector fitting learning rate')
    parser.add_argument('--patience',      type=int,   default=10)

    parser.add_argument('--device_orig',   type=str, default='cuda:0')
    parser.add_argument('--device_expert', type=str, default='cuda:1')

    parser.add_argument('--seed',       type=int,  default=42)
    parser.add_argument('--save_path',  type=str,  default='./saved_models')
    parser.add_argument('--history_dir', type=str, default='./history')
    parser.add_argument('--save_tag',   type=str,  default=None)
    parser.add_argument('--save_model', action='store_true')

    args = parser.parse_args()
    run_pipeline(args)
