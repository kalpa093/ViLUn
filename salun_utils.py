import os
from typing import Dict

import torch


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0.0
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(output, target, topk=(1,)):
    """Compute top-k accuracy in percent."""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(state: Dict, filename: str):
    os.makedirs(os.path.dirname(filename) or ".", exist_ok=True)
    torch.save(state, filename)


class SaliencyPruner:
    """Gradient-saliency based parameter mask generator.

    This keeps the original SalUn structure: compute gradients on the forget set,
    then keep only the most salient parameters during unlearning.
    """

    def __init__(self, model, device):
        self.model = model
        self.device = device

    def _unpack_batch(self, batch):
        if isinstance(batch, (list, tuple)):
            if len(batch) >= 2:
                return batch[0], batch[1]
            return batch[0], None
        return batch, None

    def compute_gradients(self, loader, criterion):
        self.model.eval()
        grads = {
            name: torch.zeros_like(param, device="cpu")
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        num_batches = 0

        for batch in loader:
            inputs, targets = self._unpack_batch(batch)
            if targets is None:
                continue
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            self.model.zero_grad(set_to_none=True)
            outputs = self.model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if not param.requires_grad or param.grad is None:
                    continue
                grads[name] += param.grad.detach().abs().cpu()
            num_batches += 1

        if num_batches > 0:
            for name in grads:
                grads[name] /= num_batches
        return grads

    def generate_mask(self, grads, ratio=0.01):
        if not grads:
            return {}

        flat_scores = torch.cat([g.reshape(-1) for g in grads.values()])
        total = flat_scores.numel()
        keep = max(1, int(total * ratio))
        if keep >= total:
            threshold = flat_scores.min() - 1e-12
        else:
            threshold = torch.topk(flat_scores, keep, largest=True).values.min()

        mask = {}
        for name, g in grads.items():
            mask[name] = (g >= threshold).float().to(self.device)
        return mask
