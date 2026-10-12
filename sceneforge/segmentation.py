"""Part segmentation: PointNet against a point Transformer.

Both consume the same (B, N, 6) XYZ+RGB tensor and emit per-point logits, so the
only thing that varies between the two arms is how points exchange information:
PointNet through a single global max-pool, the Transformer through attention over
a local neighbourhood.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def _mlp(dims: list[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for a, b in zip(dims[:-1], dims[1:]):
        layers += [nn.Conv1d(a, b, 1), nn.BatchNorm1d(b), nn.ReLU(inplace=True)]
    return nn.Sequential(*layers)


class PointNetSeg(nn.Module):
    """Per-point features, one global summary, concatenate, classify."""

    def __init__(self, in_dim: int = 6, n_classes: int = 5):
        super().__init__()
        self.local = _mlp([in_dim, 64, 64])
        self.glob = _mlp([64, 128, 1024])
        self.head = nn.Sequential(
            nn.Conv1d(64 + 1024, 512, 1), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
            nn.Conv1d(512, 128, 1), nn.BatchNorm1d(128), nn.ReLU(inplace=True),
            nn.Conv1d(128, n_classes, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, N, C)
        x = x.transpose(1, 2)                             # (B, C, N)
        local = self.local(x)
        g = self.glob(local).max(dim=2, keepdim=True).values
        fused = torch.cat([local, g.expand(-1, -1, local.shape[2])], dim=1)
        return self.head(fused).transpose(1, 2)           # (B, N, n_classes)


def knn(x: torch.Tensor, k: int) -> torch.Tensor:
    """Indices of the k nearest neighbours of every point. x is (B, N, 3)."""
    d = torch.cdist(x, x)
    return d.topk(k, largest=False).indices               # (B, N, k)


class PointTransformerSeg(nn.Module):
    """Local self-attention over a k-nearest-neighbour graph.

    The dominant cost here is the neighbourhood gather rather than the matmul,
    which is why the comparison against PointNet is reported as per-step wall
    time at a fixed batch size rather than as a FLOP count.
    """

    def __init__(self, in_dim: int = 6, n_classes: int = 5, dim: int = 64, k: int = 16):
        super().__init__()
        self.k = k
        self.embed = nn.Linear(in_dim, dim)
        self.q = nn.Linear(dim, dim)
        self.kk = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.pos = nn.Sequential(nn.Linear(3, dim), nn.ReLU(inplace=True), nn.Linear(dim, dim))
        self.head = nn.Sequential(
            nn.Linear(dim, 128), nn.ReLU(inplace=True), nn.Linear(128, n_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:   # (B, N, C)
        xyz = x[..., :3]
        h = self.embed(x)                                  # (B, N, D)
        idx = knn(xyz, self.k)                             # (B, N, k)
        B, N, D = h.shape
        gather = lambda t: torch.gather(
            t.unsqueeze(1).expand(-1, N, -1, -1), 2,
            idx.unsqueeze(-1).expand(-1, -1, -1, t.shape[-1]))
        rel = xyz.unsqueeze(2) - gather(xyz)               # (B, N, k, 3)
        pos = self.pos(rel)                                # (B, N, k, D)
        attn = (self.q(h).unsqueeze(2) - gather(self.kk(h)) + pos) / (D ** 0.5)
        attn = F.softmax(attn, dim=2)
        out = (attn * (gather(self.v(h)) + pos)).sum(dim=2)
        return self.head(out)


@torch.no_grad()
def mean_iou(logits: torch.Tensor, target: torch.Tensor, n_classes: int = 5) -> float:
    """Unweighted mean IoU over classes present in either prediction or target."""
    pred = logits.argmax(dim=-1)
    ious = []
    for c in range(n_classes):
        p, t = pred == c, target == c
        union = (p | t).sum().item()
        if union:
            ious.append((p & t).sum().item() / union)
    return float(sum(ious) / len(ious)) if ious else float("nan")
