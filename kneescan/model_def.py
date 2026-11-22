"""Model definitions — copied from the training notebook (Step 5) so saved
checkpoints load with identical parameter names."""
import torch
import torch.nn as nn
import torchvision
import timm

MODEL_NAMES = [
    "MRNet-AlexNet (Baseline)",
    "ResNet-50",
    "EfficientNetV2-S",
    "ConvNeXt-Tiny",
    "Swin-T",
]
VIEWS = ["axial", "coronal", "sagittal"]


class GatedAttentionMIL(nn.Module):
    """Slice-level MIL with Gated Attention pooling (Ilse et al., 2018)."""

    def __init__(self, bb, d, hidden=128, dropout=0.25, use_grad_ckpt=False):
        super().__init__()
        self.backbone = bb
        self.use_grad_ckpt = use_grad_ckpt
        self.attn_V = nn.Sequential(nn.Linear(d, hidden), nn.Tanh())
        self.attn_U = nn.Sequential(nn.Linear(d, hidden), nn.Sigmoid())
        self.attn_w = nn.Linear(hidden, 1)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(d, 1)

    def forward(self, x, return_attention=False):  # x: (B, K, 3, H, W)
        B, K_ = x.shape[:2]
        feats = self.backbone(x.flatten(0, 1)).view(B, K_, -1)
        feats = self.drop(feats)
        a = torch.softmax(self.attn_w(self.attn_V(feats) * self.attn_U(feats)), dim=1)
        logit = self.head((a * feats).sum(1)).squeeze(1)
        if return_attention:
            return logit, a.squeeze(-1)
        return logit


def alexnet_bb():
    m = torchvision.models.alexnet(weights=None)
    return nn.Sequential(m.features, nn.AdaptiveAvgPool2d(1), nn.Flatten()), 256


def resnet50_bb():
    m = torchvision.models.resnet50(weights=None)
    d = m.fc.in_features
    m.fc = nn.Identity()
    return m, d


def timm_bb(name):
    m = timm.create_model(name, pretrained=False, num_classes=0)
    return m, m.num_features


BUILDERS = {
    "MRNet-AlexNet (Baseline)": alexnet_bb,
    "ResNet-50": resnet50_bb,
    "EfficientNetV2-S": lambda: timm_bb("tf_efficientnetv2_s"),
    "ConvNeXt-Tiny": lambda: timm_bb("convnext_tiny"),
    "Swin-T": lambda: timm_bb("swin_tiny_patch4_window7_224"),
}


def build(name):
    bb, d = BUILDERS[name]()
    return GatedAttentionMIL(bb, d), d


def ckpt_filename(name, view):
    """Same naming rule as ckpt_path() in the notebook."""
    safe = name.replace(" ", "_").replace("(", "").replace(")", "").replace("-", "_")
    return f"{safe}_{view}.pt"


def gradcam_layer(name, bb):
    """Same target layers as gc_target() in the notebook."""
    if name.startswith("MRNet-AlexNet"):
        return bb[0][11]
    if name.startswith("ResNet"):
        return bb.layer4[-1]
    if "EfficientNet" in name:
        return bb.bn2
    if "ConvNeXt" in name:
        return bb.stages[-1]
    if "Swin" in name:
        return bb.layers[-1]
    raise ValueError(name)
