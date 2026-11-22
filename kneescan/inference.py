"""End-to-end inference: preprocessing → per-view MIL models → TTA →
learned fusion → threshold → explanation (attention + Grad-CAM)."""
import base64
import io
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from PIL import Image
import matplotlib
matplotlib.use("Agg")
from matplotlib import cm as mpl_cm

from .model_def import VIEWS, build, ckpt_filename, gradcam_layer

K = 12          # slices per view (CFG['n_slices'])
CROP = 224
N_TTA = 3       # CFG['n_tta']


# ───────────────────────── preprocessing ─────────────────────────
def pick_slices(n):
    """Same as notebook: K evenly spaced slices from the central 60%."""
    return np.linspace(int(.2 * (n - 1)), int(.8 * (n - 1)), K).round().astype(int)


def load_volume(src):
    """Accepts a path, bytes or array. Returns uint8 (L,256,256)."""
    if isinstance(src, np.ndarray):
        a = src
    elif isinstance(src, (bytes, bytearray)):
        a = np.load(io.BytesIO(src), allow_pickle=False)
    else:
        a = np.load(src, allow_pickle=False)
    if a.ndim != 3:
        raise ValueError(f"Expected a 3-D volume (slices, H, W); got shape {a.shape}")
    if a.shape[1:] != (256, 256):
        t = torch.from_numpy(a.astype(np.float32)).unsqueeze(1)
        a = F.interpolate(t, size=(256, 256), mode="bilinear", align_corners=False).squeeze(1).numpy()
    if not np.isfinite(a).all():
        raise ValueError("Volume contains NaN/Inf values")
    if a.dtype != np.uint8:
        lo, hi = float(a.min()), float(a.max())
        a = ((a - lo) / (hi - lo + 1e-8) * 255).round()
    return a.astype(np.uint8)


def preprocess(vol, mean, std):
    """uint8 (L,256,256) → tensor (1, K, 3, 224, 224), as KneeMRI eval mode."""
    idx = pick_slices(len(vol))
    x = torch.from_numpy(vol[idx]).float()
    x = TF.center_crop(x, [CROP, CROP])
    x = ((x - mean) / (std + 1e-8)).unsqueeze(1).repeat(1, 3, 1, 1)
    return x.unsqueeze(0), idx


# ───────────────────────── helpers ─────────────────────────
def png_b64(arr):
    """float [0,1] HxW or HxWx3 → base64 PNG string."""
    arr = np.clip(arr, 0, 1)
    if arr.ndim == 2:
        arr = np.stack([arr] * 3, -1)
    im = Image.fromarray((arr * 255).astype(np.uint8))
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def overlay(img, cam, alpha=.45):
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    return np.clip((1 - alpha) * np.stack([img] * 3, -1) + alpha * mpl_cm.jet(cam)[..., :3], 0, 1)


def fuzzy_fuse(P):
    """Wang et al. fuzzy-distance fusion (notebook fallback). P: (V,) or (V,N)."""
    P = np.asarray(P, dtype=float).reshape(len(P), -1)
    if len(P) == 1:
        return P[0]
    I = []
    for S in (1 - P, P):
        d = 1 - S
        e = np.linalg.norm(d, axis=0)
        m = d.sum(0)
        c = 1 - S.sum(0) / (np.sqrt(len(S)) * np.linalg.norm(S, axis=0) + 1e-9)
        I.append((e + 1e-6) * (m + 1e-6) * (c + 1e-6))
    return I[0] / (I[0] + I[1])


# ───────────────────────── engine ─────────────────────────
class Engine:
    def __init__(self, deploy_dir, device=None):
        self.dir = deploy_dir
        cfg_path = os.path.join(deploy_dir, "config.json")
        if not os.path.exists(cfg_path):
            raise FileNotFoundError(
                f"{cfg_path} not found. Run export_for_app.py in Colab after training "
                "and copy the 'deploy' folder here.")
        with open(cfg_path) as f:
            self.cfg = json.load(f)
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.cache = {}

    # models present on disk
    def available_models(self):
        out = []
        for name in self.cfg["models"]:
            ok = all(os.path.exists(os.path.join(self.dir, "ckpt", ckpt_filename(name, v))) for v in VIEWS)
            if ok:
                out.append(name)
        return out

    def _load(self, name, view):
        key = (name, view)
        if any(k[0] != name for k in self.cache):
            # keep only the active model's 3 views in memory (laptop-friendly)
            self.cache.clear()
            import gc; gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        if key not in self.cache:
            model, d = build(name)
            sd = torch.load(os.path.join(self.dir, "ckpt", ckpt_filename(name, view)), map_location="cpu")
            model.load_state_dict(sd)
            model.to(self.dev).eval()
            # keep a pristine copy: TTA passes in train mode would otherwise
            # update BatchNorm running stats and drift between requests
            pristine = {k: v.detach().clone() for k, v in model.named_buffers()}
            self.cache[key] = (model, d, pristine)
        return self.cache[key]

    @torch.no_grad()
    def _tta(self, model, pristine, x):
        """Same procedure as tta_predict() in the notebook."""
        random.seed(42)
        torch.manual_seed(42)
        ps = []
        for t in range(N_TTA):
            train_mode = t < N_TTA - 1
            model.train(train_mode)
            xx = x
            if train_mode and random.random() < .5:
                B, K_, C, H, W = x.shape
                xx = TF.hflip(x.flatten(0, 1)).view(B, K_, C, H, W)
            ps.append(torch.sigmoid(model(xx.to(self.dev)).float()).item())
        model.load_state_dict(pristine, strict=False)
        model.eval()
        return float(np.mean(ps))

    def _explain(self, name, model, d, x):
        """Attention weights (eval pass) + Grad-CAM on the most-attended slice."""
        layer = gradcam_layer(name, model.backbone)
        store = {}

        def fh(m, inp, out):
            store["a"] = out
            out.register_hook(lambda g: store.__setitem__("g", g))

        x = x.to(self.dev)
        flat = x.flatten(0, 1)                                   # (K,3,H,W)
        with torch.no_grad():                                    # attention for all slices
            _, att = model(x, return_attention=True)
            feats = model.backbone(flat)
        best = int(att[0].argmax())
        # Grad-CAM only through the most-attended slice (memory-light):
        h = layer.register_forward_hook(fh)
        try:
            with torch.enable_grad():
                f_best = model.backbone(flat[best:best + 1])
                f = torch.cat([feats[:best], f_best, feats[best + 1:]]).unsqueeze(0)
                a_ = torch.softmax(model.attn_w(model.attn_V(f) * model.attn_U(f)), dim=1)
                logit = model.head((a_ * f).sum(1)).squeeze()
                model.zero_grad(set_to_none=True)
                logit.backward()
        finally:
            h.remove()
        att = att[0].detach().cpu().numpy()
        a, g = store["a"].detach(), store["g"].detach()
        if a.shape[1] != d and a.shape[-1] == d:       # Swin: NHWC → NCHW
            a, g = a.permute(0, 3, 1, 2), g.permute(0, 3, 1, 2)
        cam = F.relu((g.mean((2, 3), keepdim=True) * a).sum(1, keepdim=True))
        cam = F.interpolate(cam, size=x.shape[-2:], mode="bilinear", align_corners=False)[:, 0]
        cam = (cam / (cam.amax((1, 2), keepdim=True) + 1e-8)).cpu().numpy()
        model.zero_grad(set_to_none=True)
        return att, cam[0]

    def predict(self, model_name, volumes, explain=True):
        """volumes: dict view -> uint8 array (L,256,256). Missing views allowed."""
        t0 = time.perf_counter()
        mcfg = self.cfg["models"][model_name]
        views_in = [v for v in VIEWS if v in volumes]
        if not views_in:
            raise ValueError("Provide at least one MRI plane")
        per_view = {}
        for v in views_in:
            vol = volumes[v]
            norm = self.cfg["normalization"][v]
            x, idx = preprocess(vol, norm["mean"], norm["std"])
            model, d, pristine = self._load(model_name, v)
            p = self._tta(model, pristine, x)
            entry = {"probability": p, "n_slices_in_volume": int(len(vol)),
                     "slice_indices": [int(i) for i in idx]}
            if explain:
                att, cam = self._explain(model_name, model, d, x)
                best = int(att.argmax())
                raw = x[0, best, 0].numpy()
                entry.update({
                    "attention": [float(a) for a in att],
                    "top_slice": best,
                    "image": png_b64((raw - raw.min()) / (raw.max() - raw.min() + 1e-8)),
                    "gradcam": png_b64(overlay(raw, cam)),
                })
            per_view[v] = entry

        # fusion
        P = np.array([per_view[v]["probability"] for v in views_in])
        fusion = mcfg.get("fusion", {})
        if fusion.get("method") == "learned" and len(views_in) == 3:
            z = float(np.dot(fusion["coef"], P) + fusion["intercept"])
            fused = 1 / (1 + np.exp(-z))
            method = "learned logistic"
        else:
            fused = float(fuzzy_fuse(P)[0]) if len(P) > 1 else float(P[0])
            method = "fuzzy-distance" if len(P) > 1 else "single view"
        thr = float(mcfg.get("threshold", 0.5))
        return {
            "model": model_name,
            "views": per_view,
            "fused_probability": float(fused),
            "threshold": thr,
            "prediction": "ACL tear" if fused > thr else "Intact",
            "fusion": method,
            "device": str(self.dev),
            "seconds": round(time.perf_counter() - t0, 2),
            "test_metrics": mcfg.get("test_metrics", {}),
        }
