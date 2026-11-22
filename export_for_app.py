"""
Run this in the SAME Colab session as the training notebook, after Step 9
(training) has finished, with:

    %run -i /content/export_for_app.py

It packages everything the application needs into /content/deploy.zip:
  deploy/config.json   per-view normalisation, learned-fusion weights,
                       Youden thresholds, test metrics, library versions
  deploy/ckpt/*.pt     the best checkpoint for every model × view
  deploy/samples/      6 held-out test exams (3 tears, 3 intact) for the live demo

The notebook discards the fitted fusion model and the normalisation stats,
so this script recomputes them from the saved checkpoints exactly as Step 9 did.
"""
import json, os, shutil, zipfile
import numpy as np, pandas as pd, torch, timm, torchvision, sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, roc_auc_score, confusion_matrix)

OUT = '/content/deploy'
os.makedirs(f'{OUT}/ckpt', exist_ok=True)
os.makedirs(f'{OUT}/samples', exist_ok=True)

cfg = {
    'versions': {'torch': torch.__version__, 'torchvision': torchvision.__version__,
                 'timm': timm.__version__, 'sklearn': sklearn.__version__},
    'n_slices': CFG['n_slices'], 'n_tta': CFG['n_tta'],
    'normalization': {v: {'mean': DATA[v]['mean'], 'std': DATA[v]['std']} for v in VIEWS},
    'models': {},
}


def load_ckpt(name, view):
    p = CFG['pretrained']; CFG['pretrained'] = False
    try:
        bb, d = BUILDERS[name]()
    finally:
        CFG['pretrained'] = p
    m = GatedAttentionMIL(bb, d)
    m.load_state_dict(torch.load(ckpt_path(name, view), map_location='cpu'))
    return m.to(DEV).eval()


for name in BUILDERS:
    if not all(os.path.exists(ckpt_path(name, v)) for v in VIEWS):
        print(f'skip {name}: checkpoints missing'); continue
    print(f'\n── {name}')
    Pv, Pt = [], []
    for v in VIEWS:
        seed_all(CFG['seed'])
        _, vad, ted = loaders(v)
        m = load_ckpt(name, v)
        Pv.append(tta_predict(m, vad)); Pt.append(tta_predict(m, ted))
        shutil.copy(ckpt_path(name, v), f"{OUT}/ckpt/{os.path.basename(ckpt_path(name, v))}")
        m.cpu(); del m; free_gpu()
    Pv, Pt = np.stack(Pv), np.stack(Pt)
    yv = DATA[VIEWS[0]]['val'][1]; yt = DATA[VIEWS[0]]['test'][1]

    lr = LogisticRegression(C=1.0, max_iter=1000, random_state=CFG['seed']).fit(Pv.T, yv)
    fv = lr.predict_proba(Pv.T)[:, 1]; ft = lr.predict_proba(Pt.T)[:, 1]
    thr = youden_thr(yv, fv)
    pred = ft > thr
    cm_ = confusion_matrix(yt, pred, labels=[0, 1])
    metrics = {
        'accuracy': float(accuracy_score(yt, pred)),
        'precision': float(precision_score(yt, pred, zero_division=0)),
        'recall': float(recall_score(yt, pred, zero_division=0)),
        'specificity': float(cm_[0, 0] / max(cm_[0].sum(), 1)),
        'f1': float(f1_score(yt, pred, zero_division=0)),
        'auc': float(roc_auc_score(yt, ft)),
        'per_view_auc': {v: float(roc_auc_score(yt, Pt[i])) for i, v in enumerate(VIEWS)},
        'confusion': {'tn': int(cm_[0, 0]), 'fp': int(cm_[0, 1]), 'fn': int(cm_[1, 0]), 'tp': int(cm_[1, 1])},
    }
    params = next((r['Parameters (M)'] for r in results if r['Model'] == name), None)
    cost = next((c for c in costs if c['Model'] == name), {})
    cfg['models'][name] = {
        'fusion': {'method': 'learned', 'coef': lr.coef_[0].tolist(), 'intercept': float(lr.intercept_[0])},
        'threshold': thr,
        'test_metrics': metrics,
        'params_m': params,
        'train_time_s': cost.get('Training Time (s)'),
        'peak_vram_mb': cost.get('Peak Train Memory (MB)'),
    }
    print(f"  thr={thr:.3f}  acc={metrics['accuracy']:.4f}  auc={metrics['auc']:.4f}")

# ── demo samples from the held-out test split ──────────────────────────────
r = CFG['data_root']
dte = pd.read_csv(f'{r}/valid-acl.csv', header=None, names=['id', 'y'])
rng = np.random.default_rng(0)
picked = list(rng.choice(dte[dte.y == 1].id.values, 3, replace=False)) + \
         list(rng.choice(dte[dte.y == 0].id.values, 3, replace=False))
samples = []
for i in picked:
    sid = f'{int(i):04d}'
    os.makedirs(f'{OUT}/samples/{sid}', exist_ok=True)
    for v in VIEWS:
        shutil.copy(f'{r}/valid/{v}/{sid}.npy', f'{OUT}/samples/{sid}/{v}.npy')
    samples.append({'id': sid, 'label': 'ACL tear' if int(dte[dte.id == i].y.iloc[0]) else 'Intact'})
cfg['samples'] = samples

with open(f'{OUT}/config.json', 'w') as f:
    json.dump(cfg, f, indent=2)

zip_path = '/content/deploy.zip'
with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as z:
    for root, _, files in os.walk(OUT):
        for fn in files:
            full = os.path.join(root, fn)
            z.write(full, os.path.relpath(full, '/content'))
print(f'\n✓ Wrote {zip_path} ({os.path.getsize(zip_path)/2**20:.0f} MB)')
print('  Library versions to install on your laptop:', cfg['versions'])
try:
    from google.colab import files; files.download(zip_path)
except Exception:
    pass
