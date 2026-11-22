"""SMOKE TEST ONLY — builds a fake deploy/ folder with RANDOM weights and
synthetic volumes so you can check the app starts before your real
checkpoints arrive. Predictions from this are meaningless.

    python tools/make_dummy_deploy.py deploy_dummy
    set KNEESCAN_DEPLOY=deploy_dummy   (Windows)   /  export KNEESCAN_DEPLOY=deploy_dummy
    uvicorn app:app
"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from kneescan.model_def import MODEL_NAMES, VIEWS, build, ckpt_filename

out = sys.argv[1] if len(sys.argv) > 1 else "deploy_dummy"
os.makedirs(f"{out}/ckpt", exist_ok=True)
torch.manual_seed(0)
cfg = {"versions": {"note": "dummy"}, "n_slices": 12, "n_tta": 3,
       "normalization": {v: {"mean": 60.0, "std": 50.0} for v in VIEWS}, "models": {}, "samples": []}
for name in MODEL_NAMES:
    for v in VIEWS:
        m, _ = build(name)
        torch.save(m.state_dict(), f"{out}/ckpt/{ckpt_filename(name, v)}")
    cfg["models"][name] = {"fusion": {"method": "learned", "coef": [1.0, 1.0, 1.0], "intercept": -1.5},
                           "threshold": 0.5, "test_metrics": {}, "params_m": None}
    print("built", name)
rng = np.random.default_rng(0)
for sid, lab in [("9001", "ACL tear"), ("9002", "Intact")]:
    os.makedirs(f"{out}/samples/{sid}", exist_ok=True)
    for v in VIEWS:
        np.save(f"{out}/samples/{sid}/{v}.npy", rng.integers(0, 255, (int(rng.integers(20, 45)), 256, 256), dtype=np.uint8))
    cfg["samples"].append({"id": sid, "label": lab})
json.dump(cfg, open(f"{out}/config.json", "w"), indent=2)
print("wrote", out)
