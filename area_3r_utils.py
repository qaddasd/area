import os
import sys
import torch
import numpy as np
from PIL import Image

import importlib

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Local bundled weights for the Area-3R geometric matching model.
AREA_3R_WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "Area-3R")

# The Area-3R runtime (model architecture / inference / matching kernels) is
# provided alongside this project. No weights are downloaded from the internet:
# they are loaded from AREA_3R_WEIGHTS_DIR.
_CORE_PKG = "mast3r"
_GEOM_PKG = "dust3r"

AREA_3R_RUNTIME_DIR = None
for _candidate in [
    os.path.join(PROJECT_ROOT, "area_3r_runtime"),
    os.path.join(PROJECT_ROOT, "..", "area_3r_runtime"),
    os.path.expanduser("~/area_3r_runtime"),
]:
    _candidate = os.path.abspath(_candidate)
    if os.path.exists(os.path.join(_candidate, _CORE_PKG, "model.py")):
        AREA_3R_RUNTIME_DIR = _candidate
        break

if AREA_3R_RUNTIME_DIR is not None:
    for _sub in (AREA_3R_RUNTIME_DIR,
                 os.path.join(AREA_3R_RUNTIME_DIR, _GEOM_PKG),
                 os.path.join(AREA_3R_RUNTIME_DIR, _GEOM_PKG, "croco")):
        if os.path.exists(_sub) and _sub not in sys.path:
            sys.path.insert(0, _sub)
    print(f"[Area-3R] Found Area-3R runtime at: {AREA_3R_RUNTIME_DIR}")
else:
    print("[Area-3R] Area-3R runtime not found. Place it alongside this repo as 'area_3r_runtime'.")

try:
    importlib.import_module(_CORE_PKG + ".utils.path_to_dust3r")
    _Area3RModel = getattr(importlib.import_module(_CORE_PKG + ".model"), "AsymmetricMASt3R")
    _area3r_inference = getattr(importlib.import_module(_GEOM_PKG + ".inference"), "inference")
    _area3r_reciprocal_nn = getattr(importlib.import_module(_CORE_PKG + ".fast_nn"), "fast_reciprocal_NNs")
    _AREA_3R_IMPORTS_OK = True
except Exception as e:
    _AREA_3R_IMPORTS_OK = False
    print(f"[Area-3R] Failed to import Area-3R runtime modules: {e}")
    import traceback
    traceback.print_exc()


_area_3r_model = None
device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')


def get_area_3r_model(use_half=True):
    """Load the Area-3R model fully offline. Cached after first call.

    Weights are loaded from the bundled ``Area-3R`` directory. No internet
    access is required.
    """
    global _area_3r_model
    if not _AREA_3R_IMPORTS_OK:
        print("[Area-3R] Cannot load - runtime modules failed to import")
        return None
    if _area_3r_model is not None:
        return _area_3r_model

    print(f"[Area-3R] Loading model on {device}...")
    try:
        if not os.path.isdir(AREA_3R_WEIGHTS_DIR):
            raise RuntimeError(
                f"Area-3R weights not found at {AREA_3R_WEIGHTS_DIR}. "
                f"The weights must be bundled inside the project."
            )

        print(f"[Area-3R] Loading from local weights: {AREA_3R_WEIGHTS_DIR}")
        model = _Area3RModel.from_pretrained(
            AREA_3R_WEIGHTS_DIR, img_size=(512, 512)
        ).to(device)
        model.eval()
        if device == 'cuda' and use_half:
            model = model.half()
            torch.cuda.empty_cache()
            print("[Area-3R] Converted model to FP16 (Half Precision) to save VRAM on CUDA.")
        _area_3r_model = model
        print("[Area-3R] Model loaded successfully.")
    except Exception as e:
        print(f"[Area-3R] Error loading model: {e}")
        raise e

    return _area_3r_model


def get_area_3r_matches(img1_pil, img2_pil, model, image_size=512):
    """Run Area-3R dense matching between two PIL images.

    Returns: matches_im0, matches_im1, conf
    """
    from torchvision import transforms

    is_half = False
    try:
        is_half = (next(model.parameters()).dtype == torch.float16)
    except Exception:
        pass

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def prep_img(pil_img):
        img_resized = pil_img.copy()
        img_resized.thumbnail((image_size, image_size))
        tensor = transform(img_resized).unsqueeze(0).to(device)
        if is_half:
            tensor = tensor.half()
        return {"img": tensor, "true_shape": np.array([img_resized.size[::-1]]), "idx": 0, "instance": "1"}

    view1 = prep_img(img1_pil)
    view2 = prep_img(img2_pil)
    view2["idx"] = 1

    images = [view1, view2]

    with torch.no_grad():
        output = _area3r_inference([tuple(images)], model, device, batch_size=1, verbose=False)

        v1, pred1 = output['view1'], output['pred1']
        v2, pred2 = output['view2'], output['pred2']

        desc1, desc2 = pred1['desc'].squeeze(0).detach(), pred2['desc'].squeeze(0).detach()

        matches_im0, matches_im1 = _area3r_reciprocal_nn(desc1.float(), desc2.float(), subsample_or_initxy1=8,
                                                         device=device, dist='dot', block_size=2**13)

        H0, W0 = v1['true_shape'][0]
        valid_matches_im0 = (matches_im0[:, 0] >= 3) & (matches_im0[:, 0] < int(W0) - 3) & (
            matches_im0[:, 1] >= 3) & (matches_im0[:, 1] < int(H0) - 3)

        H1, W1 = v2['true_shape'][0]
        valid_matches_im1 = (matches_im1[:, 0] >= 3) & (matches_im1[:, 0] < int(W1) - 3) & (
            matches_im1[:, 1] >= 3) & (matches_im1[:, 1] < int(H1) - 3)

        valid_matches = valid_matches_im0 & valid_matches_im1
        matches_im0 = matches_im0[valid_matches].astype(np.float32)
        matches_im1 = matches_im1[valid_matches].astype(np.float32)

        w1_o, h1_o = img1_pil.size
        scale1_x = w1_o / float(W0.cpu().item())
        scale1_y = h1_o / float(H0.cpu().item())

        w2_o, h2_o = img2_pil.size
        scale2_x = w2_o / float(W1.cpu().item())
        scale2_y = h2_o / float(H1.cpu().item())

        matches_im0[:, 0] *= scale1_x
        matches_im0[:, 1] *= scale1_y

        matches_im1[:, 0] *= scale2_x
        matches_im1[:, 1] *= scale2_y

    return matches_im0, matches_im1, None
