import os
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as tfm
from PIL import Image
import pickle
import threading



AREA_RAW_DIM = 8448          # Native Area-loc output dimension
AREA_PCA_DIM = 1024          # Reduced dimension for indexing (tune as needed)
AREA_INPUT_SIZE = 322        # Input resolution (multiple of 14)



_area_model = None
_area_lock = threading.Lock()
_pca_model = None


if torch.backends.mps.is_available():
    _device = 'mps'
elif torch.cuda.is_available():
    _device = 'cuda'
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision('high')
    except Exception:
        pass
else:
    _device = 'cpu'




def get_area_model(device=None):
    """Load the Area-loc model (singleton, thread-safe).

    Loads fully offline from the bundled weights file ``Area-loc.safetensors``
    using the in-project ``area_loc_model.AreaLocModel`` architecture.
    No internet access is required.
    """
    global _area_model
    if _area_model is not None:
        return _area_model

    with _area_lock:
        if _area_model is not None:
            return _area_model

        dev = device or _device
        print(f"[Area-loc] Loading Area-loc model on {dev}...")

        from area_loc_model import AreaLocModel
        from safetensors.torch import load_file

        weights_path = os.path.join(os.path.dirname(__file__), "Area-loc.safetensors")
        if not os.path.exists(weights_path):
            raise RuntimeError(
                f"Area-loc weights not found at {weights_path}. "
                f"The weights file must be bundled inside the project."
            )

        model = AreaLocModel()
        state = load_file(weights_path)
        model.load_state_dict(state)
        print(f"[Area-loc] Loaded local weights from {weights_path}")

        model = model.eval().to(dev)

        # Freeze all parameters
        for p in model.parameters():
            p.requires_grad_(False)

        # Only apply MPS patch on MPS devices; on CUDA/CPU the native cached version is faster
        if dev == 'mps' and hasattr(model, 'backbone'):
            def _patched_backbone_forward(images):
                bb = model.backbone
                B, _, H, W = images.shape
                x = bb.patch_embed(images)
                cls_tokens = bb.cls_token.expand(B, -1, -1)
                x = torch.cat((cls_tokens, x), dim=1)
                x = x + bb.interpolate_pos_encoding(x, W, H)
                for block in bb.blocks:
                    x = block(x)
                x = bb.norm(x)
                cls_token = x[:, 0]
                patch_tokens = x[:, 1:]

                patch_features = patch_tokens.contiguous().reshape(
                    B, H // bb.patch_size, W // bb.patch_size, bb.embed_dim
                ).permute(0, 3, 1, 2).contiguous()
                return patch_features, cls_token

            model.backbone.forward = _patched_backbone_forward
            print("[Area-loc] Applied MPS-compatible patches")

        _area_model = model
        print(f"[Area-loc] Model ready. Output dim: {AREA_RAW_DIM}")
        return _area_model




def _preprocess_pil(pil_img, target_size=AREA_INPUT_SIZE):
    """Convert PIL image to normalized tensor for Area.
    
    Area-loc expects [B, 3, H, W] with values in [0, 1].
    H, W must be multiples of 14. The model auto-resizes internally,
    but we pre-resize for consistency and to control memory.
    """

    size = target_size
    if size % 14 != 0:
        size = round(size / 14) * 14

    img = pil_img.convert('RGB').resize((size, size), Image.BILINEAR)
    tensor = tfm.to_tensor(img)  # [3, H, W] in [0, 1]
    return tensor




def extract_area_descriptor(img_or_tensor, apply_pca_reduction=True):
    """Extract Area descriptor from a single PIL image or torch Tensor.
    
    Args:
        img_or_tensor: PIL Image or torch.Tensor of shape [3, H, W] or [1, 3, H, W]
        apply_pca_reduction: If True and PCA is fitted, reduce dimensions
        
    Returns:
        np.ndarray of shape (AREA_PCA_DIM,) or (AREA_RAW_DIM,)
    """
    model = get_area_model()
    if isinstance(img_or_tensor, torch.Tensor):
        tensor = img_or_tensor
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        tensor = tensor.to(_device)
    else:
        tensor = _preprocess_pil(img_or_tensor).unsqueeze(0).to(_device)

    with torch.no_grad():
        desc = model(tensor)  # [1, 8448]

    desc = desc.cpu().numpy().squeeze()  # (8448,)

    if apply_pca_reduction and _pca_model is not None:
        desc = apply_pca(desc.reshape(1, -1)).squeeze()

    return desc


def batch_extract_area(images_or_tensors, batch_size=32, apply_pca_reduction=False):
    """Batch extract Area descriptors from a list of PIL images or a 4D torch.Tensor [N, 3, H, W].
    
    Args:
        images_or_tensors: List of PIL Images OR torch.Tensor of shape [N, 3, H, W]
        batch_size: Batch size for inference (default 32)
        apply_pca_reduction: If True and PCA is fitted, reduce dimensions
        
    Returns:
        np.ndarray of shape (N, dim) where dim is PCA_DIM or RAW_DIM
    """
    model = get_area_model()
    all_descs = []

    is_tensor = isinstance(images_or_tensors, torch.Tensor)
    total_count = images_or_tensors.shape[0] if is_tensor else len(images_or_tensors)

    for i in range(0, total_count, batch_size):
        if is_tensor:
            tensors = images_or_tensors[i:i + batch_size].to(_device)
        else:
            batch = images_or_tensors[i:i + batch_size]
            tensors = torch.stack([_preprocess_pil(img) for img in batch]).to(_device)

        with torch.no_grad():
            descs = model(tensors)  # [B, 8448]

        all_descs.append(descs.cpu().numpy())

        if _device == 'mps' and (i // batch_size) % 10 == 0:
            torch.mps.empty_cache()

    result = np.vstack(all_descs)

    if apply_pca_reduction and _pca_model is not None:
        result = apply_pca(result)

    return result


def area_similarity(desc1, desc2):
    """Cosine similarity between two L2-normalized descriptors."""
    return float(np.dot(desc1, desc2))






def fit_pca(descriptors, n_components=AREA_PCA_DIM, whiten=True):
    """Fit PCA model on a matrix of descriptors.
    
    Args:
        descriptors: np.ndarray of shape (N, AREA_RAW_DIM)
        n_components: Target dimensionality
        whiten: Whether to whiten (recommended for retrieval)
        
    Returns:
        Fitted PCA object
    """
    global _pca_model
    from sklearn.decomposition import PCA

    print(f"[AREA-PCA] Fitting PCA: {descriptors.shape[1]} -> {n_components} dims "
          f"on {descriptors.shape[0]} samples...")

    pca = PCA(n_components=n_components, whiten=whiten)
    pca.fit(descriptors)

    explained = pca.explained_variance_ratio_.sum()
    print(f"[AREA-PCA] PCA fitted. Explained variance: {explained:.4f} "
          f"({explained*100:.1f}%)")

    _pca_model = pca
    return pca


def apply_pca(descriptors):
    """Apply fitted PCA to descriptors, then L2-normalize.
    
    Args:
        descriptors: np.ndarray of shape (N, AREA_RAW_DIM) or (AREA_RAW_DIM,)
        
    Returns:
        np.ndarray of shape (N, AREA_PCA_DIM), L2-normalized
    """
    if _pca_model is None:
        raise RuntimeError("PCA not fitted. Call fit_pca() first or load_pca().")

    single = descriptors.ndim == 1
    if single:
        descriptors = descriptors.reshape(1, -1)

    reduced = _pca_model.transform(descriptors)


    norms = np.linalg.norm(reduced, axis=1, keepdims=True)
    norms[norms == 0] = 1
    reduced = reduced / norms

    if single:
        reduced = reduced.squeeze(0)

    return reduced


def save_pca(path):
    """Save fitted PCA model to disk."""
    if _pca_model is None:
        raise RuntimeError("No PCA model to save.")
    with open(path, 'wb') as f:
        pickle.dump(_pca_model, f)
    print(f"[AREA-PCA] Saved PCA model to {path}")


def load_pca(path):
    """Load PCA model from disk."""
    global _pca_model
    with open(path, 'rb') as f:
        _pca_model = pickle.load(f)
    print(f"[AREA-PCA] Loaded PCA model from {path} "
          f"(components: {_pca_model.n_components_})")
    return _pca_model









if __name__ == "__main__":



    model = get_area_model()
    print(f"Model loaded. feat_dim = {model.feat_dim}")


    dummy = Image.new('RGB', (256, 256), color=(128, 64, 32))
    desc = extract_area_descriptor(dummy, apply_pca_reduction=False)
    print(f"Raw descriptor shape: {desc.shape}")
    print(f"L2 norm: {np.linalg.norm(desc):.4f}")


    descs = batch_extract_area([dummy, dummy], batch_size=2)
    print(f"Batch descriptor shape: {descs.shape}")


    fake_data = np.random.randn(600, AREA_RAW_DIM).astype(np.float32)
    fit_pca(fake_data, n_components=512)
    reduced = apply_pca(desc)
    print(f"PCA-reduced descriptor shape: {reduced.shape}")
    print(f"PCA-reduced L2 norm: {np.linalg.norm(reduced):.4f}")

    print("All tests passed!")
