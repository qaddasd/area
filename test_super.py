import os
import sys
os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

import tkinter as tk
from tkinter import filedialog, ttk
from PIL import Image, ImageTk, ImageDraw, ImageOps, ImageFilter
import numpy as np
import torch
import re
import math
from collections import defaultdict
import concurrent.futures
import itertools
import threading
import queue
import time
import json
import random
import glob
import cv2
try:
    import kornia.feature as KF
except ImportError:
    KF = None
import asyncio
import aiohttp
import tkintermapview
import webbrowser
from area_utils import (
    get_area_model, extract_area_descriptor,
    area_similarity, batch_extract_area,
    fit_pca, apply_pca, save_pca, load_pca,
    AREA_RAW_DIM, AREA_PCA_DIM
)
import gc



device = 'mps' if torch.backends.mps.is_available() else ('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

extractor_lock = threading.Lock()

try:
    from area_3r_utils import get_area_3r_model, get_area_3r_matches
    AREA_3R_AVAILABLE = True
except ImportError:
    AREA_3R_AVAILABLE = False
    print("[Area-3R] area_3r_utils.py not found. Area-3R matching disabled.")


try:
    from area_hub import AreaHub, create_bundle, extract_bundle
    HUB_AVAILABLE = True
except ImportError:
    HUB_AVAILABLE = False
    print("[HUB] area_hub.py not found. Community sharing disabled.")

area_3r_model_instance = None
area_3r_lock = threading.Lock()

def get_lazy_area_3r():
    global area_3r_model_instance
    with area_3r_lock:
        if area_3r_model_instance is None:
            area_3r_model_instance = get_area_3r_model()
    return area_3r_model_instance




_potential_dir = "/Volumes/Expansion/area"
if os.path.exists(_potential_dir):
    DATA_DIR = _potential_dir
else:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "area_data")

AREA_PARTS_DIR = os.path.join(DATA_DIR, "area_parts")
EMB_CSV = os.path.join(DATA_DIR, "embeddings_index.csv")
COMPACT_INDEX_DIR = os.path.join(DATA_DIR, "index")
COMPACT_DESCS_PATH = os.path.join(COMPACT_INDEX_DIR, "area_descriptors.npy")
COMPACT_META_PATH = os.path.join(COMPACT_INDEX_DIR, "metadata.npz")
COMPACT_INFO_PATH = os.path.join(COMPACT_INDEX_DIR, "index_info.txt")


for d in [DATA_DIR, AREA_PARTS_DIR, COMPACT_INDEX_DIR]:
    os.makedirs(d, exist_ok=True)


MAX_PANOID_WORKERS = 16
MAX_HEADING_WORKERS = 1
MAX_DOWNLOAD_WORKERS = 100
MAX_MATCH_WORKERS = 6
EARLY_EXIT_INLIER_THRESHOLD = 300


_mps_cleanup_counter = 0
_mps_cleanup_lock = threading.Lock()


def aggressive_mps_cleanup(force=False):
    global _mps_cleanup_counter
    with _mps_cleanup_lock:
        _mps_cleanup_counter += 1
        should_clean = force or (_mps_cleanup_counter % 100 == 0)
    if not should_clean:
        return
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    gc.collect()
    if force or (_mps_cleanup_counter % 50 == 0):
        import subprocess
        try:
            subprocess.run(
                ['find', '/private/var/folders', '-name', 'mpsgraph-*', '-type', 'f', '-mmin', '+1', '-delete'],
                capture_output=True, timeout=10
            )
        except Exception:
            pass




def pil_to_tensor(im):
    return torch.from_numpy(np.array(im.convert('RGB'))).float().permute(2, 0, 1).unsqueeze(0).div(255.0).to(device)

def tensor_to_pil(t):
    t = t.squeeze(0).cpu().clamp(0, 1).mul(255).add_(0.5).to(torch.uint8).permute(1, 2, 0).numpy()
    if t.shape[2] == 1:
        t = t.squeeze(2)
    return Image.fromarray(t)



def draw_matches(img1, img2, kp1, kp2, matches=None, color=(0, 255, 0)):
    w1, h1 = img1.size
    w2, h2 = img2.size
    new_h = max(h1, h2)
    result = Image.new("RGB", (w1 + w2, new_h), (255, 255, 255))
    result.paste(img1, (0, 0))
    result.paste(img2, (w1, 0))
    draw = ImageDraw.Draw(result)
    if matches is None:
        return result
    if isinstance(matches, np.ndarray) and matches.ndim == 2 and matches.shape[1] == 2:
        for i in range(len(matches)):
            idx0, idx1 = matches[i]
            p1, p2 = kp1[idx0], kp2[idx1]
            draw.line(((p1[0], p1[1]), (p2[0] + w1, p2[1])), fill=color, width=1)
    elif isinstance(matches, np.ndarray) and matches.ndim == 1:
        for idx, m in enumerate(matches):
            if m > -1:
                x1, y1 = kp1[idx]
                x2, y2 = kp2[m]
                draw.line(((x1, y1), (x2 + w1, y2)), fill=color, width=1)
    return result



IMGX = 4
IMGY = 2

def _panoids_url(lat, lon):
    url = "https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{0:}!4d{1:}!2d50!3m10!2m2!1sen!2sGB!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    return url.format(lat, lon)

def panoids_from_response(text):
    matches = re.findall(r'"([A-Za-z0-9_-]{22})"', text)
    out = []
    for panoid in matches:
        latlon = re.findall(r'"' + panoid + r'".+?\[null,null,(-?\d+\.\d+),(-?\d+\.\d+)', text)
        if latlon:
            lat, lon = map(float, latlon[0])
        else:
            lat, lon = None, None
        out.append({"panoid": panoid, "lat": lat, "lon": lon})
    filtered = []
    seen = set()
    for p in out:
        if p['panoid'] not in seen:
            seen.add(p['panoid'])
            filtered.append(p)
    return filtered

def tiles_info(panoid):
    image_url = "https://streetviewpixels-pa.googleapis.com/v1/tile?panoid={0:}&x={1:}&y={2:}&zoom=2&cb_client=maps_sv.tactile"
    coord = list(itertools.product(range(IMGX), range(IMGY)))
    tiles = [(x, y, "%s_%dx%d.jpg" % (panoid, x, y), image_url.format(panoid, x, y)) for x, y in coord]
    return tiles

async def download_tile_aiohttp(session, x, y, fname, url):
    for attempt in range(2):
        try:
            async with session.get(url.replace("http://", "https://"), timeout=10) as response:
                if response.status == 200:
                    data = await response.read()
                    return x, y, data
        except Exception:
            await asyncio.sleep(2)
    return x, y, None

def fetch_single_pano(panoid, width=2048, height=1024):
    """Fast single-request download of full 360 panorama equirectangular image."""
    import urllib.request
    import io
    url = f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail?panoid={panoid}&cb_client=maps_sv.tactile&w={width}&h={height}"
    hdr = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for _ in range(2):
        try:
            req = urllib.request.Request(url, headers=hdr)
            with urllib.request.urlopen(req, timeout=8) as resp:
                if resp.status == 200:
                    data = resp.read()
                    if len(data) > 5000:
                        return Image.open(io.BytesIO(data))
        except Exception:
            pass
    return None

def download_tiles(tiles, status_callback=None, max_workers=64):
    total = len(tiles)
    results = {}
    async def main():
        connector = aiohttp.TCPConnector(limit=max_workers)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            tasks = []
            for i, (x, y, fname, url) in enumerate(tiles):
                tasks.append(download_tile_aiohttp(session, x, y, fname, url))
            for idx, coro in enumerate(asyncio.as_completed(tasks), 1):
                x, y, data = await coro
                if data:
                    results[(x, y)] = data
                if status_callback:
                    status_callback(idx, total)
    asyncio.run(main())
    return results

def stitch_tiles(tiles_data):
    tile_w, tile_h = 512, 512
    import io
    pano_np = np.zeros((IMGY * tile_h, IMGX * tile_w, 3), dtype=np.uint8)
    for (x, y), data in tiles_data.items():
        try:
            tile = Image.open(io.BytesIO(data))
            tile_np = np.array(tile)
            th, tw, _ = tile_np.shape
            pano_np[y*tile_h:y*tile_h+th, x*tile_w:x*tile_w+tw] = tile_np
            tile.close()
        except Exception:
            continue
    return Image.fromarray(pano_np)




def haversine(p1, p2):
    R = 6371
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

def grid_points(center, radius, resolution):
    lat, lon = center
    top_left = (lat - radius / 70, lon + radius / 70)
    bottom_right = (lat + radius / 70, lon - radius / 70)
    lat_diff = top_left[0] - bottom_right[0]
    lon_diff = top_left[1] - bottom_right[1]
    test_points = list(itertools.product(range(resolution + 1), range(resolution + 1)))
    test_points = [
        (bottom_right[0] + x * lat_diff / resolution, bottom_right[1] + y * lon_diff / resolution)
        for (x, y) in test_points
    ]
    test_points = [p for p in test_points if haversine(p, center) <= radius]
    return test_points

def get_panoids(points, status_callback=None, max_workers=64):
    import csv
    async def fetch_one(session, idx, lat, lon, max_attempts=12):
        url = _panoids_url(lat, lon)
        for attempt in range(max_attempts):
            try:
                async with session.get(url, timeout=120) as resp:
                    status = resp.status
                    text = await resp.text()
                    if status == 429:
                        await asyncio.sleep(1)
                        continue
                    elif status != 200:
                        continue
                    pans = panoids_from_response(text)
                    if not pans:
                        return []
                    return pans
            except asyncio.TimeoutError:
                await asyncio.sleep(0.5)
            except Exception as e:
                await asyncio.sleep(0.5)
        return []

    async def main():
        connector = aiohttp.TCPConnector(limit=max_workers)
        async with aiohttp.ClientSession(connector=connector) as session:
            tasks = []
            for idx, (lat, lon) in enumerate(points):
                task = asyncio.create_task(fetch_one(session, idx, lat, lon))
                tasks.append(task)
            results = []
            for idx, task in enumerate(asyncio.as_completed(tasks), 1):
                pans = await task
                results.extend(pans)
                if status_callback:
                    status_callback(idx, len(points))
            return results

    panoids_raw = asyncio.run(main())
    already = set()
    filtered = []
    for pan in panoids_raw:
        if pan['panoid'] not in already:
            already.add(pan['panoid'])
            filtered.append(pan)
    print(f"[SUMMARY] Fetched {len(points)} grid points, found {len(filtered)} unique panoids.")
    return filtered

def generate_circle_points(center_lat, center_lon, radius_km, num_points=36):
    points = []
    R = 6371.0
    lat_rad = math.radians(center_lat)
    lon_rad = math.radians(center_lon)
    angular_dist = radius_km / R
    for i in range(num_points):
        bearing = math.radians(i * (360 / num_points))
        new_lat = math.asin(math.sin(lat_rad) * math.cos(angular_dist) +
                            math.cos(lat_rad) * math.sin(angular_dist) * math.cos(bearing))
        new_lon = lon_rad + math.atan2(math.sin(bearing) * math.sin(angular_dist) * math.cos(lat_rad),
                                       math.cos(angular_dist) - math.sin(lat_rad) * math.sin(new_lat))
        points.append((math.degrees(new_lat), math.degrees(new_lon)))
    return points






def get_projection_base_dirs(fov_deg, out_hw):
    fov = math.radians(fov_deg)
    out_h, out_w = out_hw
    cx, cy = out_w / 2.0, out_h / 2.0
    fx = fy = (out_w / 2.0) / math.tan(fov / 2.0)
    xx, yy = torch.meshgrid(
        torch.arange(out_w, device=device, dtype=torch.float32),
        torch.arange(out_h, device=device, dtype=torch.float32),
        indexing='xy'
    )
    x = (xx - cx) / fx
    y = (yy - cy) / fy
    z = torch.ones_like(x)
    dirs = torch.stack([x, -y, z], dim=-1)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)
    return dirs.reshape(-1, 3).T

def equirectangular_to_rectilinear_torch(pano_tensor, fov_deg=90, out_hw=(400, 400), yaw_deg=0, pitch_deg=0, base_dirs=None):
    _, _, h, w = pano_tensor.shape
    out_h, out_w = out_hw
    if isinstance(yaw_deg, (float, int)):
        yaws = torch.tensor([yaw_deg], device=device, dtype=torch.float32)
    elif isinstance(yaw_deg, list):
        yaws = torch.tensor(yaw_deg, device=device, dtype=torch.float32)
    else:
        yaws = yaw_deg.to(device).float()
    B = len(yaws)
    yaws_rad = torch.deg2rad(yaws)
    cos_vals = torch.cos(yaws_rad)
    sin_vals = torch.sin(yaws_rad)
    zeros = torch.zeros_like(cos_vals)
    ones = torch.ones_like(cos_vals)
    row1 = torch.stack([cos_vals, zeros, sin_vals], dim=1)
    row2 = torch.stack([zeros, ones, zeros], dim=1)
    row3 = torch.stack([-sin_vals, zeros, cos_vals], dim=1)
    R = torch.stack([row1, row2, row3], dim=1)
    if base_dirs is None:
        base_dirs = get_projection_base_dirs(fov_deg, out_hw)
    dirs = torch.matmul(R, base_dirs.unsqueeze(0))
    dirs = dirs.permute(0, 2, 1)
    x = dirs[:, :, 0]
    y = dirs[:, :, 1]
    z = dirs[:, :, 2]
    lon = torch.atan2(x, z)
    lat = torch.asin(y.clamp(-1+1e-7, 1-1e-7))
    grid_x = lon / math.pi
    grid_y = -lat / (math.pi / 2.0)
    grid = torch.stack([grid_x, grid_y], dim=-1).reshape(B, out_h, out_w, 2)
    pano_batch = pano_tensor.expand(B, -1, -1, -1)
    out = torch.nn.functional.grid_sample(pano_batch, grid, mode='bilinear', align_corners=True)
    return out

def equirectangular_to_rectilinear(pano_img, fov_deg=90, out_hw=(400, 400), yaw_deg=0, pitch_deg=0):
    pano_tensor = pil_to_tensor(pano_img)
    out_tensor = equirectangular_to_rectilinear_torch(pano_tensor, fov_deg, out_hw, yaw_deg, pitch_deg)
    return tensor_to_pil(out_tensor)




_compact_cache = None

def parse_emb_path(emb_path):
    """Extract panoid and heading from path like '/path/to/PANOID_HEADING.npz'."""
    filename = os.path.basename(emb_path)
    name = filename.replace('.npz', '')
    parts = name.rsplit('_', 1)
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return None, None


INDEX_TARGET_DIM = 1024

def build_compact_index():
    """Build compact index from part files + CSV coordinates.
    
    Auto-applies PCA if descriptors are high-dimensional (e.g., 8448 from Area-loc).
    """
    global _compact_cache
    import glob
    os.makedirs(COMPACT_INDEX_DIR, exist_ok=True)

    area_part_pattern = os.path.join(AREA_PARTS_DIR, "area_part_*.npz")
    part_files = sorted(glob.glob(area_part_pattern))
    part_files = sorted(set(part_files))

    if not part_files:
        print(f"[INDEX] ERROR: No part files found")
        return False

    print(f"[INDEX] Found {len(part_files)} part files")


    total = 0
    raw_dim = None
    for pf in part_files:
        data = np.load(pf, allow_pickle=True)
        total += len(data['paths'])
        if raw_dim is None:
            raw_dim = data['descriptors'].shape[1]
        del data
    
    print(f"[INDEX] Total entries: {total}, raw descriptor dim: {raw_dim}")


    needs_pca = raw_dim > INDEX_TARGET_DIM
    final_dim = INDEX_TARGET_DIM if needs_pca else raw_dim
    
    if needs_pca:
        print(f"[INDEX] Will apply PCA: {raw_dim} -> {final_dim}")
        

        MAX_PCA_SAMPLES = 100_000
        pca_path = os.path.join(COMPACT_INDEX_DIR, "area_pca.pkl")
        
        print(f"[INDEX] Collecting subsample for PCA fitting (max {MAX_PCA_SAMPLES})...")
        pca_samples = []
        pca_count = 0
        for pf in part_files:
            if pca_count >= MAX_PCA_SAMPLES:
                break
            data = np.load(pf, allow_pickle=True)
            descs = data['descriptors']
            remaining = MAX_PCA_SAMPLES - pca_count
            pca_samples.append(descs[:remaining])
            pca_count += len(descs[:remaining])
            del data
        
        pca_matrix = np.vstack(pca_samples)
        del pca_samples
        final_dim = min(final_dim, pca_matrix.shape[0])
        print(f"[INDEX] Fitting PCA on {pca_matrix.shape[0]} samples (target {final_dim} dims)...")
        
        from sklearn.decomposition import PCA
        pca = PCA(n_components=final_dim, whiten=True)
        pca.fit(pca_matrix)
        explained = pca.explained_variance_ratio_.sum()
        print(f"[INDEX] PCA fitted. Explained variance: {explained*100:.1f}%")
        del pca_matrix
        

        import pickle
        with open(pca_path, 'wb') as f:
            pickle.dump(pca, f)
        print(f"[INDEX] Saved PCA model to {pca_path}")
        

        try:
            from area_utils import load_pca as _load_pca
            _load_pca(pca_path)
        except Exception:
            pass
    else:
        pca = None
        print(f"[INDEX] Descriptors already {raw_dim}-dim, no PCA needed")


    print(f"[INDEX] Loading and merging {len(part_files)} files...")
    all_descs = np.zeros((total, final_dim), dtype=np.float32)
    all_paths = []
    all_embedded_lats = []
    all_embedded_lons = []

    import time
    idx = 0
    t0 = time.time()
    for i, pf in enumerate(part_files):
        data = np.load(pf, allow_pickle=True)
        n = len(data['paths'])
        
        descs = data['descriptors']
        

        if needs_pca and descs.shape[1] > final_dim:
            descs = pca.transform(descs).astype(np.float32)
            norms = np.linalg.norm(descs, axis=1, keepdims=True)
            norms[norms == 0] = 1
            descs = descs / norms
        elif descs.shape[1] != final_dim:
            print(f"[INDEX] WARNING: Skipping {pf} — dim {descs.shape[1]} != expected {final_dim}")
            del data
            continue
        
        all_descs[idx:idx+n] = descs
        all_paths.extend(data['paths'].tolist())

        if 'lats' in data and 'lons' in data:
            all_embedded_lats.extend(data['lats'].tolist())
            all_embedded_lons.extend(data['lons'].tolist())
        else:
            all_embedded_lats.extend([0.0] * n)
            all_embedded_lons.extend([0.0] * n)

        idx += n
        del data, descs
        if (i+1) % 100 == 0:
            print(f"  Loaded {i+1}/{len(part_files)} ({idx} entries) [{time.time()-t0:.0f}s]")


    if idx < total:
        all_descs = all_descs[:idx]
        print(f"[INDEX] Trimmed to {idx} entries (skipped some incompatible files)")
        total = idx

    print(f"[INDEX] Loaded all {idx} entries in {time.time()-t0:.1f}s")


    print(f"[INDEX] Loading coordinates from {EMB_CSV}...")
    csv_locations = {}
    csv_full_locations = {}
    if os.path.exists(EMB_CSV):
        with open(EMB_CSV, 'r') as f:
            for line in f:
                parts = line.strip().split(',')
                if len(parts) >= 3:
                    try:
                        lat, lon = float(parts[1]), float(parts[2])
                        csv_full_locations[parts[0]] = (lat, lon)
                        csv_locations[os.path.basename(parts[0])] = (lat, lon)
                    except ValueError:
                        pass
    print(f"[INDEX] CSV has {len(csv_locations)} location entries")


    lats = np.zeros(idx, dtype=np.float32)
    lons = np.zeros(idx, dtype=np.float32)
    headings = np.zeros(idx, dtype=np.int16)
    panoids = []
    valid_mask = np.zeros(idx, dtype=bool)
    matched = 0

    for i, path in enumerate(all_paths):
        filename = os.path.basename(path)
        name = filename.replace('.npz', '')
        parts_split = name.rsplit('_', 1)
        panoid = parts_split[0] if len(parts_split) == 2 else None
        try:
            heading = int(parts_split[1]) if len(parts_split) == 2 else 0
        except ValueError:
            heading = 0

        panoids.append(panoid or "")
        headings[i] = heading

        emb_lat = all_embedded_lats[i]
        emb_lon = all_embedded_lons[i]

        if emb_lat != 0 or emb_lon != 0:
            lats[i], lons[i] = emb_lat, emb_lon
            valid_mask[i] = True
            matched += 1
        else:
            loc = csv_full_locations.get(path) or csv_locations.get(filename)
            if loc:
                lats[i], lons[i] = loc
                valid_mask[i] = True
                matched += 1
        if (i + 1) % 200000 == 0:
            print(f"  Matching {i+1}/{idx}... ({matched} matched)")

    print(f"[INDEX] Matched {matched}/{idx} paths to coordinates")

    valid_idx = np.where(valid_mask)[0]
    print(f"[INDEX] Keeping {len(valid_idx)} entries with valid coordinates")


    print("[INDEX] Filtering valid descriptors...")
    descs_valid = all_descs[valid_idx].copy()
    del all_descs

    print("[INDEX] Normalizing in-place...")
    norms = np.linalg.norm(descs_valid, axis=1, keepdims=True)
    norms[norms == 0] = 1
    descs_valid /= norms
    del norms

    print("[INDEX] Saving descriptors...")
    np.save(COMPACT_DESCS_PATH, descs_valid)
    del descs_valid

    print("[INDEX] Saving metadata...")
    np.savez_compressed(COMPACT_META_PATH,
        lats=lats[valid_idx], lons=lons[valid_idx], headings=headings[valid_idx],
        panoids=np.array([panoids[i] for i in valid_idx], dtype=object),
        paths=np.array([all_paths[i] for i in valid_idx], dtype=object)
    )

    COMPACT_INFO_PATH = os.path.join(COMPACT_INDEX_DIR, "index_info.txt")
    size_d = os.path.getsize(COMPACT_DESCS_PATH) / 1024 / 1024
    size_m = os.path.getsize(COMPACT_META_PATH) / 1024 / 1024
    with open(COMPACT_INFO_PATH, 'w') as f:
        f.write(f"Compact Index Info\n")
        f.write(f"Built: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Entries: {len(valid_idx)}\n")
        f.write(f"Descriptor dim: {final_dim}\n")
        f.write(f"Raw dim (pre-PCA): {raw_dim}\n")
        f.write(f"Total: {size_d + size_m:.1f} MB\n")

    print(f"\n[INDEX] [OK] Saved compact index:")
    print(f"  Descriptors: {COMPACT_DESCS_PATH} ({size_d:.1f} MB)")
    print(f"  Metadata: {COMPACT_META_PATH} ({size_m:.1f} MB)")
    print(f"  Descriptor dim: {final_dim} (from raw {raw_dim})")
    print(f"  Total: {size_d + size_m:.1f} MB")

    _compact_cache = None  # Force reload
    return True



def load_compact_index():
    """Load compact index into memory. Returns (descriptors, metadata_dict)."""
    global _compact_cache
    if _compact_cache is not None:
        return _compact_cache
    if not os.path.exists(COMPACT_DESCS_PATH) or not os.path.exists(COMPACT_META_PATH):
        print("[INDEX] ERROR: Compact index not found. Run create mode first.")
        return None, None
    print("[INDEX] Loading compact index (memory-mapped)...")
    t0 = time.time()

    descs = np.load(COMPACT_DESCS_PATH, mmap_mode='r')
    meta = np.load(COMPACT_META_PATH, allow_pickle=True)
    metadata = {
        'lats': meta['lats'].copy(), 'lons': meta['lons'].copy(),
        'headings': meta['headings'].copy(),
        'panoids': meta['panoids'], 'paths': meta['paths'],
    }
    del meta
    elapsed = time.time() - t0
    print(f"[INDEX] Loaded {len(descs)} entries ({descs.shape[1]}-dim) in {elapsed:.1f}s [mmap]")
    _compact_cache = (descs, metadata)
    return descs, metadata


def search_compact_index(query_desc, center, radius_km, top_k=500):
    """Search: radius filter → chunked dot-product → panoid dedup → top-K."""
    descs, metadata = load_compact_index()
    if descs is None:
        return []
    t0 = time.time()
    lat1 = np.radians(center[0])
    lon1 = np.radians(center[1])
    lat2 = np.radians(metadata['lats'])
    lon2 = np.radians(metadata['lons'])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    distances = 6371 * 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    radius_mask = distances <= radius_km
    radius_indices = np.where(radius_mask)[0]
    n_in_radius = len(radius_indices)
    print(f"[INDEX] Radius filter: {n_in_radius}/{len(descs)} in {radius_km}km ({time.time()-t0:.2f}s)")
    if n_in_radius == 0:
        return []

    t1 = time.time()
    query_norm = query_desc / (np.linalg.norm(query_desc) + 1e-8)
    query_norm = query_norm.astype(np.float32)


    CHUNK_SIZE = 100_000
    top_scores = np.full(top_k * 2, -np.inf, dtype=np.float32)  # keep 2x for panoid dedup
    top_indices = np.zeros(top_k * 2, dtype=np.int64)

    for chunk_start in range(0, n_in_radius, CHUNK_SIZE):
        chunk_end = min(chunk_start + CHUNK_SIZE, n_in_radius)
        chunk_idx = radius_indices[chunk_start:chunk_end]
        chunk_descs = np.array(descs[chunk_idx], dtype=np.float32)
        chunk_sims = chunk_descs @ query_norm
        del chunk_descs

        combined_scores = np.concatenate([top_scores, chunk_sims])
        combined_indices = np.concatenate([top_indices, chunk_idx])
        k = min(top_k * 2, len(combined_scores))
        best_k = np.argsort(combined_scores)[::-1][:k]
        top_scores = combined_scores[best_k]
        top_indices = combined_indices[best_k]


    seen_panoids = {}
    for gi, score in zip(top_indices, top_scores):
        if score == -np.inf:
            break
        pid = str(metadata['panoids'][gi])
        if pid not in seen_panoids or score > seen_panoids[pid]['score']:
            seen_panoids[pid] = {
                'panoid': pid,
                'heading': int(metadata['headings'][gi]),
                'lat': float(metadata['lats'][gi]),
                'lon': float(metadata['lons'][gi]),
                'score': float(score),
                'path': str(metadata['paths'][gi]),
            }

    results = sorted(seen_panoids.values(), key=lambda x: x['score'], reverse=True)[:top_k]
    print(f"[INDEX] Search: top-{len(results)} unique panoids in {time.time()-t1:.2f}s (best: {results[0]['score']:.3f})")
    return results



class ProgressTracker:
    def __init__(self, total_items, estimate_storage=False, embeddings_per_item=4, avg_bytes_per_embedding=2560):
        self.total = total_items
        self.start_time = time.time()
        self.processed = 0
        self.estimate_storage = estimate_storage
        self.embeddings_per_item = embeddings_per_item
        self.avg_bytes_per_embedding = avg_bytes_per_embedding

    def update(self, current_count):
        self.processed = current_count

    def get_status(self):
        elapsed = time.time() - self.start_time
        if elapsed > 0.5 and self.processed > 0:
            speed = self.processed / elapsed
            remaining = self.total - self.processed
    
            eta_seconds = remaining / speed if speed > 0 else 0
            eta_str = time.strftime("%H:%M:%S", time.gmtime(eta_seconds)) if eta_seconds > 3600 else time.strftime("%M:%S", time.gmtime(eta_seconds))
            speed_fmt = f"{speed:.2f}"
        else:
            eta_str = "calculating..."
            speed_fmt = "--"
        percent = int((self.processed / self.total) * 100) if self.total > 0 else 0
        storage_str = ""
        if self.estimate_storage:
            total_bytes = self.total * self.embeddings_per_item * self.avg_bytes_per_embedding
            if total_bytes < 1024 * 1024:
                storage_str = f" | Storage: {total_bytes / 1024:.1f} KB"
            elif total_bytes < 1024 * 1024 * 1024:
                storage_str = f" | Storage: {total_bytes / (1024 * 1024):.1f} MB"
            else:
                storage_str = f" | Storage: {total_bytes / (1024 * 1024 * 1024):.2f} GB"
        return f"{self.processed}/{self.total} ({percent}%) | {speed_fmt} it/s | ETA: {eta_str}{storage_str}"




class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command, width=200, height=44,
                 corner_radius=10, bg_color='#ffffff', hover_color='#e2e8f0',
                 pressed_color='#cbd5e1', text_color='#0a0c10', border_color=None,
                 font=('Inter', 10, 'bold')):
        try:
            parent_bg = parent.cget('bg')
        except:
            parent_bg = '#0a0c10'
        super().__init__(parent, width=width, height=height,
                        highlightthickness=0, bg=parent_bg, cursor='hand2')
        self.command = command
        self.bg_color = bg_color
        self.hover_color = hover_color
        self.pressed_color = pressed_color
        self.text_color = text_color
        self.border_color = border_color
        self.corner_radius = corner_radius
        self.width = width
        self.height = height
        self._text = text
        self._font = font
        self._draw_button(bg_color)
        self.bind('<Enter>', self._on_hover)
        self.bind('<Leave>', self._on_leave)
        self.bind('<Button-1>', self._on_press)
        self.bind('<ButtonRelease-1>', self._on_release)

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
                  x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _draw_button(self, color):
        self.delete('all')
        outline_kw = {'outline': self.border_color, 'width': 1} if self.border_color else {'outline': ''}
        self._create_rounded_rect(2, 2, self.width-2, self.height-2,
                                  self.corner_radius, fill=color, **outline_kw)
        self.create_text(self.width/2, self.height/2, text=self._text,
                        fill=self.text_color, font=self._font)

    def _on_hover(self, event):
        if not getattr(self, '_disabled', False): self._draw_button(self.hover_color)
    def _on_leave(self, event):
        if not getattr(self, '_disabled', False): self._draw_button(self.bg_color)
    def _on_press(self, event):
        if not getattr(self, '_disabled', False): self._draw_button(self.pressed_color)
    def _on_release(self, event):
        if not getattr(self, '_disabled', False):
            self._draw_button(self.hover_color)
            if self.command: self.command()

    def configure(self, **kwargs):
        if 'text' in kwargs: self._text = kwargs['text']
        if 'command' in kwargs: self.command = kwargs['command']
        if 'state' in kwargs:
            if kwargs['state'] == 'disabled':
                self._disabled = True
                self._draw_button('#181c24')
            else:
                self._disabled = False
                self._draw_button(self.bg_color)
            return
        self._draw_button(self.bg_color)
    config = configure


class RoundedEntry(tk.Canvas):
    def __init__(self, parent, textvariable=None, width=200, height=36, corner_radius=8,
                 bg_color='#141720', text_color='#f1f5f9', border_color='#232834',
                 focus_color='#475569', font=('Inter', 10), **kwargs):
        try:
            parent_bg = parent.cget('bg')
        except:
            parent_bg = '#0a0c10'
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg=parent_bg)
        self.corner_radius = corner_radius
        self.bg_color = bg_color
        self.border_color = border_color
        self.focus_color = focus_color
        self.width = width
        self.height = height
        self.entry = tk.Entry(self, textvariable=textvariable, font=font,
                             bg=bg_color, fg=text_color, borderwidth=0,
                             insertbackground='#ffffff', highlightthickness=0)
        self._draw_background(self.border_color)
        self.create_window(width/2, height/2, window=self.entry, width=width-20, height=height-8)
        self.entry.bind('<FocusIn>', lambda e: self._draw_background(self.focus_color))
        self.entry.bind('<FocusOut>', lambda e: self._draw_background(self.border_color))


        self.entry.bind('<Control-v>', self._paste)
        self.entry.bind('<Control-V>', self._paste)
        self.entry.bind('<Control-KeyPress-Cyrillic_em>', self._paste)
        self.entry.bind('<Control-KeyPress-Cyrillic_EM>', self._paste)

        self.entry.bind('<Control-c>', self._copy)
        self.entry.bind('<Control-C>', self._copy)
        self.entry.bind('<Control-KeyPress-Cyrillic_es>', self._copy)
        self.entry.bind('<Control-KeyPress-Cyrillic_ES>', self._copy)

        self.entry.bind('<Control-x>', self._cut)
        self.entry.bind('<Control-X>', self._cut)
        self.entry.bind('<Control-KeyPress-Cyrillic_che>', self._cut)
        self.entry.bind('<Control-KeyPress-Cyrillic_CHE>', self._cut)

        self.entry.bind('<Control-a>', self._select_all)
        self.entry.bind('<Control-A>', self._select_all)
        self.entry.bind('<Control-KeyPress-Cyrillic_ef>', self._select_all)
        self.entry.bind('<Control-KeyPress-Cyrillic_EF>', self._select_all)

    def _draw_background(self, border_col):
        super().delete('bg')
        points = [1+self.corner_radius, 1, self.width-1-self.corner_radius, 1, self.width-1, 1,
                  self.width-1, 1+self.corner_radius, self.width-1, self.height-1-self.corner_radius,
                  self.width-1, self.height-1, self.width-1-self.corner_radius, self.height-1,
                  1+self.corner_radius, self.height-1, 1, self.height-1, 1, self.height-1,
                  1, 1+self.corner_radius, 1, 1]
        self.create_polygon(points, smooth=True, fill=self.bg_color,
                          outline=border_col, width=1, tags='bg')
        self.tag_lower('bg')

    def get(self): return self.entry.get()
    def insert(self, *args): return self.entry.insert(*args)
    def delete(self, *args): return self.entry.delete(*args)

    def _paste(self, event=None):
        try:
            text = self.entry.clipboard_get()
        except tk.TclError:
            text = None
        
        if text:
            cleaned = text.strip()

            parts = []
            if ',' in cleaned:
                parts = [p.strip() for p in cleaned.split(',')]
            else:
                parts = cleaned.split()
            
            if len(parts) == 2:
                try:
                    lat_val = float(parts[0])
                    lon_val = float(parts[1])
                    curr = self.master
                    gui = None
                    while curr:
                        if hasattr(curr, 'lat_var') and hasattr(curr, 'lon_var'):
                            gui = curr
                            break
                        curr = getattr(curr, 'master', None)
                    
                    if gui is not None:
                        gui.lat_var.set(lat_val)
                        gui.lon_var.set(lon_val)
                        return "break"
                except ValueError:
                    pass

            try:
                self.entry.delete(tk.SEL_FIRST, tk.SEL_LAST)
            except tk.TclError:
                pass
            self.entry.insert(tk.INSERT, text)
        return "break"

    def _copy(self, event=None):
        self.entry.event_generate("<<Copy>>")
        return "break"

    def _cut(self, event=None):
        self.entry.event_generate("<<Cut>>")
        return "break"

    def _select_all(self, event=None):
        self.entry.select_range(0, tk.END)
        self.entry.icursor(tk.END)
        return "break"


class RoundedRadio(tk.Canvas):
    def __init__(self, parent, text, variable, value, width=120, height=32,
                 bg_color='#141720', active_bg='#222836', active_border='#3b4252',
                 text_color='#94a3b8', active_text='#ffffff', font=('Inter', 10, 'bold'), command=None):
        super().__init__(parent, width=width, height=height, highlightthickness=0, bg=bg_color, cursor='hand2')
        self.variable = variable
        self.value = value
        self.command = command
        self.bg_color = bg_color
        self.active_bg = active_bg
        self.active_border = active_border
        self.text_color = text_color
        self.active_text = active_text
        self._text = text
        self._font = font
        self.btn_w = width
        self.btn_h = height
        self.bind('<Button-1>', self._on_click)
        self.variable.trace_add("write", self._update_state)
        self._update_state()

    def _on_click(self, event):
        self.variable.set(self.value)
        if self.command: self.command()

    def _create_rounded_rect(self, x1, y1, x2, y2, r, **kwargs):
        points = [x1+r, y1, x2-r, y1, x2, y1, x2, y1+r, x2, y2-r, x2, y2,
                  x2-r, y2, x1+r, y2, x1, y2, x1, y2-r, x1, y1+r, x1, y1]
        return self.create_polygon(points, smooth=True, **kwargs)

    def _update_state(self, *args):
        self.delete('all')
        is_selected = (self.variable.get() == self.value)
        if is_selected:
            self._create_rounded_rect(1, 1, self.btn_w-1, self.btn_h-1, 8,
                                      fill=self.active_bg, outline=self.active_border, width=1)
            self.create_text(self.btn_w/2, self.btn_h/2, text=self._text,
                             fill=self.active_text, font=self._font)
        else:
            self._create_rounded_rect(1, 1, self.btn_w-1, self.btn_h-1, 8,
                                      fill=self.bg_color, outline='#232834', width=1)
            self.create_text(self.btn_w/2, self.btn_h/2, text=self._text,
                             fill=self.text_color, font=self._font)





class StreetViewMatcherGUI:
    def __init__(self, master):
        self.master = master
        master.title("Area // Visual Geolocation Workstation")
        master.configure(bg='#0a0c10')
        master.geometry("1480x1050")

        self.lat_var = tk.DoubleVar(value=43.6480)
        self.lon_var = tk.DoubleVar(value=51.1722)
        self.radius_var = tk.DoubleVar(value=1.5)
        self.res_var = tk.IntVar(value=300)
        self.match_threshold = tk.IntVar(value=50)
        self.crop_fov = tk.IntVar(value=90)
        self.crop_size = tk.IntVar(value=256)
        self.crop_step = tk.IntVar(value=90)
        self.query_img_path = None
        self.mode_var = tk.StringVar(value="search")
        self.search_option_var = tk.StringVar(value="manual")
        self.hf_token_var = tk.StringVar(value=os.getenv("HF_TOKEN", ""))

        style = ttk.Style(master)
        style.theme_use('clam')
        bg_primary = '#0a0c10'
        text_primary = '#f1f5f9'
        text_muted = '#94a3b8'

        style.configure('TFrame', background=bg_primary)
        style.configure('TLabel', background=bg_primary, foreground=text_primary, font=('Inter', 10))
        style.configure('Title.TLabel', background=bg_primary, foreground='#ffffff', font=('Inter', 22, 'bold'))
        style.configure('Subtitle.TLabel', background=bg_primary, foreground=text_muted, font=('Inter', 9))
        style.configure('Section.TLabel', background=bg_primary, foreground='#64748b', font=('Inter', 8, 'bold'))
        style.configure('Horizontal.TProgressbar', background='#ffffff', troughcolor='#141720', thickness=4)
        style.configure('TButton', background='#191d28', foreground=text_primary, font=('Inter', 10), borderwidth=0)
        style.map('TButton', background=[('active', '#222836')])

        style.configure("Treeview", 
                        background="#141720", 
                        foreground="#f1f5f9", 
                        fieldbackground="#141720", 
                        rowheight=34,
                        font=('Inter', 9),
                        borderwidth=0)
        style.map("Treeview", 
                  background=[('selected', '#222836')],
                  foreground=[('selected', '#ffffff')])
        
        style.configure("Treeview.Heading", 
                        background="#191d28", 
                        foreground="#94a3b8", 
                        font=('Inter', 8, 'bold'),
                        padding=8)
        style.map("Treeview.Heading", 
                  background=[('active', '#222836')])

        paned = ttk.PanedWindow(master, orient=tk.HORIZONTAL)
        paned.pack(fill='both', expand=True, padx=20, pady=20)

        # Left Sidebar (Controls & Photo on Left)
        sidebar_container = ttk.Frame(paned)
        self.sidebar_canvas = tk.Canvas(sidebar_container, bg='#0a0c10', highlightthickness=0, width=440)
        scrollbar = ttk.Scrollbar(sidebar_container, orient="vertical", command=self.sidebar_canvas.yview)
        self.sidebar_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        self.sidebar_canvas.configure(yscrollcommand=scrollbar.set)

        left_ctrl = tk.Frame(self.sidebar_canvas, bg='#0a0c10', padx=10, pady=5)
        self.left_ctrl = left_ctrl
        self.canvas_window = self.sidebar_canvas.create_window((0, 0), window=left_ctrl, anchor="nw")

        left_ctrl.bind("<Configure>", lambda e: self.sidebar_canvas.configure(scrollregion=self.sidebar_canvas.bbox("all")))
        self.sidebar_canvas.bind("<Configure>", lambda e: self.sidebar_canvas.itemconfig(self.canvas_window, width=max(e.width, 440)))
        self.sidebar_canvas.bind_all("<MouseWheel>", lambda e: self.sidebar_canvas.yview_scroll(int(-1*(e.delta/120)), "units"))

        # 1. Header Frame with Live Badge
        header_frame = tk.Frame(left_ctrl, bg='#0a0c10')
        header_frame.pack(fill='x', pady=(0, 14))

        badge_row = tk.Frame(header_frame, bg='#0a0c10')
        badge_row.pack(anchor='w', pady=(0, 6))
        live_pill = tk.Label(badge_row, text="  ● LIVE RESULT  ", font=('Inter', 8, 'bold'),
                             bg='#141720', fg='#34d399', highlightthickness=1, highlightbackground='#232834')
        live_pill.pack(side='left')

        tk.Label(header_frame, text="Area", font=('Inter', 22, 'bold'), bg='#0a0c10', fg='#ffffff').pack(anchor='w')
        tk.Label(header_frame, text="AI VISUAL GEOLOCATION ENGINE", font=('Inter', 8, 'bold'), bg='#0a0c10', fg='#64748b').pack(anchor='w', pady=(2, 0))

        # 2. Mode Selector Frame
        mode_card = tk.Frame(left_ctrl, bg='#141720', highlightthickness=1, highlightbackground='#232834', padx=6, pady=6)
        mode_card.pack(fill='x', pady=(0, 12))

        tk.Label(mode_card, text="OPERATING MODE", font=('Inter', 7, 'bold'), bg='#141720', fg='#64748b').pack(anchor='w', padx=4, pady=(2, 6))
        m_btns_frm = tk.Frame(mode_card, bg='#141720')
        m_btns_frm.pack(fill='x')
        RoundedRadio(m_btns_frm, text="Search Target", variable=self.mode_var, value="search", width=195, height=32, command=self._update_mode).pack(side='left', padx=3)
        RoundedRadio(m_btns_frm, text="Create Index", variable=self.mode_var, value="create", width=195, height=32, command=self._update_mode).pack(side='left', padx=3)

        # 3. Query Image Card (STAYS ON THE LEFT!)
        self.query_card = tk.Frame(left_ctrl, bg='#141720', highlightthickness=1, highlightbackground='#232834', padx=12, pady=12)
        self.query_card.pack(fill='x', pady=(0, 12))

        q_head = tk.Frame(self.query_card, bg='#141720')
        q_head.pack(fill='x', pady=(0, 8))
        tk.Label(q_head, text="TARGET QUERY IMAGE", font=('Inter', 8, 'bold'), bg='#141720', fg='#64748b').pack(side='left')
        self.query_filename_label = tk.Label(q_head, text="No photo loaded", font=('Inter', 8, 'bold'), bg='#141720', fg='#94a3b8')
        self.query_filename_label.pack(side='right')

        img_box = tk.Frame(self.query_card, bg='#0d1017', highlightthickness=1, highlightbackground='#232834', padx=4, pady=4)
        img_box.pack(fill='x', pady=(0, 8))
        self.query_img_label = tk.Label(img_box, text="📸  CLICK TO SELECT QUERY IMAGE\n(.jpg, .png · Street View photo)",
                                        font=('Inter', 9), bg='#0d1017', fg='#64748b', cursor='hand2', justify='center', pady=20)
        self.query_img_label.pack(fill='both', expand=True)
        self.query_img_label.bind("<Button-1>", lambda e: self.select_image())

        self.select_btn = RoundedButton(self.query_card, text="📸  Browse Query Photo", command=self.select_image,
                                        width=390, height=36, bg_color='#191d28', hover_color='#222734',
                                        pressed_color='#141720', text_color='#f1f5f9', border_color='#2c3340',
                                        font=('Inter', 9, 'bold'))
        self.select_btn.pack(pady=(2, 0))

        # 4. Parameters Card
        self.params_card = tk.Frame(left_ctrl, bg='#141720', highlightthickness=1, highlightbackground='#232834', padx=12, pady=12)
        self.params_card.pack(fill='x', pady=(0, 12))

        tk.Label(self.params_card, text="SEARCH PARAMETERS", font=('Inter', 8, 'bold'), bg='#141720', fg='#64748b').pack(anchor='w', pady=(0, 8))

        params = [
            ("CENTER LATITUDE", self.lat_var),
            ("CENTER LONGITUDE", self.lon_var),
            ("RADIUS (KM)", self.radius_var),
            ("GRID STEP (M)", self.res_var),
        ]
        for txt, var in params:
            row = tk.Frame(self.params_card, bg='#141720')
            row.pack(fill='x', pady=4)
            tk.Label(row, text=txt, font=('Inter', 8, 'bold'), bg='#141720', fg='#94a3b8', width=18, anchor='w').pack(side='left')
            RoundedEntry(row, textvariable=var, width=190, height=30, bg_color='#191d28', border_color='#232834', text_color='#f1f5f9').pack(side='right')

        # 5. Primary Action Button
        self.query_btn = RoundedButton(left_ctrl, text="▶  RUN SEARCH", command=self.run,
                                       width=416, height=46, bg_color='#ffffff', hover_color='#e2e8f0',
                                       pressed_color='#cbd5e1', text_color='#0a0c10', font=('Inter', 11, 'bold'))
        self.query_btn.pack(fill='x', pady=(0, 8))

        # 6. Status and Progress Bar
        self.status_label = tk.Label(left_ctrl, text="System ready", font=('Inter', 9), bg='#0a0c10', fg='#94a3b8', wraplength=410, anchor='w')
        self.status_label.pack(fill='x', pady=(4, 6))

        self.progress = ttk.Progressbar(left_ctrl, orient="horizontal", mode="determinate")
        self.progress.pack(fill='x', pady=(0, 10))

        self.canvas = ttk.Label(left_ctrl)
        self.canvas.pack(pady=4)

        # 7. Results Section
        res_header = tk.Frame(left_ctrl, bg='#0a0c10')
        res_header.pack(fill='x', pady=(12, 4))
        tk.Label(res_header, text="TOP RESULTS", font=('Inter', 8, 'bold'), bg='#0a0c10', fg='#64748b').pack(side='left')
        self.res_count_lbl = tk.Label(res_header, text="0 candidates", font=('Inter', 8), bg='#0a0c10', fg='#64748b')
        self.res_count_lbl.pack(side='right')

        tree_frame = tk.Frame(left_ctrl, bg='#141720', highlightthickness=1, highlightbackground='#232834')
        tree_frame.pack(fill='x', pady=(0, 8))

        cols = ("Rank", "Score", "Coordinates")
        self.res_tree = ttk.Treeview(tree_frame, columns=cols, show="headings", height=5, style="Treeview")
        self.res_tree.heading("Rank", text="#")
        self.res_tree.heading("Score", text="Inliers")
        self.res_tree.heading("Coordinates", text="Coordinates")
        self.res_tree.column("Rank", width=36, anchor='center')
        self.res_tree.column("Score", width=70, anchor='center')
        self.res_tree.column("Coordinates", width=290, anchor='w')

        res_scroll = ttk.Scrollbar(tree_frame, orient="vertical", command=self.res_tree.yview)
        self.res_tree.configure(yscrollcommand=res_scroll.set)
        self.res_tree.pack(side='left', fill='x', expand=True)
        res_scroll.pack(side='right', fill='y')

        self.res_menu = tk.Menu(master, tearoff=0, bg='#141720', fg='#f1f5f9', activebackground='#222836', activeforeground='#ffffff')
        self.res_menu.add_command(label="📋 Copy Coordinates", command=self.copy_res_coords)
        self.res_menu.add_command(label="🌐 Open in Google Maps", command=self.open_res_gmaps)
        self.res_tree.bind("<Button-2>" if "darwin" in sys.platform else "<Button-3>", self.show_res_menu)
        self.res_tree.bind("<<TreeviewSelect>>", self._on_res_select)

        res_btns_frm = tk.Frame(left_ctrl, bg='#0a0c10')
        res_btns_frm.pack(fill='x', pady=(0, 14))
        self.copy_btn = RoundedButton(res_btns_frm, text="📋 Copy Coords", command=self.copy_res_coords,
                                      width=200, height=36, bg_color='#191d28', hover_color='#222734',
                                      pressed_color='#141720', text_color='#f1f5f9', border_color='#2c3340',
                                      font=('Inter', 9, 'bold'))
        self.copy_btn.pack(side='left', padx=(0, 4))
        self.maps_btn = RoundedButton(res_btns_frm, text="🌐 Google Maps", command=self.open_res_gmaps,
                                      width=200, height=36, bg_color='#191d28', hover_color='#222734',
                                      pressed_color='#141720', text_color='#f1f5f9', border_color='#2c3340',
                                      font=('Inter', 9, 'bold'))
        self.maps_btn.pack(side='right', padx=(4, 0))

        # 8. Utility Actions Card
        util_card = tk.Frame(left_ctrl, bg='#141720', highlightthickness=1, highlightbackground='#232834', padx=10, pady=10)
        util_card.pack(fill='x', pady=(0, 14))
        tk.Label(util_card, text="UTILITIES & HUB", font=('Inter', 8, 'bold'), bg='#141720', fg='#64748b').pack(anchor='w', pady=(0, 6))

        self.coverage_btn = RoundedButton(util_card, text="🗺  Show Coverage Map", command=self.show_coverage_map,
                                           width=394, height=34, bg_color='#191d28', hover_color='#222734',
                                           pressed_color='#141720', text_color='#94a3b8', border_color='#2c3340',
                                           font=('Inter', 9))
        self.coverage_btn.pack(pady=3)

        self.hub_btn = RoundedButton(util_card, text="🌐  Community Hub", command=self.show_community_hub,
                                     width=394, height=34, bg_color='#191d28', hover_color='#222734',
                                     pressed_color='#141720', text_color='#94a3b8', border_color='#2c3340',
                                     font=('Inter', 9))
        self.hub_btn.pack(pady=3)

        io_row = tk.Frame(util_card, bg='#141720')
        io_row.pack(fill='x', pady=3)
        self.export_btn = RoundedButton(io_row, text="📤 Export", command=self.export_index,
                                        width=192, height=34, bg_color='#191d28', hover_color='#222734',
                                        pressed_color='#141720', text_color='#94a3b8', border_color='#2c3340',
                                        font=('Inter', 9))
        self.export_btn.pack(side='left', padx=(0, 3))
        self.import_btn = RoundedButton(io_row, text="📥 Import", command=self.import_index,
                                        width=192, height=34, bg_color='#191d28', hover_color='#222734',
                                        pressed_color='#141720', text_color='#94a3b8', border_color='#2c3340',
                                        font=('Inter', 9))
        self.import_btn.pack(side='right', padx=(3, 0))

        self.help_btn = RoundedButton(util_card, text="📖  Engine Guide & Help", command=self.show_help,
                                      width=394, height=34, bg_color='#191d28', hover_color='#222734',
                                      pressed_color='#141720', text_color='#94a3b8', border_color='#2c3340',
                                      font=('Inter', 9))
        self.help_btn.pack(pady=3)

        # 9. Right Map Frame & Bottom HUD
        self.map_frame = tk.Frame(paned, bg='#0a0c10')
        paned.add(sidebar_container, weight=0)
        paned.add(self.map_frame, weight=1)

        # Bottom HUD Bar (Matches user's reference screenshot!)
        self.hud_frame = tk.Frame(self.map_frame, bg='#141720', highlightthickness=1, highlightbackground='#232834', padx=12, pady=10)
        self.hud_frame.pack(side='bottom', fill='x', pady=(10, 0))

        c1 = tk.Frame(self.hud_frame, bg='#141720', padx=10)
        c1.pack(side='left', fill='y')
        tk.Label(c1, text="IDENTIFIED", font=('Inter', 7, 'bold'), bg='#141720', fg='#64748b').pack(anchor='w')
        self.hud_loc_lbl = tk.Label(c1, text="Aktau · Kazakhstan", font=('Inter', 11, 'bold'), bg='#141720', fg='#f1f5f9')
        self.hud_loc_lbl.pack(anchor='w')

        c2 = tk.Frame(self.hud_frame, bg='#141720', padx=15)
        c2.pack(side='left', fill='y')
        tk.Label(c2, text="COORDINATES", font=('Inter', 7, 'bold'), bg='#141720', fg='#64748b').pack(anchor='w')
        self.hud_coords_lbl = tk.Label(c2, text=f"{self.lat_var.get():.4f}°N, {self.lon_var.get():.4f}°E", font=('Inter', 11, 'bold'), bg='#141720', fg='#f1f5f9')
        self.hud_coords_lbl.pack(anchor='w')

        c3 = tk.Frame(self.hud_frame, bg='#141720', padx=15)
        c3.pack(side='left', fill='y')
        tk.Label(c3, text="RADIUS", font=('Inter', 7, 'bold'), bg='#141720', fg='#64748b').pack(anchor='w')
        self.hud_radius_lbl = tk.Label(c3, text=f"~{self.radius_var.get():.1f} km", font=('Inter', 11, 'bold'), bg='#141720', fg='#f1f5f9')
        self.hud_radius_lbl.pack(anchor='w')

        c4 = tk.Frame(self.hud_frame, bg='#141720', padx=15)
        c4.pack(side='right', fill='y')
        self.hud_match_sub = tk.Label(c4, text="MATCH · READY", font=('Inter', 7, 'bold'), bg='#141720', fg='#64748b')
        self.hud_match_sub.pack(anchor='e')
        self.hud_match_pct = tk.Label(c4, text="--%", font=('Inter', 20, 'bold'), bg='#141720', fg='#ffffff')
        self.hud_match_pct.pack(anchor='e')

        def _on_param_change(*args):
            try:
                self.hud_coords_lbl.config(text=f"{self.lat_var.get():.4f}°N, {self.lon_var.get():.4f}°E")
                self.hud_radius_lbl.config(text=f"~{self.radius_var.get():.1f} km")
            except Exception:
                pass
        self.lat_var.trace_add("write", _on_param_change)
        self.lon_var.trace_add("write", _on_param_change)
        self.radius_var.trace_add("write", _on_param_change)

        # Map Widget with Carto Dark tiles
        self.map_widget = tkintermapview.TkinterMapView(self.map_frame, corner_radius=12)
        self.map_widget.pack(fill="both", expand=True)
        self.map_widget.set_tile_server("https://a.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}.png", max_zoom=19)
        self.map_widget.set_position(self.lat_var.get(), self.lon_var.get())
        self.map_widget.set_zoom(15)

        live_map_tag = tk.Label(self.map_frame, text="  ● LIVE RESULT  ", font=('Inter', 8, 'bold'),
                                bg='#141720', fg='#34d399', highlightthickness=1, highlightbackground='#232834')
        live_map_tag.place(relx=0.02, rely=0.02, anchor="nw")

        self.monitor_label = tk.Label(self.map_frame, text="TARGET SCAN\nINACTIVE", font=('Inter', 8, 'bold'),
                                      fg="#94a3b8", bg="#141720", highlightthickness=1, highlightbackground="#232834", padx=8, pady=4)
        self.monitor_label.place(relx=0.98, rely=0.02, anchor="ne")

        self.coverage_markers, self.result_elements, self.search_nets = [], [], []

        self.match_queue, self.results_queue = queue.Queue(), queue.Queue()
        self.thumbnail_pool = []
        self._thumbnail_pool_lock = threading.Lock()
        self._update_mode()
        self.poll_match_queue()

    def _update_mode(self):
        mode = self.mode_var.get()
        if mode == "search":
            self.query_btn.config(text="▶  RUN SEARCH")
            if hasattr(self, 'query_card'):
                self.query_card.pack(fill='x', pady=(0, 12), before=self.params_card)
        else:
            self.query_btn.config(text="▶  CREATE INDEX")
            if hasattr(self, 'query_card'):
                self.query_card.pack_forget()

    def run(self):
        mode = self.mode_var.get()
        if mode == "create":
            center = (self.lat_var.get(), self.lon_var.get())
            radius = self.radius_var.get()
            res = self.res_var.get()
            fov = self.crop_fov.get()
            size = self.crop_size.get()
            step = self.crop_step.get()
            threading.Thread(target=self._create_embeddings,
                           args=(center, radius, res, fov, size, step), daemon=True).start()
            self._set_status("Creating embeddings in background...")
        else:
            self.query()



    def _create_embeddings(self, center, radius, res, crop_fov, crop_size, crop_step):
        q = self.match_queue
        q.put(('status', "Getting grid points..."))
        points = grid_points(center, radius, res)

        q.put(('status', f"Generated {len(points)} grid points. Downloading scan nodes..."))

        panoids = get_panoids(
            points,
            status_callback=lambda idx, total: q.put(('status', f"Scan node fetch {idx}/{total}...")),
            max_workers=MAX_PANOID_WORKERS
        )
        q.put(('status', f"Found {len(panoids)} scan nodes. Extracting EigenPlace features..."))

        headings_all = sorted(list(set(((h // crop_step) * crop_step) % 360 for h in range(0, 360, crop_step))))
        embeddings_per_panoid = len(headings_all)

        os.makedirs(AREA_PARTS_DIR, exist_ok=True)


        existing_files = set()
        try:
            existing_parts = glob.glob(os.path.join(AREA_PARTS_DIR, "area_part_*.npz"))
            for ep in existing_parts:
                data = np.load(ep, allow_pickle=True)
                for p in data['paths']:
                    existing_files.add(os.path.basename(str(p)))
                del data
            if existing_files:
                q.put(('status', f"Loaded {len(existing_files)} existing entries from part files. Starting..."))
        except Exception as e:
            q.put(('status', f"Warning: Could not load existing parts: {e}"))

        crop_queue = queue.Queue(maxsize=128)
        tracker = ProgressTracker(len(panoids), estimate_storage=True,
                                 embeddings_per_item=embeddings_per_panoid, avg_bytes_per_embedding=2560)
        total_extracted = 0


        def batch_extractor():
            nonlocal total_extracted
            target_batch_size = 32
            batch_buffer = []
            area_buffer_descs = []
            area_buffer_paths = []
            area_buffer_lats = []
            area_buffer_lons = []

            def save_area_chunk():
                if not area_buffer_descs:
                    return
                try:
                    timestamp = int(time.time() * 1000)
                    part_filename = os.path.join(AREA_PARTS_DIR, f"area_part_{timestamp}.npz")
                    all_descs = np.vstack(area_buffer_descs)
                    np.savez_compressed(
                        part_filename,
                        descriptors=all_descs,
                        paths=np.array(area_buffer_paths, dtype=object),
                        lats=np.array(area_buffer_lats, dtype=np.float32),
                        lons=np.array(area_buffer_lons, dtype=np.float32),
                    )
                    q.put(('status', f"Saved index chunk: {len(area_buffer_paths)} items"))
                    area_buffer_descs.clear()
                    area_buffer_paths.clear()
                    area_buffer_lats.clear()
                    area_buffer_lons.clear()
                except Exception as e:
                    print(f"Error saving EigenPlace chunk: {e}")

            def process_batch(buffer):
                nonlocal total_extracted
                crops = [b[0] for b in buffer]
                meta = [b[1] for b in buffer]
                try:
                    total_extracted += len(meta)

                    # Direct GPU batch extraction without tensor -> PIL -> tensor conversions
                    crops_batch_tensor = torch.cat(crops, dim=0)
                    cos_descs = batch_extract_area(crops_batch_tensor, batch_size=len(crops))
                    area_buffer_descs.append(cos_descs)
                    area_buffer_paths.extend([m['path'] for m in meta])
                    area_buffer_lats.extend([m['lat'] for m in meta])
                    area_buffer_lons.extend([m['lon'] for m in meta])

                    if len(area_buffer_paths) >= 5000:
                        save_area_chunk()
                except Exception as e:
                    print(f"Batch processing error: {e}")

            while True:
                item = crop_queue.get()
                if item == "DONE":
                    if batch_buffer:
                        process_batch(batch_buffer)
                    save_area_chunk()
                    crop_queue.task_done()
                    break
                batch_buffer.append(item)
                if len(batch_buffer) >= target_batch_size:
                    process_batch(batch_buffer)
                    batch_buffer = []
                crop_queue.task_done()

            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            gc.collect()

        extractor_thread = threading.Thread(target=batch_extractor)
        extractor_thread.start()

        base_dirs = get_projection_base_dirs(crop_fov, (crop_size, crop_size))

        def process_one_panoid(panoid):
            tiles = tiles_info(panoid['panoid'])
            tiles_data = download_tiles(tiles, max_workers=MAX_DOWNLOAD_WORKERS)
            if not tiles_data:
                return False
            try:
                pano_img = stitch_tiles(tiles_data)
            except Exception:
                return False
            maxw = 2048
            if pano_img.size[0] > maxw:
                pano_img = pano_img.resize((maxw, int(pano_img.size[1] * (maxw / pano_img.size[0]))), Image.BILINEAR)

            pano_t = pil_to_tensor(pano_img)
            panoid_id = panoid['panoid']


            missing_yaws = [y for y in headings_all if f"{panoid_id}_{y}.npz" not in existing_files]

            if missing_yaws:
                crops_batch = equirectangular_to_rectilinear_torch(
                    pano_t, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                    yaw_deg=missing_yaws, pitch_deg=0, base_dirs=base_dirs
                )
                for i, yaw in enumerate(missing_yaws):
                    crop_t = crops_batch[i].unsqueeze(0)
                    emb_path = f"{panoid_id}_{yaw}.npz"
                    meta = {'path': emb_path, 'lat': panoid['lat'], 'lon': panoid['lon'], 'yaw': yaw}
                    crop_queue.put((crop_t, meta))

            pano_img.close()
            del pano_t
            return True

        with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_PANOID_WORKERS) as executor:
            for idx, _ in enumerate(executor.map(process_one_panoid, panoids), 1):
                tracker.update(idx)
                q.put(('status', f"Downloading & Stitching: {tracker.get_status()}"))

        crop_queue.put("DONE")
        extractor_thread.join()

        q.put(('status', f"All embeddings saved ({total_extracted} new). Fitting PCA..."))

        try:
            part_files = sorted(glob.glob(os.path.join(AREA_PARTS_DIR, "area_part_*.npz")))
            all_raw = []
            for pf in part_files:
                data = np.load(pf, allow_pickle=True)
                all_raw.append(data['descriptors'])
            if all_raw:
                all_raw = np.vstack(all_raw)
                from area_utils import fit_pca, save_pca, apply_pca, AREA_PCA_DIM
                pca = fit_pca(all_raw, n_components=AREA_PCA_DIM, whiten=True)
                pca_path = os.path.join(COMPACT_INDEX_DIR, "area_pca.pkl")
                save_pca(pca_path)
                

            q.put(('status', f"PCA fitted. Building index..."))
        except Exception as e:
            print(f"PCA Error: {e}")
            q.put(('status', f"All embeddings saved ({total_extracted} new). Building index..."))

        build_compact_index()
        q.put(('status', f"Done! Index ready. {total_extracted} new entries added."))

        global _compact_cache
        _compact_cache = None



    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if not path:
            return
        self.query_img_path = path
        img = Image.open(path).convert('RGB')

        thumb_img = img.copy()
        thumb_img.thumbnail((390, 200))
        imgtk = ImageTk.PhotoImage(thumb_img)
        self.query_img_label.configure(image=imgtk, text="", pady=0)
        self.query_img_label.image = imgtk
        if hasattr(self, 'query_filename_label'):
            self.query_filename_label.config(text=os.path.basename(path), fg='#ffffff')
        if hasattr(self, 'select_btn'):
            self.select_btn.config(text="📸  Change Query Photo")

        self.search_nets = []
        self._clear_result_elements()
        self._set_status(f"Selected: {os.path.basename(path)}")

    def query(self):
        if not self.query_img_path:
            self.select_image()
            if not self.query_img_path:
                return

        path = self.query_img_path
        self.search_nets = []
        self._clear_result_elements()

        manual_center = (self.lat_var.get(), self.lon_var.get())
        manual_radius = self.radius_var.get()

        if self.search_option_var.get() == "ai_coarse":
            self._set_status("Requesting AI coarse geolocation...")
            self.master.update_idletasks()
            try:
                ai_guesses = self._coarse_guess_gemini(path)
                if ai_guesses:
                    for lat, lon, conf, direction, reason in ai_guesses:
                        params = self.analyze_ai_response(conf, reason)
                        self.search_nets.append((lat, lon, params['radius'], direction,
                                               params['grid_res'], params['fov'],
                                               params['direction_precision'], params['rationale']))
                    self._set_status(f"AI suggested {len(ai_guesses)} location(s).")
                else:
                    self._set_status("AI unsure → using manual center.")
                    self.search_nets = [(manual_center[0], manual_center[1], manual_radius, "UNKNOWN",
                                       self.res_var.get(), self.crop_fov.get(), 'full', 'Manual fallback')]
            except Exception as e:
                self._set_status(f"AI error: {e}")
                self.search_nets = [(manual_center[0], manual_center[1], manual_radius, "UNKNOWN",
                                   self.res_var.get(), self.crop_fov.get(), 'full', 'AI error fallback')]
        else:
            self.search_nets = [(manual_center[0], manual_center[1], manual_radius, "UNKNOWN",
                               self.res_var.get(), self.crop_fov.get(), 'full', 'Manual search')]
            self._set_status("Using manual center.")



        self.master.update_idletasks()
        self.start_full_search()

    def start_full_search(self):
        if not self.query_img_path or not self.search_nets:
            self._set_status("No query image or search area defined.")
            return

        self.query_btn.config(state='disabled', text="Searching...")
        self.stop_animation = False
        self.thumbnail_pool = []

        fov = self.crop_fov.get()
        size = self.crop_size.get()
        step = self.crop_step.get()
        threshold = self.match_threshold.get()
        res = self.res_var.get()

        def run_search_background():
            threads = []
            for net in self.search_nets:
                if len(net) == 8:
                    lat, lon, radius, direction, grid_res, net_fov, dir_precision, rationale = net
                elif len(net) == 4:
                    lat, lon, radius, direction = net
                    grid_res, net_fov = res, fov
                else:
                    continue
                center = (lat, lon)
                t = threading.Thread(target=self._run_search,
                    args=(center, radius, grid_res, threshold, net_fov, size, step, direction))
                threads.append(t)
                t.start()
            for t in threads:
                t.join()

            all_bests = []
            while not self.results_queue.empty():
                try:
                    all_bests.append(self.results_queue.get_nowait())
                except queue.Empty:
                    break

            if all_bests:
                global_best = max(all_bests, key=lambda b: b['inliers'])
                query_img_resized = Image.open(self.query_img_path).convert('RGB').resize((size, size), Image.BILINEAR)
                self.master.after(0, lambda: self._handle_match_done(
                    global_best, query_img_resized, fov, size,
                    (global_best.get('lat', self.lat_var.get()), global_best.get('lon', self.lon_var.get())),
                    max(net[2] for net in self.search_nets)))
            else:
                self.master.after(0, lambda: self._set_status("No good matches found."))

            self.master.after(0, lambda: self.query_btn.config(state='normal', text="▶  Run Search", command=self.run))
        threading.Thread(target=run_search_background, daemon=True).start()




    def _run_search(self, center, radius, res, threshold, crop_fov, crop_size, crop_step, direction="UNKNOWN"):
        q = self.match_queue
        q.put(('status', "Starting search..."))
        early_exit_event = threading.Event()

        try:

            pca_path = os.path.join(COMPACT_INDEX_DIR, "area_pca.pkl")
            if os.path.exists(pca_path):
                from area_utils import load_pca, _pca_model
                if _pca_model is None:
                    load_pca(pca_path)
            

            query_img = Image.open(self.query_img_path).convert("RGB")
            query_img_resize = query_img.resize((crop_size, crop_size), Image.BILINEAR)
            self.current_search_context = (query_img_resize, crop_fov, crop_size, center, radius)


            q.put(('status', "Extracting query Area-loc descriptor (multi-scale)..."))
            query_for_area = query_img_resize
            desc_original = extract_area_descriptor(query_for_area, apply_pca_reduction=True)


            w, h = query_img_resize.size
            margin_x, margin_y = int(w * 0.1), int(h * 0.1)
            cropped = query_img_resize.crop((margin_x, margin_y, w - margin_x, h - margin_y))
            cropped = cropped.resize((crop_size, crop_size), Image.BILINEAR)
            desc_zoom = extract_area_descriptor(cropped, apply_pca_reduction=True)
            cropped.close()


            query_area_desc = 0.65 * desc_original + 0.35 * desc_zoom
            query_area_desc = query_area_desc / (np.linalg.norm(query_area_desc) + 1e-8)


            query_img_flipped = query_img_resize.transpose(Image.FLIP_LEFT_RIGHT)
            desc_flipped = extract_area_descriptor(query_img_flipped, apply_pca_reduction=True)
            desc_flipped_zoom = extract_area_descriptor(
                query_img_flipped.crop((margin_x, margin_y, w - margin_x, h - margin_y)).resize((crop_size, crop_size), Image.BILINEAR),
                apply_pca_reduction=True
            )
            desc_flipped = 0.65 * desc_flipped + 0.35 * desc_flipped_zoom
            desc_flipped = desc_flipped / (np.linalg.norm(desc_flipped) + 1e-8)




            q.put(('status', "Searching index (original + flipped)..."))
            K_AREALOC = 1000
            results_original = search_compact_index(query_desc=query_area_desc, center=center, radius_km=radius, top_k=500)
            results_flipped = search_compact_index(query_desc=desc_flipped, center=center, radius_km=radius, top_k=500)
            

            seen = {}
            for r in results_original + results_flipped:
                key = r['panoid']
                if key not in seen or r['score'] > seen[key]['score']:
                    seen[key] = r
            compact_results = sorted(seen.values(), key=lambda x: x['score'], reverse=True)[:K_AREALOC]

            if not compact_results:
                q.put(('status', "No candidates found in radius."))
                self.results_queue.put({'inliers': 0, 'panoid': None, 'heading': None,
                                       'lat': None, 'lon': None, 'matches': None,
                                       'kp1': None, 'kp2': None, 'emb_path': None, 'confidence': 'none'})
                return

            if True:
                AREA_3R_STAGE2_TOP_N = 120
                candidates_to_check = compact_results[:AREA_3R_STAGE2_TOP_N]
                q.put(('status', f"Stage 2: Running Area-3R on top {len(candidates_to_check)} candidates..."))
                
                all_area_3r_matches = []
                best = {'inliers': 0, 'panoid': None, 'heading': None, 'lat': None, 'lon': None,
                        'matches': None, 'kp1': None, 'kp2': None, 'emb_path': None}
                
                try:
                    area_3r = get_lazy_area_3r()
                    if area_3r is not None:
                        prefetch_queue = queue.Queue(maxsize=16)
                        def prefetch_panos():
                            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pf_exec:
                                def _fetch(m):
                                    pid = m.get('panoid')
                                    if not pid: return None
                                    p_img = fetch_single_pano(pid)
                                    if p_img is not None:
                                        return p_img
                                    try:
                                        tiles = tiles_info(pid)
                                        td = download_tiles(tiles, max_workers=16)
                                        if td:
                                            p_img = stitch_tiles(td)
                                            maxw = 2048
                                            if p_img.size[0] > maxw:
                                                p_img = p_img.resize((maxw, int(p_img.size[1] * (maxw / p_img.size[0]))), Image.BILINEAR)
                                            return p_img
                                    except Exception:
                                        return None
                                    return None

                                futs = [pf_exec.submit(_fetch, m) for m in candidates_to_check]
                                for fut in futs:
                                    if early_exit_event.is_set():
                                        break
                                    prefetch_queue.put(fut.result())
                            prefetch_queue.put(None)

                        pf_thread = threading.Thread(target=prefetch_panos, daemon=True)
                        pf_thread.start()

                        for i, match in enumerate(candidates_to_check):
                            q.put(('progress', i, len(candidates_to_check)))
                            q.put(('status', f"Area-3R Match: {i+1}/{len(candidates_to_check)}"))
                            pid = match.get('panoid')
                            hdg = match.get('heading')
                            pano_img = prefetch_queue.get()

                            if not pid or hdg is None or pano_img is None:
                                if pano_img: pano_img.close()
                                continue
                            
                            if pano_img:
                                pano_t = pil_to_tensor(pano_img)
                                base_dirs_m3 = get_projection_base_dirs(crop_fov, (crop_size, crop_size))
                                crop_t_m3 = equirectangular_to_rectilinear_torch(
                                    pano_t, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                                    yaw_deg=[hdg], pitch_deg=0, base_dirs=base_dirs_m3)[0].unsqueeze(0)
                                
                                crop_pil = tensor_to_pil(crop_t_m3)
                                m3_matches0, m3_matches1, m3_conf = get_area_3r_matches(query_img_resize, crop_pil, area_3r)
                                m3_score = len(m3_matches0)
                                if m3_score > 50:
                                    print(f"[Stage 2 Area-3R] Candidate got {m3_score} dense matches")
                                
                                match_res = {
                                    'inliers': m3_score, 
                                    'panoid': pid, 'heading': hdg, 'lat': match.get('lat'), 'lon': match.get('lon'),
                                    'kp1': m3_matches0, 'kp2': m3_matches1, 'matches': np.array([[k, k] for k in range(m3_score)]),
                                    'emb_path': match.get('emb_path', '')
                                }
                                
                                if m3_score > 50:
                                    all_area_3r_matches.append(match_res)
                                    q.put(('scan_blip', match.get('lat'), match.get('lon'), m3_score, None))

                                if m3_score > best['inliers']:
                                    best = match_res.copy()
                                    if best['inliers'] >= 150:
                                        q.put(('match_update', best))
                                        q.put(('status', f"New Top Match: {best['inliers']} dense patches!"))
                                
                                del pano_t, crop_t_m3
                                pano_img.close()
                                if torch.backends.mps.is_available(): torch.mps.empty_cache()
                                
                                if best['inliers'] >= 450: # Slightly higher early exit with consensus
                                    q.put(('status', f"Ultra-Strong Area-3R match! {best['inliers']} points — stopping early"))
                                    early_exit_event.set()
                                    break
                        

                        if len(all_area_3r_matches) >= 3:
                            CELL_SIZE = 0.00045 # ~50m
                            cells = defaultdict(list)
                            for m in all_area_3r_matches:
                                cell = (round(m['lat'] / CELL_SIZE), round(m['lon'] / CELL_SIZE))
                                cells[cell].append(m)
                            
                            scored_clusters = []
                            for cell_key, cell_matches in cells.items():
                                neighborhood = []
                                for dlat in [-1, 0, 1]:
                                    for dlon in [-1, 0, 1]:
                                        neighbor = (cell_key[0] + dlat, cell_key[1] + dlon)
                                        neighborhood.extend(cells.get(neighbor, []))
                                
                                cell_score = sum(math.sqrt(m['inliers']) for m in neighborhood)
                                cluster_best = max(neighborhood, key=lambda m: m['inliers'])
                                scored_clusters.append({'score': cell_score, 'match': cluster_best})


                            scored_clusters.sort(key=lambda x: x['score'], reverse=True)
                            
                            top_results_for_sidebar = []
                            seen_pids = set()
                            for sc in scored_clusters:
                                r = sc['match']
                                if r['panoid'] not in seen_pids:
                                    top_results_for_sidebar.append(r)
                                    seen_pids.add(r['panoid'])
                                if len(top_results_for_sidebar) >= 10: break
                            
                            if top_results_for_sidebar:
                                best = top_results_for_sidebar[0].copy()
                                best['all_top_clusters'] = top_results_for_sidebar # For sidebar display
                        elif best['inliers'] > 0:
                             best['all_top_clusters'] = [best]
                            
                except Exception as e:
                    print(f"Stage 2 Area-3R error: {e}")
                
                if best['inliers'] > 0:
                    best['inliers'] = 200 + best['inliers'] // 10
                    self.results_queue.put(best)
                else:
                    self.results_queue.put({'inliers': 0, 'panoid': None, 'heading': None,
                                           'lat': None, 'lon': None, 'matches': None,
                                           'kp1': None, 'kp2': None, 'emb_path': None, 'confidence': 'none'})
                return

        except Exception as e:
            import traceback
            traceback.print_exc()
            q.put(('status', f"Search error: {e}"))
            self.results_queue.put({'inliers': 0, 'panoid': None, 'heading': None,
                                   'lat': None, 'lon': None, 'matches': None,
                                   'kp1': None, 'kp2': None, 'emb_path': None, 'confidence': 'none'})

    def show_coverage_map(self):
        from collections import defaultdict
        self._set_status("Loading coverage data...")
        locations = set()


        if os.path.exists(COMPACT_META_PATH):
            try:
                meta = np.load(COMPACT_META_PATH, allow_pickle=True)
                lats = meta['lats']
                lons = meta['lons']
                for i in range(len(lats)):
                    locations.add((round(float(lats[i]), 6), round(float(lons[i]), 6)))
                del meta
            except Exception as e:
                print(f"[COVERAGE] Error loading metadata: {e}")

        self._clear_coverage_markers()
        self._clear_result_elements()

        if locations:
            latlons = list(locations)
            center_lat = sum(lat for lat, lon in latlons) / len(latlons)
            center_lon = sum(lon for lat, lon in latlons) / len(latlons)
            self.map_widget.set_position(center_lat, center_lon)
            self.map_widget.set_zoom(13)

            MAX_CONNECT_DIST_KM = 0.03
            BUCKET_SIZE = 0.0002
            grid = defaultdict(list)
            for loc in latlons:
                bucket_key = (int(loc[0] / BUCKET_SIZE), int(loc[1] / BUCKET_SIZE))
                grid[bucket_key].append(loc)

            graph = defaultdict(list)
            for bucket_key, bucket_locs in grid.items():
                for dlat in [-1, 0, 1]:
                    for dlon in [-1, 0, 1]:
                        neighbor_key = (bucket_key[0] + dlat, bucket_key[1] + dlon)
                        if neighbor_key not in grid: continue
                        for loc1 in bucket_locs:
                            for loc2 in grid[neighbor_key]:
                                if loc1 >= loc2: continue
                                if haversine(loc1, loc2) <= MAX_CONNECT_DIST_KM:
                                    graph[loc1].append(loc2)
                                    graph[loc2].append(loc1)

            visited = set()
            line_count = 0
            for start_loc in latlons:
                if start_loc in visited: continue
                component = []
                bfs_queue = [start_loc]
                visited.add(start_loc)
                while bfs_queue:
                    current = bfs_queue.pop(0)
                    component.append(current)
                    for neighbor in graph[current]:
                        if neighbor not in visited:
                            visited.add(neighbor)
                            bfs_queue.append(neighbor)
                if len(component) >= 2:
                    path = self.map_widget.set_path(component, color="#64748b", width=2)
                    self.coverage_markers.append(path)
                    line_count += 1
                elif len(component) == 1:
                    marker = self.map_widget.set_marker(component[0][0], component[0][1], text="",
                        marker_color_circle="#94a3b8", marker_color_outside="#191d28")
                    self.coverage_markers.append(marker)

            self._set_status(f"Coverage: {len(locations)} points, {line_count} segments.")
        else:
            if self.search_nets:
                self.map_widget.set_position(self.search_nets[0][0], self.search_nets[0][1])
            else:
                self.map_widget.set_position(self.lat_var.get(), self.lon_var.get())
            self.map_widget.set_zoom(14)
            self._set_status("No index found — only showing search area(s).")


        if self.search_nets:
            for net in self.search_nets:
                net_lat, net_lon, net_radius = net[0], net[1], net[2]
                circle_points = generate_circle_points(net_lat, net_lon, net_radius)
                poly = self.map_widget.set_polygon(circle_points, outline_color="#34d399", border_width=2, fill_color=None)
                self.result_elements.append(poly)

        self.master.update_idletasks()

    

    def _handle_match_done(self, best, query_img_resize, crop_fov, crop_size, center, radius):
        self.stop_animation = True
        self._clear_coverage_markers()

        if best['inliers'] > self.match_threshold.get() and best['panoid'] is not None:
            confidence = best.get('confidence', 'UNKNOWN')
            pano_img = None
            best_crop = None

            cached_tensor = best.pop('_cached_pano_tensor', None)
            if cached_tensor is not None:
                try:
                    crop_tensor = equirectangular_to_rectilinear_torch(
                        cached_tensor, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                        yaw_deg=best['heading'], pitch_deg=0)
                    best_crop = tensor_to_pil(crop_tensor)
                    del cached_tensor, crop_tensor
                except Exception:
                    best_crop = None
                    del cached_tensor

            if best_crop is None:
                pano_img = fetch_single_pano(best['panoid'])
                if pano_img is None:
                    tiles = tiles_info(best['panoid'])
                    tiles_data = download_tiles(tiles, max_workers=MAX_DOWNLOAD_WORKERS)
                    try:
                        pano_img = stitch_tiles(tiles_data)
                    except Exception:
                        self._set_status("Failed to download visualization.")
                        return
                maxw = 2048
                if pano_img.size[0] > maxw:
                    pano_img = pano_img.resize((maxw, int(pano_img.size[1] * (maxw / pano_img.size[0]))), Image.BILINEAR)
                best_crop = equirectangular_to_rectilinear(
                    pano_img, fov_deg=crop_fov, out_hw=(crop_size, crop_size),
                    yaw_deg=best['heading'], pitch_deg=0)

            self._clear_result_elements()
            self.map_widget.set_position(center[0], center[1])
            self.map_widget.set_zoom(16)

            circle_points = generate_circle_points(center[0], center[1], radius)
            circle_poly = self.map_widget.set_polygon(circle_points, outline_color="#34d399", border_width=2, fill_color=None)
            self.result_elements.append(circle_poly)

            if best['lat'] is not None and best['lon'] is not None:
                marker = self.map_widget.set_marker(best['lat'], best['lon'],
                    text=f"📍 1. {best['lat']:.6f}, {best['lon']:.6f}\n{best['inliers']} inliers | {best['heading']}°")
                self.result_elements.append(marker)

            if best['kp1'] is not None and best['kp2'] is not None and best['matches'] is not None:
                scale1 = np.array([crop_size / query_img_resize.size[0], crop_size / query_img_resize.size[1]])
                scale2 = np.array([crop_size / best_crop.size[0], crop_size / best_crop.size[1]])
                kp1_scaled = best['kp1'] * scale1
                kp2_scaled = best['kp2'] * scale2
                match_img = draw_matches(query_img_resize.copy(), best_crop.copy(), kp1_scaled, kp2_scaled, best['matches'])
                match_img.thumbnail((2 * crop_size, crop_size))
                imgtk = ImageTk.PhotoImage(match_img)
                self._set_canvas_img(imgtk)

            try:
                if pano_img: pano_img.close()
            except: pass
            del pano_img, best_crop
            gc.collect()
            if torch.backends.mps.is_available(): torch.mps.empty_cache()

            # Update Bottom HUD Bar (like reference image)
            if best['lat'] is not None and best['lon'] is not None:
                pct = min(99, max(50, int(best['inliers'] / 3.0))) if best['inliers'] > 0 else 0
                if hasattr(self, 'hud_match_pct'):
                    self.hud_match_pct.config(text=f"{pct}%")
                    self.hud_match_sub.config(text=f"MATCH · {best['inliers']} INLIERS")
                    self.hud_coords_lbl.config(text=f"{best['lat']:.4f}°N, {best['lon']:.4f}°E")
                    self.hud_radius_lbl.config(text=f"~{radius:.1f} km")
                    self.hud_loc_lbl.config(text=f"Aktau · Match Confirmed")

            self._set_status(f"Match confirmed: {best['inliers']} inliers at heading {best['heading']}° ({confidence})")
        
            # Update results list
            self.res_tree.delete(*self.res_tree.get_children())

            results_to_show = best.get('all_top_clusters', [best])[:10]
            if hasattr(self, 'res_count_lbl'):
                self.res_count_lbl.config(text=f"{len(results_to_show)} candidates")

            for i, r in enumerate(results_to_show, 1):
                coords = f"{r['lat']:.6f}, {r['lon']:.6f}"
                self.res_tree.insert("", "end", values=(i, r['inliers'], coords))
        else:
            self._set_status("No match found.")


    def show_res_menu(self, event):
        item = self.res_tree.identify_row(event.y)
        if item:
            self.res_tree.selection_set(item)
            self.res_menu.post(event.x_root, event.y_root)

    def _on_res_select(self, event):
        item = self.res_tree.selection()
        if item:
            val = self.res_tree.item(item[0])['values']
            lat, lon = map(float, val[2].split(", "))
            self.map_widget.set_position(lat, lon)
            self.map_widget.set_zoom(18)

    def copy_res_coords(self):
        item = self.res_tree.selection()
        if item:
            coords = self.res_tree.item(item[0])['values'][2]
            self.master.clipboard_clear()
            self.master.clipboard_append(coords)
            self._set_status(f"Copied: {coords}")

    def open_res_gmaps(self):
        item = self.res_tree.selection()
        if item:
            coords = self.res_tree.item(item[0])['values'][2]
            url = f"https://www.google.com/maps/search/?api=1&query={coords.replace(' ', '')}"
            webbrowser.open(url)



    def _clear_coverage_markers(self):
        for m in self.coverage_markers: m.delete()
        self.coverage_markers = []

    def _clear_result_elements(self):
        for e in self.result_elements: e.delete()
        self.result_elements = []

    def _set_status(self, text):
        self.match_queue.put(('status', text))

    def _set_progress(self, value, maximum):
        self.match_queue.put(('progress', value, maximum))

    def _set_canvas_img(self, imgtk):
        self.canvas.configure(image=imgtk)
        self.canvas.image = imgtk

    def _add_to_thumbnail_pool(self, thumb):
        if thumb is None: return
        with self._thumbnail_pool_lock:
            self.thumbnail_pool.append(thumb)
            if len(self.thumbnail_pool) > 50:
                self.thumbnail_pool.pop(0)

    def _handle_scan_blip(self, lat, lon, inliers, thumb=None):
        if getattr(self, 'stop_animation', False): return
        if inliers > 50 and self.coverage_markers:
            self._clear_coverage_markers()
        if thumb is not None:
            self._add_to_thumbnail_pool(thumb)

        img_to_show = thumb
        if img_to_show is None and self.thumbnail_pool:
            with self._thumbnail_pool_lock:
                if self.thumbnail_pool:
                    img_to_show = random.choice(self.thumbnail_pool)

        if img_to_show:
            try:
                img_filled = ImageOps.fit(img_to_show, (128, 128), method=Image.Resampling.LANCZOS)
                border_color = (52, 211, 153) if inliers > 50 else (148, 163, 184) if inliers > 20 else (71, 85, 105)
                border = Image.new('RGB', (132, 132), border_color)
                border.paste(img_filled, (2, 2))
                photo = ImageTk.PhotoImage(border)
                self.monitor_label.config(image=photo, text=f"SCANNING...\nINLIERS: {inliers}")
                self.monitor_label.image = photo
            except Exception: pass

        try:
            current_pos = self.map_widget.get_position()
            if haversine(current_pos, (lat, lon)) > 0.05:
                self.map_widget.set_position(lat, lon)
        except Exception: pass

        color = "#34d399" if inliers > 50 else "#94a3b8" if inliers > 20 else "#475569"
        try:
            marker = self.map_widget.set_marker(lat, lon, marker_color_circle=color, marker_color_outside=color)

            self.master.after(1200, marker.delete)
        except Exception: pass



    def show_community_hub(self):
        hub_win = tk.Toplevel(self.master)
        hub_win.title("Area Community Hub")
        hub_win.configure(bg='#0a0c10')
        hub_win.geometry("720x560")
        hub_win.transient(self.master)

        header = tk.Frame(hub_win, bg='#0a0c10')
        header.pack(fill='x', padx=20, pady=(20, 10))
        tk.Label(header, text="🌐 Community Hub", font=('Inter', 18, 'bold'),
                 bg='#0a0c10', fg='#ffffff').pack(anchor='w')
        tk.Label(header, text="Download and share pre-built city indexes",
                 font=('Inter', 9), bg='#0a0c10', fg='#94a3b8').pack(anchor='w', pady=(4, 0))

        search_frame = tk.Frame(hub_win, bg='#0a0c10')
        search_frame.pack(fill='x', padx=20, pady=(10, 5))

        self._hub_search_var = tk.StringVar()
        search_entry = tk.Entry(search_frame, textvariable=self._hub_search_var,
                                font=('Inter', 10), bg='#141720', fg='#ffffff',
                                insertbackground='white', borderwidth=0, highlightthickness=1,
                                highlightcolor='#475569', highlightbackground='#232834')
        search_entry.pack(side='left', fill='x', expand=True, ipady=8, padx=(0, 10))
        search_entry.insert(0, "Search by city name...")
        search_entry.bind('<FocusIn>', lambda e: search_entry.delete(0, 'end') if search_entry.get() == "Search by city name..." else None)

        self.search_btn = RoundedButton(search_frame, text="Search",
                                       command=lambda: self._hub_search(hub_win),
                                       width=100, height=36, corner_radius=8,
                                       bg_color='#191d28', hover_color='#222734',
                                       pressed_color='#141720', text_color='#f1f5f9', border_color='#2c3340')
        self.search_btn.pack(side='right')

        list_frame = tk.Frame(hub_win, bg='#141720', highlightthickness=1, highlightbackground='#232834')
        list_frame.pack(fill='both', expand=True, padx=20, pady=10)

        self._hub_listbox = tk.Listbox(list_frame, font=('Inter', 9),
                                        bg='#141720', fg='#f1f5f9', selectbackground='#222836',
                                        selectforeground='white', borderwidth=0,
                                        highlightthickness=0, activestyle='none')
        self._hub_listbox.pack(fill='both', expand=True, side='left')

        scrollbar = tk.Scrollbar(list_frame, command=self._hub_listbox.yview)
        scrollbar.pack(side='right', fill='y')
        self._hub_listbox.config(yscrollcommand=scrollbar.set)

        self._hub_indexes = []

        bottom = tk.Frame(hub_win, bg='#0a0c10')
        bottom.pack(fill='x', padx=20, pady=(0, 20))

        self.dl_btn = RoundedButton(bottom, text="⬇ Download",
                                   command=lambda: self._hub_download(hub_win),
                                   width=160, height=40, bg_color='#ffffff',
                                   hover_color='#e2e8f0', pressed_color='#cbd5e1',
                                   text_color='#0a0c10')
        self.dl_btn.pack(side='left')

        self.up_btn = RoundedButton(bottom, text="⬆ Upload Index",
                                   command=lambda: self._hub_upload(hub_win),
                                   width=165, height=40, bg_color='#191d28',
                                   hover_color='#222734', pressed_color='#141720',
                                   text_color='#f1f5f9', border_color='#2c3340')
        self.up_btn.pack(side='right')

        self._hub_status = tk.Label(bottom, text="", font=('Inter', 9),
                                     bg='#0a0c10', fg='#94a3b8')
        self._hub_status.pack(side='left', padx=20)


        self._hub_refresh(hub_win)

    def _hub_refresh(self, hub_win):
        self._hub_status.config(text="Loading indexes...")
        hub_win.update_idletasks()

        def do_refresh():
            try:
                if not HUB_AVAILABLE:
                    self.master.after(0, lambda: self._hub_status.config(
                        text="Hub unavailable. Install: pip install huggingface_hub"))
                    return
                hub = AreaHub()
                indexes = hub.list_indexes()
                self._hub_indexes = indexes

                def update_ui():
                    self._hub_listbox.delete(0, 'end')
                    for idx in indexes:
                        size_mb = idx.get('file_size_bytes', 0) / 1024 / 1024
                        author = idx.get('author', '?')
                        badge = "🟣 Official" if idx.get('is_official') else "🟢 Community"
                        
                        line = f"📦 {idx['name']:<18} | {idx['radius_km']:>3}km | {idx['num_entries']:>6,} pts | {size_mb:>4.0f}MB | {badge} by @{author}"
                        self._hub_listbox.insert('end', line)
                    self._hub_status.config(text=f"Found {len(indexes)} indexes")

                self.master.after(0, update_ui)
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._hub_status.config(text=f"Error: {m}"))

        threading.Thread(target=do_refresh, daemon=True).start()

    def _hub_search(self, hub_win):
        query = self._hub_search_var.get().strip()
        if not query or query == "Search by city name...":
            self._hub_refresh(hub_win)
            return

        self._hub_status.config(text=f"Searching for '{query}'...")
        hub_win.update_idletasks()

        def do_search():
            try:
                hub = AreaHub()
                results = hub.search(city=query)
                self._hub_indexes = results

                def update_ui():
                    self._hub_listbox.delete(0, 'end')
                    for idx in results:
                        size_mb = idx.get('file_size_bytes', 0) / 1024 / 1024
                        author = idx.get('author', '?')
                        badge = "🟣 Official" if idx.get('is_official') else "🟢 Community"
                        
                        line = f"📦 {idx['name']:<18} | {idx['radius_km']:>3}km | {idx['num_entries']:>6,} pts | {size_mb:>4.0f}MB | {badge} by @{author}"
                        self._hub_listbox.insert('end', line)
                    self._hub_status.config(text=f"Found {len(results)} indexes for '{query}'")

                    self.master.after(0, update_ui)
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._hub_status.config(text=f"Search error: {m}"))

        threading.Thread(target=do_search, daemon=True).start()

    def _hub_download(self, hub_win):
        sel = self._hub_listbox.curselection()
        if not sel or not self._hub_indexes:
            self._hub_status.config(text="Select an index first")
            return

        idx = self._hub_indexes[sel[0]]
        repo_id = idx.get('repo_id', '')
        name = idx.get('name', 'Unknown')

        self._hub_status.config(text=f"Downloading {name}...")
        hub_win.update_idletasks()

        def do_download():
            try:
                hub = AreaHub()
                manifest = hub.download(
                    repo_id, COMPACT_INDEX_DIR,
                    progress_callback=lambda msg: self.master.after(0, lambda m=msg: self._hub_status.config(text=m))
                )

                global _compact_cache
                _compact_cache = None

                def on_done():
                    self._hub_status.config(text=f"✅ Downloaded {name}! Ready to search.")
                    self._set_status(f"Index loaded: {name}")
                    if manifest:
                        self.lat_var.set(manifest.get('center_lat', self.lat_var.get()))
                        self.lon_var.set(manifest.get('center_lon', self.lon_var.get()))
                        self.radius_var.set(manifest.get('radius_km', self.radius_var.get()))
                        self.map_widget.set_position(manifest['center_lat'], manifest['center_lon'])
                        self.map_widget.set_zoom(13)

                self.master.after(0, on_done)
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._hub_status.config(text=f"Download error: {m}"))

        threading.Thread(target=do_download, daemon=True).start()

    def _hub_upload(self, hub_win):
        if not os.path.exists(COMPACT_DESCS_PATH):
            self._hub_status.config(text="No index to upload. Create one first.")
            return


        upload_win = tk.Toplevel(hub_win)
        upload_win.title("Upload Index")
        upload_win.configure(bg='#0a0c10')
        upload_win.geometry("400x350")
        upload_win.transient(hub_win)

        tk.Label(upload_win, text="Upload to Community Hub", font=('Inter', 14, 'bold'),
                 bg='#0a0c10', fg='#ffffff').pack(pady=(20, 15))

        fields_frame = tk.Frame(upload_win, bg='#0a0c10')
        fields_frame.pack(padx=20, fill='x')

        city_var = tk.StringVar(value="")
        radius_var = tk.StringVar(value=str(self.radius_var.get()))
        lat_var = tk.StringVar(value=str(self.lat_var.get()))
        lon_var = tk.StringVar(value=str(self.lon_var.get()))
        tags_var = tk.StringVar(value="")

        for label, var in [("City name:", city_var), ("Radius (km):", radius_var),
                           ("Center Lat:", lat_var), ("Center Lon:", lon_var),
                           ("Tags (comma-sep):", tags_var)]:
            row = tk.Frame(fields_frame, bg='#0a0c10')
            row.pack(fill='x', pady=4)
            tk.Label(row, text=label, font=('Inter', 9), bg='#0a0c10', fg='#94a3b8',
                     width=16, anchor='w').pack(side='left')
            tk.Entry(row, textvariable=var, font=('Inter', 9), bg='#141720', fg='#ffffff',
                     insertbackground='white', borderwidth=0, highlightthickness=1,
                     highlightcolor='#475569', highlightbackground='#232834').pack(side='left', fill='x', expand=True, ipady=4)

        status_lbl = tk.Label(upload_win, text="", font=('Inter', 9), bg='#0a0c10', fg='#94a3b8')
        status_lbl.pack(pady=(10, 5))

        def do_upload():
            city = city_var.get().strip()
            if not city:
                status_lbl.config(text="City name required")
                return

            status_lbl.config(text="Uploading...")
            upload_win.update_idletasks()

            def upload_thread():
                try:
                    token = self.hf_token_var.get().strip()
                    if not token:
                        self.master.after(0, lambda: messagebox.showerror("Hugging Face Help", 
                            "Please connect Hugging Face to upload.\n\n"
                            "1. Click 'Get Hugging Face Token'\n"
                            "2. Generate a WRITE token\n"
                            "3. Paste it in the token field\n"
                            "4. Try uploading again"))
                        self.master.after(0, lambda: status_lbl.config(text="Token missing"))
                        return

                    os.environ["HF_TOKEN"] = token
                    hub = AreaHub(token=token)
                    
                    tags = [t.strip() for t in tags_var.get().split(',') if t.strip()]
                    url = hub.upload(
                        index_dir=COMPACT_INDEX_DIR,
                        city=city,
                        radius_km=float(radius_var.get()),
                        center_lat=float(lat_var.get()),
                        center_lon=float(lon_var.get()),
                        tags=tags,
                    )
                    self.master.after(0, lambda: status_lbl.config(text=f"✅ Uploaded! {url}"))
                    self.master.after(0, lambda: self._hub_status.config(text=f"Uploaded {city}!"))
                except Exception as e:
                    err_msg = str(e)
                    self.master.after(0, lambda m=err_msg: status_lbl.config(text=f"Error: {m}"))

            threading.Thread(target=upload_thread, daemon=True).start()

        self.final_up_btn = RoundedButton(upload_win, text="⬆  Start Upload",
                                         command=do_upload,
                                         width=180, height=42, bg_color='#ffffff',
                                         hover_color='#e2e8f0', pressed_color='#cbd5e1',
                                         text_color='#0a0c10')
        self.final_up_btn.pack(pady=(10, 20))

    def export_index(self):
        if not os.path.exists(COMPACT_DESCS_PATH):
            self._set_status("No index to export. Create one first.")
            return

        save_path = filedialog.asksaveasfilename(
            defaultextension=".area",
            filetypes=[("Area Index", "*.area")],
            title="Export Index As",
            initialfile=f"area_index_{int(self.radius_var.get())}km.area"
        )
        if not save_path:
            return

        self._set_status("Exporting index...")

        def do_export():
            try:
                from area_hub import create_bundle
                path, manifest = create_bundle(
                    index_dir=COMPACT_INDEX_DIR,
                    output_path=save_path,
                    name=f"Area Index {self.radius_var.get()}km",
                    description="Exported from Area",
                    center_lat=self.lat_var.get(),
                    center_lon=self.lon_var.get(),
                    radius_km=self.radius_var.get(),
                )
                size_mb = os.path.getsize(save_path) / 1024 / 1024
                self.master.after(0, lambda: self._set_status(
                    f"✅ Exported: {save_path} ({size_mb:.0f} MB)"))
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._set_status(f"Export error: {m}"))

        threading.Thread(target=do_export, daemon=True).start()

    def import_index(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Area Index", "*.area"), ("All files", "*.*")],
            title="Import Area Index"
        )
        if not file_path:
            return

        self._set_status("Importing index...")

        def do_import():
            try:
                from area_hub import extract_bundle
                manifest = extract_bundle(file_path, COMPACT_INDEX_DIR)

                global _compact_cache
                _compact_cache = None


                pca_path = os.path.join(COMPACT_INDEX_DIR, "area_pca.pkl")
                if os.path.exists(pca_path):
                    try:
                        from area_utils import load_pca
                        load_pca(pca_path)
                    except Exception:
                        pass

                def on_done():
                    self._set_status(f"✅ Imported: {manifest.get('name', 'Unknown')} — Ready to search!")
                    if manifest:
                        self.lat_var.set(manifest.get('center_lat', self.lat_var.get()))
                        self.lon_var.set(manifest.get('center_lon', self.lon_var.get()))
                        self.radius_var.set(manifest.get('radius_km', self.radius_var.get()))
                        self.map_widget.set_position(
                            manifest.get('center_lat', self.lat_var.get()),
                            manifest.get('center_lon', self.lon_var.get()))
                        self.map_widget.set_zoom(13)

                self.master.after(0, on_done)
            except Exception as e:
                err_msg = str(e)
                self.master.after(0, lambda m=err_msg: self._set_status(f"Import error: {m}"))

        threading.Thread(target=do_import, daemon=True).start()

    def show_help(self):
        help_win = tk.Toplevel(self.master)
        help_win.title("Area - Technical User Guide")
        help_win.geometry("850x750")
        help_win.configure(bg='#0a0c10')
        help_win.transient(self.master)

        main_frame = tk.Frame(help_win, bg='#0a0c10')
        main_frame.pack(fill='both', expand=True, padx=30, pady=30)

        title_lbl = tk.Label(main_frame, text="Area Engine Reference", 
                            font=('Inter', 20, 'bold'), bg='#0a0c10', fg='#ffffff')
        title_lbl.pack(anchor='w', pady=(0, 20))

        content_canvas = tk.Canvas(main_frame, bg='#0a0c10', highlightthickness=0)
        scrollbar = ttk.Scrollbar(main_frame, orient="vertical", command=content_canvas.yview)
        scrollable_frame = tk.Frame(content_canvas, bg='#0a0c10')

        scrollable_frame.bind(
            "<Configure>",
            lambda e: content_canvas.configure(scrollregion=content_canvas.bbox("all"))
        )

        content_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw", width=770)
        content_canvas.configure(yscrollcommand=scrollbar.set)

        content_canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        sections = [
            ("Core Logic and Technical Overview", 
             "Area is built on a global-to-local visual search pipeline. It is designed to find a specific location in an urban environment by comparing your query image against a vast database of pre-indexed street-level views. The system does not rely on GPS metadata from your photo; instead, it looks at the actual architecture, textures, and spatial relationships in the scene."),

            ("The Search Pipeline", 
             "When you run a search, Area goes through three distinct stages to ensure millimetric accuracy:\n\n"
             "1. Global Retrieval (Area-loc): The system extracts a high-level visual signature from your photo and scans the entire city index. It identifies the top 100 most similar locations based on broad visual features.\n\n"
             "2. Dense Geometric Matching (Area-3R): For the top candidates found in Stage 1, we pull the original panoramas and perform an extremely detailed point-to-point comparison. This stage finds thousands of tiny matching 'patches' between the images to confirm they are the same spot.\n\n"
             "3. Spatial Consensus: To prevent errors caused by repetitive architecture (like identical-looking chain stores), candidates are clustered into geographic groups. A location is only confirmed if multiple nearby images also match well, ensuring that isolated false positives are ignored."),

            ("Working with City Indexes", 
             "A city index is a collection of mathematical descriptors for every street-level view in a given radius. You can manage these in several ways:\n\n"
             "• **Building a New Index**: Switch the main mode to 'Create Index', set your center point and radius on the map, and click Run. The system will download panoramas and build the database locally.\n\n"
             "• **Community Hub**: Browse and download pre-built city indexes directly into your local database. This saves you hours of processing time.\n\n"
             "• **Contributing**: Have a GPU and the time to index your neighborhood? Use the 'Upload' button in the Community Hub to share your index with the world.\n\n"
             "### Hugging Face Tokens\n"
             "To upload and contribute city indexes, you need a Hugging Face Access Token:\n"
             "1. Create a free account at [huggingface.co](https://huggingface.co).\n"
             "2. Go to **Settings > Access Tokens**.\n"
             "3. Create a new token with **'Write'** permissions.\n"
             "4. Paste it into the token field in the sidebar.\n\n"
             "### Coordinate Search\n"
             "You can manually enter coordinates or paste them into the Lat/Lon fields. Use the 'Power Actions' (Copy/Open Maps) on search results to extract coordinates for external use.\n"),

            ("Search Parameters and Calibration", 
             "For the best results, you should fine-tune your search based on the city's density:\n\n"
             "• Grid Resolution: This is the gap between scan points. For broad coverage and general mapping, a 300-meter resolution is highly recommended. For extreme precision in dense urban areas, you can use 25-50 meters.\n\n"
             "• Match Threshold: This controls how picky the Stage 1 retrieval is. A higher threshold (0.80+) is faster but might miss subtle matches. Lowering it (0.60) can help in difficult conditions like light or weather changes."),

            ("Practical Tips for Researchers", 
             "• Orientation Matters: If the AI-assigned heading seems off, try rotating your query image or adjusting the step size during indexing.\n\n"
             "• Local Storage: Indexes are stored on your Expansion drive whenever possible to save space on your primary disk. Large city indexes can exceed several gigabytes.")
        ]

        for sec_title, sec_text in sections:
            s_frame = tk.Frame(scrollable_frame, bg='#0a0c10', pady=16)
            s_frame.pack(fill='x')
            
            tk.Label(s_frame, text=sec_title, font=('Inter', 13, 'bold'), 
                     bg='#0a0c10', fg='#ffffff').pack(anchor='w')
            
            tk.Label(s_frame, text=sec_text, font=('Inter', 10), 
                     bg='#0a0c10', fg='#94a3b8', justify='left', wraplength=730).pack(anchor='w', pady=(6, 0))
            
            tk.Frame(s_frame, bg='#232834', height=1).pack(fill='x', pady=(16, 0))

        close_btn = RoundedButton(help_win, text="Return to Console", font=('Inter', 10, 'bold'),
                                  bg_color='#191d28', hover_color='#222734', pressed_color='#141720',
                                  text_color='#f1f5f9', border_color='#2c3340', width=200, height=40,
                                  command=help_win.destroy)
        close_btn.pack(pady=20)

    def poll_match_queue(self):
        try:
            for _ in range(20):
                msg = self.match_queue.get_nowait()
                if msg[0] == 'status':
                    if hasattr(self, 'status_label') and self.status_label.winfo_exists():
                        self.status_label.config(text=msg[1])
                        self.master.update_idletasks()
                elif msg[0] == 'progress':
                    if hasattr(self, 'progress') and self.progress.winfo_exists():
                        self.progress['maximum'] = msg[2]
                        self.progress['value'] = msg[1]
                elif msg[0] == 'match_update':
                    match_res = msg[1]
                    if hasattr(self, 'current_search_context'):
                        self._clear_coverage_markers()
                        self._handle_match_done(match_res, *self.current_search_context)
                elif msg[0] == 'scan_blip':
                    self._handle_scan_blip(msg[1], msg[2], msg[3], msg[4] if len(msg) > 4 else None)
        except queue.Empty:
            pass
        self.master.after(100, self.poll_match_queue)





if __name__ == "__main__":

    for d in [DATA_DIR, AREA_PARTS_DIR, COMPACT_INDEX_DIR]:
        os.makedirs(d, exist_ok=True)

    root = tk.Tk()
    app = StreetViewMatcherGUI(root)
    root.mainloop()
