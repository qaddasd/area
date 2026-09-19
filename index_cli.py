#!/usr/bin/env python3
"""High-speed CLI Indexing tool for Area geolocation system.

Optimized for NVIDIA GPUs (such as GTX 1650 4GB) with zero-copy GPU pipeline,
batched neural network feature extraction, and asynchronous Street View downloading.
"""

import os
import sys
import argparse
import math
import itertools
import time
import json
import glob
import re
import queue
import threading
import asyncio
import aiohttp
import numpy as np
import torch
from PIL import Image
import io

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from area_utils import (
    get_area_model, batch_extract_area, fit_pca, save_pca,
    AREA_RAW_DIM, AREA_PCA_DIM, AREA_INPUT_SIZE
)

device = 'cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu')


def haversine(p1, p2):
    R = 6371.0
    lat1, lon1 = map(math.radians, p1)
    lat2, lon2 = map(math.radians, p2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2.0) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2.0) ** 2
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))


def grid_points(center, radius_km, resolution):
    lat, lon = center
    top_left = (lat - radius_km / 70.0, lon + radius_km / 70.0)
    bottom_right = (lat + radius_km / 70.0, lon - radius_km / 70.0)
    lat_diff = top_left[0] - bottom_right[0]
    lon_diff = top_left[1] - bottom_right[1]
    raw_points = list(itertools.product(range(resolution + 1), range(resolution + 1)))
    points = [
        (bottom_right[0] + x * lat_diff / resolution, bottom_right[1] + y * lon_diff / resolution)
        for (x, y) in raw_points
    ]
    return [p for p in points if haversine(p, center) <= radius_km]


def _panoids_url(lat, lon):
    return (
        f"https://maps.googleapis.com/maps/api/js/GeoPhotoService.SingleImageSearch"
        f"?pb=!1m5!1sapiv3!5sUS!11m2!1m1!1b0!2m4!1m2!3d{lat}!4d{lon}!2d50!3m10!2m2!1sen!2sGB"
        f"!9m1!1e2!11m4!1m3!1e2!2b1!3e2!4m10!1e1!1e2!1e3!1e4!1e8!1e6!5m1!1e2!6m1!1e2&callback=_xdc_._v2mub5"
    )


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


async def fetch_panoids_async(points, max_workers=32):
    results = []
    seen = set()
    connector = aiohttp.TCPConnector(limit=max_workers)
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}

    async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
        async def fetch_one(lat, lon):
            url = _panoids_url(lat, lon)
            for attempt in range(4):
                try:
                    async with session.get(url, timeout=15) as resp:
                        if resp.status == 200:
                            text = await resp.text()
                            return panoids_from_response(text)
                        elif resp.status == 429:
                            await asyncio.sleep(1.5)
                except Exception:
                    await asyncio.sleep(0.5)
            return []

        tasks = [asyncio.create_task(fetch_one(lat, lon)) for lat, lon in points]
        for idx, task in enumerate(asyncio.as_completed(tasks), 1):
            pans = await task
            for p in pans:
                if p['panoid'] not in seen:
                    seen.add(p['panoid'])
                    results.append(p)
            if idx % 25 == 0 or idx == len(points):
                print(f"[Discovery] Probed {idx}/{len(points)} grid points -> {len(results)} unique panoids found")

    return results


def tiles_info(panoid):
    url_template = "https://streetviewpixels-pa.googleapis.com/v1/tile?panoid={0}&x={1}&y={2}&zoom=2&cb_client=maps_sv.tactile"
    return [(x, y, url_template.format(panoid, x, y)) for x in range(4) for y in range(2)]


async def download_tiles_aiohttp(session, tiles):
    results = {}
    async def get_tile(x, y, url):
        for _ in range(3):
            try:
                async with session.get(url, timeout=12) as resp:
                    if resp.status == 200:
                        data = await resp.read()
                        return x, y, data
            except Exception:
                await asyncio.sleep(0.5)
        return x, y, None

    tasks = [get_tile(x, y, url) for x, y, url in tiles]
    for coro in asyncio.as_completed(tasks):
        x, y, data = await coro
        if data:
            results[(x, y)] = data
    return results


def stitch_tiles(tiles_data):
    tile_w, tile_h = 512, 512
    pano_np = np.zeros((2 * tile_h, 4 * tile_w, 3), dtype=np.uint8)
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


def equirectangular_to_rectilinear_torch(pano_tensor, fov_deg=90, out_hw=(322, 322), yaw_deg=0, base_dirs=None):
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


def run_indexing(city_name, lat, lon, radius_km=1.5, grid_res=25, crop_size=322, heading_step=90, max_download_workers=32):
    print("=" * 60)
    print(f"  Area High-Speed Indexing - {city_name}")
    print(f"  Coordinates: {lat:.4f}, {lon:.4f} | Radius: {radius_km:.1f} km")
    print(f"  Device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print("=" * 60)

    data_dir = os.path.join(PROJECT_ROOT, "area_data")
    parts_dir = os.path.join(data_dir, "area_parts")
    index_dir = os.path.join(data_dir, "index")
    os.makedirs(parts_dir, exist_ok=True)
    os.makedirs(index_dir, exist_ok=True)

    # Pre-load Area-loc model into GPU
    print("\n[Init] Pre-loading Area-loc model on GPU...")
    model = get_area_model(device)
    print(f"[Init] Area-loc model active. Target crop resolution: {crop_size}x{crop_size}")

    # Check existing files
    existing_panos = set()
    for pf in glob.glob(os.path.join(parts_dir, "area_part_*.npz")):
        try:
            d = np.load(pf, allow_pickle=True)
            for p in d['paths']:
                fname = os.path.basename(str(p)).replace('.npz', '')
                pid = fname.rsplit('_', 1)[0]
                existing_panos.add(pid)
        except Exception:
            pass
    if existing_panos:
        print(f"[Init] Found {len(existing_panos)} previously indexed panoids. They will be skipped.")

    # Step 1: Grid Discovery
    print(f"\n[Step 1/3] Generating grid points (radius: {radius_km} km, resolution: {grid_res})...")
    center = (lat, lon)
    pts = grid_points(center, radius_km, grid_res)
    print(f"[Step 1/3] Generated {len(pts)} grid points. Discovering Google Street View panoids...")

    panoids_list = asyncio.run(fetch_panoids_async(pts, max_workers=max_download_workers))
    print(f"[Step 1/3] Found {len(panoids_list)} total unique scan nodes in {city_name}!")

    # Filter out already indexed
    to_process = [p for p in panoids_list if p['panoid'] not in existing_panos]
    print(f"[Step 1/3] {len(to_process)} scan nodes to download and index ({len(panoids_list) - len(to_process)} already cached).")

    if not to_process and len(existing_panos) == 0:
        print("[ERROR] No scan nodes found in this area. Check coordinates or internet connection.")
        return False

    headings = sorted(list(set(((h // heading_step) * heading_step) % 360 for h in range(0, 360, heading_step))))
    base_dirs = get_projection_base_dirs(90, (crop_size, crop_size))

    # Step 2: Download & Extract
    print(f"\n[Step 2/3] Processing {len(to_process)} scan nodes (4 crops each = {len(to_process)*len(headings)} descriptors)...")

    buffer_descs = []
    buffer_paths = []
    buffer_lats = []
    buffer_lons = []
    total_indexed = 0
    t_start = time.time()

    def save_chunk():
        nonlocal buffer_descs, buffer_paths, buffer_lats, buffer_lons
        if not buffer_descs:
            return
        ts = int(time.time() * 1000)
        part_path = os.path.join(parts_dir, f"area_part_{ts}.npz")
        all_d = np.vstack(buffer_descs)
        np.savez_compressed(
            part_path,
            descriptors=all_d,
            paths=np.array(buffer_paths, dtype=object),
            lats=np.array(buffer_lats, dtype=np.float32),
            lons=np.array(buffer_lons, dtype=np.float32)
        )
        emb_csv = os.path.join(data_dir, "embeddings_index.csv")
        with open(emb_csv, "a", encoding="utf-8") as f_csv:
            for p, la, lo in zip(buffer_paths, buffer_lats, buffer_lons):
                f_csv.write(f"{p},{la},{lo}\n")
        print(f"  [Chunk Saved] {len(buffer_paths)} descriptors written to {os.path.basename(part_path)}")
        buffer_descs.clear()
        buffer_paths.clear()
        buffer_lats.clear()
        buffer_lons.clear()

    # Async download queue & GPU worker
    download_queue = queue.Queue(maxsize=32)

    async def downloader():
        connector = aiohttp.TCPConnector(limit=max_download_workers)
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        async with aiohttp.ClientSession(connector=connector, headers=headers) as session:
            async def get_pano(pinfo):
                pid = pinfo['panoid']
                # Fast single-request full panorama download
                url = f"https://streetviewpixels-pa.googleapis.com/v1/thumbnail?panoid={pid}&cb_client=maps_sv.tactile&w=2048&h=1024"
                for attempt in range(3):
                    try:
                        async with session.get(url, timeout=12) as resp:
                            if resp.status == 200:
                                data = await resp.read()
                                if len(data) > 5000:
                                    img = Image.open(io.BytesIO(data))
                                    return pinfo, img
                            elif resp.status == 429:
                                await asyncio.sleep(1.0)
                    except Exception:
                        await asyncio.sleep(0.5)

                # Fallback to tiles if thumbnail fails
                try:
                    tiles = tiles_info(pid)
                    td = await download_tiles_aiohttp(session, tiles)
                    if len(td) >= 6:
                        img = stitch_tiles(td)
                        return pinfo, img
                except Exception:
                    pass
                return pinfo, None

            sem = asyncio.Semaphore(max_download_workers)
            async def worker(pinfo):
                async with sem:
                    res = await get_pano(pinfo)
                    download_queue.put(res)

            tasks = [asyncio.create_task(worker(pinfo)) for pinfo in to_process]
            await asyncio.gather(*tasks)

        download_queue.put(None)

    def download_runner():
        asyncio.run(downloader())

    dl_thread = threading.Thread(target=download_runner, daemon=True)
    dl_thread.start()

    processed_count = 0
    gpu_batch_crops = []
    gpu_batch_meta = []

    while True:
        item = download_queue.get()
        if item is None:
            break
        pinfo, pano_img = item
        processed_count += 1

        if pano_img is not None:
            maxw = 2048
            if pano_img.size[0] > maxw:
                pano_img = pano_img.resize((maxw, int(pano_img.size[1] * (maxw / pano_img.size[0]))), Image.BILINEAR)

            pano_np = np.array(pano_img.convert('RGB'))
            pano_t = torch.from_numpy(pano_np).float().permute(2, 0, 1).unsqueeze(0).div(255.0).to(device)

            with torch.no_grad():
                crops_torch = equirectangular_to_rectilinear_torch(
                    pano_t, fov_deg=90, out_hw=(crop_size, crop_size),
                    yaw_deg=headings, base_dirs=base_dirs
                )

            for i, yaw in enumerate(headings):
                crop_t = crops_torch[i].unsqueeze(0)
                emb_name = f"{pinfo['panoid']}_{yaw}.npz"
                meta = {'path': emb_name, 'lat': pinfo['lat'], 'lon': pinfo['lon'], 'yaw': yaw}
                gpu_batch_crops.append(crop_t)
                gpu_batch_meta.append(meta)

            del pano_t, crops_torch
            pano_img.close()

        # Run batched inference when batch reaches 32
        if len(gpu_batch_crops) >= 32:
            batch_tensor = torch.cat(gpu_batch_crops, dim=0)
            descs = batch_extract_area(batch_tensor, batch_size=len(gpu_batch_crops))
            buffer_descs.append(descs)
            buffer_paths.extend([m['path'] for m in gpu_batch_meta])
            buffer_lats.extend([m['lat'] for m in gpu_batch_meta])
            buffer_lons.extend([m['lon'] for m in gpu_batch_meta])
            total_indexed += len(gpu_batch_meta)
            gpu_batch_crops.clear()
            gpu_batch_meta.clear()

            if len(buffer_paths) >= 2000:
                save_chunk()

        if processed_count % 10 == 0 or processed_count == len(to_process):
            elapsed = time.time() - t_start
            rate = processed_count / max(elapsed, 0.1)
            eta_s = (len(to_process) - processed_count) / max(rate, 0.01)
            print(f"  [{processed_count}/{len(to_process)}] Panoids: {processed_count} ({total_indexed} descriptors) | "
                  f"{rate:.1f} panos/sec | ETA: {int(eta_s//60):02d}:{int(eta_s%60):02d}")

    # Flush remaining batch
    if gpu_batch_crops:
        batch_tensor = torch.cat(gpu_batch_crops, dim=0)
        descs = batch_extract_area(batch_tensor, batch_size=len(gpu_batch_crops))
        buffer_descs.append(descs)
        buffer_paths.extend([m['path'] for m in gpu_batch_meta])
        buffer_lats.extend([m['lat'] for m in gpu_batch_meta])
        buffer_lons.extend([m['lon'] for m in gpu_batch_meta])
        total_indexed += len(gpu_batch_meta)
        gpu_batch_crops.clear()
        gpu_batch_meta.clear()

    save_chunk()
    dl_thread.join()
    print(f"\n[Step 2/3 Complete] Processed {processed_count} panoids. Total descriptors: {total_indexed} in {time.time()-t_start:.1f}s")

    # Step 3: Build compact index & PCA
    print(f"\n[Step 3/3] Building compact search index and fitting PCA...")
    from test_super import build_compact_index
    build_compact_index()

    # Update manifest.json for Aktau
    manifest_path = os.path.join(index_dir, "manifest.json")
    manifest_data = {
        "format_version": "2.0",
        "name": f"{city_name} {radius_km:.1f}km",
        "description": f"Area-loc index for {city_name}, {radius_km:.1f}km radius",
        "creator": "Area-Fast-Indexer",
        "created_at": time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
        "center_lat": lat,
        "center_lon": lon,
        "radius_km": radius_km,
        "num_entries": total_indexed,
        "num_panoids": len(panoids_list),
        "descriptor_dim": AREA_PCA_DIM,
        "raw_descriptor_dim": AREA_RAW_DIM,
        "descriptor_model": "Area-loc",
        "pca_components": AREA_PCA_DIM,
        "heading_step_deg": heading_step,
        "crop_fov_deg": 90,
        "crop_size_px": crop_size,
        "tags": [city_name.lower(), "kazakhstan"]
    }
    with open(manifest_path, 'w', encoding='utf-8') as f:
        json.dump(manifest_data, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 60)
    print(f"  INDEXING FINISHED SUCCESSFULLY!")
    print(f"  City: {city_name} ({lat}, {lon})")
    print(f"  Panoids found: {len(panoids_list)} | Descriptors: {total_indexed}")
    print(f"  Index saved to: {index_dir}")
    print("=" * 60)
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Area High-Speed Indexer")
    parser.add_argument("--city", type=str, default="Aktau", help="City name")
    parser.add_argument("--lat", type=float, default=43.6480, help="Center latitude")
    parser.add_argument("--lon", type=float, default=51.1722, help="Center longitude")
    parser.add_argument("--radius", type=float, default=1.5, help="Search radius in km")
    parser.add_argument("--res", type=int, default=25, help="Grid resolution")
    parser.add_argument("--size", type=int, default=322, help="Crop size (multiple of 14)")
    parser.add_argument("--step", type=int, default=90, help="Heading step degrees")
    parser.add_argument("--workers", type=int, default=32, help="Max download workers")
    args = parser.parse_args()

    run_indexing(
        city_name=args.city,
        lat=args.lat,
        lon=args.lon,
        radius_km=args.radius,
        grid_res=args.res,
        crop_size=args.size,
        heading_step=args.step,
        max_download_workers=args.workers
    )
