# stego_core.py
import io, json, math, random, base64, secrets
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import cv2
from scipy.spatial import KDTree
from PIL import Image, PngImagePlugin

# ---------- 基礎工具 ----------

def _list_image_files(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in exts])

def _bgr_to_lab(img_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)

def _tile_feature_mean_lab(tile_bgr: np.ndarray) -> np.ndarray:
    lab = _bgr_to_lab(tile_bgr)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)

# ---------- Atlas / Tiles ----------

def build_atlas_from_folder(tile_folder: str, tile_size: Tuple[int,int],
                            seed: int, keep_order: bool=False):
    folder = Path(tile_folder)
    files = _list_image_files(folder)
    if not files:
        raise FileNotFoundError(f"素材庫為空：{tile_folder}")

    n = len(files)
    side = int(math.sqrt(n))
    if side * side == n:
        rows = cols = side
    else:
        cols = int(math.ceil(math.sqrt(n)))
        rows = int(math.ceil(n / cols))

    idxs = list(range(n))
    if not keep_order:
        random.Random(seed).shuffle(idxs)

    tw, th = tile_size
    atlas = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    tiles = []
    for i, idx in enumerate(idxs):
        r, c = divmod(i, cols)
        img = cv2.imread(str(files[idx]), cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError(f"讀不到影像：{files[idx]}")
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        y0, y1 = r * th, (r + 1) * th
        x0, x1 = c * tw, (c + 1) * tw
        atlas[y0:y1, x0:x1] = img
        tiles.append(img)

    return atlas, tiles, rows, cols

def build_tile_feature_kdtree(tiles: List[np.ndarray]):
    feats = np.vstack([_tile_feature_mean_lab(t) for t in tiles])
    return KDTree(feats)

def split_image_into_blocks(img_bgr: np.ndarray, block_size: Tuple[int,int]) -> np.ndarray:
    bw, bh = block_size
    H, W = img_bgr.shape[:2]
    if (H % bh) or (W % bw):
        W2 = (W // bw) * bw
        H2 = (H // bh) * bh
        img_bgr = cv2.resize(img_bgr, (W2, H2), interpolation=cv2.INTER_AREA)
        H, W = H2, W2
    rows, cols = H // bh, W // bw
    blocks = np.zeros((rows, cols, bh, bw, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r*bh, (r+1)*bh
            x0, x1 = c*bw, (c+1)*bw
            blocks[r, c] = img_bgr[y0:y1, x0:x1]
    return blocks

def assign_tiles_to_blocks(blocks: np.ndarray, tile_tree: KDTree, tiles: List[np.ndarray]) -> np.ndarray:
    rows, cols = blocks.shape[:2]
    idx_map = np.zeros((rows, cols), dtype=np.int32)
    for r in range(rows):
        for c in range(cols):
            feat = _tile_feature_mean_lab(blocks[r, c])
            _, idx = tile_tree.query(feat, k=1)
            idx_map[r, c] = int(idx)
    return idx_map

def render_mosaic(idx_map: np.ndarray, tiles: List[np.ndarray], tile_size: Tuple[int,int]) -> np.ndarray:
    tw, th = tile_size
    rows, cols = idx_map.shape
    out = np.zeros((rows*th, cols*tw, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            t = tiles[idx_map[r, c]]
            if t.shape[1] != tw or t.shape[0] != th:
                t = cv2.resize(t, (tw, th), interpolation=cv2.INTER_AREA)
            y0, y1 = r*th, (r+1)*th
            x0, x1 = c*tw, (c+1)*tw
            out[y0:y1, x0:x1] = t
    return out

# ---------- MRT/SIE：mod-M 於 Y 通道 ----------

def _id_to_coords_mrt(id_val: int, M: int, N: int, perm_seed: int):
    total = M ** N
    rng = random.Random(perm_seed)
    perm = list(range(total))
    rng.shuffle(perm)
    sym = perm[id_val % total]
    coords = [0]*N
    for i in range(N-1, -1, -1):
        coords[i] = sym % M
        sym //= M
    return tuple(coords)

def embed_mrt_sie_to_y(index_map_flat: List[int], carrier_bgr: np.ndarray,
                       M: int, N: int, perm_seed: int,
                       shuffle_pixels: bool, pixel_seed: int):
    ycc = cv2.cvtColor(carrier_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)
    H, W = Y.shape
    total_pixels = H * W
    need = len(index_map_flat) * N
    if need > total_pixels:
        raise ValueError(f"容量不足：需要 {need}，實際 {total_pixels}")

    positions = list(range(total_pixels))
    if shuffle_pixels:
        rng = random.Random(pixel_seed)
        rng.shuffle(positions)

    coords_stream = []
    p = 0
    for tid in index_map_flat:
        coords = _id_to_coords_mrt(tid, M, N, perm_seed)
        for a in coords:
            pos = positions[p]
            y = int(Y.flat[pos])
            target = a % M
            delta = (target - (y % M)) % M
            if delta > M // 2:
                delta -= M
            Y.flat[pos] = np.clip(y + delta, 0, 255)
            coords_stream.append(int(target))
            p += 1

    ycc[:, :, 0] = Y.astype(np.uint8)
    stego_bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return stego_bgr, coords_stream

def extract_coords_from_stego(stego_bgr: np.ndarray, M: int,
                              shuffle_pixels: bool, pixel_seed: int) -> List[int]:
    ycc = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0]
    H, W = Y.shape
    total = H * W
    positions = list(range(total))
    if shuffle_pixels:
        rng = random.Random(pixel_seed)
        rng.shuffle(positions)
    coords = [int(Y.flat[pos]) % M for pos in positions]
    return coords

def decode_ids(coords_stream: List[int], M: int, N: int, perm_seed: int, symbols: int) -> List[int]:
    total = M ** N
    rng = random.Random(perm_seed)
    perm = list(range(total))
    rng.shuffle(perm)
    inv_perm = {v: i for i, v in enumerate(perm)}
    ids = []
    for i in range(0, symbols * N, N):
        val = 0
        for x in coords_stream[i:i+N]:
            val = val * M + x
        ids.append(inv_perm.get(val, 0))
    return ids

# ---------- 封裝：加密 / 解密 ----------

def encrypt(secret_bgr: np.ndarray, carrier_bgr: np.ndarray, tile_folder: str,
            tile_size=(16, 16),
            # 固定 M=4, N=4
            M=4, N=4,
            # 三個種子自動產生（None 時）
            perm_seed: Optional[int] = None,
            shuffle_pixels: bool = True,
            pixel_seed: Optional[int] = None,
            atlas_seed: Optional[int] = None,
            keep_order: bool = False):

    # ---------- 容量檢查與自動縮放（不改載體、不改 tile_size） ----------
    # 每符號需要 N 個像素；可嵌容量 = carrier_pixels // N
    tw, th = tile_size
    blocks = split_image_into_blocks(secret_bgr, tile_size)
    rows, cols = blocks.shape[:2]
    symbols = rows * cols
    capacity = (carrier_bgr.shape[0] * carrier_bgr.shape[1]) // N

    original_h, original_w = secret_bgr.shape[:2]

    if symbols > capacity:
        # 目標倍率（等比縮小 secret，使 symbols <= capacity）
        ratio = np.sqrt(capacity / symbols)
        new_w = max(tw, int(secret_bgr.shape[1] * ratio))
        new_h = max(th, int(secret_bgr.shape[0] * ratio))
        secret_bgr = cv2.resize(secret_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # 重新分塊並保險：若仍超過容量，逐步再縮一點點直到符合
        blocks = split_image_into_blocks(secret_bgr, tile_size)
        rows, cols = blocks.shape[:2]
        while (rows * cols) > capacity:
            new_w = max(tw, int(new_w * 0.98))
            new_h = max(th, int(new_h * 0.98))
            secret_bgr = cv2.resize(secret_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
            blocks = split_image_into_blocks(secret_bgr, tile_size)
            rows, cols = blocks.shape[:2]
        # print(f"[WARN] secret resized to {secret_bgr.shape[1]}x{secret_bgr.shape[0]} for capacity {capacity}")

    # ---------- 種子（64-bit） ----------
    if perm_seed is None:
        perm_seed = secrets.randbits(64)
    if pixel_seed is None:
        pixel_seed = secrets.randbits(64)
    if atlas_seed is None:
        atlas_seed = secrets.randbits(64)

    # ---------- Atlas / KDTree ----------
    atlas_bgr, tiles, a_rows, a_cols = build_atlas_from_folder(
        tile_folder, tile_size, seed=int(atlas_seed), keep_order=keep_order
    )
    tree = build_tile_feature_kdtree(tiles)

    # ---------- 祕密圖 → Mosaic ----------
    # （注意：上面可能已經把 blocks 準備好；這裡保險再做一次）
    blocks = split_image_into_blocks(secret_bgr, tile_size)
    idx_map = assign_tiles_to_blocks(blocks, tree, tiles)
    mosaic_bgr = render_mosaic(idx_map, tiles, tile_size)
    index_map_flat = idx_map.reshape(-1).tolist()

    # ---------- 嵌入 ----------
    stego_bgr, _ = embed_mrt_sie_to_y(
        index_map_flat, carrier_bgr, M, N, int(perm_seed),
        shuffle_pixels, int(pixel_seed)
    )

    # ---------- 打包金鑰（含 Atlas Base64） ----------
    ok, atlas_png_buf = cv2.imencode(".png", atlas_bgr)
    if not ok:
        raise RuntimeError("Atlas 轉 PNG 失敗")
    atlas_b64 = base64.b64encode(atlas_png_buf.tobytes()).decode("ascii")

    key = {
        "scheme": "MRT-SIE_modM_Y",
        "M": 4, "N": 4,
        "perm_seed": int(perm_seed),
        "shuffle_pixels": True,
        "pixel_seed": int(pixel_seed),
        "tile_size": [int(tw), int(th)],
        "mosaic_rows": int(idx_map.shape[0]),
        "mosaic_cols": int(idx_map.shape[1]),
        "symbols": len(index_map_flat),
        "coords_per_symbol": N,
        "index_map_len": len(index_map_flat),
        "secret_size_before": [int(original_w), int(original_h)],
        "secret_size_after": [int(secret_bgr.shape[1]), int(secret_bgr.shape[0])],
        "atlas": {
            "rows": int(a_rows),
            "cols": int(a_cols),
            "seed": int(atlas_seed),
            "keep_order": keep_order,
            "png_base64": atlas_b64
        }
    }

    return {"stego_bgr": stego_bgr, "mosaic_bgr": mosaic_bgr, "key": key}

def save_stego_no_metadata(stego_bgr: np.ndarray) -> bytes:
    _, buf = cv2.imencode(".png", stego_bgr)
    return buf.tobytes()


def decrypt(stego_bytes: bytes, key_json: Optional[bytes] = None) -> bytes:
    pil = Image.open(io.BytesIO(stego_bytes))
    key_text = None
    if isinstance(pil.info, dict):
        key_text = pil.info.get("mrt_sie_key")
    if (not key_text) and key_json:
        key_text = key_json.decode("utf-8")
    if not key_text:
        raise ValueError("找不到金鑰：PNG metadata 無 'mrt_sie_key'，也未提供金鑰檔。")

    key = json.loads(key_text)

    # 取座標流
    stego_bgr = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    coords_stream = extract_coords_from_stego(
        stego_bgr, M=key["M"], shuffle_pixels=key["shuffle_pixels"], pixel_seed=key["pixel_seed"]
    )
    coords_stream = coords_stream[: key["symbols"] * key["coords_per_symbol"]]

    # 解出 tile IDs
    ids = decode_ids(coords_stream, M=key["M"], N=key["N"],
                     perm_seed=key["perm_seed"], symbols=key["symbols"])

    # 還原 Atlas 與 Mosaic
    atlas_png = base64.b64decode(key["atlas"]["png_base64"])
    atlas_np = cv2.imdecode(np.frombuffer(atlas_png, np.uint8), cv2.IMREAD_COLOR)

    tw, th = key["tile_size"]
    rows, cols = key["mosaic_rows"], key["mosaic_cols"]
    a_rows, a_cols = key["atlas"]["rows"], key["atlas"]["cols"]

    # 切 atlas 成 tiles
    tiles = []
    for r in range(a_rows):
        for c in range(a_cols):
            y0, y1 = r*th, (r+1)*th
            x0, x1 = c*tw, (c+1)*tw
            tiles.append(atlas_np[y0:y1, x0:x1])

    # 拼回 mosaic
    ids = np.array(ids[: rows*cols], dtype=np.int32) % len(tiles)
    mosaic = np.zeros((rows*th, cols*tw, 3), np.uint8)
    for i, tid in enumerate(ids):
        r, c = divmod(i, cols)
        mosaic[r*th:(r+1)*th, c*tw:(c+1)*tw] = tiles[tid]

    ok, mosaic_png = cv2.imencode(".png", mosaic)
    if not ok:
        raise RuntimeError("輸出 Mosaic 失敗")
    return mosaic_png.tobytes()
