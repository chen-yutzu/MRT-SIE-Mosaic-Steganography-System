# stego_core.py
import cv2, json, random, base64, os
import numpy as np
from pathlib import Path

# ===== 無損格式限制 =====
LOSSLESS_EXT = {".png", ".tif", ".tiff", ".bmp"}

def _check_lossless(path: Path):
    if path.suffix.lower() not in LOSSLESS_EXT:
        raise ValueError(f"❌ 僅支援無損格式（PNG/TIFF/BMP），收到：{path.suffix}")

# ===== 影像 <-> Base64 =====
def _img_to_b64(img: np.ndarray) -> str:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise RuntimeError("PNG 編碼失敗")
    return base64.b64encode(buf).decode("utf-8")

def _b64_to_img(b64: str) -> np.ndarray:
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError("Base64 影像解碼失敗")
    return img

# ===== MRT (M=4, N=4) 映射 =====
def _id_to_coords_mrt(id_val: int, M: int, N: int, perm_seed: int):
    total = M ** N
    rng = random.Random(perm_seed)
    perm = list(range(total))
    rng.shuffle(perm)              # secret permutation
    sym = perm[id_val % total]     # permuted symbol in [0, total)
    coords = [0] * N
    for i in range(N - 1, -1, -1):  # base-M digits
        coords[i] = sym % M
        sym //= M
    return tuple(coords)            # 長度 N，每一位 0..M-1

# ===== Atlas：從素材資料夾生成（不可視） =====
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

def _list_imgs(folder: Path):
    return sorted([p for p in folder.iterdir() if p.suffix.lower() in IMG_EXTS])

def _build_atlas_from_folder(tile_folder: Path, tile_size=(16,16), grid=16, seed=42, keep_order=False):
    """
    從素材資料夾讀圖 → resize 成固定 tile_size → 取前 grid*grid 張 → 拼成 Atlas (grid x grid)。
    回傳：(atlas_img, rows, cols, tile_h, tile_w)
    """
    files = _list_imgs(tile_folder)
    need = grid * grid
    if len(files) < need:
        raise ValueError(f"素材不足：需要至少 {need} 張，實際 {len(files)} 張（路徑：{tile_folder}）")

    idxs = list(range(len(files)))
    if not keep_order:
        rng = random.Random(seed)
        rng.shuffle(idxs)
    idxs = idxs[:need]

    tw, th = tile_size
    atlas = np.zeros((grid * th, grid * tw, 3), dtype=np.uint8)

    for i, k in enumerate(idxs):
        r, c = divmod(i, grid)
        img = cv2.imread(str(files[k]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"讀不到影像：{files[k]}")
        img = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        y0, y1 = r * th, (r + 1) * th
        x0, x1 = c * tw, (c + 1) * tw
        atlas[y0:y1, x0:x1] = img

    return atlas, grid, grid, th, tw

def _build_atlas_random(tile_size=(16,16), grid=16, seed=42):
    rng = np.random.default_rng(seed)
    th, tw = tile_size[1], tile_size[0]
    atlas = rng.integers(0, 255, size=(grid * th, grid * tw, 3), dtype=np.uint8)
    return atlas, grid, grid, th, tw

# ===== 馬賽克索引：用 Atlas tiles 與秘密圖塊做最近色指派 =====
def _bgr2lab_mean(img_bgr: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
    return lab.reshape(-1, 3).mean(axis=0).astype(np.float32)

def _split_blocks(img_bgr: np.ndarray, tile_w: int, tile_h: int):
    H, W = img_bgr.shape[:2]
    rows, cols = H // tile_h, W // tile_w
    blocks = []
    for r in range(rows):
        for c in range(cols):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            blocks.append(img_bgr[y0:y1, x0:x1])
    return blocks, rows, cols

def _tiles_from_atlas(atlas: np.ndarray, grid_r: int, grid_c: int, tile_h: int, tile_w: int):
    tiles = []
    for r in range(grid_r):
        for c in range(grid_c):
            y0, y1 = r * tile_h, (r + 1) * tile_h
            x0, x1 = c * tile_w, (c + 1) * tile_w
            tiles.append(atlas[y0:y1, x0:x1].copy())
    return tiles  # 長度 grid_r*grid_c

def _mosaic_index_map(secret_img: np.ndarray, atlas: np.ndarray, tile_w: int, tile_h: int, grid: int):
    # 切 atlas → 建 features
    tiles = _tiles_from_atlas(atlas, grid, grid, tile_h, tile_w)
    feats = np.vstack([_bgr2lab_mean(t) for t in tiles])  # (256, 3)

    # 秘密圖裁成整數塊
    H, W = secret_img.shape[:2]
    new_W = (W // tile_w) * tile_w
    new_H = (H // tile_h) * tile_h
    if new_W != W or new_H != H:
        secret_img = cv2.resize(secret_img, (new_W, new_H), interpolation=cv2.INTER_AREA)

    blocks, rows, cols = _split_blocks(secret_img, tile_w, tile_h)

    # 逐塊配對（無 SciPy，純 numpy）
    idx = np.empty(rows * cols, dtype=np.int32)
    for i, blk in enumerate(blocks):
        f = _bgr2lab_mean(blk)
        d = np.sum((feats - f) ** 2, axis=1)  # (256,)
        idx[i] = int(np.argmin(d))           # 0..255
    return idx.reshape(rows, cols), rows, cols

# ===== Embed：只曝光「秘密圖 + 載體圖」；Atlas 內建不可視且進金鑰 =====
def embed_mrt_sie(secret_img_path: str,
                  carrier_img_path: str,
                  out_stego_path: str,
                  key_path: str,
                  *,
                  tile_size: int = 16,
                  atlas_grid: int = 16,
                  tile_folder: str | None = None,
                  atlas_seed: int = 1234):
    """
    secret_img_path: 祕密圖片（僅用來生成 mosaic 索引，不會被寫進金鑰）
    carrier_img_path: 載體（嵌入在 Y 通道）
    out_stego_path: 輸出 stego 圖
    key_path: 輸出金鑰（含 Base64 atlas 與所有必要參數）
    tile_size: 每個 tile 的像素邊長（正方）
    atlas_grid: Atlas 的格數（預設 16x16 → 256 tiles）
    tile_folder: 系統內建素材資料夾（例如 "assets/tiles_256"）；若缺少或不足則改用隨機色塊 atlas
    """
    M, N = 4, 4
    perm_seed = random.randint(0, 1_000_000_000)
    pixel_seed = random.randint(0, 1_000_000_000)

    # --- 載入圖檔 ---
    _check_lossless(Path(carrier_img_path))
    carrier = cv2.imread(carrier_img_path, cv2.IMREAD_COLOR)
    if carrier is None:
        raise FileNotFoundError("❌ 讀不到載體圖")

    secret = cv2.imread(secret_img_path, cv2.IMREAD_COLOR)
    if secret is None:
        raise FileNotFoundError("❌ 讀不到祕密圖")

    # --- 構建 Atlas（優先用素材資料夾；不足則隨機） ---
    tile_h = tile_w = int(tile_size)
    atlas = None
    atlas_meta = {"source": "random", "grid": atlas_grid, "tile_size": tile_size}
    if tile_folder:
        folder = Path(tile_folder)
        try:
            atlas, ar, ac, th, tw = _build_atlas_from_folder(
                folder, tile_size=(tile_w, tile_h), grid=atlas_grid, seed=atlas_seed, keep_order=False
            )
            atlas_meta["source"] = "folder"
            atlas_meta["folder"] = str(folder)
        except Exception as e:
            # 退回隨機 Atlas
            atlas, ar, ac, th, tw = _build_atlas_random(tile_size=(tile_w, tile_h), grid=atlas_grid, seed=atlas_seed)
    else:
        atlas, ar, ac, th, tw = _build_atlas_random(tile_size=(tile_w, tile_h), grid=atlas_grid, seed=atlas_seed)

    # --- 以 Atlas 產生馬賽克索引（0..255） ---
    idx_map, rows, cols = _mosaic_index_map(secret, atlas, tile_w, tile_h, atlas_grid)
    index_flat = idx_map.reshape(-1)  # 長度 rows*cols

    # --- 容量檢查（每個符號 → N 個像素） ---
    H, W = carrier.shape[:2]
    total_needed = len(index_flat) * N
    total_pixels = H * W
    if total_needed > total_pixels:
        raise ValueError(f"❌ 容量不足：需要 {total_needed} 像素，載體只有 {total_pixels} 像素。"
                         f"請選更大的載體或加大 tile_size 以降低符號數。")

    # --- Y 通道嵌入（mod-M） ---
    ycc = cv2.cvtColor(carrier, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)

    rng = random.Random(pixel_seed)
    pos = list(range(total_pixels))
    rng.shuffle(pos)

    wptr = 0
    for tid in index_flat:
        coords = _id_to_coords_mrt(int(tid), M, N, perm_seed)  # 長度 N，每位 0..M-1
        for a in coords:
            p = pos[wptr]
            y = int(Y.flat[p])
            delta = (a - (y % M)) % M
            if delta > M // 2:
                delta -= M
            Y.flat[p] = np.clip(y + delta, 0, 255)
            wptr += 1

    ycc[:, :, 0] = Y.astype(np.uint8)
    stego = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(out_stego_path, stego)

    # --- 寫入金鑰（含 Base64 Atlas 與必要參數） ---
    key = {
        "M": M, "N": N,
        "perm_seed": perm_seed,
        "pixel_seed": pixel_seed,
        "mosaic_rows": int(rows),
        "mosaic_cols": int(cols),
        "tile_size": int(tile_size),
        "atlas_grid": int(atlas_grid),
        "atlas_meta": atlas_meta,
        "atlas_base64": _img_to_b64(atlas),
    }
    Path(key_path).write_text(json.dumps(key, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_stego_path, key_path

# ===== Extract：用 stego 圖 + 金鑰 → 重建馬賽克圖 =====
def extract_mrt_sie(stego_img_path: str, key_path: str, out_mosaic_path: str = "decoded_mosaic.png"):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    M, N = int(key["M"]), int(key["N"])
    perm_seed, pixel_seed = int(key["perm_seed"]), int(key["pixel_seed"])
    rows, cols = int(key["mosaic_rows"]), int(key["mosaic_cols"])
    atlas_grid = int(key.get("atlas_grid", 16))
    tile_size = int(key.get("tile_size", 16))

    _check_lossless(Path(stego_img_path))
    bgr = cv2.imread(stego_img_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError("❌ 讀不到嵌入圖")
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)

    H, W = Y.shape
    tot = H * W
    rng = random.Random(pixel_seed)
    pos = list(range(tot))
    rng.shuffle(pos)

    # 還原座標流
    coords = [int(Y.flat[p] % M) for p in pos]
    # 轉回 id（以 base-M 讀回，NOTE: 這裡未做逆置換，與嵌入一致使用 perm 對應）
    ids = []
    for i in range(0, len(coords), N):
        seg = coords[i:i + N]
        if len(seg) < N: break
        v = 0
        for a in seg:
            v = v * M + a
        ids.append(int(v))
    ids = np.array(ids[:rows * cols], dtype=np.int32)  # 截斷至 mosaic 尺寸

    # 還原 Atlas
    atlas = _b64_to_img(key["atlas_base64"])
    th = tw = tile_size
    tiles = _tiles_from_atlas(atlas, atlas_grid, atlas_grid, th, tw)

    # 重建 mosaic
    out = np.zeros((rows * th, cols * tw, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            tid = int(ids[r * cols + c]) % len(tiles)
            y0, y1 = r * th, (r + 1) * th
            x0, x1 = c * tw, (c + 1) * tw
            out[y0:y1, x0:x1] = tiles[tid]

    cv2.imwrite(out_mosaic_path, out)
    return out
