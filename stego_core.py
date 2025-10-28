# stego_core.py
import cv2, json, random, base64
import numpy as np
from pathlib import Path

# 支援的無損影像格式
LOSSLESS_EXT = {".png", ".tif", ".tiff", ".bmp"}

def check_lossless(path: Path):
    if path.suffix.lower() not in LOSSLESS_EXT:
        raise ValueError(f"❌ 僅支援無損格式（PNG/TIFF/BMP），目前為：{path.suffix}")

# --- Base64 處理 ---
def encode_image_to_base64(img_path):
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片：{img_path}")
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8"), img.shape

def decode_image_from_base64(b64):
    data = base64.b64decode(b64)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# --- MRT 映射 ---
def _id_to_coords_mrt(id_val, M, N, perm_seed):
    total = M ** N
    rng = random.Random(perm_seed)
    perm = list(range(total))
    rng.shuffle(perm)
    sym = perm[id_val % total]
    coords = []
    for _ in range(N):
        coords.insert(0, sym % M)
        sym //= M
    return tuple(coords)

# --- 加密嵌入 ---
def embed_mrt_sie(index_map, carrier_path, atlas_path, out_path, key_path):
    M, N = 4, 4
    perm_seed, pixel_seed = random.randint(0, 1e9), random.randint(0, 1e9)

    # 檢查格式
    check_lossless(Path(carrier_path))
    check_lossless(Path(atlas_path))

    # 讀載體
    bgr = cv2.imread(carrier_path)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)
    H, W = Y.shape

    total_needed = len(index_map) * N
    if total_needed > H * W:
        raise ValueError("❌ 容量不足，請選更大的載體圖像")

    rng = random.Random(pixel_seed)
    pos = list(range(H * W))
    rng.shuffle(pos)

    write_ptr = 0
    for tid in index_map:
        coords = _id_to_coords_mrt(int(tid), M, N, perm_seed)
        for a in coords:
            p = pos[write_ptr]
            y_val = Y.flat[p]
            delta = (a - (y_val % M)) % M
            if delta > M // 2:
                delta -= M
            Y.flat[p] = np.clip(y_val + delta, 0, 255)
            write_ptr += 1

    ycc[:, :, 0] = Y.astype(np.uint8)
    stego = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(out_path, stego)

    # 轉 Base64 Atlas
    atlas_b64, (h, w, _) = encode_image_to_base64(atlas_path)
    key = {
        "M": M, "N": N,
        "perm_seed": perm_seed, "pixel_seed": pixel_seed,
        "mosaic_rows": int(np.sqrt(len(index_map))),
        "mosaic_cols": int(np.sqrt(len(index_map))),
        "atlas_shape": [h, w],
        "atlas_base64": atlas_b64
    }
    Path(key_path).write_text(json.dumps(key, indent=2), encoding="utf-8")
    return out_path, key_path


# --- 解密提取 ---
def extract_mrt_sie(stego_path, key_path, out_path="decoded_mosaic.png"):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    M, N = key["M"], key["N"]
    perm_seed, pixel_seed = key["perm_seed"], key["pixel_seed"]

    check_lossless(Path(stego_path))
    bgr = cv2.imread(stego_path)
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)
    H, W = Y.shape

    total = H * W
    rng = random.Random(pixel_seed)
    pos = list(range(total))
    rng.shuffle(pos)

    coords_stream = [int(Y.flat[p] % M) for p in pos]
    ids = []
    for i in range(0, len(coords_stream), N):
        seg = coords_stream[i:i+N]
        if len(seg) < N: break
        v = 0
        for a in seg:
            v = v * M + a
        ids.append(v)
    ids = np.array(ids, dtype=np.int32)

    # 從 Base64 重建 Atlas
    atlas = decode_image_from_base64(key["atlas_base64"])
    a_rows, a_cols = 16, 16
    tile_h, tile_w = atlas.shape[0] // a_rows, atlas.shape[1] // a_cols
    tiles = [atlas[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] for r in range(a_rows) for c in range(a_cols)]

    rows, cols = key["mosaic_rows"], key["mosaic_cols"]
    out = np.zeros((rows*tile_h, cols*tile_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            tid = int(ids[r*cols + c]) % len(tiles)
            out[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tiles[tid]
    cv2.imwrite(out_path, out)
    return out
