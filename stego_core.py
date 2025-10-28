# stego_core.py
import cv2, json, random, base64
import numpy as np
from pathlib import Path

# 僅支援無損影像格式
LOSSLESS_EXT = {".png", ".tif", ".tiff", ".bmp"}

def check_lossless(path: Path):
    if path.suffix.lower() not in LOSSLESS_EXT:
        raise ValueError(f"❌ 僅支援無損格式（PNG/TIFF/BMP），但收到：{path.suffix}")

# --- Base64 處理 ---
def encode_image_to_base64(img):
    _, buf = cv2.imencode(".png", img)
    return base64.b64encode(buf).decode("utf-8")

def decode_image_from_base64(b64_str):
    data = base64.b64decode(b64_str)
    arr = np.frombuffer(data, np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

# --- MRT 邏輯 ---
def _id_to_coords_mrt(id_val, M, N, perm_seed):
    total = M ** N
    rng = random.Random(perm_seed)
    perm = list(range(total))
    rng.shuffle(perm)
    sym = perm[id_val % total]
    coords = [0] * N
    for i in range(N - 1, -1, -1):
        coords[i] = sym % M
        sym //= M
    return tuple(coords)

# --- 系統內部：自動生成 Atlas ---
def generate_system_atlas(tile_size=(16,16), grid=16):
    atlas = np.zeros((tile_size[1]*grid, tile_size[0]*grid, 3), dtype=np.uint8)
    rng = np.random.default_rng(42)
    for r in range(grid):
        for c in range(grid):
            color = rng.integers(0, 255, size=(1,1,3), dtype=np.uint8)
            atlas[r*tile_size[1]:(r+1)*tile_size[1],
                  c*tile_size[0]:(c+1)*tile_size[0]] = color
    return atlas

# --- 嵌入 ---
def embed_mrt_sie(secret_img_path, carrier_img_path, out_stego_path, key_path):
    M, N = 4, 4
    perm_seed, pixel_seed = random.randint(0, 1e9), random.randint(0, 1e9)

    check_lossless(Path(carrier_img_path))
    carrier = cv2.imread(carrier_img_path)
    if carrier is None:
        raise FileNotFoundError("❌ 無法讀取載體圖")

    # 模擬馬賽克索引（假設每 tile 一個符號）
    h, w = 256, 256
    index_map = np.random.randint(0, 256, h*w)

    # 建立系統 Atlas（不讓使用者看見）
    atlas = generate_system_atlas()

    # --- 嵌入至 Y 通道 ---
    ycc = cv2.cvtColor(carrier, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)
    total_needed = len(index_map) * N
    total_pixels = Y.size
    if total_needed > total_pixels:
        raise ValueError("❌ 容量不足，請使用更大載體圖像")

    rng = random.Random(pixel_seed)
    positions = list(range(total_pixels))
    rng.shuffle(positions)

    write_ptr = 0
    for tid in index_map:
        coords = _id_to_coords_mrt(tid, M, N, perm_seed)
        for a in coords:
            pos = positions[write_ptr]
            y_val = Y.flat[pos]
            delta = (a - (y_val % M)) % M
            if delta > M // 2:
                delta -= M
            Y.flat[pos] = np.clip(y_val + delta, 0, 255)
            write_ptr += 1

    ycc[:, :, 0] = Y.astype(np.uint8)
    stego = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(out_stego_path, stego)

    # 將 Atlas 以 Base64 形式存入金鑰
    key = {
        "M": M, "N": N,
        "perm_seed": perm_seed,
        "pixel_seed": pixel_seed,
        "mosaic_rows": h,
        "mosaic_cols": w,
        "atlas_base64": encode_image_to_base64(atlas),
    }

    Path(key_path).write_text(json.dumps(key, indent=2), encoding="utf-8")
    return out_stego_path, key_path


# --- 解密 ---
def extract_mrt_sie(stego_img_path, key_path, out_mosaic_path="decoded_mosaic.png"):
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    M, N = key["M"], key["N"]
    perm_seed, pixel_seed = key["perm_seed"], key["pixel_seed"]

    check_lossless(Path(stego_img_path))
    bgr = cv2.imread(stego_img_path)
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
        coords = coords_stream[i:i+N]
        if len(coords) < N: break
        val = 0
        for a in coords:
            val = val * M + a
        ids.append(val)
    ids = np.array(ids, dtype=np.int32)

    # 從金鑰中重建 Atlas
    atlas = decode_image_from_base64(key["atlas_base64"])
    a_rows, a_cols = 16, 16
    tile_h, tile_w = atlas.shape[0] // a_rows, atlas.shape[1] // a_cols
    tiles = [atlas[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w]
             for r in range(a_rows) for c in range(a_cols)]

    rows, cols = key["mosaic_rows"], key["mosaic_cols"]
    out = np.zeros((rows*tile_h, cols*tile_w, 3), dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            tid = int(ids[r*cols + c]) % len(tiles)
            out[r*tile_h:(r+1)*tile_h, c*tile_w:(c+1)*tile_w] = tiles[tid]

    cv2.imwrite(out_mosaic_path, out)
    return out
