import cv2, json, random
import numpy as np
from pathlib import Path
from scipy.spatial import KDTree

# ============= 檔案格式檢查 =============
LOSSLESS_EXT = {".png", ".tiff", ".tif", ".bmp"}

def check_lossless(path: Path):
    if path.suffix.lower() not in LOSSLESS_EXT:
        raise ValueError(f"❌ 僅支援無損影像格式（PNG/TIFF/BMP），但收到：{path.suffix}")

# ============= MRT-SIE 函式 =============

def _id_to_coords_mrt(id_val: int, M: int, N: int, perm_seed: int):
    total = M ** N
    rng = random.Random(perm_seed)
    perm = list(range(total))
    rng.shuffle(perm)
    sym = perm[id_val % total]
    coords = [0] * N
    for i in range(N-1, -1, -1):
        coords[i] = sym % M
        sym //= M
    return tuple(coords)

def embed_mrt_sie(index_map, carrier_path, out_path, key_path):
    M, N = 4, 4
    perm_seed, pixel_seed = 13579, 24680

    bgr = cv2.imread(carrier_path)
    check_lossless(Path(carrier_path))
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)
    H, W = Y.shape
    total_needed = len(index_map) * N
    if total_needed > H * W:
        raise ValueError("❌ 容量不足，請選擇更大的載體圖像")

    rng = random.Random(pixel_seed)
    positions = list(range(H * W))
    rng.shuffle(positions)

    write_ptr = 0
    coords_stream = []
    for tid in index_map:
        coords = _id_to_coords_mrt(tid, M, N, perm_seed)
        for a in coords:
            pos = positions[write_ptr]
            y_val = Y.flat[pos]
            delta = (a - (y_val % M)) % M
            if delta > M // 2:
                delta -= M
            y_new = np.clip(y_val + delta, 0, 255)
            Y.flat[pos] = y_new
            coords_stream.append(a)
            write_ptr += 1

    ycc[:, :, 0] = Y.astype(np.uint8)
    stego = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    cv2.imwrite(out_path, stego)

    key = {
        "M": M, "N": N,
        "perm_seed": perm_seed,
        "pixel_seed": pixel_seed,
        "shuffle": True,
        "size": [H, W]
    }
    Path(key_path).write_text(json.dumps(key, indent=2), encoding="utf-8")

    return out_path, key_path


def extract_mrt_sie(stego_path, key_path):
    key = json.loads(Path(key_path).read_text())
    M, N = key["M"], key["N"]
    perm_seed, pixel_seed = key["perm_seed"], key["pixel_seed"]

    bgr = cv2.imread(stego_path)
    check_lossless(Path(stego_path))
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)
    H, W = Y.shape

    total = H * W
    rng = random.Random(pixel_seed)
    positions = list(range(total))
    rng.shuffle(positions)

    coords_stream = [int(Y.flat[pos] % M) for pos in positions]
    ids = []
    for i in range(0, len(coords_stream), N):
        coords = coords_stream[i:i+N]
        if len(coords) < N:
            break
        val = 0
        for a in coords:
            val = val * M + a
        ids.append(val)

    return np.array(ids, dtype=np.int32)
