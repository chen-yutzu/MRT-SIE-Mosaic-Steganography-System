# app.py
import io
import json
import importlib
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image
import streamlit as st

# 延遲載入 OpenCV（避免某些雲端環境的 bootstrap 衝突）
cv2 = importlib.import_module("cv2")

# 只允許無損格式
LOSSLESS_EXT = {".png", ".tif", ".tiff", ".bmp"}

# ============= 介面設定 =============
st.set_page_config(
    page_title="MRT-SIE Mosaic Steganography",
    page_icon="🔒",
    layout="centered"
)

st.markdown(
    """
<div style="text-align:center">
  <h2>🔒 MRT-SIE Mosaic Steganography System</h2>
  <p>基於馬賽克拼貼與多維索引映射的高安全性影像隱藏系統（固定 M=4, N=4；僅允許無損格式）</p>
  <hr/>
</div>
""",
    unsafe_allow_html=True,
)

# 產生顏色表（用於把 tile id 視覺化成馬賽克色塊）
def make_palette(n: int = 256) -> np.ndarray:
    x = np.linspace(0, 1, n)
    # HSV 轉 BGR
    hsv = np.stack([x, np.ones_like(x), np.ones_like(x)], axis=1).astype(np.float32)  # (n,3)
    hsv = hsv.reshape(-1, 1, 3)
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    bgr = (bgr.reshape(n, 3) * 255).astype(np.uint8)
    return bgr

PALETTE = make_palette(256)

def ensure_lossless_uploaded(name: str):
    ext = Path(name).suffix.lower()
    if ext not in LOSSLESS_EXT:
        st.error(f"❌ 僅支援無損影像：PNG / TIFF / BMP（收到 {ext}）")
        st.stop()

def imdecode_lossless(file) -> np.ndarray:
    """由上傳檔案物件讀取為 BGR np.ndarray，並強制無損檢查"""
    ensure_lossless_uploaded(file.name)
    data = file.read()
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        st.error("❌ 影像讀取失敗，請確認檔案內容")
        st.stop()
    return img

def imencode_png_bytes(img_bgr: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img_bgr)
    if not ok:
        raise RuntimeError("PNG 編碼失敗")
    return buf.tobytes()

# ============= 簡化版核心（固定 M=4, N=4；系統產生隨機金鑰） =============
def _id_to_coords_mrt(id_val: int, M: int, N: int, perm_seed: int):
    total = M ** N
    rng = np.random.default_rng(perm_seed)
    perm = np.arange(total, dtype=np.int32)
    rng.shuffle(perm)
    sym = int(perm[id_val % total])
    coords = [0] * N
    for i in range(N - 1, -1, -1):
        coords[i] = sym % M
        sym //= M
    return tuple(coords)

def embed_mrt_sie_indexmap_to_y(index_map, carrier_bgr: np.ndarray,
                                M=4, N=4, perm_seed=13579, pixel_seed=24680):
    """把 index_map（每個 tile 的 id）嵌入到載體圖 Y 通道（mod-M）"""
    ycc = cv2.cvtColor(carrier_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)
    H, W = Y.shape
    total_needed = len(index_map) * N
    total_pixels = H * W
    if total_needed > total_pixels:
        raise ValueError(f"容量不足：需要 {total_needed} 像素，只有 {total_pixels}")

    rng = np.random.default_rng(pixel_seed)
    positions = np.arange(total_pixels, dtype=np.int64)
    rng.shuffle(positions)

    write_ptr = 0
    for tid in index_map:
        coords = _id_to_coords_mrt(int(tid), M, N, perm_seed)
        for a in coords:
            pos = int(positions[write_ptr])
            y_val = int(Y.flat[pos])
            delta = (a - (y_val % M)) % M
            if delta > M // 2:
                delta -= M
            y_new = np.clip(y_val + delta, 0, 255)
            Y.flat[pos] = y_new
            write_ptr += 1

    ycc[:, :, 0] = Y.astype(np.uint8)
    stego_bgr = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2BGR)
    return stego_bgr

def extract_mrt_sie_from_y(stego_bgr: np.ndarray, key: dict):
    """依金鑰從 Y 通道取回座標流並重建 id 序列（不知實際長度則回傳全長可用序列）"""
    M = int(key["M"]); N = int(key["N"])
    perm_seed = int(key["perm_seed"]); pixel_seed = int(key["pixel_seed"])

    ycc = cv2.cvtColor(stego_bgr, cv2.COLOR_BGR2YCrCb)
    Y = ycc[:, :, 0].astype(np.int32)
    H, W = Y.shape

    total = H * W
    rng = np.random.default_rng(pixel_seed)
    positions = np.arange(total, dtype=np.int64)
    rng.shuffle(positions)

    coords_stream = [int(Y.flat[int(pos)] % M) for pos in positions]

    ids = []
    base = 1
    for _ in range(N - 1):
        base *= M

    # 反 perm 需要把座標轉回原始索引前的值；這裡使用暴力回推（同樣的 perm 方式）
    # 先生成 perm 與其 inverse
    totalM = M ** N
    rng2 = np.random.default_rng(perm_seed)
    perm = np.arange(totalM, dtype=np.int32)
    rng2.shuffle(perm)
    inv = np.empty_like(perm)
    inv[perm] = np.arange(totalM, dtype=np.int32)

    for i in range(0, len(coords_stream), N):
        chunk = coords_stream[i:i + N]
        if len(chunk) < N:
            break
        # 座標轉成排列後的符號值
        val = 0
        for a in chunk:
            val = val * M + int(a)
        # 還原至原始 id
        ids.append(int(inv[val]))
    return np.array(ids, dtype=np.int32)

# ============= UI：模式選單 =============
mode = st.sidebar.radio("選擇功能", ["🧩 加密（嵌入）", "🔍 解密（提取）"])

# 預先建立 session_state 欄位
for k in ["enc_ready", "enc_img_bytes", "enc_key_bytes", "preview_png",
          "dec_ready", "dec_png_bytes"]:
    st.session_state.setdefault(k, None)

# ============= 加密頁 =============
if mode == "🧩 加密（嵌入）":
    st.subheader("🧩 加密模式（系統固定 M=4, N=4；金鑰自動產生且不可視）")
    secret_file = st.file_uploader("上傳秘密圖（僅 PNG / TIFF / BMP）", type=["png", "tif", "tiff", "bmp"], key="sec_up")
    carrier_file = st.file_uploader("上傳載體圖（僅 PNG / TIFF / BMP）", type=["png", "tif", "tiff", "bmp"], key="car_up")
    block_size = st.slider("馬賽克塊大小（僅用於建立索引網格尺寸）", 8, 64, 16)
    run_btn = st.button("開始加密", type="primary", use_container_width=True)

    if run_btn:
        if not secret_file or not carrier_file:
            st.error("請同時上傳秘密圖與載體圖")
            st.stop()

        # 讀圖（強制無損）
        secret_bgr = imdecode_lossless(secret_file)
        carrier_bgr = imdecode_lossless(carrier_file)

        # 建立 index_map：只用網格尺寸，不做比色（簡化為展示用途）
        H, W = secret_bgr.shape[:2]
        rows, cols = max(1, H // block_size), max(1, W // block_size)
        index_map = np.arange(rows * cols, dtype=np.int32).tolist()

        # 系統產生金鑰（不可視、寫入 key 檔內容）
        perm_seed = int(np.random.SeedSequence().generate_state(1)[0] % (10**9))
        pixel_seed = int(np.random.SeedSequence().generate_state(1)[0] % (10**9))
        key = {
            "M": 4, "N": 4,
            "perm_seed": perm_seed,
            "pixel_seed": pixel_seed,
            "shuffle": True,
            # 解密需要知道馬賽克網格尺寸
            "mosaic_rows": int(rows),
            "mosaic_cols": int(cols),
            "tile_size": int(block_size)
        }

        # 進行嵌入
        try:
            stego_bgr = embed_mrt_sie_indexmap_to_y(
                index_map=index_map,
                carrier_bgr=carrier_bgr,
                M=4, N=4,
                perm_seed=perm_seed, pixel_seed=pixel_seed
            )
        except Exception as e:
            st.error(f"嵌入失敗：{e}")
            st.stop()

        # 存到 session_state（避免按下載時 rerun 造成遺失）
        st.session_state["enc_ready"] = True
        st.session_state["enc_img_bytes"] = imencode_png_bytes(stego_bgr)
        st.session_state["preview_png"] = st.session_state["enc_img_bytes"]
        st.session_state["enc_key_bytes"] = json.dumps(key, indent=2).encode("utf-8")

    # 只要有結果就顯示「同時兩個下載鍵」
    if st.session_state["enc_ready"]:
        st.success("✅ 嵌入完成！請下載下列檔案：")
        st.image(st.session_state["preview_png"], caption="嵌入後影像預覽（PNG）", use_container_width=True)

        c1, c2 = st.columns(2)
        with c1:
            st.download_button(
                "⬇️ 下載嵌入圖（PNG）",
                data=st.session_state["enc_img_bytes"],
                file_name="stego.png",
                mime="image/png",
                use_container_width=True
            )
        with c2:
            st.download_button(
                "⬇️ 下載密鑰檔（JSON）",
                data=st.session_state["enc_key_bytes"],
                file_name="stego_key.json",
                mime="application/json",
                use_container_width=True
            )

        # 加密頁的重置鈕
        if st.button("🔁 重新開始", type="secondary", use_container_width=True):
            for k in ["enc_ready", "enc_img_bytes", "enc_key_bytes", "preview_png"]:
                st.session_state[k] = None
            st.rerun()   # ← 改這行（原本是 st.experimental_rerun()）


# ============= 解密頁 =============
if mode == "🔍 解密（提取）":
    st.subheader("🔍 解密模式（上傳嵌入後影像 + 密鑰檔案）")
    stego_file = st.file_uploader("上傳嵌入後影像（僅 PNG / TIFF / BMP）", type=["png", "tif", "tiff", "bmp"], key="stego_up")
    key_file = st.file_uploader("上傳密鑰檔（JSON）", type=["json"], key="key_up")
    run_btn = st.button("開始解密", type="primary", use_container_width=True)

    if run_btn:
        if not stego_file or not key_file:
            st.error("請同時上傳嵌入圖與密鑰")
            st.stop()

        # 讀圖與金鑰
        stego_bgr = imdecode_lossless(stego_file)
        try:
            key = json.loads(key_file.read().decode("utf-8"))
        except Exception:
            st.error("❌ 密鑰檔解析失敗，請確認 JSON 格式")
            st.stop()

        # 提取 id 序列
        try:
            ids = extract_mrt_sie_from_y(stego_bgr, key)
        except Exception as e:
            st.error(f"提取失敗：{e}")
            st.stop()

        # 還原為馬賽克索引圖（用色塊視覺化）
        rows = int(key.get("mosaic_rows", 0))
        cols = int(key.get("mosaic_cols", 0))
        if rows <= 0 or cols <= 0:
            # 後備：嘗試用平方數還原
            n = int(np.sqrt(len(ids)))
            rows = cols = n

        ids = ids[: rows * cols]
        grid = ids.reshape(rows, cols)

        # 將每個 id 映射到顏色
        color_img = np.zeros((rows, cols, 3), dtype=np.uint8)
        color_img[:] = PALETTE[grid % 256]
        # 放大顯示
        tile = int(key.get("tile_size", 16))
        vis = cv2.resize(color_img, (cols * tile, rows * tile), interpolation=cv2.INTER_NEAREST)

        st.session_state["dec_ready"] = True
        st.session_state["dec_png_bytes"] = imencode_png_bytes(vis)

    if st.session_state["dec_ready"]:
        st.success("✅ 解密完成：下方為解回的『馬賽克索引圖』（色塊視覺化）")
        st.image(st.session_state["dec_png_bytes"], caption="Decoded Mosaic (Index Visualization)", use_container_width=True)

        st.download_button(
            "⬇️ 下載解碼馬賽克圖（PNG）",
            data=st.session_state["dec_png_bytes"],
            file_name="decoded_mosaic.png",
            mime="image/png",
            use_container_width=True
        )

        # 解密頁的重置鈕
        if st.button("🔁 重新開始", type="secondary", use_container_width=True):
            for k in ["dec_ready", "dec_png_bytes"]:
                st.session_state[k] = None
            st.rerun()   # ← 改這行（原本是 st.experimental_rerun()）

