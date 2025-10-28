# app.py
import streamlit as st
import numpy as np
import cv2, tempfile, json, os
from pathlib import Path
from stego_core import embed_mrt_sie, extract_mrt_sie

st.set_page_config(page_title="MRT-SIE Mosaic Steganography", layout="wide")
st.title("🧩 MRT-SIE Mosaic Steganography System")

# 系統內建素材資料夾（請把 256 張素材放到 assets/tiles_256/）
DEFAULT_TILE_FOLDER = "assets/tiles_256"   # 不顯示給使用者

tab1, tab2 = st.tabs(["🔐 加密 Embed", "🔓 解密 Extract"])

# ---------- 加密 ----------
with tab1:
    st.subheader("上傳：祕密圖片 + 載體圖片（無損）")
    secret = st.file_uploader("祕密圖片（PNG/TIFF/BMP）", type=["png","tif","tiff","bmp"])
    carrier = st.file_uploader("載體圖片（PNG/TIFF/BMP）", type=["png","tif","tiff","bmp"])

    tile_size = st.slider("馬賽克 tile 邊長", 8, 64, 16, 8)
    atlas_grid = 16  # 固定 16x16 → 256 tiles（與素材數量相符）

    if st.button("▶️ 執行加密", type="primary", use_container_width=True):
        if not (secret and carrier):
            st.warning("請同時上傳祕密圖與載體圖")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                # 寫入暫存檔
                secret_path  = f"{tmp}/secret.png"
                carrier_path = f"{tmp}/carrier.png"
                cv2.imwrite(secret_path,  cv2.imdecode(np.frombuffer(secret.read(),  np.uint8), 1))
                cv2.imwrite(carrier_path, cv2.imdecode(np.frombuffer(carrier.read(), np.uint8), 1))

                stego_path = f"{tmp}/stego.png"
                key_path   = f"{tmp}/stego_key.json"

                # 重要：Atlas 由系統內建資料夾生成，不給使用者選
                try:
                    embed_mrt_sie(secret_path, carrier_path, stego_path, key_path,
                                  tile_size=tile_size,
                                  atlas_grid=atlas_grid,
                                  tile_folder=DEFAULT_TILE_FOLDER,
                                  atlas_seed=13579)
                    st.success("✅ 嵌入完成！Atlas 已以 Base64 形式包含在金鑰檔。")
                except Exception as e:
                    st.error(f"❌ 加密失敗：{e}")
                else:
                    colA, colB = st.columns(2)
                    with colA:
                        with open(stego_path, "rb") as f:
                            st.download_button("⬇️ 下載嵌入後圖片", f, "stego.png", use_container_width=True)
                    with colB:
                        with open(key_path, "rb") as f:
                            st.download_button("🗝️ 下載金鑰檔（含 Atlas）", f, "stego_key.json", use_container_width=True)

# ---------- 解密 ----------
with tab2:
    st.subheader("上傳：嵌入後圖片 + 金鑰 JSON")
    stego = st.file_uploader("嵌入後圖片（PNG/TIFF/BMP）", type=["png","tif","tiff","bmp"])
    key   = st.file_uploader("金鑰檔（JSON）", type=["json"])

    if st.button("▶️ 執行解密", type="primary", use_container_width=True):
        if not (stego and key):
            st.warning("請同時上傳嵌入圖與金鑰檔")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                stego_path = f"{tmp}/stego.png"
                key_path   = f"{tmp}/key.json"
                cv2.imwrite(stego_path, cv2.imdecode(np.frombuffer(stego.read(), np.uint8), 1))
                Path(key_path).write_text(key.getvalue().decode("utf-8"), encoding="utf-8")

                try:
                    mosaic = extract_mrt_sie(stego_path, key_path, f"{tmp}/decoded.png")
                    st.image(mosaic, caption="解密後的馬賽克圖", use_container_width=True)
                    with open(f"{tmp}/decoded.png", "rb") as f:
                        st.download_button("⬇️ 下載解密後馬賽克圖", f, "decoded_mosaic.png", use_container_width=True)
                except Exception as e:
                    st.error(f"❌ 解密失敗：{e}")
