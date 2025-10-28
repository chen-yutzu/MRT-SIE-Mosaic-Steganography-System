# app.py
import streamlit as st
import numpy as np
import cv2, tempfile, json, base64
from stego_core import embed_mrt_sie, extract_mrt_sie

st.set_page_config(page_title="MRT-SIE Mosaic Steganography", layout="wide")

st.title("🧩 MRT-SIE Mosaic Steganography System")

tabs = st.tabs(["🔐 加密 Embed", "🔓 解密 Extract"])

# --- 加密 ---
with tabs[0]:
    st.subheader("上傳圖片")
    secret_img = st.file_uploader("秘密圖片 (mosaic 索引來源，可用任意灰階)", type=["png", "tiff", "bmp"])
    carrier_img = st.file_uploader("載體圖片 (嵌入位置)", type=["png", "tiff", "bmp"])
    atlas_img = st.file_uploader("Atlas (素材圖集)", type=["png", "tiff", "bmp"])

    tile_size = st.slider("馬賽克 Tile 邊長", 8, 64, 16, 8)

    if st.button("執行加密"):
        if not all([secret_img, carrier_img, atlas_img]):
            st.warning("請完整上傳三張圖片")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                secret_path = f"{tmp}/secret.png"
                carrier_path = f"{tmp}/carrier.png"
                atlas_path = f"{tmp}/atlas.png"
                cv2.imwrite(secret_path, cv2.imdecode(np.frombuffer(secret_img.read(), np.uint8), 1))
                cv2.imwrite(carrier_path, cv2.imdecode(np.frombuffer(carrier_img.read(), np.uint8), 1))
                cv2.imwrite(atlas_path, cv2.imdecode(np.frombuffer(atlas_img.read(), np.uint8), 1))

                idx_map = np.random.randint(0, 256, tile_size*tile_size)
                stego_path = f"{tmp}/stego.png"
                key_path = f"{tmp}/stego_key.json"
                embed_mrt_sie(idx_map, carrier_path, atlas_path, stego_path, key_path)

                with open(stego_path, "rb") as f: st.download_button("⬇️ 下載嵌入後圖片", f, "stego.png")
                with open(key_path, "rb") as f: st.download_button("🗝️ 下載金鑰檔案", f, "stego_key.json")

# --- 解密 ---
with tabs[1]:
    st.subheader("上傳密鑰與嵌入圖")
    stego_img = st.file_uploader("嵌入後的圖片", type=["png", "tiff", "bmp"])
    key_file = st.file_uploader("金鑰 JSON 檔", type=["json"])

    if st.button("執行解密"):
        if not (stego_img and key_file):
            st.warning("請上傳兩個檔案")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                stego_path = f"{tmp}/stego.png"
                key_path = f"{tmp}/key.json"
                cv2.imwrite(stego_path, cv2.imdecode(np.frombuffer(stego_img.read(), np.uint8), 1))
                key_path_obj = open(key_path, "w", encoding="utf-8")
                key_path_obj.write(key_file.getvalue().decode("utf-8"))
                key_path_obj.close()

                mosaic = extract_mrt_sie(stego_path, key_path, f"{tmp}/decoded.png")
                st.image(mosaic, caption="解密後的馬賽克圖", use_column_width=True)
                with open(f"{tmp}/decoded.png", "rb") as f:
                    st.download_button("⬇️ 下載還原圖", f, "decoded_mosaic.png")
