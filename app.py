# app.py
import streamlit as st
import cv2, tempfile, numpy as np
from stego_core import embed_mrt_sie, extract_mrt_sie

st.set_page_config(page_title="MRT-SIE Mosaic Steganography", layout="wide")

st.title("🧩 MRT-SIE Mosaic Steganography System")

tab1, tab2 = st.tabs(["🔐 加密 Embed", "🔓 解密 Extract"])

# --- 加密 ---
with tab1:
    st.header("加密：上傳秘密圖與載體圖")
    secret = st.file_uploader("秘密圖片 (PNG/TIFF/BMP)", type=["png","tiff","bmp"])
    carrier = st.file_uploader("載體圖片 (PNG/TIFF/BMP)", type=["png","tiff","bmp"])

    if st.button("執行加密"):
        if not (secret and carrier):
            st.warning("請上傳兩張圖片")
        else:
            with tempfile.TemporaryDirectory() as tmp:
                secret_path = f"{tmp}/secret.png"
                carrier_path = f"{tmp}/carrier.png"
                cv2.imwrite(secret_path, cv2.imdecode(np.frombuffer(secret.read(), np.uint8), 1))
                cv2.imwrite(carrier_path, cv2.imdecode(np.frombuffer(carrier.read(), np.uint8), 1))
                stego_path = f"{tmp}/stego.png"
                key_path = f"{tmp}/stego_key.json"
                embed_mrt_sie(secret_path, carrier_path, stego_path, key_path)
                st.success("✅ 嵌入完成！Atlas 已內嵌於金鑰中。")

                with open(stego_path, "rb") as f:
                    st.download_button("⬇️ 下載嵌入後圖片", f, "stego.png")

                with open(key_path, "rb") as f:
                    st.download_button("🗝️ 下載金鑰檔案 (含 Atlas)", f, "stego_key.json")

# --- 解密 ---
with tab2:
    st.header("解密：上傳嵌入圖與金鑰檔")
    stego = st.file_uploader("嵌入後圖片", type=["png","tiff","bmp"])
    key = st.file_uploader("金鑰 JSON 檔", type=["json"])
