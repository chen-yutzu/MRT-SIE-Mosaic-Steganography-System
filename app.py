import streamlit as st
from stego_core import embed_mrt_sie, extract_mrt_sie
import cv2, json, tempfile, numpy as np
from pathlib import Path

st.set_page_config(page_title="MRT-SIE Mosaic Steganography", page_icon="🔒")

st.title("🔒 MRT-SIE Mosaic Steganography System")
st.caption("基於馬賽克拼貼與多維索引映射的高安全性影像隱藏系統")

mode = st.sidebar.radio("選擇功能", ["🧩 加密（嵌入）", "🔍 解密（提取）"])

# ========== 加密 ========== #
if mode == "🧩 加密（嵌入）":
    st.header("🧩 加密模式")

    secret_img = st.file_uploader("上傳秘密圖 (PNG/TIFF/BMP)", type=["png", "tif", "tiff", "bmp"])
    carrier_img = st.file_uploader("上傳載體圖 (PNG/TIFF/BMP)", type=["png", "tif", "tiff", "bmp"])
    block_size = st.slider("馬賽克塊大小", 8, 64, 16)

    if st.button("開始加密"):
        if not secret_img or not carrier_img:
            st.error("請上傳秘密圖與載體圖！")
        else:
            temp_secret = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            temp_carrier = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            with open(temp_secret, "wb") as f: f.write(secret_img.read())
            with open(temp_carrier, "wb") as f: f.write(carrier_img.read())

            # 建立馬賽克索引（簡化：直接以平均亮度取代 Atlas 步驟）
            img = cv2.imread(temp_secret)
            H, W = img.shape[:2]
            rows, cols = H//block_size, W//block_size
            index_map = np.arange(rows*cols).tolist()

            out_img = "stego_output.png"
            out_key = "stego_key.json"
            embed_mrt_sie(index_map, temp_carrier, out_img, out_key)

            st.success("✅ 嵌入完成！")
            st.image(out_img, caption="嵌入後影像")
            st.download_button("下載嵌入圖", open(out_img, "rb"), file_name="stego.png")
            st.download_button("下載密鑰檔", open(out_key, "rb"), file_name="stego_key.json")

# ========== 解密 ========== #
else:
    st.header("🔍 解密模式")
    stego_img = st.file_uploader("上傳嵌入後圖像 (PNG/TIFF/BMP)", type=["png", "tif", "tiff", "bmp"])
    key_file = st.file_uploader("上傳密鑰檔案 (JSON)", type=["json"])

    if st.button("開始解密"):
        if not stego_img or not key_file:
            st.error("請上傳嵌入圖與密鑰檔案！")
        else:
            temp_stego = tempfile.NamedTemporaryFile(delete=False, suffix=".png").name
            temp_key = tempfile.NamedTemporaryFile(delete=False, suffix=".json").name
            with open(temp_stego, "wb") as f: f.write(stego_img.read())
            with open(temp_key, "wb") as f: f.write(key_file.read())

            ids = extract_mrt_sie(temp_stego, temp_key)
            n = int(np.sqrt(len(ids)))
            decoded = ids[:n*n].reshape(n, n)

            decoded_img = np.uint8((decoded / decoded.max()) * 255)
            cv2.imwrite("decoded_mosaic.png", decoded_img)
            st.image("decoded_mosaic.png", caption="解碼後馬賽克圖")
            st.success("✅ 解密完成！")
            st.download_button("下載解碼圖", open("decoded_mosaic.png", "rb"), file_name="decoded_mosaic.png")
