# app.py
import streamlit as st
import numpy as np
import cv2
from stego_core import encrypt, save_stego_with_embedded_key, decrypt

st.set_page_config(page_title="Mosaic + MRT-SIE", layout="wide")

st.title("🧩 Mosaic Stego（MRT-SIE）")
tab_enc, tab_dec = st.tabs(["🔐 加密 / Embed", "🔓 解密 / Decode"])

# ---------- 加密 ----------
with tab_enc:
    st.subheader("上傳檔案")
    secret = st.file_uploader("祕密圖（會被轉成馬賽克索引）", type=["png","jpg","jpeg","bmp","tiff"])
    carrier = st.file_uploader("載體圖（嵌入後看起來幾乎相同）", type=["png","jpg","jpeg","bmp","tiff"])

    st.divider()
    st.subheader("參數")
    c1, c2, c3, c4 = st.columns(4)
    tile_w = c1.number_input("Tile 寬", 4, 256, 16, step=4)
    tile_h = c2.number_input("Tile 高", 4, 256, 16, step=4)
    M = c3.number_input("M（mod基數）", 2, 16, 4)
    N = c4.number_input("N（維度）", 1, 8, 4)
    perm_seed = st.number_input("perm_seed（SIE 置換）", value=13579)
    pixel_seed = st.number_input("pixel_seed（像素洗牌）", value=24680)
    shuffle_pixels = st.checkbox("打亂像素順序", value=True)
    atlas_seed = st.number_input("atlas_seed（素材洗牌）", value=15)
    keep_order = st.checkbox("Atlas 按檔名順序", value=False)

    tiles_dir = st.text_input("素材方塊資料夾", value="assets/tiles_256")

    if st.button("開始加密▶"):
        try:
            secret_bgr = cv2.imdecode(np.frombuffer(secret.read(), np.uint8), cv2.IMREAD_COLOR)
            carrier_bgr = cv2.imdecode(np.frombuffer(carrier.read(), np.uint8), cv2.IMREAD_COLOR)

            result = encrypt(
                secret_bgr, carrier_bgr, tiles_dir,
                tile_size=(tile_w, tile_h), M=M, N=N, perm_seed=perm_seed,
                shuffle_pixels=shuffle_pixels, pixel_seed=pixel_seed,
                atlas_seed=atlas_seed, keep_order=keep_order
            )

            # 內嵌金鑰的 stego PNG
            stego_png_bytes = save_stego_with_embedded_key(result["stego_bgr"], result["key"])
            key_json_bytes = bytes(
                json_dumps := __import__("json").dumps(result["key"], ensure_ascii=False, indent=2),
                "utf-8"
            )
            # 預覽
            st.success("加密完成！下方提供下載。")
            st.image(cv2.cvtColor(result["mosaic_bgr"], cv2.COLOR_BGR2RGB), caption="加密用馬賽克圖（預覽）", use_column_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.download_button("⬇ 下載嵌入圖（PNG，含金鑰）", stego_png_bytes, file_name="stego.png", mime="image/png")
            with c2:
                st.download_button("⬇ 下載金鑰（JSON）", key_json_bytes, file_name="stego_key.json", mime="application/json")
        except Exception as e:
            st.error(f"加密失敗：{e}")

# ---------- 解密 ----------
with tab_dec:
    st.subheader("上傳嵌入圖（PNG）")
    stego = st.file_uploader("若 PNG 內無金鑰，可另外上傳金鑰 JSON", type=["png"], key="stego")
    key_file = st.file_uploader("（選擇性）金鑰 JSON", type=["json"], key="key")

    if st.button("開始解密▶"):
        try:
            stego_bytes = stego.read()
            key_bytes = key_file.read() if key_file else None
            mosaic_png = decrypt(stego_bytes, key_json=key_bytes)
            st.success("解密完成！")
            st.image(cv2.imdecode(np.frombuffer(mosaic_png, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1],
                     caption="還原的馬賽克加密圖", use_column_width=True)
            st.download_button("⬇ 下載還原馬賽克（PNG）", mosaic_png, file_name="decoded_mosaic.png", mime="image/png")
        except Exception as e:
            st.error(f"解密失敗：{e}")
