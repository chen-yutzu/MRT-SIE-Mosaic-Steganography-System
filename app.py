# app.py
import streamlit as st
import numpy as np
import cv2
import json
from pathlib import Path
from stego_core import encrypt, save_stego_with_embedded_key, decrypt

st.set_page_config(page_title="Mosaic Stego (M=4,N=4)", layout="wide")
st.title("🧩 Mosaic + MRT-SIE（M=4, N=4 固定，種子自動產生）")

# ---- 系統固定的素材路徑（專案相對） ----
TILES_DIR = (Path(__file__).parent / "assets" / "tiles_256").as_posix()

# 顯示唯讀資訊，方便檢查部署是否抓到正確路徑
with st.sidebar:
    st.markdown("### 系統設定（唯讀）")
    st.code(f"TILES_DIR = {TILES_DIR}")

tab_enc, tab_dec = st.tabs(["🔐 加密 / Embed", "🔓 解密 / Decode"])

# ---------- 加密 ----------
with tab_enc:
    st.subheader("上傳檔案")
    secret = st.file_uploader("祕密圖（將被轉成馬賽克索引）", type=["png","jpg","jpeg","bmp","tiff"], key="secret")
    carrier = st.file_uploader("載體圖（嵌入後看起來幾乎相同）", type=["png","jpg","jpeg","bmp","tiff"], key="carrier")

    st.divider()
    st.subheader("參數（M=4、N=4；種子由系統自動產生並寫入金鑰）")
    c1, c2 = st.columns(2)
    tile_w = c1.number_input("Tile 寬", 4, 256, 16, step=4)
    tile_h = c2.number_input("Tile 高", 4, 256, 16, step=4)

    # 檢查素材資料夾存在
    if not Path(TILES_DIR).exists():
        st.error(f"找不到素材資料夾：{TILES_DIR}\n請確認專案內有 assets/tiles_256。")
    else:
        # 小提示：顯示目前素材張數
        from os import listdir
        try:
            n_tiles = len([p for p in listdir(TILES_DIR)
                           if Path(p).suffix.lower() in {'.png','.jpg','.jpeg','.bmp','.tiff','.tif'}])
            st.caption(f"素材庫：{TILES_DIR}（偵測到 {n_tiles} 張方塊）")
        except Exception:
            st.caption(f"素材庫：{TILES_DIR}")

    if st.button("開始加密 ▶", type="primary", use_container_width=True):
        if (secret is None) or (carrier is None):
            st.error("請同時上傳「祕密圖」與「載體圖」。")
        elif not Path(TILES_DIR).exists():
            st.error(f"素材庫不存在：{TILES_DIR}")
        else:
            try:
                secret_bgr  = cv2.imdecode(np.frombuffer(secret.read(),  np.uint8), cv2.IMREAD_COLOR)
                carrier_bgr = cv2.imdecode(np.frombuffer(carrier.read(), np.uint8), cv2.IMREAD_COLOR)

                result = encrypt(
                    secret_bgr, carrier_bgr, TILES_DIR,
                    tile_size=(int(tile_w), int(tile_h)),
                    # M=4, N=4（預設），三個 seed 不傳 → 自動產生
                    perm_seed=None, pixel_seed=None, atlas_seed=None,
                    shuffle_pixels=True, keep_order=False
                )

                stego_png_bytes = save_stego_with_embedded_key(result["stego_bgr"], result["key"])
                key_json_bytes  = json.dumps(result["key"], ensure_ascii=False, indent=2).encode("utf-8")

                st.success("加密完成！（金鑰已內嵌於 PNG；亦提供 JSON 備份下載）")
                st.image(cv2.cvtColor(result["mosaic_bgr"], cv2.COLOR_BGR2RGB),
                         caption="加密用馬賽克圖（預覽）", use_column_width=True)

                c1, c2 = st.columns(2)
                with c1:
                    st.download_button("⬇ 下載嵌入圖（PNG，含金鑰）",
                                       stego_png_bytes, file_name="stego.png", mime="image/png", use_container_width=True)
                with c2:
                    st.download_button("⬇ 下載金鑰（JSON 備份）",
                                       key_json_bytes, file_name="stego_key.json", mime="application/json", use_container_width=True)

                with st.expander("查看此次固定參數 / 種子（唯讀）"):
                    st.json({
                        "M": 4, "N": 4,
                        "perm_seed": result["key"]["perm_seed"],
                        "pixel_seed": result["key"]["pixel_seed"],
                        "atlas_seed": result["key"]["atlas"]["seed"],
                        "shuffle_pixels": True,
                        "tile_size": result["key"]["tile_size"],
                        "mosaic_rows": result["key"]["mosaic_rows"],
                        "mosaic_cols": result["key"]["mosaic_cols"],
                        "symbols": result["key"]["symbols"],
                        "tiles_dir": TILES_DIR,
                    })

            except Exception as e:
                st.error(f"加密失敗：{e}")

# ---------- 解密 ----------
with tab_dec:
    st.subheader("上傳嵌入圖（PNG）")
    stego = st.file_uploader("若 PNG metadata 無金鑰，可另外上傳金鑰 JSON", type=["png"], key="stego_png")
    key_file = st.file_uploader("（選擇性）金鑰 JSON", type=["json"], key="key_json")

    if st.button("開始解密 ▶", use_container_width=True):
        if stego is None:
            st.error("請上傳嵌入後 PNG。")
        else:
            try:
                stego_bytes = stego.read()
                key_bytes = key_file.read() if key_file else None
                mosaic_png = decrypt(stego_bytes, key_json=key_bytes)

                st.success("解密完成！")
                st.image(cv2.imdecode(np.frombuffer(mosaic_png, np.uint8), cv2.IMREAD_COLOR)[:, :, ::-1],
                         caption="還原的馬賽克加密圖（PNG 無損）", use_column_width=True)
                st.download_button("⬇ 下載還原馬賽克（PNG）",
                                   mosaic_png, file_name="decoded_mosaic.png", mime="image/png", use_container_width=True)
            except Exception as e:
                st.error(f"解密失敗：{e}")
