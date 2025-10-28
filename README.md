<p align="center">
  <br>
  <b style="font-size:24px;">馬賽克拼貼式影像隱寫系統</b>
  <br> 
  <b>MRT-SIE Mosaic Steganography System</b>
  <br>
  <i>Secure Image Hiding with Multi-Dimensional Reference Tensor and Scalable Index Encoding</i>
  <br>
  <sub>台中科技大學 資訊工程系 專題製作</sub>
  <br>
  作者：陳宥慈、郭姿廷、黃薏俽、王孝淳　｜　指導教授：洪維恩
  <br><br>
</p>

<p align="center">
  🎯 <b>線上展示（Streamlit Demo）</b><br><br>
  <a href="https://mrt-sie-mosaic-steganography-system-c5imaur3ojmfhgkehjuvpr.streamlit.app/" target="_blank">
    <img src="https://img.shields.io/badge/🚀%20Try%20Now%20on%20Streamlit-blue?style=for-the-badge" alt="Streamlit Demo"/>
  </a>
</p>


---

## 🧠 專題簡介
本系統結合 **馬賽克拼貼** 與 **影像隱寫術 (Steganography)**，  
建立一套能「將彩色圖片藏進另一張圖片中」的安全影像傳輸系統。  

系統先將秘密圖像轉換成由素材庫拼成的馬賽克圖，  
再透過 **MRT-SIE 演算法** 將其索引座標嵌入載體圖片。  
傳輸時僅呈現一張普通圖片，第三方無法察覺其中隱含的資訊。  

✨ 主要特點：
- 隱匿傳輸，外觀不具可疑特徵  
- 馬賽克拼貼具視覺意義，可驗證性高  
- 金鑰分離式保存，提升安全性  
- 自動容量調整，適應多種圖像大小  

---

## ⚙️ 系統架構
```text
Secret Image ─┐
               │
               ▼
        [Mosaic Encoding]
               │
        MRT-SIE Coordinate Encoding
               │
Carrier Image ─┤
               ▼
         Stego Image + Key
```
---

## ⚙️ 系統參數與使用限制

### 🔸 固定參數設定
| 參數 | 值 | 說明 |
|------|----|------|
| **M** | 4 | MRT 的邊長，用於控制嵌入容量 |
| **N** | 4 | MRT 的維度，用於控制安全性強度 |
| **Tile 數量** | 256 張 | 對應 Mⁿ 的索引總數 |
| **Tile 尺寸** |可自行調整 | 每個素材圖塊的大小 |
| **種子產生方式** | 系統自動生成 (64-bit) | 包含 `perm_seed`, `pixel_seed`, `atlas_seed` |
| **嵌入通道** | Y（亮度通道） | 人眼較無法識別 |
| **金鑰結構** | JSON 格式 | 儲存 MRT-SIE 參數與素材庫種子 |


### ⚠️ 使用注意事項
1. **請勿使用亮度過高（白底、淺色背景）的載體圖片。**  
   - 若亮度通道（Y）平均值過高，嵌入後訊號易被壓縮或平滑處理抹除。  

2. **素材圖庫須固定為 256 張圖片。**  
   - 系統會依 `M=4, N=4` 自動建立 4⁴ = 256 組索引。  
   - 若素材數量不同，將導致解碼錯位或顯示異常。  

3. **秘密圖像若超出載體容量，系統會自動縮放。**  
   - 加密前會依可嵌入容量調整尺寸，不會造成資料遺失。  

4. **金鑰與嵌入圖必須配對使用。**  
   - 若使用不同次生成的金鑰，解碼結果會錯亂。  
   - 系統會自動於金鑰中加入雜湊驗證碼以檢查匹配性。  



---
## ❓ FAQ

**Q1. 為什麼有些白底圖片解不出來？**  
A. 目前嵌在 Y 通道，過高亮度會降低可辨識訊號。請改用紋理較多或亮度較低的載體。

**Q2. 我只有 stego.png，沒有金鑰可以解嗎？**  
A. 不行。本系統採分離式金鑰，必須同時提供 `stego.png + stego_key.json`。

**Q3. 素材庫一定要 256 張嗎？**  
A. 因為該系統的固定設置 M=4、N=4 → 4^4=256。如果要更改素材庫量，M、N 也要搭配做更改。

---

## 📜 授權與使用

本專案僅供 **學術研究與教育用途**，禁止任何形式的商業使用。  
著作權歸屬作者團隊所有，如需引用或轉載，請標註出處：

> 陳宥慈、郭姿廷、黃薏俽、王孝淳（2025）。  
> *MRT-SIE Mosaic Steganography System: Secure Image Hiding with Multi-Dimensional Reference Tensor and Scalable Index Encoding.*  
> 國立台中科技大學 資訊工程系 專題製作。  
