# MRT-SIE Mosaic Steganography System

This project implements an image steganography workflow based on mosaic tile-index representation and MRT-SIE embedding.

The core research idea is:

```text
Secret Image
-> Mosaic Tile-Index Representation
-> MRT-SIE Embedding
-> Stego Image
-> MRT-SIE Extraction
-> Decoded Mosaic Image
```

In this system, the secret image is not embedded directly into the carrier image. The secret image is first converted into a mosaic representation. Each block of the secret image is represented by the index of a matched tile from a predefined tile library. The resulting tile-index sequence is then embedded into a carrier image by using MRT-SIE.

Therefore, the main contribution of this research is the mosaic-based secret representation layer. MRT-SIE is used as the embedding method for implementing and testing the proposed representation.

## Project Structure

```text
.
|-- app.py                         # Streamlit web interface
|-- stego_core.py                  # Shared mosaic and steganography functions
|-- mosaic_collage_converter.py    # Secret image -> mosaic image + index JSON
|-- mrt_sie_embedder.py            # Index JSON + carrier image -> stego image + key JSON
|-- mrt_sie_decoder.py             # Stego image + key JSON + tile library -> decoded mosaic
|-- assets/
|   `-- tiles_256/                 # Tile library used for mosaic representation
|-- requirements.txt               # Python packages
`-- README.md
```

## Installation

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Run the Web Interface

```bash
streamlit run app.py
```

The web interface provides a simple way to test the full workflow:

1. Upload a secret image.
2. Select or use the prepared tile library.
3. Generate the mosaic tile-index representation.
4. Embed the index data into a carrier image.
5. Decode the stego image and reconstruct the mosaic representation.

## Command-Line Usage

### Step 1: Convert Secret Image to Mosaic Representation

```bash
python mosaic_collage_converter.py ^
  --secret data/secret/Splash.jpg ^
  --tiles assets/tiles_256 ^
  --out outputs/mosaic ^
  --block-size 16 ^
  --tile-seed 15
```

Main outputs:

```text
Splash_cropped.png
Splash_mosaic_b16.png
Splash_mosaic_index_b16.json
tile_atlas_b16.png
```

The JSON file stores the tile-index sequence. It records which tile is used for each block of the secret image.

### Step 2: Embed the Tile-Index Sequence

```bash
python mrt_sie_embedder.py ^
  --index outputs/mosaic/Splash_mosaic_index_b16.json ^
  --carrier data/carrier/Lena.jpg ^
  --out outputs/stego ^
  --m 4 ^
  --n 4 ^
  --perm-seed 13579 ^
  --pixel-seed 24680
```

Main outputs:

```text
Lena_stego_M4_N4.png
Lena_embedding_key_M4_N4.json
```

The stego image contains the embedded index data. The key JSON stores the parameters required for extraction.

### Step 3: Decode and Reconstruct the Mosaic Image

```bash
python mrt_sie_decoder.py ^
  --stego outputs/stego/Lena_stego_M4_N4.png ^
  --key outputs/stego/Lena_embedding_key_M4_N4.json ^
  --tiles assets/tiles_256 ^
  --out outputs/decoded/decoded_mosaic.png
```

To reconstruct the hidden mosaic image, the receiver needs:

```text
stego image
embedding key JSON
same tile library and tile order
```

## Research Position

Traditional image hiding methods often focus on the embedding algorithm itself. This project emphasizes the representation of the secret image before embedding.

The secret image is transformed into a mosaic tile-index sequence. Even if the embedded data is extracted, the data is not the original image pixels. It must be interpreted with the correct tile library and reconstruction parameters.

This design separates the system into two layers:

```text
Representation Layer: Secret image -> mosaic tile-index sequence
Embedding Layer: Tile-index sequence -> stego image by MRT-SIE
```

Because these two layers are separated, the mosaic representation can also be tested with other embedding methods in future work.

## Notes

- The tile library should remain fixed during encoding and decoding.
- Changing tile order, block size, or seed values will affect reconstruction.
- MRT-SIE is used here as the embedding method, while the research focus is the mosaic tile-index representation.
