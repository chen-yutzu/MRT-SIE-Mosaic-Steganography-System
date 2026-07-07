# MRT-SIE Mosaic Steganography System

This project implements a mosaic-index image steganography workflow.

The main idea is:

```text
Secret image
-> Mosaic tile-index representation
-> MRT-SIE embedding
-> Stego image
-> MRT-SIE extraction
-> Decoded mosaic image
```

In this research, the secret image is not embedded directly. It is first converted into a tile index sequence. The index sequence can only be interpreted with the correct tile library and parameters.

## Project Structure

```text
.
├── mosaic_collage_converter.py   # Secret image -> mosaic image + index JSON
├── mrt_sie_embedder.py           # Index JSON + carrier image -> stego image + key JSON
├── mrt_sie_decoder.py            # Stego image + key JSON + tile library -> decoded mosaic
├── requirements.txt              # Python packages
├── .gitignore
└── README.md
```

## Installation

Use Python 3.10 or newer.

```bash
pip install -r requirements.txt
```

## Required Input Data

Prepare:

```text
secret image      example: data/secret/Splash.jpg
carrier image     example: data/carrier/Lena.jpg
tile library      example: data/tiles_256/
```

The tile library should contain image files such as:

```text
000001.jpg
000002.jpg
...
000256.jpg
```

## Step 1: Convert Secret Image to Mosaic Index Code

```bash
python mosaic_collage_converter.py ^
  --secret data/secret/Splash.jpg ^
  --tiles data/tiles_256 ^
  --out outputs/mosaic ^
  --block-size 16 ^
  --tile-seed 15
```

Outputs:

```text
Splash_cropped.png
Splash_mosaic_b16.png
Splash_mosaic_index_b16.json
tile_atlas_b16.png
```

The most important file is:

```text
Splash_mosaic_index_b16.json
```

It contains the tile index sequence:

```text
index_map_flat = [t1, t2, ..., tn]
```

## Step 2: Embed Index Code with MRT-SIE

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

Outputs:

```text
Lena_stego_M4_N4.png
Lena_embedding_key_M4_N4.json
```

When `M=4` and `N=4`, MRT-SIE can represent:

```text
M^N = 4^4 = 256 symbols
```

This matches a 256-image tile library.

## Step 3: Decode Stego Image

```bash
python mrt_sie_decoder.py ^
  --stego outputs/stego/Lena_stego_M4_N4.png ^
  --key outputs/stego/Lena_embedding_key_M4_N4.json ^
  --tiles data/tiles_256 ^
  --out outputs/decoded/decoded_mosaic.png
```

The decoder extracts the tile index sequence from the stego image and reconstructs the mosaic image with the correct tile library.

## Important Concepts

### Mosaic Index Representation

Each secret image block is matched to the closest tile image by Lab mean color:

```text
F(B_i) = [mean_L(B_i), mean_a(B_i), mean_b(B_i)]
```

Tile selection:

```text
t_i = argmin_j || F(B_i) - F(T_j) ||_2
```

Where:

```text
B_i = secret image block
T_j = tile image
t_i = selected tile index
```

### MRT-SIE Embedding

Each tile index is converted into MRT-SIE coordinates.

Example:

```text
tile index 173 -> [2, 3, 0, 1]
```

The coordinate values are embedded into the carrier image Y channel by modulo adjustment.

### Recovery Conditions

To decode correctly, the receiver needs:

```text
1. stego image
2. MRT-SIE key JSON
3. correct tile library
4. correct tile order / tile seed
```

Without these conditions, the extracted code cannot be correctly interpreted as a mosaic image.

## Research Position

This project focuses on using mosaic collage as an index representation layer for image steganography.

MRT-SIE is used as the embedding layer. The main research contribution is not improving MRT-SIE itself, but converting a secret image into a tile-index code before embedding.

