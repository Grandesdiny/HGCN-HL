# HGCN-HL
Dual-branch GSDG-style dynamic hypergraph classification for hyperspectral
and LiDAR data.

## Step-by-step demo

`train_original.py` remains the Stage-0 fixed-incidence HGCN-HL baseline.
`demo_train.py` is currently Stage 7:

```bash
python demo_train.py \
  --dataset muufl \
  --train-samples-per-class 20 \
  --graph-layout separate \
  --lidar-segmentation slic \
  --lidar-graph-prior rag-height-knn \
  --lidar-rag-hops 2 \
  --lidar-height-knn-k 5 \
  --lidar-modulation rag-lowhigh \
  --cross-modal-interaction overlap-gate \
  --fdsm-scope hsi \
  --device cuda
```

The demo keeps the original
joint `PCA(HSI)+LiDAR` input, two WMF blocks, original `5x5/5x5` CNN,
lambda fusion, and classifier. In the default `--graph-layout separate`,
HSI-SLIC and LiDAR-SLIC use independent WMF graph encoders, spatial priors,
Q/K graph builders, and GATs. Their graph features are independently
projected to pixels and fused with `--graph-modality-lambda`; the unchanged
joint CNN is then fused with that graph result. Use `--graph-layout joint`
to retain the previous concatenated-node single graph. LiDAR regions default
to SLIC; `--lidar-segmentation felzenszwalb` remains available.

`--fdsm-scope hsi` applies the original GSDG frequency-domain modulation
after HSI superpixel pooling. For LiDAR, `--lidar-modulation rag-lowhigh`
uses a geometry-gated decomposition into RAG-smoothed low-frequency and
RAG-residual high-frequency node features. The optional
`--lidar-graph-prior rag-height-knn` restricts LiDAR dynamic Top-k edges with
a local RAG and elevation similarity. `--cross-modal-interaction
overlap-gate` exchanges information between GAT stages using the spatial
overlap of the independent HSI and LiDAR superpixels; `overlap-attention`
provides the attention-based alternative. Each option defaults to `none` or
the earlier centroid prior for ablation compatibility. The demo does not use
a hypergraph or the GSDG CNN. Outputs are isolated under
`model_demo/stage7_optional_lidar_rag_lowhigh`.

## Supported datasets

The command-line entry point supports these local datasets:

- `muufl`: `/root/hsi/MUUFL`
- `houston`: `/root/hsi/dataset/Houston2013`
- `trento`: `/root/hsi/dataset/Trento`

Install the dependencies and run:

```bash
python -m pip install -r requirements.txt

# MUUFL
python train.py \
  --dataset muufl \
  --train-samples-per-class 20 \
  --lidar-segmentation slic \
  --backbone hypergraph \
  --graph-mode dynamic \
  --dynamic-topk 8 \
  --dynamic-dk 16 \
  --spatial-prior-k 15 \
  --fdsm-scope hsi \
  --cnn-style gsdg \
  --device cuda

# Houston 2013
python train.py --dataset houston --lidar-segmentation slic --device cuda

# Trento
python train.py --dataset trento --lidar-segmentation slic --device cuda
```

For each run, exactly 20 labeled pixels from every class are randomly selected
for training. All other labeled pixels are used for testing.

Dataset defaults are PCA/scale `10/100` for MUUFL, `25/300` for Houston, and
`10/100` for Trento. Use `--data-dir`, `--pca-components`, or `--scales` to
override them.

`--lidar-segmentation` accepts `slic` or `felzenszwalb`; the latter remains the
default used by the original pipeline.

The network uses two independent modality branches:

```text
HSI pixels   -> HSI superpixels   -> dynamic hyperedges -> two HGCNs --\
                                                                        -> project -> concat -> classifier
LiDAR pixels -> LiDAR superpixels -> dynamic hyperedges -> two HGCNs --/
```

Each branch first pools pixels into superpixel vertices with its assignment
matrix. Following GSDG, FDSM is applied immediately after HSI
pixel-to-superpixel pooling. In the default `--backbone hypergraph` mode, every
superpixel is the center of one neighborhood hyperedge. Each HGCN layer
combines learned Q/K similarity, sinusoidal position encoding, and the
centroid-based spatial prior, keeps Top-k members, and then performs weighted
node-to-hyperedge-to-node propagation. The second dynamic HGCN has a residual
connection. Because each layer receives updated features and has its own Q/K
projection, its hyperedges are rebuilt independently on every forward pass.

Use `--fdsm-scope both` to also modulate LiDAR latent features or
`--fdsm-scope none` for ablation. Each projected graph feature passes through
`Linear -> BatchNorm1d -> LeakyReLU` before it is fused with the pixel-CNN
feature. Both modalities use independent copies of the GSDG CNN path:
`stem_dim -> 64` depthwise-separable convolution with a `3x3` kernel, followed
by `64 -> 64` with a `7x7` kernel (`64` is controlled by `--graph-dim`).
HSI and LiDAR do not share assignments, CNNs, Q/K projections, HGCNs,
projection blocks, or spatial priors. `--fusion-lambda` controls hypergraph
and pixel-CNN fusion inside each modality branch.

Set `--cnn-style original` to restore the earlier HGCN-HL CNN path with two
`5x5` SSConv blocks and Kaiming initialization. The default
`--cnn-style gsdg` uses the GSDG `3x3`/`7x7` DwsConv path. Both styles output
`--graph-dim` channels and are instantiated independently for HSI and LiDAR.

Use `--prototype-scope hsi`, `lidar`, or `both` to add the optional HiH-style
class-prototype global hyperedges. Use `--graph-mode static` for fixed
geometry-based neighborhood hyperedges. The ordinary GSDG GAT implementation
is still available with `--backbone gsdg-graph`; prototype hyperedges must be
disabled in that mode. The legacy spelling `--hgcn-mode` remains an alias for
`--graph-mode`.

## Dual GSDG with HGCN fusion

`--architecture dual-gsdg-hgcn-fusion` selects the additional architecture:

```text
HSI pixels
  -> HSI SLIC -> stem -> FDSM -> dynamic Q/K Top-k -> two GATs --\
  -> GSDG graph decoder + 3x3/7x7 CNN -> GSDG weighted fusion      \
                                                                    -> HGCN weighted sum -> classifier
LiDAR pixels                                                        /
  -> height/residual/slope/curvature/roughness                     /
  -> Geometry-SLIC -> robust height descriptor MLP                /
  -> 1/2-hop RAG -> log(structure prior) Top-k -> Weighted GAT1   /
  -> rebuild graph -> Weighted GAT2                               /
  -> graph decoder + parallel 3x3/5x5/7x7 CNN -> weighted fusion -/
```

The HSI path retains the original GSDG pipeline. The LiDAR path replaces FDSM
with seven standardized geometric channels: elevation, local elevation
residual, x/y slopes, gradient magnitude, curvature, and local roughness.
Geometry-SLIC is computed from this stack. Every LiDAR superpixel node uses
the robust height descriptor
`[q10, q25, q50, q75, q90, mean, std, max-min]`, standardized per component
and encoded by an independent two-layer MLP.

LiDAR dynamic Top-k is hard-masked to the one- or two-hop region adjacency
graph. For every directly adjacent pair, the common-boundary term is the
mean DSM gradient across horizontal/vertical pixel pairs straddling that
boundary. The structural prior is
`P=A_xy^beta1 * A_z^beta2 * A_r^beta3 * A_b^beta4`, where the four terms
represent centroid proximity, mean-height similarity, height-standard-
deviation similarity, and common-boundary smoothness. Dynamic logits use
`QK/sqrt(d) + log(P+epsilon)`.

Positive dynamic edge weights are also added to GAT attention logits, so
the LiDAR Q/K graph builders receive gradients. GAT1 and GAT2 use independent
builders, and the graph is rebuilt between them. The final modality fusion
follows the fixed convex fusion used by HGCN:

```text
fused = modality_lambda * HSI + (1 - modality_lambda) * LiDAR
```

The new architecture defaults to two-hop LiDAR RAG candidates, the original
GSDG graph/CNN weight `0.95`, and equal HSI/LiDAR fusion. Train MUUFL with
20 pixels per class using:

```bash
python train.py --architecture dual-gsdg-hgcn-fusion --dataset muufl --train-samples-per-class 20 --device cuda
```

Use `--fusion-lambda` to change the graph/CNN fusion inside each GSDG branch,
and `--modality-fusion-lambda` to change the HSI/LiDAR fusion. Prototype
hyperedges are intentionally excluded from this option. LiDAR geometry can
be controlled with `--lidar-geometry-window`, `--lidar-rag-hops`,
`--lidar-{spatial,height,roughness,boundary}-weight`, and
`--lidar-edge-weight-beta`.

## Original HGCN-HL

`train_original.py` preserves the original pixel-vertex fixed hypergraph:
HSI SLIC regions and LiDAR Felzenszwalb regions are concatenated as static
hyperedges, followed by two HGCN layers and the original `5x5`/`5x5` CNN.
It uses exactly 20 randomly selected training pixels per class:

```bash
python train_original.py \
  --dataset muufl \
  --train-samples-per-class 20 \
  --lidar-segmentation felzenszwalb \
  --device cuda
```

Use `--lidar-segmentation slic` to replace the original LiDAR Felzenszwalb
region hyperedges with LiDAR SLIC region hyperedges.
