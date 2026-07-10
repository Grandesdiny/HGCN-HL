# HGCN-HL
Dual-branch GSDG-style dynamic superpixel graph classification for
hyperspectral and LiDAR data.

## Step-by-step demo

`train_original.py` remains the Stage-0 fixed-incidence HGCN-HL baseline.
`demo_train.py` is currently Stage 8:

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
  --fdsm-scope hsi \
  --device cuda
```

A mediator consensus graph can be enabled as the only exposed post-GAT2
cross-modal branch:

```bash
--post-gat-consensus-graph intersection-mediator \
--consensus-graph-fusion residual-c \
--consensus-graph-residual-init 0 \
--consensus-graph-weight 0.1 \
--consensus-graph-spatial-prior-weight 1.0 \
--consensus-graph-hsi-prior-weight 0.5 \
--consensus-graph-lidar-prior-weight 0.5 \
--bridge-attention-dk 32 \
--bridge-attention-topk 8
```

It is disabled by default with `--post-gat-consensus-graph none`. In the
default mediator readout, the C branch does not write messages back to HSI or
LiDAR superpixel nodes: the HSI and LiDAR private GAT2 nodes are projected to
pixels unchanged, and the C graph is a third pixel branch. The optional
`--consensus-graph-transport bidirectional` mode instead uses the same C graph
as a mediator transport kernel before HSI/LiDAR pixel projection. In both
cases, mediator C nodes aggregate HSI/LiDAR node features:

```text
H_C = B_CH V_H(H)
L_C = B_CL V_L(L)
R = concat(R_HC, R_LC)
X_HL = concat(V_H(H), V_L(L))
I_C = degree_norm(R^T X_HL)
C0 = LN(phi([H_C, L_C, I_C, abs(H_C - L_C), H_C * L_C, attrs]) + E_C)
```

`attrs` are the cell attributes from `build_common_refinement_cells()`.
Each C node is a nonempty HSI-superpixel/LiDAR-superpixel intersection cell,
so this branch requires exactly one superpixel scale.
`I_C` is the explicit incidence context from all HSI/LiDAR parent nodes to
the intersection C nodes; it is separate from the two modality-specific parent
contexts `H_C` and `L_C`.

The mediator graph is then constructed by C's own Q/K, not by HSI-to-LiDAR
bipartite attention. The HSI/LiDAR private GAT2 adjacencies only modulate the
C-QK logits as projected structure priors:

```text
P_C^H = row_norm(B_CH A_H B_HC)
P_C^L = row_norm(B_CL A_L B_LC)
P_C^S = intersection-cell RAG prior

M_C = (P_C^S > 0) OR (P_C^H > 0) OR (P_C^L > 0) OR I
logits_C = Q_C K_C^T / sqrt(d)
         + alpha_s log(P_C^S + eps)
         + alpha_h log(P_C^H + eps)
         + alpha_l log(P_C^L + eps)
A_C = TopKSoftmax(mask(logits_C, M_C))
```

To use the C graph as an HSI/LiDAR mediator instead of a third pixel branch:

```bash
--post-gat-consensus-graph intersection-mediator \
--consensus-graph-transport bidirectional \
--consensus-graph-transport-lambda 0.5 \
--consensus-graph-transport-gamma-init 0.0
```

This reuses the existing incidence matrices and the hard-supported C-QK graph:

```text
T_C = (1 - lambda_t) I + lambda_t A_C
Z_L_to_H = R_HC T_C R_CL V_L(L)
Z_H_to_L = R_LC T_C R_CH V_H(H)
H' = H + gamma_H g_H W_LH Z_L_to_H
L' = L + gamma_L g_L W_HL Z_H_to_L
```

The gates are local node gates:

```text
g_H = sigmoid(MLP([H, W_LH Z_L_to_H, abs(H - W_LH Z_L_to_H)]))
g_L = sigmoid(MLP([L, W_HL Z_H_to_L, abs(L - W_HL Z_H_to_L)]))
```

When this mode is enabled, the original pixel-level C residual/fixed/gated
third-branch fusion is skipped. The updated `H'` and `L'` are projected to
pixels and fused with `--graph-modality-lambda`.

When `--consensus-graph-transport none`, this adjacency is then used by an
independent C-GAT third branch. The propagation layer reuses the same
`MultiHeadGAT` implementation as the private HSI/LiDAR graph branches, with
the dynamic C-QK adjacency supplied as a weighted graph:

```text
C_GAT = MultiHeadGAT(C0, A_C)
Z_C1 = LN(C0 + gamma_c * C_GAT)
Z_C2 = LN(Z_C1 + FFN(Z_C1))
F_C = Q_C_pixel Z_C2
```

So C is not merely an attention prior: it is a third graph branch with its own
message passing. `Z_C2` is projected directly to pixels as
`consensus_graph_features`. By default, the final graph readout uses
`residual-c`, which starts exactly from the private HSI/LiDAR baseline:

```text
F_base = lambda * F_HSI + (1 - lambda) * F_LiDAR
r = sigmoid(MLP([F_base, F_consensus, abs(F_consensus - F_base)]))
F_graph = F_base + gamma_c * r * (F_consensus - F_base)
```

The pixel residual scale `gamma_c` is initialized by
`--consensus-graph-residual-init` and defaults to zero. The local gate is
initialized with a small sigmoid output. This makes the first forward pass
exactly match the two-private-graph baseline, useful for testing whether
accuracy drops are caused by over-strong C injection. The internal C-GAT
residual scale is separately controlled by `--consensus-graph-c-gamma-init`
and defaults to `0.1`.

For a gated ablation, use `--consensus-graph-fusion c-guided-gate`. Its final
linear layer is zero-weight initialized and biased to the fixed prior
`[lambda(1-w_c), (1-lambda)(1-w_c), w_c]`, not to uniform 1/3:

```text
gate = softmax(MLP([
    F_H, F_L, F_C,
    abs(F_H - F_C),
    abs(F_L - F_C),
    abs(F_H - F_L)
]))

F_graph = pi_H F_H + pi_L F_L + pi_C F_C
```

For a fixed-weight ablation, use `--consensus-graph-fusion fixed`:

```text
F_graph =
  (1 - w_c) * lambda * F_HSI
+ (1 - w_c) * (1 - lambda) * F_LiDAR
+ w_c * F_consensus
```

where `w_c` is `--consensus-graph-weight` and `lambda` is
`--graph-modality-lambda`.

For `intersection-mediator`, the C graph prior can be upgraded from binary
cell RAG to a LiDAR-aware weighted RAG:

```bash
--post-gat-consensus-graph intersection-mediator \
--consensus-graph-cell-edge spectral-height-boundary \
--cell-sam-weight 1.0 \
--cell-height-weight 1.0 \
--cell-boundary-weight 1.0 \
--cell-conflict-weight 0.0
```

This edge mode uses HSI spectral angle, LiDAR height difference, LiDAR boundary
gradient, and optional HSI/LiDAR boundary conflict. It only changes the
mediator C-C prior; there is no parent-cell-parent feedback branch in the
main entry point.

The joint pixel CNN is independently selectable:

```bash
--cnn-branch original  # default HGCN-HL 5x5/5x5 SSConv
--cnn-branch gsdg      # GSDG 3x3/7x7 depthwise-separable CNN
```

Both choices use the existing joint `PCA(HSI)+LiDAR` input and two-layer
1x1 WMF stem, which is structurally equivalent to the GSDG stem. The GSDG
choice uses equal-width `hidden_dim -> hidden_dim -> hidden_dim` DwsConv
blocks with kernel sizes 3 and 7. Thus it retains the GSDG spatial operator
but removes its 64-channel bottleneck and the former 64-to-hidden adapter,
making the comparison against the original 5x5/5x5 branch primarily a
kernel-layout comparison. The option is currently available with
`--graph-layout separate`; `original` remains the default.

The demo keeps the original
joint `PCA(HSI)+LiDAR` input, two WMF blocks, original `5x5/5x5` CNN,
lambda fusion, and classifier. In the default `--graph-layout separate`,
HSI-SLIC and LiDAR-SLIC use independent WMF graph encoders, spatial priors,
Q/K graph builders, and GATs. Their graph features are independently
projected to pixels and fused with `--graph-modality-lambda`; the unchanged
joint CNN is then fused with that graph result. When
`--post-gat-consensus-graph intersection-mediator` is enabled, the graph
result becomes a private-HSI/private-LiDAR/consensus-C fusion, but the HSI
and LiDAR private nodes remain unchanged. Use `--graph-layout joint` to
retain the previous concatenated-node single graph. LiDAR regions default to
SLIC; `--lidar-segmentation felzenszwalb` remains available.

`--fdsm-scope hsi` applies the original GSDG frequency-domain modulation
after HSI superpixel pooling. For LiDAR, `--lidar-modulation rag-lowhigh`
uses a geometry-gated decomposition into RAG-smoothed low-frequency and
RAG-residual high-frequency node features. The optional
`--lidar-graph-prior rag-height-knn` restricts LiDAR dynamic Top-k edges with
a local RAG and elevation similarity. The previous GAT1/GAT2 cross-modal
interaction, contrastive losses, MSSAGF/SACR anchor write-back, SPSN
prototype fusion, center-block, center-mediator, and parent-cell-parent cell
feedback routes have been removed from `demo_train.py`. The main line is
private dual graphs plus the optional intersection C-GAT third branch. The
demo does not use a fixed-incidence hypergraph. Outputs are isolated under
`model_demo/stage8_intersection_mediator_cgat`.

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

## Four-relation heterogeneous GSDG

`--architecture four-relation-hetero` combines the HSI GSDG encoder and
LiDAR Geometry-GSDG encoder with four directed relations:

```text
HSI   -- spectral-spatial --> HSI
LiDAR -- height-geometry  --> LiDAR
HSI   -- overlap          --> LiDAR
LiDAR -- overlap          --> HSI
```

The two cross-modal directions do not share Q/K/V projections or edge MLPs.
Their attention logits combine learned feature affinity, log superpixel
overlap, centroid distance, and relative x/y direction. A per-node relation
gate then performs a two-way softmax over the intra-modal and cross-modal
messages. With `--hetero-layers 2`, the first gated update rebuilds both
modality-specific dynamic graphs before GAT2, and a second four-relation
update follows GAT2. Use `--hetero-layers 1` for the one-layer ablation.

`--pixel-fusion adaptive` learns one HSI/LiDAR gate for every pixel from the
two projected features and their absolute difference. Use `fixed` to recover
the scalar `--modality-fusion-lambda` baseline.

```bash
python train.py \
  --architecture four-relation-hetero \
  --dataset muufl \
  --train-samples-per-class 20 \
  --hetero-layers 2 \
  --hetero-cross-dk 16 \
  --pixel-fusion adaptive \
  --device cuda
```

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
