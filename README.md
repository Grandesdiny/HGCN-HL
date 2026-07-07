# HGCN-HL
Dual-branch GSDG-style dynamic hypergraph classification for hyperspectral
and LiDAR data.

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
  --cross-modal-interaction overlap-qk-condition \
  --overlap-metric iou \
  --fdsm-scope hsi \
  --device cuda
```

An SPSN-inspired post-GAT prototype-correlation branch can be enabled on
this base pipeline:

```bash
--post-gat-prototype-fusion spsn-correlation \
--spsn-prototype-count 32 \
--spsn-correlation-temperature 0.2
```

It is disabled by default with `--post-gat-prototype-fusion none`. HSI and
LiDAR GAT2 nodes already represent modality-specific superpixel prototypes,
so no second SLIC/GAP stage is added. Independent classification-trained
selectors retain a fixed number of prototypes per modality. Node-to-selected-
prototype cosine correlations are computed before pixel projection and then
mapped through the sparse HSI/LiDAR assignment matrices. Two small residual
adapters inject the correlation maps into the corresponding pixel graph
features. A pixel-wise two-way reliability gate replaces fixed HSI/LiDAR
graph fusion and is initialized from `--graph-modality-lambda`; its output
continues through the unchanged graph/CNN `--fusion-lambda` fusion.

Unlike the original saliency-oriented SPSN, this branch does not copy
foreground-superpixel BCE or reliability pseudo-label losses. Pixel
classification supervision trains the selectors, correlation adapters, and
reliability gate end to end. The selected indices, mean selection scores, and
mean modality reliabilities are stored in each run's result JSON at logging
epochs.

A separate MSSAGF-inspired post-graph consensus interaction is available:

```bash
--post-gat-consensus mssagf-anchor \
--consensus-anchor-count 0 \
--consensus-temperature 0.2 \
--consensus-gamma-init 0 \
--consensus-fusion fixed \
--consensus-writeback direct \
--consensus-reliability-temperature 1.0
```

It is disabled by default with `--post-gat-consensus none`. An anchor count
of zero resolves to twice the dataset class count. After both modality-private
GAT2 layers, HSI and LiDAR nodes independently obtain soft assignments to one
learnable shared anchor bank. The one-layer modality projections and shared
anchor queries are L2-normalized before assignment so HSI and LiDAR feature
scales cannot make one assignment uniformly diffuse and the other collapse.
Each modality forms anchor features with raw superpixel-area weighting, the
two anchor sets are fused either with fixed 0.5/0.5 weights or per-anchor
modality reliability. Adaptive reliability is the softmax of the negative
area-weighted within-anchor reconstruction errors. The shared node features
are L2-normalized only in this error-estimation path so HSI FDSM and LiDAR
geometry features have comparable error units; the consensus values
themselves remain unchanged. These errors are detached before the softmax,
preventing the feature encoders from manipulating them to collapse the
weights. `direct` writes the consensus anchor itself back;
`difference` writes `consensus - modality_anchor`, removing self-copy and
making the message an explicit cross-modal correction. Both residual scales
start at zero by default, so the initial forward pass exactly recovers the
existing graph branch. Updated nodes are projected exactly once with their
original sparse assignment matrices, combined by `graph-modality-lambda`, and
then follow the unchanged graph/CNN fusion.

The three intended ablations are:

```bash
# Fixed 0.5 consensus + direct write-back
--post-gat-consensus mssagf-anchor \
--consensus-fusion fixed \
--consensus-writeback direct

# Adaptive reliability + direct write-back
--post-gat-consensus mssagf-anchor \
--consensus-fusion adaptive \
--consensus-writeback direct \
--consensus-reliability-temperature 1.0

# Adaptive reliability + difference write-back
--post-gat-consensus mssagf-anchor \
--consensus-fusion adaptive \
--consensus-writeback difference \
--consensus-reliability-temperature 1.0
```

Each logging record stores both per-anchor modality weights, per-anchor
weight entropy, detached reconstruction errors, both gamma values, and the
full area mass of every anchor plus empty-anchor counts.

The SACR extension remains in the same post-GAT2 location and is also
disabled unless requested. It keeps A3 as the main consensus:

```text
C0 = r_H * U_H + r_L * U_L
```

Then it builds an HSI anchor graph and a LiDAR anchor graph from the two
modality-specific anchor features and only adds a small structure-aligned
residual:

```text
C = C0 + eta * structure_gate *
    (r_H * (G_H U_H - U_H) + r_L * (G_L U_L - U_L))
```

`eta` is learnable and initializes to zero by default, so the initial forward
pass exactly recovers A3. This first version deliberately does not add a
learned anchor graph, an extra GCN block, LayerNorm/GELU, selective write-back
gate, TV loss, or orthogonal projection loss. It is meant to answer one
question cleanly: does cross-modal anchor-graph structure alignment improve
A3?

The intended follow-up ablations are:

```bash
# E0: A3 main baseline
--post-gat-consensus mssagf-anchor \
--consensus-fusion adaptive \
--consensus-writeback difference

# E1: A3 + SACR, eta initialized to zero, no adaptive structure reliability
--post-gat-consensus mssagf-anchor \
--consensus-fusion adaptive \
--consensus-writeback difference \
--consensus-anchor-reasoning sacr \
--consensus-structure-eta-init 0 \
--consensus-structure-reliability none

# E2: E1 + adaptive structure reliability from the HSI/LiDAR graph gap
--post-gat-consensus mssagf-anchor \
--consensus-fusion adaptive \
--consensus-writeback difference \
--consensus-anchor-reasoning sacr \
--consensus-structure-eta-init 0 \
--consensus-structure-reliability adaptive \
--consensus-structure-temperature 0.1 \
--consensus-anchor-graph-topk 8

# E3: weak orthogonal projection loss is intentionally not implemented yet
```

When SACR is enabled, logs additionally include eta, HSI/LiDAR anchor-graph
entropy, graph gap, structure gate min/mean/max, structure residual norm, and
HSI/LiDAR message norms.

This first ablation has no node gate, extra contrastive loss, or dense
HSI-by-LiDAR attention. To preserve exactly one cross-modal interaction and
pure modality-private graph construction, it requires `cross-modal-interaction
none`, `contrastive-mode none`, `cell-interaction none`, and
`post-gat-prototype-fusion none`. The implementation adapts the multiview
anchor-consensus principle of
[MSSAGF](https://github.com/W-Xinxin/MSSAGF); the reference repository itself
is a MATLAB clustering method rather than a neural fusion layer. The SACR
residual borrows the anchor-graph structure-alignment idea from
[OSMAGC](https://github.com/ZhangYongshan/OSMAGC) without importing its
orthogonal loss in this first pass.

A separate center-bridge block interaction is available as another post-GAT2
branch:

```bash
--post-gat-bridge center-block \
--bridge-anchor-count 0 \
--bridge-attention-dk 32 \
--bridge-attention-topk 8 \
--bridge-overlap-metric coverage \
--bridge-overlap-weight 1.0 \
--bridge-spatial-weight 1.0 \
--bridge-height-weight 1.0 \
--bridge-gamma-init 0
```

It is disabled by default with `--post-gat-bridge none`. A bridge count of
zero resolves to twice the dataset class count. The module first constructs a
public spatial bridge assignment `Q_C` from a deterministic grid over the
image. It then builds `Q_H^T Q_C` and `Q_L^T Q_C` overlap priors, adds
centroid-distance bias for both modalities, and adds LiDAR height-distribution
bias for the LiDAR-to-bridge relations. After HSI/LiDAR GAT2, the first
version uses only the minimal block closure:

```text
H receives: H self-view + C bridge-view
C receives: H view + C self-view + L view
L receives: C bridge-view + L self-view
```

The mediated direct HSI-LiDAR blocks `A_HL^C` and `A_LH^C` are intentionally
left off in this first ablation. Residual scales for H/C/L initialize to zero,
so the first forward pass is equivalent to the original late graph fusion.
This branch is mutually exclusive with consensus anchors, SPSN-style
prototype fusion, contrastive loss, cell interaction, and earlier
cross-modal graph interaction to keep the attribution clean.

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

An optional training-only contrastive objective can be added to this exact
Stage-8 pipeline:

```bash
--contrastive-mode overlap-prototype \
--prototype-objective cosine \
--contrastive-weight 0.05 \
--contrastive-temperature 0.2 \
--contrastive-dim 32
```

It is inserted after the two modality-specific GAT2 layers and immediately
before their node features are projected back to pixels. Row-normalized
shared-pixel counts `q_HL` and `q_LH` construct an opposite-modal structural
prototype for every node: `prototype_L = q_HL @ z_L` and
`prototype_H = q_LH @ z_H`. The default prototype objective minimizes
`1 - cosine(node, own_opposite_modal_prototype)` in both directions, without
using other nodes or prototypes as global negatives. Both directions are
weighted by
`1 - entropy(q) / log(number_of_overlapping_nodes)`. Independent projectors
align only a low-dimensional shared subspace; pixel projection and
classification continue to use the unprojected modality-private GAT2 node
features. The projectors and contrastive similarity matrix are skipped in
evaluation. `--prototype-objective infonce` retains the earlier global
prototype-negative formulation, while `overlap-soft` remains the
distribution-cross-entropy ablation. `--contrastive-mode none` is the default.

Overlap-constrained semantic transport distillation is available as a larger
training-only ablation and is also disabled by default:

```bash
--contrastive-mode overlap-transport \
--transport-semantic-weight 1.0 \
--transport-iterations 10 \
--transport-warmup-epochs 50 \
--contrastive-weight 0.05 \
--variance-weight 0.01 \
--variance-target 1.0 \
--contrastive-temperature 0.2 \
--contrastive-dim 32
```

The raw shared-pixel matrix `M = Q_H^T Q_L`, normalized by the image pixel
count, is already a feasible transport plan with HSI/LiDAR superpixel-area
marginals. For the first `transport-warmup-epochs`, this fixed plan is used
exactly. Semantic cosine similarity is then linearly introduced over the same
number of epochs. Log-domain Sinkhorn scaling preserves the area marginals,
and entries outside the true overlap support remain exactly zero. The
transport produces bidirectional opposite-modal structural prototypes.
Independent SimSiam-style predictors match each node to a stop-gradient
prototype; a small per-dimension variance penalty protects the projected
shared spaces from collapse. Projectors, predictors, transport, and losses
are absent from inference. This mode requires one superpixel scale so each
assignment matrix is a true pixel partition and its area marginals are
well-defined.

At each logging epoch, the result JSON stores both modalities' full
per-dimension projector standard deviations, bidirectional and mean
node-prototype cosine, maximum Sinkhorn row/column marginal errors, transport
entropy, and semantic-ramp progress. OA, AA, and Kappa remain recorded for
every run. The retained comparison modes are `none`, `overlap-soft`,
`overlap-prototype` with either `infonce` or `cosine`, and
`overlap-transport`.

Intersection-cell RAG interaction is an optional addition to this exact
pipeline and is disabled by default. Enable it by appending:

```bash
--cell-interaction rag
```

With this option, the two modality-specific GAT1 outputs are sent to one
node per nonempty HSI/LiDAR superpixel intersection. The cell nodes run one
sparse residual GCN over the common-refinement map's 1-hop RAG and return
coverage-weighted messages to both parent graphs through independent gates.
The updated parent features are then used by the existing IoU-conditioned
second-layer Q/K builders. Thus the original joint CNN, HSI FDSM, LiDAR
RAG-height-KNN, LiDAR low/high modulation, and overlap-Q/K condition remain
active. This initial cell mode requires one superpixel scale.

Four enhancements can be independently enabled on top of that command:

```bash
--cell-pixel-descriptor mean \
--cell-edge-mode spectral-height-boundary \
--cell-interaction-stages 2 \
--cell-output-branch fixed \
--cell-output-weight 0.3333333333 \
--cell-conflict-weight 1.0 \
--cell-topology-veto soft
```

`mean` adds the mean HSI-PCA vector and mean LiDAR value of the pixels
inside each cell. `spectral-height-boundary` replaces binary cell edge
values with sparse weights based on normalized spectral angle, mean-height
difference, and average DSM gradient along the shared cell boundary. Two
interaction stages run separate parent-cell-parent layers after GAT1 and
GAT2. The fixed output branch projects the latest cell features back with
the pixel-to-cell assignment and mixes them with the already fused
HSI/LiDAR graph output. The HSI boundary strength is the average spectral
angle of pixel pairs across the shared boundary, while the LiDAR boundary
strength is its average DSM gradient. After separate robust normalization,
`--cell-conflict-weight` adds their absolute disagreement
`abs(B_HSI - B_LiDAR)` to the edge penalty. It defaults to `0`, so this
additional term is disabled unless explicitly requested. The other three
edge terms can be controlled with
`--cell-sam-weight`, `--cell-height-weight`, and
`--cell-boundary-weight`.

Cell-derived topology veto is separately disabled by default. It maps the
cell adjacency back to both parent graphs:

```text
S_h = R_h<-c A_c R_c<-h
S_l = R_l<-c A_c R_c<-l
```

`--cell-topology-veto soft` applies independent learnable sigmoid gates to
these support matrices and inserts them into the second-layer Q/K logits
before Top-K. `hard` masks support below `--cell-veto-threshold` before
softmax and Top-K; self-loops are always retained. The veto combines with,
rather than replaces, the LiDAR RAG-height-KNN candidate mask.

All optional enhancements are disabled by default: descriptor `none`, edge
mode `binary`, one interaction stage, cell output `none`, conflict weight
`0`, and topology veto `none`. They require `--cell-interaction rag`.

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
overlap-qk-condition` aggregates the other modality through the directional
superpixel-overlap matrix after GAT1 and injects that context into both
modalities' independent Q/K projections when rebuilding the second graph.
It does not directly add cross-modal messages to node features.
`--overlap-metric iou` computes intersection over union before normalizing
the HSI-to-LiDAR and LiDAR-to-HSI aggregation rows; `coverage` retains the
earlier intersection-count weighting.
`overlap-gate` and `overlap-attention` remain available as earlier
interaction ablations. Each option defaults to `none` or the earlier
centroid prior for compatibility. The demo does not use a hypergraph or the
GSDG CNN. Outputs are isolated under
`model_demo/stage8_overlap_conditioned_qk`.

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

## Common-refinement Cell Bridge

`--architecture common-refinement-cell` keeps the HSI GSDG and LiDAR
Geometry-GSDG internal graphs, but can replace direct cross-modal attention
with explicit intersection-cell nodes. The feature is disabled by default:

```bash
python train.py \
  --architecture common-refinement-cell \
  --cell-interaction rag \
  --dataset muufl \
  --train-samples-per-class 20 \
  --device cuda
```

Every nonempty pair `S_h intersect S_l` becomes one cell. A binary parent
lookup gathers the HSI/LiDAR parent features without attenuation. Cell
features encode the two parents together with log-area, HSI/LiDAR coverage,
IoU, centroid distance, and relative y/x. Separate coverage-normalized
scatter weights then return the cell messages to HSI and LiDAR parents.
`rag` first propagates the fused cell features through one sparse residual
GCN on the common-refinement map's binary 1-hop RAG. Independent relation
gates see the original parent feature, the modality-internal GAT1 message,
and the returned cell message:

```text
HSI/LiDAR GAT1 (separate modality graphs)
  -> parent-to-cell incidence + Cell MLP
  -> optional cell-cell RAG-GCN
  -> gated Cell-to-HSI and Cell-to-LiDAR injection
  -> rebuild both modality-specific graphs
  -> GAT2
```

Use `--cell-interaction bridge` to remove only the cell-cell RAG-GCN, or
`--cell-interaction none` for the matched no-cell ablation. This initial
`train.py` architecture uses one cell interaction after GAT1 and does not
enable the four enhanced `demo_train.py` cell options above. It currently
requires exactly one superpixel scale so that the cells form a true image
partition.

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
