"""Stage-8 demo: overlap-conditioned second-layer dynamic graphs.

The fixed hypergraph/HGCN path is replaced by GSDG graph/GAT propagation.
The default uses independent HSI and LiDAR graphs; the previous concatenated
node graph remains selectable. The LiDAR graph can additionally restrict its
dynamic neighbors with a local RAG and an elevation-similarity KNN. The
original joint CNN and fusion remain. An intersection-cell RAG can be inserted
after GAT1, but is disabled by default.
"""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.sparse import coo_matrix, hstack, issparse
from sklearn.decomposition import PCA

from train import (
    DATASET_CONFIG,
    DwsConv,
    DynamicGraphBuilder,
    FDSM,
    MultiHeadGAT,
    OriginalSSConv,
    WMF,
    build_common_boundary_strength,
    build_common_refinement_cells,
    build_rag_hop_candidates,
    build_superpixel_spatial_prior,
    minmax_normalize,
    normalized_sparse_assignments,
    scipy_sparse_to_torch,
    set_seed,
    superpixel_centroids,
    split_fixed_samples_per_class,
    superpixel_height_distribution,
    symmetrically_normalize_sparse_adjacency,
)
from utils import (
    get_felzenszwalb_Segs,
    get_HSI_LiDAR_data,
    get_HSI_performance,
    get_SLIC_Segs,
)


STAGE = "stage8_overlap_conditioned_qk"


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Original HGCN-HL with its fixed hypergraph replaced by "
            "a GSDG-style dynamic superpixel graph."
        )
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIG),
        default="muufl",
    )
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument(
        "--train-samples-per-class",
        type=int,
        default=20,
    )
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--scales", type=int, nargs="+", default=None)
    parser.add_argument(
        "--lidar-segmentation",
        choices=("slic", "felzenszwalb"),
        default="slic",
        help=(
            "LiDAR superpixel generator. SLIC is the new default; "
            "Felzenszwalb preserves the earlier option."
        ),
    )
    parser.add_argument(
        "--graph-layout",
        choices=("separate", "joint"),
        default="separate",
        help=(
            "Use independent HSI/LiDAR GSDG graphs, or retain the "
            "previous concatenated-node joint graph."
        ),
    )
    parser.add_argument("--dynamic-dk", type=int, default=16)
    parser.add_argument("--dynamic-topk", type=int, default=8)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--spatial-prior-k", type=int, default=15)
    parser.add_argument(
        "--lidar-graph-prior",
        choices=("centroid", "rag-height-knn"),
        default="centroid",
        help=(
            "LiDAR dynamic-graph prior. 'centroid' preserves Stage 3; "
            "'rag-height-knn' applies a hard local RAG constraint and "
            "keeps the elevation-nearest regions inside that RAG."
        ),
    )
    parser.add_argument(
        "--lidar-rag-hops",
        type=int,
        choices=(1, 2),
        default=2,
        help="One- or two-hop LiDAR RAG used by rag-height-knn.",
    )
    parser.add_argument(
        "--lidar-height-knn-k",
        type=int,
        default=5,
        help=(
            "Number of elevation-nearest LiDAR regions retained per "
            "node inside the local RAG."
        ),
    )
    parser.add_argument(
        "--graph-modality-lambda",
        type=float,
        default=0.5,
        help="HSI weight when fusing the two separate graph outputs.",
    )
    parser.add_argument(
        "--cross-modal-interaction",
        choices=(
            "none",
            "overlap-gate",
            "overlap-attention",
            "overlap-qk-condition",
        ),
        default="none",
        help=(
            "Interaction between independent HSI/LiDAR graphs after "
            "GAT1. overlap-qk-condition uses cross-partition overlap "
            "context only to condition each modality's second Q/K graph. "
            "The default 'none' exactly preserves late fusion."
        ),
    )
    parser.add_argument(
        "--cross-attention-dk",
        type=int,
        default=16,
        help="Query/key dimension of overlap-constrained cross-attention.",
    )
    parser.add_argument(
        "--overlap-metric",
        choices=("iou", "coverage"),
        default="iou",
        help=(
            "Cross-modal superpixel correspondence. 'iou' uses "
            "intersection over union; 'coverage' preserves the earlier "
            "directional intersection/target-area weighting."
        ),
    )
    parser.add_argument(
        "--contrastive-mode",
        choices=(
            "none",
            "overlap-soft",
            "overlap-prototype",
            "overlap-transport",
        ),
        default="none",
        help=(
            "Training-only cross-modal contrastive alignment of GAT2 "
            "node features before pixel projection. overlap-transport "
            "uses overlap-supported semantic transport and SimSiam-style "
            "distillation. Default: none."
        ),
    )
    parser.add_argument(
        "--contrastive-weight",
        type=float,
        default=0.05,
        help="Lambda of the cross-modal contrastive loss. Default: 0.05.",
    )
    parser.add_argument(
        "--contrastive-temperature",
        type=float,
        default=0.2,
        help=(
            "Similarity temperature for overlap-soft, prototype "
            "InfoNCE, and transport semantics; unused by prototype "
            "cosine consistency. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--contrastive-dim",
        type=int,
        default=32,
        help="Shared projector output dimension. Default: 32.",
    )
    parser.add_argument(
        "--prototype-objective",
        choices=("cosine", "infonce"),
        default="cosine",
        help=(
            "Objective used by overlap-prototype. 'cosine' performs "
            "bidirectional node-prototype consistency without global "
            "negatives; 'infonce' retains the earlier prototype "
            "classification ablation. Default: cosine."
        ),
    )
    parser.add_argument(
        "--post-gat-prototype-fusion",
        choices=("none", "spsn-correlation"),
        default="none",
        help=(
            "Optional SPSN-inspired prototype selection, correlation "
            "maps, and pixel-wise modality reliability fusion after "
            "GAT2. Default: none."
        ),
    )
    parser.add_argument(
        "--post-gat-consensus",
        choices=("none", "mssagf-anchor"),
        default="none",
        help=(
            "Optional single post-GAT2 cross-modal interaction through "
            "shared consensus anchors. Default: none."
        ),
    )
    parser.add_argument(
        "--consensus-anchor-count",
        type=int,
        default=0,
        help=(
            "Number of shared consensus anchors. Zero resolves to twice "
            "the dataset class count. Default: 0."
        ),
    )
    parser.add_argument(
        "--consensus-temperature",
        type=float,
        default=0.2,
        help=(
            "Soft node-to-consensus-anchor assignment temperature. "
            "Default: 0.2."
        ),
    )
    parser.add_argument(
        "--consensus-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial learnable residual write-back scale for both "
            "modalities. Default: 0."
        ),
    )
    parser.add_argument(
        "--consensus-fusion",
        choices=("fixed", "adaptive"),
        default="fixed",
        help=(
            "Fuse modality-specific anchor features with fixed 0.5/0.5 "
            "or detached reconstruction-error reliability. "
            "Default: fixed."
        ),
    )
    parser.add_argument(
        "--consensus-writeback",
        choices=("direct", "difference"),
        default="direct",
        help=(
            "Write the consensus anchor directly or write only its "
            "difference from each modality anchor. Default: direct."
        ),
    )
    parser.add_argument(
        "--consensus-reliability-temperature",
        type=float,
        default=1.0,
        help=(
            "Temperature tau_r of adaptive per-anchor modality "
            "reliability. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--consensus-anchor-reasoning",
        choices=("none", "sacr"),
        default="none",
        help=(
            "Optional post-GAT2 reasoning on the shared consensus "
            "anchors. 'sacr' applies structure-aligned consensus "
            "refinement as a small residual on top of A3. Default: "
            "none."
        ),
    )
    parser.add_argument(
        "--consensus-structure-reliability",
        choices=("none", "adaptive"),
        default="none",
        help=(
            "Whether SACR uses the HSI/LiDAR anchor-graph gap to "
            "downweight structurally inconsistent anchors. Default: "
            "none."
        ),
    )
    parser.add_argument(
        "--consensus-anchor-graph-topk",
        type=int,
        default=8,
        help=(
            "Top-K neighbors per anchor when building HSI/LiDAR "
            "anchor graphs for SACR. Default: 8."
        ),
    )
    parser.add_argument(
        "--consensus-structure-temperature",
        type=float,
        default=0.1,
        help=(
            "Temperature for converting the HSI/LiDAR anchor-graph "
            "gap into SACR adaptive structure reliability. Default: "
            "0.1."
        ),
    )
    parser.add_argument(
        "--consensus-structure-eta-init",
        type=float,
        default=0.0,
        help=(
            "Initial eta for SACR residual refinement. Eta is "
            "learnable and defaults to 0 so the initial forward pass "
            "exactly recovers A3."
        ),
    )
    parser.add_argument(
        "--post-gat-bridge",
        choices=("none", "center-block"),
        default="none",
        help=(
            "Optional post-GAT2 center-bridge block interaction. "
            "'center-block' creates public spatial bridge anchors and "
            "uses H<->C<->L block attention before pixel projection. "
            "Default: none."
        ),
    )
    parser.add_argument(
        "--post-gat-consensus-graph",
        choices=("none", "center-mediator"),
        default="none",
        help=(
            "Optional post-GAT2 mediator consensus graph as an "
            "independent third pixel branch. It reuses public center "
            "bridge anchors but does not write messages back to HSI or "
            "LiDAR private graph nodes. Default: none."
        ),
    )
    parser.add_argument(
        "--consensus-graph-weight",
        type=float,
        default=0.333,
        help=(
            "Pixel-fusion weight of the mediator consensus graph "
            "branch. The remaining weight is split between HSI and "
            "LiDAR according to --graph-modality-lambda. Default: "
            "0.333."
        ),
    )
    parser.add_argument(
        "--bridge-anchor-count",
        type=int,
        default=0,
        help=(
            "Number of public center bridge anchors. Zero resolves to "
            "twice the dataset class count. Default: 0."
        ),
    )
    parser.add_argument(
        "--bridge-attention-dk",
        type=int,
        default=32,
        help="Query/key dimension of center bridge attention. Default: 32.",
    )
    parser.add_argument(
        "--bridge-attention-topk",
        type=int,
        default=8,
        help="Top-K entries per center bridge relation row. Default: 8.",
    )
    parser.add_argument(
        "--bridge-overlap-metric",
        choices=("coverage", "iou"),
        default="coverage",
        help=(
            "Overlap prior between modality superpixels and public "
            "bridge anchors. Default: coverage."
        ),
    )
    parser.add_argument(
        "--bridge-overlap-weight",
        type=float,
        default=1.0,
        help="Weight of log-overlap bridge attention bias. Default: 1.",
    )
    parser.add_argument(
        "--bridge-spatial-weight",
        type=float,
        default=1.0,
        help="Weight of centroid-distance bridge bias. Default: 1.",
    )
    parser.add_argument(
        "--bridge-height-weight",
        type=float,
        default=1.0,
        help=(
            "Weight of LiDAR height-distribution bridge bias. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--bridge-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scales for H/C/L bridge updates. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--spsn-prototype-count",
        type=int,
        default=32,
        help=(
            "Number of independently selected GAT2 prototypes per "
            "modality for spsn-correlation. Default: 32."
        ),
    )
    parser.add_argument(
        "--spsn-correlation-temperature",
        type=float,
        default=0.2,
        help=(
            "Cosine-correlation softmax temperature for the selected "
            "post-GAT prototypes. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--transport-semantic-weight",
        type=float,
        default=1.0,
        help=(
            "Beta multiplying semantic cosine similarity inside the "
            "overlap-supported Sinkhorn kernel. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--transport-iterations",
        type=int,
        default=10,
        help="Number of log-domain Sinkhorn iterations. Default: 10.",
    )
    parser.add_argument(
        "--transport-warmup-epochs",
        type=int,
        default=50,
        help=(
            "Epochs using fixed normalized overlap M before linearly "
            "introducing semantic transport over the same duration. "
            "Default: 50."
        ),
    )
    parser.add_argument(
        "--variance-weight",
        type=float,
        default=0.01,
        help=(
            "Lambda of the overlap-transport projector variance "
            "regularizer. Ignored by other modes. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--variance-target",
        type=float,
        default=1.0,
        help=(
            "Minimum per-dimension projector standard deviation gamma "
            "for overlap-transport. Default: 1.0."
        ),
    )
    parser.add_argument(
        "--cell-interaction",
        choices=("none", "rag"),
        default="none",
        help=(
            "Optional intersection-cell interaction after GAT1. "
            "'rag' performs parent-to-cell fusion, one sparse 1-hop "
            "cell RAG-GCN, and gated cell-to-parent feedback before "
            "the existing second-layer graph construction. Default: none."
        ),
    )
    parser.add_argument(
        "--cell-pixel-descriptor",
        choices=("none", "mean"),
        default="none",
        help=(
            "Optionally append mean HSI-PCA and LiDAR pixels inside "
            "each intersection cell to its node encoder. Default: none."
        ),
    )
    parser.add_argument(
        "--cell-edge-mode",
        choices=("binary", "spectral-height-boundary"),
        default="binary",
        help=(
            "Binary 1-hop cell RAG or a sparse RAG weighted by cell "
            "spectral angle, mean-height difference, and shared-boundary "
            "LiDAR gradient. Default: binary."
        ),
    )
    parser.add_argument(
        "--cell-interaction-stages",
        type=int,
        choices=(1, 2),
        default=1,
        help=(
            "Run cell interaction only after GAT1, or again after GAT2. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--cell-output-branch",
        choices=("none", "fixed"),
        default="none",
        help=(
            "Optionally project the latest cell features to pixels as "
            "a third fixed-weight graph branch. Default: none."
        ),
    )
    parser.add_argument(
        "--cell-output-weight",
        type=float,
        default=1.0 / 3.0,
        help="Cell pixel-branch weight when --cell-output-branch fixed.",
    )
    parser.add_argument("--cell-sam-weight", type=float, default=1.0)
    parser.add_argument("--cell-height-weight", type=float, default=1.0)
    parser.add_argument("--cell-boundary-weight", type=float, default=1.0)
    parser.add_argument(
        "--cell-conflict-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the absolute normalized HSI/LiDAR shared-boundary "
            "strength disagreement. Default: 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--cell-topology-veto",
        choices=("none", "soft", "hard"),
        default="none",
        help=(
            "Use the cell graph to attenuate or mask second-layer "
            "HSI/LiDAR Q/K edges before Top-K. Default: none."
        ),
    )
    parser.add_argument(
        "--cell-veto-threshold",
        type=float,
        default=0.05,
        help=(
            "Minimum mapped cell support retained by hard topology veto. "
            "Default: 0.05."
        ),
    )
    parser.add_argument(
        "--fdsm-scope",
        choices=("none", "hsi"),
        default="none",
        help=(
            "Apply the original GSDG frequency-domain modulation to "
            "pooled HSI superpixel features. Default: none."
        ),
    )
    parser.add_argument(
        "--lidar-modulation",
        choices=("none", "rag-lowhigh"),
        default="none",
        help=(
            "LiDAR-specific node modulation. 'rag-lowhigh' separates "
            "RAG-smoothed and RAG-residual features and fuses them with "
            "a geometry-conditioned gate. Default: none."
        ),
    )
    parser.add_argument(
        "--cnn-branch",
        choices=("original", "gsdg"),
        default="original",
        help=(
            "Joint pixel CNN fused with the graph output. 'original' "
            "keeps HGCN-HL 5x5/5x5 SSConv; 'gsdg' uses GSDG 3x3/7x7 "
            "depthwise-separable convolution at the same hidden width. "
            "Default: original."
        ),
    )
    parser.add_argument("--fusion-lambda", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_demo") / STAGE,
    )
    return parser.parse_args()


def resolve_options(args):
    config = DATASET_CONFIG[args.dataset]
    if args.data_dir is None:
        args.data_dir = config["data_dir"]
    if args.pca_components is None:
        args.pca_components = config["pca_components"]
    if args.scales is None:
        args.scales = config["scales"].copy()
    return config


def cell_configuration_tag(args):
    edge_tag = (
        "shb"
        if args.cell_edge_mode == "spectral-height-boundary"
        else "bin"
    )
    return (
        f"cell-{args.cell_interaction}_"
        f"cd-{args.cell_pixel_descriptor}_"
        f"ce-{edge_tag}_"
        f"cs-{args.cell_interaction_stages}_"
        f"co-{args.cell_output_branch}_"
        f"cw-{args.cell_sam_weight:g}-"
        f"{args.cell_height_weight:g}-"
        f"{args.cell_boundary_weight:g}-"
        f"{args.cell_conflict_weight:g}_"
        f"tv-{args.cell_topology_veto}-"
        f"{args.cell_veto_threshold:g}"
    )


def contrastive_configuration_tag(args):
    if args.contrastive_mode == "none":
        return "cm-none"
    mode_tag = {
        "overlap-soft": "soft",
        "overlap-prototype": "proto",
        "overlap-transport": "transport",
    }[args.contrastive_mode]
    objective_tag = (
        f"-p{args.prototype_objective}"
        if args.contrastive_mode == "overlap-prototype"
        else ""
    )
    transport_tag = (
        f"-b{args.transport_semantic_weight:g}"
        f"-i{args.transport_iterations}"
        f"-wu{args.transport_warmup_epochs}"
        f"-vw{args.variance_weight:g}"
        f"-vg{args.variance_target:g}"
        if args.contrastive_mode == "overlap-transport"
        else ""
    )
    return (
        f"cm-{mode_tag}{objective_tag}{transport_tag}-"
        f"w{args.contrastive_weight:g}-"
        f"t{args.contrastive_temperature:g}-"
        f"d{args.contrastive_dim}"
    )


def prototype_fusion_configuration_tag(args):
    if args.post_gat_prototype_fusion == "none":
        return "pgpf-none"
    return (
        f"pgpf-spsn-k{args.spsn_prototype_count}-"
        f"t{args.spsn_correlation_temperature:g}"
    )


def consensus_configuration_tag(args, class_count):
    if args.post_gat_consensus == "none":
        return "consensus-none"
    anchor_count = (
        args.consensus_anchor_count
        if args.consensus_anchor_count > 0
        else 2 * class_count
    )
    tag = (
        f"consensus-anchor-k{anchor_count}-"
        f"t{args.consensus_temperature:g}-"
        f"g{args.consensus_gamma_init:g}-"
        f"f{args.consensus_fusion}-"
        f"w{args.consensus_writeback}-"
        f"rt{args.consensus_reliability_temperature:g}"
    )
    if args.consensus_anchor_reasoning != "none":
        tag = (
            f"{tag}-ar{args.consensus_anchor_reasoning}-"
            f"k{args.consensus_anchor_graph_topk}-"
            f"sr{args.consensus_structure_reliability}-"
            f"st{args.consensus_structure_temperature:g}-"
            f"eta{args.consensus_structure_eta_init:g}"
        )
    return tag


def bridge_configuration_tag(args, class_count):
    if args.post_gat_bridge == "none":
        return "bridge-none"
    anchor_count = (
        args.bridge_anchor_count
        if args.bridge_anchor_count > 0
        else 2 * class_count
    )
    return (
        f"bridge-center-k{anchor_count}-"
        f"dk{args.bridge_attention_dk}-"
        f"top{args.bridge_attention_topk}-"
        f"ov{args.bridge_overlap_metric}-"
        f"w{args.bridge_overlap_weight:g}-"
        f"{args.bridge_spatial_weight:g}-"
        f"{args.bridge_height_weight:g}-"
        f"g{args.bridge_gamma_init:g}"
    )


def consensus_graph_configuration_tag(args, class_count):
    if args.post_gat_consensus_graph == "none":
        return "cg-none"
    anchor_count = (
        args.bridge_anchor_count
        if args.bridge_anchor_count > 0
        else 2 * class_count
    )
    return (
        f"cg-center-mediator-k{anchor_count}-"
        f"dk{args.bridge_attention_dk}-"
        f"top{args.bridge_attention_topk}-"
        f"ov{args.bridge_overlap_metric}-"
        f"w{args.consensus_graph_weight:g}-"
        f"bias{args.bridge_overlap_weight:g}-"
        f"{args.bridge_spatial_weight:g}-"
        f"{args.bridge_height_weight:g}-"
        f"g{args.bridge_gamma_init:g}"
    )


def safe_output_filename(stem, suffix, maximum_bytes=240):
    """Keep experiment filenames below common filesystem limits."""
    filename = f"{stem}{suffix}"
    if len(filename.encode("utf-8")) <= maximum_bytes:
        return filename
    digest = hashlib.sha1(
        stem.encode("utf-8")
    ).hexdigest()[:12]
    available = maximum_bytes - len(suffix) - len(digest) - 1
    return f"{stem[:available]}-{digest}{suffix}"


def build_modality_superpixel_assignments(
    hsi,
    lidar,
    scales,
    lidar_segmentation,
):
    """Build HSI SLIC and selectable LiDAR region node assignments."""
    height, width, bands = hsi.shape
    reduced = PCA(n_components=min(3, bands)).fit_transform(
        hsi.reshape(-1, bands)
    )
    reduced = reduced.reshape(height, width, -1)
    hsi_assignments = []
    lidar_assignments = []
    for scale in scales:
        hsi_assignments.append(
            get_SLIC_Segs(
            reduced,
            scale,
            sparse_output=True,
        )
        )
        if lidar_segmentation == "slic":
            lidar_assignment = get_SLIC_Segs(
                lidar,
                scale,
                sparse_output=True,
            )
        else:
            lidar_assignment = get_felzenszwalb_Segs(
                lidar,
                scale,
                sparse_output=True,
            )
        lidar_assignments.append(lidar_assignment)

    hsi_assignment = hstack(hsi_assignments, format="csr")
    lidar_assignment = hstack(lidar_assignments, format="csr")
    joint_assignment = hstack(
        [hsi_assignment, lidar_assignment],
        format="csr",
    )
    return (
        hsi_assignment,
        lidar_assignment,
        joint_assignment,
        lidar_assignments,
    )


def build_lidar_rag_modulation_structure(
    lidar_assignments,
    elevation,
    rag_hops,
):
    """Build normalized multi-scale RAG and physical node descriptors."""
    height, width = elevation.shape
    node_count = sum(part.shape[1] for part in lidar_assignments)
    rag_adjacency = np.zeros(
        (node_count, node_count),
        dtype=np.float32,
    )
    descriptors = np.zeros((node_count, 3), dtype=np.float32)
    gradient_y, gradient_x = np.gradient(
        np.asarray(elevation, dtype=np.float32)
    )
    gradient_magnitude = np.sqrt(
        gradient_x * gradient_x + gradient_y * gradient_y
    )
    offset = 0

    for assignment in lidar_assignments:
        scale_nodes = assignment.shape[1]
        scale_slice = slice(offset, offset + scale_nodes)
        segments = np.asarray(
            assignment.argmax(axis=1)
        ).reshape(height, width)
        scale_rag = build_rag_hop_candidates(
            segments,
            hops=rag_hops,
        ).toarray().astype(np.float32)
        scale_rag /= np.maximum(
            scale_rag.sum(axis=1, keepdims=True),
            1.0,
        )
        rag_adjacency[scale_slice, scale_slice] = scale_rag

        raw_height, _ = superpixel_height_distribution(
            assignment,
            elevation,
        )
        boundary_strength = build_common_boundary_strength(
            segments,
            gradient_magnitude,
        )
        direct_rag = build_rag_hop_candidates(
            segments,
            hops=1,
        ).toarray().astype(np.float32)
        np.fill_diagonal(direct_rag, 0.0)
        mean_boundary = np.asarray(
            boundary_strength.sum(axis=1)
        ).reshape(-1) / np.maximum(
            direct_rag.sum(axis=1),
            1.0,
        )
        descriptors[scale_slice] = np.stack(
            [
                raw_height[:, 6],
                raw_height[:, 7],
                mean_boundary,
            ],
            axis=1,
        )
        offset += scale_nodes

    descriptor_mean = descriptors.mean(axis=0, keepdims=True)
    descriptor_std = descriptors.std(axis=0, keepdims=True)
    descriptors = (
        (descriptors - descriptor_mean)
        / np.maximum(descriptor_std, 1e-6)
    ).astype(np.float32)
    return rag_adjacency, descriptors


def build_lidar_rag_height_knn_prior(
    lidar_assignments,
    elevation,
    centroid_prior,
    rag_adjacency,
    height_knn_k,
):
    """Build a hard local RAG mask and an elevation-weighted prior.

    Each segmentation scale is handled independently. For every region, the
    K closest mean-elevation regions are retained only from its 1/2-hop RAG.
    """
    node_count = sum(part.shape[1] for part in lidar_assignments)
    candidate_mask = np.zeros((node_count, node_count), dtype=bool)
    constrained_prior = np.zeros(
        (node_count, node_count),
        dtype=np.float32,
    )
    offset = 0

    for assignment in lidar_assignments:
        scale_nodes = assignment.shape[1]
        scale_slice = slice(offset, offset + scale_nodes)
        rag = rag_adjacency[
            scale_slice,
            scale_slice,
        ] > 0
        raw_height, _ = superpixel_height_distribution(
            assignment,
            elevation,
        )
        mean_height = raw_height[:, 5]
        height_distance = np.abs(
            mean_height[:, None] - mean_height[None, :]
        )
        positive_distance = height_distance[
            np.logical_and(rag, height_distance > 0)
        ]
        height_sigma = (
            float(np.median(positive_distance))
            if positive_distance.size
            else 1.0
        )
        height_similarity = np.exp(
            -(height_distance * height_distance)
            / (2.0 * max(height_sigma, 1e-6) ** 2)
        ).astype(np.float32)

        scale_candidates = np.eye(scale_nodes, dtype=bool)
        for node_index in range(scale_nodes):
            local_neighbors = np.flatnonzero(rag[node_index])
            local_neighbors = local_neighbors[
                local_neighbors != node_index
            ]
            if local_neighbors.size == 0:
                continue
            k = min(height_knn_k, local_neighbors.size)
            nearest = local_neighbors[
                np.argpartition(
                    height_distance[node_index, local_neighbors],
                    kth=k - 1,
                )[:k]
            ]
            scale_candidates[node_index, nearest] = True

        # Undirected GAT support: retain an edge selected in either direction.
        scale_candidates = np.logical_or(
            scale_candidates,
            scale_candidates.T,
        )
        scale_candidates &= rag
        np.fill_diagonal(scale_candidates, True)

        scale_prior = centroid_prior[scale_slice, scale_slice].copy()
        scale_prior *= height_similarity
        scale_prior[~scale_candidates] = 0.0
        np.fill_diagonal(scale_prior, 1.0)
        candidate_mask[scale_slice, scale_slice] = scale_candidates
        constrained_prior[scale_slice, scale_slice] = scale_prior
        offset += scale_nodes

    return constrained_prior, candidate_mask


class MaskedDynamicGraphBuilder(DynamicGraphBuilder):
    """GSDG Q/K Top-k with an exact hard candidate-edge mask."""

    def __init__(self, *args, candidate_mask, **kwargs):
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "candidate_mask",
            torch.as_tensor(candidate_mask, dtype=torch.bool),
            persistent=False,
        )

    def forward(self, node_features, spatial_prior):
        query = self.query(node_features) + self.position_encoding
        key = self.key(node_features) + self.position_encoding
        logits = (
            query @ key.t() * self.scale
            + torch.log(spatial_prior + 1e-6)
        ) / self.tau
        logits = logits.masked_fill(
            ~self.candidate_mask,
            torch.finfo(logits.dtype).min,
        )
        scores = torch.softmax(logits, dim=-1)

        k = min(self.topk, scores.size(1))
        values, indices = torch.topk(scores, k=k, dim=-1)
        adjacency = torch.zeros_like(scores).scatter_(
            dim=-1,
            index=indices,
            src=values,
        )
        adjacency *= self.candidate_mask.to(adjacency.dtype)
        if self.symmetrize:
            adjacency = torch.maximum(adjacency, adjacency.t())
        adjacency /= adjacency.sum(dim=-1, keepdim=True) + 1e-6
        self.last_adjacency = adjacency.detach()
        return adjacency


class CrossConditionedDynamicGraphBuilder(DynamicGraphBuilder):
    """Second-layer Q/K graph conditioned by overlap-aligned other nodes."""

    def __init__(
        self,
        *args,
        cross_channels,
        candidate_mask=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        d_k = self.query.out_features
        self.cross_query = nn.Linear(cross_channels, d_k)
        self.cross_key = nn.Linear(cross_channels, d_k)
        self.register_buffer(
            "candidate_mask",
            (
                torch.as_tensor(candidate_mask, dtype=torch.bool)
                if candidate_mask is not None
                else None
            ),
            persistent=False,
        )
        self.last_cross_context = None
        self.last_topology_gate = None
        self.last_topology_mask = None

    def forward(
        self,
        node_features,
        spatial_prior,
        cross_context,
        topology_gate=None,
        topology_mask=None,
    ):
        if cross_context.shape[0] != node_features.shape[0]:
            raise ValueError(
                "Cross context must have one row per target node."
            )
        query = (
            self.query(node_features)
            + self.cross_query(cross_context)
            + self.position_encoding
        )
        key = (
            self.key(node_features)
            + self.cross_key(cross_context)
            + self.position_encoding
        )
        logits = (
            query @ key.transpose(0, 1) * self.scale
            + torch.log(spatial_prior + 1e-6)
        ) / self.tau
        if topology_gate is not None:
            if topology_gate.shape != logits.shape:
                raise ValueError(
                    "Topology gate must match the Q/K graph shape."
                )
            logits = logits + torch.log(topology_gate + 1e-6)
        combined_mask = topology_mask
        if self.candidate_mask is not None:
            combined_mask = (
                self.candidate_mask
                if combined_mask is None
                else torch.logical_and(
                    combined_mask,
                    self.candidate_mask,
                )
            )
        if combined_mask is not None:
            logits = logits.masked_fill(
                ~combined_mask,
                torch.finfo(logits.dtype).min,
            )
        scores = torch.softmax(logits, dim=-1)

        k = min(self.topk, scores.size(1))
        values, indices = torch.topk(scores, k=k, dim=-1)
        adjacency = torch.zeros_like(scores).scatter_(
            dim=-1,
            index=indices,
            src=values,
        )
        if combined_mask is not None:
            adjacency = adjacency * combined_mask.to(
                adjacency.dtype
            )
        if self.symmetrize:
            adjacency = torch.maximum(
                adjacency,
                adjacency.transpose(0, 1),
            )
        if topology_gate is not None:
            adjacency = adjacency * topology_gate
        if combined_mask is not None:
            adjacency = adjacency * combined_mask.to(
                adjacency.dtype
            )
        adjacency = adjacency / (
            adjacency.sum(dim=-1, keepdim=True) + 1e-6
        )
        self.last_cross_context = cross_context.detach()
        self.last_topology_gate = (
            topology_gate.detach()
            if topology_gate is not None
            else None
        )
        self.last_topology_mask = (
            topology_mask.detach()
            if topology_mask is not None
            else None
        )
        self.last_adjacency = adjacency.detach()
        return adjacency


def build_cross_modal_overlap(
    hsi_assignment,
    lidar_assignment,
    metric="iou",
):
    """Return directional row-normalized IoU or coverage weights."""
    overlap = hsi_assignment.transpose() @ lidar_assignment
    if issparse(overlap):
        overlap = overlap.toarray()
    overlap = np.asarray(overlap, dtype=np.float32)
    if metric == "iou":
        hsi_area = np.asarray(
            hsi_assignment.sum(axis=0)
        ).reshape(-1).astype(np.float32)
        lidar_area = np.asarray(
            lidar_assignment.sum(axis=0)
        ).reshape(-1).astype(np.float32)
        union = (
            hsi_area[:, None]
            + lidar_area[None, :]
            - overlap
        )
        correspondence = np.divide(
            overlap,
            np.maximum(union, 1.0),
            out=np.zeros_like(overlap),
            where=overlap > 0,
        )
    elif metric == "coverage":
        correspondence = overlap
    else:
        raise ValueError(
            "Overlap metric must be 'iou' or 'coverage'."
        )

    hsi_to_lidar = correspondence / np.maximum(
        correspondence.sum(axis=1, keepdims=True),
        1e-6,
    )
    lidar_to_hsi = correspondence.T / np.maximum(
        correspondence.T.sum(axis=1, keepdims=True),
        1e-6,
    )
    return (
        hsi_to_lidar.astype(np.float32),
        lidar_to_hsi.astype(np.float32),
    )


def build_center_bridge_assignment(height, width, anchor_count):
    """Build an exact-K public spatial grid assignment Q_C."""
    if anchor_count <= 0:
        raise ValueError("Bridge anchor count must be positive.")
    target_rows = int(
        round(np.sqrt(anchor_count * height / max(width, 1)))
    )
    rows = int(np.clip(target_rows, 1, anchor_count))
    while rows > 1 and anchor_count // rows == 0:
        rows -= 1
    base_columns = anchor_count // rows
    extra_columns = anchor_count % rows
    if base_columns == 0:
        rows = 1
        base_columns = anchor_count
        extra_columns = 0

    labels = np.zeros((height, width), dtype=np.int64)
    offset = 0
    for row_index in range(rows):
        y_start = int(np.floor(row_index * height / rows))
        y_end = int(np.floor((row_index + 1) * height / rows))
        if row_index == rows - 1:
            y_end = height
        columns = base_columns + (
            1 if row_index < extra_columns else 0
        )
        x_bins = np.floor(
            np.arange(width, dtype=np.float32)
            * columns
            / max(width, 1)
        ).astype(np.int64)
        x_bins = np.minimum(x_bins, columns - 1)
        labels[y_start:y_end, :] = offset + x_bins[None, :]
        offset += columns
    if offset != anchor_count:
        raise RuntimeError(
            "Internal bridge grid construction did not produce "
            "the requested number of anchors."
        )
    pixel_indices = np.arange(height * width, dtype=np.int64)
    return coo_matrix(
        (
            np.ones(height * width, dtype=np.float32),
            (pixel_indices, labels.reshape(-1)),
        ),
        shape=(height * width, anchor_count),
        dtype=np.float32,
    ).tocsr()


def _dense_overlap(source_assignment, target_assignment):
    overlap = source_assignment.transpose() @ target_assignment
    if issparse(overlap):
        overlap = overlap.toarray()
    return np.asarray(overlap, dtype=np.float32)


def _directional_overlap_prior(
    overlap,
    source_area,
    target_area,
    metric,
):
    if metric == "iou":
        union = (
            source_area[:, None]
            + target_area[None, :]
            - overlap
        )
        correspondence = np.divide(
            overlap,
            np.maximum(union, 1.0),
            out=np.zeros_like(overlap),
            where=overlap > 0,
        )
    elif metric == "coverage":
        correspondence = overlap
    else:
        raise ValueError(
            "Bridge overlap metric must be 'coverage' or 'iou'."
        )
    return (
        correspondence
        / np.maximum(correspondence.sum(axis=1, keepdims=True), 1e-6)
    ).astype(np.float32)


def _negative_centroid_distance_bias(source_centroids, target_centroids):
    delta = source_centroids[:, None, :] - target_centroids[None, :, :]
    squared_distance = np.sum(delta * delta, axis=2).astype(np.float32)
    positive = squared_distance[squared_distance > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    return (-squared_distance / max(sigma, 1e-6)).astype(np.float32)


def _negative_height_distribution_bias(source_height, target_height):
    source_descriptor = source_height[:, [5, 6, 1, 2, 3]]
    target_descriptor = target_height[:, [5, 6, 1, 2, 3]]
    distance = np.mean(
        np.abs(
            source_descriptor[:, None, :]
            - target_descriptor[None, :, :]
        ),
        axis=2,
    ).astype(np.float32)
    positive = distance[distance > 0]
    sigma = float(np.median(positive)) if positive.size else 1.0
    return (-distance / max(sigma, 1e-6)).astype(np.float32)


def build_center_bridge_data(
    hsi_assignment,
    lidar_assignment,
    elevation,
    height,
    width,
    anchor_count,
    overlap_metric="coverage",
    overlap_weight=1.0,
    spatial_weight=1.0,
    height_weight=1.0,
):
    """Build Q_C and all fixed priors for center-bridge attention."""
    bridge_assignment = build_center_bridge_assignment(
        height,
        width,
        anchor_count,
    )
    hsi_area = np.asarray(hsi_assignment.sum(axis=0)).reshape(-1)
    lidar_area = np.asarray(lidar_assignment.sum(axis=0)).reshape(-1)
    bridge_area = np.asarray(
        bridge_assignment.sum(axis=0)
    ).reshape(-1)

    overlap_hc = _dense_overlap(hsi_assignment, bridge_assignment)
    overlap_lc = _dense_overlap(lidar_assignment, bridge_assignment)
    prior_hc = _directional_overlap_prior(
        overlap_hc,
        hsi_area,
        bridge_area,
        overlap_metric,
    )
    prior_ch = _directional_overlap_prior(
        overlap_hc.T,
        bridge_area,
        hsi_area,
        overlap_metric,
    )
    prior_lc = _directional_overlap_prior(
        overlap_lc,
        lidar_area,
        bridge_area,
        overlap_metric,
    )
    prior_cl = _directional_overlap_prior(
        overlap_lc.T,
        bridge_area,
        lidar_area,
        overlap_metric,
    )

    hsi_centroids = superpixel_centroids(
        hsi_assignment,
        height,
        width,
    )
    lidar_centroids = superpixel_centroids(
        lidar_assignment,
        height,
        width,
    )
    bridge_centroids = superpixel_centroids(
        bridge_assignment,
        height,
        width,
    )
    spatial_hc = _negative_centroid_distance_bias(
        hsi_centroids,
        bridge_centroids,
    )
    spatial_lc = _negative_centroid_distance_bias(
        lidar_centroids,
        bridge_centroids,
    )
    spatial_cc = _negative_centroid_distance_bias(
        bridge_centroids,
        bridge_centroids,
    )
    lidar_height, _ = superpixel_height_distribution(
        lidar_assignment,
        elevation,
    )
    bridge_height, _ = superpixel_height_distribution(
        bridge_assignment,
        elevation,
    )
    height_lc = _negative_height_distribution_bias(
        lidar_height,
        bridge_height,
    )

    eps = 1e-6
    bias_hc = (
        overlap_weight * np.log(prior_hc + eps)
        + spatial_weight * spatial_hc
    )
    bias_ch = (
        overlap_weight * np.log(prior_ch + eps)
        + spatial_weight * spatial_hc.T
    )
    bias_lc = (
        overlap_weight * np.log(prior_lc + eps)
        + spatial_weight * spatial_lc
        + height_weight * height_lc
    )
    bias_cl = (
        overlap_weight * np.log(prior_cl + eps)
        + spatial_weight * spatial_lc.T
        + height_weight * height_lc.T
    )
    bias_cc = spatial_weight * spatial_cc
    return {
        "assignment": bridge_assignment.astype(np.float32),
        "prior_hc": prior_hc.astype(np.float32),
        "prior_ch": prior_ch.astype(np.float32),
        "prior_lc": prior_lc.astype(np.float32),
        "prior_cl": prior_cl.astype(np.float32),
        "bias_hc": bias_hc.astype(np.float32),
        "bias_ch": bias_ch.astype(np.float32),
        "bias_lc": bias_lc.astype(np.float32),
        "bias_cl": bias_cl.astype(np.float32),
        "bias_cc": bias_cc.astype(np.float32),
        "anchor_count": int(anchor_count),
        "area": bridge_area.astype(np.float32),
    }


def build_overlap_distribution_targets(
    hsi_assignment,
    lidar_assignment,
):
    """Build count-normalized overlap targets and entropy confidence."""
    overlap = hsi_assignment.transpose() @ lidar_assignment
    if issparse(overlap):
        overlap = overlap.toarray()
    overlap = np.asarray(overlap, dtype=np.float32)

    def directional_targets(counts):
        row_sum = counts.sum(axis=1, keepdims=True)
        targets = counts / np.maximum(row_sum, 1e-6)
        positive_count = np.count_nonzero(counts, axis=1)
        entropy = -np.sum(
            np.where(
                targets > 0,
                targets * np.log(np.maximum(targets, 1e-12)),
                0.0,
            ),
            axis=1,
        )
        confidence = np.ones(counts.shape[0], dtype=np.float32)
        ambiguous = positive_count > 1
        confidence[ambiguous] = (
            1.0
            - entropy[ambiguous]
            / np.log(positive_count[ambiguous])
        )
        confidence[positive_count == 0] = 0.0
        return (
            targets.astype(np.float32),
            np.clip(confidence, 0.0, 1.0).astype(np.float32),
        )

    hsi_to_lidar, hsi_confidence = directional_targets(overlap)
    lidar_to_hsi, lidar_confidence = directional_targets(
        overlap.transpose()
    )
    return {
        "hsi_to_lidar": hsi_to_lidar,
        "lidar_to_hsi": lidar_to_hsi,
        "hsi_confidence": hsi_confidence,
        "lidar_confidence": lidar_confidence,
    }


def build_overlap_transport_targets(
    hsi_assignment,
    lidar_assignment,
):
    """Build a feasible overlap transport plan and its area marginals."""
    overlap = hsi_assignment.transpose() @ lidar_assignment
    if issparse(overlap):
        overlap = overlap.toarray()
    overlap = np.asarray(overlap, dtype=np.float32)
    pixel_count = float(overlap.sum())
    if pixel_count <= 0:
        raise ValueError("Cross-modal overlap matrix is empty.")
    row_marginal = overlap.sum(axis=1) / pixel_count
    column_marginal = overlap.sum(axis=0) / pixel_count
    if np.any(row_marginal <= 0) or np.any(column_marginal <= 0):
        raise ValueError(
            "Every superpixel must overlap at least one opposite-modal "
            "superpixel."
        )
    return {
        "overlap_count": overlap,
        "overlap_mass": overlap / pixel_count,
        "overlap_support": overlap > 0,
        "row_marginal": row_marginal.astype(np.float32),
        "column_marginal": column_marginal.astype(np.float32),
    }


class OverlapDistributionContrastiveLoss(nn.Module):
    """Multi-positive cross-modal alignment in a small shared subspace."""

    def __init__(
        self,
        channels,
        projection_dim,
        temperature,
        target_data,
    ):
        super().__init__()
        self.temperature = temperature
        self.hsi_projector = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, projection_dim),
        )
        self.lidar_projector = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, projection_dim),
        )
        for name, value in target_data.items():
            self.register_buffer(
                name,
                torch.as_tensor(value, dtype=torch.float32),
                persistent=False,
            )
        self.last_hsi_to_lidar_loss = None
        self.last_lidar_to_hsi_loss = None

    @staticmethod
    def _directional_loss(logits, targets, confidence):
        log_probability = F.log_softmax(logits, dim=1)
        per_anchor = -torch.sum(
            targets * log_probability,
            dim=1,
        )
        return torch.sum(confidence * per_anchor) / (
            confidence.sum() + 1e-6
        )

    def forward(self, hsi_features, lidar_features):
        hsi_shared = F.normalize(
            self.hsi_projector(hsi_features),
            dim=1,
        )
        lidar_shared = F.normalize(
            self.lidar_projector(lidar_features),
            dim=1,
        )
        logits = (
            hsi_shared @ lidar_shared.transpose(0, 1)
        ) / self.temperature
        hsi_to_lidar_loss = self._directional_loss(
            logits,
            self.hsi_to_lidar,
            self.hsi_confidence,
        )
        lidar_to_hsi_loss = self._directional_loss(
            logits.transpose(0, 1),
            self.lidar_to_hsi,
            self.lidar_confidence,
        )
        self.last_hsi_to_lidar_loss = (
            hsi_to_lidar_loss.detach()
        )
        self.last_lidar_to_hsi_loss = (
            lidar_to_hsi_loss.detach()
        )
        return 0.5 * (
            hsi_to_lidar_loss + lidar_to_hsi_loss
        )


class OverlapPrototypeContrastiveLoss(nn.Module):
    """Contrast nodes against overlap-aggregated opposite-modal prototypes."""

    def __init__(
        self,
        channels,
        projection_dim,
        temperature,
        target_data,
        objective="cosine",
    ):
        super().__init__()
        self.temperature = temperature
        self.objective = objective
        self.hsi_projector = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, projection_dim),
        )
        self.lidar_projector = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, projection_dim),
        )
        for name, value in target_data.items():
            self.register_buffer(
                name,
                torch.as_tensor(value, dtype=torch.float32),
                persistent=False,
            )
        self.last_hsi_to_lidar_loss = None
        self.last_lidar_to_hsi_loss = None
        self.last_lidar_prototypes = None
        self.last_hsi_prototypes = None

    def _directional_loss(
        self,
        anchors,
        prototypes,
        confidence,
    ):
        if self.objective == "cosine":
            per_anchor = 1.0 - torch.sum(
                anchors * prototypes,
                dim=1,
            )
        else:
            logits = (
                anchors @ prototypes.transpose(0, 1)
            ) / self.temperature
            labels = torch.arange(
                anchors.shape[0],
                device=anchors.device,
            )
            per_anchor = F.cross_entropy(
                logits,
                labels,
                reduction="none",
            )
        return torch.sum(confidence * per_anchor) / (
            confidence.sum() + 1e-6
        )

    def forward(self, hsi_features, lidar_features):
        hsi_shared = F.normalize(
            self.hsi_projector(hsi_features),
            dim=1,
        )
        lidar_shared = F.normalize(
            self.lidar_projector(lidar_features),
            dim=1,
        )
        lidar_prototypes = F.normalize(
            self.hsi_to_lidar @ lidar_shared,
            dim=1,
        )
        hsi_prototypes = F.normalize(
            self.lidar_to_hsi @ hsi_shared,
            dim=1,
        )
        hsi_to_lidar_loss = self._directional_loss(
            hsi_shared,
            lidar_prototypes,
            self.hsi_confidence,
        )
        lidar_to_hsi_loss = self._directional_loss(
            lidar_shared,
            hsi_prototypes,
            self.lidar_confidence,
        )
        self.last_hsi_to_lidar_loss = (
            hsi_to_lidar_loss.detach()
        )
        self.last_lidar_to_hsi_loss = (
            lidar_to_hsi_loss.detach()
        )
        self.last_lidar_prototypes = lidar_prototypes.detach()
        self.last_hsi_prototypes = hsi_prototypes.detach()
        return 0.5 * (
            hsi_to_lidar_loss + lidar_to_hsi_loss
        )


class OverlapTransportDistillationLoss(nn.Module):
    """Overlap-supported semantic transport with SimSiam distillation."""

    def __init__(
        self,
        channels,
        projection_dim,
        temperature,
        target_data,
        semantic_weight=1.0,
        sinkhorn_iterations=10,
        warmup_epochs=50,
        variance_target=1.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.semantic_weight = semantic_weight
        self.sinkhorn_iterations = sinkhorn_iterations
        self.warmup_epochs = warmup_epochs
        self.variance_target = variance_target
        self.current_epoch = 0
        self.hsi_projector = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, projection_dim),
        )
        self.lidar_projector = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, projection_dim),
        )
        self.hsi_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LeakyReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        self.lidar_predictor = nn.Sequential(
            nn.Linear(projection_dim, projection_dim),
            nn.LeakyReLU(),
            nn.Linear(projection_dim, projection_dim),
        )
        for name, value in target_data.items():
            tensor = torch.as_tensor(value)
            if tensor.dtype != torch.bool:
                tensor = tensor.to(dtype=torch.float32)
            self.register_buffer(name, tensor, persistent=False)
        self.last_variance_loss = None
        self.last_diagnostics = None
        self.last_transport = None

    def set_epoch(self, epoch):
        self.current_epoch = int(epoch)

    def _semantic_progress(self):
        if self.warmup_epochs == 0:
            return 1.0
        if self.current_epoch <= self.warmup_epochs:
            return 0.0
        return min(
            1.0,
            (self.current_epoch - self.warmup_epochs)
            / float(self.warmup_epochs),
        )

    def _sinkhorn(self, semantic_similarity, semantic_progress):
        if semantic_progress == 0.0 or self.semantic_weight == 0.0:
            return self.overlap_mass
        negative_infinity = torch.full_like(
            self.overlap_count,
            -torch.inf,
        )
        log_kernel = torch.where(
            self.overlap_support,
            torch.log(self.overlap_count.clamp_min(1e-12))
            + self.semantic_weight
            * semantic_progress
            * semantic_similarity,
            negative_infinity,
        )
        log_row = torch.log(self.row_marginal.clamp_min(1e-12))
        log_column = torch.log(
            self.column_marginal.clamp_min(1e-12)
        )
        log_u = torch.zeros_like(log_row)
        log_v = torch.zeros_like(log_column)
        for _ in range(self.sinkhorn_iterations):
            log_u = log_row - torch.logsumexp(
                log_kernel + log_v.unsqueeze(0),
                dim=1,
            )
            log_v = log_column - torch.logsumexp(
                log_kernel + log_u.unsqueeze(1),
                dim=0,
            )
        return torch.exp(
            log_kernel
            + log_u.unsqueeze(1)
            + log_v.unsqueeze(0)
        )

    def _variance_loss(self, hsi_projection, lidar_projection):
        hsi_std = torch.sqrt(
            hsi_projection.var(dim=0, unbiased=False) + 1e-4
        )
        lidar_std = torch.sqrt(
            lidar_projection.var(dim=0, unbiased=False) + 1e-4
        )
        loss = 0.5 * (
            F.relu(self.variance_target - hsi_std).mean()
            + F.relu(self.variance_target - lidar_std).mean()
        )
        return loss, hsi_std, lidar_std

    def forward(self, hsi_features, lidar_features):
        hsi_projection = self.hsi_projector(hsi_features)
        lidar_projection = self.lidar_projector(lidar_features)
        hsi_shared = F.normalize(hsi_projection, dim=1)
        lidar_shared = F.normalize(lidar_projection, dim=1)
        semantic_similarity = (
            hsi_shared @ lidar_shared.transpose(0, 1)
        ) / self.temperature
        semantic_progress = self._semantic_progress()
        transport = self._sinkhorn(
            semantic_similarity.detach(),
            semantic_progress,
        )
        lidar_prototypes = F.normalize(
            (
                transport @ lidar_shared
            )
            / self.row_marginal.unsqueeze(1).clamp_min(1e-12),
            dim=1,
        )
        hsi_prototypes = F.normalize(
            (
                transport.transpose(0, 1) @ hsi_shared
            )
            / self.column_marginal.unsqueeze(1).clamp_min(1e-12),
            dim=1,
        )
        hsi_prediction = F.normalize(
            self.hsi_predictor(hsi_projection),
            dim=1,
        )
        lidar_prediction = F.normalize(
            self.lidar_predictor(lidar_projection),
            dim=1,
        )
        hsi_to_lidar_cosine = torch.sum(
            hsi_prediction * lidar_prototypes.detach(),
            dim=1,
        )
        lidar_to_hsi_cosine = torch.sum(
            lidar_prediction * hsi_prototypes.detach(),
            dim=1,
        )
        distillation_loss = 0.5 * (
            (1.0 - hsi_to_lidar_cosine).mean()
            + (1.0 - lidar_to_hsi_cosine).mean()
        )
        (
            variance_loss,
            hsi_std,
            lidar_std,
        ) = self._variance_loss(
            hsi_projection,
            lidar_projection,
        )
        row_error = torch.max(
            torch.abs(
                transport.sum(dim=1) - self.row_marginal
            )
        )
        column_error = torch.max(
            torch.abs(
                transport.sum(dim=0) - self.column_marginal
            )
        )
        transport_entropy = -torch.sum(
            transport
            * torch.log(transport.clamp_min(1e-12))
        )
        self.last_variance_loss = variance_loss
        self.last_transport = transport.detach()
        self.last_diagnostics = {
            "semantic_progress": float(semantic_progress),
            "hsi_projector_std": hsi_std.detach(),
            "lidar_projector_std": lidar_std.detach(),
            "hsi_to_lidar_cosine": (
                hsi_to_lidar_cosine.mean().detach()
            ),
            "lidar_to_hsi_cosine": (
                lidar_to_hsi_cosine.mean().detach()
            ),
            "mean_node_prototype_cosine": (
                0.5
                * (
                    hsi_to_lidar_cosine.mean()
                    + lidar_to_hsi_cosine.mean()
                )
            ).detach(),
            "sinkhorn_row_max_error": row_error.detach(),
            "sinkhorn_column_max_error": column_error.detach(),
            "transport_entropy": transport_entropy.detach(),
        }
        return distillation_loss

    def diagnostics(self):
        if self.last_diagnostics is None:
            return None
        output = {}
        for name, value in self.last_diagnostics.items():
            if torch.is_tensor(value):
                value = value.detach().cpu()
                output[name] = (
                    value.tolist()
                    if value.ndim > 0
                    else float(value.item())
                )
            else:
                output[name] = value
        return output


class PostGATPrototypeCorrelationFusion(nn.Module):
    """SPSN-inspired selected-prototype correlation and reliability fusion."""

    def __init__(
        self,
        channels,
        prototype_count,
        temperature,
        initial_hsi_weight=0.5,
    ):
        super().__init__()
        self.prototype_count = prototype_count
        self.temperature = temperature
        selector_hidden = max(channels // 2, 16)

        def make_selector():
            return nn.Sequential(
                nn.LayerNorm(channels),
                nn.Linear(channels, selector_hidden),
                nn.LeakyReLU(),
                nn.Linear(selector_hidden, 1),
            )

        self.hsi_selector = make_selector()
        self.lidar_selector = make_selector()
        self.hsi_correlation_adapter = nn.Linear(
            prototype_count,
            channels,
        )
        self.lidar_correlation_adapter = nn.Linear(
            prototype_count,
            channels,
        )
        reliability_hidden = max(prototype_count, 16)
        self.reliability_gate = nn.Sequential(
            nn.Linear(2 * prototype_count, reliability_hidden),
            nn.LeakyReLU(),
            nn.Linear(reliability_hidden, 2),
        )
        for adapter in (
            self.hsi_correlation_adapter,
            self.lidar_correlation_adapter,
        ):
            nn.init.normal_(adapter.weight, std=1e-3)
            nn.init.zeros_(adapter.bias)
        output_layer = self.reliability_gate[-1]
        nn.init.zeros_(output_layer.weight)
        clipped_hsi_weight = min(
            max(float(initial_hsi_weight), 1e-4),
            1.0 - 1e-4,
        )
        with torch.no_grad():
            output_layer.bias.copy_(
                torch.log(
                    torch.tensor(
                        [
                            clipped_hsi_weight,
                            1.0 - clipped_hsi_weight,
                        ],
                        dtype=output_layer.bias.dtype,
                    )
                )
            )
        self.last_diagnostics = None

    def _correlation_map(
        self,
        node_features,
        projection_assignment,
        selector,
    ):
        selection_scores = torch.sigmoid(
            selector(node_features).squeeze(-1)
        )
        selected_scores, selected_indices = torch.topk(
            selection_scores,
            k=self.prototype_count,
            dim=0,
            sorted=True,
        )
        selected_prototypes = node_features.index_select(
            0,
            selected_indices,
        )
        normalized_nodes = F.normalize(node_features, dim=1)
        normalized_prototypes = F.normalize(
            selected_prototypes,
            dim=1,
        )
        correlation_logits = (
            normalized_nodes @ normalized_prototypes.transpose(0, 1)
        ) / self.temperature
        correlation_logits = (
            correlation_logits
            + torch.log(selected_scores.clamp_min(1e-6)).unsqueeze(0)
        )
        node_correlation = F.softmax(
            correlation_logits,
            dim=1,
        )
        pixel_correlation = torch.sparse.mm(
            projection_assignment,
            node_correlation,
        )
        return (
            pixel_correlation,
            selected_scores,
            selected_indices,
        )

    def forward(
        self,
        hsi_nodes,
        lidar_nodes,
        hsi_pixel_features,
        lidar_pixel_features,
        hsi_projection_assignment,
        lidar_projection_assignment,
    ):
        (
            hsi_correlation,
            hsi_selected_scores,
            hsi_selected_indices,
        ) = self._correlation_map(
            hsi_nodes,
            hsi_projection_assignment,
            self.hsi_selector,
        )
        (
            lidar_correlation,
            lidar_selected_scores,
            lidar_selected_indices,
        ) = self._correlation_map(
            lidar_nodes,
            lidar_projection_assignment,
            self.lidar_selector,
        )
        hsi_enhanced = (
            hsi_pixel_features
            + self.hsi_correlation_adapter(hsi_correlation)
        )
        lidar_enhanced = (
            lidar_pixel_features
            + self.lidar_correlation_adapter(lidar_correlation)
        )
        reliability = F.softmax(
            self.reliability_gate(
                torch.cat(
                    [hsi_correlation, lidar_correlation],
                    dim=1,
                )
            ),
            dim=1,
        )
        fused_features = (
            reliability[:, :1] * hsi_enhanced
            + reliability[:, 1:] * lidar_enhanced
        )
        self.last_diagnostics = {
            "hsi_selected_indices": (
                hsi_selected_indices.detach().cpu().tolist()
            ),
            "lidar_selected_indices": (
                lidar_selected_indices.detach().cpu().tolist()
            ),
            "hsi_selected_score_mean": float(
                hsi_selected_scores.detach().mean().item()
            ),
            "lidar_selected_score_mean": float(
                lidar_selected_scores.detach().mean().item()
            ),
            "hsi_reliability_mean": float(
                reliability[:, 0].detach().mean().item()
            ),
            "lidar_reliability_mean": float(
                reliability[:, 1].detach().mean().item()
            ),
        }
        return fused_features

    def diagnostics(self):
        return self.last_diagnostics


class PostGATConsensusAnchorInteraction(nn.Module):
    """A3 shared anchors with optional SACR residual refinement."""

    def __init__(
        self,
        channels,
        anchor_count,
        temperature,
        hsi_area,
        lidar_area,
        gamma_init=0.0,
        fusion_mode="fixed",
        writeback_mode="direct",
        reliability_temperature=1.0,
        anchor_reasoning="none",
        structure_reliability="none",
        anchor_graph_topk=8,
        structure_temperature=0.1,
        structure_eta_init=0.0,
    ):
        super().__init__()
        self.temperature = temperature
        self.fusion_mode = fusion_mode
        self.writeback_mode = writeback_mode
        self.reliability_temperature = reliability_temperature
        self.anchor_reasoning = anchor_reasoning
        self.structure_reliability = structure_reliability
        self.anchor_graph_topk = anchor_graph_topk
        self.structure_temperature = structure_temperature
        self.empty_anchor_mass_threshold = 1.0
        self.consensus_anchors = nn.Parameter(
            torch.empty(anchor_count, channels)
        )
        nn.init.xavier_uniform_(self.consensus_anchors)
        self.hsi_key_projection = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        self.lidar_key_projection = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        self.hsi_value_projection = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        self.lidar_value_projection = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        nn.init.eye_(self.hsi_value_projection.weight)
        nn.init.eye_(self.lidar_value_projection.weight)
        self.structure_eta = nn.Parameter(
            torch.tensor(float(structure_eta_init))
        )
        self.hsi_gamma = nn.Parameter(
            torch.tensor(float(gamma_init))
        )
        self.lidar_gamma = nn.Parameter(
            torch.tensor(float(gamma_init))
        )
        self.register_buffer(
            "hsi_area",
            torch.as_tensor(hsi_area, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "lidar_area",
            torch.as_tensor(lidar_area, dtype=torch.float32),
            persistent=False,
        )
        self.last_diagnostics = None

    def _soft_assignment(self, nodes, projection):
        node_keys = F.normalize(projection(nodes), dim=1)
        anchor_keys = F.normalize(
            self.consensus_anchors,
            dim=1,
        )
        return F.softmax(
            (
                node_keys
                @ anchor_keys.transpose(0, 1)
            )
            / self.temperature,
            dim=1,
        )

    @staticmethod
    def _area_weighted_anchor_features(
        assignment,
        area,
        nodes,
    ):
        weighted_nodes = area.unsqueeze(1) * nodes
        numerator = assignment.transpose(0, 1) @ weighted_nodes
        denominator = (
            assignment.transpose(0, 1) @ area
        ).unsqueeze(1)
        return (
            numerator / denominator.clamp_min(1e-6),
            denominator.squeeze(1),
        )

    @staticmethod
    def _anchor_reconstruction_error(
        assignment,
        area,
        node_features,
        anchor_features,
        anchor_mass,
    ):
        node_norm = torch.sum(
            node_features * node_features,
            dim=1,
            keepdim=True,
        )
        anchor_norm = torch.sum(
            anchor_features * anchor_features,
            dim=1,
        ).unsqueeze(0)
        squared_distance = (
            node_norm
            + anchor_norm
            - 2.0
            * (
                node_features
                @ anchor_features.transpose(0, 1)
            )
        ).clamp_min(0.0)
        weighted_assignment = area.unsqueeze(1) * assignment
        return torch.sum(
            weighted_assignment * squared_distance,
            dim=0,
        ) / anchor_mass.clamp_min(1e-6)

    @staticmethod
    def _node_assignment_entropy(assignment):
        return -torch.sum(
            assignment * torch.log(assignment.clamp_min(1e-12)),
            dim=1,
            keepdim=True,
        )

    @staticmethod
    def _mean_assignment_entropy(assignment):
        return (
            PostGATConsensusAnchorInteraction
            ._node_assignment_entropy(assignment)
            .mean()
        )

    @staticmethod
    def _row_entropy(graph):
        return -torch.sum(
            graph * torch.log(graph.clamp_min(1e-12)),
            dim=1,
        )

    @staticmethod
    def _topk_row_softmax(logits, topk):
        if topk <= 0 or topk >= logits.shape[1]:
            return F.softmax(logits, dim=1)
        values, indices = torch.topk(
            logits,
            k=min(topk, logits.shape[1]),
            dim=1,
        )
        masked_logits = torch.full_like(
            logits,
            torch.finfo(logits.dtype).min,
        )
        masked_logits.scatter_(1, indices, values)
        return F.softmax(masked_logits, dim=1)

    def _anchor_graph_from_features(self, anchor_features):
        normalized = F.normalize(anchor_features, dim=1)
        logits = normalized @ normalized.transpose(0, 1)
        return self._topk_row_softmax(
            logits,
            self.anchor_graph_topk,
        )

    def _apply_sacr_refinement(
        self,
        consensus,
        hsi_anchors,
        lidar_anchors,
        reliability,
    ):
        hsi_anchor_graph = self._anchor_graph_from_features(
            hsi_anchors
        )
        lidar_anchor_graph = self._anchor_graph_from_features(
            lidar_anchors
        )
        graph_gap = torch.mean(
            torch.abs(hsi_anchor_graph - lidar_anchor_graph),
            dim=1,
            keepdim=True,
        )
        if self.structure_reliability == "adaptive":
            structure_gate = torch.exp(
                -graph_gap / self.structure_temperature
            )
        else:
            structure_gate = torch.ones_like(graph_gap)
        structure_residual = (
            reliability[:, :1]
            * (hsi_anchor_graph @ hsi_anchors - hsi_anchors)
            + reliability[:, 1:]
            * (lidar_anchor_graph @ lidar_anchors - lidar_anchors)
        )
        refined = (
            consensus
            + self.structure_eta
            * structure_gate
            * structure_residual
        )
        return (
            refined,
            hsi_anchor_graph,
            lidar_anchor_graph,
            graph_gap,
            structure_gate,
            structure_residual,
        )

    def forward(self, hsi_nodes, lidar_nodes):
        hsi_assignment = self._soft_assignment(
            hsi_nodes,
            self.hsi_key_projection,
        )
        lidar_assignment = self._soft_assignment(
            lidar_nodes,
            self.lidar_key_projection,
        )
        (
            hsi_anchor_features,
            hsi_anchor_mass,
        ) = self._area_weighted_anchor_features(
            hsi_assignment,
            self.hsi_area,
            hsi_nodes,
        )
        (
            lidar_anchor_features,
            lidar_anchor_mass,
        ) = self._area_weighted_anchor_features(
            lidar_assignment,
            self.lidar_area,
            lidar_nodes,
        )
        hsi_shared_nodes = self.hsi_value_projection(hsi_nodes)
        lidar_shared_nodes = self.lidar_value_projection(
            lidar_nodes
        )
        hsi_shared_anchors = self.hsi_value_projection(
            hsi_anchor_features
        )
        lidar_shared_anchors = self.lidar_value_projection(
            lidar_anchor_features
        )
        hsi_reliability_nodes = F.normalize(
            hsi_shared_nodes,
            dim=1,
        )
        lidar_reliability_nodes = F.normalize(
            lidar_shared_nodes,
            dim=1,
        )
        (
            hsi_reliability_anchors,
            _,
        ) = self._area_weighted_anchor_features(
            hsi_assignment,
            self.hsi_area,
            hsi_reliability_nodes,
        )
        (
            lidar_reliability_anchors,
            _,
        ) = self._area_weighted_anchor_features(
            lidar_assignment,
            self.lidar_area,
            lidar_reliability_nodes,
        )
        hsi_error = self._anchor_reconstruction_error(
            hsi_assignment,
            self.hsi_area,
            hsi_reliability_nodes,
            hsi_reliability_anchors,
            hsi_anchor_mass,
        ).detach()
        lidar_error = self._anchor_reconstruction_error(
            lidar_assignment,
            self.lidar_area,
            lidar_reliability_nodes,
            lidar_reliability_anchors,
            lidar_anchor_mass,
        ).detach()
        if self.fusion_mode == "fixed":
            reliability = torch.full(
                (
                    hsi_shared_anchors.shape[0],
                    2,
                ),
                0.5,
                dtype=hsi_nodes.dtype,
                device=hsi_nodes.device,
            )
        else:
            reliability = F.softmax(
                -torch.stack(
                    [hsi_error, lidar_error],
                    dim=1,
                )
                / self.reliability_temperature,
                dim=1,
            )
        consensus = (
            reliability[:, :1] * hsi_shared_anchors
            + reliability[:, 1:] * lidar_shared_anchors
        )
        hsi_anchor_graph = None
        lidar_anchor_graph = None
        graph_gap = None
        structure_gate = None
        structure_residual = None
        if self.anchor_reasoning == "sacr":
            (
                consensus,
                hsi_anchor_graph,
                lidar_anchor_graph,
                graph_gap,
                structure_gate,
                structure_residual,
            ) = self._apply_sacr_refinement(
                consensus,
                hsi_shared_anchors,
                lidar_shared_anchors,
                reliability,
            )
        if self.writeback_mode == "difference":
            hsi_anchor_message = (
                consensus - hsi_shared_anchors
            )
            lidar_anchor_message = (
                consensus - lidar_shared_anchors
            )
        else:
            hsi_anchor_message = consensus
            lidar_anchor_message = consensus
        hsi_message = hsi_assignment @ hsi_anchor_message
        lidar_message = (
            lidar_assignment @ lidar_anchor_message
        )
        updated_hsi = (
            hsi_nodes + self.hsi_gamma * hsi_message
        )
        updated_lidar = (
            lidar_nodes + self.lidar_gamma * lidar_message
        )
        reliability_entropy = -torch.sum(
            reliability
            * torch.log(reliability.clamp_min(1e-12)),
            dim=1,
        )
        diagnostics = {
            "fusion_mode": self.fusion_mode,
            "writeback_mode": self.writeback_mode,
            "anchor_reasoning": self.anchor_reasoning,
            "structure_reliability": self.structure_reliability,
            "anchor_graph_topk": self.anchor_graph_topk,
            "structure_temperature": self.structure_temperature,
            "structure_eta": float(
                self.structure_eta.detach().item()
            ),
            "hsi_gamma": float(
                self.hsi_gamma.detach().item()
            ),
            "lidar_gamma": float(
                self.lidar_gamma.detach().item()
            ),
            "hsi_assignment_entropy": float(
                self._mean_assignment_entropy(
                    hsi_assignment
                ).detach().item()
            ),
            "lidar_assignment_entropy": float(
                self._mean_assignment_entropy(
                    lidar_assignment
                ).detach().item()
            ),
            "hsi_anchor_reliability": (
                reliability[:, 0].detach().cpu().tolist()
            ),
            "lidar_anchor_reliability": (
                reliability[:, 1].detach().cpu().tolist()
            ),
            "reliability_entropy": (
                reliability_entropy.detach().cpu().tolist()
            ),
            "mean_reliability_entropy": float(
                reliability_entropy.detach().mean().item()
            ),
            "hsi_reconstruction_error": (
                hsi_error.cpu().tolist()
            ),
            "lidar_reconstruction_error": (
                lidar_error.cpu().tolist()
            ),
            "hsi_anchor_mass": (
                hsi_anchor_mass.detach().cpu().tolist()
            ),
            "lidar_anchor_mass": (
                lidar_anchor_mass.detach().cpu().tolist()
            ),
            "hsi_anchor_mass_min": float(
                hsi_anchor_mass.detach().min().item()
            ),
            "lidar_anchor_mass_min": float(
                lidar_anchor_mass.detach().min().item()
            ),
            "empty_anchor_mass_threshold": (
                self.empty_anchor_mass_threshold
            ),
            "hsi_empty_anchor_count": int(
                (
                    hsi_anchor_mass
                    < self.empty_anchor_mass_threshold
                ).detach().sum().item()
            ),
            "lidar_empty_anchor_count": int(
                (
                    lidar_anchor_mass
                    < self.empty_anchor_mass_threshold
                ).detach().sum().item()
            ),
            "hsi_message_norm": float(
                hsi_message.detach().norm(dim=1).mean().item()
            ),
            "lidar_message_norm": float(
                lidar_message.detach().norm(dim=1).mean().item()
            ),
        }
        if hsi_anchor_graph is not None:
            diagnostics.update(
                {
                    "hsi_anchor_graph_entropy": float(
                        self._row_entropy(hsi_anchor_graph)
                        .detach()
                        .mean()
                        .item()
                    ),
                    "lidar_anchor_graph_entropy": float(
                        self._row_entropy(lidar_anchor_graph)
                        .detach()
                        .mean()
                        .item()
                    ),
                    "anchor_graph_l1_gap": float(
                        graph_gap.detach().mean().item()
                    ),
                    "anchor_graph_l1_gap_per_anchor": (
                        graph_gap.squeeze(1)
                        .detach()
                        .cpu()
                        .tolist()
                    ),
                    "structure_gate_mean": float(
                        structure_gate.detach().mean().item()
                    ),
                    "structure_gate_min": float(
                        structure_gate.detach().min().item()
                    ),
                    "structure_gate_max": float(
                        structure_gate.detach().max().item()
                    ),
                    "structure_residual_norm": float(
                        structure_residual.detach()
                        .norm(dim=1)
                        .mean()
                        .item()
                    ),
                }
            )
        self.last_diagnostics = diagnostics
        return updated_hsi, updated_lidar

    def diagnostics(self):
        return self.last_diagnostics


class CenterBridgeBlockInteraction(nn.Module):
    """Post-GAT2 H<->C<->L block relation through public bridge anchors."""

    def __init__(
        self,
        channels,
        bridge_data,
        attention_d_k=32,
        topk=8,
        gamma_init=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.attention_d_k = attention_d_k
        self.topk = topk
        anchor_count = int(bridge_data["anchor_count"])
        self.bridge_embedding = nn.Parameter(
            torch.empty(anchor_count, channels)
        )
        nn.init.xavier_uniform_(self.bridge_embedding)
        self.bridge_norm = nn.LayerNorm(channels)

        self.h_query = nn.Linear(channels, attention_d_k, bias=False)
        self.h_key = nn.Linear(channels, attention_d_k, bias=False)
        self.h_value = nn.Linear(channels, channels, bias=False)
        self.c_query = nn.Linear(channels, attention_d_k, bias=False)
        self.c_key = nn.Linear(channels, attention_d_k, bias=False)
        self.c_value = nn.Linear(channels, channels, bias=False)
        self.l_query = nn.Linear(channels, attention_d_k, bias=False)
        self.l_key = nn.Linear(channels, attention_d_k, bias=False)
        self.l_value = nn.Linear(channels, channels, bias=False)
        self.scale = attention_d_k ** -0.5

        self.h_view_score = nn.Linear(channels, 1)
        self.c_view_score = nn.Linear(channels, 1)
        self.l_view_score = nn.Linear(channels, 1)
        self.h_ffn = nn.Linear(channels, channels)
        self.c_ffn = nn.Linear(channels, channels)
        self.l_ffn = nn.Linear(channels, channels)
        self.h_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.c_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.l_gamma = nn.Parameter(torch.tensor(float(gamma_init)))

        for name in (
            "prior_hc",
            "prior_ch",
            "prior_lc",
            "prior_cl",
            "bias_hc",
            "bias_ch",
            "bias_lc",
            "bias_cl",
            "bias_cc",
        ):
            self.register_buffer(
                name,
                torch.as_tensor(
                    bridge_data[name],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        self.last_diagnostics = None

    @staticmethod
    def _topk_softmax(logits, topk):
        if topk <= 0 or topk >= logits.shape[1]:
            return F.softmax(logits, dim=1)
        values, indices = torch.topk(
            logits,
            k=min(topk, logits.shape[1]),
            dim=1,
        )
        masked = torch.full_like(
            logits,
            torch.finfo(logits.dtype).min,
        )
        masked.scatter_(1, indices, values)
        return F.softmax(masked, dim=1)

    @staticmethod
    def _row_entropy(attention):
        return -torch.sum(
            attention * torch.log(attention.clamp_min(1e-12)),
            dim=1,
        )

    @staticmethod
    def _row_normalize(matrix):
        return matrix / matrix.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

    def _attention(self, query, key, bias):
        logits = query @ key.transpose(0, 1) * self.scale + bias
        return self._topk_softmax(logits, self.topk)

    @staticmethod
    def _fuse_views(views, scorer):
        scores = scorer(views).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * views, dim=1)
        return fused, weights

    def forward(self, hsi_nodes, lidar_nodes):
        bridge_from_hsi = self.prior_ch @ hsi_nodes
        bridge_from_lidar = self.prior_cl @ lidar_nodes
        bridge_nodes = self.bridge_norm(
            0.5 * bridge_from_hsi
            + 0.5 * bridge_from_lidar
            + self.bridge_embedding
        )

        h_query = self.h_query(hsi_nodes)
        h_key = self.h_key(hsi_nodes)
        h_value = self.h_value(hsi_nodes)
        c_query = self.c_query(bridge_nodes)
        c_key = self.c_key(bridge_nodes)
        c_value = self.c_value(bridge_nodes)
        l_query = self.l_query(lidar_nodes)
        l_key = self.l_key(lidar_nodes)
        l_value = self.l_value(lidar_nodes)

        attention_hc = self._attention(
            h_query,
            c_key,
            self.bias_hc,
        )
        attention_ch = self._attention(
            c_query,
            h_key,
            self.bias_ch,
        )
        attention_lc = self._attention(
            l_query,
            c_key,
            self.bias_lc,
        )
        attention_cl = self._attention(
            c_query,
            l_key,
            self.bias_cl,
        )
        attention_cc = self._attention(
            c_query,
            c_key,
            self.bias_cc,
        )

        attention_hl_via_c = self._row_normalize(
            attention_hc @ attention_cl
        )
        attention_lh_via_c = self._row_normalize(
            attention_lc @ attention_ch
        )
        bridge_views = torch.stack(
            [
                attention_ch @ h_value,
                attention_cc @ c_value,
                attention_cl @ l_value,
            ],
            dim=1,
        )
        bridge_message, bridge_view_weights = self._fuse_views(
            bridge_views,
            self.c_view_score,
        )
        updated_bridge = (
            bridge_nodes + self.c_gamma * self.c_ffn(bridge_message)
        )
        updated_bridge_value = self.c_value(updated_bridge)

        hsi_views = torch.stack(
            [
                h_value,
                attention_hc @ updated_bridge_value,
                attention_hl_via_c @ l_value,
            ],
            dim=1,
        )
        lidar_views = torch.stack(
            [
                attention_lh_via_c @ h_value,
                attention_lc @ updated_bridge_value,
                l_value,
            ],
            dim=1,
        )
        hsi_message, hsi_view_weights = self._fuse_views(
            hsi_views,
            self.h_view_score,
        )
        lidar_message, lidar_view_weights = self._fuse_views(
            lidar_views,
            self.l_view_score,
        )

        updated_hsi = (
            hsi_nodes + self.h_gamma * self.h_ffn(hsi_message)
        )
        updated_lidar = (
            lidar_nodes + self.l_gamma * self.l_ffn(lidar_message)
        )
        self.last_diagnostics = {
            "h_gamma": float(self.h_gamma.detach().item()),
            "c_gamma": float(self.c_gamma.detach().item()),
            "l_gamma": float(self.l_gamma.detach().item()),
            "bridge_count": int(bridge_nodes.shape[0]),
            "attention_hc_entropy": float(
                self._row_entropy(attention_hc).detach().mean().item()
            ),
            "attention_ch_entropy": float(
                self._row_entropy(attention_ch).detach().mean().item()
            ),
            "attention_lc_entropy": float(
                self._row_entropy(attention_lc).detach().mean().item()
            ),
            "attention_cl_entropy": float(
                self._row_entropy(attention_cl).detach().mean().item()
            ),
            "attention_cc_entropy": float(
                self._row_entropy(attention_cc).detach().mean().item()
            ),
            "attention_hl_via_c_entropy": float(
                self._row_entropy(attention_hl_via_c)
                .detach()
                .mean()
                .item()
            ),
            "attention_lh_via_c_entropy": float(
                self._row_entropy(attention_lh_via_c)
                .detach()
                .mean()
                .item()
            ),
            "hsi_view_weight_mean": (
                hsi_view_weights.detach().mean(dim=0).cpu().tolist()
            ),
            "bridge_view_weight_mean": (
                bridge_view_weights.detach().mean(dim=0).cpu().tolist()
            ),
            "lidar_view_weight_mean": (
                lidar_view_weights.detach().mean(dim=0).cpu().tolist()
            ),
            "hsi_message_norm": float(
                hsi_message.detach().norm(dim=1).mean().item()
            ),
            "bridge_message_norm": float(
                bridge_message.detach().norm(dim=1).mean().item()
            ),
            "lidar_message_norm": float(
                lidar_message.detach().norm(dim=1).mean().item()
            ),
        }
        return updated_hsi, updated_lidar

    def diagnostics(self):
        return self.last_diagnostics


class PostGATMediatedConsensusGraph(nn.Module):
    """Independent mediator graph branch over public center anchors.

    The module consumes post-GAT2 HSI/LiDAR private graph nodes, builds
    public center anchors from both modalities, reasons only on those
    mediator anchors, and projects the updated mediator graph directly
    back to pixels. It intentionally does not write messages back to the
    HSI or LiDAR superpixel nodes.
    """

    def __init__(
        self,
        channels,
        bridge_data,
        attention_d_k=32,
        topk=8,
        gamma_init=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.attention_d_k = attention_d_k
        self.topk = topk
        anchor_count = int(bridge_data["anchor_count"])
        self.bridge_embedding = nn.Parameter(
            torch.empty(anchor_count, channels)
        )
        nn.init.xavier_uniform_(self.bridge_embedding)
        self.bridge_norm = nn.LayerNorm(channels)

        self.h_key = nn.Linear(channels, attention_d_k, bias=False)
        self.h_value = nn.Linear(channels, channels, bias=False)
        self.c_query = nn.Linear(channels, attention_d_k, bias=False)
        self.c_key = nn.Linear(channels, attention_d_k, bias=False)
        self.c_value = nn.Linear(channels, channels, bias=False)
        self.l_key = nn.Linear(channels, attention_d_k, bias=False)
        self.l_value = nn.Linear(channels, channels, bias=False)
        self.scale = attention_d_k ** -0.5

        self.c_view_score = nn.Linear(channels, 1)
        self.c_ffn = nn.Linear(channels, channels)
        self.c_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.graph_projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(),
        )

        for name in (
            "prior_ch",
            "prior_cl",
            "bias_ch",
            "bias_cl",
            "bias_cc",
        ):
            self.register_buffer(
                name,
                torch.as_tensor(
                    bridge_data[name],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        _, bridge_projection_assignment = normalized_sparse_assignments(
            bridge_data["assignment"]
        )
        self.register_buffer(
            "bridge_projection_assignment",
            bridge_projection_assignment,
            persistent=False,
        )
        self.last_diagnostics = None

    @staticmethod
    def _topk_softmax(logits, topk):
        if topk <= 0 or topk >= logits.shape[1]:
            return F.softmax(logits, dim=1)
        values, indices = torch.topk(
            logits,
            k=min(topk, logits.shape[1]),
            dim=1,
        )
        masked = torch.full_like(
            logits,
            torch.finfo(logits.dtype).min,
        )
        masked.scatter_(1, indices, values)
        return F.softmax(masked, dim=1)

    @staticmethod
    def _row_entropy(attention):
        return -torch.sum(
            attention * torch.log(attention.clamp_min(1e-12)),
            dim=1,
        )

    def _attention(self, query, key, bias):
        logits = query @ key.transpose(0, 1) * self.scale + bias
        return self._topk_softmax(logits, self.topk)

    @staticmethod
    def _fuse_views(views, scorer):
        scores = scorer(views).squeeze(-1)
        weights = F.softmax(scores, dim=1)
        fused = torch.sum(weights.unsqueeze(-1) * views, dim=1)
        return fused, weights

    def forward(self, hsi_nodes, lidar_nodes):
        h_key = self.h_key(hsi_nodes)
        h_value = self.h_value(hsi_nodes)
        l_key = self.l_key(lidar_nodes)
        l_value = self.l_value(lidar_nodes)

        bridge_nodes = self.bridge_norm(
            0.5 * (self.prior_ch @ h_value)
            + 0.5 * (self.prior_cl @ l_value)
            + self.bridge_embedding
        )
        c_query = self.c_query(bridge_nodes)
        c_key = self.c_key(bridge_nodes)
        c_value = self.c_value(bridge_nodes)

        attention_ch = self._attention(
            c_query,
            h_key,
            self.bias_ch,
        )
        attention_cc = self._attention(
            c_query,
            c_key,
            self.bias_cc,
        )
        attention_cl = self._attention(
            c_query,
            l_key,
            self.bias_cl,
        )
        bridge_views = torch.stack(
            [
                attention_ch @ h_value,
                attention_cc @ c_value,
                attention_cl @ l_value,
            ],
            dim=1,
        )
        bridge_message, bridge_view_weights = self._fuse_views(
            bridge_views,
            self.c_view_score,
        )
        updated_bridge = (
            bridge_nodes + self.c_gamma * self.c_ffn(bridge_message)
        )
        consensus_pixel_features = torch.sparse.mm(
            self.bridge_projection_assignment,
            updated_bridge,
        )
        consensus_pixel_features = self.graph_projection(
            consensus_pixel_features
        )
        self.last_diagnostics = {
            "c_gamma": float(self.c_gamma.detach().item()),
            "bridge_count": int(bridge_nodes.shape[0]),
            "attention_ch_entropy": float(
                self._row_entropy(attention_ch).detach().mean().item()
            ),
            "attention_cc_entropy": float(
                self._row_entropy(attention_cc).detach().mean().item()
            ),
            "attention_cl_entropy": float(
                self._row_entropy(attention_cl).detach().mean().item()
            ),
            "bridge_view_weight_mean": (
                bridge_view_weights.detach().mean(dim=0).cpu().tolist()
            ),
            "bridge_message_norm": float(
                bridge_message.detach().norm(dim=1).mean().item()
            ),
            "consensus_pixel_norm": float(
                consensus_pixel_features.detach()
                .norm(dim=1)
                .mean()
                .item()
            ),
        }
        return consensus_pixel_features

    def diagnostics(self):
        return self.last_diagnostics


def aggregate_cell_pixel_means(
    pixel_features,
    pixel_cell_index,
    cell_count,
):
    """Average a pixel feature cube inside every intersection cell."""
    flat_features = np.asarray(
        pixel_features,
        dtype=np.float32,
    ).reshape(-1, pixel_features.shape[-1])
    pixel_cell_index = np.asarray(
        pixel_cell_index,
        dtype=np.int64,
    ).reshape(-1)
    sums = np.zeros(
        (cell_count, flat_features.shape[1]),
        dtype=np.float32,
    )
    np.add.at(sums, pixel_cell_index, flat_features)
    counts = np.bincount(
        pixel_cell_index,
        minlength=cell_count,
    ).astype(np.float32)
    return sums / np.maximum(counts[:, None], 1.0)


def attach_cell_pixel_descriptors(
    cell_data,
    hsi_features,
    lidar_features,
):
    """Attach standardized mean HSI/LiDAR pixels to cell nodes."""
    cell_count = cell_data["cell_count"]
    pixel_cell_index = cell_data["pixel_cell_index"]
    hsi_cell_mean = aggregate_cell_pixel_means(
        hsi_features,
        pixel_cell_index,
        cell_count,
    )
    lidar_cell_mean = aggregate_cell_pixel_means(
        lidar_features,
        pixel_cell_index,
        cell_count,
    )
    raw_descriptor = np.concatenate(
        [hsi_cell_mean, lidar_cell_mean],
        axis=1,
    ).astype(np.float32)
    descriptor_mean = raw_descriptor.mean(axis=0, keepdims=True)
    descriptor_std = raw_descriptor.std(axis=0, keepdims=True)
    cell_data["pixel_descriptors"] = (
        (raw_descriptor - descriptor_mean)
        / np.maximum(descriptor_std, 1e-6)
    ).astype(np.float32)
    cell_data["raw_pixel_descriptors"] = raw_descriptor


def build_sparse_cell_boundary_strengths(
    cell_segments,
    hsi,
    elevation,
):
    """Return average HSI spectral and LiDAR gradient boundary strengths."""
    cell_count = int(cell_segments.max()) + 1
    hsi = np.asarray(hsi, dtype=np.float32)
    gradient_y, gradient_x = np.gradient(
        np.asarray(elevation, dtype=np.float32)
    )
    gradient = np.sqrt(
        gradient_x * gradient_x + gradient_y * gradient_y
    )
    key_parts = []
    hsi_value_parts = []
    lidar_value_parts = []
    for (
        first,
        second,
        first_spectrum,
        second_spectrum,
        first_gradient,
        second_gradient,
    ) in (
        (
            cell_segments[:, :-1],
            cell_segments[:, 1:],
            hsi[:, :-1, :],
            hsi[:, 1:, :],
            gradient[:, :-1],
            gradient[:, 1:],
        ),
        (
            cell_segments[:-1, :],
            cell_segments[1:, :],
            hsi[:-1, :, :],
            hsi[1:, :, :],
            gradient[:-1, :],
            gradient[1:, :],
        ),
    ):
        boundary = first != second
        if not np.any(boundary):
            continue
        left = first[boundary].astype(np.int64)
        right = second[boundary].astype(np.int64)
        lower = np.minimum(left, right)
        upper = np.maximum(left, right)
        key_parts.append(lower * cell_count + upper)
        boundary_first_spectrum = first_spectrum[boundary]
        boundary_second_spectrum = second_spectrum[boundary]
        numerator = np.sum(
            boundary_first_spectrum * boundary_second_spectrum,
            axis=1,
        )
        denominator = (
            np.linalg.norm(boundary_first_spectrum, axis=1)
            * np.linalg.norm(boundary_second_spectrum, axis=1)
        )
        hsi_value_parts.append(
            np.arccos(
                np.clip(
                    numerator / np.maximum(denominator, 1e-8),
                    -1.0,
                    1.0,
                )
            ).astype(np.float32)
        )
        lidar_value_parts.append(
            0.5
            * (
                first_gradient[boundary]
                + second_gradient[boundary]
            )
        )
    if not key_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )
    keys = np.concatenate(key_parts)
    hsi_values = np.concatenate(hsi_value_parts).astype(np.float32)
    lidar_values = np.concatenate(lidar_value_parts).astype(
        np.float32
    )
    unique_keys, inverse = np.unique(keys, return_inverse=True)
    hsi_boundary_sum = np.zeros(unique_keys.size, dtype=np.float32)
    lidar_boundary_sum = np.zeros(unique_keys.size, dtype=np.float32)
    boundary_count = np.zeros(unique_keys.size, dtype=np.float32)
    np.add.at(hsi_boundary_sum, inverse, hsi_values)
    np.add.at(lidar_boundary_sum, inverse, lidar_values)
    np.add.at(boundary_count, inverse, 1.0)
    boundary_count = np.maximum(boundary_count, 1.0)
    return (
        unique_keys,
        hsi_boundary_sum / boundary_count,
        lidar_boundary_sum / boundary_count,
    )


def build_multimodal_weighted_cell_rag(
    cell_data,
    hsi,
    lidar,
    height,
    width,
    sam_weight=1.0,
    height_weight=1.0,
    boundary_weight=1.0,
    conflict_weight=0.0,
):
    """Weight cell RAG by spectrum, height, boundary, and conflict."""
    cell_count = cell_data["cell_count"]
    pixel_cell_index = cell_data["pixel_cell_index"]
    cell_segments = pixel_cell_index.reshape(height, width)
    hsi_cell_mean = aggregate_cell_pixel_means(
        hsi,
        pixel_cell_index,
        cell_count,
    )
    lidar_cell_mean = aggregate_cell_pixel_means(
        lidar[:, :, None],
        pixel_cell_index,
        cell_count,
    )[:, 0]

    binary_rag = cell_data["cell_rag_adjacency"].tocoo()
    rows = binary_rag.row.astype(np.int64)
    columns = binary_rag.col.astype(np.int64)
    nonself = rows != columns
    edge_rows = rows[nonself]
    edge_columns = columns[nonself]

    numerator = np.sum(
        hsi_cell_mean[edge_rows] * hsi_cell_mean[edge_columns],
        axis=1,
    )
    denominator = (
        np.linalg.norm(hsi_cell_mean[edge_rows], axis=1)
        * np.linalg.norm(hsi_cell_mean[edge_columns], axis=1)
    )
    spectral_angle = np.arccos(
        np.clip(
            numerator / np.maximum(denominator, 1e-8),
            -1.0,
            1.0,
        )
    ).astype(np.float32)
    height_difference = np.abs(
        lidar_cell_mean[edge_rows]
        - lidar_cell_mean[edge_columns]
    ).astype(np.float32)

    (
        boundary_keys,
        hsi_boundary_values,
        lidar_boundary_values,
    ) = (
        build_sparse_cell_boundary_strengths(
            cell_segments,
            hsi,
            lidar,
        )
    )
    lower = np.minimum(edge_rows, edge_columns)
    upper = np.maximum(edge_rows, edge_columns)
    edge_keys = lower * cell_count + upper
    boundary_positions = np.searchsorted(
        boundary_keys,
        edge_keys,
    )
    if (
        boundary_keys.size == 0
        or np.any(boundary_positions >= boundary_keys.size)
        or not np.array_equal(
            boundary_keys[boundary_positions],
            edge_keys,
        )
    ):
        raise RuntimeError(
            "Cell RAG edge does not match a shared pixel boundary."
        )
    hsi_boundary_strength = hsi_boundary_values[
        boundary_positions
    ].astype(np.float32)
    lidar_boundary_strength = lidar_boundary_values[
        boundary_positions
    ].astype(np.float32)

    def scaled(values):
        positive = values[values > 0]
        bandwidth = (
            float(np.median(positive))
            if positive.size
            else 1.0
        )
        return values / max(bandwidth, 1e-6)

    scaled_hsi_boundary = scaled(hsi_boundary_strength)
    scaled_lidar_boundary = scaled(lidar_boundary_strength)
    boundary_conflict = np.abs(
        scaled_hsi_boundary - scaled_lidar_boundary
    )
    edge_weights = np.exp(
        -sam_weight * scaled(spectral_angle)
        -height_weight * scaled(height_difference)
        -boundary_weight * scaled_lidar_boundary
        -conflict_weight * boundary_conflict
    ).astype(np.float32)
    values = np.ones(rows.size, dtype=np.float32)
    values[nonself] = edge_weights
    weighted_rag = coo_matrix(
        (values, (rows, columns)),
        shape=(cell_count, cell_count),
        dtype=np.float32,
    ).tocsr()
    cell_data["cell_weighted_adjacency"] = (
        symmetrically_normalize_sparse_adjacency(weighted_rag)
    )
    if edge_weights.size:
        cell_data["cell_weight_stats"] = {
            "minimum": float(edge_weights.min()),
            "mean": float(edge_weights.mean()),
            "maximum": float(edge_weights.max()),
        }
    else:
        cell_data["cell_weight_stats"] = {
            "minimum": 1.0,
            "mean": 1.0,
            "maximum": 1.0,
        }
    cell_data["cell_boundary_conflict_stats"] = {
        "minimum": (
            float(boundary_conflict.min())
            if boundary_conflict.size
            else 0.0
        ),
        "mean": (
            float(boundary_conflict.mean())
            if boundary_conflict.size
            else 0.0
        ),
        "maximum": (
            float(boundary_conflict.max())
            if boundary_conflict.size
            else 0.0
        ),
    }


def build_parent_topology_support(
    cell_data,
    parent_key,
    coverage_key,
    adjacency_key,
):
    """Map A_cell to a symmetric [0, 1] parent-edge support matrix."""
    parent_index = np.asarray(
        cell_data[parent_key],
        dtype=np.int64,
    )
    coverage = np.asarray(
        cell_data[coverage_key],
        dtype=np.float32,
    )
    cell_count = cell_data["cell_count"]
    parent_count = int(parent_index.max()) + 1
    cell_indices = np.arange(cell_count, dtype=np.int64)
    cell_from_parent = coo_matrix(
        (
            np.ones(cell_count, dtype=np.float32),
            (cell_indices, parent_index),
        ),
        shape=(cell_count, parent_count),
        dtype=np.float32,
    ).tocsr()
    parent_from_cell = coo_matrix(
        (
            coverage,
            (parent_index, cell_indices),
        ),
        shape=(parent_count, cell_count),
        dtype=np.float32,
    ).tocsr()
    support = (
        parent_from_cell
        @ cell_data[adjacency_key]
        @ cell_from_parent
    ).tocsr()
    row_maximum = np.asarray(
        support.max(axis=1).toarray()
    ).reshape(-1)
    support = support.tocoo()
    support.data /= np.maximum(
        row_maximum[support.row],
        1e-6,
    )
    support = support.tocsr().maximum(
        support.transpose().tocsr()
    )
    support.setdiag(1.0)
    support.eliminate_zeros()
    return np.clip(
        support.toarray().astype(np.float32),
        0.0,
        1.0,
    )


def attach_cell_parent_topology_supports(
    cell_data,
    edge_mode,
):
    """Attach cell-derived support matrices for both modality graphs."""
    adjacency_key = (
        "cell_weighted_adjacency"
        if edge_mode == "spectral-height-boundary"
        else "cell_rag_adjacency"
    )
    cell_data["hsi_topology_support"] = (
        build_parent_topology_support(
            cell_data,
            "hsi_parent",
            "hsi_coverage",
            adjacency_key,
        )
    )
    cell_data["lidar_topology_support"] = (
        build_parent_topology_support(
            cell_data,
            "lidar_parent",
            "lidar_coverage",
            adjacency_key,
        )
    )


class IntersectionCellRAGLayer(nn.Module):
    """Parent-cell-parent exchange with optional pixel descriptors."""

    def __init__(
        self,
        channels,
        cell_data,
        use_pixel_descriptors=False,
        edge_mode="binary",
    ):
        super().__init__()
        self.hsi_node_count = int(
            np.max(cell_data["hsi_parent"])
        ) + 1
        self.lidar_node_count = int(
            np.max(cell_data["lidar_parent"])
        ) + 1
        for name, value, dtype in (
            ("hsi_parent", cell_data["hsi_parent"], torch.long),
            ("lidar_parent", cell_data["lidar_parent"], torch.long),
            (
                "hsi_coverage",
                cell_data["hsi_coverage"],
                torch.float32,
            ),
            (
                "lidar_coverage",
                cell_data["lidar_coverage"],
                torch.float32,
            ),
            (
                "cell_attributes",
                cell_data["attributes"],
                torch.float32,
            ),
        ):
            self.register_buffer(
                name,
                torch.as_tensor(value, dtype=dtype),
                persistent=False,
            )
        if use_pixel_descriptors:
            if "pixel_descriptors" not in cell_data:
                raise ValueError(
                    "Cell pixel descriptors were not prepared."
                )
            self.register_buffer(
                "pixel_descriptors",
                torch.as_tensor(
                    cell_data["pixel_descriptors"],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            descriptor_channels = cell_data[
                "pixel_descriptors"
            ].shape[1]
        else:
            self.pixel_descriptors = None
            descriptor_channels = 0

        adjacency_key = (
            "cell_rag_adjacency"
            if edge_mode == "binary"
            else "cell_weighted_adjacency"
        )
        if adjacency_key not in cell_data:
            raise ValueError(
                f"Missing {adjacency_key} for cell edge mode."
            )
        self.register_buffer(
            "cell_adjacency",
            scipy_sparse_to_torch(cell_data[adjacency_key]),
            persistent=False,
        )
        attribute_channels = cell_data["attributes"].shape[1]
        self.cell_encoder = nn.Sequential(
            nn.Linear(
                2 * channels
                + attribute_channels
                + descriptor_channels,
                channels,
            ),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
        )
        self.cell_graph_projection = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        self.cell_graph_norm = nn.LayerNorm(channels)
        self.hsi_gate = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.Sigmoid(),
        )
        self.lidar_gate = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.Sigmoid(),
        )
        self.hsi_norm = nn.LayerNorm(channels)
        self.lidar_norm = nn.LayerNorm(channels)
        self.last_cell_features = None
        self.last_hsi_gate = None
        self.last_lidar_gate = None

    def forward(
        self,
        hsi_nodes,
        lidar_nodes,
        hsi_intra,
        lidar_intra,
    ):
        if hsi_nodes.shape[0] != self.hsi_node_count:
            raise ValueError("Unexpected number of HSI parent nodes.")
        if lidar_nodes.shape[0] != self.lidar_node_count:
            raise ValueError("Unexpected number of LiDAR parent nodes.")
        cell_inputs = [
            hsi_intra.index_select(0, self.hsi_parent),
            lidar_intra.index_select(0, self.lidar_parent),
            self.cell_attributes,
        ]
        if self.pixel_descriptors is not None:
            cell_inputs.append(self.pixel_descriptors)
        cell_features = self.cell_encoder(
            torch.cat(cell_inputs, dim=-1)
        )
        cell_graph_message = torch.sparse.mm(
            self.cell_adjacency,
            self.cell_graph_projection(cell_features),
        )
        cell_features = self.cell_graph_norm(
            cell_features + F.leaky_relu(cell_graph_message)
        )

        hsi_message = torch.zeros_like(hsi_nodes)
        hsi_message.index_add_(
            0,
            self.hsi_parent,
            cell_features * self.hsi_coverage.unsqueeze(1),
        )
        lidar_message = torch.zeros_like(lidar_nodes)
        lidar_message.index_add_(
            0,
            self.lidar_parent,
            cell_features * self.lidar_coverage.unsqueeze(1),
        )
        hsi_gate = self.hsi_gate(
            torch.cat(
                [hsi_nodes, hsi_intra, hsi_message],
                dim=-1,
            )
        )
        lidar_gate = self.lidar_gate(
            torch.cat(
                [lidar_nodes, lidar_intra, lidar_message],
                dim=-1,
            )
        )
        self.last_cell_features = cell_features.detach()
        self.last_hsi_gate = hsi_gate.detach()
        self.last_lidar_gate = lidar_gate.detach()
        return (
            self.hsi_norm(
                hsi_nodes + hsi_intra + hsi_gate * hsi_message
            ),
            self.lidar_norm(
                lidar_nodes
                + lidar_intra
                + lidar_gate * lidar_message
            ),
            cell_features,
        )


class OverlapCrossModalInteraction(nn.Module):
    """Bidirectional gated exchange over spatially overlapping regions."""

    def __init__(
        self,
        channels,
        hsi_to_lidar,
        lidar_to_hsi,
        mode="overlap-gate",
        attention_d_k=16,
    ):
        super().__init__()
        if mode not in {"overlap-gate", "overlap-attention"}:
            raise ValueError(f"Unsupported cross-modal mode: {mode}")
        self.mode = mode
        self.register_buffer(
            "hsi_to_lidar",
            torch.as_tensor(hsi_to_lidar, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "lidar_to_hsi",
            torch.as_tensor(lidar_to_hsi, dtype=torch.float32),
            persistent=False,
        )
        if mode == "overlap-attention":
            self.hsi_query = nn.Linear(channels, attention_d_k)
            self.lidar_query = nn.Linear(channels, attention_d_k)
            self.hsi_key = nn.Linear(channels, attention_d_k)
            self.lidar_key = nn.Linear(channels, attention_d_k)
            self.hsi_value = nn.Linear(channels, channels)
            self.lidar_value = nn.Linear(channels, channels)
            self.attention_scale = attention_d_k ** -0.5

        self.hsi_message = nn.Linear(channels, channels)
        self.lidar_message = nn.Linear(channels, channels)
        self.hsi_gate = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.Sigmoid(),
        )
        self.lidar_gate = nn.Sequential(
            nn.Linear(2 * channels, channels),
            nn.Sigmoid(),
        )
        self.hsi_norm = nn.LayerNorm(channels)
        self.lidar_norm = nn.LayerNorm(channels)
        self.last_hsi_to_lidar_attention = None
        self.last_lidar_to_hsi_attention = None

    def _overlap_attention(
        self,
        query,
        key,
        value,
        overlap,
    ):
        logits = (
            query @ key.transpose(0, 1) * self.attention_scale
            + torch.log(overlap + 1e-6)
        )
        overlap_mask = overlap > 0
        logits = logits.masked_fill(
            ~overlap_mask,
            torch.finfo(logits.dtype).min,
        )
        attention = torch.softmax(logits, dim=-1)
        attention = attention * overlap_mask.to(attention.dtype)
        attention = attention / (
            attention.sum(dim=-1, keepdim=True) + 1e-6
        )
        return attention @ value, attention

    def forward(self, hsi_features, lidar_features):
        if self.mode == "overlap-gate":
            hsi_context = self.hsi_to_lidar @ lidar_features
            lidar_context = self.lidar_to_hsi @ hsi_features
        else:
            hsi_context, hsi_attention = self._overlap_attention(
                self.hsi_query(hsi_features),
                self.lidar_key(lidar_features),
                self.lidar_value(lidar_features),
                self.hsi_to_lidar,
            )
            lidar_context, lidar_attention = self._overlap_attention(
                self.lidar_query(lidar_features),
                self.hsi_key(hsi_features),
                self.hsi_value(hsi_features),
                self.lidar_to_hsi,
            )
            self.last_hsi_to_lidar_attention = hsi_attention.detach()
            self.last_lidar_to_hsi_attention = lidar_attention.detach()

        hsi_message = self.hsi_message(hsi_context)
        lidar_message = self.lidar_message(lidar_context)
        hsi_gate = self.hsi_gate(
            torch.cat([hsi_features, hsi_message], dim=-1)
        )
        lidar_gate = self.lidar_gate(
            torch.cat([lidar_features, lidar_message], dim=-1)
        )
        return (
            self.hsi_norm(hsi_features + hsi_gate * hsi_message),
            self.lidar_norm(
                lidar_features + lidar_gate * lidar_message
            ),
        )


class RAGLowHighModulation(nn.Module):
    """Geometry-gated low/high graph-frequency modulation for LiDAR."""

    def __init__(
        self,
        channels,
        rag_adjacency,
        geometry_descriptors,
    ):
        super().__init__()
        self.register_buffer(
            "rag_adjacency",
            torch.as_tensor(rag_adjacency, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "geometry_descriptors",
            torch.as_tensor(
                geometry_descriptors,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.low_projection = nn.Linear(channels, channels)
        self.high_projection = nn.Linear(channels, channels)
        self.geometry_gate = nn.Sequential(
            nn.Linear(geometry_descriptors.shape[1], channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
        self.output_norm = nn.LayerNorm(channels)
        self.last_gate = None

    def forward(self, node_features):
        low_frequency = self.rag_adjacency @ node_features
        high_frequency = node_features - low_frequency
        high_gate = self.geometry_gate(self.geometry_descriptors)
        modulated = (
            node_features
            + (1.0 - high_gate)
            * self.low_projection(low_frequency)
            + high_gate
            * self.high_projection(high_frequency)
        )
        self.last_gate = high_gate.detach()
        return self.output_norm(modulated)


class OriginalHGCNHLWithGSDGGraph(nn.Module):
    """Original HGCN-HL except for GSDG graph construction/propagation."""

    def __init__(
        self,
        height,
        width,
        input_channels,
        class_count,
        assignment,
        spatial_prior,
        hidden_dim=128,
        fusion_lambda=0.5,
        dropout=0.4,
        dynamic_d_k=16,
        dynamic_topk=8,
        dynamic_tau=1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.fusion_lambda = fusion_lambda
        pooling_assignment, projection_assignment = (
            normalized_sparse_assignments(assignment)
        )
        self.register_buffer(
            "pooling_assignment",
            pooling_assignment,
            persistent=False,
        )
        self.register_buffer(
            "projection_assignment",
            projection_assignment,
            persistent=False,
        )
        self.register_buffer(
            "spatial_prior",
            torch.as_tensor(spatial_prior, dtype=torch.float32),
            persistent=False,
        )

        # Unchanged original HGCN-HL joint HSI+LiDAR feature mapping.
        self.feature_mapping = nn.Sequential(
            WMF(input_channels, hidden_dim),
            WMF(hidden_dim, hidden_dim),
        )
        # Unchanged original HGCN-HL pixel CNN path.
        self.cnn_branch = nn.Sequential(
            OriginalSSConv(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
            ),
            OriginalSSConv(
                hidden_dim,
                hidden_dim,
                kernel_size=5,
            ),
        )

        # The only architectural replacement: fixed HGCN -> GSDG graph/GAT.
        self.graph_builder = DynamicGraphBuilder(
            hidden_dim,
            assignment.shape[1],
            d_k=dynamic_d_k,
            topk=dynamic_topk,
            tau=dynamic_tau,
        )
        self.gat1 = MultiHeadGAT(
            hidden_dim,
            head_channels=30,
            out_channels=hidden_dim,
            dropout=0.1,
            heads=4,
            alpha=0.2,
        )
        self.gat2 = MultiHeadGAT(
            hidden_dim,
            head_channels=60,
            out_channels=hidden_dim,
            dropout=0.2,
            heads=4,
            alpha=0.2,
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
        )
        self.classifier = nn.Linear(hidden_dim, class_count)

    def forward(self, joint_input):
        mapped = self.feature_mapping(
            joint_input.permute(2, 0, 1).unsqueeze(0)
        )
        cnn_features = (
            self.cnn_branch(mapped)
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(self.height * self.width, -1)
        )

        pixel_features = (
            mapped.squeeze(0)
            .permute(1, 2, 0)
            .reshape(self.height * self.width, -1)
        )
        node_features = torch.sparse.mm(
            self.pooling_assignment.transpose(0, 1),
            pixel_features,
        )
        adjacency = self.graph_builder(
            node_features,
            self.spatial_prior,
        )
        first_graph_features = self.gat1(
            node_features,
            adjacency,
        )
        graph_features = (
            self.gat2(first_graph_features, adjacency)
            + first_graph_features
        )
        graph_features = torch.sparse.mm(
            self.projection_assignment,
            graph_features,
        )
        graph_features = self.graph_projection(graph_features)

        fused_features = (
            self.fusion_lambda * graph_features
            + (1.0 - self.fusion_lambda) * cnn_features
        )
        return self.classifier(fused_features)


class ModalityGSDGGraphEncoder(nn.Module):
    """One modality-specific superpixel graph without a CNN branch."""

    def __init__(
        self,
        in_channels,
        assignment,
        spatial_prior,
        hidden_dim=128,
        dynamic_d_k=16,
        dynamic_topk=8,
        dynamic_tau=1.0,
        candidate_mask=None,
        use_edge_weights=False,
        use_fdsm=False,
        lidar_modulation="none",
        rag_adjacency=None,
        geometry_descriptors=None,
        use_cross_conditioned_builder=False,
    ):
        super().__init__()
        pooling_assignment, projection_assignment = (
            normalized_sparse_assignments(assignment)
        )
        self.register_buffer(
            "pooling_assignment",
            pooling_assignment,
            persistent=False,
        )
        self.register_buffer(
            "projection_assignment",
            projection_assignment,
            persistent=False,
        )
        self.register_buffer(
            "spatial_prior",
            torch.as_tensor(spatial_prior, dtype=torch.float32),
            persistent=False,
        )
        self.feature_mapping = nn.Sequential(
            WMF(in_channels, hidden_dim),
            WMF(hidden_dim, hidden_dim),
        )
        self.frequency_modulation = (
            FDSM(hidden_dim) if use_fdsm else nn.Identity()
        )
        if lidar_modulation == "none":
            self.structure_modulation = nn.Identity()
        elif lidar_modulation == "rag-lowhigh":
            if (
                rag_adjacency is None
                or geometry_descriptors is None
            ):
                raise ValueError(
                    "rag-lowhigh requires a RAG adjacency and "
                    "LiDAR geometry descriptors."
                )
            self.structure_modulation = RAGLowHighModulation(
                hidden_dim,
                rag_adjacency,
                geometry_descriptors,
            )
        else:
            raise ValueError(
                f"Unsupported LiDAR modulation: {lidar_modulation}"
            )
        graph_builder_options = {
            "d_k": dynamic_d_k,
            "topk": dynamic_topk,
            "tau": dynamic_tau,
        }
        if candidate_mask is None:
            self.graph_builder = DynamicGraphBuilder(
                hidden_dim,
                assignment.shape[1],
                **graph_builder_options,
            )
        else:
            self.graph_builder = MaskedDynamicGraphBuilder(
                hidden_dim,
                assignment.shape[1],
                candidate_mask=candidate_mask,
                **graph_builder_options,
            )
        self.cross_conditioned_graph_builder = (
            CrossConditionedDynamicGraphBuilder(
                hidden_dim,
                assignment.shape[1],
                cross_channels=hidden_dim,
                candidate_mask=candidate_mask,
                **graph_builder_options,
            )
            if use_cross_conditioned_builder
            else None
        )
        self.gat1 = MultiHeadGAT(
            hidden_dim,
            head_channels=30,
            out_channels=hidden_dim,
            dropout=0.1,
            heads=4,
            alpha=0.2,
            use_edge_weights=use_edge_weights,
        )
        self.gat2 = MultiHeadGAT(
            hidden_dim,
            head_channels=60,
            out_channels=hidden_dim,
            dropout=0.2,
            heads=4,
            alpha=0.2,
            use_edge_weights=use_edge_weights,
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.LeakyReLU(),
        )

    def encode_nodes(self, modality_input):
        height, width, _ = modality_input.shape
        mapped = self.feature_mapping(
            modality_input.permute(2, 0, 1).unsqueeze(0)
        )
        pixel_features = (
            mapped.squeeze(0)
            .permute(1, 2, 0)
            .reshape(height * width, -1)
        )
        node_features = torch.sparse.mm(
            self.pooling_assignment.transpose(0, 1),
            pixel_features,
        )
        node_features = self.frequency_modulation(node_features)
        return self.structure_modulation(node_features)

    def apply_gat1(self, node_features):
        adjacency = self.graph_builder(
            node_features,
            self.spatial_prior,
        )
        first_graph_features = self.gat1(node_features, adjacency)
        return first_graph_features, adjacency

    def apply_gat2_and_project(
        self,
        first_graph_features,
        adjacency=None,
        rebuild_graph=False,
        cross_context=None,
        topology_gate=None,
        topology_mask=None,
    ):
        graph_features = self.apply_gat2_nodes(
            first_graph_features,
            adjacency=adjacency,
            rebuild_graph=rebuild_graph,
            cross_context=cross_context,
            topology_gate=topology_gate,
            topology_mask=topology_mask,
        )
        return self.project_nodes(graph_features)

    def apply_gat2_nodes(
        self,
        first_graph_features,
        adjacency=None,
        rebuild_graph=False,
        cross_context=None,
        topology_gate=None,
        topology_mask=None,
        return_intra=False,
    ):
        if cross_context is not None:
            if self.cross_conditioned_graph_builder is None:
                raise ValueError(
                    "Cross-conditioned graph builder is not enabled."
                )
            adjacency = self.cross_conditioned_graph_builder(
                first_graph_features,
                self.spatial_prior,
                cross_context,
                topology_gate=topology_gate,
                topology_mask=topology_mask,
            )
        elif rebuild_graph:
            adjacency = self.graph_builder(
                first_graph_features,
                self.spatial_prior,
            )
        if adjacency is None:
            raise ValueError("GAT2 requires an adjacency matrix.")
        intra_features = self.gat2(first_graph_features, adjacency)
        if return_intra:
            return intra_features
        return intra_features + first_graph_features

    def project_nodes(self, graph_features):
        pixel_graph_features = torch.sparse.mm(
            self.projection_assignment,
            graph_features,
        )
        return self.graph_projection(pixel_graph_features)

    def forward(self, modality_input):
        node_features = self.encode_nodes(modality_input)
        first_graph_features, adjacency = self.apply_gat1(
            node_features
        )
        return self.apply_gat2_and_project(
            first_graph_features,
            adjacency=adjacency,
            rebuild_graph=False,
        )


class OriginalHGCNHLWithSeparateGSDGGraphs(nn.Module):
    """Independent modality graphs with the original joint CNN."""

    def __init__(
        self,
        height,
        width,
        hsi_channels,
        class_count,
        hsi_assignment,
        lidar_assignment,
        hsi_spatial_prior,
        lidar_spatial_prior,
        lidar_candidate_mask=None,
        hidden_dim=128,
        graph_modality_lambda=0.5,
        fusion_lambda=0.5,
        dynamic_d_k=16,
        dynamic_topk=8,
        dynamic_tau=1.0,
        cross_modal_interaction="none",
        cross_attention_d_k=16,
        overlap_metric="iou",
        contrastive_mode="none",
        contrastive_temperature=0.2,
        contrastive_dim=32,
        prototype_objective="cosine",
        post_gat_prototype_fusion="none",
        spsn_prototype_count=32,
        spsn_correlation_temperature=0.2,
        post_gat_consensus="none",
        consensus_anchor_count=0,
        consensus_temperature=0.2,
        consensus_gamma_init=0.0,
        consensus_fusion="fixed",
        consensus_writeback="direct",
        consensus_reliability_temperature=1.0,
        consensus_anchor_reasoning="none",
        consensus_structure_reliability="none",
        consensus_anchor_graph_topk=8,
        consensus_structure_temperature=0.1,
        consensus_structure_eta_init=0.0,
        post_gat_bridge="none",
        post_gat_consensus_graph="none",
        consensus_graph_weight=0.333,
        bridge_data=None,
        bridge_attention_d_k=32,
        bridge_attention_topk=8,
        bridge_gamma_init=0.0,
        transport_semantic_weight=1.0,
        transport_iterations=10,
        transport_warmup_epochs=50,
        variance_target=1.0,
        cell_interaction="none",
        cell_data=None,
        cell_pixel_descriptor="none",
        cell_edge_mode="binary",
        cell_interaction_stages=1,
        cell_output_branch="none",
        cell_output_weight=1.0 / 3.0,
        cell_topology_veto="none",
        cell_veto_threshold=0.05,
        fdsm_scope="none",
        lidar_modulation="none",
        cnn_branch="original",
        lidar_rag_adjacency=None,
        lidar_geometry_descriptors=None,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.graph_modality_lambda = graph_modality_lambda
        self.fusion_lambda = fusion_lambda
        self.cross_modal_interaction = cross_modal_interaction
        self.overlap_metric = overlap_metric
        self.contrastive_mode = contrastive_mode
        self.last_contrastive_loss = None
        self.last_variance_loss = None
        self.post_gat_prototype_fusion = (
            post_gat_prototype_fusion
        )
        self.post_gat_consensus = post_gat_consensus
        self.post_gat_consensus_graph = post_gat_consensus_graph
        self.consensus_graph_weight = consensus_graph_weight
        self.cell_interaction = cell_interaction
        self.cell_interaction_stages = cell_interaction_stages
        self.cell_output_branch = cell_output_branch
        self.cell_output_weight = cell_output_weight
        self.cell_topology_veto = cell_topology_veto
        self.cell_veto_threshold = cell_veto_threshold
        self.cnn_branch_mode = cnn_branch
        use_qk_condition = (
            cross_modal_interaction == "overlap-qk-condition"
        )
        if contrastive_mode == "none":
            self.contrastive_module = None
        else:
            if contrastive_mode == "overlap-transport":
                transport_targets = build_overlap_transport_targets(
                    hsi_assignment,
                    lidar_assignment,
                )
                self.contrastive_module = (
                    OverlapTransportDistillationLoss(
                        hidden_dim,
                        contrastive_dim,
                        contrastive_temperature,
                        transport_targets,
                        semantic_weight=(
                            transport_semantic_weight
                        ),
                        sinkhorn_iterations=transport_iterations,
                        warmup_epochs=transport_warmup_epochs,
                        variance_target=variance_target,
                    )
                )
            elif contrastive_mode == "overlap-prototype":
                contrastive_targets = (
                    build_overlap_distribution_targets(
                        hsi_assignment,
                        lidar_assignment,
                    )
                )
                self.contrastive_module = (
                    OverlapPrototypeContrastiveLoss(
                        hidden_dim,
                        contrastive_dim,
                        contrastive_temperature,
                        contrastive_targets,
                        objective=prototype_objective,
                    )
                )
            else:
                contrastive_targets = (
                    build_overlap_distribution_targets(
                        hsi_assignment,
                        lidar_assignment,
                    )
                )
                self.contrastive_module = (
                    OverlapDistributionContrastiveLoss(
                        hidden_dim,
                        contrastive_dim,
                        contrastive_temperature,
                        contrastive_targets,
                    )
                )
        self.hsi_graph = ModalityGSDGGraphEncoder(
            in_channels=hsi_channels,
            assignment=hsi_assignment,
            spatial_prior=hsi_spatial_prior,
            hidden_dim=hidden_dim,
            dynamic_d_k=dynamic_d_k,
            dynamic_topk=dynamic_topk,
            dynamic_tau=dynamic_tau,
            use_edge_weights=use_qk_condition,
            use_fdsm=fdsm_scope == "hsi",
            use_cross_conditioned_builder=use_qk_condition,
        )
        self.lidar_graph = ModalityGSDGGraphEncoder(
            in_channels=1,
            assignment=lidar_assignment,
            spatial_prior=lidar_spatial_prior,
            hidden_dim=hidden_dim,
            dynamic_d_k=dynamic_d_k,
            dynamic_topk=dynamic_topk,
            dynamic_tau=dynamic_tau,
            candidate_mask=lidar_candidate_mask,
            use_edge_weights=(
                lidar_candidate_mask is not None
                or use_qk_condition
            ),
            lidar_modulation=lidar_modulation,
            rag_adjacency=lidar_rag_adjacency,
            geometry_descriptors=lidar_geometry_descriptors,
            use_cross_conditioned_builder=use_qk_condition,
        )
        if post_gat_prototype_fusion == "spsn-correlation":
            if spsn_prototype_count > min(
                hsi_assignment.shape[1],
                lidar_assignment.shape[1],
            ):
                raise ValueError(
                    "--spsn-prototype-count cannot exceed either "
                    "modality's superpixel count."
                )
            self.prototype_correlation_fusion = (
                PostGATPrototypeCorrelationFusion(
                    hidden_dim,
                    spsn_prototype_count,
                    spsn_correlation_temperature,
                    initial_hsi_weight=graph_modality_lambda,
                )
            )
        else:
            self.prototype_correlation_fusion = None
        if cross_modal_interaction == "overlap-qk-condition":
            hsi_to_lidar, lidar_to_hsi = build_cross_modal_overlap(
                hsi_assignment,
                lidar_assignment,
                metric=overlap_metric,
            )
            self.register_buffer(
                "hsi_to_lidar_overlap",
                torch.as_tensor(
                    hsi_to_lidar,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "lidar_to_hsi_overlap",
                torch.as_tensor(
                    lidar_to_hsi,
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.cross_interaction = None
        elif cross_modal_interaction == "none":
            self.register_buffer(
                "hsi_to_lidar_overlap",
                None,
                persistent=False,
            )
            self.register_buffer(
                "lidar_to_hsi_overlap",
                None,
                persistent=False,
            )
            self.cross_interaction = None
        else:
            self.register_buffer(
                "hsi_to_lidar_overlap",
                None,
                persistent=False,
            )
            self.register_buffer(
                "lidar_to_hsi_overlap",
                None,
                persistent=False,
            )
            hsi_to_lidar, lidar_to_hsi = build_cross_modal_overlap(
                hsi_assignment,
                lidar_assignment,
                metric=overlap_metric,
            )
            self.cross_interaction = OverlapCrossModalInteraction(
                hidden_dim,
                hsi_to_lidar,
                lidar_to_hsi,
                mode=cross_modal_interaction,
                attention_d_k=cross_attention_d_k,
            )
        if cell_interaction == "rag":
            if cell_data is None:
                raise ValueError(
                    "cell_data is required for cell interaction."
                )
            cell_layer_options = {
                "channels": hidden_dim,
                "cell_data": cell_data,
                "use_pixel_descriptors": (
                    cell_pixel_descriptor == "mean"
                ),
                "edge_mode": cell_edge_mode,
            }
            self.cell_layer = IntersectionCellRAGLayer(
                **cell_layer_options,
            )
            self.cell_layer2 = (
                IntersectionCellRAGLayer(
                    **cell_layer_options,
                )
                if cell_interaction_stages == 2
                else None
            )
            if cell_topology_veto != "none":
                self.register_buffer(
                    "hsi_topology_support",
                    torch.as_tensor(
                        cell_data["hsi_topology_support"],
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "lidar_topology_support",
                    torch.as_tensor(
                        cell_data["lidar_topology_support"],
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "hsi_topology_support",
                    None,
                    persistent=False,
                )
                self.register_buffer(
                    "lidar_topology_support",
                    None,
                    persistent=False,
                )
            if cell_topology_veto == "soft":
                self.hsi_topology_projection = nn.Linear(1, 1)
                self.lidar_topology_projection = nn.Linear(1, 1)
                for projection in (
                    self.hsi_topology_projection,
                    self.lidar_topology_projection,
                ):
                    nn.init.constant_(projection.weight, 10.0)
                    nn.init.constant_(
                        projection.bias,
                        -10.0 * cell_veto_threshold,
                    )
            else:
                self.hsi_topology_projection = None
                self.lidar_topology_projection = None
            if cell_output_branch == "fixed":
                pixel_cell_index = np.asarray(
                    cell_data["pixel_cell_index"],
                    dtype=np.int64,
                )
                pixel_indices = np.arange(
                    pixel_cell_index.size,
                    dtype=np.int64,
                )
                cell_assignment = coo_matrix(
                    (
                        np.ones(
                            pixel_cell_index.size,
                            dtype=np.float32,
                        ),
                        (pixel_indices, pixel_cell_index),
                    ),
                    shape=(
                        pixel_cell_index.size,
                        cell_data["cell_count"],
                    ),
                    dtype=np.float32,
                )
                self.register_buffer(
                    "cell_projection_assignment",
                    scipy_sparse_to_torch(cell_assignment),
                    persistent=False,
                )
            else:
                self.register_buffer(
                    "cell_projection_assignment",
                    None,
                    persistent=False,
                )
        elif cell_interaction == "none":
            self.cell_layer = None
            self.cell_layer2 = None
            self.register_buffer(
                "hsi_topology_support",
                None,
                persistent=False,
            )
            self.register_buffer(
                "lidar_topology_support",
                None,
                persistent=False,
            )
            self.hsi_topology_projection = None
            self.lidar_topology_projection = None
            self.register_buffer(
                "cell_projection_assignment",
                None,
                persistent=False,
            )
        else:
            raise ValueError(
                "cell_interaction must be none or rag."
            )

        # Both choices retain the same joint PCA(HSI)+LiDAR input stem.
        self.joint_feature_mapping = nn.Sequential(
            WMF(hsi_channels + 1, hidden_dim),
            WMF(hidden_dim, hidden_dim),
        )
        if cnn_branch == "original":
            self.cnn_branch = nn.Sequential(
                OriginalSSConv(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=5,
                ),
                OriginalSSConv(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=5,
                ),
            )
        elif cnn_branch == "gsdg":
            self.cnn_branch = nn.Sequential(
                DwsConv(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=3,
                ),
                DwsConv(
                    hidden_dim,
                    hidden_dim,
                    kernel_size=7,
                ),
            )
        else:
            raise ValueError(
                "cnn_branch must be 'original' or 'gsdg'."
            )
        self.classifier = nn.Linear(hidden_dim, class_count)
        if post_gat_consensus == "mssagf-anchor":
            resolved_anchor_count = (
                consensus_anchor_count
                if consensus_anchor_count > 0
                else 2 * class_count
            )
            hsi_area = np.asarray(
                hsi_assignment.sum(axis=0)
            ).reshape(-1)
            lidar_area = np.asarray(
                lidar_assignment.sum(axis=0)
            ).reshape(-1)
            with torch.random.fork_rng(devices=[]):
                self.consensus_anchor_interaction = (
                    PostGATConsensusAnchorInteraction(
                        hidden_dim,
                        resolved_anchor_count,
                        consensus_temperature,
                        hsi_area,
                        lidar_area,
                        gamma_init=consensus_gamma_init,
                        fusion_mode=consensus_fusion,
                        writeback_mode=consensus_writeback,
                        reliability_temperature=(
                            consensus_reliability_temperature
                        ),
                        anchor_reasoning=(
                            consensus_anchor_reasoning
                        ),
                        structure_reliability=(
                            consensus_structure_reliability
                        ),
                        anchor_graph_topk=(
                            consensus_anchor_graph_topk
                        ),
                        structure_temperature=(
                            consensus_structure_temperature
                        ),
                        structure_eta_init=(
                            consensus_structure_eta_init
                        ),
                    )
                )
        else:
            self.consensus_anchor_interaction = None
        if post_gat_bridge == "center-block":
            if bridge_data is None:
                raise ValueError(
                    "bridge_data is required for center bridge block."
                )
            self.bridge_block_interaction = (
                CenterBridgeBlockInteraction(
                    hidden_dim,
                    bridge_data,
                    attention_d_k=bridge_attention_d_k,
                    topk=bridge_attention_topk,
                    gamma_init=bridge_gamma_init,
                )
            )
        else:
            self.bridge_block_interaction = None
        if post_gat_consensus_graph == "center-mediator":
            if bridge_data is None:
                raise ValueError(
                    "bridge_data is required for center mediator "
                    "consensus graph."
                )
            self.consensus_graph_branch = (
                PostGATMediatedConsensusGraph(
                    hidden_dim,
                    bridge_data,
                    attention_d_k=bridge_attention_d_k,
                    topk=bridge_attention_topk,
                    gamma_init=bridge_gamma_init,
                )
            )
        else:
            self.consensus_graph_branch = None

    def set_contrastive_epoch(self, epoch):
        if (
            self.contrastive_module is not None
            and hasattr(self.contrastive_module, "set_epoch")
        ):
            self.contrastive_module.set_epoch(epoch)

    def _update_contrastive_loss(
        self,
        hsi_features,
        lidar_features,
    ):
        if self.training and self.contrastive_module is not None:
            self.last_contrastive_loss = self.contrastive_module(
                hsi_features,
                lidar_features,
            )
            self.last_variance_loss = getattr(
                self.contrastive_module,
                "last_variance_loss",
                None,
            )
        else:
            self.last_contrastive_loss = None
            self.last_variance_loss = None

    def _cell_topology_constraints(self):
        if self.cell_topology_veto == "none":
            return None, None, None, None
        if self.cell_topology_veto == "soft":
            hsi_gate = torch.sigmoid(
                self.hsi_topology_projection(
                    self.hsi_topology_support.unsqueeze(-1)
                ).squeeze(-1)
            )
            lidar_gate = torch.sigmoid(
                self.lidar_topology_projection(
                    self.lidar_topology_support.unsqueeze(-1)
                ).squeeze(-1)
            )
            return hsi_gate, lidar_gate, None, None
        return (
            None,
            None,
            self.hsi_topology_support
            >= self.cell_veto_threshold,
            self.lidar_topology_support
            >= self.cell_veto_threshold,
        )

    def _has_post_node_interaction(self):
        return (
            self.consensus_anchor_interaction is not None
            or self.bridge_block_interaction is not None
        )

    def _needs_explicit_node_path(self):
        return (
            self._has_post_node_interaction()
            or self.consensus_graph_branch is not None
        )

    def forward(self, hsi, lidar, joint_input):
        self.last_contrastive_loss = None
        self.last_variance_loss = None
        if self.cell_layer is not None:
            hsi_nodes = self.hsi_graph.encode_nodes(hsi)
            lidar_nodes = self.lidar_graph.encode_nodes(lidar)
            hsi_features, _ = self.hsi_graph.apply_gat1(hsi_nodes)
            lidar_features, _ = self.lidar_graph.apply_gat1(
                lidar_nodes
            )
            (
                hsi_features,
                lidar_features,
                latest_cell_features,
            ) = self.cell_layer(
                hsi_nodes,
                lidar_nodes,
                hsi_features,
                lidar_features,
            )
            (
                hsi_topology_gate,
                lidar_topology_gate,
                hsi_topology_mask,
                lidar_topology_mask,
            ) = self._cell_topology_constraints()
            if (
                self.cross_modal_interaction
                == "overlap-qk-condition"
            ):
                hsi_cross_context = (
                    self.hsi_to_lidar_overlap @ lidar_features
                )
                lidar_cross_context = (
                    self.lidar_to_hsi_overlap @ hsi_features
                )
                hsi_intra2 = self.hsi_graph.apply_gat2_nodes(
                    hsi_features,
                    cross_context=hsi_cross_context,
                    topology_gate=hsi_topology_gate,
                    topology_mask=hsi_topology_mask,
                    return_intra=True,
                )
                lidar_intra2 = self.lidar_graph.apply_gat2_nodes(
                    lidar_features,
                    cross_context=lidar_cross_context,
                    topology_gate=lidar_topology_gate,
                    topology_mask=lidar_topology_mask,
                    return_intra=True,
                )
            else:
                if self.cross_interaction is not None:
                    hsi_features, lidar_features = (
                        self.cross_interaction(
                            hsi_features,
                            lidar_features,
                        )
                    )
                hsi_intra2 = self.hsi_graph.apply_gat2_nodes(
                    hsi_features,
                    rebuild_graph=True,
                    return_intra=True,
                )
                lidar_intra2 = self.lidar_graph.apply_gat2_nodes(
                    lidar_features,
                    rebuild_graph=True,
                    return_intra=True,
                )
            if self.cell_layer2 is not None:
                (
                    hsi_final_nodes,
                    lidar_final_nodes,
                    latest_cell_features,
                ) = self.cell_layer2(
                    hsi_features,
                    lidar_features,
                    hsi_intra2,
                    lidar_intra2,
                )
            else:
                hsi_final_nodes = hsi_features + hsi_intra2
                lidar_final_nodes = lidar_features + lidar_intra2
            self._update_contrastive_loss(
                hsi_final_nodes,
                lidar_final_nodes,
            )
            if not self._needs_explicit_node_path():
                hsi_graph_features = self.hsi_graph.project_nodes(
                    hsi_final_nodes
                )
                lidar_graph_features = (
                    self.lidar_graph.project_nodes(
                        lidar_final_nodes
                    )
                )
        elif self.cross_modal_interaction == "none":
            if (
                self.contrastive_module is None
                and self.prototype_correlation_fusion is None
                and not self._needs_explicit_node_path()
            ):
                # Preserve the original Stage-3 path exactly.
                hsi_graph_features = self.hsi_graph(hsi)
                lidar_graph_features = self.lidar_graph(lidar)
            else:
                hsi_nodes = self.hsi_graph.encode_nodes(hsi)
                (
                    hsi_features,
                    hsi_adjacency,
                ) = self.hsi_graph.apply_gat1(hsi_nodes)
                hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                    hsi_features,
                    adjacency=hsi_adjacency,
                )
                if not self._needs_explicit_node_path():
                    hsi_graph_features = (
                        self.hsi_graph.project_nodes(
                            hsi_final_nodes
                        )
                    )
                lidar_nodes = self.lidar_graph.encode_nodes(lidar)
                (
                    lidar_features,
                    lidar_adjacency,
                ) = self.lidar_graph.apply_gat1(lidar_nodes)
                lidar_final_nodes = (
                    self.lidar_graph.apply_gat2_nodes(
                        lidar_features,
                        adjacency=lidar_adjacency,
                    )
                )
                if not self._needs_explicit_node_path():
                    lidar_graph_features = (
                        self.lidar_graph.project_nodes(
                            lidar_final_nodes
                        )
                    )
                self._update_contrastive_loss(
                    hsi_final_nodes,
                    lidar_final_nodes,
                )
        elif (
            self.cross_modal_interaction
            == "overlap-qk-condition"
        ):
            hsi_nodes = self.hsi_graph.encode_nodes(hsi)
            lidar_nodes = self.lidar_graph.encode_nodes(lidar)
            hsi_features, _ = self.hsi_graph.apply_gat1(hsi_nodes)
            lidar_features, _ = self.lidar_graph.apply_gat1(
                lidar_nodes
            )
            hsi_cross_context = (
                self.hsi_to_lidar_overlap @ lidar_features
            )
            lidar_cross_context = (
                self.lidar_to_hsi_overlap @ hsi_features
            )
            if (
                self.contrastive_module is None
                and self.prototype_correlation_fusion is None
                and not self._needs_explicit_node_path()
            ):
                hsi_graph_features = (
                    self.hsi_graph.apply_gat2_and_project(
                        hsi_features,
                        cross_context=hsi_cross_context,
                    )
                )
                lidar_graph_features = (
                    self.lidar_graph.apply_gat2_and_project(
                        lidar_features,
                        cross_context=lidar_cross_context,
                    )
                )
            else:
                hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                    hsi_features,
                    cross_context=hsi_cross_context,
                )
                lidar_final_nodes = (
                    self.lidar_graph.apply_gat2_nodes(
                        lidar_features,
                        cross_context=lidar_cross_context,
                    )
                )
                self._update_contrastive_loss(
                    hsi_final_nodes,
                    lidar_final_nodes,
                )
                if not self._needs_explicit_node_path():
                    hsi_graph_features = (
                        self.hsi_graph.project_nodes(
                            hsi_final_nodes
                        )
                    )
                    lidar_graph_features = (
                        self.lidar_graph.project_nodes(
                            lidar_final_nodes
                        )
                    )
        else:
            hsi_nodes = self.hsi_graph.encode_nodes(hsi)
            lidar_nodes = self.lidar_graph.encode_nodes(lidar)
            hsi_features, _ = self.hsi_graph.apply_gat1(hsi_nodes)
            lidar_features, _ = self.lidar_graph.apply_gat1(
                lidar_nodes
            )
            hsi_features, lidar_features = self.cross_interaction(
                hsi_features,
                lidar_features,
            )
            # Cross-modal information participates in the second graph.
            if (
                self.contrastive_module is None
                and self.prototype_correlation_fusion is None
                and not self._needs_explicit_node_path()
            ):
                hsi_graph_features = (
                    self.hsi_graph.apply_gat2_and_project(
                        hsi_features,
                        rebuild_graph=True,
                    )
                )
                lidar_graph_features = (
                    self.lidar_graph.apply_gat2_and_project(
                        lidar_features,
                        rebuild_graph=True,
                    )
                )
            else:
                hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                    hsi_features,
                    rebuild_graph=True,
                )
                lidar_final_nodes = (
                    self.lidar_graph.apply_gat2_nodes(
                        lidar_features,
                        rebuild_graph=True,
                    )
                )
                self._update_contrastive_loss(
                    hsi_final_nodes,
                    lidar_final_nodes,
                )
                if not self._needs_explicit_node_path():
                    hsi_graph_features = (
                        self.hsi_graph.project_nodes(
                            hsi_final_nodes
                        )
                    )
                    lidar_graph_features = (
                        self.lidar_graph.project_nodes(
                            lidar_final_nodes
                        )
                    )
        if self._has_post_node_interaction():
            if self.consensus_anchor_interaction is not None:
                (
                    hsi_final_nodes,
                    lidar_final_nodes,
                ) = self.consensus_anchor_interaction(
                    hsi_final_nodes,
                    lidar_final_nodes,
                )
            if self.bridge_block_interaction is not None:
                (
                    hsi_final_nodes,
                    lidar_final_nodes,
                ) = self.bridge_block_interaction(
                    hsi_final_nodes,
                    lidar_final_nodes,
                )
            hsi_graph_features = self.hsi_graph.project_nodes(
                hsi_final_nodes
            )
            lidar_graph_features = self.lidar_graph.project_nodes(
                lidar_final_nodes
            )
        elif self._needs_explicit_node_path():
            hsi_graph_features = self.hsi_graph.project_nodes(
                hsi_final_nodes
            )
            lidar_graph_features = self.lidar_graph.project_nodes(
                lidar_final_nodes
            )

        if self.consensus_graph_branch is not None:
            consensus_graph_features = self.consensus_graph_branch(
                hsi_final_nodes,
                lidar_final_nodes,
            )
            consensus_weight = self.consensus_graph_weight
            private_weight = 1.0 - consensus_weight
            graph_features = (
                private_weight
                * self.graph_modality_lambda
                * hsi_graph_features
                + private_weight
                * (1.0 - self.graph_modality_lambda)
                * lidar_graph_features
                + consensus_weight * consensus_graph_features
            )
        elif self.prototype_correlation_fusion is None:
            graph_features = (
                self.graph_modality_lambda * hsi_graph_features
                + (1.0 - self.graph_modality_lambda)
                * lidar_graph_features
            )
        else:
            graph_features = self.prototype_correlation_fusion(
                hsi_final_nodes,
                lidar_final_nodes,
                hsi_graph_features,
                lidar_graph_features,
                self.hsi_graph.projection_assignment,
                self.lidar_graph.projection_assignment,
            )
        if self.cell_projection_assignment is not None:
            cell_pixel_features = torch.sparse.mm(
                self.cell_projection_assignment,
                latest_cell_features,
            )
            graph_features = (
                (1.0 - self.cell_output_weight) * graph_features
                + self.cell_output_weight * cell_pixel_features
            )

        mapped_joint = self.joint_feature_mapping(
            joint_input.permute(2, 0, 1).unsqueeze(0)
        )
        cnn_features = (
            self.cnn_branch(mapped_joint)
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(self.height * self.width, -1)
        )
        fused_features = (
            self.fusion_lambda * graph_features
            + (1.0 - self.fusion_lambda) * cnn_features
        )
        return self.classifier(fused_features)


def prepare_data(args, config):
    hsi, lidar, gt, class_count, _, _ = get_HSI_LiDAR_data(
        config["loader_name"],
        args.data_dir,
    )
    hsi = minmax_normalize(hsi)
    lidar = minmax_normalize(lidar)
    if lidar.ndim == 3:
        lidar = lidar[:, :, 0]

    (
        hsi_assignment,
        lidar_assignment,
        assignment,
        lidar_assignment_parts,
    ) = build_modality_superpixel_assignments(
        hsi,
        lidar[:, :, np.newaxis],
        args.scales,
        args.lidar_segmentation,
    )
    hsi_spatial_prior = build_superpixel_spatial_prior(
        hsi_assignment,
        hsi.shape[0],
        hsi.shape[1],
        args.spatial_prior_k,
    )
    lidar_spatial_prior = build_superpixel_spatial_prior(
        lidar_assignment,
        hsi.shape[0],
        hsi.shape[1],
        args.spatial_prior_k,
    )
    lidar_rag_adjacency = None
    lidar_geometry_descriptors = None
    if (
        args.lidar_graph_prior == "rag-height-knn"
        or args.lidar_modulation == "rag-lowhigh"
    ):
        (
            lidar_rag_adjacency,
            lidar_geometry_descriptors,
        ) = build_lidar_rag_modulation_structure(
            lidar_assignment_parts,
            lidar,
            args.lidar_rag_hops,
        )
    lidar_candidate_mask = None
    if args.lidar_graph_prior == "rag-height-knn":
        (
            lidar_spatial_prior,
            lidar_candidate_mask,
        ) = build_lidar_rag_height_knn_prior(
            lidar_assignment_parts,
            lidar,
            lidar_spatial_prior,
            lidar_rag_adjacency,
            args.lidar_height_knn_k,
        )
    joint_spatial_prior = build_superpixel_spatial_prior(
        assignment,
        hsi.shape[0],
        hsi.shape[1],
        args.spatial_prior_k,
    )
    cell_data = None
    if args.cell_interaction == "rag":
        cell_data = build_common_refinement_cells(
            hsi_assignment,
            lidar_assignment,
            hsi.shape[0],
            hsi.shape[1],
        )
        if (
            args.cell_edge_mode
            == "spectral-height-boundary"
        ):
            build_multimodal_weighted_cell_rag(
                cell_data,
                hsi,
                lidar,
                hsi.shape[0],
                hsi.shape[1],
                sam_weight=args.cell_sam_weight,
                height_weight=args.cell_height_weight,
                boundary_weight=args.cell_boundary_weight,
                conflict_weight=args.cell_conflict_weight,
            )
        if args.cell_topology_veto != "none":
            attach_cell_parent_topology_supports(
                cell_data,
                args.cell_edge_mode,
            )

    height, width, bands = hsi.shape
    component_count = min(args.pca_components, bands)
    reduced_hsi = PCA(
        n_components=component_count,
        random_state=args.seed,
    ).fit_transform(hsi.reshape(-1, bands))
    reduced_hsi = reduced_hsi.reshape(
        height,
        width,
        component_count,
    ).astype(np.float32)
    joint_input = np.concatenate(
        [reduced_hsi, lidar[:, :, np.newaxis]],
        axis=2,
    ).astype(np.float32)
    lidar_features = lidar[:, :, np.newaxis].astype(np.float32)
    if (
        cell_data is not None
        and args.cell_pixel_descriptor == "mean"
    ):
        attach_cell_pixel_descriptors(
            cell_data,
            reduced_hsi,
            lidar_features,
        )
    bridge_data = None
    if (
        args.post_gat_bridge == "center-block"
        or args.post_gat_consensus_graph == "center-mediator"
    ):
        bridge_anchor_count = (
            args.bridge_anchor_count
            if args.bridge_anchor_count > 0
            else 2 * class_count
        )
        bridge_data = build_center_bridge_data(
            hsi_assignment,
            lidar_assignment,
            lidar,
            height,
            width,
            bridge_anchor_count,
            overlap_metric=args.bridge_overlap_metric,
            overlap_weight=args.bridge_overlap_weight,
            spatial_weight=args.bridge_spatial_weight,
            height_weight=args.bridge_height_weight,
        )
    return (
        reduced_hsi,
        lidar_features,
        joint_input,
        gt,
        class_count,
        hsi_assignment.astype(np.float32),
        lidar_assignment.astype(np.float32),
        assignment.astype(np.float32),
        hsi_spatial_prior,
        lidar_spatial_prior,
        lidar_candidate_mask,
        lidar_rag_adjacency,
        lidar_geometry_descriptors,
        cell_data,
        bridge_data,
        joint_spatial_prior,
    )


def train_one_run(
    args,
    hsi_features,
    lidar_features,
    joint_input,
    gt,
    class_count,
    hsi_assignment,
    lidar_assignment,
    joint_assignment,
    hsi_spatial_prior,
    lidar_spatial_prior,
    lidar_candidate_mask,
    lidar_rag_adjacency,
    lidar_geometry_descriptors,
    cell_data,
    bridge_data,
    joint_spatial_prior,
    run_index,
):
    run_seed = args.seed + run_index
    set_seed(run_seed)
    train_indices, test_indices = split_fixed_samples_per_class(
        gt,
        class_count,
        args.train_samples_per_class,
        run_seed,
    )
    device = torch.device(args.device)
    hsi_x = torch.from_numpy(hsi_features).to(device)
    lidar_x = torch.from_numpy(lidar_features).to(device)
    joint_x = torch.from_numpy(joint_input).to(device)
    flat_gt = gt.reshape(-1)
    train_index = torch.from_numpy(train_indices).long().to(device)
    test_index = torch.from_numpy(test_indices).long().to(device)
    train_labels = torch.from_numpy(
        flat_gt[train_indices] - 1
    ).long().to(device)
    test_labels = torch.from_numpy(
        flat_gt[test_indices] - 1
    ).long().to(device)

    common_options = {
        "height": joint_input.shape[0],
        "width": joint_input.shape[1],
        "class_count": class_count,
        "hidden_dim": args.hidden_dim,
        "fusion_lambda": args.fusion_lambda,
        "dynamic_d_k": args.dynamic_dk,
        "dynamic_topk": args.dynamic_topk,
        "dynamic_tau": args.dynamic_tau,
    }
    if args.graph_layout == "separate":
        model = OriginalHGCNHLWithSeparateGSDGGraphs(
            hsi_channels=hsi_features.shape[2],
            hsi_assignment=hsi_assignment,
            lidar_assignment=lidar_assignment,
            hsi_spatial_prior=hsi_spatial_prior,
            lidar_spatial_prior=lidar_spatial_prior,
            lidar_candidate_mask=lidar_candidate_mask,
            lidar_rag_adjacency=lidar_rag_adjacency,
            lidar_geometry_descriptors=(
                lidar_geometry_descriptors
            ),
            lidar_modulation=args.lidar_modulation,
            graph_modality_lambda=args.graph_modality_lambda,
            cross_modal_interaction=args.cross_modal_interaction,
            cross_attention_d_k=args.cross_attention_dk,
            overlap_metric=args.overlap_metric,
            contrastive_mode=args.contrastive_mode,
            contrastive_temperature=(
                args.contrastive_temperature
            ),
            contrastive_dim=args.contrastive_dim,
            prototype_objective=args.prototype_objective,
            post_gat_prototype_fusion=(
                args.post_gat_prototype_fusion
            ),
            spsn_prototype_count=args.spsn_prototype_count,
            spsn_correlation_temperature=(
                args.spsn_correlation_temperature
            ),
            post_gat_consensus=args.post_gat_consensus,
            consensus_anchor_count=(
                args.consensus_anchor_count
            ),
            consensus_temperature=args.consensus_temperature,
            consensus_gamma_init=args.consensus_gamma_init,
            consensus_fusion=args.consensus_fusion,
            consensus_writeback=args.consensus_writeback,
            consensus_reliability_temperature=(
                args.consensus_reliability_temperature
            ),
            consensus_anchor_reasoning=(
                args.consensus_anchor_reasoning
            ),
            consensus_structure_reliability=(
                args.consensus_structure_reliability
            ),
            consensus_anchor_graph_topk=(
                args.consensus_anchor_graph_topk
            ),
            consensus_structure_temperature=(
                args.consensus_structure_temperature
            ),
            consensus_structure_eta_init=(
                args.consensus_structure_eta_init
            ),
            post_gat_bridge=args.post_gat_bridge,
            post_gat_consensus_graph=(
                args.post_gat_consensus_graph
            ),
            consensus_graph_weight=args.consensus_graph_weight,
            bridge_attention_d_k=args.bridge_attention_dk,
            bridge_attention_topk=args.bridge_attention_topk,
            bridge_gamma_init=args.bridge_gamma_init,
            transport_semantic_weight=(
                args.transport_semantic_weight
            ),
            transport_iterations=args.transport_iterations,
            transport_warmup_epochs=(
                args.transport_warmup_epochs
            ),
            variance_target=args.variance_target,
            cell_interaction=args.cell_interaction,
            cell_data=cell_data,
            bridge_data=bridge_data,
            cell_pixel_descriptor=args.cell_pixel_descriptor,
            cell_edge_mode=args.cell_edge_mode,
            cell_interaction_stages=args.cell_interaction_stages,
            cell_output_branch=args.cell_output_branch,
            cell_output_weight=args.cell_output_weight,
            cell_topology_veto=args.cell_topology_veto,
            cell_veto_threshold=args.cell_veto_threshold,
            fdsm_scope=args.fdsm_scope,
            cnn_branch=args.cnn_branch,
            **common_options,
        ).to(device)

        def forward_model():
            return model(hsi_x, lidar_x, joint_x)
    else:
        model = OriginalHGCNHLWithGSDGGraph(
            input_channels=joint_input.shape[2],
            assignment=joint_assignment,
            spatial_prior=joint_spatial_prior,
            dropout=args.dropout,
            **common_options,
        ).to(device)

        def forward_model():
            return model(joint_x)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    best_loss = float("inf")
    best_state = None
    transport_diagnostics = []
    prototype_fusion_diagnostics = []
    consensus_diagnostics = []
    bridge_diagnostics = []
    consensus_graph_diagnostics = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        if hasattr(model, "set_contrastive_epoch"):
            model.set_contrastive_epoch(epoch)
        optimizer.zero_grad()
        logits = forward_model()
        classification_loss = criterion(
            logits.index_select(0, train_index),
            train_labels,
        )
        contrastive_loss = getattr(
            model,
            "last_contrastive_loss",
            None,
        )
        if contrastive_loss is None:
            contrastive_loss = classification_loss.new_zeros(())
        variance_loss = getattr(
            model,
            "last_variance_loss",
            None,
        )
        if variance_loss is None:
            variance_loss = classification_loss.new_zeros(())
        loss = (
            classification_loss
            + args.contrastive_weight * contrastive_loss
            + args.variance_weight * variance_loss
        )
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % args.log_interval == 0:
            transport_record = None
            contrastive_module = getattr(
                model,
                "contrastive_module",
                None,
            )
            if (
                contrastive_module is not None
                and hasattr(contrastive_module, "diagnostics")
            ):
                transport_record = (
                    contrastive_module.diagnostics()
                )
                if transport_record is not None:
                    transport_record = {
                        "epoch": epoch,
                        **transport_record,
                    }
                    transport_diagnostics.append(
                        transport_record
                    )
            prototype_fusion_record = None
            prototype_fusion_module = getattr(
                model,
                "prototype_correlation_fusion",
                None,
            )
            if prototype_fusion_module is not None:
                prototype_fusion_record = (
                    prototype_fusion_module.diagnostics()
                )
                if prototype_fusion_record is not None:
                    prototype_fusion_record = {
                        "epoch": epoch,
                        **prototype_fusion_record,
                    }
                    prototype_fusion_diagnostics.append(
                        prototype_fusion_record
                    )
            consensus_record = None
            consensus_module = getattr(
                model,
                "consensus_anchor_interaction",
                None,
            )
            if consensus_module is not None:
                consensus_record = consensus_module.diagnostics()
                if consensus_record is not None:
                    consensus_record = {
                        "epoch": epoch,
                        **consensus_record,
                    }
                    consensus_diagnostics.append(
                        consensus_record
                    )
            bridge_record = None
            bridge_module = getattr(
                model,
                "bridge_block_interaction",
                None,
            )
            if bridge_module is not None:
                bridge_record = bridge_module.diagnostics()
                if bridge_record is not None:
                    bridge_record = {
                        "epoch": epoch,
                        **bridge_record,
                    }
                    bridge_diagnostics.append(bridge_record)
            consensus_graph_record = None
            consensus_graph_module = getattr(
                model,
                "consensus_graph_branch",
                None,
            )
            if consensus_graph_module is not None:
                consensus_graph_record = (
                    consensus_graph_module.diagnostics()
                )
                if consensus_graph_record is not None:
                    consensus_graph_record = {
                        "epoch": epoch,
                        **consensus_graph_record,
                    }
                    consensus_graph_diagnostics.append(
                        consensus_graph_record
                    )
            train_predictions = (
                logits.index_select(0, train_index).argmax(dim=1)
            )
            train_oa = (
                train_predictions == train_labels
            ).float().mean().item()
            print(
                f"Run {run_index + 1}/{args.runs} | "
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"loss={loss.item():.6f} | "
                f"cls={classification_loss.item():.6f} | "
                f"cm={contrastive_loss.item():.6f} | "
                f"var={variance_loss.item():.6f} | "
                f"train_OA={train_oa:.4f}"
            )
            if transport_record is not None:
                hsi_std = np.asarray(
                    transport_record["hsi_projector_std"]
                )
                lidar_std = np.asarray(
                    transport_record["lidar_projector_std"]
                )
                print(
                    "  transport | semantic-progress="
                    f"{transport_record['semantic_progress']:.3f} | "
                    "node-prototype-cos="
                    f"{transport_record['mean_node_prototype_cosine']:.4f} | "
                    "row-error="
                    f"{transport_record['sinkhorn_row_max_error']:.2e} | "
                    "column-error="
                    f"{transport_record['sinkhorn_column_max_error']:.2e} | "
                    "entropy="
                    f"{transport_record['transport_entropy']:.4f} | "
                    "projector-std H/L mean="
                    f"{hsi_std.mean():.4f}/{lidar_std.mean():.4f}, "
                    "min="
                    f"{hsi_std.min():.4f}/{lidar_std.min():.4f}"
                )
            if prototype_fusion_record is not None:
                print(
                    "  SPSN correlation | selected-score H/L="
                    f"{prototype_fusion_record['hsi_selected_score_mean']:.4f}/"
                    f"{prototype_fusion_record['lidar_selected_score_mean']:.4f} | "
                    "reliability H/L="
                    f"{prototype_fusion_record['hsi_reliability_mean']:.4f}/"
                    f"{prototype_fusion_record['lidar_reliability_mean']:.4f}"
                )
            if consensus_record is not None:
                hsi_reliability = np.asarray(
                    consensus_record[
                        "hsi_anchor_reliability"
                    ]
                )
                lidar_reliability = np.asarray(
                    consensus_record[
                        "lidar_anchor_reliability"
                    ]
                )
                print(
                    "  consensus anchors | gamma H/L="
                    f"{consensus_record['hsi_gamma']:.5f}/"
                    f"{consensus_record['lidar_gamma']:.5f} | "
                    "anchor reliability H/L mean="
                    f"{hsi_reliability.mean():.4f}/"
                    f"{lidar_reliability.mean():.4f} | "
                    "reliability entropy="
                    f"{consensus_record['mean_reliability_entropy']:.4f} | "
                    "assignment entropy H/L="
                    f"{consensus_record['hsi_assignment_entropy']:.4f}/"
                    f"{consensus_record['lidar_assignment_entropy']:.4f} | "
                    "empty anchors H/L="
                    f"{consensus_record['hsi_empty_anchor_count']}/"
                    f"{consensus_record['lidar_empty_anchor_count']}"
                )
                if (
                    consensus_record.get("anchor_reasoning")
                    != "none"
                    and "anchor_graph_l1_gap"
                    in consensus_record
                ):
                    print(
                        "  SACR | eta="
                        f"{consensus_record['structure_eta']:.5f} | "
                        "H/L anchor graph entropy="
                        f"{consensus_record['hsi_anchor_graph_entropy']:.4f}/"
                        f"{consensus_record['lidar_anchor_graph_entropy']:.4f} | "
                        "gap="
                        f"{consensus_record['anchor_graph_l1_gap']:.4f} | "
                        "gate mean/min/max="
                        f"{consensus_record['structure_gate_mean']:.4f}/"
                        f"{consensus_record['structure_gate_min']:.4f}/"
                        f"{consensus_record['structure_gate_max']:.4f} | "
                        "residual-norm="
                        f"{consensus_record['structure_residual_norm']:.4f} | "
                        "message-norm H/L="
                        f"{consensus_record['hsi_message_norm']:.4f}/"
                        f"{consensus_record['lidar_message_norm']:.4f}"
                    )
            if bridge_record is not None:
                hsi_weights = np.asarray(
                    bridge_record["hsi_view_weight_mean"]
                )
                bridge_weights = np.asarray(
                    bridge_record["bridge_view_weight_mean"]
                )
                lidar_weights = np.asarray(
                    bridge_record["lidar_view_weight_mean"]
                )
                print(
                    "  center bridge | gamma H/C/L="
                    f"{bridge_record['h_gamma']:.5f}/"
                    f"{bridge_record['c_gamma']:.5f}/"
                    f"{bridge_record['l_gamma']:.5f} | "
                    "entropy HC/CH/LC/CL/CC/HLc/LHc="
                    f"{bridge_record['attention_hc_entropy']:.4f}/"
                    f"{bridge_record['attention_ch_entropy']:.4f}/"
                    f"{bridge_record['attention_lc_entropy']:.4f}/"
                    f"{bridge_record['attention_cl_entropy']:.4f}/"
                    f"{bridge_record['attention_cc_entropy']:.4f}/"
                    f"{bridge_record['attention_hl_via_c_entropy']:.4f}/"
                    f"{bridge_record['attention_lh_via_c_entropy']:.4f} | "
                    "view H/C/L="
                    f"{hsi_weights.round(3).tolist()}/"
                    f"{bridge_weights.round(3).tolist()}/"
                    f"{lidar_weights.round(3).tolist()}"
                )
            if consensus_graph_record is not None:
                bridge_weights = np.asarray(
                    consensus_graph_record["bridge_view_weight_mean"]
                )
                print(
                    "  mediator consensus graph | gamma C="
                    f"{consensus_graph_record['c_gamma']:.5f} | "
                    "entropy CH/CC/CL="
                    f"{consensus_graph_record['attention_ch_entropy']:.4f}/"
                    f"{consensus_graph_record['attention_cc_entropy']:.4f}/"
                    f"{consensus_graph_record['attention_cl_entropy']:.4f} | "
                    "view C="
                    f"{bridge_weights.round(3).tolist()} | "
                    "message-norm="
                    f"{consensus_graph_record['bridge_message_norm']:.4f} | "
                    "pixel-norm="
                    f"{consensus_graph_record['consensus_pixel_norm']:.4f}"
                )

    training_time = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        predictions = (
            forward_model()
            .index_select(0, test_index)
            .argmax(dim=1)
            .cpu()
            .numpy()
        )
    truth = test_labels.cpu().numpy()
    oa, aa, kappa, class_accuracy, _ = get_HSI_performance(
        truth,
        predictions,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = (
        f"{args.dataset}_{args.train_samples_per_class}px_"
        f"{STAGE}_{args.graph_layout}_"
        f"lidar-{args.lidar_segmentation}_"
        f"prior-{args.lidar_graph_prior}_"
        f"cross-{args.cross_modal_interaction}_"
        f"overlap-{args.overlap_metric}_"
        f"{contrastive_configuration_tag(args)}_"
        f"{prototype_fusion_configuration_tag(args)}_"
        f"{consensus_configuration_tag(args, class_count)}_"
        f"{bridge_configuration_tag(args, class_count)}_"
        f"{consensus_graph_configuration_tag(args, class_count)}_"
        f"{cell_configuration_tag(args)}_"
        f"fdsm-{args.fdsm_scope}_"
        f"cnn-{args.cnn_branch}_"
        f"lidarmod-{args.lidar_modulation}_"
        f"run{run_index + 1}"
    )
    checkpoint = args.output_dir / safe_output_filename(
        checkpoint_stem,
        ".pt",
    )
    torch.save(best_state, checkpoint)
    print(
        f"Run {run_index + 1}: OA={oa:.4f}, AA={aa:.4f}, "
        f"Kappa={kappa:.4f}, time={training_time:.2f}s"
    )
    return {
        "seed": run_seed,
        "OA": float(oa),
        "AA": float(aa),
        "Kappa": float(kappa),
        "class_accuracy": class_accuracy.tolist(),
        "training_time": training_time,
        "checkpoint": str(checkpoint),
        "transport_diagnostics": transport_diagnostics,
        "prototype_fusion_diagnostics": (
            prototype_fusion_diagnostics
        ),
        "consensus_diagnostics": consensus_diagnostics,
        "bridge_diagnostics": bridge_diagnostics,
        "consensus_graph_diagnostics": (
            consensus_graph_diagnostics
        ),
    }


def validate_args(args):
    if args.train_samples_per_class <= 0:
        raise ValueError("--train-samples-per-class must be positive.")
    if args.epochs <= 0 or args.runs <= 0:
        raise ValueError("--epochs and --runs must be positive.")
    if args.learning_rate <= 0 or args.pca_components <= 0:
        raise ValueError("Learning rate and PCA components must be positive.")
    if not args.scales or any(scale <= 0 for scale in args.scales):
        raise ValueError("--scales must contain positive integers.")
    if not 0.0 <= args.fusion_lambda <= 1.0:
        raise ValueError("--fusion-lambda must be between 0 and 1.")
    if not 0.0 <= args.graph_modality_lambda <= 1.0:
        raise ValueError(
            "--graph-modality-lambda must be between 0 and 1."
        )
    if args.dynamic_dk <= 0 or args.dynamic_topk <= 0:
        raise ValueError("Dynamic graph dimensions must be positive.")
    if args.dynamic_tau <= 0 or args.spatial_prior_k <= 0:
        raise ValueError("Dynamic tau and spatial prior k must be positive.")
    if args.lidar_height_knn_k <= 0:
        raise ValueError("--lidar-height-knn-k must be positive.")
    if args.cross_attention_dk <= 0:
        raise ValueError("--cross-attention-dk must be positive.")
    if args.contrastive_weight < 0:
        raise ValueError("--contrastive-weight must be nonnegative.")
    if args.contrastive_temperature <= 0:
        raise ValueError(
            "--contrastive-temperature must be positive."
        )
    if args.contrastive_dim <= 0:
        raise ValueError("--contrastive-dim must be positive.")
    if args.spsn_prototype_count <= 0:
        raise ValueError("--spsn-prototype-count must be positive.")
    if args.spsn_correlation_temperature <= 0:
        raise ValueError(
            "--spsn-correlation-temperature must be positive."
        )
    if args.consensus_anchor_count < 0:
        raise ValueError(
            "--consensus-anchor-count must be nonnegative."
        )
    if args.consensus_temperature <= 0:
        raise ValueError(
            "--consensus-temperature must be positive."
        )
    if args.consensus_gamma_init < 0:
        raise ValueError(
            "--consensus-gamma-init must be nonnegative."
        )
    if args.consensus_reliability_temperature <= 0:
        raise ValueError(
            "--consensus-reliability-temperature must be positive."
        )
    if args.consensus_anchor_graph_topk <= 0:
        raise ValueError(
            "--consensus-anchor-graph-topk must be positive."
        )
    if args.consensus_structure_temperature <= 0:
        raise ValueError(
            "--consensus-structure-temperature must be positive."
        )
    if args.consensus_structure_eta_init < 0:
        raise ValueError(
            "--consensus-structure-eta-init must be nonnegative."
        )
    if args.bridge_anchor_count < 0:
        raise ValueError("--bridge-anchor-count must be nonnegative.")
    if args.bridge_attention_dk <= 0:
        raise ValueError("--bridge-attention-dk must be positive.")
    if args.bridge_attention_topk <= 0:
        raise ValueError("--bridge-attention-topk must be positive.")
    if args.bridge_gamma_init < 0:
        raise ValueError("--bridge-gamma-init must be nonnegative.")
    if any(
        weight < 0
        for weight in (
            args.bridge_overlap_weight,
            args.bridge_spatial_weight,
            args.bridge_height_weight,
        )
    ):
        raise ValueError("Bridge bias weights must be nonnegative.")
    if args.transport_semantic_weight < 0:
        raise ValueError(
            "--transport-semantic-weight must be nonnegative."
        )
    if args.transport_iterations <= 0:
        raise ValueError("--transport-iterations must be positive.")
    if args.transport_warmup_epochs < 0:
        raise ValueError(
            "--transport-warmup-epochs must be nonnegative."
        )
    if args.variance_weight < 0:
        raise ValueError("--variance-weight must be nonnegative.")
    if args.variance_target <= 0:
        raise ValueError("--variance-target must be positive.")
    if (
        args.graph_layout == "joint"
        and args.contrastive_mode != "none"
    ):
        raise ValueError(
            "--contrastive-mode requires --graph-layout separate."
        )
    if (
        args.contrastive_mode == "overlap-transport"
        and len(args.scales) != 1
    ):
        raise ValueError(
            "--contrastive-mode overlap-transport requires exactly "
            "one superpixel scale so Q_H and Q_L are partitions."
        )
    if (
        args.graph_layout == "joint"
        and args.cross_modal_interaction != "none"
    ):
        raise ValueError(
            "--cross-modal-interaction requires "
            "--graph-layout separate."
        )
    if (
        args.graph_layout == "joint"
        and args.post_gat_prototype_fusion != "none"
    ):
        raise ValueError(
            "--post-gat-prototype-fusion requires "
            "--graph-layout separate."
        )
    if (
        args.graph_layout == "joint"
        and args.post_gat_consensus != "none"
    ):
        raise ValueError(
            "--post-gat-consensus requires "
            "--graph-layout separate."
        )
    if args.post_gat_consensus != "none":
        if (
            args.consensus_writeback == "difference"
            and args.consensus_fusion != "adaptive"
        ):
            raise ValueError(
                "--consensus-writeback difference requires "
                "--consensus-fusion adaptive."
            )
        if args.consensus_anchor_reasoning == "sacr":
            if args.consensus_fusion != "adaptive":
                raise ValueError(
                    "--consensus-anchor-reasoning sacr requires "
                    "--consensus-fusion adaptive."
                )
            if args.consensus_writeback != "difference":
                raise ValueError(
                    "--consensus-anchor-reasoning sacr requires "
                    "--consensus-writeback difference."
                )
        elif args.consensus_structure_reliability != "none":
            raise ValueError(
                "--consensus-structure-reliability requires "
                "--consensus-anchor-reasoning sacr."
            )
        if args.cross_modal_interaction != "none":
            raise ValueError(
                "--post-gat-consensus requires "
                "--cross-modal-interaction none so both GAT stages "
                "remain modality-private."
            )
        if args.contrastive_mode != "none":
            raise ValueError(
                "--post-gat-consensus first ablation requires "
                "--contrastive-mode none."
            )
        if args.cell_interaction != "none":
            raise ValueError(
                "--post-gat-consensus first ablation requires "
                "--cell-interaction none."
            )
        if args.post_gat_prototype_fusion != "none":
            raise ValueError(
                "--post-gat-consensus and "
                "--post-gat-prototype-fusion are alternative "
                "single-interaction ablations."
            )
    if (
        args.graph_layout == "joint"
        and args.post_gat_bridge != "none"
    ):
        raise ValueError(
            "--post-gat-bridge requires --graph-layout separate."
        )
    if (
        args.graph_layout == "joint"
        and args.post_gat_consensus_graph != "none"
    ):
        raise ValueError(
            "--post-gat-consensus-graph requires "
            "--graph-layout separate."
        )
    if args.post_gat_bridge != "none":
        if args.cross_modal_interaction != "none":
            raise ValueError(
                "--post-gat-bridge requires "
                "--cross-modal-interaction none so bridge is the "
                "only cross-modal graph interaction."
            )
        if args.post_gat_consensus != "none":
            raise ValueError(
                "--post-gat-bridge and --post-gat-consensus are "
                "alternative post-GAT2 interactions."
            )
        if args.post_gat_prototype_fusion != "none":
            raise ValueError(
                "--post-gat-bridge and --post-gat-prototype-fusion "
                "are alternative post-GAT2 interactions."
            )
        if args.contrastive_mode != "none":
            raise ValueError(
                "--post-gat-bridge first ablation requires "
                "--contrastive-mode none."
            )
        if args.cell_interaction != "none":
            raise ValueError(
                "--post-gat-bridge first ablation requires "
                "--cell-interaction none."
            )
    if not 0.0 <= args.consensus_graph_weight <= 1.0:
        raise ValueError(
            "--consensus-graph-weight must be between 0 and 1."
        )
    if args.post_gat_consensus_graph != "none":
        if args.cross_modal_interaction != "none":
            raise ValueError(
                "--post-gat-consensus-graph requires "
                "--cross-modal-interaction none so both private "
                "GAT stages remain modality-only."
            )
        if args.post_gat_consensus != "none":
            raise ValueError(
                "--post-gat-consensus-graph and "
                "--post-gat-consensus are alternative post-GAT2 "
                "fusion mechanisms."
            )
        if args.post_gat_bridge != "none":
            raise ValueError(
                "--post-gat-consensus-graph center-mediator reuses "
                "center bridge priors internally; do not also enable "
                "--post-gat-bridge."
            )
        if args.post_gat_prototype_fusion != "none":
            raise ValueError(
                "--post-gat-consensus-graph and "
                "--post-gat-prototype-fusion are alternative "
                "post-GAT2 fusion mechanisms."
            )
        if args.contrastive_mode != "none":
            raise ValueError(
                "--post-gat-consensus-graph first ablation requires "
                "--contrastive-mode none."
            )
        if args.cell_interaction != "none":
            raise ValueError(
                "--post-gat-consensus-graph first ablation requires "
                "--cell-interaction none."
            )
    if (
        args.graph_layout == "joint"
        and args.cnn_branch != "original"
    ):
        raise ValueError(
            "--cnn-branch gsdg currently requires "
            "--graph-layout separate."
        )
    if args.graph_layout == "joint" and args.fdsm_scope != "none":
        raise ValueError(
            "--fdsm-scope hsi requires --graph-layout separate."
        )
    if (
        args.graph_layout == "joint"
        and args.lidar_modulation != "none"
    ):
        raise ValueError(
            "--lidar-modulation requires --graph-layout separate."
        )
    if (
        args.graph_layout == "joint"
        and args.cell_interaction != "none"
    ):
        raise ValueError(
            "--cell-interaction requires --graph-layout separate."
        )
    if (
        args.cell_interaction != "none"
        and len(args.scales) != 1
    ):
        raise ValueError(
            "--cell-interaction rag currently requires exactly one "
            "superpixel scale."
        )
    cell_enhancement_requested = (
        args.cell_pixel_descriptor != "none"
        or args.cell_edge_mode != "binary"
        or args.cell_interaction_stages != 1
        or args.cell_output_branch != "none"
        or args.cell_conflict_weight > 0
        or args.cell_topology_veto != "none"
    )
    if (
        cell_enhancement_requested
        and args.cell_interaction != "rag"
    ):
        raise ValueError(
            "Cell descriptor, weighted edges, two-stage interaction, "
            "and cell output require --cell-interaction rag."
        )
    if not 0.0 <= args.cell_output_weight <= 1.0:
        raise ValueError("--cell-output-weight must be between 0 and 1.")
    if (
        args.cell_conflict_weight > 0
        and args.cell_edge_mode
        != "spectral-height-boundary"
    ):
        raise ValueError(
            "--cell-conflict-weight requires "
            "--cell-edge-mode spectral-height-boundary."
        )
    if not 0.0 < args.cell_veto_threshold <= 1.0:
        raise ValueError(
            "--cell-veto-threshold must be in (0, 1]."
        )
    if (
        args.cell_topology_veto != "none"
        and args.cross_modal_interaction
        != "overlap-qk-condition"
    ):
        raise ValueError(
            "--cell-topology-veto currently requires "
            "--cross-modal-interaction overlap-qk-condition."
        )
    if any(
        weight < 0
        for weight in (
            args.cell_sam_weight,
            args.cell_height_weight,
            args.cell_boundary_weight,
            args.cell_conflict_weight,
        )
    ):
        raise ValueError("Cell edge weights must be nonnegative.")
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")


def main():
    args = parse_args()
    config = resolve_options(args)
    validate_args(args)
    set_seed(args.seed)
    (
        hsi_features,
        lidar_features,
        joint_input,
        gt,
        class_count,
        hsi_assignment,
        lidar_assignment,
        joint_assignment,
        hsi_spatial_prior,
        lidar_spatial_prior,
        lidar_candidate_mask,
        lidar_rag_adjacency,
        lidar_geometry_descriptors,
        cell_data,
        bridge_data,
        joint_spatial_prior,
    ) = prepare_data(args, config)

    print("=" * 72)
    print(
        "demo_train | Stage 8: overlap-conditioned second-layer Q/K"
    )
    if args.cnn_branch == "original":
        print(
            "CNN branch: original joint PCA(HSI)+LiDAR -> WMF -> "
            "HGCN-HL 5x5/5x5 SSConv (default)"
        )
    else:
        print(
            "CNN branch: joint PCA(HSI)+LiDAR -> GSDG stem -> "
            f"equal-width {args.hidden_dim}->"
            f"{args.hidden_dim}->{args.hidden_dim} "
            "3x3/7x7 DwsConv"
        )
    if args.graph_layout == "separate":
        print(
            "Graph layout: independent HSI-SLIC and "
            f"LiDAR-{args.lidar_segmentation} GSDG graphs"
        )
        if args.post_gat_prototype_fusion == "none":
            print(
                "Graph fusion after pixel projection: "
                f"HSI={args.graph_modality_lambda:g}, "
                f"LiDAR={1.0 - args.graph_modality_lambda:g}"
            )
        else:
            print(
                "Graph fusion after pixel projection: adaptive "
                "pixel-wise HSI/LiDAR reliability, initialized at "
                f"{args.graph_modality_lambda:g}/"
                f"{1.0 - args.graph_modality_lambda:g}"
            )
        if args.lidar_graph_prior == "rag-height-knn":
            print(
                "LiDAR prior: hard "
                f"{args.lidar_rag_hops}-hop RAG + local elevation "
                f"{args.lidar_height_knn_k}-NN + weighted GAT"
            )
        else:
            print("LiDAR prior: Stage-3 centroid KNN")
        print(
            "Cross-modal graph interaction: "
            f"{args.cross_modal_interaction}"
        )
        if args.cross_modal_interaction != "none":
            print(
                "Cross-modal overlap metric: "
                f"{args.overlap_metric}"
            )
        if (
            args.cross_modal_interaction
            == "overlap-qk-condition"
        ):
            print(
                "Cross condition: C_HL @ L and C_LH @ H modify "
                "both modalities' second-layer Q/K; no direct "
                "overlap feature gate"
            )
        print(
            "Post-GAT2 cross-modal contrastive loss: "
            f"{args.contrastive_mode}"
        )
        if args.contrastive_mode != "none":
            contrastive_target = {
                "overlap-soft": "soft overlap distribution",
                "overlap-prototype": (
                    "opposite-modal overlap prototype"
                ),
                "overlap-transport": (
                    "overlap-supported semantic transport prototype"
                ),
            }[args.contrastive_mode]
            prototype_detail = (
                f", prototype objective={args.prototype_objective}"
                if args.contrastive_mode == "overlap-prototype"
                else ""
            )
            transport_detail = (
                ", semantic beta="
                f"{args.transport_semantic_weight:g}, Sinkhorn "
                f"iterations={args.transport_iterations}, fixed-overlap "
                f"warmup={args.transport_warmup_epochs} epochs, "
                "linear semantic ramp="
                f"{args.transport_warmup_epochs} epochs, variance "
                f"lambda={args.variance_weight:g}, gamma="
                f"{args.variance_target:g}"
                if args.contrastive_mode == "overlap-transport"
                else ""
            )
            temperature_detail = (
                ""
                if (
                    args.contrastive_mode == "overlap-prototype"
                    and args.prototype_objective == "cosine"
                )
                else f", tau={args.contrastive_temperature:g}"
            )
            print(
                "Contrastive configuration: lambda="
                f"{args.contrastive_weight:g}"
                f"{temperature_detail}, projection dim="
                f"{args.contrastive_dim}; target="
                f"{contrastive_target}{prototype_detail}"
                f"{transport_detail}"
            )
        print(
            "Post-GAT2 prototype correlation fusion: "
            f"{args.post_gat_prototype_fusion}"
        )
        if (
            args.post_gat_prototype_fusion
            == "spsn-correlation"
        ):
            print(
                "SPSN-style path: independent HSI/LiDAR GAT2 "
                f"prototype Top-{args.spsn_prototype_count} -> "
                "node correlation -> sparse pixel projection -> "
                "pixel-wise reliability -> unchanged CNN fusion; "
                "correlation tau="
                f"{args.spsn_correlation_temperature:g}"
            )
        print(
            "Post-GAT2 consensus interaction: "
            f"{args.post_gat_consensus}"
        )
        if args.post_gat_consensus == "mssagf-anchor":
            resolved_anchor_count = (
                args.consensus_anchor_count
                if args.consensus_anchor_count > 0
                else 2 * class_count
            )
            print(
                "Consensus path: pure modality-private GAT1/GAT2 "
                f"-> {resolved_anchor_count} shared anchors -> "
                "area-weighted "
                f"{args.consensus_fusion} anchor fusion -> "
                f"{args.consensus_writeback} "
                "zero-initialized learnable residual write-back -> "
                "separate pixel projection; tau="
                f"{args.consensus_temperature:g}, gamma-init="
                f"{args.consensus_gamma_init:g}, reliability-tau="
                f"{args.consensus_reliability_temperature:g}"
            )
            if args.consensus_anchor_reasoning != "none":
                print(
                    "Consensus anchor reasoning: "
                    f"{args.consensus_anchor_reasoning}, "
                    "structure reliability="
                    f"{args.consensus_structure_reliability}, "
                    "anchor top-k="
                    f"{args.consensus_anchor_graph_topk}, "
                    "structure temperature="
                    f"{args.consensus_structure_temperature:g}, "
                    "eta-init="
                    f"{args.consensus_structure_eta_init:g}"
                )
        print(
            "Post-GAT2 center bridge interaction: "
            f"{args.post_gat_bridge}"
        )
        if args.post_gat_bridge == "center-block":
            resolved_bridge_count = (
                args.bridge_anchor_count
                if args.bridge_anchor_count > 0
                else 2 * class_count
            )
            print(
                "Center bridge path: HSI/LiDAR GAT2 nodes -> "
                f"{resolved_bridge_count} public spatial bridge "
                "anchors -> H<->C<->L block attention -> separate "
                "pixel projection; d_k="
                f"{args.bridge_attention_dk}, top-k="
                f"{args.bridge_attention_topk}, overlap="
                f"{args.bridge_overlap_metric}, bias weights "
                "overlap/spatial/height="
                f"{args.bridge_overlap_weight:g}/"
                f"{args.bridge_spatial_weight:g}/"
                f"{args.bridge_height_weight:g}, gamma-init="
                f"{args.bridge_gamma_init:g}"
            )
        print(
            "Post-GAT2 mediator consensus graph: "
            f"{args.post_gat_consensus_graph}"
        )
        if args.post_gat_consensus_graph == "center-mediator":
            resolved_bridge_count = (
                args.bridge_anchor_count
                if args.bridge_anchor_count > 0
                else 2 * class_count
            )
            private_weight = 1.0 - args.consensus_graph_weight
            print(
                "Mediator graph path: HSI/LiDAR private GAT2 nodes "
                f"stay unchanged -> {resolved_bridge_count} public "
                "center anchors -> C receives H/C/L block attention "
                "-> consensus pixels as third graph branch; fusion "
                "weights H/L/C="
                f"{private_weight * args.graph_modality_lambda:g}/"
                f"{private_weight * (1.0 - args.graph_modality_lambda):g}/"
                f"{args.consensus_graph_weight:g}; d_k="
                f"{args.bridge_attention_dk}, top-k="
                f"{args.bridge_attention_topk}, overlap="
                f"{args.bridge_overlap_metric}, bias weights "
                "overlap/spatial/height="
                f"{args.bridge_overlap_weight:g}/"
                f"{args.bridge_spatial_weight:g}/"
                f"{args.bridge_height_weight:g}, gamma-init="
                f"{args.bridge_gamma_init:g}"
            )
        print(f"Intersection-cell interaction: {args.cell_interaction}")
        if args.cell_interaction == "rag":
            print(
                "Cell path: parent-to-cell -> sparse 1-hop RAG-GCN "
                "-> gated cell-to-parent; cells="
                f"{cell_data['cell_count']}, directed RAG entries "
                "excluding self="
                f"{cell_data['cell_rag_edge_count']}"
            )
            print(
                "Cell enhancements: pixel descriptor="
                f"{args.cell_pixel_descriptor}, edge mode="
                f"{args.cell_edge_mode}, interaction stages="
                f"{args.cell_interaction_stages}, pixel output="
                f"{args.cell_output_branch}"
            )
            print(
                "Cell topology veto: "
                f"{args.cell_topology_veto}"
            )
            if args.cell_topology_veto != "none":
                hsi_support = cell_data[
                    "hsi_topology_support"
                ]
                lidar_support = cell_data[
                    "lidar_topology_support"
                ]
                print(
                    "Mapped cell support retained at threshold "
                    f"{args.cell_veto_threshold:g}: HSI="
                    f"{np.mean(hsi_support >= args.cell_veto_threshold):.4f}, "
                    "LiDAR="
                    f"{np.mean(lidar_support >= args.cell_veto_threshold):.4f}"
                )
            if (
                args.cell_edge_mode
                == "spectral-height-boundary"
            ):
                weight_stats = cell_data["cell_weight_stats"]
                print(
                    "Cell multimodal edge weights: "
                    f"min={weight_stats['minimum']:.4f}, "
                    f"mean={weight_stats['mean']:.4f}, "
                    f"max={weight_stats['maximum']:.4f}"
                )
                if args.cell_conflict_weight > 0:
                    conflict_stats = cell_data[
                        "cell_boundary_conflict_stats"
                    ]
                    print(
                        "HSI/LiDAR boundary conflict: "
                        f"weight={args.cell_conflict_weight:g}, "
                        f"min={conflict_stats['minimum']:.4f}, "
                        f"mean={conflict_stats['mean']:.4f}, "
                        f"max={conflict_stats['maximum']:.4f}"
                    )
            if args.cell_output_branch == "fixed":
                print(
                    "Cell third pixel branch weight: "
                    f"{args.cell_output_weight:g}"
                )
        print(f"HSI FDSM: {args.fdsm_scope}")
        print(f"LiDAR modulation: {args.lidar_modulation}")
    else:
        print(
            "Graph layout: retained concatenated-node joint graph "
            f"(HSI-SLIC + LiDAR-{args.lidar_segmentation})"
        )
    print("Not enabled: fixed-incidence hypergraph")
    print("=" * 72)
    print(
        f"Dataset: {config['loader_name']} | "
        f"joint-input={joint_input.shape} | "
        f"HSI-nodes={hsi_assignment.shape[1]} | "
        f"LiDAR-nodes={lidar_assignment.shape[1]}"
    )
    print(
        f"Dynamic graph: d_k={args.dynamic_dk}, "
        f"top-k={args.dynamic_topk}, tau={args.dynamic_tau}, "
        f"spatial-prior-k={args.spatial_prior_k}"
    )
    print(
        f"Split: exactly {args.train_samples_per_class} "
        "training pixels per class"
    )

    results = [
        train_one_run(
            args,
            hsi_features,
            lidar_features,
            joint_input,
            gt,
            class_count,
            hsi_assignment,
            lidar_assignment,
            joint_assignment,
            hsi_spatial_prior,
            lidar_spatial_prior,
            lidar_candidate_mask,
            lidar_rag_adjacency,
            lidar_geometry_descriptors,
            cell_data,
            bridge_data,
            joint_spatial_prior,
            run_index,
        )
        for run_index in range(args.runs)
    ]
    summary = {}
    for metric_name in ("OA", "AA", "Kappa"):
        values = np.asarray(
            [result[metric_name] for result in results],
            dtype=np.float64,
        )
        summary[metric_name] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    output = {
        "stage": STAGE,
        "config": {
            **vars(args),
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
        },
        "runs": results,
        "summary": summary,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result_stem = (
        f"{args.dataset}_{args.train_samples_per_class}px_"
        f"{STAGE}_{args.graph_layout}_"
        f"lidar-{args.lidar_segmentation}_"
        f"prior-{args.lidar_graph_prior}_"
        f"cross-{args.cross_modal_interaction}_"
        f"overlap-{args.overlap_metric}_"
        f"{contrastive_configuration_tag(args)}_"
        f"{prototype_fusion_configuration_tag(args)}_"
        f"{consensus_configuration_tag(args, class_count)}_"
        f"{bridge_configuration_tag(args, class_count)}_"
        f"{consensus_graph_configuration_tag(args, class_count)}_"
        f"{cell_configuration_tag(args)}_"
        f"fdsm-{args.fdsm_scope}_"
        f"cnn-{args.cnn_branch}_"
        f"lidarmod-{args.lidar_modulation}_results"
    )
    result_path = args.output_dir / safe_output_filename(
        result_stem,
        ".json",
    )
    result_path.write_text(
        json.dumps(output, indent=2),
        encoding="utf-8",
    )
    print("Summary")
    for metric_name, values in summary.items():
        print(
            f"{metric_name}: {values['mean']:.4f} "
            f"± {values['std']:.4f}"
        )
    print(f"Results saved to: {result_path.resolve()}")


if __name__ == "__main__":
    main()
