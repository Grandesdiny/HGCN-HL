"""Stage-8 demo: private dual GSDG graphs plus mediator C-GAT.

The fixed hypergraph/HGCN path is replaced by GSDG graph/GAT propagation.
The default uses independent HSI and LiDAR graphs; the previous concatenated
node graph remains selectable. The LiDAR graph can additionally restrict its
dynamic neighbors with a local RAG and an elevation-similarity KNN. The
original joint CNN and fusion remain. The only exposed cross-modal graph
branch is a post-GAT2 intersection-cell mediator C graph with its own C-GAT
message passing and pixel readout.
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


STAGE = "stage8_intersection_mediator_cgat"


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
    # Archived ablation switches are kept as hidden compatibility flags
    # but their active choices are removed from the main entry point.
    parser.add_argument(
        "--post-gat-consensus-graph",
        choices=("none", "intersection-mediator"),
        default="none",
        help=(
            "Optional post-GAT2 mediator consensus graph as an "
            "independent third pixel branch. 'intersection-mediator' "
            "uses HSI-superpixel ∩ LiDAR-superpixel cells. "
            "Default: none."
        ),
    )
    parser.add_argument(
        "--consensus-graph-weight",
        type=float,
        default=0.1,
        help=(
            "Pixel-fusion weight of the mediator consensus graph "
            "branch. The remaining weight is split between HSI and "
            "LiDAR according to --graph-modality-lambda. Default: "
            "0.1."
        ),
    )
    parser.add_argument(
        "--consensus-graph-fusion",
        choices=("fixed", "c-guided-gate", "residual-c"),
        default="residual-c",
        help=(
            "Fuse HSI/LiDAR/mediator graph pixels with fixed weights "
            "or a C-guided pixel-wise tri-graph gate. 'residual-c' "
            "starts exactly from the private HSI/LiDAR baseline. "
            "Default: residual-c."
        ),
    )
    parser.add_argument(
        "--consensus-graph-residual-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scale for residual-c fusion. Zero makes "
            "the initial forward pass exactly match private HSI/LiDAR "
            "graph fusion. Default: 0."
        ),
    )
    parser.add_argument(
        "--consensus-graph-c-gamma-init",
        type=float,
        default=0.1,
        help=(
            "Initial residual scale of the mediator C-GAT branch "
            "inside the C graph. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--consensus-graph-transport",
        choices=("none", "bidirectional"),
        default="none",
        help=(
            "Optional post-GAT2 C-mediated bidirectional transport. "
            "When set to 'bidirectional', HSI/LiDAR nodes exchange "
            "messages through the intersection C graph before pixel "
            "projection, and the third pixel C branch is not fused. "
            "Default: none."
        ),
    )
    parser.add_argument(
        "--consensus-graph-transport-fusion",
        choices=("residual", "tri-gate", "concat", "bilinear"),
        default="residual",
        help=(
            "Fusion rule inside bidirectional C-mediated transport. "
            "'residual' keeps the existing two-source residual update; "
            "'tri-gate' mixes node state, private intra-graph message, "
            "and C-mediated inter-modal message with a softmax gate; "
            "'concat' fuses the same sources with an MLP; "
            "'bilinear' adds low-rank multiplicative evidence before "
            "the MLP. "
            "Default: residual."
        ),
    )
    parser.add_argument(
        "--consensus-graph-transport-message",
        choices=("fixed", "qk-prior"),
        default="fixed",
        help=(
            "How to build C-mediated inter-modal transport messages. "
            "'fixed' uses Beta V directly; 'qk-prior' uses cross-modal "
            "Q/K attention regularized and hard-supported by Beta. "
            "Default: fixed."
        ),
    )
    parser.add_argument(
        "--consensus-graph-transport-state",
        choices=("topology-only", "cgnn-reliability"),
        default="topology-only",
        help=(
            "How the mediator C graph controls bidirectional transport. "
            "'topology-only' preserves the current implementation. "
            "'cgnn-reliability' runs C-GAT first and uses the propagated "
            "C state to modulate the C transport kernel. "
            "Default: topology-only."
        ),
    )
    parser.add_argument(
        "--consensus-graph-transport-prior-weight",
        type=float,
        default=1.0,
        help=(
            "Eta multiplying log(Beta) in qk-prior transport message "
            "attention. Default: 1."
        ),
    )
    parser.add_argument(
        "--consensus-graph-transport-lambda",
        type=float,
        default=0.5,
        help=(
            "Lambda in T_C=(1-lambda)I+lambda A_C for C-mediated "
            "transport. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--consensus-graph-transport-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scale for both HSI and LiDAR "
            "C-mediated transport updates. Default: 0."
        ),
    )
    parser.add_argument(
        "--consensus-graph-cell-edge",
        choices=("binary", "spectral-height-boundary"),
        default="binary",
        help=(
            "Cell RAG prior used by intersection-mediator. "
            "'spectral-height-boundary' uses HSI SAM, LiDAR height "
            "difference, LiDAR boundary gradient, and HSI/LiDAR "
            "boundary conflict. Default: binary."
        ),
    )
    parser.add_argument(
        "--consensus-graph-spatial-prior-weight",
        type=float,
        default=1.0,
        help=(
            "Alpha_s for the mediator C-QK spatial prior. Default: 1."
        ),
    )
    parser.add_argument(
        "--consensus-graph-hsi-prior-weight",
        type=float,
        default=0.5,
        help=(
            "Alpha_h for the HSI private-graph prior projected into "
            "mediator C space. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--consensus-graph-lidar-prior-weight",
        type=float,
        default=0.5,
        help=(
            "Alpha_l for the LiDAR private-graph prior projected into "
            "mediator C space. Default: 0.5."
        ),
    )
    parser.add_argument(
        "--bridge-attention-dk",
        type=int,
        default=32,
        help=(
            "Query/key dimension of the mediator C graph attention. "
            "Default: 32."
        ),
    )
    parser.add_argument(
        "--bridge-attention-topk",
        type=int,
        default=8,
        help="Top-K entries per mediator C graph row. Default: 8.",
    )
    parser.add_argument(
        "--cell-sam-weight",
        type=float,
        default=1.0,
        help=(
            "HSI spectral-angle weight for the intersection-mediator "
            "spectral-height-boundary C-C prior. Default: 1."
        ),
    )
    parser.add_argument(
        "--cell-height-weight",
        type=float,
        default=1.0,
        help=(
            "LiDAR mean-height difference weight for the "
            "intersection-mediator spectral-height-boundary C-C prior. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--cell-boundary-weight",
        type=float,
        default=1.0,
        help=(
            "LiDAR boundary-gradient weight for the "
            "intersection-mediator spectral-height-boundary C-C prior. "
            "Default: 1."
        ),
    )
    parser.add_argument(
        "--cell-conflict-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of the absolute normalized HSI/LiDAR shared-boundary "
            "strength disagreement in the intersection-mediator "
            "spectral-height-boundary C-C prior. Default: 0 (disabled)."
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












def consensus_graph_configuration_tag(args, class_count):
    if args.post_gat_consensus_graph == "none":
        return "cg-none"
    anchor_tag = "cells"
    return (
        f"cg-{args.post_gat_consensus_graph}-k{anchor_tag}-"
        f"dk{args.bridge_attention_dk}-"
        f"top{args.bridge_attention_topk}-"
        f"w{args.consensus_graph_weight:g}-"
        f"f{args.consensus_graph_fusion}-"
        f"rg{args.consensus_graph_residual_init:g}-"
        f"cg{args.consensus_graph_c_gamma_init:g}-"
        f"tp{args.consensus_graph_transport}-"
        f"tf{args.consensus_graph_transport_fusion}-"
        f"tm{args.consensus_graph_transport_message}-"
        f"ts{args.consensus_graph_transport_state}-"
        f"tw{args.consensus_graph_transport_prior_weight:g}-"
        f"tl{args.consensus_graph_transport_lambda:g}-"
        f"tg{args.consensus_graph_transport_gamma_init:g}-"
        f"edge{args.consensus_graph_cell_edge}-"
        f"a{args.consensus_graph_spatial_prior_weight:g}-"
        f"{args.consensus_graph_hsi_prior_weight:g}-"
        f"{args.consensus_graph_lidar_prior_weight:g}"
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


def build_intersection_mediator_data(
    cell_data,
    hsi_node_count,
    lidar_node_count,
    edge_mode="binary",
):
    """Build mediator matrices from HSI/LiDAR common-refinement cells."""
    cell_count = int(cell_data["cell_count"])
    cell_indices = np.arange(cell_count, dtype=np.int64)
    hsi_parent = np.asarray(
        cell_data["hsi_parent"],
        dtype=np.int64,
    )
    lidar_parent = np.asarray(
        cell_data["lidar_parent"],
        dtype=np.int64,
    )
    hsi_coverage = np.asarray(
        cell_data["hsi_coverage"],
        dtype=np.float32,
    )
    lidar_coverage = np.asarray(
        cell_data["lidar_coverage"],
        dtype=np.float32,
    )

    prior_ch = coo_matrix(
        (
            np.ones(cell_count, dtype=np.float32),
            (cell_indices, hsi_parent),
        ),
        shape=(cell_count, hsi_node_count),
        dtype=np.float32,
    ).toarray()
    prior_cl = coo_matrix(
        (
            np.ones(cell_count, dtype=np.float32),
            (cell_indices, lidar_parent),
        ),
        shape=(cell_count, lidar_node_count),
        dtype=np.float32,
    ).toarray()
    prior_hc = coo_matrix(
        (
            hsi_coverage,
            (hsi_parent, cell_indices),
        ),
        shape=(hsi_node_count, cell_count),
        dtype=np.float32,
    ).toarray()
    prior_lc = coo_matrix(
        (
            lidar_coverage,
            (lidar_parent, cell_indices),
        ),
        shape=(lidar_node_count, cell_count),
        dtype=np.float32,
    ).toarray()

    pixel_cell_index = np.asarray(
        cell_data["pixel_cell_index"],
        dtype=np.int64,
    )
    pixel_indices = np.arange(pixel_cell_index.size, dtype=np.int64)
    assignment = coo_matrix(
        (
            np.ones(pixel_cell_index.size, dtype=np.float32),
            (pixel_indices, pixel_cell_index),
        ),
        shape=(pixel_cell_index.size, cell_count),
        dtype=np.float32,
    ).tocsr()

    adjacency_key = (
        "cell_weighted_adjacency"
        if edge_mode == "spectral-height-boundary"
        else "cell_rag_adjacency"
    )
    if adjacency_key not in cell_data:
        raise ValueError(
            f"Missing {adjacency_key} for intersection mediator."
        )
    cc_prior = cell_data[adjacency_key].tocsr().astype(np.float32)
    cc_prior = cc_prior.copy()
    cc_prior.setdiag(1.0)
    cc_prior.eliminate_zeros()
    row_sum = np.asarray(cc_prior.sum(axis=1)).reshape(-1)
    cc_prior = cc_prior.tocoo()
    cc_values = cc_prior.data / np.maximum(row_sum[cc_prior.row], 1e-6)
    cc_prior = coo_matrix(
        (cc_values.astype(np.float32), (cc_prior.row, cc_prior.col)),
        shape=(cell_count, cell_count),
        dtype=np.float32,
    ).toarray()
    return {
        "assignment": assignment.astype(np.float32),
        "prior_hc": prior_hc.astype(np.float32),
        "prior_ch": prior_ch.astype(np.float32),
        "prior_lc": prior_lc.astype(np.float32),
        "prior_cl": prior_cl.astype(np.float32),
        "cc_prior": cc_prior.astype(np.float32),
        "attributes": np.asarray(
            cell_data["attributes"],
            dtype=np.float32,
        ),
        "anchor_count": cell_count,
        "area": np.asarray(
            cell_data["cell_area"],
            dtype=np.float32,
        ),
        "mediator_kind": "intersection",
        "edge_mode": edge_mode,
    }


















class PostGATMediatedConsensusGraph(nn.Module):
    """Independent mediator graph branch over public center anchors.

    The module consumes post-GAT2 HSI/LiDAR private graph nodes, builds
    public center anchors from both modalities, builds a C-C dynamic graph
    with C's own Q/K and HSI/LiDAR projected topology priors, then projects
    the updated mediator graph directly back to pixels. It intentionally
    does not write messages back to HSI or LiDAR superpixel nodes.
    """

    def __init__(
        self,
        channels,
        bridge_data,
        attention_d_k=32,
        topk=8,
        gamma_init=0.0,
        transport_lambda=0.5,
        transport_fusion="residual",
        transport_message="fixed",
        transport_state="topology-only",
        transport_prior_weight=1.0,
        transport_gamma_init=0.0,
        spatial_prior_weight=1.0,
        hsi_prior_weight=0.5,
        lidar_prior_weight=0.5,
    ):
        super().__init__()
        self.channels = channels
        self.attention_d_k = attention_d_k
        self.topk = topk
        self.spatial_prior_weight = spatial_prior_weight
        self.hsi_prior_weight = hsi_prior_weight
        self.lidar_prior_weight = lidar_prior_weight
        self.transport_lambda = transport_lambda
        self.transport_fusion = transport_fusion
        self.transport_message = transport_message
        self.transport_state = transport_state
        self.transport_prior_weight = transport_prior_weight
        anchor_count = int(bridge_data["anchor_count"])
        self.bridge_embedding = nn.Parameter(
            torch.empty(anchor_count, channels)
        )
        nn.init.xavier_uniform_(self.bridge_embedding)
        self.bridge_norm = nn.LayerNorm(channels)
        attribute_channels = (
            bridge_data["attributes"].shape[1]
            if "attributes" in bridge_data
            else 0
        )
        self.cell_encoder = nn.Sequential(
            nn.Linear(5 * channels + attribute_channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
        )

        self.h_value = nn.Linear(channels, channels, bias=False)
        self.c_query = nn.Linear(channels, attention_d_k, bias=False)
        self.c_key = nn.Linear(channels, attention_d_k, bias=False)
        self.l_value = nn.Linear(channels, channels, bias=False)
        self.scale = attention_d_k ** -0.5

        self.c_gat = MultiHeadGAT(
            channels,
            head_channels=60,
            out_channels=channels,
            dropout=0.2,
            heads=4,
            alpha=0.2,
            use_edge_weights=True,
        )
        self.c_gat_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.c_graph_norm = nn.LayerNorm(channels)
        self.c_ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.LeakyReLU(),
            nn.Linear(2 * channels, channels),
        )
        self.c_ffn_norm = nn.LayerNorm(channels)
        reliability_input_dim = channels + attribute_channels + 1
        reliability_hidden = max(16, channels // 4)
        self.c_reliability_head = nn.Sequential(
            nn.Linear(reliability_input_dim, reliability_hidden),
            nn.LeakyReLU(),
            nn.Linear(reliability_hidden, 1),
        )
        nn.init.normal_(
            self.c_reliability_head[-1].weight,
            mean=0.0,
            std=1e-3,
        )
        nn.init.zeros_(self.c_reliability_head[-1].bias)
        self.graph_projection = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.LeakyReLU(),
        )
        self.l_to_h_transport = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        self.h_to_l_transport = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        self.h_transport_query = nn.Linear(
            channels,
            attention_d_k,
            bias=False,
        )
        self.l_transport_key = nn.Linear(
            channels,
            attention_d_k,
            bias=False,
        )
        self.l_transport_query = nn.Linear(
            channels,
            attention_d_k,
            bias=False,
        )
        self.h_transport_key = nn.Linear(
            channels,
            attention_d_k,
            bias=False,
        )
        self.h_intra_proj = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        self.l_intra_proj = nn.Linear(
            channels,
            channels,
            bias=False,
        )
        h_gate_output = nn.Linear(channels, 1)
        l_gate_output = nn.Linear(channels, 1)
        nn.init.zeros_(h_gate_output.weight)
        nn.init.zeros_(l_gate_output.weight)
        nn.init.constant_(h_gate_output.bias, -3.0)
        nn.init.constant_(l_gate_output.bias, -3.0)
        self.l_to_h_gate = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.LeakyReLU(),
            h_gate_output,
        )
        self.h_to_l_gate = nn.Sequential(
            nn.Linear(3 * channels, channels),
            nn.LeakyReLU(),
            l_gate_output,
        )
        h_tri_gate_output = nn.Linear(channels, 3)
        l_tri_gate_output = nn.Linear(channels, 3)
        nn.init.zeros_(h_tri_gate_output.weight)
        nn.init.zeros_(l_tri_gate_output.weight)
        tri_gate_init = torch.log(
            torch.tensor([0.80, 0.15, 0.05], dtype=torch.float32)
        )
        with torch.no_grad():
            h_tri_gate_output.bias.copy_(tri_gate_init)
            l_tri_gate_output.bias.copy_(tri_gate_init)
        self.h_tri_gate = nn.Sequential(
            nn.Linear(4 * channels, channels),
            nn.LeakyReLU(),
            h_tri_gate_output,
        )
        self.l_tri_gate = nn.Sequential(
            nn.Linear(4 * channels, channels),
            nn.LeakyReLU(),
            l_tri_gate_output,
        )
        self.h_concat_fuse = nn.Sequential(
            nn.Linear(4 * channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.l_concat_fuse = nn.Sequential(
            nn.Linear(4 * channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        bilinear_rank = max(16, channels // 2)
        self.h_bilinear_left = nn.Linear(
            channels,
            bilinear_rank,
            bias=False,
        )
        self.h_bilinear_right = nn.Linear(
            channels,
            bilinear_rank,
            bias=False,
        )
        self.h_bilinear_out = nn.Linear(
            bilinear_rank,
            channels,
            bias=False,
        )
        self.l_bilinear_left = nn.Linear(
            channels,
            bilinear_rank,
            bias=False,
        )
        self.l_bilinear_right = nn.Linear(
            channels,
            bilinear_rank,
            bias=False,
        )
        self.l_bilinear_out = nn.Linear(
            bilinear_rank,
            channels,
            bias=False,
        )
        self.h_bilinear_fuse = nn.Sequential(
            nn.Linear(5 * channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.l_bilinear_fuse = nn.Sequential(
            nn.Linear(5 * channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.h_transport_gamma = nn.Parameter(
            torch.tensor(float(transport_gamma_init))
        )
        self.l_transport_gamma = nn.Parameter(
            torch.tensor(float(transport_gamma_init))
        )
        self.mediator_kind = bridge_data.get(
            "mediator_kind",
            "center",
        )

        for name in (
            "prior_hc",
            "prior_ch",
            "prior_lc",
            "prior_cl",
        ):
            self.register_buffer(
                name,
                torch.as_tensor(
                    bridge_data[name],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        if "attributes" in bridge_data:
            self.register_buffer(
                "cell_attributes",
                torch.as_tensor(
                    bridge_data["attributes"],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        else:
            self.register_buffer(
                "cell_attributes",
                None,
                persistent=False,
            )
        if "cc_prior" in bridge_data:
            self.register_buffer(
                "cc_prior",
                torch.as_tensor(
                    bridge_data["cc_prior"],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "bias_cc",
                None,
                persistent=False,
            )
        else:
            self.register_buffer(
                "cc_prior",
                None,
                persistent=False,
            )
            self.register_buffer(
                "bias_cc",
                torch.as_tensor(
                    bridge_data["bias_cc"],
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

    @staticmethod
    def _row_normalize(matrix):
        return matrix / matrix.sum(
            dim=1,
            keepdim=True,
        ).clamp_min(1e-6)

    def _project_private_graph_prior(self, left, adjacency, right):
        projected = left @ adjacency.detach() @ right
        projected = projected.clamp_min(0.0)
        return self._row_normalize(projected)

    def _spatial_prior(self):
        if self.cc_prior is not None:
            return self._row_normalize(self.cc_prior)
        shifted = self.bias_cc - self.bias_cc.max(
            dim=1,
            keepdim=True,
        ).values
        return self._row_normalize(torch.exp(shifted))

    def _build_c_graph(
        self,
        hsi_nodes,
        lidar_nodes,
        hsi_adjacency,
        lidar_adjacency,
    ):
        h_value = self.h_value(hsi_nodes)
        l_value = self.l_value(lidar_nodes)
        h_context = self.prior_ch @ h_value
        l_context = self.prior_cl @ l_value
        node_value = torch.cat([h_value, l_value], dim=0)
        node_to_c = torch.cat([self.prior_hc, self.prior_lc], dim=0)
        c_degree = node_to_c.sum(dim=0).clamp_min(1e-6)
        incidence_context = (
            node_to_c.transpose(0, 1) @ node_value
        ) / c_degree.unsqueeze(1)
        cell_inputs = [
            h_context,
            l_context,
            incidence_context,
            torch.abs(h_context - l_context),
            h_context * l_context,
        ]
        if self.cell_attributes is not None:
            cell_inputs.append(self.cell_attributes)

        bridge_nodes = self.bridge_norm(
            self.cell_encoder(torch.cat(cell_inputs, dim=1))
            + self.bridge_embedding
        )
        c_query = self.c_query(bridge_nodes)
        c_key = self.c_key(bridge_nodes)

        spatial_prior = self._spatial_prior()
        hsi_prior = self._project_private_graph_prior(
            self.prior_ch,
            hsi_adjacency,
            self.prior_hc,
        )
        lidar_prior = self._project_private_graph_prior(
            self.prior_cl,
            lidar_adjacency,
            self.prior_lc,
        )
        logits = c_query @ c_key.transpose(0, 1) * self.scale
        logits = (
            logits
            + self.spatial_prior_weight
            * torch.log(spatial_prior.clamp_min(1e-6))
            + self.hsi_prior_weight
            * torch.log(hsi_prior.clamp_min(1e-6))
            + self.lidar_prior_weight
            * torch.log(lidar_prior.clamp_min(1e-6))
        )
        identity_support = torch.eye(
            logits.shape[0],
            dtype=torch.bool,
            device=logits.device,
        )
        support_mask = (
            (spatial_prior > 0.0)
            | (hsi_prior > 0.0)
            | (lidar_prior > 0.0)
            | identity_support
        )
        masked_logits = logits.masked_fill(
            ~support_mask,
            torch.finfo(logits.dtype).min,
        )
        attention_cc = self._topk_softmax(masked_logits, self.topk)
        supported_logits = logits.detach()[support_mask]
        diagnostics = {
            "c_gamma": float(self.c_gat_gamma.detach().item()),
            "c_gat_gamma": float(self.c_gat_gamma.detach().item()),
            "bridge_count": int(bridge_nodes.shape[0]),
            "mediator_kind": self.mediator_kind,
            "attention_cc_entropy": float(
                self._row_entropy(attention_cc).detach().mean().item()
            ),
            "support_density": float(
                support_mask.float().detach().mean().item()
            ),
            "spatial_prior_entropy": float(
                self._row_entropy(spatial_prior).detach().mean().item()
            ),
            "hsi_projected_prior_entropy": float(
                self._row_entropy(hsi_prior).detach().mean().item()
            ),
            "lidar_projected_prior_entropy": float(
                self._row_entropy(lidar_prior).detach().mean().item()
            ),
            "qk_logit_mean": float(logits.detach().mean().item()),
            "qk_logit_std": float(
                logits.detach().std(unbiased=False).item()
            ),
            "supported_qk_logit_mean": float(
                supported_logits.mean().item()
            ),
            "supported_qk_logit_std": float(
                supported_logits.std(unbiased=False).item()
            ),
            "prior_weights": (
                float(self.spatial_prior_weight),
                float(self.hsi_prior_weight),
                float(self.lidar_prior_weight),
            ),
            "initial_c_norm": float(
                bridge_nodes.detach().norm(dim=1).mean().item()
            ),
        }
        return {
            "h_value": h_value,
            "l_value": l_value,
            "bridge_nodes": bridge_nodes,
            "attention_cc": attention_cc,
            "diagnostics": diagnostics,
        }

    def _propagate_c_state(self, c_graph):
        bridge_nodes = c_graph["bridge_nodes"]
        attention_cc = c_graph["attention_cc"]

        c_gat_output = self.c_gat(bridge_nodes, attention_cc)
        c_graph_features = self.c_graph_norm(
            bridge_nodes + self.c_gat_gamma * c_gat_output
        )
        c_ffn_delta = self.c_ffn(c_graph_features)
        updated_bridge = self.c_ffn_norm(
            c_graph_features + c_ffn_delta
        )

        propagation_diagnostics = {
            "c_gat_output_norm": float(
                c_gat_output.detach().norm(dim=1).mean().item()
            ),
            "c_ffn_delta_norm": float(
                c_ffn_delta.detach().norm(dim=1).mean().item()
            ),
            "updated_c_state_norm": float(
                updated_bridge.detach().norm(dim=1).mean().item()
            ),
        }
        return updated_bridge, propagation_diagnostics

    def _c_state_reliability(self, updated_bridge, attention_cc):
        entropy = self._row_entropy(attention_cc)
        entropy_scale = max(
            float(np.log(max(attention_cc.shape[1], 2))),
            1e-6,
        )
        normalized_entropy = (entropy / entropy_scale).unsqueeze(1)
        reliability_inputs = [updated_bridge, normalized_entropy]
        if self.cell_attributes is not None:
            reliability_inputs.append(self.cell_attributes)
        reliability_logit = self.c_reliability_head(
            torch.cat(reliability_inputs, dim=1)
        )
        reliability = torch.exp(
            0.5 * torch.tanh(reliability_logit)
        ).squeeze(1)
        return reliability

    def forward(self, hsi_nodes, lidar_nodes, hsi_adjacency, lidar_adjacency):
        c_graph = self._build_c_graph(
            hsi_nodes,
            lidar_nodes,
            hsi_adjacency,
            lidar_adjacency,
        )
        updated_bridge, propagation_diagnostics = (
            self._propagate_c_state(c_graph)
        )
        consensus_pixel_features = torch.sparse.mm(
            self.bridge_projection_assignment,
            updated_bridge,
        )
        consensus_pixel_features = self.graph_projection(
            consensus_pixel_features
        )
        self.last_diagnostics = {
            **c_graph["diagnostics"],
            **propagation_diagnostics,
            "transport_mode": "none",
            "consensus_pixel_norm": float(
                consensus_pixel_features.detach()
                .norm(dim=1)
                .mean()
                .item()
            ),
        }
        return consensus_pixel_features

    def transport_nodes(
        self,
        hsi_nodes,
        lidar_nodes,
        hsi_adjacency,
        lidar_adjacency,
    ):
        c_graph = self._build_c_graph(
            hsi_nodes,
            lidar_nodes,
            hsi_adjacency,
            lidar_adjacency,
        )
        h_value = c_graph["h_value"]
        l_value = c_graph["l_value"]
        attention_cc = c_graph["attention_cc"]
        identity = torch.eye(
            attention_cc.shape[0],
            dtype=attention_cc.dtype,
            device=attention_cc.device,
        )
        base_transport_kernel = (
            (1.0 - self.transport_lambda) * identity
            + self.transport_lambda * attention_cc
        )
        propagation_diagnostics = {}
        c_reliability = None
        if self.transport_state == "topology-only":
            transport_kernel = base_transport_kernel
        elif self.transport_state == "cgnn-reliability":
            updated_bridge, propagation_diagnostics = (
                self._propagate_c_state(c_graph)
            )
            c_reliability = self._c_state_reliability(
                updated_bridge,
                attention_cc,
            )
            pair_reliability = torch.sqrt(
                c_reliability.unsqueeze(1)
                * c_reliability.unsqueeze(0)
            )
            transport_kernel = base_transport_kernel * pair_reliability
        else:
            raise ValueError(
                f"Unsupported transport state: {self.transport_state}"
            )

        beta_l_to_h = self._row_normalize(
            (self.prior_hc @ transport_kernel @ self.prior_cl)
            .clamp_min(0.0)
        )
        beta_h_to_l = self._row_normalize(
            (self.prior_lc @ transport_kernel @ self.prior_ch)
            .clamp_min(0.0)
        )
        if self.transport_message == "fixed":
            l_to_h_raw = beta_l_to_h @ l_value
            h_to_l_raw = beta_h_to_l @ h_value
            l_to_h_attention_entropy = None
            h_to_l_attention_entropy = None
        elif self.transport_message == "qk-prior":
            l_to_h_logits = (
                self.h_transport_query(hsi_nodes)
                @ self.l_transport_key(lidar_nodes).transpose(0, 1)
                * self.scale
                + self.transport_prior_weight
                * torch.log(beta_l_to_h.clamp_min(1e-6))
            )
            l_to_h_support = beta_l_to_h > 0.0
            l_to_h_logits = l_to_h_logits.masked_fill(
                ~l_to_h_support,
                torch.finfo(l_to_h_logits.dtype).min,
            )
            l_to_h_attention = F.softmax(l_to_h_logits, dim=1)
            l_to_h_raw = l_to_h_attention @ l_value
            h_to_l_logits = (
                self.l_transport_query(lidar_nodes)
                @ self.h_transport_key(hsi_nodes).transpose(0, 1)
                * self.scale
                + self.transport_prior_weight
                * torch.log(beta_h_to_l.clamp_min(1e-6))
            )
            h_to_l_support = beta_h_to_l > 0.0
            h_to_l_logits = h_to_l_logits.masked_fill(
                ~h_to_l_support,
                torch.finfo(h_to_l_logits.dtype).min,
            )
            h_to_l_attention = F.softmax(h_to_l_logits, dim=1)
            h_to_l_raw = h_to_l_attention @ h_value
            l_to_h_attention_entropy = float(
                self._row_entropy(l_to_h_attention)
                .detach()
                .mean()
                .item()
            )
            h_to_l_attention_entropy = float(
                self._row_entropy(h_to_l_attention)
                .detach()
                .mean()
                .item()
            )
        else:
            raise ValueError(
                f"Unsupported transport message: {self.transport_message}"
            )
        l_to_h_message = self.l_to_h_transport(l_to_h_raw)
        h_to_l_message = self.h_to_l_transport(h_to_l_raw)

        h_intra = self.h_intra_proj(hsi_adjacency @ hsi_nodes)
        l_intra = self.l_intra_proj(lidar_adjacency @ lidar_nodes)
        if self.transport_fusion == "residual":
            h_gate = torch.sigmoid(
                self.l_to_h_gate(
                    torch.cat(
                        [
                            hsi_nodes,
                            l_to_h_message,
                            torch.abs(hsi_nodes - l_to_h_message),
                        ],
                        dim=1,
                    )
                )
            )
            updated_hsi = (
                hsi_nodes
                + self.h_transport_gamma * h_gate * l_to_h_message
            )
            l_gate = torch.sigmoid(
                self.h_to_l_gate(
                    torch.cat(
                        [
                            lidar_nodes,
                            h_to_l_message,
                            torch.abs(lidar_nodes - h_to_l_message),
                        ],
                        dim=1,
                    )
                )
            )
            updated_lidar = (
                lidar_nodes
                + self.l_transport_gamma * l_gate * h_to_l_message
            )
            h_gate_mean = float(h_gate.detach().mean().item())
            l_gate_mean = float(l_gate.detach().mean().item())
            h_tri_gate_mean = None
            l_tri_gate_mean = None
            h_concat_delta_norm = None
            l_concat_delta_norm = None
            h_bilinear_product_norm = None
            l_bilinear_product_norm = None
        elif self.transport_fusion == "tri-gate":
            h_tri_gate = F.softmax(
                self.h_tri_gate(
                    torch.cat(
                        [
                            hsi_nodes,
                            h_intra,
                            l_to_h_message,
                            torch.abs(hsi_nodes - l_to_h_message),
                        ],
                        dim=1,
                    )
                ),
                dim=1,
            )
            h_mix = (
                h_tri_gate[:, 0:1] * hsi_nodes
                + h_tri_gate[:, 1:2] * h_intra
                + h_tri_gate[:, 2:3] * l_to_h_message
            )
            updated_hsi = (
                hsi_nodes
                + self.h_transport_gamma * (h_mix - hsi_nodes)
            )
            l_tri_gate = F.softmax(
                self.l_tri_gate(
                    torch.cat(
                        [
                            lidar_nodes,
                            l_intra,
                            h_to_l_message,
                            torch.abs(lidar_nodes - h_to_l_message),
                        ],
                        dim=1,
                    )
                ),
                dim=1,
            )
            l_mix = (
                l_tri_gate[:, 0:1] * lidar_nodes
                + l_tri_gate[:, 1:2] * l_intra
                + l_tri_gate[:, 2:3] * h_to_l_message
            )
            updated_lidar = (
                lidar_nodes
                + self.l_transport_gamma * (l_mix - lidar_nodes)
            )
            h_gate_mean = None
            l_gate_mean = None
            h_tri_gate_mean = (
                h_tri_gate.detach().mean(dim=0).cpu().tolist()
            )
            l_tri_gate_mean = (
                l_tri_gate.detach().mean(dim=0).cpu().tolist()
            )
            h_concat_delta_norm = None
            l_concat_delta_norm = None
            h_bilinear_product_norm = None
            l_bilinear_product_norm = None
        elif self.transport_fusion == "concat":
            h_concat_input = torch.cat(
                [
                    hsi_nodes,
                    h_intra,
                    l_to_h_message,
                    torch.abs(hsi_nodes - l_to_h_message),
                ],
                dim=1,
            )
            h_fused = self.h_concat_fuse(h_concat_input)
            updated_hsi = (
                hsi_nodes
                + self.h_transport_gamma * (h_fused - hsi_nodes)
            )
            l_concat_input = torch.cat(
                [
                    lidar_nodes,
                    l_intra,
                    h_to_l_message,
                    torch.abs(lidar_nodes - h_to_l_message),
                ],
                dim=1,
            )
            l_fused = self.l_concat_fuse(l_concat_input)
            updated_lidar = (
                lidar_nodes
                + self.l_transport_gamma * (l_fused - lidar_nodes)
            )
            h_gate_mean = None
            l_gate_mean = None
            h_tri_gate_mean = None
            l_tri_gate_mean = None
            h_concat_delta_norm = float(
                (h_fused - hsi_nodes).detach().norm(dim=1).mean().item()
            )
            l_concat_delta_norm = float(
                (l_fused - lidar_nodes)
                .detach()
                .norm(dim=1)
                .mean()
                .item()
            )
            h_bilinear_product_norm = None
            l_bilinear_product_norm = None
        elif self.transport_fusion == "bilinear":
            h_product = self.h_bilinear_out(
                self.h_bilinear_left(hsi_nodes)
                * self.h_bilinear_right(l_to_h_message)
            )
            h_fused = self.h_bilinear_fuse(
                torch.cat(
                    [
                        hsi_nodes,
                        h_intra,
                        l_to_h_message,
                        torch.abs(hsi_nodes - l_to_h_message),
                        h_product,
                    ],
                    dim=1,
                )
            )
            updated_hsi = (
                hsi_nodes
                + self.h_transport_gamma * (h_fused - hsi_nodes)
            )
            l_product = self.l_bilinear_out(
                self.l_bilinear_left(lidar_nodes)
                * self.l_bilinear_right(h_to_l_message)
            )
            l_fused = self.l_bilinear_fuse(
                torch.cat(
                    [
                        lidar_nodes,
                        l_intra,
                        h_to_l_message,
                        torch.abs(lidar_nodes - h_to_l_message),
                        l_product,
                    ],
                    dim=1,
                )
            )
            updated_lidar = (
                lidar_nodes
                + self.l_transport_gamma * (l_fused - lidar_nodes)
            )
            h_gate_mean = None
            l_gate_mean = None
            h_tri_gate_mean = None
            l_tri_gate_mean = None
            h_concat_delta_norm = float(
                (h_fused - hsi_nodes).detach().norm(dim=1).mean().item()
            )
            l_concat_delta_norm = float(
                (l_fused - lidar_nodes)
                .detach()
                .norm(dim=1)
                .mean()
                .item()
            )
            h_bilinear_product_norm = float(
                h_product.detach().norm(dim=1).mean().item()
            )
            l_bilinear_product_norm = float(
                l_product.detach().norm(dim=1).mean().item()
            )
        else:
            raise ValueError(
                f"Unsupported transport fusion: {self.transport_fusion}"
            )

        self.last_diagnostics = {
            **c_graph["diagnostics"],
            **propagation_diagnostics,
            "transport_mode": "bidirectional",
            "transport_fusion": self.transport_fusion,
            "transport_message": self.transport_message,
            "transport_state": self.transport_state,
            "transport_prior_weight": float(self.transport_prior_weight),
            "transport_lambda": float(self.transport_lambda),
            "c_reliability_mean": (
                None
                if c_reliability is None
                else float(c_reliability.detach().mean().item())
            ),
            "c_reliability_min": (
                None
                if c_reliability is None
                else float(c_reliability.detach().min().item())
            ),
            "c_reliability_max": (
                None
                if c_reliability is None
                else float(c_reliability.detach().max().item())
            ),
            "l_to_h_beta_density": float(
                (beta_l_to_h > 0.0).float().detach().mean().item()
            ),
            "h_to_l_beta_density": float(
                (beta_h_to_l > 0.0).float().detach().mean().item()
            ),
            "l_to_h_attention_entropy": l_to_h_attention_entropy,
            "h_to_l_attention_entropy": h_to_l_attention_entropy,
            "h_transport_gamma": float(
                self.h_transport_gamma.detach().item()
            ),
            "l_transport_gamma": float(
                self.l_transport_gamma.detach().item()
            ),
            "h_transport_gate_mean": h_gate_mean,
            "l_transport_gate_mean": l_gate_mean,
            "h_tri_gate_mean": h_tri_gate_mean,
            "l_tri_gate_mean": l_tri_gate_mean,
            "h_concat_delta_norm": h_concat_delta_norm,
            "l_concat_delta_norm": l_concat_delta_norm,
            "h_bilinear_product_norm": h_bilinear_product_norm,
            "l_bilinear_product_norm": l_bilinear_product_norm,
            "h_intra_message_norm": float(
                h_intra.detach().norm(dim=1).mean().item()
            ),
            "l_intra_message_norm": float(
                l_intra.detach().norm(dim=1).mean().item()
            ),
            "l_to_h_message_norm": float(
                l_to_h_message.detach().norm(dim=1).mean().item()
            ),
            "h_to_l_message_norm": float(
                h_to_l_message.detach().norm(dim=1).mean().item()
            ),
        }
        return updated_hsi, updated_lidar

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
        self.last_gat2_adjacency = None

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
        self.last_gat2_adjacency = adjacency
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
    """Private HSI/LiDAR GSDG graphs plus optional mediator C-GAT."""

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
        consensus_graph_weight=0.1,
        consensus_graph_fusion="residual-c",
        consensus_graph_residual_init=0.0,
        consensus_graph_transport="none",
        consensus_graph_transport_fusion="residual",
        consensus_graph_transport_message="fixed",
        consensus_graph_transport_state="topology-only",
        consensus_graph_transport_prior_weight=1.0,
        consensus_graph_transport_lambda=0.5,
        consensus_graph_transport_gamma_init=0.0,
        consensus_graph_spatial_prior_weight=1.0,
        consensus_graph_hsi_prior_weight=0.5,
        consensus_graph_lidar_prior_weight=0.5,
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
        self.post_gat_consensus_graph = post_gat_consensus_graph
        self.consensus_graph_weight = consensus_graph_weight
        self.consensus_graph_fusion = consensus_graph_fusion
        self.consensus_graph_transport = consensus_graph_transport
        self.last_contrastive_loss = None
        self.last_variance_loss = None
        self.last_consensus_graph_gate_diagnostics = None
        self.cnn_branch_mode = cnn_branch

        self.hsi_graph = ModalityGSDGGraphEncoder(
            in_channels=hsi_channels,
            assignment=hsi_assignment,
            spatial_prior=hsi_spatial_prior,
            hidden_dim=hidden_dim,
            dynamic_d_k=dynamic_d_k,
            dynamic_topk=dynamic_topk,
            dynamic_tau=dynamic_tau,
            use_fdsm=fdsm_scope == "hsi",
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
            use_edge_weights=lidar_candidate_mask is not None,
            lidar_modulation=lidar_modulation,
            rag_adjacency=lidar_rag_adjacency,
            geometry_descriptors=lidar_geometry_descriptors,
        )

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

        if post_gat_consensus_graph != "none":
            if bridge_data is None:
                raise ValueError(
                    "mediator data is required for mediator "
                    "consensus graph."
                )
            self.consensus_graph_branch = (
                PostGATMediatedConsensusGraph(
                    hidden_dim,
                    bridge_data,
                    attention_d_k=bridge_attention_d_k,
                    topk=bridge_attention_topk,
                    gamma_init=bridge_gamma_init,
                    transport_lambda=consensus_graph_transport_lambda,
                    transport_fusion=consensus_graph_transport_fusion,
                    transport_message=consensus_graph_transport_message,
                    transport_state=consensus_graph_transport_state,
                    transport_prior_weight=(
                        consensus_graph_transport_prior_weight
                    ),
                    transport_gamma_init=(
                        consensus_graph_transport_gamma_init
                    ),
                    spatial_prior_weight=(
                        consensus_graph_spatial_prior_weight
                    ),
                    hsi_prior_weight=consensus_graph_hsi_prior_weight,
                    lidar_prior_weight=(
                        consensus_graph_lidar_prior_weight
                    ),
                )
            )
            if consensus_graph_fusion == "c-guided-gate":
                gate_output = nn.Linear(hidden_dim, 3)
                nn.init.zeros_(gate_output.weight)
                initial_weights = torch.tensor(
                    [
                        graph_modality_lambda
                        * (1.0 - consensus_graph_weight),
                        (1.0 - graph_modality_lambda)
                        * (1.0 - consensus_graph_weight),
                        consensus_graph_weight,
                    ],
                    dtype=torch.float32,
                ).clamp_min(1e-6)
                with torch.no_grad():
                    gate_output.bias.copy_(torch.log(initial_weights))
                self.consensus_graph_gate = nn.Sequential(
                    nn.Linear(6 * hidden_dim, hidden_dim),
                    nn.LeakyReLU(),
                    gate_output,
                )
            else:
                self.consensus_graph_gate = None
            if consensus_graph_fusion == "residual-c":
                self.consensus_graph_residual_gamma = nn.Parameter(
                    torch.tensor(float(consensus_graph_residual_init))
                )
                residual_gate_output = nn.Linear(hidden_dim, 1)
                nn.init.zeros_(residual_gate_output.weight)
                nn.init.constant_(residual_gate_output.bias, -3.0)
                self.consensus_graph_residual_gate = nn.Sequential(
                    nn.Linear(3 * hidden_dim, hidden_dim),
                    nn.LeakyReLU(),
                    residual_gate_output,
                )
            else:
                self.consensus_graph_residual_gamma = None
                self.consensus_graph_residual_gate = None
        else:
            self.consensus_graph_branch = None
            self.consensus_graph_gate = None
            self.consensus_graph_residual_gamma = None
            self.consensus_graph_residual_gate = None

    def _encode_private_graph_nodes(self, hsi, lidar):
        hsi_nodes = self.hsi_graph.encode_nodes(hsi)
        hsi_features, hsi_adjacency = self.hsi_graph.apply_gat1(
            hsi_nodes
        )
        hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
            hsi_features,
            adjacency=hsi_adjacency,
        )
        lidar_nodes = self.lidar_graph.encode_nodes(lidar)
        lidar_features, lidar_adjacency = self.lidar_graph.apply_gat1(
            lidar_nodes
        )
        lidar_final_nodes = self.lidar_graph.apply_gat2_nodes(
            lidar_features,
            adjacency=lidar_adjacency,
        )
        return hsi_final_nodes, lidar_final_nodes

    def forward(self, hsi, lidar, joint_input):
        self.last_contrastive_loss = None
        self.last_variance_loss = None
        self.last_consensus_graph_gate_diagnostics = None
        if self.consensus_graph_branch is None:
            hsi_graph_features = self.hsi_graph(hsi)
            lidar_graph_features = self.lidar_graph(lidar)
            graph_features = (
                self.graph_modality_lambda * hsi_graph_features
                + (1.0 - self.graph_modality_lambda)
                * lidar_graph_features
            )
        else:
            (
                hsi_final_nodes,
                lidar_final_nodes,
            ) = self._encode_private_graph_nodes(hsi, lidar)
            hsi_adjacency = self.hsi_graph.last_gat2_adjacency
            lidar_adjacency = self.lidar_graph.last_gat2_adjacency
            if hsi_adjacency is None or lidar_adjacency is None:
                raise RuntimeError(
                    "Mediator consensus graph requires both private "
                    "GAT2 adjacencies."
                )
            if self.consensus_graph_transport == "bidirectional":
                (
                    hsi_final_nodes,
                    lidar_final_nodes,
                ) = self.consensus_graph_branch.transport_nodes(
                    hsi_final_nodes,
                    lidar_final_nodes,
                    hsi_adjacency,
                    lidar_adjacency,
                )
                hsi_graph_features = self.hsi_graph.project_nodes(
                    hsi_final_nodes
                )
                lidar_graph_features = self.lidar_graph.project_nodes(
                    lidar_final_nodes
                )
                graph_features = (
                    self.graph_modality_lambda * hsi_graph_features
                    + (1.0 - self.graph_modality_lambda)
                    * lidar_graph_features
                )
                self.last_consensus_graph_gate_diagnostics = {
                    "fusion_mode": "transport-private-fusion",
                    "private_weights": [
                        self.graph_modality_lambda,
                        1.0 - self.graph_modality_lambda,
                    ],
                }
            elif self.consensus_graph_transport != "none":
                raise ValueError(
                    f"Unsupported consensus graph transport: "
                    f"{self.consensus_graph_transport}"
                )
            else:
                hsi_graph_features = self.hsi_graph.project_nodes(
                    hsi_final_nodes
                )
                lidar_graph_features = self.lidar_graph.project_nodes(
                    lidar_final_nodes
                )
                consensus_graph_features = self.consensus_graph_branch(
                    hsi_final_nodes,
                    lidar_final_nodes,
                    hsi_adjacency,
                    lidar_adjacency,
                )
                private_graph_features = (
                    self.graph_modality_lambda * hsi_graph_features
                    + (1.0 - self.graph_modality_lambda)
                    * lidar_graph_features
                )
                if self.consensus_graph_fusion == "c-guided-gate":
                    gate_input = torch.cat(
                        [
                            hsi_graph_features,
                            lidar_graph_features,
                            consensus_graph_features,
                            torch.abs(
                                hsi_graph_features
                                - consensus_graph_features
                            ),
                            torch.abs(
                                lidar_graph_features
                                - consensus_graph_features
                            ),
                            torch.abs(
                                hsi_graph_features
                                - lidar_graph_features
                            ),
                        ],
                        dim=1,
                    )
                    gate = F.softmax(
                        self.consensus_graph_gate(gate_input),
                        dim=1,
                    )
                    graph_features = (
                        gate[:, 0:1] * hsi_graph_features
                        + gate[:, 1:2] * lidar_graph_features
                        + gate[:, 2:3] * consensus_graph_features
                    )
                    gate_entropy = -torch.sum(
                        gate * torch.log(gate.clamp_min(1e-12)),
                        dim=1,
                    )
                    self.last_consensus_graph_gate_diagnostics = {
                        "fusion_mode": self.consensus_graph_fusion,
                        "gate_mean": (
                            gate.detach().mean(dim=0).cpu().tolist()
                        ),
                        "gate_entropy": float(
                            gate_entropy.detach().mean().item()
                        ),
                        "gate_min": (
                            gate.detach().min(dim=0).values.cpu().tolist()
                        ),
                        "gate_max": (
                            gate.detach().max(dim=0).values.cpu().tolist()
                        ),
                    }
                elif self.consensus_graph_fusion == "fixed":
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
                    self.last_consensus_graph_gate_diagnostics = {
                        "fusion_mode": self.consensus_graph_fusion,
                        "fixed_weights": [
                            private_weight * self.graph_modality_lambda,
                            private_weight
                            * (1.0 - self.graph_modality_lambda),
                            consensus_weight,
                        ],
                    }
                elif self.consensus_graph_fusion == "residual-c":
                    residual_gamma = self.consensus_graph_residual_gamma
                    residual_input = torch.cat(
                        [
                            private_graph_features,
                            consensus_graph_features,
                            torch.abs(
                                consensus_graph_features
                                - private_graph_features
                            ),
                        ],
                        dim=1,
                    )
                    residual_gate = torch.sigmoid(
                        self.consensus_graph_residual_gate(residual_input)
                    )
                    graph_features = (
                        private_graph_features
                        + residual_gamma
                        * residual_gate
                        * (
                            consensus_graph_features
                            - private_graph_features
                        )
                    )
                    self.last_consensus_graph_gate_diagnostics = {
                        "fusion_mode": self.consensus_graph_fusion,
                        "residual_gamma": float(
                            residual_gamma.detach().item()
                        ),
                        "residual_gate_mean": float(
                            residual_gate.detach().mean().item()
                        ),
                        "residual_gate_min": float(
                            residual_gate.detach().min().item()
                        ),
                        "residual_gate_max": float(
                            residual_gate.detach().max().item()
                        ),
                        "baseline_weights": [
                            self.graph_modality_lambda,
                            1.0 - self.graph_modality_lambda,
                            0.0,
                        ],
                    }
                else:
                    raise ValueError(
                        f"Unsupported consensus graph fusion: "
                        f"{self.consensus_graph_fusion}"
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
    needs_intersection_cells = (
        args.post_gat_consensus_graph == "intersection-mediator"
    )
    if needs_intersection_cells:
        cell_data = build_common_refinement_cells(
            hsi_assignment,
            lidar_assignment,
            hsi.shape[0],
            hsi.shape[1],
        )
        if (
            args.consensus_graph_cell_edge
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
    bridge_data = None
    if args.post_gat_consensus_graph == "intersection-mediator":
        bridge_data = build_intersection_mediator_data(
            cell_data,
            hsi_assignment.shape[1],
            lidar_assignment.shape[1],
            edge_mode=args.consensus_graph_cell_edge,
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
            post_gat_consensus_graph=(
                args.post_gat_consensus_graph
            ),
            consensus_graph_weight=args.consensus_graph_weight,
            consensus_graph_fusion=args.consensus_graph_fusion,
            consensus_graph_residual_init=(
                args.consensus_graph_residual_init
            ),
            consensus_graph_transport=args.consensus_graph_transport,
            consensus_graph_transport_fusion=(
                args.consensus_graph_transport_fusion
            ),
            consensus_graph_transport_message=(
                args.consensus_graph_transport_message
            ),
            consensus_graph_transport_state=(
                args.consensus_graph_transport_state
            ),
            consensus_graph_transport_prior_weight=(
                args.consensus_graph_transport_prior_weight
            ),
            consensus_graph_transport_lambda=(
                args.consensus_graph_transport_lambda
            ),
            consensus_graph_transport_gamma_init=(
                args.consensus_graph_transport_gamma_init
            ),
            consensus_graph_spatial_prior_weight=(
                args.consensus_graph_spatial_prior_weight
            ),
            consensus_graph_hsi_prior_weight=(
                args.consensus_graph_hsi_prior_weight
            ),
            consensus_graph_lidar_prior_weight=(
                args.consensus_graph_lidar_prior_weight
            ),
            bridge_attention_d_k=args.bridge_attention_dk,
            bridge_attention_topk=args.bridge_attention_topk,
            bridge_gamma_init=args.consensus_graph_c_gamma_init,
            bridge_data=bridge_data,
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
    consensus_graph_diagnostics = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_model()
        classification_loss = criterion(
            logits.index_select(0, train_index),
            train_labels,
        )
        loss = classification_loss
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % args.log_interval == 0:
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
                    gate_record = getattr(
                        model,
                        "last_consensus_graph_gate_diagnostics",
                        None,
                    )
                    if gate_record is not None:
                        consensus_graph_record = {
                            **consensus_graph_record,
                            **gate_record,
                        }
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
                f"train_OA={train_oa:.4f}"
            )
            if consensus_graph_record is not None:
                fusion_mode = consensus_graph_record.get(
                    "fusion_mode",
                    "unknown",
                )
                fusion_suffix = ""
                transport_mode = consensus_graph_record.get(
                    "transport_mode",
                    "none",
                )
                if transport_mode == "bidirectional":
                    fusion_mode = "bidirectional-transport"
                    transport_fusion = consensus_graph_record.get(
                        "transport_fusion",
                        "residual",
                    )
                    transport_message = consensus_graph_record.get(
                        "transport_message",
                        "fixed",
                    )
                    transport_state = consensus_graph_record.get(
                        "transport_state",
                        "topology-only",
                    )
                    if transport_fusion == "tri-gate":
                        h_tri = np.asarray(
                            consensus_graph_record["h_tri_gate_mean"]
                        )
                        l_tri = np.asarray(
                            consensus_graph_record["l_tri_gate_mean"]
                        )
                        gate_text = (
                            f"tri={h_tri.round(2).tolist()}/"
                            f"{l_tri.round(2).tolist()}"
                        )
                    elif transport_fusion == "concat":
                        gate_text = (
                            "concat_delta="
                            f"{consensus_graph_record['h_concat_delta_norm']:.4f}/"
                            f"{consensus_graph_record['l_concat_delta_norm']:.4f}"
                        )
                    elif transport_fusion == "bilinear":
                        gate_text = (
                            "bilinear_delta="
                            f"{consensus_graph_record['h_concat_delta_norm']:.4f}/"
                            f"{consensus_graph_record['l_concat_delta_norm']:.4f}"
                            ", product="
                            f"{consensus_graph_record['h_bilinear_product_norm']:.4f}/"
                            f"{consensus_graph_record['l_bilinear_product_norm']:.4f}"
                        )
                    else:
                        gate_text = (
                            "gates="
                            f"{consensus_graph_record['h_transport_gate_mean']:.4f}/"
                            f"{consensus_graph_record['l_transport_gate_mean']:.4f}"
                        )
                    fusion_suffix = (
                        ", lambda="
                        f"{consensus_graph_record['transport_lambda']:.2f}"
                        ", mode="
                        f"{transport_fusion}"
                        ", msg="
                        f"{transport_message}"
                        ", state="
                        f"{transport_state}"
                        ", h_gamma="
                        f"{consensus_graph_record['h_transport_gamma']:.4f}"
                        ", l_gamma="
                        f"{consensus_graph_record['l_transport_gamma']:.4f}"
                        ", "
                        f"{gate_text}"
                    )
                    c_reliability_mean = consensus_graph_record.get(
                        "c_reliability_mean"
                    )
                    if c_reliability_mean is not None:
                        fusion_suffix += (
                            ", c_rel="
                            f"{c_reliability_mean:.3f}"
                        )
                elif "residual_gamma" in consensus_graph_record:
                    fusion_suffix = (
                        ", residual="
                        f"{consensus_graph_record['residual_gamma']:.4f}"
                        ", gate="
                        f"{consensus_graph_record.get('residual_gate_mean', 0.0):.4f}"
                    )
                elif "fixed_weights" in consensus_graph_record:
                    fixed_weights = np.asarray(
                        consensus_graph_record["fixed_weights"]
                    )
                    fusion_suffix = (
                        ", weights="
                        f"{fixed_weights.round(2).tolist()}"
                    )
                elif "gate_mean" in consensus_graph_record:
                    gate_mean = np.asarray(
                        consensus_graph_record["gate_mean"]
                    )
                    fusion_suffix = (
                        ", gate="
                        f"{gate_mean.round(2).tolist()}"
                    )
                print(
                    "  C-mediator: "
                    f"gamma={consensus_graph_record['c_gamma']:.4f}, "
                    f"entropy={consensus_graph_record['attention_cc_entropy']:.4f}, "
                    f"fusion={fusion_mode}{fusion_suffix}"
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
        f"{consensus_graph_configuration_tag(args, class_count)}_"
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
    if args.bridge_attention_dk <= 0:
        raise ValueError("--bridge-attention-dk must be positive.")
    if args.bridge_attention_topk <= 0:
        raise ValueError("--bridge-attention-topk must be positive.")
    if not 0.0 <= args.consensus_graph_weight <= 1.0:
        raise ValueError(
            "--consensus-graph-weight must be between 0 and 1."
        )
    if args.consensus_graph_residual_init < 0:
        raise ValueError(
            "--consensus-graph-residual-init must be nonnegative."
        )
    if args.consensus_graph_c_gamma_init < 0:
        raise ValueError(
            "--consensus-graph-c-gamma-init must be nonnegative."
        )
    if not 0.0 <= args.consensus_graph_transport_lambda <= 1.0:
        raise ValueError(
            "--consensus-graph-transport-lambda must be between 0 and 1."
        )
    if args.consensus_graph_transport_gamma_init < 0:
        raise ValueError(
            "--consensus-graph-transport-gamma-init must be nonnegative."
        )
    if args.consensus_graph_transport_prior_weight < 0:
        raise ValueError(
            "--consensus-graph-transport-prior-weight must be nonnegative."
        )
    if any(
        weight < 0
        for weight in (
            args.consensus_graph_spatial_prior_weight,
            args.consensus_graph_hsi_prior_weight,
            args.consensus_graph_lidar_prior_weight,
        )
    ):
        raise ValueError(
            "Consensus graph prior weights must be nonnegative."
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
        raise ValueError("Mediator cell edge weights must be nonnegative.")
    if (
        args.consensus_graph_cell_edge != "binary"
        and args.post_gat_consensus_graph != "intersection-mediator"
    ):
        raise ValueError(
            "--consensus-graph-cell-edge spectral-height-boundary "
            "requires --post-gat-consensus-graph "
            "intersection-mediator."
        )
    if args.post_gat_consensus_graph != "none":
        if args.graph_layout != "separate":
            raise ValueError(
                "--post-gat-consensus-graph requires "
                "--graph-layout separate."
            )
        if len(args.scales) != 1:
            raise ValueError(
                "--post-gat-consensus-graph intersection-mediator "
                "requires exactly one superpixel scale."
            )
    elif args.consensus_graph_transport != "none":
        raise ValueError(
            "--consensus-graph-transport requires "
            "--post-gat-consensus-graph intersection-mediator."
        )
    if (
        args.consensus_graph_transport == "none"
        and args.consensus_graph_transport_fusion != "residual"
    ):
        raise ValueError(
            "--consensus-graph-transport-fusion tri-gate/concat/bilinear "
            "requires --consensus-graph-transport bidirectional."
        )
    if (
        args.consensus_graph_transport == "none"
        and args.consensus_graph_transport_message != "fixed"
    ):
        raise ValueError(
            "--consensus-graph-transport-message qk-prior requires "
            "--consensus-graph-transport bidirectional."
        )
    if (
        args.consensus_graph_transport == "none"
        and args.consensus_graph_transport_state != "topology-only"
    ):
        raise ValueError(
            "--consensus-graph-transport-state cgnn-reliability requires "
            "--consensus-graph-transport bidirectional."
        )
    if args.graph_layout == "joint" and args.cnn_branch != "original":
        raise ValueError(
            "--cnn-branch gsdg currently requires "
            "--graph-layout separate."
        )
    if args.graph_layout == "joint" and args.fdsm_scope != "none":
        raise ValueError(
            "--fdsm-scope hsi requires --graph-layout separate."
        )
    if args.graph_layout == "joint" and args.lidar_modulation != "none":
        raise ValueError(
            "--lidar-modulation requires --graph-layout separate."
        )
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
        "demo_train | Stage 8: private dual GSDG + mediator C-GAT"
    )
    print(
        f"CNN: {args.cnn_branch} | graph-layout: {args.graph_layout} | "
        f"LiDAR segments: {args.lidar_segmentation}"
    )
    if args.graph_layout == "separate":
        print(
            "Private graphs: HSI-SLIC + "
            f"LiDAR-{args.lidar_graph_prior}"
            f"({args.lidar_rag_hops}hop,"
            f"k={args.lidar_height_knn_k}) | "
            f"FDSM={args.fdsm_scope} | "
            f"LiDAR-mod={args.lidar_modulation}"
        )
        if args.post_gat_consensus_graph != "none":
            resolved_mediator_count = cell_data["cell_count"]
            private_weight = 1.0 - args.consensus_graph_weight
            if args.consensus_graph_transport == "bidirectional":
                fusion_detail = (
                    "C-mediated bidirectional transport, "
                    f"fusion={args.consensus_graph_transport_fusion}, "
                    f"message={args.consensus_graph_transport_message}, "
                    f"state={args.consensus_graph_transport_state}, "
                    f"lambda={args.consensus_graph_transport_lambda:g}, "
                    f"eta={args.consensus_graph_transport_prior_weight:g}, "
                    f"gamma-init={args.consensus_graph_transport_gamma_init:g}; "
                    "pixel C fusion disabled"
                )
            elif args.consensus_graph_fusion == "c-guided-gate":
                fusion_detail = (
                    "gate init H/L/C="
                    f"{private_weight * args.graph_modality_lambda:g}/"
                    f"{private_weight * (1.0 - args.graph_modality_lambda):g}/"
                    f"{args.consensus_graph_weight:g}"
                )
            elif args.consensus_graph_fusion == "residual-c":
                fusion_detail = (
                    "local residual gate, "
                    f"gamma-init={args.consensus_graph_residual_init:g}"
                )
            else:
                fusion_detail = (
                    "fixed H/L/C="
                    f"{private_weight * args.graph_modality_lambda:g}/"
                    f"{private_weight * (1.0 - args.graph_modality_lambda):g}/"
                    f"{args.consensus_graph_weight:g}"
                )
            print(
                "Mediator: "
                f"{args.post_gat_consensus_graph} | "
                f"cells={resolved_mediator_count} | "
                f"edge={args.consensus_graph_cell_edge} | "
                f"C-QK d_k={args.bridge_attention_dk}, "
                f"topk={args.bridge_attention_topk} | "
                f"c-gamma-init={args.consensus_graph_c_gamma_init:g} | "
                f"alpha S/H/L="
                f"{args.consensus_graph_spatial_prior_weight:g}/"
                f"{args.consensus_graph_hsi_prior_weight:g}/"
                f"{args.consensus_graph_lidar_prior_weight:g} | "
                f"{fusion_detail}"
            )
            if args.consensus_graph_cell_edge == "spectral-height-boundary":
                weight_stats = cell_data["cell_weight_stats"]
                print(
                    "Mediator C-C multimodal edge weights: "
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
        else:
            print(
                "Mediator: none | graph fusion H/L="
                f"{args.graph_modality_lambda:g}/"
                f"{1.0 - args.graph_modality_lambda:g}"
            )
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
        f"{consensus_graph_configuration_tag(args, class_count)}_"
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
