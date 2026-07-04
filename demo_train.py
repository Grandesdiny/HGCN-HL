"""Stage-8 demo: overlap-conditioned second-layer dynamic graphs.

The fixed hypergraph/HGCN path is replaced by GSDG graph/GAT propagation.
The default uses independent HSI and LiDAR graphs; the previous concatenated
node graph remains selectable. The LiDAR graph can additionally restrict its
dynamic neighbors with a local RAG and an elevation-similarity KNN. The
original joint CNN and fusion remain. An intersection-cell RAG can be inserted
after GAT1, but is disabled by default.
"""

import argparse
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
        self.cell_interaction = cell_interaction
        self.cell_interaction_stages = cell_interaction_stages
        self.cell_output_branch = cell_output_branch
        self.cell_output_weight = cell_output_weight
        self.cell_topology_veto = cell_topology_veto
        self.cell_veto_threshold = cell_veto_threshold
        use_qk_condition = (
            cross_modal_interaction == "overlap-qk-condition"
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

        # This is intentionally still the original joint-input CNN path.
        self.joint_feature_mapping = nn.Sequential(
            WMF(hsi_channels + 1, hidden_dim),
            WMF(hidden_dim, hidden_dim),
        )
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
        self.classifier = nn.Linear(hidden_dim, class_count)

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

    def forward(self, hsi, lidar, joint_input):
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
            hsi_graph_features = self.hsi_graph.project_nodes(
                hsi_final_nodes
            )
            lidar_graph_features = self.lidar_graph.project_nodes(
                lidar_final_nodes
            )
        elif self.cross_modal_interaction == "none":
            # Preserve the original Stage-3 path exactly.
            hsi_graph_features = self.hsi_graph(hsi)
            lidar_graph_features = self.lidar_graph(lidar)
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
        graph_features = (
            self.graph_modality_lambda * hsi_graph_features
            + (1.0 - self.graph_modality_lambda)
            * lidar_graph_features
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
            cell_interaction=args.cell_interaction,
            cell_data=cell_data,
            cell_pixel_descriptor=args.cell_pixel_descriptor,
            cell_edge_mode=args.cell_edge_mode,
            cell_interaction_stages=args.cell_interaction_stages,
            cell_output_branch=args.cell_output_branch,
            cell_output_weight=args.cell_output_weight,
            cell_topology_veto=args.cell_topology_veto,
            cell_veto_threshold=args.cell_veto_threshold,
            fdsm_scope=args.fdsm_scope,
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
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_model()
        loss = criterion(
            logits.index_select(0, train_index),
            train_labels,
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
            train_predictions = (
                logits.index_select(0, train_index).argmax(dim=1)
            )
            train_oa = (
                train_predictions == train_labels
            ).float().mean().item()
            print(
                f"Run {run_index + 1}/{args.runs} | "
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"loss={loss.item():.6f} | train_OA={train_oa:.4f}"
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
    checkpoint = args.output_dir / (
        f"{args.dataset}_{args.train_samples_per_class}px_"
        f"{STAGE}_{args.graph_layout}_"
        f"lidar-{args.lidar_segmentation}_"
        f"prior-{args.lidar_graph_prior}_"
        f"cross-{args.cross_modal_interaction}_"
        f"overlap-{args.overlap_metric}_"
        f"{cell_configuration_tag(args)}_"
        f"fdsm-{args.fdsm_scope}_"
        f"lidarmod-{args.lidar_modulation}_"
        f"run{run_index + 1}.pt"
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
    if (
        args.graph_layout == "joint"
        and args.cross_modal_interaction != "none"
    ):
        raise ValueError(
            "--cross-modal-interaction requires "
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
        joint_spatial_prior,
    ) = prepare_data(args, config)

    print("=" * 72)
    print(
        "demo_train | Stage 8: overlap-conditioned second-layer Q/K"
    )
    print(
        "Unchanged CNN: joint PCA(HSI)+LiDAR -> WMF -> "
        "original 5x5/5x5 CNN"
    )
    if args.graph_layout == "separate":
        print(
            "Graph layout: independent HSI-SLIC and "
            f"LiDAR-{args.lidar_segmentation} GSDG graphs"
        )
        print(
            "Graph fusion after pixel projection: "
            f"HSI={args.graph_modality_lambda:g}, "
            f"LiDAR={1.0 - args.graph_modality_lambda:g}"
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
    print("Not enabled: hypergraph or GSDG CNN")
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
    result_path = args.output_dir / (
        f"{args.dataset}_{args.train_samples_per_class}px_"
        f"{STAGE}_{args.graph_layout}_"
        f"lidar-{args.lidar_segmentation}_"
        f"prior-{args.lidar_graph_prior}_"
        f"cross-{args.cross_modal_interaction}_"
        f"overlap-{args.overlap_metric}_"
        f"{cell_configuration_tag(args)}_"
        f"fdsm-{args.fdsm_scope}_"
        f"lidarmod-{args.lidar_modulation}_results.json"
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
