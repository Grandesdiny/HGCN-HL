import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from scipy.ndimage import laplace, uniform_filter
from scipy.sparse import (
    block_diag,
    coo_matrix,
    hstack,
    identity,
    issparse,
)
from skimage.segmentation import slic

from utils import get_HSI_LiDAR_data, get_HSI_performance, obtain_H_from_HSI_with_LiDAR


DATASET_CONFIG = {
    "muufl": {
        "loader_name": "MUUFL",
        "data_dir": Path("/root/hsi/MUUFL"),
        "pca_components": 10,
        "scales": [100],
    },
    "houston": {
        "loader_name": "Houston2013",
        "data_dir": Path("/root/hsi/dataset/Houston2013"),
        "pca_components": 25,
        "scales": [300],
    },
    "trento": {
        "loader_name": "Trento",
        "data_dir": Path("/root/hsi/dataset/Trento"),
        "pca_components": 10,
        "scales": [100],
    },
}


def normalized_sparse_assignments(assignment):
    if issparse(assignment):
        coo = assignment.tocoo()
        indices = torch.from_numpy(
            np.vstack([coo.row, coo.col])
        ).long()
        values = torch.from_numpy(coo.data.astype(np.float32))
        shape = coo.shape
    else:
        dense = torch.as_tensor(assignment, dtype=torch.float32)
        sparse = dense.to_sparse_coo().coalesce()
        indices = sparse.indices()
        values = sparse.values()
        shape = tuple(dense.shape)

    row_indices, column_indices = indices
    row_sum = torch.zeros(shape[0], dtype=torch.float32)
    column_sum = torch.zeros(shape[1], dtype=torch.float32)
    row_sum.index_add_(0, row_indices, values)
    column_sum.index_add_(0, column_indices, values)

    with torch.sparse.check_sparse_tensor_invariants():
        pooling = torch.sparse_coo_tensor(
            indices,
            values / column_sum[column_indices].clamp_min(1.0),
            shape,
        ).coalesce()
        projection = torch.sparse_coo_tensor(
            indices,
            values / row_sum[row_indices].clamp_min(1.0),
            shape,
        ).coalesce()
    return pooling, projection


def standardize_feature_cube(features):
    features = np.asarray(features, dtype=np.float32)
    flat_features = features.reshape(-1, features.shape[-1])
    mean = flat_features.mean(axis=0, keepdims=True)
    std = flat_features.std(axis=0, keepdims=True)
    standardized = (flat_features - mean) / np.maximum(std, 1e-6)
    return standardized.reshape(features.shape).astype(np.float32)


def build_lidar_geometry_features(lidar, local_window=5):
    """Create elevation, slope, curvature, and roughness channels."""
    if local_window <= 0 or local_window % 2 == 0:
        raise ValueError("LiDAR geometry window must be a positive odd integer.")

    elevation = np.asarray(lidar, dtype=np.float32)
    local_mean = uniform_filter(
        elevation,
        size=local_window,
        mode="reflect",
    )
    local_second_moment = uniform_filter(
        elevation * elevation,
        size=local_window,
        mode="reflect",
    )
    local_std = np.sqrt(
        np.maximum(local_second_moment - local_mean * local_mean, 0.0)
    )
    gradient_y, gradient_x = np.gradient(elevation)
    gradient_magnitude = np.sqrt(
        gradient_x * gradient_x + gradient_y * gradient_y
    )
    curvature = laplace(elevation, mode="reflect")
    geometry = np.stack(
        [
            elevation,
            elevation - local_mean,
            gradient_x,
            gradient_y,
            gradient_magnitude,
            curvature,
            local_std,
        ],
        axis=-1,
    )
    return standardize_feature_cube(geometry)


def segments_to_sparse_assignment(segments):
    flat_segments = np.asarray(segments, dtype=np.int64).reshape(-1)
    _, contiguous_segments = np.unique(
        flat_segments,
        return_inverse=True,
    )
    rows = np.arange(flat_segments.size, dtype=np.int64)
    values = np.ones(flat_segments.size, dtype=np.float32)
    return coo_matrix(
        (
            values,
            (rows, contiguous_segments.astype(np.int64)),
        ),
        shape=(
            flat_segments.size,
            int(contiguous_segments.max()) + 1,
        ),
        dtype=np.float32,
    ).tocsr(), contiguous_segments.reshape(segments.shape)


def build_rag_hop_candidates(segments, hops=2):
    """Return a sparse self-inclusive 1..hops region adjacency mask."""
    if hops < 1:
        raise ValueError("RAG hops must be at least 1.")

    node_count = int(segments.max()) + 1
    edge_left = []
    edge_right = []
    for first, second in (
        (segments[:, :-1], segments[:, 1:]),
        (segments[:-1, :], segments[1:, :]),
    ):
        boundary = first != second
        if np.any(boundary):
            edge_left.append(first[boundary].reshape(-1))
            edge_right.append(second[boundary].reshape(-1))

    if edge_left:
        left = np.concatenate(edge_left).astype(np.int64)
        right = np.concatenate(edge_right).astype(np.int64)
        rows = np.concatenate([left, right])
        columns = np.concatenate([right, left])
        adjacency = coo_matrix(
            (
                np.ones(rows.size, dtype=np.float32),
                (rows, columns),
            ),
            shape=(node_count, node_count),
        ).tocsr()
        adjacency.data[:] = 1.0
        adjacency.eliminate_zeros()
    else:
        adjacency = coo_matrix(
            (node_count, node_count),
            dtype=np.float32,
        ).tocsr()

    reachability = identity(
        node_count,
        dtype=np.float32,
        format="csr",
    )
    frontier = adjacency
    reachability = reachability.maximum(frontier)
    for _ in range(2, hops + 1):
        frontier = frontier @ adjacency
        if frontier.nnz:
            frontier.data[:] = 1.0
            frontier.eliminate_zeros()
        reachability = reachability.maximum(frontier)
    reachability.data[:] = 1.0
    return reachability.tocsr()


def build_common_boundary_strength(segments, gradient_magnitude):
    """Average DSM gradient along every adjacent region boundary."""
    node_count = int(segments.max()) + 1
    boundary_sum = np.zeros(
        (node_count, node_count),
        dtype=np.float32,
    )
    boundary_count = np.zeros(
        (node_count, node_count),
        dtype=np.float32,
    )
    for first, second, first_gradient, second_gradient in (
        (
            segments[:, :-1],
            segments[:, 1:],
            gradient_magnitude[:, :-1],
            gradient_magnitude[:, 1:],
        ),
        (
            segments[:-1, :],
            segments[1:, :],
            gradient_magnitude[:-1, :],
            gradient_magnitude[1:, :],
        ),
    ):
        boundary = first != second
        if not np.any(boundary):
            continue
        left = first[boundary].astype(np.int64)
        right = second[boundary].astype(np.int64)
        strength = (
            0.5
            * (
                first_gradient[boundary]
                + second_gradient[boundary]
            )
        ).astype(np.float32)
        np.add.at(boundary_sum, (left, right), strength)
        np.add.at(boundary_sum, (right, left), strength)
        np.add.at(boundary_count, (left, right), 1.0)
        np.add.at(boundary_count, (right, left), 1.0)

    boundary_strength = np.divide(
        boundary_sum,
        np.maximum(boundary_count, 1.0),
        out=np.zeros_like(boundary_sum),
        where=boundary_count > 0,
    )
    return coo_matrix(boundary_strength).tocsr()


def build_geometry_slic_structure(
    geometry_features,
    elevation,
    scales,
    compactness=0.1,
    rag_hops=2,
):
    """Build multi-scale Geometry-SLIC assignments and block RAG masks."""
    height, width, _ = geometry_features.shape
    assignments = []
    rag_candidates = []
    boundary_strengths = []
    elevation_gradient_y, elevation_gradient_x = np.gradient(
        np.asarray(elevation, dtype=np.float32)
    )
    elevation_gradient = np.sqrt(
        elevation_gradient_x * elevation_gradient_x
        + elevation_gradient_y * elevation_gradient_y
    )
    for scale in scales:
        segment_count = max(
            2,
            int(round(height * width / scale)),
        )
        segments = slic(
            geometry_features,
            n_segments=segment_count,
            compactness=compactness,
            sigma=1.0,
            convert2lab=False,
            enforce_connectivity=True,
            min_size_factor=0.1,
            max_size_factor=2.0,
            start_label=0,
            channel_axis=-1,
        )
        assignment, segments = segments_to_sparse_assignment(segments)
        assignments.append(assignment)
        rag_candidates.append(
            build_rag_hop_candidates(segments, hops=rag_hops)
        )
        boundary_strengths.append(
            build_common_boundary_strength(
                segments,
                elevation_gradient,
            )
        )

    return (
        hstack(assignments, format="csr"),
        block_diag(rag_candidates, format="csr"),
        block_diag(boundary_strengths, format="csr"),
    )


def superpixel_height_distribution(assignment, elevation):
    """Return q10/q25/q50/q75/q90/mean/std/range per region."""
    flat_elevation = np.asarray(
        elevation,
        dtype=np.float32,
    ).reshape(-1)
    assignment_csc = assignment.tocsc()
    descriptors = np.zeros(
        (assignment.shape[1], 8),
        dtype=np.float32,
    )
    quantiles = (10, 25, 50, 75, 90)
    for node_index in range(assignment.shape[1]):
        start = assignment_csc.indptr[node_index]
        end = assignment_csc.indptr[node_index + 1]
        pixel_indices = assignment_csc.indices[start:end]
        if pixel_indices.size == 0:
            continue
        region_height = flat_elevation[pixel_indices]
        descriptors[node_index, :5] = np.percentile(
            region_height,
            quantiles,
        )
        descriptors[node_index, 5] = region_height.mean()
        descriptors[node_index, 6] = region_height.std()
        descriptors[node_index, 7] = (
            region_height.max() - region_height.min()
        )

    descriptor_mean = descriptors.mean(axis=0, keepdims=True)
    descriptor_std = descriptors.std(axis=0, keepdims=True)
    standardized = (
        (descriptors - descriptor_mean)
        / np.maximum(descriptor_std, 1e-6)
    )
    return descriptors, standardized.astype(np.float32)


def superpixel_centroids(assignment, height, width):
    counts = np.asarray(
        assignment.sum(axis=0, dtype=np.float64)
    ).reshape(-1)
    counts = np.maximum(counts, 1.0)
    y_coordinates = np.repeat(
        np.arange(height, dtype=np.float32),
        width,
    )
    x_coordinates = np.tile(
        np.arange(width, dtype=np.float32),
        height,
    )
    center_y = np.asarray(
        assignment.T @ y_coordinates
    ).reshape(-1) / counts
    center_x = np.asarray(
        assignment.T @ x_coordinates
    ).reshape(-1) / counts
    return np.stack(
        [
            center_y / max(height - 1, 1),
            center_x / max(width - 1, 1),
        ],
        axis=1,
    ).astype(np.float32)


def robust_bandwidth(values):
    values = np.asarray(values, dtype=np.float32)
    positive = values[values > 0]
    return float(np.median(positive)) if positive.size else 1.0


def build_lidar_structure_prior(
    centroids,
    height_descriptors,
    boundary_strength,
    spatial_beta=1.0,
    height_beta=1.0,
    roughness_beta=1.0,
    boundary_beta=1.0,
):
    """Build P=A_xy^b1 A_z^b2 A_r^b3 A_b^b4."""
    delta_y = centroids[:, None, 0] - centroids[None, :, 0]
    delta_x = centroids[:, None, 1] - centroids[None, :, 1]
    spatial_squared = delta_y * delta_y + delta_x * delta_x
    spatial_sigma = robust_bandwidth(np.sqrt(spatial_squared))
    spatial_prior = np.exp(
        -spatial_squared / (2.0 * spatial_sigma * spatial_sigma)
    )

    mean_height = height_descriptors[:, 5]
    height_difference = (
        mean_height[:, None] - mean_height[None, :]
    )
    height_sigma = robust_bandwidth(np.abs(height_difference))
    height_prior = np.exp(
        -(height_difference * height_difference)
        / (2.0 * height_sigma * height_sigma)
    )

    # Region height standard deviation is the robust roughness statistic.
    roughness = height_descriptors[:, 6]
    roughness_difference = (
        roughness[:, None] - roughness[None, :]
    )
    roughness_sigma = robust_bandwidth(
        np.abs(roughness_difference)
    )
    roughness_prior = np.exp(
        -(roughness_difference * roughness_difference)
        / (2.0 * roughness_sigma * roughness_sigma)
    )

    boundary_dense = (
        boundary_strength.toarray()
        if issparse(boundary_strength)
        else np.asarray(boundary_strength)
    ).astype(np.float32)
    boundary_sigma = robust_bandwidth(boundary_dense)
    boundary_prior = np.exp(
        -boundary_dense / max(boundary_sigma, 1e-6)
    )

    structure_prior = (
        np.power(spatial_prior, spatial_beta)
        * np.power(height_prior, height_beta)
        * np.power(roughness_prior, roughness_beta)
        * np.power(boundary_prior, boundary_beta)
    )
    np.fill_diagonal(structure_prior, 1.0)
    return np.clip(
        structure_prior,
        1e-12,
        1.0,
    ).astype(np.float32)


def sinusoidal_position_encoding(length, channels):
    positions = torch.arange(length, dtype=torch.float32).unsqueeze(1)
    frequencies = torch.exp(
        torch.arange(0, channels, 2, dtype=torch.float32)
        * (-np.log(10000.0) / channels)
    )
    encoding = torch.zeros(length, channels, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(positions * frequencies)
    if channels > 1:
        encoding[:, 1::2] = torch.cos(
            positions * frequencies[: encoding[:, 1::2].shape[1]]
        )
    return encoding


class DynamicGraphBuilder(nn.Module):
    """Build the GSDG Q/K Top-k adjacency between superpixel nodes."""

    def __init__(
        self,
        channels,
        num_nodes,
        d_k=16,
        topk=8,
        tau=1.0,
        symmetrize=True,
    ):
        super().__init__()
        self.topk = topk
        self.tau = tau
        self.symmetrize = symmetrize
        self.query = nn.Linear(channels, d_k)
        self.key = nn.Linear(channels, d_k)
        self.scale = d_k ** -0.5
        self.register_buffer(
            "position_encoding",
            sinusoidal_position_encoding(num_nodes, d_k),
            persistent=False,
        )
        self.last_adjacency = None

    def forward(self, node_features, spatial_prior):
        query = self.query(node_features) + self.position_encoding
        key = self.key(node_features) + self.position_encoding
        logits = (
            query @ key.t() * self.scale
            + torch.log(spatial_prior + 1e-6)
        ) / self.tau
        scores = F.softmax(logits, dim=-1)

        k = min(self.topk, scores.size(1))
        values, indices = torch.topk(scores, k=k, dim=-1)
        adjacency = torch.zeros_like(scores).scatter_(
            dim=-1,
            index=indices,
            src=values,
        )
        if self.symmetrize:
            adjacency = torch.maximum(adjacency, adjacency.t())
        adjacency = adjacency / (
            adjacency.sum(dim=-1, keepdim=True) + 1e-6
        )
        self.last_adjacency = adjacency.detach()
        return adjacency


class GeometryDynamicGraphBuilder(nn.Module):
    """LiDAR dynamic graph with hard RAG and geometric penalties."""

    def __init__(
        self,
        channels,
        geometry_descriptors,
        rag_candidates,
        structure_prior,
        d_k=16,
        topk=8,
        tau=1.0,
        symmetrize=True,
    ):
        super().__init__()
        descriptors = torch.as_tensor(
            geometry_descriptors,
            dtype=torch.float32,
        )
        candidate_mask = torch.as_tensor(
            rag_candidates.toarray()
            if issparse(rag_candidates)
            else rag_candidates,
            dtype=torch.bool,
        )
        prior = torch.as_tensor(
            structure_prior,
            dtype=torch.float32,
        )
        self.register_buffer(
            "geometry_descriptors",
            descriptors,
            persistent=False,
        )
        self.register_buffer(
            "candidate_mask",
            candidate_mask,
            persistent=False,
        )
        self.register_buffer(
            "structure_prior",
            prior,
            persistent=False,
        )
        self.query = nn.Linear(channels, d_k)
        self.key = nn.Linear(channels, d_k)
        self.geometry_query = nn.Linear(descriptors.shape[1], d_k)
        self.geometry_key = nn.Linear(descriptors.shape[1], d_k)
        self.scale = d_k ** -0.5
        self.topk = topk
        self.tau = tau
        self.symmetrize = symmetrize
        self.last_adjacency = None

    def forward(self, node_features):
        query = (
            self.query(node_features)
            + self.geometry_query(self.geometry_descriptors)
        )
        key = (
            self.key(node_features)
            + self.geometry_key(self.geometry_descriptors)
        )
        logits = (
            query @ key.t() * self.scale
            + torch.log(self.structure_prior + 1e-6)
        ) / max(self.tau, 1e-6)
        logits = logits.masked_fill(
            ~self.candidate_mask,
            torch.finfo(logits.dtype).min,
        )
        scores = F.softmax(logits, dim=-1)

        k = min(self.topk, scores.size(1))
        values, indices = torch.topk(scores, k=k, dim=-1)
        adjacency = torch.zeros_like(scores).scatter_(
            dim=-1,
            index=indices,
            src=values,
        )
        adjacency = adjacency * self.candidate_mask.to(adjacency.dtype)
        if self.symmetrize:
            adjacency = torch.maximum(adjacency, adjacency.t())
        adjacency = adjacency / (
            adjacency.sum(dim=-1, keepdim=True) + 1e-6
        )
        self.last_adjacency = adjacency.detach()
        return adjacency


class GraphAttentionLayer(nn.Module):
    """Memory-efficient equivalent of the dense GSDG GAT layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        alpha=0.2,
        concat=True,
        use_edge_weights=False,
        edge_weight_beta=1.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.concat = concat
        self.use_edge_weights = use_edge_weights
        self.edge_weight_beta = edge_weight_beta
        self.weight = nn.Parameter(
            torch.empty(in_channels, out_channels)
        )
        self.attention = nn.Parameter(
            torch.empty(2 * out_channels, 1)
        )
        self.leaky_relu = nn.LeakyReLU(alpha)
        nn.init.xavier_uniform_(self.weight, gain=1.414)
        nn.init.xavier_uniform_(self.attention, gain=1.414)

    def forward(self, node_features, adjacency):
        transformed = node_features @ self.weight
        source_score = (
            transformed @ self.attention[: transformed.shape[1]]
        )
        target_score = (
            transformed @ self.attention[transformed.shape[1] :]
        )
        edge_score = self.leaky_relu(
            source_score + target_score.t()
        )
        if self.use_edge_weights:
            edge_score = (
                edge_score
                + self.edge_weight_beta
                * torch.log(adjacency.clamp_min(1e-12))
            )
        masked_score = edge_score.masked_fill(
            adjacency <= 0,
            torch.finfo(edge_score.dtype).min,
        )
        attention = F.softmax(masked_score, dim=1)
        attention = F.dropout(
            attention,
            self.dropout,
            training=self.training,
        )
        output = attention @ transformed
        return F.elu(output) if self.concat else output


class MultiHeadGAT(nn.Module):
    def __init__(
        self,
        in_channels,
        head_channels,
        out_channels,
        dropout,
        heads=4,
        alpha=0.2,
        use_edge_weights=False,
        edge_weight_beta=1.0,
    ):
        super().__init__()
        self.dropout = dropout
        self.heads = nn.ModuleList(
            [
                GraphAttentionLayer(
                    in_channels,
                    head_channels,
                    dropout=dropout,
                    alpha=alpha,
                    concat=True,
                    use_edge_weights=use_edge_weights,
                    edge_weight_beta=edge_weight_beta,
                )
                for _ in range(heads)
            ]
        )
        self.output_attention = GraphAttentionLayer(
            head_channels * heads,
            out_channels,
            dropout=dropout,
            alpha=alpha,
            concat=False,
            use_edge_weights=use_edge_weights,
            edge_weight_beta=edge_weight_beta,
        )

    def forward(self, node_features, adjacency):
        node_features = F.dropout(
            node_features,
            self.dropout,
            training=self.training,
        )
        node_features = torch.cat(
            [
                attention(node_features, adjacency)
                for attention in self.heads
            ],
            dim=1,
        )
        node_features = F.dropout(
            node_features,
            self.dropout,
            training=self.training,
        )
        return F.elu(
            self.output_attention(node_features, adjacency)
        )


class DynamicHyperedgeBuilder(nn.Module):
    """Build one GSDG-style neighborhood hyperedge per superpixel."""

    def __init__(
        self,
        channels,
        num_nodes,
        dynamic=False,
        d_k=16,
        topk=8,
        tau=1.0,
    ):
        super().__init__()
        self.dynamic = dynamic
        self.topk = topk
        self.tau = tau
        if dynamic:
            self.query = nn.Linear(channels, d_k)
            self.key = nn.Linear(channels, d_k)
            self.scale = d_k ** -0.5
            self.register_buffer(
                "position_encoding",
                sinusoidal_position_encoding(num_nodes, d_k),
                persistent=False,
            )

    def forward(self, node_features, spatial_prior):
        if self.dynamic:
            query = self.query(node_features) + self.position_encoding
            key = self.key(node_features) + self.position_encoding
            logits = (
                query @ key.t() * self.scale
                + torch.log(spatial_prior + 1e-6)
            ) / self.tau
            scores = F.softmax(logits, dim=-1)
        else:
            scores = spatial_prior / (
                spatial_prior.sum(dim=-1, keepdim=True) + 1e-6
            )

        k = min(self.topk, scores.size(1))
        values, indices = torch.topk(scores, k=k, dim=-1)
        incidence = torch.zeros_like(scores).scatter_(
            dim=-1,
            index=indices,
            src=values,
        )
        # Every row is a hyperedge centered on the same-index node.
        self_membership = torch.diag(values[:, 0])
        return torch.maximum(incidence, self_membership)


class ClassPrototypeHyperedgeBuilder(nn.Module):
    """Build HiH-style soft global hyperedges from class prototypes."""

    def __init__(
        self,
        channels,
        class_count,
        prototypes_per_class=3,
        temperature=0.1,
    ):
        super().__init__()
        self.global_edge_count = class_count * prototypes_per_class
        self.temperature = temperature
        self.norm = nn.LayerNorm(channels)
        self.prototypes = nn.Parameter(
            torch.empty(
                class_count,
                prototypes_per_class,
                channels,
            )
        )
        self.log_hyperedge_weight = nn.Parameter(
            torch.zeros(self.global_edge_count)
        )
        self.last_incidence = None
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, node_features):
        support = F.normalize(self.norm(node_features), dim=1)
        prototypes = F.normalize(
            self.prototypes.reshape(self.global_edge_count, -1),
            dim=1,
        )
        incidence = F.softmax(
            support @ prototypes.t() / self.temperature,
            dim=1,
        )
        self.last_incidence = incidence.detach()
        return incidence, torch.exp(self.log_hyperedge_weight)


class SuperpixelHGCN(nn.Module):
    """Weighted node-hyperedge-node propagation on superpixel vertices."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_superpixels,
        dynamic=False,
        dynamic_d_k=16,
        dynamic_topk=8,
        dynamic_tau=1.0,
        class_count=None,
        use_prototype_hyperedges=False,
        prototypes_per_class=3,
        prototype_temperature=0.1,
    ):
        super().__init__()
        self.linear = nn.Linear(
            in_channels,
            out_channels,
            bias=False,
        )
        self.hyperedge_weight = nn.Parameter(
            torch.ones(num_superpixels)
        )
        self.hyperedge_builder = DynamicHyperedgeBuilder(
            in_channels,
            num_superpixels,
            dynamic=dynamic,
            d_k=dynamic_d_k,
            topk=dynamic_topk,
            tau=dynamic_tau,
        )
        if use_prototype_hyperedges:
            if class_count is None or class_count <= 0:
                raise ValueError(
                    "class_count must be positive when prototype "
                    "hyperedges are enabled."
                )
            self.prototype_hyperedge_builder = (
                ClassPrototypeHyperedgeBuilder(
                    in_channels,
                    class_count,
                    prototypes_per_class=prototypes_per_class,
                    temperature=prototype_temperature,
                )
            )
            self.prototype_gamma = nn.Parameter(torch.tensor(1.0))
        else:
            self.prototype_hyperedge_builder = None
            self.register_parameter("prototype_gamma", None)

    def forward(self, node_features, spatial_prior):
        incidence = self.hyperedge_builder(
            node_features,
            spatial_prior,
        )
        transformed_features = self.linear(node_features)

        node_degree = (
            incidence.sum(dim=0).clamp_min(1e-6).pow(-0.5)
        )
        edge_degree = (
            incidence.sum(dim=1).clamp_min(1e-6).reciprocal()
        )
        normalized_nodes = (
            transformed_features * node_degree.unsqueeze(1)
        )
        edge_features = incidence @ normalized_nodes
        edge_features = (
            edge_features
            * edge_degree.unsqueeze(1)
            * self.hyperedge_weight.unsqueeze(1)
        )
        local_out = (
            incidence.t() @ edge_features
        ) * node_degree.unsqueeze(1)

        if self.prototype_hyperedge_builder is None:
            return local_out

        global_incidence, global_edge_weight = (
            self.prototype_hyperedge_builder(node_features)
        )
        global_edge_degree = (
            global_incidence.sum(dim=0).clamp_min(1e-6)
        )
        global_edge_features = (
            global_incidence.t() @ transformed_features
        )
        global_edge_features = (
            global_edge_features
            / global_edge_degree.unsqueeze(1)
            * global_edge_weight.unsqueeze(1)
        )
        global_node_degree = (
            global_incidence @ global_edge_weight.unsqueeze(1)
        ).squeeze(1).clamp_min(1e-6)
        global_out = (
            global_incidence @ global_edge_features
        ) / global_node_degree.unsqueeze(1)
        return local_out + self.prototype_gamma * global_out


class FDSM(nn.Module):
    """GSDG frequency-domain modulation over superpixel features."""

    def __init__(self, channels):
        super().__init__()
        frequency_components = channels // 2 + 1
        self.complex_weight = nn.Parameter(
            torch.randn(
                frequency_components,
                2,
                dtype=torch.float32,
            ) * 0.01
        )

    def forward(self, x):
        x = x.to(torch.float32)
        frequency_features = torch.fft.rfft(
            x,
            dim=-1,
            norm="ortho",
        )
        complex_weight = torch.view_as_complex(
            self.complex_weight
        ).view(1, -1)
        frequency_features = frequency_features * complex_weight
        return torch.fft.irfft(
            frequency_features,
            n=x.size(-1),
            dim=-1,
            norm="ortho",
        )


class WMF(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm2d(in_dim),
            nn.Conv2d(in_dim, out_dim, kernel_size=1),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DwsConv(nn.Module):
    """GSDG depthwise-separable convolution block."""

    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        self.depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels,
        )
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            groups=1,
            bias=False,
        )
        self.norm = nn.BatchNorm2d(in_channels)
        self.activation = nn.LeakyReLU()

    def forward(self, x):
        x = self.activation(self.pointwise(self.norm(x)))
        return self.activation(self.depthwise(x))


class OriginalSSConv(nn.Module):
    """The original HGCN-HL spatial-spectral CNN block."""

    def __init__(self, in_channels, out_channels, kernel_size=5):
        super().__init__()
        self.norm = nn.BatchNorm2d(in_channels)
        self.pointwise = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=1,
            bias=False,
        )
        self.depthwise = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=out_channels,
        )
        self.activation = nn.LeakyReLU()

        nn.init.kaiming_normal_(
            self.pointwise.weight,
            mode="fan_out",
            nonlinearity="leaky_relu",
        )
        nn.init.kaiming_normal_(
            self.depthwise.weight,
            mode="fan_out",
            nonlinearity="leaky_relu",
        )

    def forward(self, x):
        x = self.activation(self.pointwise(self.norm(x)))
        return self.activation(self.depthwise(x))


class ModalitySuperpixelBranch(nn.Module):
    def __init__(
        self,
        height,
        width,
        in_channels,
        assignment,
        superpixel_spatial_prior,
        backbone="hypergraph",
        graph_mode="dynamic",
        dynamic_d_k=16,
        dynamic_topk=8,
        dynamic_tau=1.0,
        class_count=None,
        use_prototype_hyperedges=False,
        prototypes_per_class=3,
        prototype_temperature=0.1,
        use_fdsm=False,
        cnn_style="gsdg",
        fusion_lambda=0.5,
        hidden_dim=128,
        graph_dim=64,
        dropout=0.4,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.backbone = backbone
        self.cnn_style = cnn_style
        self.fusion_lambda = fusion_lambda
        self.graph_mode = graph_mode
        self.output_dim = graph_dim
        self.frequency_modulation = (
            FDSM(hidden_dim) if use_fdsm else nn.Identity()
        )
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
            "superpixel_spatial_prior",
            torch.as_tensor(
                superpixel_spatial_prior,
                dtype=torch.float32,
            ),
            persistent=False,
        )
        num_superpixels = assignment.shape[1]

        self.feature_mapping = nn.Sequential(
            WMF(in_channels, hidden_dim),
            WMF(hidden_dim, hidden_dim),
        )
        if backbone == "hypergraph":
            hgcn_options = {
                "dynamic": graph_mode == "dynamic",
                "dynamic_d_k": dynamic_d_k,
                "dynamic_topk": dynamic_topk,
                "dynamic_tau": dynamic_tau,
                "class_count": class_count,
                "use_prototype_hyperedges": (
                    use_prototype_hyperedges
                ),
                "prototypes_per_class": prototypes_per_class,
                "prototype_temperature": prototype_temperature,
            }
            self.hgcn1 = SuperpixelHGCN(
                hidden_dim,
                graph_dim,
                num_superpixels,
                **hgcn_options,
            )
            self.hgcn2 = SuperpixelHGCN(
                graph_dim,
                graph_dim,
                num_superpixels,
                **hgcn_options,
            )
            self.dynamic_graph_builder = None
            self.gat1 = None
            self.gat2 = None
        else:
            self.hgcn1 = None
            self.hgcn2 = None
            self.dynamic_graph_builder = (
                DynamicGraphBuilder(
                    hidden_dim,
                    num_superpixels,
                    d_k=dynamic_d_k,
                    topk=dynamic_topk,
                    tau=dynamic_tau,
                )
                if graph_mode == "dynamic"
                else None
            )
            self.gat1 = MultiHeadGAT(
                hidden_dim,
                head_channels=30,
                out_channels=graph_dim,
                dropout=0.1,
                heads=4,
                alpha=0.2,
            )
            self.gat2 = MultiHeadGAT(
                graph_dim,
                head_channels=60,
                out_channels=graph_dim,
                dropout=0.2,
                heads=4,
                alpha=0.2,
            )

        if cnn_style == "gsdg":
            self.cnn_branch = nn.Sequential(
                DwsConv(
                    hidden_dim,
                    graph_dim,
                    kernel_size=3,
                ),
                DwsConv(
                    graph_dim,
                    graph_dim,
                    kernel_size=7,
                ),
            )
        else:
            self.cnn_branch = nn.Sequential(
                OriginalSSConv(
                    hidden_dim,
                    graph_dim,
                    kernel_size=5,
                ),
                OriginalSSConv(
                    graph_dim,
                    graph_dim,
                    kernel_size=5,
                ),
            )
        self.graph_projection = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.BatchNorm1d(self.output_dim),
            nn.LeakyReLU(),
        )
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        mapped = self.feature_mapping(x.permute(2, 0, 1).unsqueeze(0))

        cnn_features = self.cnn_branch(mapped)
        cnn_features = cnn_features.squeeze(0).permute(1, 2, 0).reshape(
            self.height * self.width,
            -1,
        )

        pixel_features = mapped.squeeze(0).permute(1, 2, 0).reshape(
            self.height * self.width,
            -1,
        )
        graph_features = torch.sparse.mm(
            self.pooling_assignment.transpose(0, 1),
            pixel_features,
        )
        graph_features = self.frequency_modulation(graph_features)

        if self.backbone == "hypergraph":
            graph_features = self.activation(
                self.dropout(graph_features)
            )
            first_graph_features = self.hgcn1(
                graph_features,
                self.superpixel_spatial_prior,
            )
            first_graph_features = self.activation(
                self.dropout(first_graph_features)
            )
            graph_features = self.hgcn2(
                first_graph_features,
                self.superpixel_spatial_prior,
            )
            if self.graph_mode == "dynamic":
                graph_features = (
                    graph_features + first_graph_features
                )
            graph_features = self.activation(
                self.dropout(graph_features)
            )
        else:
            if self.dynamic_graph_builder is None:
                adjacency = self.superpixel_spatial_prior
            else:
                adjacency = self.dynamic_graph_builder(
                    graph_features,
                    self.superpixel_spatial_prior,
                )
            first_graph_features = self.gat1(
                graph_features,
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

        features = (
            self.fusion_lambda * graph_features
            + (1.0 - self.fusion_lambda) * cnn_features
        )
        return features


class LiDARGeometryGSDGBranch(nn.Module):
    """Geometry-aware LiDAR GSDG with layer-wise graph rebuilding."""

    def __init__(
        self,
        height,
        width,
        in_channels,
        assignment,
        height_descriptors,
        geometry_descriptors,
        rag_candidates,
        structure_prior,
        dynamic_d_k=16,
        dynamic_topk=8,
        dynamic_tau=1.0,
        fusion_lambda=0.95,
        hidden_dim=128,
        graph_dim=64,
        edge_weight_beta=1.0,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.fusion_lambda = fusion_lambda
        _, projection_assignment = (
            normalized_sparse_assignments(assignment)
        )
        self.register_buffer(
            "projection_assignment",
            projection_assignment,
            persistent=False,
        )
        self.register_buffer(
            "height_descriptors",
            torch.as_tensor(
                height_descriptors,
                dtype=torch.float32,
            ),
            persistent=False,
        )

        self.feature_mapping = nn.Sequential(
            WMF(in_channels, hidden_dim),
            WMF(hidden_dim, hidden_dim),
        )
        self.height_descriptor_encoder = nn.Sequential(
            nn.Linear(height_descriptors.shape[1], hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(),
        )
        self.graph_builder1 = GeometryDynamicGraphBuilder(
            hidden_dim,
            geometry_descriptors,
            rag_candidates,
            structure_prior,
            d_k=dynamic_d_k,
            topk=dynamic_topk,
            tau=dynamic_tau,
        )
        self.gat1 = MultiHeadGAT(
            hidden_dim,
            head_channels=30,
            out_channels=graph_dim,
            dropout=0.1,
            heads=4,
            alpha=0.2,
            use_edge_weights=True,
            edge_weight_beta=edge_weight_beta,
        )
        self.graph_builder2 = GeometryDynamicGraphBuilder(
            graph_dim,
            geometry_descriptors,
            rag_candidates,
            structure_prior,
            d_k=dynamic_d_k,
            topk=dynamic_topk,
            tau=dynamic_tau,
        )
        self.gat2 = MultiHeadGAT(
            graph_dim,
            head_channels=60,
            out_channels=graph_dim,
            dropout=0.2,
            heads=4,
            alpha=0.2,
            use_edge_weights=True,
            edge_weight_beta=edge_weight_beta,
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(graph_dim, graph_dim),
            nn.BatchNorm1d(graph_dim),
            nn.LeakyReLU(),
        )

        self.cnn_scales = nn.ModuleList(
            [
                DwsConv(
                    hidden_dim,
                    graph_dim,
                    kernel_size=kernel_size,
                )
                for kernel_size in (3, 5, 7)
            ]
        )
        self.cnn_scale_fusion = nn.Sequential(
            nn.Conv2d(
                graph_dim * len(self.cnn_scales),
                graph_dim,
                kernel_size=1,
                bias=False,
            ),
            nn.BatchNorm2d(graph_dim),
            nn.LeakyReLU(),
        )

    def forward(self, lidar_geometry):
        mapped = self.feature_mapping(
            lidar_geometry.permute(2, 0, 1).unsqueeze(0)
        )
        cnn_features = self.cnn_scale_fusion(
            torch.cat(
                [branch(mapped) for branch in self.cnn_scales],
                dim=1,
            )
        )
        cnn_features = (
            cnn_features.squeeze(0)
            .permute(1, 2, 0)
            .reshape(self.height * self.width, -1)
        )

        node_features = self.height_descriptor_encoder(
            self.height_descriptors
        )

        adjacency1 = self.graph_builder1(node_features)
        first_graph_features = self.gat1(
            node_features,
            adjacency1,
        )
        adjacency2 = self.graph_builder2(first_graph_features)
        graph_features = (
            self.gat2(first_graph_features, adjacency2)
            + first_graph_features
        )
        graph_features = torch.sparse.mm(
            self.projection_assignment,
            graph_features,
        )
        graph_features = self.graph_projection(graph_features)
        return (
            self.fusion_lambda * graph_features
            + (1.0 - self.fusion_lambda) * cnn_features
        )


class DualBranchSuperpixelNetwork(nn.Module):
    def __init__(
        self,
        height,
        width,
        hsi_channels,
        lidar_channels,
        class_count,
        hsi_assignment,
        lidar_assignment,
        hsi_superpixel_spatial_prior,
        lidar_superpixel_spatial_prior,
        backbone="hypergraph",
        graph_mode="dynamic",
        dynamic_d_k=16,
        dynamic_topk=8,
        dynamic_tau=1.0,
        fdsm_scope="hsi",
        cnn_style="gsdg",
        prototype_scope="none",
        prototypes_per_class=3,
        prototype_temperature=0.1,
        fusion_lambda=0.5,
        hidden_dim=128,
        graph_dim=64,
        dropout=0.4,
    ):
        super().__init__()
        branch_options = {
            "height": height,
            "width": width,
            "backbone": backbone,
            "graph_mode": graph_mode,
            "dynamic_d_k": dynamic_d_k,
            "dynamic_topk": dynamic_topk,
            "dynamic_tau": dynamic_tau,
            "class_count": class_count,
            "prototypes_per_class": prototypes_per_class,
            "prototype_temperature": prototype_temperature,
            "cnn_style": cnn_style,
            "fusion_lambda": fusion_lambda,
            "hidden_dim": hidden_dim,
            "graph_dim": graph_dim,
            "dropout": dropout,
        }
        self.hsi_branch = ModalitySuperpixelBranch(
            in_channels=hsi_channels,
            assignment=hsi_assignment,
            superpixel_spatial_prior=hsi_superpixel_spatial_prior,
            use_fdsm=fdsm_scope in {"hsi", "both"},
            use_prototype_hyperedges=prototype_scope in {"hsi", "both"},
            **branch_options,
        )
        self.lidar_branch = ModalitySuperpixelBranch(
            in_channels=lidar_channels,
            assignment=lidar_assignment,
            superpixel_spatial_prior=lidar_superpixel_spatial_prior,
            use_fdsm=fdsm_scope == "both",
            use_prototype_hyperedges=prototype_scope in {"lidar", "both"},
            **branch_options,
        )
        self.classifier = nn.Linear(graph_dim * 2, class_count)

    def forward(
        self,
        hsi,
        lidar,
    ):
        hsi_features = self.hsi_branch(hsi)
        lidar_features = self.lidar_branch(lidar)
        fused_features = torch.cat(
            [hsi_features, lidar_features],
            dim=-1,
        )
        return self.classifier(fused_features)


class DualGSDGHGCNFusionNetwork(nn.Module):
    """Two complete GSDG encoders with HGCN-style modality fusion."""

    def __init__(
        self,
        height,
        width,
        hsi_channels,
        lidar_channels,
        class_count,
        hsi_assignment,
        lidar_assignment,
        hsi_superpixel_spatial_prior,
        lidar_height_descriptors,
        lidar_geometry_descriptors,
        lidar_rag_candidates,
        lidar_structure_prior,
        dynamic_d_k=16,
        dynamic_topk=8,
        dynamic_tau=1.0,
        gsdg_fusion_lambda=0.95,
        modality_fusion_lambda=0.5,
        hidden_dim=128,
        graph_dim=64,
        dropout=0.4,
        lidar_edge_weight_beta=1.0,
    ):
        super().__init__()
        self.modality_fusion_lambda = modality_fusion_lambda
        branch_options = {
            "height": height,
            "width": width,
            "backbone": "gsdg-graph",
            "graph_mode": "dynamic",
            "dynamic_d_k": dynamic_d_k,
            "dynamic_topk": dynamic_topk,
            "dynamic_tau": dynamic_tau,
            "class_count": class_count,
            "use_prototype_hyperedges": False,
            "use_fdsm": True,
            "cnn_style": "gsdg",
            "fusion_lambda": gsdg_fusion_lambda,
            "hidden_dim": hidden_dim,
            "graph_dim": graph_dim,
            "dropout": dropout,
        }
        self.hsi_gsdg = ModalitySuperpixelBranch(
            in_channels=hsi_channels,
            assignment=hsi_assignment,
            superpixel_spatial_prior=hsi_superpixel_spatial_prior,
            **branch_options,
        )
        self.lidar_gsdg = LiDARGeometryGSDGBranch(
            height=height,
            width=width,
            in_channels=lidar_channels,
            assignment=lidar_assignment,
            height_descriptors=lidar_height_descriptors,
            geometry_descriptors=lidar_geometry_descriptors,
            rag_candidates=lidar_rag_candidates,
            structure_prior=lidar_structure_prior,
            dynamic_d_k=dynamic_d_k,
            dynamic_topk=dynamic_topk,
            dynamic_tau=dynamic_tau,
            fusion_lambda=gsdg_fusion_lambda,
            hidden_dim=hidden_dim,
            graph_dim=graph_dim,
            edge_weight_beta=lidar_edge_weight_beta,
        )
        self.classifier = nn.Linear(graph_dim, class_count)

    def forward(self, hsi, lidar):
        hsi_features = self.hsi_gsdg(hsi)
        lidar_features = self.lidar_gsdg(lidar)
        fused_features = (
            self.modality_fusion_lambda * hsi_features
            + (1.0 - self.modality_fusion_lambda) * lidar_features
        )
        return self.classifier(fused_features)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train a dual-branch GSDG-style superpixel graph/hypergraph "
            "network "
            "with a fixed number "
            "of pixels per class."
        )
    )
    parser.add_argument(
        "--architecture",
        choices=(
            "dual-superpixel",
            "dual-gsdg-hgcn-fusion",
        ),
        default="dual-superpixel",
        help=(
            "Keep the configurable dual-superpixel model, or use HSI "
            "GSDG plus LiDAR Geometry-GSDG followed by HGCN-style "
            "weighted modality fusion."
        ),
    )
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_CONFIG),
        default="muufl",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Override the dataset's default directory.",
    )
    parser.add_argument("--train-samples-per-class", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--pca-components", type=int, default=None)
    parser.add_argument("--scales", type=int, nargs="+", default=None)
    parser.add_argument(
        "--lidar-segmentation",
        choices=("felzenszwalb", "slic"),
        default=None,
        help=(
            "Segmentation used to construct LiDAR superpixels. Defaults "
            "to Geometry-SLIC for dual-gsdg-hgcn-fusion and "
            "Felzenszwalb for the existing architecture."
        ),
    )
    parser.add_argument(
        "--backbone",
        choices=("hypergraph", "gsdg-graph"),
        default="hypergraph",
        help=(
            "Use GSDG-style dynamic neighborhood hyperedges with HGCN "
            "propagation, or the ordinary GSDG GAT graph."
        ),
    )
    parser.add_argument(
        "--graph-mode",
        "--hgcn-mode",
        dest="graph_mode",
        choices=("static", "dynamic"),
        default="dynamic",
        help=(
            "Use fixed spatial neighborhoods or GSDG Q/K Top-k "
            "dynamic neighborhoods. --hgcn-mode is a legacy alias."
        ),
    )
    parser.add_argument("--dynamic-dk", type=int, default=16)
    parser.add_argument("--dynamic-topk", type=int, default=8)
    parser.add_argument("--dynamic-tau", type=float, default=1.0)
    parser.add_argument("--spatial-prior-k", type=int, default=15)
    parser.add_argument(
        "--lidar-geometry-window",
        type=int,
        default=5,
        help="Odd local window for LiDAR residual and roughness features.",
    )
    parser.add_argument(
        "--lidar-slic-compactness",
        type=float,
        default=0.1,
        help="Compactness of Geometry-SLIC in the LiDAR GSDG branch.",
    )
    parser.add_argument(
        "--lidar-rag-hops",
        type=int,
        choices=(1, 2),
        default=2,
        help="Hard RAG candidate radius for LiDAR dynamic Top-k.",
    )
    parser.add_argument(
        "--lidar-spatial-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lidar-height-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lidar-boundary-weight",
        "--lidar-slope-weight",
        dest="lidar_boundary_weight",
        type=float,
        default=1.0,
        help=(
            "Exponent beta4 of the common-boundary gradient prior. "
            "--lidar-slope-weight is retained as a legacy alias."
        ),
    )
    parser.add_argument(
        "--lidar-roughness-weight",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lidar-edge-weight-beta",
        type=float,
        default=1.0,
        help="Strength of dynamic LiDAR edge weights in Weighted GAT.",
    )
    parser.add_argument(
        "--fdsm-scope",
        choices=("none", "hsi", "both"),
        default="hsi",
        help=(
            "Apply GSDG FDSM to no branch, the HSI branch, "
            "or both modality branches."
        ),
    )
    parser.add_argument(
        "--cnn-style",
        choices=("gsdg", "original"),
        default="gsdg",
        help=(
            "Use the GSDG 3x3/7x7 DwsConv path or the original "
            "HGCN-HL 5x5/5x5 SSConv path in both modalities."
        ),
    )
    parser.add_argument(
        "--prototype-scope",
        choices=("none", "hsi", "lidar", "both"),
        default="none",
        help=(
            "Add HiH class-prototype global hyperedges to selected "
            "branches of the hypergraph backbone."
        ),
    )
    parser.add_argument("--prototypes-per-class", type=int, default=3)
    parser.add_argument("--prototype-temperature", type=float, default=0.1)
    parser.add_argument(
        "--fusion-lambda",
        type=float,
        default=None,
        help=(
            "Graph/CNN feature fusion weight inside each modality. "
            "Defaults to 0.95 for the dual GSDG architecture and 0.5 "
            "for the existing architecture."
        ),
    )
    parser.add_argument(
        "--modality-fusion-lambda",
        type=float,
        default=0.5,
        help=(
            "HSI weight in the HGCN-style HSI/LiDAR weighted fusion "
            "used by dual-gsdg-hgcn-fusion."
        ),
    )
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--graph-dim", type=int, default=64)
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.4,
        help="Dropout around HGCN layers in the hypergraph backbone.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("model"))
    return parser.parse_args()


def resolve_dataset_options(args):
    config = DATASET_CONFIG[args.dataset]
    if args.data_dir is None:
        args.data_dir = config["data_dir"]
    if args.pca_components is None:
        args.pca_components = config["pca_components"]
    if args.scales is None:
        args.scales = config["scales"].copy()
    if args.lidar_segmentation is None:
        args.lidar_segmentation = (
            "slic"
            if args.architecture == "dual-gsdg-hgcn-fusion"
            else "felzenszwalb"
        )
    if args.fusion_lambda is None:
        args.fusion_lambda = (
            0.95
            if args.architecture == "dual-gsdg-hgcn-fusion"
            else 0.5
        )
    return config["loader_name"]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def minmax_normalize(x):
    x = x.astype(np.float32, copy=False)
    minimum = float(x.min())
    value_range = float(x.max()) - minimum
    if value_range == 0:
        return np.zeros_like(x)
    return (x - minimum) / value_range


def split_fixed_samples_per_class(gt, class_count, samples_per_class, seed):
    if samples_per_class <= 0:
        raise ValueError("--train-samples-per-class must be positive.")

    rng = np.random.default_rng(seed)
    flat_gt = gt.reshape(-1)
    train_indices = []
    test_indices = []

    for class_id in range(1, class_count + 1):
        class_indices = np.flatnonzero(flat_gt == class_id)
        if class_indices.size <= samples_per_class:
            raise ValueError(
                f"Class {class_id} has {class_indices.size} pixels, "
                f"which is not enough for {samples_per_class} training samples "
                "and at least one test sample."
            )
        selected = rng.choice(
            class_indices,
            size=samples_per_class,
            replace=False,
        )
        train_indices.append(selected)
        test_indices.append(np.setdiff1d(class_indices, selected, assume_unique=True))

    return np.concatenate(train_indices), np.concatenate(test_indices)


def build_superpixel_spatial_prior(
    assignment,
    height,
    width,
    neighbor_count,
):
    """Build a centroid-based spatial prior between superpixel nodes."""
    num_superpixels = assignment.shape[1]
    if num_superpixels == 1:
        return np.ones((1, 1), dtype=np.float32)

    y_coordinates = np.repeat(
        np.arange(height, dtype=np.float32),
        width,
    )
    x_coordinates = np.tile(
        np.arange(width, dtype=np.float32),
        height,
    )
    counts = np.asarray(
        assignment.sum(axis=0, dtype=np.float64)
    ).reshape(-1)
    counts = np.maximum(counts, 1.0)
    center_y = np.asarray(
        assignment.T @ y_coordinates
    ).reshape(-1) / counts
    center_x = np.asarray(
        assignment.T @ x_coordinates
    ).reshape(-1) / counts
    center_y /= max(height - 1, 1)
    center_x /= max(width - 1, 1)

    delta_y = center_y[:, None] - center_y[None, :]
    delta_x = center_x[:, None] - center_x[None, :]
    squared_distance = delta_y ** 2 + delta_x ** 2
    nonzero_distance = np.sqrt(
        squared_distance[squared_distance > 0]
    )
    sigma = (
        float(np.median(nonzero_distance))
        if nonzero_distance.size
        else 1.0
    )
    weights = np.exp(
        -squared_distance / (2.0 * sigma ** 2)
    ).astype(np.float32)
    np.fill_diagonal(weights, 0.0)

    k = max(1, min(neighbor_count, num_superpixels - 1))
    neighbor_indices = np.argpartition(
        -weights,
        kth=k - 1,
        axis=1,
    )[:, :k]
    rows = np.arange(num_superpixels)[:, None]
    prior = np.zeros_like(weights)
    prior[rows, neighbor_indices] = weights[rows, neighbor_indices]
    prior = np.maximum(prior, prior.T)
    prior /= prior.sum(axis=1, keepdims=True) + 1e-6
    return prior.astype(np.float32)


def prepare_superpixel_structure(assignment, height, width, args):
    spatial_prior = build_superpixel_spatial_prior(
        assignment,
        height,
        width,
        args.spatial_prior_k,
    )
    return {
        "assignment": assignment.astype(np.float32),
        "num_superpixels": assignment.shape[1],
        "spatial_prior": spatial_prior,
    }


def prepare_lidar_geometry_structure(lidar, args):
    geometry_features = build_lidar_geometry_features(
        lidar,
        local_window=args.lidar_geometry_window,
    )
    (
        assignment,
        rag_candidates,
        boundary_strength,
    ) = build_geometry_slic_structure(
        geometry_features,
        lidar,
        args.scales,
        compactness=args.lidar_slic_compactness,
        rag_hops=args.lidar_rag_hops,
    )
    (
        raw_height_descriptors,
        height_descriptors,
    ) = superpixel_height_distribution(
        assignment,
        lidar,
    )
    centroids = superpixel_centroids(
        assignment,
        lidar.shape[0],
        lidar.shape[1],
    )
    geometry_descriptors = np.concatenate(
        [height_descriptors, centroids],
        axis=1,
    ).astype(np.float32)
    structure_prior = build_lidar_structure_prior(
        centroids,
        raw_height_descriptors,
        boundary_strength,
        spatial_beta=args.lidar_spatial_weight,
        height_beta=args.lidar_height_weight,
        roughness_beta=args.lidar_roughness_weight,
        boundary_beta=args.lidar_boundary_weight,
    )
    return geometry_features, {
        "assignment": assignment.astype(np.float32),
        "num_superpixels": assignment.shape[1],
        "height_descriptors": height_descriptors,
        "raw_height_descriptors": raw_height_descriptors,
        "geometry_descriptors": geometry_descriptors,
        "rag_candidates": rag_candidates,
        "boundary_strength": boundary_strength,
        "structure_prior": structure_prior,
    }


def prepare_data(args):
    loader_name = DATASET_CONFIG[args.dataset]["loader_name"]
    hsi, lidar, gt, class_count, _, _ = get_HSI_LiDAR_data(
        loader_name,
        args.data_dir,
    )
    hsi = minmax_normalize(hsi)
    lidar = minmax_normalize(lidar)

    hsi_assignment, default_lidar_assignment = (
        obtain_H_from_HSI_with_LiDAR(
        hsi,
        lidar[:, :, np.newaxis],
        args.scales,
        lidar_segmentation=args.lidar_segmentation,
        return_separate=True,
        sparse_output=True,
        )
    )
    hsi_structure = prepare_superpixel_structure(
        hsi_assignment,
        hsi.shape[0],
        hsi.shape[1],
        args,
    )
    if args.architecture == "dual-gsdg-hgcn-fusion":
        lidar_features, lidar_structure = (
            prepare_lidar_geometry_structure(lidar, args)
        )
    else:
        lidar_structure = prepare_superpixel_structure(
            default_lidar_assignment,
            hsi.shape[0],
            hsi.shape[1],
            args,
        )
        lidar_features = lidar[:, :, np.newaxis].astype(np.float32)

    height, width, bands = hsi.shape
    component_count = min(args.pca_components, bands)
    reduced_hsi = PCA(
        n_components=component_count,
        random_state=args.seed,
    ).fit_transform(hsi.reshape(-1, bands))
    hsi_features = reduced_hsi.reshape(
        height,
        width,
        component_count,
    ).astype(np.float32)

    return (
        hsi_features,
        lidar_features,
        gt,
        class_count,
        hsi_structure,
        lidar_structure,
    )


def experiment_name(args):
    prefix = (
        f"{args.dataset.upper()}_{args.train_samples_per_class}px_"
    )
    if args.architecture == "dual-gsdg-hgcn-fusion":
        return (
            f"{prefix}dual_gsdg_geometry_lidar_hgcn_fusion_"
            f"rag-{args.lidar_rag_hops}hop_"
            f"topk-{args.dynamic_topk}_"
            f"gsdg-lambda-{args.fusion_lambda:g}_"
            f"modality-lambda-{args.modality_fusion_lambda:g}"
        )
    return (
        f"{prefix}dual_superpixel_{args.lidar_segmentation}_"
        f"{args.backbone}-{args.graph_mode}_"
        f"cnn-{args.cnn_style}_"
        f"fdsm-{args.fdsm_scope}_"
        f"prototype-{args.prototype_scope}"
    )


def train_one_run(
    args,
    hsi_features,
    lidar_features,
    gt,
    class_count,
    hsi_structure,
    lidar_structure,
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
    flat_gt = gt.reshape(-1)
    train_labels = torch.from_numpy(flat_gt[train_indices] - 1).long().to(device)
    test_labels = torch.from_numpy(flat_gt[test_indices] - 1).long().to(device)
    train_index_tensor = torch.from_numpy(train_indices).long().to(device)
    test_index_tensor = torch.from_numpy(test_indices).long().to(device)

    common_model_options = {
        "height": hsi_features.shape[0],
        "width": hsi_features.shape[1],
        "hsi_channels": hsi_features.shape[2],
        "lidar_channels": lidar_features.shape[2],
        "class_count": class_count,
        "hsi_assignment": hsi_structure["assignment"],
        "lidar_assignment": lidar_structure["assignment"],
        "hsi_superpixel_spatial_prior": (
            hsi_structure["spatial_prior"]
        ),
        "dynamic_d_k": args.dynamic_dk,
        "dynamic_topk": args.dynamic_topk,
        "dynamic_tau": args.dynamic_tau,
        "hidden_dim": args.hidden_dim,
        "graph_dim": args.graph_dim,
        "dropout": args.dropout,
    }
    if args.architecture == "dual-gsdg-hgcn-fusion":
        model = DualGSDGHGCNFusionNetwork(
            lidar_height_descriptors=(
                lidar_structure["height_descriptors"]
            ),
            lidar_geometry_descriptors=(
                lidar_structure["geometry_descriptors"]
            ),
            lidar_rag_candidates=(
                lidar_structure["rag_candidates"]
            ),
            lidar_structure_prior=(
                lidar_structure["structure_prior"]
            ),
            gsdg_fusion_lambda=args.fusion_lambda,
            modality_fusion_lambda=args.modality_fusion_lambda,
            lidar_edge_weight_beta=args.lidar_edge_weight_beta,
            **common_model_options,
        ).to(device)
    else:
        model = DualBranchSuperpixelNetwork(
            lidar_superpixel_spatial_prior=(
                lidar_structure["spatial_prior"]
            ),
            backbone=args.backbone,
            graph_mode=args.graph_mode,
            fdsm_scope=args.fdsm_scope,
            cnn_style=args.cnn_style,
            prototype_scope=args.prototype_scope,
            prototypes_per_class=args.prototypes_per_class,
            prototype_temperature=args.prototype_temperature,
            fusion_lambda=args.fusion_lambda,
            **common_model_options,
        ).to(device)
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
        logits = model(
            hsi_x,
            lidar_x,
        )
        loss = criterion(logits[train_index_tensor], train_labels)
        loss.backward()
        optimizer.step()

        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

        if epoch == 1 or epoch % args.log_interval == 0:
            train_predictions = logits[train_index_tensor].argmax(dim=1)
            train_accuracy = (
                (train_predictions == train_labels).float().mean().item()
            )
            print(
                f"Run {run_index + 1}/{args.runs} | "
                f"Epoch {epoch:4d}/{args.epochs} | "
                f"loss={loss.item():.6f} | train_OA={train_accuracy:.4f}"
            )

    training_time = time.perf_counter() - start_time
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(
            hsi_x,
            lidar_x,
        )
        predictions = logits[test_index_tensor].argmax(dim=1).cpu().numpy()

    truth = test_labels.cpu().numpy()
    oa, aa, kappa, class_accuracy, _ = get_HSI_performance(truth, predictions)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / (
        f"{experiment_name(args)}_run{run_index + 1}.pt"
    )
    torch.save(model.state_dict(), checkpoint)

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
    }


def main():
    args = parse_args()
    dataset_name = resolve_dataset_options(args)
    if args.epochs <= 0:
        raise ValueError("--epochs must be positive.")
    if args.runs <= 0:
        raise ValueError("--runs must be positive.")
    if not 0.0 <= args.fusion_lambda <= 1.0:
        raise ValueError("--fusion-lambda must be between 0 and 1.")
    if not 0.0 <= args.modality_fusion_lambda <= 1.0:
        raise ValueError(
            "--modality-fusion-lambda must be between 0 and 1."
        )
    if args.dynamic_dk <= 0:
        raise ValueError("--dynamic-dk must be positive.")
    if args.dynamic_topk <= 0:
        raise ValueError("--dynamic-topk must be positive.")
    if args.dynamic_tau <= 0:
        raise ValueError("--dynamic-tau must be positive.")
    if args.spatial_prior_k <= 0:
        raise ValueError("--spatial-prior-k must be positive.")
    if (
        args.lidar_geometry_window <= 0
        or args.lidar_geometry_window % 2 == 0
    ):
        raise ValueError(
            "--lidar-geometry-window must be a positive odd integer."
        )
    if args.lidar_slic_compactness <= 0:
        raise ValueError("--lidar-slic-compactness must be positive.")
    lidar_geometry_weights = (
        args.lidar_spatial_weight,
        args.lidar_height_weight,
        args.lidar_roughness_weight,
        args.lidar_boundary_weight,
        args.lidar_edge_weight_beta,
    )
    if any(weight < 0 for weight in lidar_geometry_weights):
        raise ValueError(
            "LiDAR geometry and edge-weight coefficients must be "
            "non-negative."
        )
    if args.hidden_dim <= 0:
        raise ValueError("--hidden-dim must be positive.")
    if args.graph_dim <= 0:
        raise ValueError("--graph-dim must be positive.")
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")
    if args.prototypes_per_class <= 0:
        raise ValueError("--prototypes-per-class must be positive.")
    if args.prototype_temperature <= 0:
        raise ValueError("--prototype-temperature must be positive.")
    if (
        args.architecture == "dual-superpixel"
        and
        args.backbone != "hypergraph"
        and args.prototype_scope != "none"
    ):
        raise ValueError(
            "--prototype-scope is only available with "
            "--backbone hypergraph."
        )
    if (
        args.architecture == "dual-gsdg-hgcn-fusion"
        and args.prototype_scope != "none"
    ):
        raise ValueError(
            "--prototype-scope is not part of the pure dual GSDG "
            "architecture; use --prototype-scope none."
        )
    if (
        args.architecture == "dual-gsdg-hgcn-fusion"
        and args.lidar_segmentation != "slic"
    ):
        raise ValueError(
            "dual-gsdg-hgcn-fusion uses Geometry-SLIC; set "
            "--lidar-segmentation slic."
        )
    if args.pca_components <= 0:
        raise ValueError("--pca-components must be positive.")
    if not args.scales or any(scale <= 0 for scale in args.scales):
        raise ValueError("--scales must contain positive integers.")

    set_seed(args.seed)
    (
        hsi_features,
        lidar_features,
        gt,
        class_count,
        hsi_structure,
        lidar_structure,
    ) = prepare_data(args)

    print(
        f"Dataset: {dataset_name} "
        f"({hsi_features.shape[0]} x {hsi_features.shape[1]})"
    )
    print(f"Data directory: {args.data_dir.resolve()}")
    print(f"LiDAR segmentation: {args.lidar_segmentation}")
    print(f"Architecture: {args.architecture}")
    if args.architecture == "dual-gsdg-hgcn-fusion":
        print(
            "Branches: HSI GSDG + LiDAR Geometry-GSDG"
        )
        print(
            "LiDAR Geometry-GSDG: Geometry-SLIC -> robust height "
            "quantiles/MLP -> "
            f"{args.lidar_rag_hops}-hop RAG -> "
            "log(Axy*Az*Ar*Ab) Top-k -> Weighted GAT1 -> rebuild "
            "-> Weighted GAT2 -> multi-scale CNN"
        )
        print(
            "LiDAR prior betas: "
            f"xy={args.lidar_spatial_weight}, "
            f"height={args.lidar_height_weight}, "
            f"roughness={args.lidar_roughness_weight}, "
            f"boundary={args.lidar_boundary_weight}"
        )
        print(
            "Fusion: HGCN-style weighted sum; "
            f"HSI lambda={args.modality_fusion_lambda}, "
            f"LiDAR lambda={1.0 - args.modality_fusion_lambda}"
        )
        print(
            f"Within-branch GSDG graph/CNN lambda={args.fusion_lambda}"
        )
    else:
        print(f"Architecture backbone: {args.backbone}")
        print(f"Topology mode: {args.graph_mode}")
        print(f"FDSM scope: {args.fdsm_scope}")
        print(f"CNN style: {args.cnn_style}")
    if (
        args.architecture == "dual-superpixel"
        and args.backbone == "hypergraph"
    ):
        print(
            "Propagation: weighted node -> hyperedge -> node; "
            f"prototype-scope={args.prototype_scope}"
        )
    elif args.architecture == "dual-superpixel":
        print("Propagation: ordinary graph multi-head GAT")
    if (
        args.architecture == "dual-gsdg-hgcn-fusion"
        or args.graph_mode == "dynamic"
    ):
        if args.architecture == "dual-gsdg-hgcn-fusion":
            print(
                f"Dynamic graph: d_k={args.dynamic_dk}, "
                f"top-k={args.dynamic_topk}, tau={args.dynamic_tau}, "
                f"LiDAR edge beta={args.lidar_edge_weight_beta}"
            )
        else:
            print(
                f"Dynamic graph: d_k={args.dynamic_dk}, "
                f"top-k={args.dynamic_topk}, tau={args.dynamic_tau}, "
                f"spatial-prior-k={args.spatial_prior_k}"
            )
    print(
        f"Split: {args.train_samples_per_class} training pixels/class "
        f"({args.train_samples_per_class * class_count} total)"
    )
    print(
        f"HSI: channels={hsi_features.shape[2]}, "
        f"graph-nodes={hsi_structure['num_superpixels']}"
    )
    print(
        f"LiDAR: channels={lidar_features.shape[2]}, "
        f"graph-nodes={lidar_structure['num_superpixels']}"
    )

    results = [
        train_one_run(
            args,
            hsi_features,
            lidar_features,
            gt,
            class_count,
            hsi_structure,
            lidar_structure,
            run_index,
        )
        for run_index in range(args.runs)
    ]

    summary = {}
    for metric in ("OA", "AA", "Kappa"):
        values = np.array([result[metric] for result in results])
        summary[metric] = {
            "mean": float(values.mean()),
            "std": float(values.std()),
        }

    output = {
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
        f"{experiment_name(args)}_results.json"
    )
    result_path.write_text(json.dumps(output, indent=2), encoding="utf-8")

    print("\nSummary")
    for metric, values in summary.items():
        print(f"{metric}: {values['mean']:.4f} ± {values['std']:.4f}")
    print(f"Results saved to: {result_path.resolve()}")


if __name__ == "__main__":
    main()
