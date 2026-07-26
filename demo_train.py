"""Stage-8 demo: private dual GSDG graphs plus mediator C-GAT.

The fixed hypergraph/HGCN path is replaced by GSDG graph/GAT propagation.
The default uses independent HSI and LiDAR graphs; the previous concatenated
node graph remains selectable. The LiDAR graph can additionally restrict its
dynamic neighbors with a local RAG and an elevation-similarity KNN. The
original joint CNN and fusion remain. Cross-modal interaction is optional:
the older post-GAT2 intersection-cell mediator C graph is retained, while the
newer path treats C as a cross-modal overlap bipartite graph over HSI/LiDAR
superpixel nodes and performs alternating propagation between private GAT
stages.
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
from scipy.sparse import coo_matrix, hstack
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
    parser.add_argument(
        "--topology-rewiring",
        choices=("none", "cell-veto-advocacy"),
        default="none",
        help=(
            "Cross-modal topology rewiring for private dynamic graphs. "
            "'cell-veto-advocacy' uses intersection cells to build "
            "opposite-modality evidence profiles and modulates Q/K "
            "graph logits without feature transport. Default: none."
        ),
    )
    parser.add_argument(
        "--topology-advocacy-k",
        type=int,
        default=0,
        help=(
            "Extra opposite-evidence KNN candidates used for advocacy "
            "rewiring when a hard support mask exists. Keep 0 for "
            "pure veto/logit modulation. Default: 0."
        ),
    )
    parser.add_argument(
        "--topology-advocacy-rag-hops",
        type=int,
        choices=(1, 2),
        default=2,
        help=(
            "Local LiDAR RAG support used only for advocacy candidates. "
            "This does not change --lidar-rag-hops for the base LiDAR "
            "graph. Default: 2."
        ),
    )
    parser.add_argument(
        "--topology-audit-side",
        choices=("both", "hsi", "lidar"),
        default="both",
        help=(
            "Which private graph is audited by opposite-modality cell "
            "evidence. 'hsi' means LiDAR evidence rewires HSI edges; "
            "'lidar' means HSI evidence rewires LiDAR edges. "
            "Default: both."
        ),
    )
    parser.add_argument(
        "--topology-evidence",
        choices=("learned", "physical"),
        default="learned",
        help=(
            "Opposite evidence profile used by topology rewiring. "
            "'learned' uses opposite encoded node features; 'physical' "
            "uses fixed LiDAR height/gradient statistics for HSI and "
            "fixed HSI PCA means for LiDAR. Default: learned."
        ),
    )
    parser.add_argument(
        "--topology-impurity-mode",
        choices=("none", "target", "source-temperature", "both"),
        default="target",
        help=(
            "How fragmentation affects rewired logits. 'target' "
            "penalizes impure message-source nodes; "
            "'source-temperature' flattens attention emitted by impure "
            "receiver/source rows; 'both' applies both. Default: target."
        ),
    )
    parser.add_argument(
        "--topology-freeze-advocacy",
        action="store_true",
        help=(
            "Freeze the advocacy alpha scalar. Use with "
            "--topology-advocacy-init 0 for strict veto-only ablation."
        ),
    )
    parser.add_argument(
        "--topology-veto-init",
        type=float,
        default=0.0,
        help=(
            "Initial beta for opposite-evidence distance veto. Zero "
            "keeps the initial logits equal to the base graph."
        ),
    )
    parser.add_argument(
        "--topology-advocacy-init",
        type=float,
        default=0.0,
        help=(
            "Initial alpha for opposite-evidence similarity advocacy. "
            "Zero keeps the initial logits equal to the base graph."
        ),
    )
    parser.add_argument(
        "--topology-impurity-init",
        type=float,
        default=0.0,
        help=(
            "Initial node fragmentation penalty read from the common "
            "refinement cells. Zero disables it at initialization."
        ),
    )
    parser.add_argument(
        "--cross-overlap-relation",
        choices=("none", "sparse"),
        default="none",
        help=(
            "Optional sparse cross-modal relation layer using only the "
            "nonzero HSI/LiDAR superpixel overlap matrix. It does not "
            "create explicit C nodes or a C-C graph. Default: none."
        ),
    )
    parser.add_argument(
        "--cross-overlap-stage",
        choices=("post-gat2", "post-gat", "postgat", "inter-gat", "alternating"),
        default="inter-gat",
        help=(
            "Where to apply --cross-overlap-relation sparse. "
            "'inter-gat' performs GAT1 -> overlap transport -> GAT2; "
            "'post-gat2'/'post-gat'/'postgat' performs a single late "
            "transport after GAT2; "
            "'alternating' also applies a second transport after GAT2. "
            "Default: inter-gat."
        ),
    )
    parser.add_argument(
        "--cross-overlap-message",
        choices=("fixed", "qk-prior"),
        default="qk-prior",
        help=(
            "Message weights on sparse overlap edges. 'fixed' uses "
            "coverage-normalized overlap; 'qk-prior' computes sparse "
            "edge Q/K attention regularized by log overlap coverage. "
            "Default: qk-prior."
        ),
    )
    parser.add_argument(
        "--cross-overlap-fusion",
        choices=("bilinear", "dual-channel"),
        default="dual-channel",
        help=(
            "Node update after sparse overlap transport. 'bilinear' "
            "uses inter-modal message plus multiplicative evidence; "
            "'dual-channel' additionally aggregates consensus/product "
            "and conflict/difference edge channels. Default: "
            "dual-channel."
        ),
    )
    parser.add_argument(
        "--cross-overlap-prior-weight",
        type=float,
        default=1.0,
        help=(
            "Weight on log overlap coverage in sparse qk-prior edge "
            "attention. Default: 1."
        ),
    )
    parser.add_argument(
        "--cross-overlap-gamma-init",
        type=float,
        default=0.1,
        help="Initial residual scale for the first sparse overlap update.",
    )
    parser.add_argument(
        "--cross-overlap-second-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scale for the second sparse overlap update "
            "when --cross-overlap-stage alternating. Default: 0."
        ),
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
        "--consensus-graph-transport-stage",
        choices=("post-gat2", "inter-gat", "alternating"),
        default="post-gat2",
        help=(
            "Where to apply intersection-mediator transport. "
            "'post-gat2' keeps the previous single late transport; "
            "'inter-gat' applies transport after GAT1 and before GAT2; "
            "'alternating' applies transport after GAT1 and again "
            "after GAT2. Default: post-gat2."
        ),
    )
    parser.add_argument(
        "--consensus-graph-transport-operator",
        choices=("transport", "sheaf", "dcsi", "cgsa"),
        default="transport",
        help=(
            "Operator used inside bidirectional intersection-mediator "
            "transport. 'transport' keeps the existing C-mediated "
            "Beta/QK message route. 'sheaf' treats intersection cells "
            "as edge stalks on the HSI-LiDAR bipartite graph and applies "
            "a cellular-sheaf diffusion half-step. 'dcsi' applies the "
            "dual-channel contextual sheaf interaction with consensus/"
            "conflict stalk reasoning. 'cgsa' applies context-gated "
            "sheaf alignment where the edge graph only calibrates the "
            "conflict-correction step size. Default: transport."
        ),
    )
    parser.add_argument(
        "--sheaf-restriction",
        choices=("diag", "lowrank"),
        default="diag",
        help=(
            "Restriction-map parameterization for "
            "--consensus-graph-transport-operator sheaf. Default: diag."
        ),
    )
    parser.add_argument(
        "--sheaf-rank",
        type=int,
        default=4,
        help="Low-rank restriction rank for --sheaf-restriction lowrank.",
    )
    parser.add_argument(
        "--sheaf-steps",
        type=int,
        default=1,
        help="Number of cellular-sheaf diffusion half-steps. Default: 1.",
    )
    parser.add_argument(
        "--sheaf-energy-weight",
        type=float,
        default=0.0,
        help=(
            "Optional weight for the sheaf Dirichlet energy regularizer. "
            "Use 0 to validate pure diffusion. Default: 0."
        ),
    )
    parser.add_argument(
        "--dcsi-edge-topk",
        type=int,
        default=8,
        help=(
            "Top-K edge-stalk neighbors used by DCSI contextual "
            "reasoning. Default: 8."
        ),
    )
    parser.add_argument(
        "--dcsi-consensus-mix",
        type=float,
        default=0.1,
        help="Mu_C for DCSI consensus-channel edge reasoning.",
    )
    parser.add_argument(
        "--dcsi-conflict-mix",
        type=float,
        default=0.1,
        help="Mu_D for DCSI conflict-channel edge reasoning.",
    )
    parser.add_argument(
        "--dcsi-semantic-gamma-init",
        type=float,
        default=0.1,
        help="Initial DCSI consensus writeback scale. Default: 0.1.",
    )
    parser.add_argument(
        "--dcsi-conflict-gamma-init",
        type=float,
        default=0.05,
        help="Initial DCSI conflict correction scale. Default: 0.05.",
    )
    parser.add_argument(
        "--cgsa-alpha-max",
        type=float,
        default=2.0,
        help=(
            "Maximum per-cell correction multiplier in CGSA. With "
            "zero-initialized step head and alpha-max=2, CGSA starts "
            "from alpha_e=1, matching plain sheaf step sizing. "
            "Default: 2."
        ),
    )
    parser.add_argument(
        "--cgsa-consensus-channel",
        action="store_true",
        help=(
            "Enable CGSA's optional zero-initialized consensus-increment "
            "channel. Keep off for direction A."
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
        "--consensus-graph-transport-second-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scale for the second transport round "
            "when --consensus-graph-transport-stage alternating. "
            "It is ignored by post-gat2 and inter-gat. "
            "Default: 0."
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
    parser.add_argument(
        "--cnn-layout",
        choices=("joint", "separate"),
        default="joint",
        help=(
            "Pixel CNN modality layout. 'joint' keeps the existing "
            "early-fusion HSI+LiDAR CNN; 'separate' uses independent "
            "HSI and LiDAR CNN branches and fuses their pixel features "
            "with --graph-modality-lambda. Default: joint."
        ),
    )
    parser.add_argument(
        "--cnn-share-weights",
        action="store_true",
        help=(
            "Share the CNN body between HSI and LiDAR when "
            "--cnn-layout separate is used. The modality-specific WMF "
            "input mappings remain separate because their channel "
            "counts differ. Default: disabled."
        ),
    )
    parser.add_argument("--fusion-lambda", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument(
        "--dummy-logit-dim",
        type=int,
        default=0,
        help=(
            "DuRM-style enlarged classifier output dimension. If set "
            "larger than the dataset class count, cross entropy is "
            "computed over all logits but train/test predictions use "
            "only the first real dataset classes. Default: 0 "
            "(disabled)."
        ),
    )
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
    sheaf_tag = ""
    if args.consensus_graph_transport_operator == "sheaf":
        sheaf_tag = (
            f"sh{args.sheaf_restriction}-"
            f"r{args.sheaf_rank}-"
            f"s{args.sheaf_steps}-"
            f"e{args.sheaf_energy_weight:g}-"
        )
    elif args.consensus_graph_transport_operator == "dcsi":
        sheaf_tag = (
            f"sh{args.sheaf_restriction}-"
            f"r{args.sheaf_rank}-"
            f"s{args.sheaf_steps}-"
            f"e{args.sheaf_energy_weight:g}-"
            f"et{args.dcsi_edge_topk}-"
            f"mu{args.dcsi_consensus_mix:g}-"
            f"{args.dcsi_conflict_mix:g}-"
            f"gm{args.dcsi_semantic_gamma_init:g}-"
            f"{args.dcsi_conflict_gamma_init:g}-"
        )
    elif args.consensus_graph_transport_operator == "cgsa":
        sheaf_tag = (
            f"sh{args.sheaf_restriction}-"
            f"r{args.sheaf_rank}-"
            f"s{args.sheaf_steps}-"
            f"e{args.sheaf_energy_weight:g}-"
            f"et{args.dcsi_edge_topk}-"
            f"amax{args.cgsa_alpha_max:g}-"
            f"cc{int(args.cgsa_consensus_channel)}-"
        )
    return (
        f"cg-{args.post_gat_consensus_graph}-k{anchor_tag}-"
        f"dk{args.bridge_attention_dk}-"
        f"top{args.bridge_attention_topk}-"
        f"w{args.consensus_graph_weight:g}-"
        f"f{args.consensus_graph_fusion}-"
        f"rg{args.consensus_graph_residual_init:g}-"
        f"cg{args.consensus_graph_c_gamma_init:g}-"
        f"tp{args.consensus_graph_transport}-"
        f"to{args.consensus_graph_transport_operator}-"
        f"{sheaf_tag}"
        f"tf{args.consensus_graph_transport_fusion}-"
        f"tm{args.consensus_graph_transport_message}-"
        f"ts{args.consensus_graph_transport_stage}-"
        f"tw{args.consensus_graph_transport_prior_weight:g}-"
        f"tl{args.consensus_graph_transport_lambda:g}-"
        f"tg{args.consensus_graph_transport_gamma_init:g}-"
        f"tg2{args.consensus_graph_transport_second_gamma_init:g}-"
        f"edge{args.consensus_graph_cell_edge}-"
        f"a{args.consensus_graph_spatial_prior_weight:g}-"
        f"{args.consensus_graph_hsi_prior_weight:g}-"
        f"{args.consensus_graph_lidar_prior_weight:g}"
    )


def dummy_logit_configuration_tag(args, class_count):
    if args.dummy_logit_dim <= 0:
        return "dummy-none"
    return f"dummy{args.dummy_logit_dim}-real{class_count}"


def topology_rewiring_configuration_tag(args):
    if args.topology_rewiring == "none":
        return "tw-none"
    return (
        f"tw-{args.topology_rewiring}-"
        f"side{args.topology_audit_side}-"
        f"ev{args.topology_evidence}-"
        f"im{args.topology_impurity_mode}-"
        f"ak{args.topology_advocacy_k}-"
        f"ar{args.topology_advocacy_rag_hops}-"
        f"fa{int(args.topology_freeze_advocacy)}-"
        f"v{args.topology_veto_init:g}-"
        f"a{args.topology_advocacy_init:g}-"
        f"i{args.topology_impurity_init:g}"
    )


def cross_overlap_configuration_tag(args):
    if args.cross_overlap_relation == "none":
        return "xo-none"
    return (
        f"xo-{args.cross_overlap_relation}-"
        f"st{args.cross_overlap_stage}-"
        f"msg{args.cross_overlap_message}-"
        f"fu{args.cross_overlap_fusion}-"
        f"pw{args.cross_overlap_prior_weight:g}-"
        f"g{args.cross_overlap_gamma_init:g}-"
        f"g2{args.cross_overlap_second_gamma_init:g}"
    )


def cnn_configuration_tag(args):
    share_tag = ""
    if args.cnn_layout == "separate":
        share_tag = f"-share{int(args.cnn_share_weights)}"
    return f"cnn-{args.cnn_layout}-{args.cnn_branch}{share_tag}"


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


class CellEvidenceRewiringDynamicGraphBuilder(DynamicGraphBuilder):
    """Q/K graph rewiring audited by opposite-modality cell evidence."""

    uses_topology_rewiring = True

    def __init__(
        self,
        *args,
        candidate_mask=None,
        advocacy_k=0,
        veto_init=0.0,
        advocacy_init=0.0,
        impurity_init=0.0,
        impurity_mode="target",
        freeze_advocacy=False,
        advocacy_candidate_mask=None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.advocacy_k = int(advocacy_k)
        self.impurity_mode = impurity_mode
        self.veto_beta = nn.Parameter(
            torch.tensor(float(veto_init), dtype=torch.float32)
        )
        self.advocacy_alpha = nn.Parameter(
            torch.tensor(float(advocacy_init), dtype=torch.float32)
        )
        if freeze_advocacy:
            self.advocacy_alpha.requires_grad_(False)
        self.impurity_eta = nn.Parameter(
            torch.tensor(float(impurity_init), dtype=torch.float32)
        )
        self.advocacy_prior_floor = nn.Parameter(
            torch.tensor(np.log(1e-6), dtype=torch.float32)
        )
        self.register_buffer(
            "candidate_mask",
            (
                torch.as_tensor(candidate_mask, dtype=torch.bool)
                if candidate_mask is not None
                else None
            ),
            persistent=False,
        )
        self.register_buffer(
            "advocacy_candidate_mask",
            (
                torch.as_tensor(
                    (
                        advocacy_candidate_mask.toarray()
                        if hasattr(advocacy_candidate_mask, "toarray")
                        else advocacy_candidate_mask
                    ),
                    dtype=torch.bool,
                )
                if advocacy_candidate_mask is not None
                else None
            ),
            persistent=False,
        )
        self.last_diagnostics = None

    def _advocacy_mask(self, similarity, base_mask):
        if self.advocacy_k <= 0:
            return torch.zeros_like(base_mask)
        node_count = similarity.shape[0]
        k = min(self.advocacy_k, max(node_count - 1, 1))
        eye = torch.eye(
            node_count,
            dtype=torch.bool,
            device=similarity.device,
        )
        allowed_mask = (
            torch.ones_like(base_mask, dtype=torch.bool)
            if self.advocacy_candidate_mask is None
            else self.advocacy_candidate_mask
        )
        allowed_mask = torch.logical_and(allowed_mask, ~base_mask)
        allowed_mask = torch.logical_and(allowed_mask, ~eye)
        scores = similarity.masked_fill(
            ~allowed_mask,
            torch.finfo(similarity.dtype).min,
        )
        _, indices = torch.topk(scores, k=k, dim=-1)
        mask = torch.zeros_like(base_mask)
        mask.scatter_(dim=-1, index=indices, value=True)
        mask = torch.logical_and(mask, allowed_mask)
        return mask

    def forward(
        self,
        node_features,
        spatial_prior,
        audit_context,
        fragmentation=None,
        audit_uncertainty=None,
    ):
        if audit_context.shape[0] != node_features.shape[0]:
            raise ValueError(
                "Audit context must have one row per target node."
            )
        query = self.query(node_features) + self.position_encoding
        key = self.key(node_features) + self.position_encoding
        qk_logits = query @ key.t() * self.scale
        prior_logits = torch.log(spatial_prior + 1e-6)
        base_raw_logits = qk_logits + prior_logits

        normalized_context = F.normalize(
            audit_context,
            p=2,
            dim=1,
            eps=1e-6,
        )
        similarity = normalized_context @ normalized_context.t()
        if audit_uncertainty is None:
            distance = torch.cdist(audit_context, audit_context, p=2)
            positive_distance = distance.detach()[
                distance.detach() > 0
            ]
            distance_scale = (
                positive_distance.mean()
                if positive_distance.numel()
                else distance.new_tensor(1.0)
            ).clamp_min(1e-6)
            normalized_distance = distance / distance_scale
        else:
            if audit_uncertainty.shape != audit_context.shape:
                raise ValueError(
                    "Audit uncertainty must match audit context shape."
                )
            delta = audit_context[:, None, :] - audit_context[None, :, :]
            scale = torch.sqrt(
                audit_uncertainty[:, None, :].pow(2)
                + audit_uncertainty[None, :, :].pow(2)
                + 1e-6
            )
            normalized_distance = torch.sqrt(
                torch.mean((delta / scale).pow(2), dim=-1) + 1e-6
            )

        modulation = (
            self.advocacy_alpha * similarity
            - self.veto_beta * normalized_distance
        )
        source_temperature = None
        if fragmentation is not None:
            if fragmentation.shape[0] != node_features.shape[0]:
                raise ValueError(
                    "Fragmentation must have one value per node."
                )
            if self.impurity_mode in ("target", "both"):
                modulation = (
                    modulation
                    - self.impurity_eta * fragmentation[None, :]
                )
            elif self.impurity_mode == "none":
                pass
            elif self.impurity_mode != "source-temperature":
                raise ValueError(
                    "Unsupported impurity mode: "
                    f"{self.impurity_mode}"
                )
            if self.impurity_mode in ("source-temperature", "both"):
                source_temperature = torch.exp(
                    self.impurity_eta * fragmentation[:, None]
                )

        base_mask = (
            torch.ones_like(base_raw_logits, dtype=torch.bool)
            if self.candidate_mask is None
            else self.candidate_mask
        )
        support_mask = base_mask
        nonbase_advocacy_mask = torch.zeros_like(base_mask)
        if (
            self.candidate_mask is not None
            and self.advocacy_k > 0
            and abs(float(self.advocacy_alpha.detach().item())) > 1e-12
        ):
            advocacy_mask = self._advocacy_mask(similarity, base_mask)
            nonbase_advocacy_mask = torch.logical_and(
                advocacy_mask,
                ~base_mask,
            )
            support_mask = torch.logical_or(base_mask, advocacy_mask)

        raw_logits = base_raw_logits + modulation
        if nonbase_advocacy_mask.any():
            advocacy_logits = (
                qk_logits
                + self.advocacy_prior_floor
                + modulation
            )
            raw_logits = torch.where(
                nonbase_advocacy_mask,
                advocacy_logits,
                raw_logits,
            )
        if source_temperature is not None:
            raw_logits = raw_logits / source_temperature.clamp_min(1e-6)
        logits = raw_logits / self.tau
        logits = logits.masked_fill(
            ~support_mask,
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
        adjacency = adjacency * support_mask.to(adjacency.dtype)
        if self.symmetrize:
            adjacency = torch.maximum(adjacency, adjacency.t())
        adjacency = adjacency * support_mask.to(adjacency.dtype)
        adjacency = adjacency / (
            adjacency.sum(dim=-1, keepdim=True) + 1e-6
        )
        fragmentation_mean = (
            float(fragmentation.detach().mean().item())
            if fragmentation is not None
            else 0.0
        )
        self.last_diagnostics = {
            "alpha": float(self.advocacy_alpha.detach().item()),
            "beta": float(self.veto_beta.detach().item()),
            "impurity": float(self.impurity_eta.detach().item()),
            "impurity_mode": self.impurity_mode,
            "source_temperature_mean": (
                float(source_temperature.detach().mean().item())
                if source_temperature is not None
                else 1.0
            ),
            "fragmentation_mean": fragmentation_mean,
            "context_distance_mean": float(
                normalized_distance.detach().mean().item()
            ),
            "support_density": float(
                support_mask.detach().float().mean().item()
            ),
            "advocacy_density": float(
                nonbase_advocacy_mask.detach().float().mean().item()
            ),
        }
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


def build_sparse_overlap_relation_data(hsi_assignment, lidar_assignment):
    """Build sparse HSI/LiDAR superpixel co-occurrence edges.

    Edges are the nonzero entries of M = Q_H^T Q_L. Directional edge
    weights are receiver-normalized coverages:
    H<-L uses M_ij / |S_i^H| and L<-H uses M_ij / |S_j^L|.
    """
    hsi_sparse = coo_matrix(hsi_assignment, dtype=np.float32).tocsr()
    lidar_sparse = coo_matrix(lidar_assignment, dtype=np.float32).tocsr()
    overlap = (hsi_sparse.transpose() @ lidar_sparse).tocoo()
    overlap.sum_duplicates()
    valid = overlap.data > 0
    h_index = overlap.row[valid].astype(np.int64)
    l_index = overlap.col[valid].astype(np.int64)
    overlap_values = overlap.data[valid].astype(np.float32)
    h_area = np.asarray(hsi_sparse.sum(axis=0)).reshape(-1).astype(np.float32)
    l_area = (
        np.asarray(lidar_sparse.sum(axis=0)).reshape(-1).astype(np.float32)
    )
    h_coverage = overlap_values / np.maximum(h_area[h_index], 1e-6)
    l_coverage = overlap_values / np.maximum(l_area[l_index], 1e-6)
    h_node_count = hsi_sparse.shape[1]
    l_node_count = lidar_sparse.shape[1]
    return {
        "h_index": h_index,
        "l_index": l_index,
        "overlap": overlap_values.astype(np.float32),
        "h_coverage": h_coverage.astype(np.float32),
        "l_coverage": l_coverage.astype(np.float32),
        "h_area": h_area.astype(np.float32),
        "l_area": l_area.astype(np.float32),
        "h_node_count": int(h_node_count),
        "l_node_count": int(l_node_count),
        "edge_count": int(overlap_values.size),
        "density": float(
            overlap_values.size / max(h_node_count * l_node_count, 1)
        ),
    }


def normalized_cell_fragmentation(
    cell_data,
    node_count,
    parent_key,
    coverage_key,
):
    """Entropy of how strongly the opposite partition cuts each node."""
    parent = np.asarray(cell_data[parent_key], dtype=np.int64)
    coverage = np.asarray(cell_data[coverage_key], dtype=np.float32)
    entropy = np.zeros(node_count, dtype=np.float32)
    cell_counts = np.zeros(node_count, dtype=np.float32)
    np.add.at(
        entropy,
        parent,
        -coverage * np.log(np.maximum(coverage, 1e-12)),
    )
    np.add.at(cell_counts, parent, 1.0)
    denominator = np.log(np.maximum(cell_counts, 2.0))
    fragmentation = np.divide(
        entropy,
        np.maximum(denominator, 1e-6),
        out=np.zeros_like(entropy),
        where=cell_counts > 1,
    )
    return np.clip(fragmentation, 0.0, 1.0).astype(np.float32)


def _cell_feature_mean(cell_index, cell_count, pixel_features):
    feature_dim = pixel_features.shape[1]
    sums = np.zeros((cell_count, feature_dim), dtype=np.float32)
    np.add.at(sums, cell_index, pixel_features.astype(np.float32))
    counts = np.bincount(
        cell_index,
        minlength=cell_count,
    ).astype(np.float32)
    return sums / np.maximum(counts[:, None], 1.0)


def _standardize_profile(mean_profile, uncertainty_profile):
    profile_mean = mean_profile.mean(axis=0, keepdims=True)
    profile_std = np.maximum(
        mean_profile.std(axis=0, keepdims=True),
        1e-6,
    )
    standardized = (mean_profile - profile_mean) / profile_std
    standardized_uncertainty = np.maximum(
        uncertainty_profile / profile_std,
        0.05,
    )
    return (
        standardized.astype(np.float32),
        standardized_uncertainty.astype(np.float32),
    )


def build_topology_physical_evidence(cell_data, hsi, lidar):
    """Fixed opposite-modality profiles for cell-based topology audit."""
    height, width, hsi_channels = hsi.shape
    cell_index = np.asarray(
        cell_data["pixel_cell_index"],
        dtype=np.int64,
    )
    cell_count = int(cell_data["cell_count"])
    hsi_parent = np.asarray(cell_data["hsi_parent"], dtype=np.int64)
    lidar_parent = np.asarray(cell_data["lidar_parent"], dtype=np.int64)
    hsi_coverage = np.asarray(
        cell_data["hsi_coverage"],
        dtype=np.float32,
    )
    lidar_coverage = np.asarray(
        cell_data["lidar_coverage"],
        dtype=np.float32,
    )

    flat_hsi = hsi.reshape(height * width, hsi_channels).astype(np.float32)
    hsi_cell_mean = _cell_feature_mean(
        cell_index,
        cell_count,
        flat_hsi,
    )

    lidar = lidar.astype(np.float32)
    grad_y, grad_x = np.gradient(lidar)
    lidar_gradient = np.sqrt(grad_y * grad_y + grad_x * grad_x)
    lidar_cell_mean = _cell_feature_mean(
        cell_index,
        cell_count,
        lidar.reshape(-1, 1),
    )
    lidar_cell_second = _cell_feature_mean(
        cell_index,
        cell_count,
        (lidar * lidar).reshape(-1, 1),
    )
    lidar_cell_std = np.sqrt(
        np.maximum(
            lidar_cell_second - lidar_cell_mean * lidar_cell_mean,
            0.0,
        )
    )
    lidar_cell_gradient = _cell_feature_mean(
        cell_index,
        cell_count,
        lidar_gradient.reshape(-1, 1).astype(np.float32),
    )
    lidar_cell_profile = np.concatenate(
        [lidar_cell_mean, lidar_cell_std, lidar_cell_gradient],
        axis=1,
    ).astype(np.float32)

    hsi_node_count = int(hsi_parent.max()) + 1
    lidar_node_count = int(lidar_parent.max()) + 1
    hsi_audit_mean = np.zeros(
        (hsi_node_count, lidar_cell_profile.shape[1]),
        dtype=np.float32,
    )
    lidar_audit_mean = np.zeros(
        (lidar_node_count, hsi_channels),
        dtype=np.float32,
    )
    np.add.at(
        hsi_audit_mean,
        hsi_parent,
        hsi_coverage[:, None] * lidar_cell_profile,
    )
    np.add.at(
        lidar_audit_mean,
        lidar_parent,
        lidar_coverage[:, None] * hsi_cell_mean,
    )

    hsi_audit_var = np.zeros_like(hsi_audit_mean)
    lidar_audit_var = np.zeros_like(lidar_audit_mean)
    np.add.at(
        hsi_audit_var,
        hsi_parent,
        hsi_coverage[:, None]
        * np.square(lidar_cell_profile - hsi_audit_mean[hsi_parent]),
    )
    np.add.at(
        lidar_audit_var,
        lidar_parent,
        lidar_coverage[:, None]
        * np.square(hsi_cell_mean - lidar_audit_mean[lidar_parent]),
    )
    hsi_audit_uncertainty = np.sqrt(np.maximum(hsi_audit_var, 0.0))
    lidar_audit_uncertainty = np.sqrt(np.maximum(lidar_audit_var, 0.0))
    hsi_audit_mean, hsi_audit_uncertainty = _standardize_profile(
        hsi_audit_mean,
        hsi_audit_uncertainty,
    )
    lidar_audit_mean, lidar_audit_uncertainty = _standardize_profile(
        lidar_audit_mean,
        lidar_audit_uncertainty,
    )
    return {
        "hsi_context": hsi_audit_mean,
        "hsi_uncertainty": hsi_audit_uncertainty,
        "lidar_context": lidar_audit_mean,
        "lidar_uncertainty": lidar_audit_uncertainty,
    }


















def inverse_softplus_scalar(value):
    """Return x where softplus(x) is approximately value."""
    value = float(max(value, 1e-8))
    return torch.log(torch.expm1(torch.tensor(value)))


def apply_edge_restriction(restriction, edge_features, transpose=False):
    mode, payload = restriction
    if mode == "diag":
        return edge_features * payload
    if mode != "lowrank":
        raise ValueError(f"Unsupported sheaf restriction mode: {mode}")
    left, right = payload
    if transpose:
        latent = torch.einsum("cdr,cd->cr", left, edge_features)
        return edge_features + torch.einsum("cdr,cr->cd", right, latent)
    latent = torch.einsum("cdr,cd->cr", right, edge_features)
    return edge_features + torch.einsum("cdr,cr->cd", left, latent)


class RestrictionMapGenerator(nn.Module):
    """Generate edge-conditioned sheaf restriction maps."""

    def __init__(self, descriptor_dim, channels, mode="diag", rank=4):
        super().__init__()
        self.mode = mode
        self.channels = channels
        self.rank = rank
        if mode == "diag":
            self.diag_head = nn.Sequential(
                nn.Linear(descriptor_dim, channels),
                nn.LeakyReLU(),
                nn.Linear(channels, channels),
            )
            nn.init.zeros_(self.diag_head[-1].weight)
            nn.init.zeros_(self.diag_head[-1].bias)
        elif mode == "lowrank":
            self.left_head = nn.Sequential(
                nn.Linear(descriptor_dim, channels),
                nn.LeakyReLU(),
                nn.Linear(channels, channels * rank),
            )
            self.right_head = nn.Sequential(
                nn.Linear(descriptor_dim, channels),
                nn.LeakyReLU(),
                nn.Linear(channels, channels * rank),
            )
            nn.init.normal_(self.left_head[-1].weight, std=1e-3)
            nn.init.zeros_(self.left_head[-1].bias)
            nn.init.zeros_(self.right_head[-1].weight)
            nn.init.zeros_(self.right_head[-1].bias)
        else:
            raise ValueError(
                "--sheaf-restriction must be 'diag' or 'lowrank'."
            )

    def forward(self, descriptor):
        if self.mode == "diag":
            scale = 1.0 + 0.5 * torch.tanh(self.diag_head(descriptor))
            return "diag", scale
        left = self.left_head(descriptor).view(
            descriptor.shape[0],
            self.channels,
            self.rank,
        )
        right = self.right_head(descriptor).view(
            descriptor.shape[0],
            self.channels,
            self.rank,
        )
        return "lowrank", (left, right)


class SheafConsensusDiffusion(nn.Module):
    """Cellular-sheaf half-step over HSI/LiDAR superpixel bipartite cells."""

    def __init__(
        self,
        channels,
        descriptor_dim,
        restriction="diag",
        rank=4,
        steps=1,
        alpha_init=0.1,
    ):
        super().__init__()
        self.channels = channels
        self.restriction = restriction
        self.rank = rank
        self.steps = steps
        self.h_restriction = RestrictionMapGenerator(
            descriptor_dim,
            channels,
            mode=restriction,
            rank=rank,
        )
        self.l_restriction = RestrictionMapGenerator(
            descriptor_dim,
            channels,
            mode=restriction,
            rank=rank,
        )
        initial_alpha = inverse_softplus_scalar(alpha_init)
        self.h_alpha_raw = nn.Parameter(initial_alpha.clone())
        self.l_alpha_raw = nn.Parameter(initial_alpha.clone())

    @staticmethod
    def _scatter_mean(edge_values, parent_index, node_count):
        sums = edge_values.new_zeros(node_count, edge_values.shape[1])
        sums.index_add_(0, parent_index, edge_values)
        counts = torch.bincount(
            parent_index,
            minlength=node_count,
        ).to(edge_values.dtype).clamp_min(1.0)
        return sums / counts.unsqueeze(1)

    def forward(
        self,
        hsi_nodes,
        lidar_nodes,
        h_parent_index,
        l_parent_index,
        descriptor,
    ):
        h_current = hsi_nodes
        l_current = lidar_nodes
        h_alpha = F.softplus(self.h_alpha_raw)
        l_alpha = F.softplus(self.l_alpha_raw)
        final_delta = None

        h_map = self.h_restriction(descriptor)
        l_map = self.l_restriction(descriptor)
        for _ in range(self.steps):
            h_edge = h_current.index_select(0, h_parent_index)
            l_edge = l_current.index_select(0, l_parent_index)
            h_stalk = apply_edge_restriction(h_map, h_edge)
            l_stalk = apply_edge_restriction(l_map, l_edge)
            delta = h_stalk - l_stalk
            final_delta = delta
            h_edge_grad = apply_edge_restriction(
                h_map,
                delta,
                transpose=True,
            )
            l_edge_grad = -apply_edge_restriction(
                l_map,
                delta,
                transpose=True,
            )
            h_grad = self._scatter_mean(
                h_edge_grad,
                h_parent_index,
                h_current.shape[0],
            )
            l_grad = self._scatter_mean(
                l_edge_grad,
                l_parent_index,
                l_current.shape[0],
            )
            h_current = h_current - h_alpha * h_grad
            l_current = l_current - l_alpha * l_grad

        energy = final_delta.pow(2).sum(dim=1)
        diagnostics = {
            "sheaf_restriction": self.restriction,
            "sheaf_rank": int(self.rank),
            "sheaf_steps": int(self.steps),
            "sheaf_alpha_h": float(h_alpha.detach().item()),
            "sheaf_alpha_l": float(l_alpha.detach().item()),
            "sheaf_energy_mean": float(energy.detach().mean().item()),
            "sheaf_energy_std": float(
                energy.detach().std(unbiased=False).item()
            ),
            "sheaf_delta_norm": float(
                final_delta.detach().norm(dim=1).mean().item()
            ),
        }
        return h_current, l_current, energy, diagnostics


class DualChannelContextualSheafInteraction(nn.Module):
    """Dual-channel contextual sheaf interaction over overlap edge stalks."""

    def __init__(
        self,
        channels,
        descriptor_dim,
        attribute_dim=0,
        attention_d_k=32,
        topk=8,
        restriction="lowrank",
        rank=4,
        steps=1,
        consensus_mix_init=0.1,
        conflict_mix_init=0.1,
        semantic_gamma_init=0.1,
        conflict_gamma_init=0.05,
    ):
        super().__init__()
        self.channels = channels
        self.restriction = restriction
        self.rank = rank
        self.steps = steps
        self.topk = topk
        self.scale = attention_d_k ** -0.5
        self.h_base = nn.Linear(channels, channels, bias=False)
        self.l_base = nn.Linear(channels, channels, bias=False)
        nn.init.eye_(self.h_base.weight)
        nn.init.eye_(self.l_base.weight)
        self.h_restriction = RestrictionMapGenerator(
            descriptor_dim,
            channels,
            mode=restriction,
            rank=rank,
        )
        self.l_restriction = RestrictionMapGenerator(
            descriptor_dim,
            channels,
            mode=restriction,
            rank=rank,
        )
        edge_token_dim = 4 * channels + attribute_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_token_dim, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.edge_query = nn.Linear(channels, attention_d_k, bias=False)
        self.edge_key = nn.Linear(channels, attention_d_k, bias=False)
        self.consensus_value = nn.Linear(channels, channels, bias=False)
        self.conflict_value = nn.Linear(channels, channels, bias=False)
        self.consensus_mix = nn.Parameter(
            torch.tensor(float(consensus_mix_init))
        )
        self.conflict_mix = nn.Parameter(
            torch.tensor(float(conflict_mix_init))
        )
        self.consensus_norm = nn.LayerNorm(channels)
        self.conflict_norm = nn.LayerNorm(channels)
        self.consensus_ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.LeakyReLU(),
            nn.Linear(2 * channels, channels),
        )
        self.conflict_ffn = nn.Sequential(
            nn.Linear(channels, 2 * channels),
            nn.LeakyReLU(),
            nn.Linear(2 * channels, channels),
        )
        self.consensus_ffn_norm = nn.LayerNorm(channels)
        self.conflict_ffn_norm = nn.LayerNorm(channels)
        self.h_update = nn.Linear(2 * channels, channels, bias=False)
        self.l_update = nn.Linear(2 * channels, channels, bias=False)
        h_gate_output = nn.Linear(channels, 1)
        l_gate_output = nn.Linear(channels, 1)
        nn.init.zeros_(h_gate_output.weight)
        nn.init.zeros_(l_gate_output.weight)
        nn.init.constant_(h_gate_output.bias, -3.0)
        nn.init.constant_(l_gate_output.bias, -3.0)
        self.h_gate = nn.Sequential(
            nn.Linear(5 * channels, channels),
            nn.LeakyReLU(),
            h_gate_output,
        )
        self.l_gate = nn.Sequential(
            nn.Linear(5 * channels, channels),
            nn.LeakyReLU(),
            l_gate_output,
        )
        self.h_update_norm = nn.LayerNorm(channels)
        self.l_update_norm = nn.LayerNorm(channels)
        self.semantic_gamma = nn.Parameter(
            torch.tensor(float(semantic_gamma_init))
        )
        self.conflict_gamma = nn.Parameter(
            torch.tensor(float(conflict_gamma_init))
        )

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
    def _scatter_mean(edge_values, parent_index, node_count):
        sums = edge_values.new_zeros(node_count, edge_values.shape[1])
        sums.index_add_(0, parent_index, edge_values)
        counts = torch.bincount(
            parent_index,
            minlength=node_count,
        ).to(edge_values.dtype).clamp_min(1.0)
        return sums / counts.unsqueeze(1)

    @staticmethod
    def _base_adjoint(base_layer, edge_values):
        return F.linear(edge_values, base_layer.weight.t())

    def _edge_attention(
        self,
        consensus,
        conflict,
        cell_attributes,
        spatial_prior,
        hsi_prior,
        lidar_prior,
        spatial_prior_weight,
        hsi_prior_weight,
        lidar_prior_weight,
    ):
        edge_parts = [
            consensus,
            conflict,
            torch.abs(conflict),
            consensus * conflict,
        ]
        if cell_attributes is not None:
            edge_parts.append(cell_attributes)
        edge_token = self.edge_encoder(torch.cat(edge_parts, dim=1))
        logits = (
            self.edge_query(edge_token)
            @ self.edge_key(edge_token).transpose(0, 1)
            * self.scale
        )
        logits = (
            logits
            + spatial_prior_weight
            * torch.log(spatial_prior.clamp_min(1e-6))
            + hsi_prior_weight
            * torch.log(hsi_prior.clamp_min(1e-6))
            + lidar_prior_weight
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
        attention = self._topk_softmax(masked_logits, self.topk)
        supported_logits = logits.detach()[support_mask]
        return edge_token, attention, support_mask, supported_logits

    def _restrict_pair(
        self,
        h_current,
        l_current,
        h_parent_index,
        l_parent_index,
        h_map,
        l_map,
    ):
        h_edge = h_current.index_select(0, h_parent_index)
        l_edge = l_current.index_select(0, l_parent_index)
        h_base_edge = self.h_base(h_edge)
        l_base_edge = self.l_base(l_edge)
        h_stalk = apply_edge_restriction(h_map, h_base_edge)
        l_stalk = apply_edge_restriction(l_map, l_base_edge)
        consensus = 0.5 * (h_stalk + l_stalk)
        conflict = h_stalk - l_stalk
        return consensus, conflict

    def _pullback_h(self, h_map, edge_values, h_parent_index, node_count):
        edge_grad = apply_edge_restriction(
            h_map,
            edge_values,
            transpose=True,
        )
        edge_grad = self._base_adjoint(self.h_base, edge_grad)
        return self._scatter_mean(edge_grad, h_parent_index, node_count)

    def _pullback_l(self, l_map, edge_values, l_parent_index, node_count):
        edge_grad = apply_edge_restriction(
            l_map,
            edge_values,
            transpose=True,
        )
        edge_grad = self._base_adjoint(self.l_base, edge_grad)
        return self._scatter_mean(edge_grad, l_parent_index, node_count)

    def forward(
        self,
        hsi_nodes,
        lidar_nodes,
        h_parent_index,
        l_parent_index,
        descriptor,
        cell_attributes,
        spatial_prior,
        hsi_prior,
        lidar_prior,
        spatial_prior_weight=1.0,
        hsi_prior_weight=0.5,
        lidar_prior_weight=0.5,
    ):
        h_current = hsi_nodes
        l_current = lidar_nodes
        h_map = self.h_restriction(descriptor)
        l_map = self.l_restriction(descriptor)
        diagnostics = {}
        final_energy = None

        for _ in range(self.steps):
            consensus, conflict = self._restrict_pair(
                h_current,
                l_current,
                h_parent_index,
                l_parent_index,
                h_map,
                l_map,
            )
            (
                edge_token,
                attention,
                support_mask,
                supported_logits,
            ) = self._edge_attention(
                consensus,
                conflict,
                cell_attributes,
                spatial_prior,
                hsi_prior,
                lidar_prior,
                spatial_prior_weight,
                hsi_prior_weight,
                lidar_prior_weight,
            )
            consensus_message = attention @ self.consensus_value(
                consensus
            )
            conflict_message = attention @ self.conflict_value(conflict)
            consensus_ctx = self.consensus_norm(
                consensus + self.consensus_mix * consensus_message
            )
            conflict_ctx = self.conflict_norm(
                conflict + self.conflict_mix * conflict_message
            )
            consensus_ctx = self.consensus_ffn_norm(
                consensus_ctx + self.consensus_ffn(consensus_ctx)
            )
            conflict_ctx = self.conflict_ffn_norm(
                conflict_ctx + self.conflict_ffn(conflict_ctx)
            )

            h_semantic = self._pullback_h(
                h_map,
                consensus_ctx,
                h_parent_index,
                h_current.shape[0],
            )
            l_semantic = self._pullback_l(
                l_map,
                consensus_ctx,
                l_parent_index,
                l_current.shape[0],
            )
            h_correction = -self._pullback_h(
                h_map,
                conflict_ctx,
                h_parent_index,
                h_current.shape[0],
            )
            l_correction = self._pullback_l(
                l_map,
                conflict_ctx,
                l_parent_index,
                l_current.shape[0],
            )

            h_gate = torch.sigmoid(
                self.h_gate(
                    torch.cat(
                        [
                            h_current,
                            h_semantic,
                            h_correction,
                            h_current * h_semantic,
                            torch.abs(h_correction),
                        ],
                        dim=1,
                    )
                )
            )
            l_gate = torch.sigmoid(
                self.l_gate(
                    torch.cat(
                        [
                            l_current,
                            l_semantic,
                            l_correction,
                            l_current * l_semantic,
                            torch.abs(l_correction),
                        ],
                        dim=1,
                    )
                )
            )
            h_update = self.h_update(
                torch.cat(
                    [
                        self.semantic_gamma * h_semantic,
                        self.conflict_gamma * h_correction,
                    ],
                    dim=1,
                )
            )
            l_update = self.l_update(
                torch.cat(
                    [
                        self.semantic_gamma * l_semantic,
                        self.conflict_gamma * l_correction,
                    ],
                    dim=1,
                )
            )
            h_current = self.h_update_norm(
                h_current + h_gate * h_update
            )
            l_current = self.l_update_norm(
                l_current + l_gate * l_update
            )
            final_energy = conflict.pow(2).sum(dim=1)
            diagnostics = {
                "dcsi_restriction": self.restriction,
                "dcsi_rank": int(self.rank),
                "dcsi_steps": int(self.steps),
                "dcsi_edge_topk": int(self.topk),
                "dcsi_attention_entropy": float(
                    self._row_entropy(attention).detach().mean().item()
                ),
                "dcsi_support_density": float(
                    support_mask.float().detach().mean().item()
                ),
                "dcsi_supported_logit_mean": float(
                    supported_logits.mean().item()
                ),
                "dcsi_supported_logit_std": float(
                    supported_logits.std(unbiased=False).item()
                ),
                "dcsi_consensus_mix": float(
                    self.consensus_mix.detach().item()
                ),
                "dcsi_conflict_mix": float(
                    self.conflict_mix.detach().item()
                ),
                "dcsi_semantic_gamma": float(
                    self.semantic_gamma.detach().item()
                ),
                "dcsi_conflict_gamma": float(
                    self.conflict_gamma.detach().item()
                ),
                "dcsi_h_gate_mean": float(h_gate.detach().mean().item()),
                "dcsi_l_gate_mean": float(l_gate.detach().mean().item()),
                "dcsi_consensus_norm": float(
                    consensus.detach().norm(dim=1).mean().item()
                ),
                "dcsi_conflict_norm": float(
                    conflict.detach().norm(dim=1).mean().item()
                ),
                "dcsi_edge_token_norm": float(
                    edge_token.detach().norm(dim=1).mean().item()
                ),
                "sheaf_energy_mean": float(
                    final_energy.detach().mean().item()
                ),
                "sheaf_energy_std": float(
                    final_energy.detach().std(unbiased=False).item()
                ),
            }

        return h_current, l_current, final_energy, diagnostics


class ContextGatedSheafAlignment(nn.Module):
    """Context-Gated Cellular Sheaf Alignment (CGSA).

    Intersection cells are treated as edge stalks on the HSI/LiDAR
    superpixel bipartite graph. The edge-stalk graph only predicts a
    scalar correction step size; the feature writeback remains strictly
    proportional to the measured sheaf conflict.
    """

    def __init__(
        self,
        channels,
        descriptor_dim,
        attribute_dim=0,
        attention_d_k=32,
        topk=8,
        restriction="lowrank",
        rank=4,
        steps=1,
        gamma_init=0.1,
        alpha_max=2.0,
        use_consensus_channel=False,
        consensus_channel_gamma_init=0.05,
    ):
        super().__init__()
        self.channels = channels
        self.restriction = restriction
        self.rank = rank
        self.steps = steps
        self.topk = topk
        self.alpha_max = float(alpha_max)
        self.use_consensus_channel = use_consensus_channel
        self.scale = attention_d_k ** -0.5

        self.h_base = nn.Linear(channels, channels, bias=False)
        self.l_base = nn.Linear(channels, channels, bias=False)
        nn.init.eye_(self.h_base.weight)
        nn.init.eye_(self.l_base.weight)
        self.h_restriction = RestrictionMapGenerator(
            descriptor_dim,
            channels,
            mode=restriction,
            rank=rank,
        )
        self.l_restriction = RestrictionMapGenerator(
            descriptor_dim,
            channels,
            mode=restriction,
            rank=rank,
        )

        edge_token_dim = 3 * channels + attribute_dim
        self.edge_encoder = nn.Sequential(
            nn.Linear(edge_token_dim, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.edge_query = nn.Linear(channels, attention_d_k, bias=False)
        self.edge_key = nn.Linear(channels, attention_d_k, bias=False)
        self.edge_value = nn.Linear(channels, channels, bias=False)
        self.step_head = nn.Linear(2 * channels, 1)
        nn.init.zeros_(self.step_head.weight)
        nn.init.zeros_(self.step_head.bias)

        self.h_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.l_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        if use_consensus_channel:
            self.consensus_out = nn.Linear(
                channels,
                channels,
                bias=False,
            )
            nn.init.zeros_(self.consensus_out.weight)
            self.c_gamma = nn.Parameter(
                torch.tensor(float(consensus_channel_gamma_init))
            )
        else:
            self.consensus_out = None
            self.c_gamma = None

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
    def _scatter_mean(edge_values, parent_index, node_count):
        sums = edge_values.new_zeros(node_count, edge_values.shape[1])
        sums.index_add_(0, parent_index, edge_values)
        counts = torch.bincount(
            parent_index,
            minlength=node_count,
        ).to(edge_values.dtype).clamp_min(1.0)
        return sums / counts.unsqueeze(1)

    @staticmethod
    def _base_adjoint(base_layer, edge_values):
        return F.linear(edge_values, base_layer.weight.t())

    def _edge_attention(
        self,
        consensus,
        conflict,
        cell_attributes,
        spatial_prior,
        hsi_prior,
        lidar_prior,
        spatial_prior_weight,
        hsi_prior_weight,
        lidar_prior_weight,
    ):
        token_parts = [consensus, conflict, torch.abs(conflict)]
        if cell_attributes is not None:
            token_parts.append(cell_attributes)
        edge_token = self.edge_encoder(torch.cat(token_parts, dim=1))
        logits = (
            self.edge_query(edge_token)
            @ self.edge_key(edge_token).transpose(0, 1)
            * self.scale
            + spatial_prior_weight
            * torch.log(spatial_prior.clamp_min(1e-6))
            + hsi_prior_weight
            * torch.log(hsi_prior.clamp_min(1e-6))
            + lidar_prior_weight
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
        attention = self._topk_softmax(masked_logits, self.topk)
        context = attention @ self.edge_value(edge_token)
        supported_logits = logits.detach()[support_mask]
        return edge_token, context, attention, support_mask, supported_logits

    def _pullback_h(self, h_map, edge_values, h_parent_index, node_count):
        edge_grad = apply_edge_restriction(
            h_map,
            edge_values,
            transpose=True,
        )
        edge_grad = self._base_adjoint(self.h_base, edge_grad)
        return self._scatter_mean(edge_grad, h_parent_index, node_count)

    def _pullback_l(self, l_map, edge_values, l_parent_index, node_count):
        edge_grad = apply_edge_restriction(
            l_map,
            edge_values,
            transpose=True,
        )
        edge_grad = self._base_adjoint(self.l_base, edge_grad)
        return self._scatter_mean(edge_grad, l_parent_index, node_count)

    def forward(
        self,
        hsi_nodes,
        lidar_nodes,
        h_parent_index,
        l_parent_index,
        descriptor,
        cell_attributes,
        spatial_prior,
        hsi_prior,
        lidar_prior,
        spatial_prior_weight=1.0,
        hsi_prior_weight=0.5,
        lidar_prior_weight=0.5,
    ):
        h_current = hsi_nodes
        l_current = lidar_nodes
        h_map = self.h_restriction(descriptor)
        l_map = self.l_restriction(descriptor)
        diagnostics = {}
        final_energy = None

        for _ in range(self.steps):
            h_edge = self.h_base(
                h_current.index_select(0, h_parent_index)
            )
            l_edge = self.l_base(
                l_current.index_select(0, l_parent_index)
            )
            h_stalk = apply_edge_restriction(h_map, h_edge)
            l_stalk = apply_edge_restriction(l_map, l_edge)
            conflict = h_stalk - l_stalk
            consensus = 0.5 * (h_stalk + l_stalk)

            (
                edge_token,
                context,
                attention,
                support_mask,
                supported_logits,
            ) = self._edge_attention(
                consensus,
                conflict,
                cell_attributes,
                spatial_prior,
                hsi_prior,
                lidar_prior,
                spatial_prior_weight,
                hsi_prior_weight,
                lidar_prior_weight,
            )
            alpha = self.alpha_max * torch.sigmoid(
                self.step_head(torch.cat([edge_token, context], dim=1))
            )
            scaled_conflict = alpha * conflict
            h_grad = self._pullback_h(
                h_map,
                scaled_conflict,
                h_parent_index,
                h_current.shape[0],
            )
            l_grad = self._pullback_l(
                l_map,
                scaled_conflict,
                l_parent_index,
                l_current.shape[0],
            )
            h_current = h_current - self.h_gamma * h_grad
            l_current = l_current + self.l_gamma * l_grad

            consensus_delta_norm = None
            if self.consensus_out is not None:
                consensus_delta = self.consensus_out(
                    attention @ consensus - consensus
                )
                h_current = h_current + self.c_gamma * self._pullback_h(
                    h_map,
                    consensus_delta,
                    h_parent_index,
                    h_current.shape[0],
                )
                l_current = l_current + self.c_gamma * self._pullback_l(
                    l_map,
                    consensus_delta,
                    l_parent_index,
                    l_current.shape[0],
                )
                consensus_delta_norm = float(
                    consensus_delta.detach().norm(dim=1).mean().item()
                )

            final_energy = conflict.pow(2).sum(dim=1)
            diagnostics = {
                "cgsa_restriction": self.restriction,
                "cgsa_rank": int(self.rank),
                "cgsa_steps": int(self.steps),
                "cgsa_edge_topk": int(self.topk),
                "cgsa_alpha_max": float(self.alpha_max),
                "cgsa_alpha_mean": float(alpha.detach().mean().item()),
                "cgsa_alpha_std": float(
                    alpha.detach().std(unbiased=False).item()
                ),
                "cgsa_alpha_min": float(alpha.detach().min().item()),
                "cgsa_alpha_max_observed": float(
                    alpha.detach().max().item()
                ),
                "cgsa_gamma_h": float(self.h_gamma.detach().item()),
                "cgsa_gamma_l": float(self.l_gamma.detach().item()),
                "cgsa_attention_entropy": float(
                    self._row_entropy(attention).detach().mean().item()
                ),
                "cgsa_support_density": float(
                    support_mask.float().detach().mean().item()
                ),
                "cgsa_supported_logit_mean": float(
                    supported_logits.mean().item()
                ),
                "cgsa_supported_logit_std": float(
                    supported_logits.std(unbiased=False).item()
                ),
                "cgsa_conflict_norm": float(
                    conflict.detach().norm(dim=1).mean().item()
                ),
                "cgsa_consensus_norm": float(
                    consensus.detach().norm(dim=1).mean().item()
                ),
                "cgsa_context_norm": float(
                    context.detach().norm(dim=1).mean().item()
                ),
                "cgsa_consensus_channel": bool(
                    self.consensus_out is not None
                ),
                "cgsa_consensus_delta_norm": consensus_delta_norm,
                "sheaf_energy_mean": float(
                    final_energy.detach().mean().item()
                ),
                "sheaf_energy_std": float(
                    final_energy.detach().std(unbiased=False).item()
                ),
            }
            if self.c_gamma is not None:
                diagnostics["cgsa_gamma_c"] = float(
                    self.c_gamma.detach().item()
                )

        return h_current, l_current, final_energy, diagnostics


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
        transport_operator="transport",
        transport_prior_weight=1.0,
        transport_gamma_init=0.0,
        transport_second_gamma_init=0.0,
        sheaf_restriction="diag",
        sheaf_rank=4,
        sheaf_steps=1,
        dcsi_edge_topk=8,
        dcsi_consensus_mix=0.1,
        dcsi_conflict_mix=0.1,
        dcsi_semantic_gamma_init=0.1,
        dcsi_conflict_gamma_init=0.05,
        cgsa_alpha_max=2.0,
        cgsa_consensus_channel=False,
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
        self.transport_operator = transport_operator
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
        self.h_transport_gamma2 = nn.Parameter(
            torch.tensor(float(transport_second_gamma_init))
        )
        self.l_transport_gamma2 = nn.Parameter(
            torch.tensor(float(transport_second_gamma_init))
        )
        descriptor_channels = 5 * channels + attribute_channels
        self.sheaf_diffusion = None
        if transport_operator == "sheaf":
            self.sheaf_diffusion = SheafConsensusDiffusion(
                channels,
                descriptor_channels,
                restriction=sheaf_restriction,
                rank=sheaf_rank,
                steps=sheaf_steps,
                alpha_init=transport_gamma_init,
            )
        self.dcsi_interaction = None
        if transport_operator == "dcsi":
            self.dcsi_interaction = (
                DualChannelContextualSheafInteraction(
                    channels,
                    descriptor_channels,
                    attribute_dim=attribute_channels,
                    attention_d_k=attention_d_k,
                    topk=dcsi_edge_topk,
                    restriction=sheaf_restriction,
                    rank=sheaf_rank,
                    steps=sheaf_steps,
                    consensus_mix_init=dcsi_consensus_mix,
                    conflict_mix_init=dcsi_conflict_mix,
                    semantic_gamma_init=dcsi_semantic_gamma_init,
                    conflict_gamma_init=dcsi_conflict_gamma_init,
                )
            )
        self.cgsa_interaction = None
        if transport_operator == "cgsa":
            self.cgsa_interaction = ContextGatedSheafAlignment(
                channels,
                descriptor_channels,
                attribute_dim=attribute_channels,
                attention_d_k=attention_d_k,
                topk=dcsi_edge_topk,
                restriction=sheaf_restriction,
                rank=sheaf_rank,
                steps=sheaf_steps,
                gamma_init=transport_gamma_init,
                alpha_max=cgsa_alpha_max,
                use_consensus_channel=cgsa_consensus_channel,
                consensus_channel_gamma_init=dcsi_conflict_gamma_init,
            )
        self.mediator_kind = bridge_data.get(
            "mediator_kind",
            "center",
        )

        prior_ch_array = np.asarray(bridge_data["prior_ch"])
        prior_cl_array = np.asarray(bridge_data["prior_cl"])
        self.register_buffer(
            "h_parent_index",
            torch.as_tensor(
                prior_ch_array.argmax(axis=1),
                dtype=torch.long,
            ),
            persistent=False,
        )
        self.register_buffer(
            "l_parent_index",
            torch.as_tensor(
                prior_cl_array.argmax(axis=1),
                dtype=torch.long,
            ),
            persistent=False,
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
        self.last_sheaf_energy_loss = None

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

    def _build_sheaf_descriptor(self, hsi_nodes, lidar_nodes):
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
        descriptor_parts = [
            h_context,
            l_context,
            incidence_context,
            torch.abs(h_context - l_context),
            h_context * l_context,
        ]
        if self.cell_attributes is not None:
            descriptor_parts.append(self.cell_attributes)
        return torch.cat(descriptor_parts, dim=1)

    def forward(self, hsi_nodes, lidar_nodes, hsi_adjacency, lidar_adjacency):
        c_graph = self._build_c_graph(
            hsi_nodes,
            lidar_nodes,
            hsi_adjacency,
            lidar_adjacency,
        )
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
        consensus_pixel_features = torch.sparse.mm(
            self.bridge_projection_assignment,
            updated_bridge,
        )
        consensus_pixel_features = self.graph_projection(
            consensus_pixel_features
        )
        self.last_diagnostics = {
            **c_graph["diagnostics"],
            "transport_mode": "none",
            "c_gat_output_norm": float(
                c_gat_output.detach().norm(dim=1).mean().item()
            ),
            "c_ffn_delta_norm": float(
                c_ffn_delta.detach().norm(dim=1).mean().item()
            ),
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
        round_index=1,
    ):
        if round_index == 1:
            h_gamma = self.h_transport_gamma
            l_gamma = self.l_transport_gamma
        elif round_index == 2:
            h_gamma = self.h_transport_gamma2
            l_gamma = self.l_transport_gamma2
        else:
            raise ValueError("round_index must be 1 or 2.")
        self.last_sheaf_energy_loss = None
        if self.transport_operator == "sheaf":
            if self.sheaf_diffusion is None:
                raise ValueError("Sheaf diffusion module is not initialized.")
            sheaf_descriptor = self._build_sheaf_descriptor(
                hsi_nodes,
                lidar_nodes,
            )
            (
                updated_hsi,
                updated_lidar,
                energy,
                sheaf_diagnostics,
            ) = self.sheaf_diffusion(
                hsi_nodes,
                lidar_nodes,
                self.h_parent_index,
                self.l_parent_index,
                sheaf_descriptor,
            )
            self.last_sheaf_energy_loss = (
                energy.mean() / float(self.channels)
            )
            self.last_diagnostics = {
                "transport_operator": "sheaf",
                "transport_mode": "bidirectional",
                "transport_stage": getattr(
                    self,
                    "active_transport_stage",
                    "post-gat2",
                ),
                "transport_round": round_index,
                "transport_fusion": "sheaf-diffusion",
                "transport_message": "sheaf-restriction",
                "h_transport_gamma": sheaf_diagnostics["sheaf_alpha_h"],
                "l_transport_gamma": sheaf_diagnostics["sheaf_alpha_l"],
                "bridge_count": int(self.prior_ch.shape[0]),
                "mediator_kind": self.mediator_kind,
                **sheaf_diagnostics,
            }
            return updated_hsi, updated_lidar

        if self.transport_operator == "cgsa":
            if self.cgsa_interaction is None:
                raise ValueError("CGSA module is not initialized.")
            sheaf_descriptor = self._build_sheaf_descriptor(
                hsi_nodes,
                lidar_nodes,
            )
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
            (
                updated_hsi,
                updated_lidar,
                energy,
                cgsa_diagnostics,
            ) = self.cgsa_interaction(
                hsi_nodes,
                lidar_nodes,
                self.h_parent_index,
                self.l_parent_index,
                sheaf_descriptor,
                self.cell_attributes,
                spatial_prior,
                hsi_prior,
                lidar_prior,
                spatial_prior_weight=self.spatial_prior_weight,
                hsi_prior_weight=self.hsi_prior_weight,
                lidar_prior_weight=self.lidar_prior_weight,
            )
            self.last_sheaf_energy_loss = (
                energy.mean() / float(self.channels)
            )
            self.last_diagnostics = {
                "transport_operator": "cgsa",
                "transport_mode": "bidirectional",
                "transport_stage": getattr(
                    self,
                    "active_transport_stage",
                    "post-gat2",
                ),
                "transport_round": round_index,
                "transport_fusion": "context-gated-sheaf-alignment",
                "transport_message": "context-gated-conflict",
                "bridge_count": int(self.prior_ch.shape[0]),
                "mediator_kind": self.mediator_kind,
                "prior_weights": (
                    float(self.spatial_prior_weight),
                    float(self.hsi_prior_weight),
                    float(self.lidar_prior_weight),
                ),
                **cgsa_diagnostics,
            }
            return updated_hsi, updated_lidar

        if self.transport_operator == "dcsi":
            if self.dcsi_interaction is None:
                raise ValueError("DCSI module is not initialized.")
            sheaf_descriptor = self._build_sheaf_descriptor(
                hsi_nodes,
                lidar_nodes,
            )
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
            (
                updated_hsi,
                updated_lidar,
                energy,
                dcsi_diagnostics,
            ) = self.dcsi_interaction(
                hsi_nodes,
                lidar_nodes,
                self.h_parent_index,
                self.l_parent_index,
                sheaf_descriptor,
                self.cell_attributes,
                spatial_prior,
                hsi_prior,
                lidar_prior,
                spatial_prior_weight=self.spatial_prior_weight,
                hsi_prior_weight=self.hsi_prior_weight,
                lidar_prior_weight=self.lidar_prior_weight,
            )
            self.last_sheaf_energy_loss = (
                energy.mean() / float(self.channels)
            )
            self.last_diagnostics = {
                "transport_operator": "dcsi",
                "transport_mode": "bidirectional",
                "transport_stage": getattr(
                    self,
                    "active_transport_stage",
                    "post-gat2",
                ),
                "transport_round": round_index,
                "transport_fusion": "dual-channel-contextual-sheaf",
                "transport_message": "edge-stalk-reasoning",
                "bridge_count": int(self.prior_ch.shape[0]),
                "mediator_kind": self.mediator_kind,
                "prior_weights": (
                    float(self.spatial_prior_weight),
                    float(self.hsi_prior_weight),
                    float(self.lidar_prior_weight),
                ),
                **dcsi_diagnostics,
            }
            return updated_hsi, updated_lidar

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
        transport_kernel = (
            (1.0 - self.transport_lambda) * identity
            + self.transport_lambda * attention_cc
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
                + h_gamma * h_gate * l_to_h_message
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
                + l_gamma * l_gate * h_to_l_message
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
                + h_gamma * (h_mix - hsi_nodes)
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
                + l_gamma * (l_mix - lidar_nodes)
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
                + h_gamma * (h_fused - hsi_nodes)
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
                + l_gamma * (l_fused - lidar_nodes)
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
                + h_gamma * (h_fused - hsi_nodes)
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
                + l_gamma * (l_fused - lidar_nodes)
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
            "transport_operator": self.transport_operator,
            "transport_mode": "bidirectional",
            "transport_stage": getattr(
                self,
                "active_transport_stage",
                "post-gat2",
            ),
            "transport_round": round_index,
            "transport_fusion": self.transport_fusion,
            "transport_message": self.transport_message,
            "transport_prior_weight": float(self.transport_prior_weight),
            "transport_lambda": float(self.transport_lambda),
            "l_to_h_beta_density": float(
                (beta_l_to_h > 0.0).float().detach().mean().item()
            ),
            "h_to_l_beta_density": float(
                (beta_h_to_l > 0.0).float().detach().mean().item()
            ),
            "l_to_h_attention_entropy": l_to_h_attention_entropy,
            "h_to_l_attention_entropy": h_to_l_attention_entropy,
            "h_transport_gamma": float(
                h_gamma.detach().item()
            ),
            "l_transport_gamma": float(
                l_gamma.detach().item()
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


class SparseOverlapCrossModalRelation(nn.Module):
    """Sparse HSI/LiDAR overlap relation without explicit C nodes.

    The relation graph is the nonzero support of M = Q_H^T Q_L. Messages
    are directional because H<-L normalizes by HSI superpixel coverage and
    L<-H normalizes by LiDAR superpixel coverage.
    """

    def __init__(
        self,
        channels,
        relation_data,
        attention_d_k=32,
        message="qk-prior",
        fusion="dual-channel",
        prior_weight=1.0,
        gamma_init=0.1,
        second_gamma_init=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.message = message
        self.fusion = fusion
        self.prior_weight = prior_weight
        self.scale = attention_d_k ** -0.5
        self.register_buffer(
            "h_index",
            torch.as_tensor(relation_data["h_index"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "l_index",
            torch.as_tensor(relation_data["l_index"], dtype=torch.long),
            persistent=False,
        )
        self.register_buffer(
            "h_coverage",
            torch.as_tensor(
                relation_data["h_coverage"],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "l_coverage",
            torch.as_tensor(
                relation_data["l_coverage"],
                dtype=torch.float32,
            ),
            persistent=False,
        )
        self.register_buffer(
            "overlap",
            torch.as_tensor(relation_data["overlap"], dtype=torch.float32),
            persistent=False,
        )
        self.h_node_count = int(relation_data["h_node_count"])
        self.l_node_count = int(relation_data["l_node_count"])
        self.edge_count = int(relation_data["edge_count"])
        self.density = float(relation_data["density"])

        self.h_query = nn.Linear(channels, attention_d_k, bias=False)
        self.l_key = nn.Linear(channels, attention_d_k, bias=False)
        self.l_query = nn.Linear(channels, attention_d_k, bias=False)
        self.h_key = nn.Linear(channels, attention_d_k, bias=False)
        self.l_to_h_value = nn.Linear(channels, channels, bias=False)
        self.h_to_l_value = nn.Linear(channels, channels, bias=False)

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
            nn.Linear(4 * channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.l_bilinear_fuse = nn.Sequential(
            nn.Linear(4 * channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.h_dual_fuse = nn.Sequential(
            nn.Linear(5 * channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.l_dual_fuse = nn.Sequential(
            nn.Linear(5 * channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )

        def make_gate(input_channels):
            output = nn.Linear(channels, 1)
            nn.init.zeros_(output.weight)
            nn.init.constant_(output.bias, -3.0)
            return nn.Sequential(
                nn.Linear(input_channels, channels),
                nn.LeakyReLU(),
                output,
            )

        self.h_bilinear_gate = make_gate(4 * channels)
        self.l_bilinear_gate = make_gate(4 * channels)
        self.h_dual_gate = make_gate(5 * channels)
        self.l_dual_gate = make_gate(5 * channels)
        self.h_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.l_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.h_gamma2 = nn.Parameter(torch.tensor(float(second_gamma_init)))
        self.l_gamma2 = nn.Parameter(torch.tensor(float(second_gamma_init)))
        self.last_diagnostics = None

    @staticmethod
    def _segment_softmax(logits, index, segment_count):
        if logits.numel() == 0:
            return logits
        if hasattr(logits, "scatter_reduce_"):
            try:
                max_values = torch.full(
                    (segment_count,),
                    torch.finfo(logits.dtype).min,
                    dtype=logits.dtype,
                    device=logits.device,
                )
                max_values.scatter_reduce_(
                    0,
                    index,
                    logits,
                    reduce="amax",
                    include_self=True,
                )
                exp_values = torch.exp(logits - max_values[index])
                denom = torch.zeros(
                    segment_count,
                    dtype=logits.dtype,
                    device=logits.device,
                )
                denom.index_add_(0, index, exp_values)
                return exp_values / denom[index].clamp_min(1e-12)
            except TypeError:
                pass

        weights = torch.zeros_like(logits)
        for segment in torch.unique(index):
            mask = index == segment
            weights[mask] = F.softmax(logits[mask], dim=0)
        return weights

    @staticmethod
    def _aggregate(edge_values, edge_weights, receiver_index, receiver_count):
        output = edge_values.new_zeros((receiver_count, edge_values.shape[1]))
        output.index_add_(
            0,
            receiver_index,
            edge_values * edge_weights.unsqueeze(1),
        )
        return output

    @staticmethod
    def _edge_entropy(edge_weights, receiver_index, receiver_count):
        entropy_edges = -edge_weights * torch.log(edge_weights.clamp_min(1e-12))
        entropy = edge_weights.new_zeros(receiver_count)
        entropy.index_add_(0, receiver_index, entropy_edges)
        return entropy

    def _edge_weights(self, hsi_nodes, lidar_nodes):
        if self.message == "fixed":
            return self.h_coverage, self.l_coverage
        if self.message != "qk-prior":
            raise ValueError(f"Unsupported sparse overlap message: {self.message}")

        h_logits = (
            self.h_query(hsi_nodes)[self.h_index]
            * self.l_key(lidar_nodes)[self.l_index]
        ).sum(dim=1) * self.scale
        h_logits = h_logits + self.prior_weight * torch.log(
            self.h_coverage.clamp_min(1e-6)
        )
        h_weights = self._segment_softmax(
            h_logits,
            self.h_index,
            self.h_node_count,
        )

        l_logits = (
            self.l_query(lidar_nodes)[self.l_index]
            * self.h_key(hsi_nodes)[self.h_index]
        ).sum(dim=1) * self.scale
        l_logits = l_logits + self.prior_weight * torch.log(
            self.l_coverage.clamp_min(1e-6)
        )
        l_weights = self._segment_softmax(
            l_logits,
            self.l_index,
            self.l_node_count,
        )
        return h_weights, l_weights

    def _update_side(
        self,
        nodes,
        message,
        consensus,
        conflict,
        gamma,
        bilinear_left,
        bilinear_right,
        bilinear_out,
        bilinear_fuse,
        dual_fuse,
        bilinear_gate,
        dual_gate,
    ):
        product = bilinear_out(
            bilinear_left(nodes) * bilinear_right(message)
        )
        if self.fusion == "bilinear":
            fuse_input = torch.cat(
                [
                    nodes,
                    message,
                    torch.abs(nodes - message),
                    product,
                ],
                dim=1,
            )
            delta = bilinear_fuse(fuse_input)
            gate = torch.sigmoid(bilinear_gate(fuse_input))
        elif self.fusion == "dual-channel":
            fuse_input = torch.cat(
                [
                    nodes,
                    message,
                    consensus,
                    conflict,
                    product,
                ],
                dim=1,
            )
            delta = dual_fuse(fuse_input)
            gate = torch.sigmoid(dual_gate(fuse_input))
        else:
            raise ValueError(f"Unsupported sparse overlap fusion: {self.fusion}")
        updated = nodes + gamma * gate * delta
        return updated, gate, product, delta

    def forward(self, hsi_nodes, lidar_nodes, round_index=1):
        if round_index == 1:
            h_gamma = self.h_gamma
            l_gamma = self.l_gamma
        elif round_index == 2:
            h_gamma = self.h_gamma2
            l_gamma = self.l_gamma2
        else:
            raise ValueError("round_index must be 1 or 2.")

        h_weights, l_weights = self._edge_weights(hsi_nodes, lidar_nodes)
        h_source = self.l_to_h_value(lidar_nodes)
        l_source = self.h_to_l_value(hsi_nodes)

        h_edge_message = h_source[self.l_index]
        l_edge_message = l_source[self.h_index]
        h_receiver = hsi_nodes[self.h_index]
        l_receiver = lidar_nodes[self.l_index]

        h_message = self._aggregate(
            h_edge_message,
            h_weights,
            self.h_index,
            self.h_node_count,
        )
        l_message = self._aggregate(
            l_edge_message,
            l_weights,
            self.l_index,
            self.l_node_count,
        )
        h_consensus = self._aggregate(
            h_receiver * h_edge_message,
            h_weights,
            self.h_index,
            self.h_node_count,
        )
        l_consensus = self._aggregate(
            l_receiver * l_edge_message,
            l_weights,
            self.l_index,
            self.l_node_count,
        )
        h_conflict = self._aggregate(
            torch.abs(h_receiver - h_edge_message),
            h_weights,
            self.h_index,
            self.h_node_count,
        )
        l_conflict = self._aggregate(
            torch.abs(l_receiver - l_edge_message),
            l_weights,
            self.l_index,
            self.l_node_count,
        )

        updated_hsi, h_gate, h_product, h_delta = self._update_side(
            hsi_nodes,
            h_message,
            h_consensus,
            h_conflict,
            h_gamma,
            self.h_bilinear_left,
            self.h_bilinear_right,
            self.h_bilinear_out,
            self.h_bilinear_fuse,
            self.h_dual_fuse,
            self.h_bilinear_gate,
            self.h_dual_gate,
        )
        updated_lidar, l_gate, l_product, l_delta = self._update_side(
            lidar_nodes,
            l_message,
            l_consensus,
            l_conflict,
            l_gamma,
            self.l_bilinear_left,
            self.l_bilinear_right,
            self.l_bilinear_out,
            self.l_bilinear_fuse,
            self.l_dual_fuse,
            self.l_bilinear_gate,
            self.l_dual_gate,
        )

        h_entropy = self._edge_entropy(
            h_weights,
            self.h_index,
            self.h_node_count,
        )
        l_entropy = self._edge_entropy(
            l_weights,
            self.l_index,
            self.l_node_count,
        )
        self.last_diagnostics = {
            "transport_operator": "sparse-overlap",
            "transport_mode": "sparse-overlap",
            "transport_round": int(round_index),
            "transport_message": self.message,
            "transport_fusion": self.fusion,
            "edge_count": int(self.edge_count),
            "edge_density": float(self.density),
            "h_transport_gamma": float(h_gamma.detach().item()),
            "l_transport_gamma": float(l_gamma.detach().item()),
            "h_attention_entropy": float(
                h_entropy.detach().mean().item()
            ),
            "l_attention_entropy": float(
                l_entropy.detach().mean().item()
            ),
            "h_gate_mean": float(h_gate.detach().mean().item()),
            "l_gate_mean": float(l_gate.detach().mean().item()),
            "h_message_norm": float(
                h_message.detach().norm(dim=1).mean().item()
            ),
            "l_message_norm": float(
                l_message.detach().norm(dim=1).mean().item()
            ),
            "h_delta_norm": float(h_delta.detach().norm(dim=1).mean().item()),
            "l_delta_norm": float(l_delta.detach().norm(dim=1).mean().item()),
            "h_product_norm": float(
                h_product.detach().norm(dim=1).mean().item()
            ),
            "l_product_norm": float(
                l_product.detach().norm(dim=1).mean().item()
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
        topology_rewiring=False,
        topology_advocacy_k=0,
        topology_veto_init=0.0,
        topology_advocacy_init=0.0,
        topology_impurity_init=0.0,
        topology_impurity_mode="target",
        topology_freeze_advocacy=False,
        topology_advocacy_candidate_mask=None,
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
        if topology_rewiring:
            self.graph_builder = CellEvidenceRewiringDynamicGraphBuilder(
                hidden_dim,
                assignment.shape[1],
                candidate_mask=candidate_mask,
                advocacy_k=topology_advocacy_k,
                veto_init=topology_veto_init,
                advocacy_init=topology_advocacy_init,
                impurity_init=topology_impurity_init,
                impurity_mode=topology_impurity_mode,
                freeze_advocacy=topology_freeze_advocacy,
                advocacy_candidate_mask=topology_advocacy_candidate_mask,
                **graph_builder_options,
            )
        elif candidate_mask is None:
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

    def apply_gat1(
        self,
        node_features,
        topology_context=None,
        topology_fragmentation=None,
        topology_uncertainty=None,
    ):
        if topology_context is None:
            adjacency = self.graph_builder(
                node_features,
                self.spatial_prior,
            )
        else:
            if not getattr(
                self.graph_builder,
                "uses_topology_rewiring",
                False,
            ):
                raise ValueError(
                    "Topology context requires a topology-rewiring "
                    "graph builder."
                )
            adjacency = self.graph_builder(
                node_features,
                self.spatial_prior,
                topology_context,
                fragmentation=topology_fragmentation,
                audit_uncertainty=topology_uncertainty,
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
        post_gat_consensus_graph="none",
        consensus_graph_weight=0.1,
        consensus_graph_fusion="residual-c",
        consensus_graph_residual_init=0.0,
        consensus_graph_transport="none",
        consensus_graph_transport_stage="post-gat2",
        consensus_graph_transport_fusion="residual",
        consensus_graph_transport_message="fixed",
        consensus_graph_transport_operator="transport",
        consensus_graph_transport_prior_weight=1.0,
        consensus_graph_transport_lambda=0.5,
        consensus_graph_transport_gamma_init=0.0,
        consensus_graph_transport_second_gamma_init=0.0,
        sheaf_restriction="diag",
        sheaf_rank=4,
        sheaf_steps=1,
        sheaf_energy_weight=0.0,
        dcsi_edge_topk=8,
        dcsi_consensus_mix=0.1,
        dcsi_conflict_mix=0.1,
        dcsi_semantic_gamma_init=0.1,
        dcsi_conflict_gamma_init=0.05,
        cgsa_alpha_max=2.0,
        cgsa_consensus_channel=False,
        consensus_graph_spatial_prior_weight=1.0,
        consensus_graph_hsi_prior_weight=0.5,
        consensus_graph_lidar_prior_weight=0.5,
        bridge_data=None,
        cross_overlap_relation="none",
        cross_overlap_stage="inter-gat",
        cross_overlap_message="qk-prior",
        cross_overlap_fusion="dual-channel",
        cross_overlap_prior_weight=1.0,
        cross_overlap_gamma_init=0.1,
        cross_overlap_second_gamma_init=0.0,
        cross_overlap_data=None,
        bridge_attention_d_k=32,
        bridge_attention_topk=8,
        bridge_gamma_init=0.0,
        cell_data=None,
        fdsm_scope="none",
        lidar_modulation="none",
        cnn_branch="original",
        cnn_layout="joint",
        cnn_share_weights=False,
        topology_rewiring="none",
        topology_advocacy_k=0,
        topology_audit_side="both",
        topology_evidence="learned",
        topology_impurity_mode="target",
        topology_freeze_advocacy=False,
        topology_veto_init=0.0,
        topology_advocacy_init=0.0,
        topology_impurity_init=0.0,
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
        self.consensus_graph_transport_stage = (
            consensus_graph_transport_stage
        )
        self.cross_overlap_relation = cross_overlap_relation
        self.cross_overlap_stage = cross_overlap_stage
        self.sheaf_energy_weight = sheaf_energy_weight
        self.last_consensus_graph_gate_diagnostics = None
        self.last_cross_overlap_diagnostics = None
        self.cnn_branch_mode = cnn_branch
        self.cnn_layout = cnn_layout
        self.cnn_share_weights = cnn_share_weights
        self.topology_rewiring = topology_rewiring
        self.topology_audit_side = topology_audit_side
        self.topology_evidence = topology_evidence
        self.last_topology_rewiring_diagnostics = None
        topology_rewiring_enabled = topology_rewiring != "none"
        hsi_topology_rewiring = topology_rewiring_enabled and (
            topology_audit_side in ("both", "hsi")
        )
        lidar_topology_rewiring = topology_rewiring_enabled and (
            topology_audit_side in ("both", "lidar")
        )
        lidar_advocacy_candidate_mask = lidar_rag_adjacency
        if (
            topology_rewiring_enabled
            and cell_data is not None
            and "topology_lidar_advocacy_adjacency" in cell_data
        ):
            lidar_advocacy_candidate_mask = cell_data[
                "topology_lidar_advocacy_adjacency"
            ]

        self.hsi_graph = ModalityGSDGGraphEncoder(
            in_channels=hsi_channels,
            assignment=hsi_assignment,
            spatial_prior=hsi_spatial_prior,
            hidden_dim=hidden_dim,
            dynamic_d_k=dynamic_d_k,
            dynamic_topk=dynamic_topk,
            dynamic_tau=dynamic_tau,
            use_fdsm=fdsm_scope == "hsi",
            topology_rewiring=hsi_topology_rewiring,
            topology_advocacy_k=topology_advocacy_k,
            topology_veto_init=topology_veto_init,
            topology_advocacy_init=topology_advocacy_init,
            topology_impurity_init=topology_impurity_init,
            topology_impurity_mode=topology_impurity_mode,
            topology_freeze_advocacy=topology_freeze_advocacy,
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
            topology_rewiring=lidar_topology_rewiring,
            topology_advocacy_k=topology_advocacy_k,
            topology_veto_init=topology_veto_init,
            topology_advocacy_init=topology_advocacy_init,
            topology_impurity_init=topology_impurity_init,
            topology_impurity_mode=topology_impurity_mode,
            topology_freeze_advocacy=topology_freeze_advocacy,
            topology_advocacy_candidate_mask=lidar_advocacy_candidate_mask,
        )

        if topology_rewiring_enabled:
            if cell_data is None:
                raise ValueError(
                    "cell-veto-advocacy topology rewiring requires "
                    "common-refinement cell data."
                )
            self.register_buffer(
                "rewiring_hsi_parent",
                torch.as_tensor(
                    cell_data["hsi_parent"],
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "rewiring_lidar_parent",
                torch.as_tensor(
                    cell_data["lidar_parent"],
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "rewiring_hsi_coverage",
                torch.as_tensor(
                    cell_data["hsi_coverage"],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "rewiring_lidar_coverage",
                torch.as_tensor(
                    cell_data["lidar_coverage"],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "rewiring_hsi_fragmentation",
                torch.as_tensor(
                    normalized_cell_fragmentation(
                        cell_data,
                        hsi_assignment.shape[1],
                        "hsi_parent",
                        "hsi_coverage",
                    ),
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            self.register_buffer(
                "rewiring_lidar_fragmentation",
                torch.as_tensor(
                    normalized_cell_fragmentation(
                        cell_data,
                        lidar_assignment.shape[1],
                        "lidar_parent",
                        "lidar_coverage",
                    ),
                    dtype=torch.float32,
                ),
                persistent=False,
            )
            if topology_evidence == "physical":
                physical_evidence = cell_data.get(
                    "topology_physical_evidence"
                )
                if physical_evidence is None:
                    raise ValueError(
                        "--topology-evidence physical requires "
                        "precomputed physical evidence in cell_data."
                    )
                self.register_buffer(
                    "rewiring_hsi_physical_context",
                    torch.as_tensor(
                        physical_evidence["hsi_context"],
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "rewiring_hsi_physical_uncertainty",
                    torch.as_tensor(
                        physical_evidence["hsi_uncertainty"],
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "rewiring_lidar_physical_context",
                    torch.as_tensor(
                        physical_evidence["lidar_context"],
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
                self.register_buffer(
                    "rewiring_lidar_physical_uncertainty",
                    torch.as_tensor(
                        physical_evidence["lidar_uncertainty"],
                        dtype=torch.float32,
                    ),
                    persistent=False,
                )
            else:
                self.rewiring_hsi_physical_context = None
                self.rewiring_hsi_physical_uncertainty = None
                self.rewiring_lidar_physical_context = None
                self.rewiring_lidar_physical_uncertainty = None
        else:
            self.rewiring_hsi_parent = None
            self.rewiring_lidar_parent = None
            self.rewiring_hsi_coverage = None
            self.rewiring_lidar_coverage = None
            self.rewiring_hsi_fragmentation = None
            self.rewiring_lidar_fragmentation = None
            self.rewiring_hsi_physical_context = None
            self.rewiring_hsi_physical_uncertainty = None
            self.rewiring_lidar_physical_context = None
            self.rewiring_lidar_physical_uncertainty = None

        def make_cnn_branch():
            if cnn_branch == "original":
                return nn.Sequential(
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
            if cnn_branch == "gsdg":
                return nn.Sequential(
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
            raise ValueError(
                "cnn_branch must be 'original' or 'gsdg'."
            )

        if cnn_layout == "joint":
            self.joint_feature_mapping = nn.Sequential(
                WMF(hsi_channels + 1, hidden_dim),
                WMF(hidden_dim, hidden_dim),
            )
            self.cnn_branch = make_cnn_branch()
            self.hsi_cnn_feature_mapping = None
            self.lidar_cnn_feature_mapping = None
            self.hsi_cnn_branch = None
            self.lidar_cnn_branch = None
            self.shared_cnn_branch = None
        elif cnn_layout == "separate":
            self.joint_feature_mapping = None
            self.cnn_branch = None
            self.hsi_cnn_feature_mapping = nn.Sequential(
                WMF(hsi_channels, hidden_dim),
                WMF(hidden_dim, hidden_dim),
            )
            self.lidar_cnn_feature_mapping = nn.Sequential(
                WMF(1, hidden_dim),
                WMF(hidden_dim, hidden_dim),
            )
            if cnn_share_weights:
                self.shared_cnn_branch = make_cnn_branch()
                self.hsi_cnn_branch = None
                self.lidar_cnn_branch = None
            else:
                self.shared_cnn_branch = None
                self.hsi_cnn_branch = make_cnn_branch()
                self.lidar_cnn_branch = make_cnn_branch()
        else:
            raise ValueError("cnn_layout must be 'joint' or 'separate'.")
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
                    transport_operator=(
                        consensus_graph_transport_operator
                    ),
                    transport_prior_weight=(
                        consensus_graph_transport_prior_weight
                    ),
                    transport_gamma_init=(
                        consensus_graph_transport_gamma_init
                    ),
                    transport_second_gamma_init=(
                        consensus_graph_transport_second_gamma_init
                    ),
                    sheaf_restriction=sheaf_restriction,
                    sheaf_rank=sheaf_rank,
                    sheaf_steps=sheaf_steps,
                    dcsi_edge_topk=dcsi_edge_topk,
                    dcsi_consensus_mix=dcsi_consensus_mix,
                    dcsi_conflict_mix=dcsi_conflict_mix,
                    dcsi_semantic_gamma_init=(
                        dcsi_semantic_gamma_init
                    ),
                    dcsi_conflict_gamma_init=(
                        dcsi_conflict_gamma_init
                    ),
                    cgsa_alpha_max=cgsa_alpha_max,
                    cgsa_consensus_channel=cgsa_consensus_channel,
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

        if cross_overlap_relation == "sparse":
            if cross_overlap_data is None:
                cross_overlap_data = build_sparse_overlap_relation_data(
                    hsi_assignment,
                    lidar_assignment,
                )
            self.cross_overlap_branch = SparseOverlapCrossModalRelation(
                hidden_dim,
                cross_overlap_data,
                attention_d_k=bridge_attention_d_k,
                message=cross_overlap_message,
                fusion=cross_overlap_fusion,
                prior_weight=cross_overlap_prior_weight,
                gamma_init=cross_overlap_gamma_init,
                second_gamma_init=cross_overlap_second_gamma_init,
            )
        else:
            self.cross_overlap_branch = None

    def _has_topology_rewiring(self):
        return self.topology_rewiring != "none"

    def _cross_topology_contexts(self, hsi_nodes, lidar_nodes):
        if self.topology_evidence == "physical":
            return (
                self.rewiring_hsi_physical_context,
                self.rewiring_lidar_physical_context,
                self.rewiring_hsi_fragmentation,
                self.rewiring_lidar_fragmentation,
                self.rewiring_hsi_physical_uncertainty,
                self.rewiring_lidar_physical_uncertainty,
            )
        hsi_context = hsi_nodes.new_zeros(hsi_nodes.shape)
        hsi_context.index_add_(
            0,
            self.rewiring_hsi_parent,
            self.rewiring_hsi_coverage.unsqueeze(1)
            * lidar_nodes.index_select(
                0,
                self.rewiring_lidar_parent,
            ),
        )
        lidar_context = lidar_nodes.new_zeros(lidar_nodes.shape)
        lidar_context.index_add_(
            0,
            self.rewiring_lidar_parent,
            self.rewiring_lidar_coverage.unsqueeze(1)
            * hsi_nodes.index_select(
                0,
                self.rewiring_hsi_parent,
            ),
        )
        return (
            hsi_context,
            lidar_context,
            self.rewiring_hsi_fragmentation,
            self.rewiring_lidar_fragmentation,
            None,
            None,
        )

    def _apply_private_gat1(self, hsi_nodes, lidar_nodes):
        if not self._has_topology_rewiring():
            hsi_features, hsi_adjacency = self.hsi_graph.apply_gat1(
                hsi_nodes
            )
            lidar_features, lidar_adjacency = self.lidar_graph.apply_gat1(
                lidar_nodes
            )
            return (
                hsi_features,
                lidar_features,
                hsi_adjacency,
                lidar_adjacency,
            )

        (
            hsi_context,
            lidar_context,
            hsi_fragmentation,
            lidar_fragmentation,
            hsi_uncertainty,
            lidar_uncertainty,
        ) = self._cross_topology_contexts(hsi_nodes, lidar_nodes)
        if self.topology_audit_side in ("both", "hsi"):
            hsi_features, hsi_adjacency = self.hsi_graph.apply_gat1(
                hsi_nodes,
                topology_context=hsi_context,
                topology_fragmentation=hsi_fragmentation,
                topology_uncertainty=hsi_uncertainty,
            )
        else:
            hsi_features, hsi_adjacency = self.hsi_graph.apply_gat1(
                hsi_nodes
            )
        if self.topology_audit_side in ("both", "lidar"):
            lidar_features, lidar_adjacency = self.lidar_graph.apply_gat1(
                lidar_nodes,
                topology_context=lidar_context,
                topology_fragmentation=lidar_fragmentation,
                topology_uncertainty=lidar_uncertainty,
            )
        else:
            lidar_features, lidar_adjacency = self.lidar_graph.apply_gat1(
                lidar_nodes
            )
        self.last_topology_rewiring_diagnostics = {
            "mode": self.topology_rewiring,
            "evidence": self.topology_evidence,
            "side": self.topology_audit_side,
            "hsi": getattr(
                self.hsi_graph.graph_builder,
                "last_diagnostics",
                None,
            ),
            "lidar": getattr(
                self.lidar_graph.graph_builder,
                "last_diagnostics",
                None,
            ),
        }
        return (
            hsi_features,
            lidar_features,
            hsi_adjacency,
            lidar_adjacency,
        )

    def _encode_private_graph_nodes(self, hsi, lidar):
        hsi_nodes = self.hsi_graph.encode_nodes(hsi)
        lidar_nodes = self.lidar_graph.encode_nodes(lidar)
        (
            hsi_features,
            lidar_features,
            hsi_adjacency,
            lidar_adjacency,
        ) = self._apply_private_gat1(hsi_nodes, lidar_nodes)
        hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
            hsi_features,
            adjacency=hsi_adjacency,
        )
        lidar_final_nodes = self.lidar_graph.apply_gat2_nodes(
            lidar_features,
            adjacency=lidar_adjacency,
        )
        return hsi_final_nodes, lidar_final_nodes

    def _forward_mediator_transport(self, hsi, lidar):
        self.consensus_graph_branch.active_transport_stage = (
            self.consensus_graph_transport_stage
        )
        hsi_nodes = self.hsi_graph.encode_nodes(hsi)
        lidar_nodes = self.lidar_graph.encode_nodes(lidar)
        (
            hsi_features,
            lidar_features,
            hsi_adjacency,
            lidar_adjacency,
        ) = self._apply_private_gat1(
            hsi_nodes,
            lidar_nodes,
        )
        if self.consensus_graph_transport_stage == "post-gat2":
            hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                hsi_features,
                adjacency=hsi_adjacency,
            )
            lidar_final_nodes = self.lidar_graph.apply_gat2_nodes(
                lidar_features,
                adjacency=lidar_adjacency,
            )
            (
                hsi_final_nodes,
                lidar_final_nodes,
            ) = self.consensus_graph_branch.transport_nodes(
                hsi_final_nodes,
                lidar_final_nodes,
                hsi_adjacency,
                lidar_adjacency,
                round_index=1,
            )
        elif self.consensus_graph_transport_stage == "inter-gat":
            (
                hsi_features,
                lidar_features,
            ) = self.consensus_graph_branch.transport_nodes(
                hsi_features,
                lidar_features,
                hsi_adjacency,
                lidar_adjacency,
                round_index=1,
            )
            hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                hsi_features,
                adjacency=hsi_adjacency,
            )
            lidar_final_nodes = self.lidar_graph.apply_gat2_nodes(
                lidar_features,
                adjacency=lidar_adjacency,
            )
        elif self.consensus_graph_transport_stage == "alternating":
            (
                hsi_features,
                lidar_features,
            ) = self.consensus_graph_branch.transport_nodes(
                hsi_features,
                lidar_features,
                hsi_adjacency,
                lidar_adjacency,
                round_index=1,
            )
            hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                hsi_features,
                adjacency=hsi_adjacency,
            )
            lidar_final_nodes = self.lidar_graph.apply_gat2_nodes(
                lidar_features,
                adjacency=lidar_adjacency,
            )
            (
                hsi_final_nodes,
                lidar_final_nodes,
            ) = self.consensus_graph_branch.transport_nodes(
                hsi_final_nodes,
                lidar_final_nodes,
                hsi_adjacency,
                lidar_adjacency,
                round_index=2,
            )
        else:
            raise ValueError(
                "Unsupported consensus graph transport stage: "
                f"{self.consensus_graph_transport_stage}"
            )
        hsi_graph_features = self.hsi_graph.project_nodes(
            hsi_final_nodes
        )
        lidar_graph_features = self.lidar_graph.project_nodes(
            lidar_final_nodes
        )
        self.last_consensus_graph_gate_diagnostics = {
            "fusion_mode": "transport-private-fusion",
            "private_weights": [
                self.graph_modality_lambda,
                1.0 - self.graph_modality_lambda,
            ],
        }
        return (
            self.graph_modality_lambda * hsi_graph_features
            + (1.0 - self.graph_modality_lambda)
            * lidar_graph_features
        )

    def _forward_cross_overlap_relation(self, hsi, lidar):
        hsi_nodes = self.hsi_graph.encode_nodes(hsi)
        lidar_nodes = self.lidar_graph.encode_nodes(lidar)
        (
            hsi_features,
            lidar_features,
            hsi_adjacency,
            lidar_adjacency,
        ) = self._apply_private_gat1(
            hsi_nodes,
            lidar_nodes,
        )
        if self.cross_overlap_stage == "post-gat2":
            hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                hsi_features,
                adjacency=hsi_adjacency,
            )
            lidar_final_nodes = self.lidar_graph.apply_gat2_nodes(
                lidar_features,
                adjacency=lidar_adjacency,
            )
            (
                hsi_final_nodes,
                lidar_final_nodes,
            ) = self.cross_overlap_branch(
                hsi_final_nodes,
                lidar_final_nodes,
                round_index=1,
            )
        elif self.cross_overlap_stage == "inter-gat":
            (
                hsi_features,
                lidar_features,
            ) = self.cross_overlap_branch(
                hsi_features,
                lidar_features,
                round_index=1,
            )
            hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                hsi_features,
                adjacency=hsi_adjacency,
            )
            lidar_final_nodes = self.lidar_graph.apply_gat2_nodes(
                lidar_features,
                adjacency=lidar_adjacency,
            )
        elif self.cross_overlap_stage == "alternating":
            (
                hsi_features,
                lidar_features,
            ) = self.cross_overlap_branch(
                hsi_features,
                lidar_features,
                round_index=1,
            )
            hsi_final_nodes = self.hsi_graph.apply_gat2_nodes(
                hsi_features,
                adjacency=hsi_adjacency,
            )
            lidar_final_nodes = self.lidar_graph.apply_gat2_nodes(
                lidar_features,
                adjacency=lidar_adjacency,
            )
            (
                hsi_final_nodes,
                lidar_final_nodes,
            ) = self.cross_overlap_branch(
                hsi_final_nodes,
                lidar_final_nodes,
                round_index=2,
            )
        else:
            raise ValueError(
                "Unsupported sparse overlap stage: "
                f"{self.cross_overlap_stage}"
            )
        hsi_graph_features = self.hsi_graph.project_nodes(hsi_final_nodes)
        lidar_graph_features = self.lidar_graph.project_nodes(
            lidar_final_nodes
        )
        self.last_cross_overlap_diagnostics = {
            "stage": self.cross_overlap_stage,
            **(self.cross_overlap_branch.diagnostics() or {}),
        }
        return (
            self.graph_modality_lambda * hsi_graph_features
            + (1.0 - self.graph_modality_lambda)
            * lidar_graph_features
        )

    def _flatten_cnn_features(self, cnn_output):
        return (
            cnn_output.squeeze(0)
            .permute(1, 2, 0)
            .reshape(self.height * self.width, -1)
        )

    def _forward_cnn_features(self, hsi, lidar, joint_input):
        if self.cnn_layout == "joint":
            mapped_joint = self.joint_feature_mapping(
                joint_input.permute(2, 0, 1).unsqueeze(0)
            )
            return self._flatten_cnn_features(
                self.cnn_branch(mapped_joint)
            )
        if self.cnn_layout == "separate":
            mapped_hsi = self.hsi_cnn_feature_mapping(
                hsi.permute(2, 0, 1).unsqueeze(0)
            )
            mapped_lidar = self.lidar_cnn_feature_mapping(
                lidar.permute(2, 0, 1).unsqueeze(0)
            )
            if self.cnn_share_weights:
                hsi_cnn_features = self._flatten_cnn_features(
                    self.shared_cnn_branch(mapped_hsi)
                )
                lidar_cnn_features = self._flatten_cnn_features(
                    self.shared_cnn_branch(mapped_lidar)
                )
                return (
                    self.graph_modality_lambda * hsi_cnn_features
                    + (1.0 - self.graph_modality_lambda)
                    * lidar_cnn_features
                )
            hsi_cnn_features = self._flatten_cnn_features(
                self.hsi_cnn_branch(mapped_hsi)
            )
            lidar_cnn_features = self._flatten_cnn_features(
                self.lidar_cnn_branch(mapped_lidar)
            )
            return (
                self.graph_modality_lambda * hsi_cnn_features
                + (1.0 - self.graph_modality_lambda)
                * lidar_cnn_features
            )
        raise ValueError(
            f"Unsupported CNN layout: {self.cnn_layout}"
        )

    def forward(self, hsi, lidar, joint_input):
        self.last_consensus_graph_gate_diagnostics = None
        self.last_cross_overlap_diagnostics = None
        if self.cross_overlap_branch is not None:
            graph_features = self._forward_cross_overlap_relation(hsi, lidar)
        elif self.consensus_graph_branch is None:
            if self._has_topology_rewiring():
                (
                    hsi_final_nodes,
                    lidar_final_nodes,
                ) = self._encode_private_graph_nodes(hsi, lidar)
                hsi_graph_features = self.hsi_graph.project_nodes(
                    hsi_final_nodes
                )
                lidar_graph_features = self.lidar_graph.project_nodes(
                    lidar_final_nodes
                )
            else:
                hsi_graph_features = self.hsi_graph(hsi)
                lidar_graph_features = self.lidar_graph(lidar)
            graph_features = (
                self.graph_modality_lambda * hsi_graph_features
                + (1.0 - self.graph_modality_lambda)
                * lidar_graph_features
            )
        elif self.consensus_graph_transport == "bidirectional":
            graph_features = self._forward_mediator_transport(hsi, lidar)
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
            if self.consensus_graph_transport != "none":
                raise ValueError(
                    f"Unsupported consensus graph transport: "
                    f"{self.consensus_graph_transport}"
                )
            if self.consensus_graph_transport == "none":
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

        cnn_features = self._forward_cnn_features(
            hsi,
            lidar,
            joint_input,
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
        or args.topology_rewiring != "none"
    )
    if needs_intersection_cells:
        cell_data = build_common_refinement_cells(
            hsi_assignment,
            lidar_assignment,
            hsi.shape[0],
            hsi.shape[1],
        )
        if (
            args.topology_rewiring != "none"
            and args.topology_advocacy_k > 0
        ):
            if (
                lidar_rag_adjacency is not None
                and args.topology_advocacy_rag_hops
                == args.lidar_rag_hops
            ):
                topology_lidar_advocacy_adjacency = lidar_rag_adjacency
            else:
                (
                    topology_lidar_advocacy_adjacency,
                    _,
                ) = build_lidar_rag_modulation_structure(
                    lidar_assignment_parts,
                    lidar,
                    args.topology_advocacy_rag_hops,
                )
            cell_data["topology_lidar_advocacy_adjacency"] = (
                topology_lidar_advocacy_adjacency
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
    if (
        cell_data is not None
        and args.topology_rewiring != "none"
        and args.topology_evidence == "physical"
    ):
        cell_data["topology_physical_evidence"] = (
            build_topology_physical_evidence(
                cell_data,
                reduced_hsi,
                lidar,
            )
        )
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
    classifier_output_dim = (
        class_count
        if args.dummy_logit_dim <= 0
        else args.dummy_logit_dim
    )
    if classifier_output_dim < class_count:
        raise ValueError(
            "--dummy-logit-dim must be zero/disabled or at least the "
            f"dataset class count ({class_count})."
        )
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
        "class_count": classifier_output_dim,
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
            consensus_graph_transport_stage=(
                args.consensus_graph_transport_stage
            ),
            consensus_graph_transport_fusion=(
                args.consensus_graph_transport_fusion
            ),
            consensus_graph_transport_message=(
                args.consensus_graph_transport_message
            ),
            consensus_graph_transport_operator=(
                args.consensus_graph_transport_operator
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
            consensus_graph_transport_second_gamma_init=(
                args.consensus_graph_transport_second_gamma_init
            ),
            sheaf_restriction=args.sheaf_restriction,
            sheaf_rank=args.sheaf_rank,
            sheaf_steps=args.sheaf_steps,
            sheaf_energy_weight=args.sheaf_energy_weight,
            dcsi_edge_topk=args.dcsi_edge_topk,
            dcsi_consensus_mix=args.dcsi_consensus_mix,
            dcsi_conflict_mix=args.dcsi_conflict_mix,
            dcsi_semantic_gamma_init=args.dcsi_semantic_gamma_init,
            dcsi_conflict_gamma_init=args.dcsi_conflict_gamma_init,
            cgsa_alpha_max=args.cgsa_alpha_max,
            cgsa_consensus_channel=args.cgsa_consensus_channel,
            consensus_graph_spatial_prior_weight=(
                args.consensus_graph_spatial_prior_weight
            ),
            consensus_graph_hsi_prior_weight=(
                args.consensus_graph_hsi_prior_weight
            ),
            consensus_graph_lidar_prior_weight=(
                args.consensus_graph_lidar_prior_weight
            ),
            cross_overlap_relation=args.cross_overlap_relation,
            cross_overlap_stage=args.cross_overlap_stage,
            cross_overlap_message=args.cross_overlap_message,
            cross_overlap_fusion=args.cross_overlap_fusion,
            cross_overlap_prior_weight=args.cross_overlap_prior_weight,
            cross_overlap_gamma_init=args.cross_overlap_gamma_init,
            cross_overlap_second_gamma_init=(
                args.cross_overlap_second_gamma_init
            ),
            bridge_attention_d_k=args.bridge_attention_dk,
            bridge_attention_topk=args.bridge_attention_topk,
            bridge_gamma_init=args.consensus_graph_c_gamma_init,
            bridge_data=bridge_data,
            cell_data=cell_data,
            fdsm_scope=args.fdsm_scope,
            cnn_branch=args.cnn_branch,
            cnn_layout=args.cnn_layout,
            cnn_share_weights=args.cnn_share_weights,
            topology_rewiring=args.topology_rewiring,
            topology_advocacy_k=args.topology_advocacy_k,
            topology_audit_side=args.topology_audit_side,
            topology_evidence=args.topology_evidence,
            topology_impurity_mode=args.topology_impurity_mode,
            topology_freeze_advocacy=args.topology_freeze_advocacy,
            topology_veto_init=args.topology_veto_init,
            topology_advocacy_init=args.topology_advocacy_init,
            topology_impurity_init=args.topology_impurity_init,
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
    topology_rewiring_diagnostics = []
    cross_overlap_diagnostics = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_model()
        real_class_logits = logits[:, :class_count]
        classification_loss = criterion(
            logits.index_select(0, train_index),
            train_labels,
        )
        loss = classification_loss
        consensus_graph_module = getattr(
            model,
            "consensus_graph_branch",
            None,
        )
        sheaf_energy_loss = None
        if consensus_graph_module is not None:
            sheaf_energy = getattr(
                consensus_graph_module,
                "last_sheaf_energy_loss",
                None,
            )
            if sheaf_energy is not None and args.sheaf_energy_weight > 0:
                sheaf_energy_loss = args.sheaf_energy_weight * sheaf_energy
                loss = loss + sheaf_energy_loss
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
            topology_rewiring_record = getattr(
                model,
                "last_topology_rewiring_diagnostics",
                None,
            )
            if topology_rewiring_record is not None:
                topology_rewiring_record = {
                    "epoch": epoch,
                    **topology_rewiring_record,
                }
                topology_rewiring_diagnostics.append(
                    topology_rewiring_record
                )
            cross_overlap_record = getattr(
                model,
                "last_cross_overlap_diagnostics",
                None,
            )
            if cross_overlap_record is not None:
                cross_overlap_record = {
                    "epoch": epoch,
                    **cross_overlap_record,
                }
                cross_overlap_diagnostics.append(cross_overlap_record)
            train_predictions = (
                real_class_logits.index_select(
                    0,
                    train_index,
                ).argmax(dim=1)
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
            if topology_rewiring_record is not None:

                def topology_side_text(record):
                    if record is None:
                        return "off"
                    return (
                        f"a={record['alpha']:.3f},"
                        f"b={record['beta']:.3f},"
                        f"i={record['impurity']:.3f},"
                        f"T={record['source_temperature_mean']:.3f},"
                        f"sup={record['support_density']:.3f},"
                        f"adv={record['advocacy_density']:.3f}"
                    )

                print(
                    "  Topology audit: "
                    f"side={topology_rewiring_record['side']}, "
                    f"evidence={topology_rewiring_record['evidence']}, "
                    f"H[{topology_side_text(topology_rewiring_record['hsi'])}] "
                    f"L[{topology_side_text(topology_rewiring_record['lidar'])}]"
                )
            if cross_overlap_record is not None:
                print(
                    "  Sparse overlap: "
                    f"stage={cross_overlap_record['stage']}, "
                    f"round={cross_overlap_record['transport_round']}, "
                    f"msg={cross_overlap_record['transport_message']}, "
                    f"fusion={cross_overlap_record['transport_fusion']}, "
                    f"edges={cross_overlap_record['edge_count']}, "
                    f"gamma="
                    f"{cross_overlap_record['h_transport_gamma']:.3f}/"
                    f"{cross_overlap_record['l_transport_gamma']:.3f}, "
                    f"gate="
                    f"{cross_overlap_record['h_gate_mean']:.3f}/"
                    f"{cross_overlap_record['l_gate_mean']:.3f}"
                )
            if consensus_graph_record is not None:
                if (
                    consensus_graph_record.get("transport_operator")
                    == "cgsa"
                ):
                    print(
                        "  C-CGSA: "
                        f"stage={consensus_graph_record.get('transport_stage', 'post-gat2')}, "
                        f"round={consensus_graph_record.get('transport_round', 1)}, "
                        f"restriction={consensus_graph_record['cgsa_restriction']}, "
                        f"rank={consensus_graph_record['cgsa_rank']}, "
                        f"steps={consensus_graph_record['cgsa_steps']}, "
                        f"edge_topk={consensus_graph_record['cgsa_edge_topk']}, "
                        "alpha="
                        f"{consensus_graph_record['cgsa_alpha_mean']:.3f}"
                        "±"
                        f"{consensus_graph_record['cgsa_alpha_std']:.3f}, "
                        "gamma="
                        f"{consensus_graph_record['cgsa_gamma_h']:.3f}/"
                        f"{consensus_graph_record['cgsa_gamma_l']:.3f}, "
                        "entropy="
                        f"{consensus_graph_record['cgsa_attention_entropy']:.3f}, "
                        "conflict="
                        f"{consensus_graph_record['cgsa_conflict_norm']:.3f}, "
                        "energy="
                        f"{consensus_graph_record['sheaf_energy_mean']:.4f}"
                    )
                elif (
                    consensus_graph_record.get("transport_operator")
                    == "dcsi"
                ):
                    print(
                        "  C-DCSI: "
                        f"stage={consensus_graph_record.get('transport_stage', 'post-gat2')}, "
                        f"round={consensus_graph_record.get('transport_round', 1)}, "
                        f"restriction={consensus_graph_record['dcsi_restriction']}, "
                        f"rank={consensus_graph_record['dcsi_rank']}, "
                        f"blocks={consensus_graph_record['dcsi_steps']}, "
                        f"edge_topk={consensus_graph_record['dcsi_edge_topk']}, "
                        "mu="
                        f"{consensus_graph_record['dcsi_consensus_mix']:.3f}/"
                        f"{consensus_graph_record['dcsi_conflict_mix']:.3f}, "
                        "gamma="
                        f"{consensus_graph_record['dcsi_semantic_gamma']:.3f}/"
                        f"{consensus_graph_record['dcsi_conflict_gamma']:.3f}, "
                        "gate="
                        f"{consensus_graph_record['dcsi_h_gate_mean']:.3f}/"
                        f"{consensus_graph_record['dcsi_l_gate_mean']:.3f}, "
                        "entropy="
                        f"{consensus_graph_record['dcsi_attention_entropy']:.3f}, "
                        "energy="
                        f"{consensus_graph_record['sheaf_energy_mean']:.4f}"
                    )
                elif (
                    consensus_graph_record.get("transport_operator")
                    == "sheaf"
                ):
                    print(
                        "  C-sheaf: "
                        f"stage={consensus_graph_record.get('transport_stage', 'post-gat2')}, "
                        f"round={consensus_graph_record.get('transport_round', 1)}, "
                        f"restriction={consensus_graph_record['sheaf_restriction']}, "
                        f"rank={consensus_graph_record['sheaf_rank']}, "
                        f"steps={consensus_graph_record['sheaf_steps']}, "
                        "alpha="
                        f"{consensus_graph_record['sheaf_alpha_h']:.4f}/"
                        f"{consensus_graph_record['sheaf_alpha_l']:.4f}, "
                        "energy="
                        f"{consensus_graph_record['sheaf_energy_mean']:.4f}"
                    )
                else:
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
                        if transport_fusion == "tri-gate":
                            h_tri = np.asarray(
                                consensus_graph_record[
                                    "h_tri_gate_mean"
                                ]
                            )
                            l_tri = np.asarray(
                                consensus_graph_record[
                                    "l_tri_gate_mean"
                                ]
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
                            ", stage="
                            f"{consensus_graph_record.get('transport_stage', 'post-gat2')}"
                            ", round="
                            f"{consensus_graph_record.get('transport_round', 1)}"
                            ", lambda="
                            f"{consensus_graph_record['transport_lambda']:.2f}"
                            ", mode="
                            f"{transport_fusion}"
                            ", msg="
                            f"{transport_message}"
                            ", h_gamma="
                            f"{consensus_graph_record['h_transport_gamma']:.4f}"
                            ", l_gamma="
                            f"{consensus_graph_record['l_transport_gamma']:.4f}"
                            ", "
                            f"{gate_text}"
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
            [:, :class_count]
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
        f"{topology_rewiring_configuration_tag(args)}_"
        f"{cross_overlap_configuration_tag(args)}_"
        f"fdsm-{args.fdsm_scope}_"
        f"{cnn_configuration_tag(args)}_"
        f"lidarmod-{args.lidar_modulation}_"
        f"{dummy_logit_configuration_tag(args, class_count)}_"
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
        "classifier_output_dim": classifier_output_dim,
        "real_class_count": class_count,
        "dummy_logit_count": classifier_output_dim - class_count,
        "consensus_graph_diagnostics": (
            consensus_graph_diagnostics
        ),
        "topology_rewiring_diagnostics": (
            topology_rewiring_diagnostics
        ),
        "cross_overlap_diagnostics": cross_overlap_diagnostics,
    }


def normalize_joint_layout_args(args):
    """Treat separate-only switches as no-ops for the joint graph layout."""
    if args.graph_layout != "joint":
        return

    ignored = []

    def reset_if_needed(attribute, default, flag):
        if getattr(args, attribute) != default:
            ignored.append(flag)
            setattr(args, attribute, default)

    reset_if_needed(
        "post_gat_consensus_graph",
        "none",
        "--post-gat-consensus-graph",
    )
    reset_if_needed(
        "consensus_graph_transport",
        "none",
        "--consensus-graph-transport",
    )
    reset_if_needed(
        "consensus_graph_transport_stage",
        "post-gat2",
        "--consensus-graph-transport-stage",
    )
    reset_if_needed(
        "consensus_graph_transport_operator",
        "transport",
        "--consensus-graph-transport-operator",
    )
    reset_if_needed(
        "consensus_graph_transport_fusion",
        "residual",
        "--consensus-graph-transport-fusion",
    )
    reset_if_needed(
        "consensus_graph_transport_message",
        "fixed",
        "--consensus-graph-transport-message",
    )
    reset_if_needed(
        "consensus_graph_transport_second_gamma_init",
        0.0,
        "--consensus-graph-transport-second-gamma-init",
    )
    reset_if_needed("sheaf_restriction", "diag", "--sheaf-restriction")
    reset_if_needed("sheaf_rank", 4, "--sheaf-rank")
    reset_if_needed("sheaf_steps", 1, "--sheaf-steps")
    reset_if_needed("sheaf_energy_weight", 0.0, "--sheaf-energy-weight")
    reset_if_needed("dcsi_edge_topk", 8, "--dcsi-edge-topk")
    reset_if_needed("dcsi_consensus_mix", 0.1, "--dcsi-consensus-mix")
    reset_if_needed("dcsi_conflict_mix", 0.1, "--dcsi-conflict-mix")
    reset_if_needed(
        "dcsi_semantic_gamma_init",
        0.1,
        "--dcsi-semantic-gamma-init",
    )
    reset_if_needed(
        "dcsi_conflict_gamma_init",
        0.05,
        "--dcsi-conflict-gamma-init",
    )
    reset_if_needed("cgsa_alpha_max", 2.0, "--cgsa-alpha-max")
    reset_if_needed(
        "cgsa_consensus_channel",
        False,
        "--cgsa-consensus-channel",
    )
    reset_if_needed(
        "consensus_graph_cell_edge",
        "binary",
        "--consensus-graph-cell-edge",
    )
    reset_if_needed(
        "topology_rewiring",
        "none",
        "--topology-rewiring",
    )
    reset_if_needed(
        "topology_advocacy_k",
        0,
        "--topology-advocacy-k",
    )
    reset_if_needed(
        "topology_advocacy_rag_hops",
        2,
        "--topology-advocacy-rag-hops",
    )
    reset_if_needed(
        "topology_audit_side",
        "both",
        "--topology-audit-side",
    )
    reset_if_needed(
        "topology_evidence",
        "learned",
        "--topology-evidence",
    )
    reset_if_needed(
        "topology_impurity_mode",
        "target",
        "--topology-impurity-mode",
    )
    reset_if_needed(
        "topology_freeze_advocacy",
        False,
        "--topology-freeze-advocacy",
    )
    reset_if_needed(
        "topology_veto_init",
        0.0,
        "--topology-veto-init",
    )
    reset_if_needed(
        "topology_advocacy_init",
        0.0,
        "--topology-advocacy-init",
    )
    reset_if_needed(
        "topology_impurity_init",
        0.0,
        "--topology-impurity-init",
    )
    reset_if_needed(
        "cross_overlap_relation",
        "none",
        "--cross-overlap-relation",
    )
    reset_if_needed(
        "cross_overlap_stage",
        "inter-gat",
        "--cross-overlap-stage",
    )
    reset_if_needed(
        "cross_overlap_message",
        "qk-prior",
        "--cross-overlap-message",
    )
    reset_if_needed(
        "cross_overlap_fusion",
        "dual-channel",
        "--cross-overlap-fusion",
    )
    reset_if_needed(
        "cross_overlap_prior_weight",
        1.0,
        "--cross-overlap-prior-weight",
    )
    reset_if_needed(
        "cross_overlap_gamma_init",
        0.1,
        "--cross-overlap-gamma-init",
    )
    reset_if_needed(
        "cross_overlap_second_gamma_init",
        0.0,
        "--cross-overlap-second-gamma-init",
    )
    reset_if_needed(
        "lidar_graph_prior",
        "centroid",
        "--lidar-graph-prior",
    )
    reset_if_needed(
        "lidar_modulation",
        "none",
        "--lidar-modulation",
    )
    reset_if_needed("fdsm_scope", "none", "--fdsm-scope")
    reset_if_needed("cnn_branch", "original", "--cnn-branch")
    reset_if_needed("cnn_layout", "joint", "--cnn-layout")
    reset_if_needed("cnn_share_weights", False, "--cnn-share-weights")

    if ignored:
        print(
            "Joint graph layout: ignoring separate-only options: "
            + ", ".join(ignored)
        )


def validate_args(args):
    normalize_joint_layout_args(args)
    if args.cross_overlap_stage in ("post-gat", "postgat"):
        args.cross_overlap_stage = "post-gat2"
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
    if args.topology_advocacy_k < 0:
        raise ValueError("--topology-advocacy-k must be nonnegative.")
    if args.topology_advocacy_rag_hops not in (1, 2):
        raise ValueError("--topology-advocacy-rag-hops must be 1 or 2.")
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
    if args.consensus_graph_transport_second_gamma_init < 0:
        raise ValueError(
            "--consensus-graph-transport-second-gamma-init must be "
            "nonnegative."
        )
    if args.consensus_graph_transport_prior_weight < 0:
        raise ValueError(
            "--consensus-graph-transport-prior-weight must be nonnegative."
        )
    if args.sheaf_rank <= 0:
        raise ValueError("--sheaf-rank must be positive.")
    if args.sheaf_steps <= 0:
        raise ValueError("--sheaf-steps must be positive.")
    if args.sheaf_energy_weight < 0:
        raise ValueError("--sheaf-energy-weight must be nonnegative.")
    if args.dcsi_edge_topk <= 0:
        raise ValueError("--dcsi-edge-topk must be positive.")
    if args.dcsi_consensus_mix < 0 or args.dcsi_conflict_mix < 0:
        raise ValueError("DCSI mix values must be nonnegative.")
    if (
        args.dcsi_semantic_gamma_init < 0
        or args.dcsi_conflict_gamma_init < 0
    ):
        raise ValueError("DCSI gamma init values must be nonnegative.")
    if args.cgsa_alpha_max <= 0:
        raise ValueError("--cgsa-alpha-max must be positive.")
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
    if args.topology_rewiring != "none":
        if args.graph_layout != "separate":
            raise ValueError(
                "--topology-rewiring requires --graph-layout separate."
            )
        if len(args.scales) != 1:
            raise ValueError(
                "--topology-rewiring uses common-refinement cells "
                "and requires exactly one superpixel scale."
            )
        if args.post_gat_consensus_graph != "none":
            raise ValueError(
                "--topology-rewiring treats intersection cells as a "
                "cross-modal topology auditor; do not enable "
                "--post-gat-consensus-graph at the same time."
            )
        if args.consensus_graph_transport != "none":
            raise ValueError(
                "--topology-rewiring is mutually exclusive with "
                "--consensus-graph-transport."
            )
    if args.cross_overlap_relation != "none":
        if args.graph_layout != "separate":
            raise ValueError(
                "--cross-overlap-relation sparse requires "
                "--graph-layout separate."
            )
        if args.post_gat_consensus_graph != "none":
            raise ValueError(
                "--cross-overlap-relation sparse does not create C nodes; "
                "do not enable --post-gat-consensus-graph at the same time."
            )
        if args.consensus_graph_transport != "none":
            raise ValueError(
                "--cross-overlap-relation sparse is mutually exclusive with "
                "--consensus-graph-transport."
            )
        if args.cross_overlap_prior_weight < 0:
            raise ValueError(
                "--cross-overlap-prior-weight must be nonnegative."
            )
        if args.cross_overlap_gamma_init < 0:
            raise ValueError(
                "--cross-overlap-gamma-init must be nonnegative."
            )
        if args.cross_overlap_second_gamma_init < 0:
            raise ValueError(
                "--cross-overlap-second-gamma-init must be nonnegative."
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
        and args.consensus_graph_transport_stage != "post-gat2"
    ):
        raise ValueError(
            "--consensus-graph-transport-stage inter-gat/alternating "
            "requires --consensus-graph-transport bidirectional."
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
        and args.consensus_graph_transport_operator != "transport"
    ):
        raise ValueError(
            "--consensus-graph-transport-operator sheaf/dcsi/cgsa requires "
            "--consensus-graph-transport bidirectional."
        )
    if (
        args.consensus_graph_transport_operator
        in ("sheaf", "dcsi", "cgsa")
        and args.post_gat_consensus_graph != "intersection-mediator"
    ):
        raise ValueError(
            "--consensus-graph-transport-operator sheaf/dcsi/cgsa requires "
            "--post-gat-consensus-graph intersection-mediator."
        )
    if args.graph_layout == "joint" and args.cnn_branch != "original":
        raise ValueError(
            "--cnn-branch gsdg currently requires "
            "--graph-layout separate."
        )
    if args.graph_layout == "joint" and args.cnn_layout != "joint":
        raise ValueError(
            "--cnn-layout separate currently requires "
            "--graph-layout separate."
        )
    if args.cnn_share_weights and args.cnn_layout != "separate":
        raise ValueError(
            "--cnn-share-weights requires --cnn-layout separate."
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
    if args.dummy_logit_dim < 0:
        raise ValueError("--dummy-logit-dim must be nonnegative.")
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
        "demo_train | Stage 8: private dual GSDG + optional cross-modal relation"
    )
    print(
        f"CNN: {cnn_configuration_tag(args)} | "
        f"graph-layout: {args.graph_layout} | "
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
        if args.topology_rewiring != "none":
            print(
                "Topology rewiring: "
                f"{args.topology_rewiring} | "
                f"side={args.topology_audit_side} | "
                f"evidence={args.topology_evidence} | "
                f"impurity={args.topology_impurity_mode} | "
                f"cells={cell_data['cell_count']} | "
                "veto/advocacy/impurity-init="
                f"{args.topology_veto_init:g}/"
                f"{args.topology_advocacy_init:g}/"
                f"{args.topology_impurity_init:g} | "
                f"advocacy-k={args.topology_advocacy_k} | "
                f"advocacy-rag={args.topology_advocacy_rag_hops}hop | "
                f"freeze-adv={int(args.topology_freeze_advocacy)} | "
                "mediator disabled"
            )
        if args.cross_overlap_relation != "none":
            print(
                "Sparse overlap relation: "
                f"{args.cross_overlap_relation} | "
                f"stage={args.cross_overlap_stage} | "
                f"message={args.cross_overlap_message} | "
                f"fusion={args.cross_overlap_fusion} | "
                f"prior-weight={args.cross_overlap_prior_weight:g} | "
                "gamma-init="
                f"{args.cross_overlap_gamma_init:g}/"
                f"{args.cross_overlap_second_gamma_init:g} | "
                "explicit C graph disabled"
            )
        if args.post_gat_consensus_graph != "none":
            resolved_mediator_count = cell_data["cell_count"]
            private_weight = 1.0 - args.consensus_graph_weight
            if args.consensus_graph_transport == "bidirectional":
                if args.consensus_graph_transport_operator == "cgsa":
                    fusion_detail = (
                        "context-gated sheaf alignment, "
                        f"stage={args.consensus_graph_transport_stage}, "
                        f"restriction={args.sheaf_restriction}, "
                        f"rank={args.sheaf_rank}, "
                        f"steps={args.sheaf_steps}, "
                        f"edge-topk={args.dcsi_edge_topk}, "
                        f"alpha-max={args.cgsa_alpha_max:g}, "
                        "gamma-init="
                        f"{args.consensus_graph_transport_gamma_init:g}, "
                        f"consensus-channel={int(args.cgsa_consensus_channel)}, "
                        f"lambdaE={args.sheaf_energy_weight:g}; "
                        "pixel C fusion disabled"
                    )
                elif args.consensus_graph_transport_operator == "dcsi":
                    fusion_detail = (
                        "dual-channel contextual sheaf interaction, "
                        f"stage={args.consensus_graph_transport_stage}, "
                        f"restriction={args.sheaf_restriction}, "
                        f"rank={args.sheaf_rank}, "
                        f"blocks={args.sheaf_steps}, "
                        f"edge-topk={args.dcsi_edge_topk}, "
                        "mu="
                        f"{args.dcsi_consensus_mix:g}/"
                        f"{args.dcsi_conflict_mix:g}, "
                        "gamma-sem/conf="
                        f"{args.dcsi_semantic_gamma_init:g}/"
                        f"{args.dcsi_conflict_gamma_init:g}, "
                        f"lambdaE={args.sheaf_energy_weight:g}; "
                        "pixel C fusion disabled"
                    )
                elif args.consensus_graph_transport_operator == "sheaf":
                    fusion_detail = (
                        "cellular-sheaf diffusion, "
                        f"stage={args.consensus_graph_transport_stage}, "
                        f"restriction={args.sheaf_restriction}, "
                        f"rank={args.sheaf_rank}, "
                        f"steps={args.sheaf_steps}, "
                        f"lambdaE={args.sheaf_energy_weight:g}, "
                        "alpha-init="
                        f"{args.consensus_graph_transport_gamma_init:g}; "
                        "pixel C fusion disabled"
                    )
                else:
                    fusion_detail = (
                        "C-mediated bidirectional transport, "
                        f"stage={args.consensus_graph_transport_stage}, "
                        f"fusion={args.consensus_graph_transport_fusion}, "
                        f"message={args.consensus_graph_transport_message}, "
                        f"lambda={args.consensus_graph_transport_lambda:g}, "
                        f"eta={args.consensus_graph_transport_prior_weight:g}, "
                        "gamma-init="
                        f"{args.consensus_graph_transport_gamma_init:g}/"
                        f"{args.consensus_graph_transport_second_gamma_init:g}; "
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
    classifier_output_dim = (
        class_count
        if args.dummy_logit_dim <= 0
        else args.dummy_logit_dim
    )
    print(
        f"Classifier logits: {classifier_output_dim} | "
        f"evaluated real classes: {class_count} | "
        f"dummy logits: {max(0, classifier_output_dim - class_count)}"
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
        f"{topology_rewiring_configuration_tag(args)}_"
        f"{cross_overlap_configuration_tag(args)}_"
        f"fdsm-{args.fdsm_scope}_"
        f"{cnn_configuration_tag(args)}_"
        f"lidarmod-{args.lidar_modulation}_"
        f"{dummy_logit_configuration_tag(args, class_count)}_results"
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
