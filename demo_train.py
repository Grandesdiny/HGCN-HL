"""Demo trainer: private dual GSDG graphs plus active lightweight ablations.

The fixed hypergraph/HGCN path is replaced by GSDG graph/GAT propagation.
The default uses independent HSI and LiDAR graphs; the previous concatenated
node graph remains selectable. Retired experimental branches are archived
outside this main demo entry.
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
    build_rag_hop_candidates,
    build_superpixel_spatial_prior,
    minmax_normalize,
    normalized_sparse_assignments,
    set_seed,
    split_fixed_samples_per_class,
    superpixel_height_distribution,
)
from utils import (
    get_felzenszwalb_Segs,
    get_HSI_LiDAR_data,
    get_HSI_performance,
    get_SLIC_Segs,
)


STAGE = "stage8_clean_demo"


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
        "--dual-transport-arbitration",
        choices=("none", "post-gat2", "post-gat", "postgat"),
        default="none",
        help=(
            "Geometry-semantic dual transport arbitration after private "
            "GAT2 and before graph readout. It compares overlap coverage "
            "transport with feature-similarity transport on the same "
            "nonzero overlap support. Default: none."
        ),
    )
    parser.add_argument(
        "--dta-dk",
        type=int,
        default=64,
        help="Query/key dimension for dual transport arbitration.",
    )
    parser.add_argument(
        "--dta-tau",
        type=float,
        default=0.1,
        help="Semantic softmax temperature for dual transport arbitration.",
    )
    parser.add_argument(
        "--dta-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scale for dual transport arbitration. "
            "Zero makes the initial forward pass match the baseline."
        ),
    )
    parser.add_argument(
        "--dta-variant",
        choices=("simple", "full"),
        default="simple",
        help=(
            "Dual transport arbitration variant. 'simple' keeps semantic "
            "transport on overlap edges only; 'full' adds optional "
            "nonlocal retrieval, Sinkhorn normalization, and auxiliary "
            "regularization. Default: simple."
        ),
    )
    parser.add_argument(
        "--dta-nonlocal-topk",
        type=int,
        default=10,
        help=(
            "Extra full-image semantic neighbors per receiver node for "
            "--dta-variant full. Set 0 to disable nonlocal retrieval. "
            "Default: 10."
        ),
    )
    parser.add_argument(
        "--dta-normalization",
        choices=("softmax", "sinkhorn"),
        default="sinkhorn",
        help=(
            "Semantic transport normalization used by --dta-variant full. "
            "The simple variant always uses sparse row-softmax. "
            "Default: sinkhorn."
        ),
    )
    parser.add_argument(
        "--dta-sinkhorn-iters",
        type=int,
        default=5,
        help="Sinkhorn iterations for --dta-variant full. Default: 5.",
    )
    parser.add_argument(
        "--dta-aux-sparse-weight",
        type=float,
        default=0.05,
        help=(
            "Weight for the full-DTA gate sparsity auxiliary loss. "
            "Only active with --dta-variant full. Default: 0.05."
        ),
    )
    parser.add_argument(
        "--dta-aux-align-weight",
        type=float,
        default=0.10,
        help=(
            "Weight for the full-DTA harmonious overlap alignment loss. "
            "Only active with --dta-variant full. Default: 0.10."
        ),
    )
    parser.add_argument(
        "--dta-aux-entropy-weight",
        type=float,
        default=0.01,
        help=(
            "Weight for the full-DTA semantic transport entropy penalty. "
            "Only active with --dta-variant full. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--dta-film",
        action="store_true",
        help=(
            "Enable conflict-aware FiLM after DTA. It uses DTA's "
            "arbitrated cross-modal context to modulate only a shared "
            "prefix of each modality node feature, with conflict-driven "
            "shrinkage. Default: disabled."
        ),
    )
    parser.add_argument(
        "--dta-film-ds",
        type=int,
        default=64,
        help=(
            "Number of leading channels treated as the shared FiLM "
            "subspace. Remaining channels are private bypass. Default: 64."
        ),
    )
    parser.add_argument(
        "--dta-film-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scale for DTA-FiLM. Zero makes the initial "
            "forward pass match DTA without FiLM. Default: 0."
        ),
    )
    parser.add_argument(
        "--assignment-graph",
        choices=("none", "post-gat2", "post-gat", "postgat"),
        default="none",
        help=(
            "SEGMN-style overlap-restricted assignment graph. It runs "
            "after private HSI/LiDAR GAT2, treats every nonzero overlap "
            "pair (HSI-SP i, LiDAR-SP j) as an assignment node, applies "
            "one assignment graph convolution, and writes back to the "
            "private nodes with zero-init residual scales. Default: none."
        ),
    )
    parser.add_argument(
        "--assignment-graph-topk",
        type=int,
        default=8,
        help=(
            "Top-k assignment neighbors retained per overlap-pair node. "
            "Default: 8."
        ),
    )
    parser.add_argument(
        "--assignment-graph-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scale for assignment graph writeback. "
            "Zero makes the initial forward pass match the private "
            "GAT2 baseline. Default: 0."
        ),
    )
    parser.add_argument(
        "--assignment-graph-output",
        choices=("writeback", "third-branch", "both"),
        default="writeback",
        help=(
            "How the assignment graph contributes after its C-GNN. "
            "'writeback' applies post-GAT2 residual node correction; "
            "'third-branch' projects assignment nodes directly to pixels "
            "and fuses them with private graph pixels; 'both' does both. "
            "Default: writeback."
        ),
    )
    parser.add_argument(
        "--assignment-graph-weight",
        type=float,
        default=0.1,
        help=(
            "Weight of the assignment third branch when "
            "--assignment-graph-output is third-branch or both. "
            "Default: 0.1."
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
        choices=("bilinear", "dual-channel", "dual-path"),
        default="dual-channel",
        help=(
            "Node update after sparse overlap transport. 'bilinear' "
            "uses inter-modal message plus multiplicative evidence; "
            "'dual-channel' additionally aggregates consensus/product "
            "and conflict/difference edge channels in one MLP; "
            "'dual-path' keeps consensus and conflict in separate "
            "MLP/gate/gamma update paths. Default: dual-channel."
        ),
    )
    parser.add_argument(
        "--cross-overlap-edge-attrs",
        choices=("none", "physical"),
        default="none",
        help=(
            "Optional edge descriptor bias for sparse overlap qk-prior "
            "attention. 'physical' adds [coverage_H, coverage_L, IoU, "
            "HSI-SAM, delta-height, DSM-gradient] through an edge MLP. "
            "Default: none."
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
    parser.add_argument(
        "--cross-overlap-min-coverage",
        type=float,
        default=0.0,
        help=(
            "Prune sparse overlap edges whose HSI and LiDAR coverage "
            "are both below this threshold. Use 0.05 for the 2a "
            "edge-pruning ablation. Default: 0 disables coverage "
            "pruning."
        ),
    )
    parser.add_argument(
        "--cross-overlap-iou-topk",
        type=int,
        default=0,
        help=(
            "After coverage pruning, keep only edges that are in the "
            "IoU top-k of both endpoint nodes. Use 3 for the 2a "
            "edge-pruning ablation. Default: 0 disables top-k pruning."
        ),
    )
    parser.add_argument(
        "--cross-overlap-fragmentation-alpha",
        type=float,
        default=0.0,
        help=(
            "Fixed alpha for 2c fragmentation gating on sparse overlap "
            "node updates: gate=exp(-alpha*(1-max_iou)). Try 1, 2, "
            "or 4 after 2b/2a is validated. Default: 0 disables it."
        ),
    )
    parser.add_argument(
        "--overlap-distill-weight",
        type=float,
        default=0.0,
        help=(
            "Weight of confidence-weighted mutual distillation on "
            "HSI/LiDAR overlap edges. Default: 0 disables it."
        ),
    )
    parser.add_argument(
        "--overlap-distill-temperature",
        type=float,
        default=2.0,
        help="Temperature used by overlap-edge distillation. Default: 2.",
    )
    parser.add_argument(
        "--overlap-distill-iou-threshold",
        type=float,
        default=0.3,
        help=(
            "Only overlap edges with IoU above this threshold are used "
            "for distillation. Default: 0.3."
        ),
    )
    parser.add_argument(
        "--overlap-distill-margin",
        type=float,
        default=0.1,
        help=(
            "Teacher confidence must exceed student confidence by this "
            "margin. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--overlap-distill-warmup-ratio",
        type=float,
        default=0.2,
        help=(
            "Fraction of epochs with zero distillation before linear "
            "ramp-up. Default: 0.2."
        ),
    )
    parser.add_argument(
        "--overlap-distill-confidence",
        choices=("max-prob", "neg-entropy"),
        default="max-prob",
        help=(
            "Confidence score used to pick the teacher on each overlap "
            "edge. 'neg-entropy' is normalized as 1-H(p)/log(C). "
            "Default: max-prob."
        ),
    )
    parser.add_argument(
        "--evidence-fusion",
        choices=("none", "dirichlet"),
        default="none",
        help=(
            "Optional evidence decision fusion for separate HSI/LiDAR "
            "graph branches. 'dirichlet' uses two branch evidence heads, "
            "Dempster pixel fusion, and EDL classification loss. "
            "Default: none."
        ),
    )
    parser.add_argument(
        "--bcq-interaction",
        choices=("none", "post-gat2", "post-gat", "postgat"),
        default="none",
        help=(
            "Bipartite-grounded Consensus Quotient interaction. "
            "The first implemented slot is post-GAT2: private GAT1/GAT2 "
            "then BCQ membership-consistency residual writeback. "
            "Default: none."
        ),
    )
    parser.add_argument(
        "--bcq-anchor-ratio",
        type=int,
        default=4,
        help=(
            "Number of shared BCQ anchors per real dataset class. "
            "Default: 4."
        ),
    )
    parser.add_argument(
        "--bcq-anchor-topk",
        type=int,
        default=8,
        help=(
            "Top-k anchors retained in every node-to-anchor membership "
            "distribution. Default: 8."
        ),
    )
    parser.add_argument(
        "--bcq-anchor-dk",
        type=int,
        default=32,
        help="BCQ node/anchor key dimension. Default: 32.",
    )
    parser.add_argument(
        "--bcq-conflict-alpha",
        type=float,
        default=2.0,
        help=(
            "Fixed JSD conflict gate strength in BCQ: "
            "gate=exp(-alpha*JSD). Default: 2."
        ),
    )
    parser.add_argument(
        "--bcq-gamma-init",
        type=float,
        default=0.0,
        help=(
            "Initial residual scale for BCQ writeback. Zero makes the "
            "initial forward pass identical to the private-graph path. "
            "Default: 0."
        ),
    )
    parser.add_argument(
        "--consensus-token-fusion",
        choices=("none", "post-gat2", "post-gat", "postgat"),
        default="none",
        help=(
            "Consensus-token classification fusion head. It runs after "
            "the two private GAT branches, uses shared class-grouped "
            "tokens to build HSI/LiDAR membership distributions, and "
            "fuses token logits into the main logits with a zero-init "
            "logit gate. Default: none."
        ),
    )
    parser.add_argument(
        "--consensus-token-ratio",
        type=int,
        default=4,
        help=(
            "Number of consensus tokens per real dataset class. "
            "Default: 4."
        ),
    )
    parser.add_argument(
        "--consensus-token-heads",
        type=int,
        default=4,
        help="Number of heads for token-to-node attention. Default: 4.",
    )
    parser.add_argument(
        "--consensus-token-topk",
        type=int,
        default=0,
        help=(
            "Optional top-k over token membership per node. Default: 0 "
            "keeps dense membership."
        ),
    )
    parser.add_argument(
        "--consensus-token-tau",
        type=float,
        default=1.0,
        help="Temperature for node-to-token membership. Default: 1.",
    )
    parser.add_argument(
        "--consensus-token-ce-weight",
        type=float,
        default=0.4,
        help=(
            "Auxiliary CE weight for HSI/LiDAR consensus-token pixel "
            "probabilities. Active only when --consensus-token-fusion "
            "is enabled. Default: 0.4."
        ),
    )
    parser.add_argument(
        "--consensus-token-agreement-weight",
        type=float,
        default=0.1,
        help=(
            "JSD agreement weight between semantic memberships and "
            "overlap-transported memberships. Default: 0.1."
        ),
    )
    parser.add_argument(
        "--consensus-token-usage-weight",
        type=float,
        default=0.01,
        help=(
            "Anti-collapse token-usage KL weight. Default: 0.01."
        ),
    )
    parser.add_argument(
        "--consensus-token-fusion-gate-init",
        type=float,
        default=0.0,
        help=(
            "Initial logit-space fusion gate. Zero makes final logits "
            "start exactly from the main classifier. Default: 0."
        ),
    )
    parser.add_argument(
        "--edl-kl-weight",
        type=float,
        default=0.1,
        help=(
            "KL weight in EDL loss when --evidence-fusion dirichlet. "
            "Default: 0.1."
        ),
    )
    parser.add_argument(
        "--edl-anneal-ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of epochs used to anneal the EDL KL term from 0 "
            "to 1. Default: 0.5."
        ),
    )
    # Archived ablation switches are kept as hidden compatibility flags
    # but their active choices are removed from the main entry point.
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
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Base random seed. Run k uses seed + k, so --runs 10 with "
            "--seed 100 uses seeds 100..109. Default: 0."
        ),
    )
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













def dummy_logit_configuration_tag(args, class_count):
    if args.dummy_logit_dim <= 0:
        return "dummy-none"
    return f"dummy{args.dummy_logit_dim}-real{class_count}"


def seed_configuration_tag(args):
    return f"seed{args.seed}"


def dta_configuration_tag(args):
    if args.dual_transport_arbitration == "none":
        return "dta-none"
    tag = (
        f"dta-{args.dual_transport_arbitration}-"
        f"{args.dta_variant}-"
        f"dk{args.dta_dk}-"
        f"tau{args.dta_tau:g}-"
        f"g{args.dta_gamma_init:g}"
    )
    if args.dta_variant == "full":
        tag += (
            f"-nl{args.dta_nonlocal_topk}-"
            f"{args.dta_normalization}-"
            f"sh{args.dta_sinkhorn_iters}"
        )
    if args.dta_film:
        tag += (
            f"-filmds{args.dta_film_ds}-"
            f"filmg{args.dta_film_gamma_init:g}"
        )
    return tag


def assignment_graph_configuration_tag(args):
    if args.assignment_graph == "none":
        return "ag-none"
    return (
        f"ag-{args.assignment_graph}-"
        f"top{args.assignment_graph_topk}-"
        f"g{args.assignment_graph_gamma_init:g}-"
        f"out{args.assignment_graph_output}-"
        f"w{args.assignment_graph_weight:g}"
    )



def cross_overlap_configuration_tag(args):
    if args.cross_overlap_relation == "none":
        return "xo-none"
    return (
        f"xo-{args.cross_overlap_relation}-"
        f"st{args.cross_overlap_stage}-"
        f"msg{args.cross_overlap_message}-"
        f"fu{args.cross_overlap_fusion}-"
        f"ea{args.cross_overlap_edge_attrs}-"
        f"pw{args.cross_overlap_prior_weight:g}-"
        f"g{args.cross_overlap_gamma_init:g}-"
        f"g2{args.cross_overlap_second_gamma_init:g}-"
        f"mc{args.cross_overlap_min_coverage:g}-"
        f"ik{args.cross_overlap_iou_topk}-"
        f"fa{args.cross_overlap_fragmentation_alpha:g}"
    )


def overlap_distill_configuration_tag(args):
    if args.overlap_distill_weight <= 0:
        return "od-none"
    return (
        f"od-w{args.overlap_distill_weight:g}-"
        f"t{args.overlap_distill_temperature:g}-"
        f"iou{args.overlap_distill_iou_threshold:g}-"
        f"m{args.overlap_distill_margin:g}-"
        f"wr{args.overlap_distill_warmup_ratio:g}-"
        f"conf{args.overlap_distill_confidence}"
    )


def evidence_fusion_configuration_tag(args):
    if args.evidence_fusion == "none":
        return "ef-none"
    return (
        f"ef-{args.evidence_fusion}-"
        f"kl{args.edl_kl_weight:g}-"
        f"ar{args.edl_anneal_ratio:g}"
    )


def bcq_configuration_tag(args):
    if args.bcq_interaction == "none":
        return "bcq-none"
    return (
        f"bcq-{args.bcq_interaction}-"
        f"r{args.bcq_anchor_ratio}-"
        f"top{args.bcq_anchor_topk}-"
        f"dk{args.bcq_anchor_dk}-"
        f"a{args.bcq_conflict_alpha:g}-"
        f"g{args.bcq_gamma_init:g}"
    )


def consensus_token_configuration_tag(args):
    if args.consensus_token_fusion == "none":
        return "ct-none"
    return (
        f"ct-{args.consensus_token_fusion}-"
        f"r{args.consensus_token_ratio}-"
        f"h{args.consensus_token_heads}-"
        f"top{args.consensus_token_topk}-"
        f"tau{args.consensus_token_tau:g}-"
        f"ce{args.consensus_token_ce_weight:g}-"
        f"ag{args.consensus_token_agreement_weight:g}-"
        f"us{args.consensus_token_usage_weight:g}-"
        f"g{args.consensus_token_fusion_gate_init:g}"
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







def _superpixel_feature_means(assignment, pixel_features):
    assignment = coo_matrix(assignment, dtype=np.float32).tocsr()
    flat_features = np.asarray(pixel_features, dtype=np.float32).reshape(
        assignment.shape[0],
        -1,
    )
    area = np.asarray(assignment.sum(axis=0)).reshape(-1).astype(np.float32)
    sums = assignment.transpose() @ flat_features
    return np.asarray(sums, dtype=np.float32) / np.maximum(
        area[:, None],
        1e-6,
    )


def _robust_unit_scale(values):
    values = np.asarray(values, dtype=np.float32)
    positive = values[np.isfinite(values) & (values > 0)]
    if positive.size == 0:
        return np.zeros_like(values, dtype=np.float32)
    scale = np.percentile(positive, 95)
    return np.clip(values / max(float(scale), 1e-6), 0.0, 1.0).astype(
        np.float32
    )


def _topk_edge_mask_by_group(scores, groups, k):
    scores = np.asarray(scores)
    groups = np.asarray(groups, dtype=np.int64)
    keep = np.zeros(scores.shape[0], dtype=bool)
    if k <= 0 or scores.size == 0:
        keep[:] = True
        return keep
    for group in np.unique(groups):
        edge_index = np.flatnonzero(groups == group)
        if edge_index.size <= k:
            keep[edge_index] = True
        else:
            order = np.argsort(scores[edge_index], kind="mergesort")
            keep[edge_index[order[-k:]]] = True
    return keep


def _max_edge_value_by_group(values, groups, group_count):
    max_values = np.zeros(int(group_count), dtype=np.float32)
    if values.size == 0:
        return max_values
    np.maximum.at(
        max_values,
        np.asarray(groups, dtype=np.int64),
        np.asarray(values, dtype=np.float32),
    )
    return max_values


def build_sparse_overlap_relation_data(
    hsi_assignment,
    lidar_assignment,
    hsi_features=None,
    lidar_image=None,
    edge_attrs="none",
    min_coverage=0.0,
    iou_topk=0,
):
    """Build sparse HSI/LiDAR superpixel co-occurrence edges.

    Edges are the nonzero entries of M = Q_H^T Q_L. Directional edge
    weights are receiver-normalized coverages:
    H<-L uses M_ij / |S_i^H| and L<-H uses M_ij / |S_j^L|.
    Optional 2a pruning first removes tiny bidirectional coverages and
    then keeps only bidirectional endpoint IoU top-k edges.
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
    iou = overlap_values / np.maximum(
        h_area[h_index] + l_area[l_index] - overlap_values,
        1e-6,
    )
    h_node_count = hsi_sparse.shape[1]
    l_node_count = lidar_sparse.shape[1]
    original_edge_count = int(overlap_values.size)
    coverage_pruned_edges = 0
    topk_pruned_edges = 0
    min_coverage = float(min_coverage)
    iou_topk = int(iou_topk)
    if min_coverage > 0:
        coverage_keep = (
            (h_coverage >= min_coverage)
            | (l_coverage >= min_coverage)
        )
        coverage_pruned_edges = int(
            coverage_keep.size - coverage_keep.sum()
        )
        h_index = h_index[coverage_keep]
        l_index = l_index[coverage_keep]
        overlap_values = overlap_values[coverage_keep]
        h_coverage = h_coverage[coverage_keep]
        l_coverage = l_coverage[coverage_keep]
        iou = iou[coverage_keep]
    if iou_topk > 0 and iou.size:
        before_topk = int(iou.size)
        h_topk_keep = _topk_edge_mask_by_group(iou, h_index, iou_topk)
        l_topk_keep = _topk_edge_mask_by_group(iou, l_index, iou_topk)
        topk_keep = h_topk_keep & l_topk_keep
        topk_pruned_edges = int(before_topk - topk_keep.sum())
        h_index = h_index[topk_keep]
        l_index = l_index[topk_keep]
        overlap_values = overlap_values[topk_keep]
        h_coverage = h_coverage[topk_keep]
        l_coverage = l_coverage[topk_keep]
        iou = iou[topk_keep]
    if overlap_values.size == 0:
        raise ValueError(
            "Sparse overlap pruning removed every HSI/LiDAR edge. "
            "Lower --cross-overlap-min-coverage or increase "
            "--cross-overlap-iou-topk."
        )
    h_max_iou = _max_edge_value_by_group(iou, h_index, h_node_count)
    l_max_iou = _max_edge_value_by_group(iou, l_index, l_node_count)
    h_fragmentation = np.clip(1.0 - h_max_iou, 0.0, 1.0)
    l_fragmentation = np.clip(1.0 - l_max_iou, 0.0, 1.0)
    relation = {
        "h_index": h_index,
        "l_index": l_index,
        "overlap": overlap_values.astype(np.float32),
        "h_coverage": h_coverage.astype(np.float32),
        "l_coverage": l_coverage.astype(np.float32),
        "iou": iou.astype(np.float32),
        "h_area": h_area.astype(np.float32),
        "l_area": l_area.astype(np.float32),
        "h_fragmentation": h_fragmentation.astype(np.float32),
        "l_fragmentation": l_fragmentation.astype(np.float32),
        "h_node_count": int(h_node_count),
        "l_node_count": int(l_node_count),
        "edge_count": int(overlap_values.size),
        "original_edge_count": original_edge_count,
        "coverage_pruned_edges": coverage_pruned_edges,
        "topk_pruned_edges": topk_pruned_edges,
        "retained_edge_fraction": float(
            overlap_values.size / max(original_edge_count, 1)
        ),
        "min_coverage": float(min_coverage),
        "iou_topk": int(iou_topk),
        "density": float(
            overlap_values.size / max(h_node_count * l_node_count, 1)
        ),
        "edge_attribute_mode": "none",
    }
    if edge_attrs == "none":
        return relation
    if edge_attrs != "physical":
        raise ValueError(f"Unsupported sparse overlap edge attrs: {edge_attrs}")
    if hsi_features is None or lidar_image is None:
        raise ValueError(
            "Physical sparse overlap edge attributes require HSI and LiDAR "
            "pixel features."
        )

    hsi_features = np.asarray(hsi_features, dtype=np.float32)
    lidar_image = np.asarray(lidar_image, dtype=np.float32)
    if lidar_image.ndim == 3:
        lidar_image = lidar_image[:, :, 0]
    hsi_mean_h = _superpixel_feature_means(hsi_sparse, hsi_features)
    hsi_mean_l = _superpixel_feature_means(lidar_sparse, hsi_features)
    hsi_left = hsi_mean_h[h_index]
    hsi_right = hsi_mean_l[l_index]
    sam_numerator = np.sum(hsi_left * hsi_right, axis=1)
    sam_denominator = (
        np.linalg.norm(hsi_left, axis=1)
        * np.linalg.norm(hsi_right, axis=1)
    )
    cosine = np.clip(
        sam_numerator / np.maximum(sam_denominator, 1e-6),
        -1.0,
        1.0,
    )
    sam = (np.arccos(cosine) / np.pi).astype(np.float32)

    lidar_stack = lidar_image[:, :, np.newaxis]
    h_height = _superpixel_feature_means(hsi_sparse, lidar_stack)[:, 0]
    l_height = _superpixel_feature_means(lidar_sparse, lidar_stack)[:, 0]
    delta_height = np.abs(h_height[h_index] - l_height[l_index]).astype(
        np.float32
    )

    gradient_y, gradient_x = np.gradient(lidar_image)
    gradient = np.sqrt(gradient_x * gradient_x + gradient_y * gradient_y)
    weighted_lidar = lidar_sparse.multiply(
        gradient.reshape(-1, 1).astype(np.float32)
    )
    gradient_overlap = (hsi_sparse.transpose() @ weighted_lidar).tocoo()
    gradient_lookup = {
        int(row) * l_node_count + int(col): float(value)
        for row, col, value in zip(
            gradient_overlap.row,
            gradient_overlap.col,
            gradient_overlap.data,
        )
    }
    edge_keys = h_index * l_node_count + l_index
    gradient_sum = np.asarray(
        [gradient_lookup.get(int(key), 0.0) for key in edge_keys],
        dtype=np.float32,
    )
    boundary = gradient_sum / np.maximum(overlap_values, 1e-6)

    edge_attributes = np.stack(
        [
            h_coverage,
            l_coverage,
            iou.astype(np.float32),
            sam,
            _robust_unit_scale(delta_height),
            _robust_unit_scale(boundary),
        ],
        axis=1,
    ).astype(np.float32)
    relation["edge_attributes"] = edge_attributes
    relation["edge_attribute_mode"] = "physical"
    relation["edge_attribute_names"] = [
        "coverage_h",
        "coverage_l",
        "iou",
        "sam",
        "delta_height",
        "boundary",
    ]
    relation["edge_attribute_stats"] = {
        name: {
            "mean": float(edge_attributes[:, index].mean()),
            "std": float(edge_attributes[:, index].std()),
        }
        for index, name in enumerate(relation["edge_attribute_names"])
    }
    return relation


def build_overlap_pair_assignment(
    hsi_assignment,
    lidar_assignment,
    relation_data,
):
    """Build pixel-to-overlap-pair assignment from current H/L assignments."""

    hsi_sparse = coo_matrix(hsi_assignment, dtype=np.float32).tocsr()
    lidar_sparse = coo_matrix(lidar_assignment, dtype=np.float32).tocsr()
    if hsi_sparse.shape[0] != lidar_sparse.shape[0]:
        raise ValueError(
            "HSI and LiDAR assignments must have the same pixel count."
        )
    h_indices = np.asarray(relation_data["h_index"], dtype=np.int64)
    l_indices = np.asarray(relation_data["l_index"], dtype=np.int64)
    edge_lookup = {
        (int(h_node), int(l_node)): int(edge_index)
        for edge_index, (h_node, l_node) in enumerate(
            zip(h_indices, l_indices)
        )
    }
    rows = []
    columns = []
    values = []
    for pixel in range(hsi_sparse.shape[0]):
        h_start = hsi_sparse.indptr[pixel]
        h_end = hsi_sparse.indptr[pixel + 1]
        l_start = lidar_sparse.indptr[pixel]
        l_end = lidar_sparse.indptr[pixel + 1]
        pixel_h_nodes = hsi_sparse.indices[h_start:h_end]
        pixel_l_nodes = lidar_sparse.indices[l_start:l_end]
        pixel_h_values = hsi_sparse.data[h_start:h_end]
        pixel_l_values = lidar_sparse.data[l_start:l_end]
        for h_node, h_value in zip(pixel_h_nodes, pixel_h_values):
            for l_node, l_value in zip(pixel_l_nodes, pixel_l_values):
                edge_index = edge_lookup.get((int(h_node), int(l_node)))
                if edge_index is None:
                    continue
                rows.append(pixel)
                columns.append(edge_index)
                values.append(float(h_value * l_value))
    if not values:
        raise ValueError("No pixel-to-overlap-pair assignments were built.")
    return coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (
                np.asarray(rows, dtype=np.int64),
                np.asarray(columns, dtype=np.int64),
            ),
        ),
        shape=(hsi_sparse.shape[0], int(relation_data["edge_count"])),
        dtype=np.float32,
    )


def build_overlap_distill_tensors(overlap_data, device):
    if overlap_data is None:
        return None
    iou = overlap_data.get("iou")
    if iou is None:
        overlap = np.asarray(overlap_data["overlap"], dtype=np.float32)
        h_index = np.asarray(overlap_data["h_index"], dtype=np.int64)
        l_index = np.asarray(overlap_data["l_index"], dtype=np.int64)
        h_area = np.asarray(overlap_data["h_area"], dtype=np.float32)
        l_area = np.asarray(overlap_data["l_area"], dtype=np.float32)
        iou = overlap / np.maximum(
            h_area[h_index] + l_area[l_index] - overlap,
            1e-6,
        )
    return {
        "h_index": torch.as_tensor(
            overlap_data["h_index"],
            dtype=torch.long,
            device=device,
        ),
        "l_index": torch.as_tensor(
            overlap_data["l_index"],
            dtype=torch.long,
            device=device,
        ),
        "iou": torch.as_tensor(iou, dtype=torch.float32, device=device),
    }


def overlap_distill_ramp(epoch, total_epochs, warmup_ratio):
    warmup_epochs = int(round(total_epochs * warmup_ratio))
    if warmup_epochs <= 0:
        return 1.0
    if epoch <= warmup_epochs:
        return 0.0
    return min(
        1.0,
        (epoch - warmup_epochs)
        / max(total_epochs - warmup_epochs, 1),
    )


def _prediction_confidence(probabilities, mode):
    if mode == "max-prob":
        return probabilities.max(dim=1).values
    if mode == "neg-entropy":
        class_count = probabilities.shape[1]
        entropy = -torch.sum(
            probabilities * torch.log(probabilities.clamp_min(1e-12)),
            dim=1,
        )
        return 1.0 - entropy / np.log(max(class_count, 2))
    raise ValueError(f"Unsupported overlap distill confidence: {mode}")


def _distill_edge_probabilities(values, temperature, inputs_are_probabilities):
    if inputs_are_probabilities:
        log_values = torch.log(values.clamp_min(1e-12))
        return F.softmax(log_values / temperature, dim=-1)
    return F.softmax(values / temperature, dim=-1)


def _distill_edge_log_probabilities(
    values,
    temperature,
    inputs_are_probabilities,
):
    if inputs_are_probabilities:
        log_values = torch.log(values.clamp_min(1e-12))
        return F.log_softmax(log_values / temperature, dim=-1)
    return F.log_softmax(values / temperature, dim=-1)


def confidence_weighted_overlap_distill_loss(
    hsi_node_logits,
    lidar_node_logits,
    overlap_tensors,
    class_count,
    temperature=2.0,
    iou_threshold=0.3,
    margin=0.1,
    confidence_mode="max-prob",
    inputs_are_probabilities=False,
):
    """IoU-weighted strong-to-weak KL on HSI/LiDAR overlap edges."""
    if overlap_tensors is None:
        raise ValueError("Overlap distillation requires overlap tensors.")
    zero = (
        hsi_node_logits[:, :class_count].sum()
        + lidar_node_logits[:, :class_count].sum()
    ) * 0.0
    h_index = overlap_tensors["h_index"]
    l_index = overlap_tensors["l_index"]
    edge_iou = overlap_tensors["iou"]
    mask = edge_iou > iou_threshold
    selected_count = int(mask.detach().sum().item())
    if selected_count == 0:
        return zero, {
            "selected_edges": 0,
            "h_to_l_edges": 0,
            "l_to_h_edges": 0,
            "h_to_l_ratio": 0.0,
            "l_to_h_ratio": 0.0,
            "mean_iou": 0.0,
            "mean_conf_h": 0.0,
            "mean_conf_l": 0.0,
            "raw_loss": 0.0,
        }

    h_edges = h_index[mask]
    l_edges = l_index[mask]
    weights = edge_iou[mask]
    h_logits = hsi_node_logits[:, :class_count]
    l_logits = lidar_node_logits[:, :class_count]
    h_selected = h_logits.index_select(0, h_edges)
    l_selected = l_logits.index_select(0, l_edges)
    p_h = _distill_edge_probabilities(
        h_selected,
        temperature,
        inputs_are_probabilities,
    )
    p_l = _distill_edge_probabilities(
        l_selected,
        temperature,
        inputs_are_probabilities,
    )
    conf_h = _prediction_confidence(p_h, confidence_mode)
    conf_l = _prediction_confidence(p_l, confidence_mode)

    h_teaches_l = conf_h > conf_l + margin
    l_teaches_h = conf_l > conf_h + margin
    h_to_l_loss = F.kl_div(
        _distill_edge_log_probabilities(
            l_selected,
            temperature,
            inputs_are_probabilities,
        ),
        p_h.detach(),
        reduction="none",
    ).sum(dim=-1)
    l_to_h_loss = F.kl_div(
        _distill_edge_log_probabilities(
            h_selected,
            temperature,
            inputs_are_probabilities,
        ),
        p_l.detach(),
        reduction="none",
    ).sum(dim=-1)
    edge_loss = (
        h_teaches_l.float() * h_to_l_loss
        + l_teaches_h.float() * l_to_h_loss
    )
    loss = (
        weights * edge_loss
    ).sum() / weights.sum().clamp_min(1e-8)
    loss = loss * temperature * temperature
    h_to_l_count = int(h_teaches_l.detach().sum().item())
    l_to_h_count = int(l_teaches_h.detach().sum().item())
    diagnostics = {
        "selected_edges": selected_count,
        "h_to_l_edges": h_to_l_count,
        "l_to_h_edges": l_to_h_count,
        "h_to_l_ratio": h_to_l_count / max(selected_count, 1),
        "l_to_h_ratio": l_to_h_count / max(selected_count, 1),
        "mean_iou": float(weights.detach().mean().item()),
        "mean_conf_h": float(conf_h.detach().mean().item()),
        "mean_conf_l": float(conf_l.detach().mean().item()),
        "raw_loss": float(loss.detach().item()),
    }
    return loss, diagnostics


def dirichlet_kl_to_uniform(alpha):
    beta = torch.ones_like(alpha)
    sum_alpha = alpha.sum(dim=-1, keepdim=True)
    sum_beta = beta.sum(dim=-1, keepdim=True)
    log_beta_alpha = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=-1, keepdim=True)
    )
    log_beta_uniform = (
        torch.lgamma(beta).sum(dim=-1, keepdim=True)
        - torch.lgamma(sum_beta)
    )
    digamma_delta = torch.digamma(alpha) - torch.digamma(sum_alpha)
    kl = (
        log_beta_alpha
        + log_beta_uniform
        + ((alpha - beta) * digamma_delta).sum(dim=-1, keepdim=True)
    )
    return kl.squeeze(-1)


def edl_classification_loss(
    alpha,
    labels,
    class_count,
    epoch,
    total_epochs,
    kl_weight=0.1,
    anneal_ratio=0.5,
):
    y_onehot = F.one_hot(labels, num_classes=class_count).float()
    strength = alpha.sum(dim=-1, keepdim=True)
    ce = (
        y_onehot
        * (torch.digamma(strength) - torch.digamma(alpha))
    ).sum(dim=-1)
    alpha_t = y_onehot + (1.0 - y_onehot) * alpha
    anneal_epochs = max(float(total_epochs) * float(anneal_ratio), 1.0)
    anneal = min(1.0, float(epoch) / anneal_epochs)
    kl = dirichlet_kl_to_uniform(alpha_t)
    return (ce + anneal * float(kl_weight) * kl).mean(), {
        "edl_ce": float(ce.detach().mean().item()),
        "edl_kl": float(kl.detach().mean().item()),
        "edl_anneal": float(anneal),
    }


def dempster_combine_dirichlet(alpha_h, alpha_l):
    """Combine two Dirichlet opinions with the simplified TMC rule."""
    class_count = alpha_h.shape[-1]
    evidence_h = (alpha_h - 1.0).clamp_min(0.0)
    evidence_l = (alpha_l - 1.0).clamp_min(0.0)
    strength_h = alpha_h.sum(dim=-1, keepdim=True)
    strength_l = alpha_l.sum(dim=-1, keepdim=True)
    belief_h = evidence_h / strength_h.clamp_min(1e-8)
    belief_l = evidence_l / strength_l.clamp_min(1e-8)
    uncertainty_h = class_count / strength_h.clamp_min(1e-8)
    uncertainty_l = class_count / strength_l.clamp_min(1e-8)
    agreement = (belief_h * belief_l).sum(dim=-1, keepdim=True)
    conflict = (
        belief_h.sum(dim=-1, keepdim=True)
        * belief_l.sum(dim=-1, keepdim=True)
        - agreement
    ).clamp(0.0, 1.0 - 1e-6)
    normalizer = (1.0 - conflict).clamp_min(1e-8)
    fused_belief = (
        belief_h * belief_l
        + belief_h * uncertainty_l
        + belief_l * uncertainty_h
    ) / normalizer
    fused_uncertainty = (
        uncertainty_h * uncertainty_l
    ) / normalizer
    fused_strength = class_count / fused_uncertainty.clamp_min(1e-8)
    fused_evidence = fused_belief * fused_strength
    fused_alpha = fused_evidence + 1.0
    diagnostics = {
        "h_uncertainty_mean": float(
            uncertainty_h.detach().mean().item()
        ),
        "l_uncertainty_mean": float(
            uncertainty_l.detach().mean().item()
        ),
        "fused_uncertainty_mean": float(
            fused_uncertainty.detach().mean().item()
        ),
        "dempster_conflict_mean": float(
            conflict.detach().mean().item()
        ),
        "dempster_conflict_max": float(
            conflict.detach().max().item()
        ),
    }
    return fused_alpha, diagnostics



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
        fragmentation_alpha=0.0,
    ):
        super().__init__()
        self.channels = channels
        self.message = message
        self.fusion = fusion
        self.prior_weight = prior_weight
        self.fragmentation_alpha = float(fragmentation_alpha)
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
        h_fragmentation = relation_data.get("h_fragmentation")
        if h_fragmentation is None:
            h_fragmentation = np.zeros(
                int(relation_data["h_node_count"]),
                dtype=np.float32,
            )
        l_fragmentation = relation_data.get("l_fragmentation")
        if l_fragmentation is None:
            l_fragmentation = np.zeros(
                int(relation_data["l_node_count"]),
                dtype=np.float32,
            )
        self.register_buffer(
            "h_fragmentation",
            torch.as_tensor(h_fragmentation, dtype=torch.float32),
            persistent=False,
        )
        self.register_buffer(
            "l_fragmentation",
            torch.as_tensor(l_fragmentation, dtype=torch.float32),
            persistent=False,
        )
        self.h_node_count = int(relation_data["h_node_count"])
        self.l_node_count = int(relation_data["l_node_count"])
        self.edge_count = int(relation_data["edge_count"])
        self.original_edge_count = int(
            relation_data.get("original_edge_count", self.edge_count)
        )
        self.coverage_pruned_edges = int(
            relation_data.get("coverage_pruned_edges", 0)
        )
        self.topk_pruned_edges = int(
            relation_data.get("topk_pruned_edges", 0)
        )
        self.retained_edge_fraction = float(
            relation_data.get("retained_edge_fraction", 1.0)
        )
        self.min_coverage = float(relation_data.get("min_coverage", 0.0))
        self.iou_topk = int(relation_data.get("iou_topk", 0))
        self.density = float(relation_data["density"])
        self.edge_attribute_mode = relation_data.get(
            "edge_attribute_mode",
            "none",
        )
        edge_attributes = relation_data.get("edge_attributes")
        if edge_attributes is not None:
            edge_attributes = np.asarray(edge_attributes, dtype=np.float32)
            self.register_buffer(
                "edge_attributes",
                torch.as_tensor(edge_attributes, dtype=torch.float32),
                persistent=False,
            )
            edge_hidden = max(8, 2 * edge_attributes.shape[1])
            edge_bias_output = nn.Linear(edge_hidden, 1)
            nn.init.zeros_(edge_bias_output.weight)
            nn.init.zeros_(edge_bias_output.bias)
            self.edge_bias = nn.Sequential(
                nn.Linear(edge_attributes.shape[1], edge_hidden),
                nn.LeakyReLU(),
                edge_bias_output,
            )
        else:
            self.register_buffer(
                "edge_attributes",
                None,
                persistent=False,
            )
            self.edge_bias = None

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
        self.h_consensus_path = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.h_conflict_path = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.l_consensus_path = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.l_conflict_path = nn.Sequential(
            nn.Linear(channels, channels),
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
        self.h_consensus_gate = make_gate(3 * channels)
        self.h_conflict_gate = make_gate(3 * channels)
        self.l_consensus_gate = make_gate(3 * channels)
        self.l_conflict_gate = make_gate(3 * channels)
        self.h_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.l_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.h_gamma2 = nn.Parameter(torch.tensor(float(second_gamma_init)))
        self.l_gamma2 = nn.Parameter(torch.tensor(float(second_gamma_init)))
        self.h_consensus_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.h_conflict_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.l_consensus_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.l_conflict_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.h_consensus_gamma2 = nn.Parameter(
            torch.tensor(float(second_gamma_init))
        )
        self.h_conflict_gamma2 = nn.Parameter(
            torch.tensor(float(second_gamma_init))
        )
        self.l_consensus_gamma2 = nn.Parameter(
            torch.tensor(float(second_gamma_init))
        )
        self.l_conflict_gamma2 = nn.Parameter(
            torch.tensor(float(second_gamma_init))
        )
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
        edge_bias = None
        if self.edge_bias is not None:
            edge_bias = self.edge_bias(self.edge_attributes).squeeze(1)

        h_logits = (
            self.h_query(hsi_nodes)[self.h_index]
            * self.l_key(lidar_nodes)[self.l_index]
        ).sum(dim=1) * self.scale
        h_logits = h_logits + self.prior_weight * torch.log(
            self.h_coverage.clamp_min(1e-6)
        )
        if edge_bias is not None:
            h_logits = h_logits + edge_bias
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
        if edge_bias is not None:
            l_logits = l_logits + edge_bias
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
        consensus_gamma,
        conflict_gamma,
        consensus_path,
        conflict_path,
        consensus_gate,
        conflict_gate,
        fragmentation_gate,
    ):
        product = bilinear_out(
            bilinear_left(nodes) * bilinear_right(message)
        )
        path_stats = {
            "consensus_gate_mean": None,
            "conflict_gate_mean": None,
            "consensus_delta_norm": None,
            "conflict_delta_norm": None,
            "consensus_gamma": None,
            "conflict_gamma": None,
        }
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
        elif self.fusion == "dual-path":
            consensus_delta = consensus_path(consensus)
            conflict_delta = conflict_path(conflict)
            consensus_gate_value = torch.sigmoid(
                consensus_gate(
                    torch.cat(
                        [
                            nodes,
                            consensus,
                            product,
                        ],
                        dim=1,
                    )
                )
            )
            conflict_gate_value = torch.sigmoid(
                conflict_gate(
                    torch.cat(
                        [
                            nodes,
                            conflict,
                            torch.abs(nodes - message),
                        ],
                        dim=1,
                    )
                )
            )
            updated = (
                nodes
                + fragmentation_gate
                * consensus_gamma
                * consensus_gate_value
                * consensus_delta
                + fragmentation_gate
                * conflict_gamma
                * conflict_gate_value
                * conflict_delta
            )
            gate = 0.5 * (consensus_gate_value + conflict_gate_value)
            delta = consensus_delta + conflict_delta
            path_stats = {
                "consensus_gate_mean": float(
                    consensus_gate_value.detach().mean().item()
                ),
                "conflict_gate_mean": float(
                    conflict_gate_value.detach().mean().item()
                ),
                "consensus_delta_norm": float(
                    consensus_delta.detach().norm(dim=1).mean().item()
                ),
                "conflict_delta_norm": float(
                    conflict_delta.detach().norm(dim=1).mean().item()
                ),
                "consensus_gamma": float(consensus_gamma.detach().item()),
                "conflict_gamma": float(conflict_gamma.detach().item()),
            }
            return updated, gate, product, delta, path_stats
        else:
            raise ValueError(f"Unsupported sparse overlap fusion: {self.fusion}")
        updated = nodes + fragmentation_gate * gamma * gate * delta
        return updated, gate, product, delta, path_stats

    def forward(self, hsi_nodes, lidar_nodes, round_index=1):
        if round_index == 1:
            h_gamma = self.h_gamma
            l_gamma = self.l_gamma
            h_consensus_gamma = self.h_consensus_gamma
            h_conflict_gamma = self.h_conflict_gamma
            l_consensus_gamma = self.l_consensus_gamma
            l_conflict_gamma = self.l_conflict_gamma
        elif round_index == 2:
            h_gamma = self.h_gamma2
            l_gamma = self.l_gamma2
            h_consensus_gamma = self.h_consensus_gamma2
            h_conflict_gamma = self.h_conflict_gamma2
            l_consensus_gamma = self.l_consensus_gamma2
            l_conflict_gamma = self.l_conflict_gamma2
        else:
            raise ValueError("round_index must be 1 or 2.")

        h_weights, l_weights = self._edge_weights(hsi_nodes, lidar_nodes)
        if self.fragmentation_alpha > 0:
            h_fragmentation_gate = torch.exp(
                -self.fragmentation_alpha * self.h_fragmentation
            ).unsqueeze(1)
            l_fragmentation_gate = torch.exp(
                -self.fragmentation_alpha * self.l_fragmentation
            ).unsqueeze(1)
        else:
            h_fragmentation_gate = hsi_nodes.new_ones(
                (self.h_node_count, 1)
            )
            l_fragmentation_gate = lidar_nodes.new_ones(
                (self.l_node_count, 1)
            )
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

        (
            updated_hsi,
            h_gate,
            h_product,
            h_delta,
            h_path_stats,
        ) = self._update_side(
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
            h_consensus_gamma,
            h_conflict_gamma,
            self.h_consensus_path,
            self.h_conflict_path,
            self.h_consensus_gate,
            self.h_conflict_gate,
            h_fragmentation_gate,
        )
        (
            updated_lidar,
            l_gate,
            l_product,
            l_delta,
            l_path_stats,
        ) = self._update_side(
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
            l_consensus_gamma,
            l_conflict_gamma,
            self.l_consensus_path,
            self.l_conflict_path,
            self.l_consensus_gate,
            self.l_conflict_gate,
            l_fragmentation_gate,
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
        edge_bias_mean = None
        edge_bias_std = None
        if self.edge_bias is not None:
            edge_bias_values = self.edge_bias(self.edge_attributes).detach()
            edge_bias_mean = float(edge_bias_values.mean().item())
            edge_bias_std = float(
                edge_bias_values.std(unbiased=False).item()
            )
        self.last_diagnostics = {
            "transport_operator": "sparse-overlap",
            "transport_mode": "sparse-overlap",
            "transport_round": int(round_index),
            "transport_message": self.message,
            "transport_fusion": self.fusion,
            "edge_attribute_mode": self.edge_attribute_mode,
            "edge_bias_mean": edge_bias_mean,
            "edge_bias_std": edge_bias_std,
            "edge_count": int(self.edge_count),
            "original_edge_count": int(self.original_edge_count),
            "coverage_pruned_edges": int(self.coverage_pruned_edges),
            "topk_pruned_edges": int(self.topk_pruned_edges),
            "retained_edge_fraction": float(self.retained_edge_fraction),
            "min_coverage": float(self.min_coverage),
            "iou_topk": int(self.iou_topk),
            "fragmentation_alpha": float(self.fragmentation_alpha),
            "h_fragmentation_mean": float(
                self.h_fragmentation.detach().mean().item()
            ),
            "l_fragmentation_mean": float(
                self.l_fragmentation.detach().mean().item()
            ),
            "h_fragmentation_gate_mean": float(
                h_fragmentation_gate.detach().mean().item()
            ),
            "l_fragmentation_gate_mean": float(
                l_fragmentation_gate.detach().mean().item()
            ),
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
            "h_consensus_gate_mean": h_path_stats[
                "consensus_gate_mean"
            ],
            "h_conflict_gate_mean": h_path_stats[
                "conflict_gate_mean"
            ],
            "l_consensus_gate_mean": l_path_stats[
                "consensus_gate_mean"
            ],
            "l_conflict_gate_mean": l_path_stats[
                "conflict_gate_mean"
            ],
            "h_consensus_delta_norm": h_path_stats[
                "consensus_delta_norm"
            ],
            "h_conflict_delta_norm": h_path_stats[
                "conflict_delta_norm"
            ],
            "l_consensus_delta_norm": l_path_stats[
                "consensus_delta_norm"
            ],
            "l_conflict_delta_norm": l_path_stats[
                "conflict_delta_norm"
            ],
            "h_consensus_gamma": h_path_stats["consensus_gamma"],
            "h_conflict_gamma": h_path_stats["conflict_gamma"],
            "l_consensus_gamma": l_path_stats["consensus_gamma"],
            "l_conflict_gamma": l_path_stats["conflict_gamma"],
        }
        return updated_hsi, updated_lidar

    def diagnostics(self):
        return self.last_diagnostics


class DualTransportArbitration(nn.Module):
    """Geometry/semantic transport arbitration on sparse overlap edges.

    The simple variant keeps semantic transport on the same sparse overlap
    support as geometry. The full variant optionally augments that support
    with nonlocal semantic top-k neighbors, applies Sinkhorn column limiting,
    and exposes auxiliary regularization losses.
    """

    def __init__(
        self,
        channels,
        relation_data,
        attention_d_k=64,
        tau=0.1,
        gamma_init=0.0,
        variant="simple",
        nonlocal_topk=10,
        normalization="sinkhorn",
        sinkhorn_iters=5,
    ):
        super().__init__()
        self.channels = int(channels)
        self.variant = variant
        self.tau = float(tau)
        self.scale = attention_d_k ** -0.5
        self.nonlocal_topk = int(nonlocal_topk)
        self.normalization = normalization
        self.sinkhorn_iters = int(sinkhorn_iters)
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
        self.h_node_count = int(relation_data["h_node_count"])
        self.l_node_count = int(relation_data["l_node_count"])
        self.edge_count = int(relation_data["edge_count"])
        h_coverage = torch.as_tensor(
            relation_data["h_coverage"],
            dtype=torch.float32,
        )
        l_coverage = torch.as_tensor(
            relation_data["l_coverage"],
            dtype=torch.float32,
        )
        self.register_buffer(
            "h_geo",
            self._normalize_edge_weights(
                h_coverage,
                self.h_index,
                self.h_node_count,
            ),
            persistent=False,
        )
        self.register_buffer(
            "l_geo",
            self._normalize_edge_weights(
                l_coverage,
                self.l_index,
                self.l_node_count,
            ),
            persistent=False,
        )
        self.register_buffer(
            "h_degree",
            self._segment_sum(
                torch.ones_like(h_coverage),
                self.h_index,
                self.h_node_count,
            ),
            persistent=False,
        )
        self.register_buffer(
            "l_degree",
            self._segment_sum(
                torch.ones_like(l_coverage),
                self.l_index,
                self.l_node_count,
            ),
            persistent=False,
        )
        h_geo_dense = torch.zeros(
            (self.h_node_count, self.l_node_count),
            dtype=torch.float32,
        )
        h_geo_dense[
            torch.as_tensor(relation_data["h_index"], dtype=torch.long),
            torch.as_tensor(relation_data["l_index"], dtype=torch.long),
        ] = self.h_geo
        self.register_buffer(
            "h_geo_dense",
            h_geo_dense,
            persistent=False,
        )
        l_geo_dense = torch.zeros(
            (self.l_node_count, self.h_node_count),
            dtype=torch.float32,
        )
        l_geo_dense[
            torch.as_tensor(relation_data["l_index"], dtype=torch.long),
            torch.as_tensor(relation_data["h_index"], dtype=torch.long),
        ] = self.l_geo
        self.register_buffer(
            "l_geo_dense",
            l_geo_dense,
            persistent=False,
        )
        self.register_buffer(
            "h_support_dense",
            h_geo_dense > 0,
            persistent=False,
        )
        self.register_buffer(
            "l_support_dense",
            l_geo_dense > 0,
            persistent=False,
        )

        self.q_h = nn.Linear(channels, attention_d_k, bias=False)
        self.k_l = nn.Linear(channels, attention_d_k, bias=False)
        self.q_l = nn.Linear(channels, attention_d_k, bias=False)
        self.k_h = nn.Linear(channels, attention_d_k, bias=False)
        self.v_l2h = nn.Linear(channels, channels)
        self.v_h2l = nn.Linear(channels, channels)
        self.gate_scale_h = nn.Parameter(torch.ones(()))
        self.gate_bias_h = nn.Parameter(torch.zeros(()))
        self.gate_scale_l = nn.Parameter(torch.ones(()))
        self.gate_bias_l = nn.Parameter(torch.zeros(()))
        self.gate_mlp_h = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.gate_mlp_l = nn.Sequential(
            nn.Linear(3, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )
        self.align_l_to_h = nn.Linear(channels, channels, bias=False)
        self.align_h_to_l = nn.Linear(channels, channels, bias=False)
        self.gamma_h = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_l = nn.Parameter(torch.tensor(float(gamma_init)))
        self.ln_h = nn.LayerNorm(channels)
        self.ln_l = nn.LayerNorm(channels)
        self.last_diagnostics = None
        self.last_D_h = None
        self.last_D_l = None
        self.last_gate_h = None
        self.last_gate_l = None
        self.last_ctx_h = None
        self.last_ctx_l = None
        self.last_aux_losses = {}

    @staticmethod
    def _segment_sum(values, index, segment_count):
        output = values.new_zeros(segment_count)
        output.index_add_(0, index, values)
        return output

    @classmethod
    def _normalize_edge_weights(cls, values, index, segment_count):
        denom = cls._segment_sum(values, index, segment_count)
        return values / denom[index].clamp_min(1e-12)

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

    @staticmethod
    def _mean_valid(values, valid):
        if bool(valid.any()):
            return float(values[valid].detach().mean().item())
        return 0.0

    @staticmethod
    def _standardize(values, valid):
        output = torch.zeros_like(values)
        if bool(valid.any()):
            selected = values[valid]
            centered = selected - selected.mean()
            scale = selected.std(unbiased=False).clamp_min(1e-8)
            output[valid] = centered / scale
        return output

    @staticmethod
    def _masked_row_softmax(logits, support):
        masked_logits = logits.masked_fill(~support, -1e9)
        weights = torch.softmax(masked_logits, dim=1)
        weights = weights * support.to(weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.nan_to_num(weights, nan=0.0)

    def _masked_sinkhorn(self, logits, support):
        log_weights = logits.masked_fill(~support, -1e9)
        valid_rows = support.any(dim=1)
        valid_cols = support.any(dim=0)
        for _ in range(self.sinkhorn_iters):
            row_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
            log_weights = torch.where(
                valid_rows[:, None],
                log_weights - row_norm,
                log_weights,
            )
            col_norm = torch.logsumexp(log_weights, dim=0, keepdim=True)
            log_weights = torch.where(
                valid_cols[None, :],
                log_weights - col_norm,
                log_weights,
            )
        row_norm = torch.logsumexp(log_weights, dim=1, keepdim=True)
        log_weights = torch.where(
            valid_rows[:, None],
            log_weights - row_norm,
            log_weights,
        )
        weights = log_weights.exp() * support.to(log_weights.dtype)
        weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)
        return torch.nan_to_num(weights, nan=0.0)

    def _one_direction(
        self,
        query_nodes,
        source_nodes,
        receiver_index,
        source_index,
        geo_weights,
        receiver_degree,
        receiver_count,
        q_proj,
        k_proj,
        v_proj,
        gate_scale,
        gate_bias,
        layer_norm,
    ):
        logits = (
            q_proj(query_nodes)[receiver_index]
            * k_proj(source_nodes)[source_index]
        ).sum(dim=1) * self.scale / self.tau
        sem_weights = self._segment_softmax(
            logits,
            receiver_index,
            receiver_count,
        )

        kl_edges = sem_weights * (
            torch.log(sem_weights.clamp_min(1e-12))
            - torch.log(geo_weights.clamp_min(1e-12))
        )
        conflict = self._segment_sum(
            kl_edges,
            receiver_index,
            receiver_count,
        )
        valid = receiver_degree > 0
        conflict_std = conflict.new_zeros(conflict.shape)
        if bool(valid.any()):
            valid_conflict = conflict[valid]
            centered = valid_conflict - valid_conflict.mean()
            scale = valid_conflict.std(unbiased=False).clamp_min(1e-8)
            conflict_std[valid] = centered / scale
        gate = torch.zeros(
            (receiver_count, 1),
            dtype=query_nodes.dtype,
            device=query_nodes.device,
        )
        gate[valid] = torch.sigmoid(
            gate_scale * conflict_std[valid] + gate_bias
        ).unsqueeze(1)

        source_values = v_proj(source_nodes)
        edge_values = source_values[source_index]
        geo_message = self._aggregate(
            edge_values,
            geo_weights,
            receiver_index,
            receiver_count,
        )
        sem_message = self._aggregate(
            edge_values,
            sem_weights,
            receiver_index,
            receiver_count,
        )
        ctx = (1.0 - gate) * geo_message + gate * sem_message
        delta = layer_norm(ctx)
        delta = delta.masked_fill(~valid.unsqueeze(1), 0.0)
        entropy_geo = self._edge_entropy(
            geo_weights,
            receiver_index,
            receiver_count,
        )
        entropy_sem = self._edge_entropy(
            sem_weights,
            receiver_index,
            receiver_count,
        )
        stats = {
            "conflict": conflict,
            "gate": gate,
            "geo_entropy": entropy_geo,
            "sem_entropy": entropy_sem,
            "delta": delta,
            "ctx": ctx,
        }
        return delta, stats

    def _one_direction_full(
        self,
        query_nodes,
        source_nodes,
        geo_dense,
        support_geo,
        q_proj,
        k_proj,
        v_proj,
        gate_mlp,
        layer_norm,
    ):
        eps = 1e-12
        logits = q_proj(query_nodes) @ k_proj(source_nodes).t()
        logits = logits * self.scale / self.tau
        support = support_geo
        if self.nonlocal_topk > 0 and source_nodes.shape[0] > 0:
            k = min(self.nonlocal_topk, source_nodes.shape[0])
            nonlocal_index = torch.topk(logits, k=k, dim=1).indices
            nonlocal_support = torch.zeros_like(support_geo)
            nonlocal_support.scatter_(1, nonlocal_index, True)
            support = support_geo | nonlocal_support

        if self.normalization == "sinkhorn":
            sem_dense = self._masked_sinkhorn(logits, support)
        else:
            sem_dense = self._masked_row_softmax(logits, support)

        valid = support_geo.any(dim=1)
        escape = (sem_dense * (~support_geo).to(sem_dense.dtype)).sum(dim=1)
        sem_in_geo = sem_dense * support_geo.to(sem_dense.dtype)
        sem_in_geo = (
            sem_in_geo
            / sem_in_geo.sum(dim=1, keepdim=True).clamp_min(eps)
        )
        kl = (
            sem_in_geo
            * (
                torch.log(sem_in_geo.clamp_min(eps))
                - torch.log(geo_dense.clamp_min(eps))
            )
            * support_geo.to(sem_dense.dtype)
        ).sum(dim=1)
        sem_entropy = -(
            sem_dense * torch.log(sem_dense.clamp_min(eps))
        ).sum(dim=1)
        geo_entropy = -(
            geo_dense * torch.log(geo_dense.clamp_min(eps))
        ).sum(dim=1)

        gate_input = torch.stack(
            [
                self._standardize(kl, valid),
                self._standardize(escape, valid),
                self._standardize(sem_entropy, valid),
            ],
            dim=1,
        )
        gate = torch.zeros(
            (query_nodes.shape[0], 1),
            dtype=query_nodes.dtype,
            device=query_nodes.device,
        )
        gate[valid] = torch.sigmoid(gate_mlp(gate_input[valid]))

        source_values = v_proj(source_nodes)
        geo_message = geo_dense @ source_values
        sem_message = sem_dense @ source_values
        ctx = (1.0 - gate) * geo_message + gate * sem_message
        delta = layer_norm(ctx)
        delta = delta.masked_fill(~valid.unsqueeze(1), 0.0)
        stats = {
            "conflict": kl,
            "gate": gate,
            "escape": escape,
            "geo_entropy": geo_entropy,
            "sem_entropy": sem_entropy,
            "delta": delta,
            "ctx": ctx,
            "sem_dense": sem_dense,
        }
        return delta, stats

    def _forward_simple(self, hsi_nodes, lidar_nodes):
        delta_h, h_stats = self._one_direction(
            hsi_nodes,
            lidar_nodes,
            self.h_index,
            self.l_index,
            self.h_geo,
            self.h_degree,
            self.h_node_count,
            self.q_h,
            self.k_l,
            self.v_l2h,
            self.gate_scale_h,
            self.gate_bias_h,
            self.ln_h,
        )
        delta_l, l_stats = self._one_direction(
            lidar_nodes,
            hsi_nodes,
            self.l_index,
            self.h_index,
            self.l_geo,
            self.l_degree,
            self.l_node_count,
            self.q_l,
            self.k_h,
            self.v_h2l,
            self.gate_scale_l,
            self.gate_bias_l,
            self.ln_l,
        )
        return delta_h, delta_l, h_stats, l_stats

    def _forward_full(self, hsi_nodes, lidar_nodes):
        delta_h, h_stats = self._one_direction_full(
            hsi_nodes,
            lidar_nodes,
            self.h_geo_dense,
            self.h_support_dense,
            self.q_h,
            self.k_l,
            self.v_l2h,
            self.gate_mlp_h,
            self.ln_h,
        )
        delta_l, l_stats = self._one_direction_full(
            lidar_nodes,
            hsi_nodes,
            self.l_geo_dense,
            self.l_support_dense,
            self.q_l,
            self.k_h,
            self.v_h2l,
            self.gate_mlp_l,
            self.ln_l,
        )
        self.last_aux_losses = {}
        if self.training:
            h_valid = self.h_degree > 0
            l_valid = self.l_degree > 0
            h_gate = h_stats["gate"].squeeze(1)
            l_gate = l_stats["gate"].squeeze(1)
            sparse_terms = []
            if bool(h_valid.any()):
                sparse_terms.append(h_gate[h_valid].mean())
            if bool(l_valid.any()):
                sparse_terms.append(l_gate[l_valid].mean())
            if sparse_terms:
                self.last_aux_losses["sparse"] = torch.stack(sparse_terms).sum()

            h_norm = F.normalize(hsi_nodes, dim=1)
            l_as_h = F.normalize(self.align_l_to_h(lidar_nodes), dim=1)
            h_cos = h_norm @ l_as_h.t()
            h_weight = (1.0 - h_gate.detach()).unsqueeze(1) * self.h_geo_dense
            h_align = ((1.0 - h_cos) * h_weight).sum() / (
                h_weight.sum().clamp_min(1e-12)
            )
            l_norm = F.normalize(lidar_nodes, dim=1)
            h_as_l = F.normalize(self.align_h_to_l(hsi_nodes), dim=1)
            l_cos = l_norm @ h_as_l.t()
            l_weight = (1.0 - l_gate.detach()).unsqueeze(1) * self.l_geo_dense
            l_align = ((1.0 - l_cos) * l_weight).sum() / (
                l_weight.sum().clamp_min(1e-12)
            )
            self.last_aux_losses["align"] = 0.5 * (h_align + l_align)

            ent_terms = []
            if bool(h_valid.any()):
                ent_terms.append(h_stats["sem_entropy"][h_valid].mean())
            if bool(l_valid.any()):
                ent_terms.append(l_stats["sem_entropy"][l_valid].mean())
            if ent_terms:
                self.last_aux_losses["entropy"] = (
                    torch.stack(ent_terms).mean()
                )
        return delta_h, delta_l, h_stats, l_stats

    def forward(self, hsi_nodes, lidar_nodes):
        self.last_aux_losses = {}
        if self.variant == "full":
            delta_h, delta_l, h_stats, l_stats = self._forward_full(
                hsi_nodes,
                lidar_nodes,
            )
        else:
            delta_h, delta_l, h_stats, l_stats = self._forward_simple(
                hsi_nodes,
                lidar_nodes,
            )
        updated_hsi = hsi_nodes + self.gamma_h * delta_h
        updated_lidar = lidar_nodes + self.gamma_l * delta_l
        self.last_D_h = h_stats["conflict"].detach()
        self.last_D_l = l_stats["conflict"].detach()
        self.last_gate_h = h_stats["gate"].detach()
        self.last_gate_l = l_stats["gate"].detach()
        self.last_ctx_h = h_stats["ctx"]
        self.last_ctx_l = l_stats["ctx"]

        h_valid = self.h_degree > 0
        l_valid = self.l_degree > 0

        self.last_diagnostics = {
            "mode": f"dual-transport-arbitration-{self.variant}",
            "edge_count": int(self.edge_count),
            "h_gamma": float(self.gamma_h.detach().item()),
            "l_gamma": float(self.gamma_l.detach().item()),
            "h_conflict_mean": self._mean_valid(h_stats["conflict"], h_valid),
            "l_conflict_mean": self._mean_valid(l_stats["conflict"], l_valid),
            "h_gate_mean": self._mean_valid(
                h_stats["gate"].squeeze(1),
                h_valid,
            ),
            "l_gate_mean": self._mean_valid(
                l_stats["gate"].squeeze(1),
                l_valid,
            ),
            "h_geo_entropy": self._mean_valid(h_stats["geo_entropy"], h_valid),
            "l_geo_entropy": self._mean_valid(l_stats["geo_entropy"], l_valid),
            "h_sem_entropy": self._mean_valid(h_stats["sem_entropy"], h_valid),
            "l_sem_entropy": self._mean_valid(l_stats["sem_entropy"], l_valid),
            "h_delta_norm": self._mean_valid(
                h_stats["delta"].norm(dim=1),
                h_valid,
            ),
            "l_delta_norm": self._mean_valid(
                l_stats["delta"].norm(dim=1),
                l_valid,
            ),
        }
        if self.variant == "full":
            self.last_diagnostics.update(
                {
                    "nonlocal_topk": int(self.nonlocal_topk),
                    "normalization": self.normalization,
                    "sinkhorn_iters": int(self.sinkhorn_iters),
                    "h_escape_mean": self._mean_valid(
                        h_stats["escape"],
                        h_valid,
                    ),
                    "l_escape_mean": self._mean_valid(
                        l_stats["escape"],
                        l_valid,
                    ),
                }
            )
        return updated_hsi, updated_lidar

    def diagnostics(self):
        return self.last_diagnostics


class ConflictAwareFiLM(nn.Module):
    """Conflict-aware FiLM generated from DTA arbitrated context.

    Only the first d_s channels are modulated as a shared subspace. The
    remaining channels bypass the module as modality-private features.
    """

    def __init__(self, channels, d_s=64, gamma_init=0.0):
        super().__init__()
        self.channels = int(channels)
        self.d_s = min(int(d_s), self.channels)
        self.private_dim = self.channels - self.d_s

        self.film_h = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, 2 * self.d_s),
        )
        self.film_l = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, 2 * self.d_s),
        )
        nn.init.zeros_(self.film_h[-1].weight)
        nn.init.zeros_(self.film_h[-1].bias)
        nn.init.zeros_(self.film_l[-1].weight)
        nn.init.zeros_(self.film_l[-1].bias)

        self.alpha_h = nn.Parameter(torch.ones(()))
        self.alpha_l = nn.Parameter(torch.ones(()))
        self.gamma_h = nn.Parameter(torch.tensor(float(gamma_init)))
        self.gamma_l = nn.Parameter(torch.tensor(float(gamma_init)))
        self.last_diagnostics = None

    @staticmethod
    def _shrink(conflict, alpha):
        if conflict.numel() <= 1:
            standardized = torch.zeros_like(conflict)
        else:
            standardized = (conflict - conflict.mean()) / conflict.std(
                unbiased=False,
            ).clamp_min(1e-8)
        return torch.exp(
            -alpha.clamp_min(0.0) * standardized.clamp_min(0.0)
        ).unsqueeze(1)

    def _one_side(self, nodes, ctx, conflict, film, alpha, gamma):
        film_params = film(ctx)
        gamma_raw = film_params[:, : self.d_s]
        beta = film_params[:, self.d_s :]
        shrink = self._shrink(conflict, alpha)
        strength = gamma * shrink
        shared = nodes[:, : self.d_s]
        private = nodes[:, self.d_s :]
        modulated_shared = (
            (1.0 + strength * torch.tanh(gamma_raw)) * shared
            + strength * beta
        )
        if self.private_dim > 0:
            return torch.cat([modulated_shared, private], dim=1), shrink
        return modulated_shared, shrink

    def forward(self, hsi_nodes, lidar_nodes, ctx_h, ctx_l, D_h, D_l):
        updated_hsi, shrink_h = self._one_side(
            hsi_nodes,
            ctx_h,
            D_h,
            self.film_h,
            self.alpha_h,
            self.gamma_h,
        )
        updated_lidar, shrink_l = self._one_side(
            lidar_nodes,
            ctx_l,
            D_l,
            self.film_l,
            self.alpha_l,
            self.gamma_l,
        )
        h_delta = updated_hsi[:, : self.d_s] - hsi_nodes[:, : self.d_s]
        l_delta = updated_lidar[:, : self.d_s] - lidar_nodes[:, : self.d_s]
        self.last_diagnostics = {
            "mode": "conflict-aware-film",
            "shared_dim": int(self.d_s),
            "private_dim": int(self.private_dim),
            "gamma_h": float(self.gamma_h.detach().item()),
            "gamma_l": float(self.gamma_l.detach().item()),
            "alpha_h": float(self.alpha_h.detach().item()),
            "alpha_l": float(self.alpha_l.detach().item()),
            "shrink_h_mean": float(shrink_h.detach().mean().item()),
            "shrink_l_mean": float(shrink_l.detach().mean().item()),
            "delta_h_norm": float(
                h_delta.detach().norm(dim=1).mean().item()
            ),
            "delta_l_norm": float(
                l_delta.detach().norm(dim=1).mean().item()
            ),
        }
        return updated_hsi, updated_lidar

    def diagnostics(self):
        return self.last_diagnostics


class AssignmentGraphInteraction(nn.Module):
    """SEGMN-style assignment graph over nonzero HSI/LiDAR overlap pairs.

    Assignment nodes are overlap pairs (h_i, l_j). The assignment graph uses
    private graph topology after GAT2: pair e=(i,j) receives from f=(u,v)
    according to A_H[i,u] * A_L[j,v]. The convolved pair features are pulled
    back to HSI/LiDAR nodes with receiver-normalized overlap coverage.
    """

    def __init__(
        self,
        channels,
        relation_data,
        pair_assignment=None,
        topk=8,
        gamma_init=0.0,
        output="writeback",
    ):
        super().__init__()
        self.channels = int(channels)
        self.topk = int(topk)
        self.output = output
        self.h_node_count = int(relation_data["h_node_count"])
        self.l_node_count = int(relation_data["l_node_count"])
        self.edge_count = int(relation_data["edge_count"])
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
        h_coverage = torch.as_tensor(
            relation_data["h_coverage"],
            dtype=torch.float32,
        )
        l_coverage = torch.as_tensor(
            relation_data["l_coverage"],
            dtype=torch.float32,
        )
        self.register_buffer(
            "h_coverage",
            h_coverage,
            persistent=False,
        )
        self.register_buffer(
            "l_coverage",
            l_coverage,
            persistent=False,
        )
        self.register_buffer(
            "iou",
            torch.as_tensor(relation_data["iou"], dtype=torch.float32),
            persistent=False,
        )
        overlap = torch.as_tensor(relation_data["overlap"], dtype=torch.float32)
        h_area = torch.as_tensor(
            relation_data["h_area"],
            dtype=torch.float32,
        )
        l_area = torch.as_tensor(
            relation_data["l_area"],
            dtype=torch.float32,
        )
        self.register_buffer(
            "overlap_scale",
            overlap / overlap.max().clamp_min(1.0),
            persistent=False,
        )
        self.register_buffer(
            "h_area_scale",
            h_area[self.h_index] / h_area.max().clamp_min(1.0),
            persistent=False,
        )
        self.register_buffer(
            "l_area_scale",
            l_area[self.l_index] / l_area.max().clamp_min(1.0),
            persistent=False,
        )
        self.register_buffer(
            "h_geo",
            self._normalize_edge_weights(
                h_coverage,
                self.h_index,
                self.h_node_count,
            ),
            persistent=False,
        )
        self.register_buffer(
            "l_geo",
            self._normalize_edge_weights(
                l_coverage,
                self.l_index,
                self.l_node_count,
            ),
            persistent=False,
        )
        self.register_buffer(
            "h_degree",
            self._segment_sum(
                torch.ones_like(h_coverage),
                self.h_index,
                self.h_node_count,
            ),
            persistent=False,
        )
        self.register_buffer(
            "l_degree",
            self._segment_sum(
                torch.ones_like(l_coverage),
                self.l_index,
                self.l_node_count,
            ),
            persistent=False,
        )
        if pair_assignment is not None:
            _, pair_projection_assignment = normalized_sparse_assignments(
                pair_assignment
            )
            self.register_buffer(
                "pair_projection_assignment",
                pair_projection_assignment,
                persistent=False,
            )
        else:
            self.pair_projection_assignment = None

        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * channels + 7, channels),
            nn.LayerNorm(channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.pair_value = nn.Linear(channels, channels, bias=False)
        self.pair_gamma = nn.Parameter(torch.ones(()))
        self.pair_norm = nn.LayerNorm(channels)
        self.pair_ffn = nn.Sequential(
            nn.Linear(channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.pair_ffn_norm = nn.LayerNorm(channels)
        self.match_head = nn.Linear(channels, 1)
        self.h_pair_to_node = nn.Linear(channels, channels)
        self.l_pair_to_node = nn.Linear(channels, channels)
        self.h_gate = nn.Sequential(
            nn.Linear(4 * channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.l_gate = nn.Sequential(
            nn.Linear(4 * channels, channels),
            nn.LeakyReLU(),
            nn.Linear(channels, channels),
        )
        self.h_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.l_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.graph_projection = nn.Linear(channels, channels)
        self.last_diagnostics = None
        self.last_pair_context = None

    @staticmethod
    def _segment_sum(values, index, segment_count):
        output = values.new_zeros(segment_count)
        output.index_add_(0, index, values)
        return output

    @classmethod
    def _normalize_edge_weights(cls, values, index, segment_count):
        denom = cls._segment_sum(values, index, segment_count)
        return values / denom[index].clamp_min(1e-12)

    @staticmethod
    def _aggregate(edge_values, edge_weights, receiver_index, receiver_count):
        output = edge_values.new_zeros((receiver_count, edge_values.shape[1]))
        output.index_add_(
            0,
            receiver_index,
            edge_values * edge_weights.unsqueeze(1),
        )
        return output

    def _assignment_adjacency(self, hsi_adjacency, lidar_adjacency):
        hsi_adjacency = hsi_adjacency.detach()
        lidar_adjacency = lidar_adjacency.detach()
        h_prior = hsi_adjacency.index_select(
            0,
            self.h_index,
        ).index_select(1, self.h_index)
        l_prior = lidar_adjacency.index_select(
            0,
            self.l_index,
        ).index_select(1, self.l_index)
        prior = h_prior * l_prior
        pair_count = prior.shape[0]
        diagonal = torch.eye(
            pair_count,
            dtype=torch.bool,
            device=prior.device,
        )
        support = prior > 0
        support = support | diagonal
        prior = prior.masked_fill(~support, 0.0)
        diagonal_values = prior.diagonal().clamp_min(1.0)
        prior = (
            prior * (~diagonal).to(prior.dtype)
            + torch.diag(diagonal_values)
        )
        logits = torch.log(prior.clamp_min(1e-12))
        if self.topk > 0 and pair_count > self.topk:
            k = min(self.topk, pair_count)
            masked_logits = logits.masked_fill(
                ~support,
                torch.finfo(logits.dtype).min,
            )
            _, top_indices = torch.topk(masked_logits, k=k, dim=1)
            top_support = torch.zeros_like(support)
            top_support.scatter_(1, top_indices, True)
            support = support & top_support
            logits = logits.masked_fill(
                ~support,
                torch.finfo(logits.dtype).min,
            )
        else:
            logits = logits.masked_fill(
                ~support,
                torch.finfo(logits.dtype).min,
            )
        adjacency = torch.softmax(logits, dim=1)
        adjacency = adjacency * support.to(adjacency.dtype)
        adjacency = adjacency / adjacency.sum(dim=1, keepdim=True).clamp_min(
            1e-12
        )
        return torch.nan_to_num(adjacency, nan=0.0), support

    def forward(
        self,
        hsi_nodes,
        lidar_nodes,
        hsi_adjacency,
        lidar_adjacency,
    ):
        h_pair = hsi_nodes.index_select(0, self.h_index)
        l_pair = lidar_nodes.index_select(0, self.l_index)
        pair_similarity = F.cosine_similarity(
            h_pair,
            l_pair,
            dim=1,
        ).unsqueeze(1)
        edge_attributes = torch.stack(
            [
                self.overlap_scale,
                self.h_coverage,
                self.l_coverage,
                self.iou,
                self.h_area_scale,
                self.l_area_scale,
            ],
            dim=1,
        )
        pair_nodes = self.pair_encoder(
            torch.cat(
                [
                    h_pair,
                    l_pair,
                    torch.abs(h_pair - l_pair),
                    h_pair * l_pair,
                    edge_attributes,
                    pair_similarity,
                ],
                dim=1,
            )
        )
        assignment_adjacency, assignment_support = self._assignment_adjacency(
            hsi_adjacency,
            lidar_adjacency,
        )
        pair_message = assignment_adjacency @ self.pair_value(pair_nodes)
        pair_context = self.pair_norm(
            pair_nodes + self.pair_gamma * pair_message
        )
        pair_context = self.pair_ffn_norm(
            pair_context + self.pair_ffn(pair_context)
        )
        self.last_pair_context = pair_context
        pair_match = torch.sigmoid(self.match_head(pair_context))
        h_pair_message = self.h_pair_to_node(pair_context) * pair_match
        l_pair_message = self.l_pair_to_node(pair_context) * pair_match
        h_message = self._aggregate(
            h_pair_message,
            self.h_geo,
            self.h_index,
            self.h_node_count,
        )
        l_message = self._aggregate(
            l_pair_message,
            self.l_geo,
            self.l_index,
            self.l_node_count,
        )
        h_valid = self.h_degree > 0
        l_valid = self.l_degree > 0
        h_gate = torch.sigmoid(
            self.h_gate(
                torch.cat(
                    [
                        hsi_nodes,
                        h_message,
                        torch.abs(hsi_nodes - h_message),
                        hsi_nodes * h_message,
                    ],
                    dim=1,
                )
            )
        ) * h_valid.to(hsi_nodes.dtype).unsqueeze(1)
        l_gate = torch.sigmoid(
            self.l_gate(
                torch.cat(
                    [
                        lidar_nodes,
                        l_message,
                        torch.abs(lidar_nodes - l_message),
                        lidar_nodes * l_message,
                    ],
                    dim=1,
                )
            )
        ) * l_valid.to(lidar_nodes.dtype).unsqueeze(1)
        if self.output in ("writeback", "both"):
            updated_hsi = hsi_nodes + self.h_gamma * h_gate * (
                h_message - hsi_nodes
            )
            updated_lidar = lidar_nodes + self.l_gamma * l_gate * (
                l_message - lidar_nodes
            )
        else:
            updated_hsi = hsi_nodes
            updated_lidar = lidar_nodes
        assignment_entropy = -(
            assignment_adjacency
            * torch.log(assignment_adjacency.clamp_min(1e-12))
        ).sum(dim=1)

        def mean_valid(values, valid=None):
            if valid is None:
                return float(values.detach().mean().item())
            if bool(valid.any()):
                return float(values[valid].detach().mean().item())
            return 0.0

        self.last_diagnostics = {
            "mode": "assignment-graph-post-gat2",
            "output": self.output,
            "assignment_node_count": int(self.edge_count),
            "assignment_topk": int(self.topk),
            "assignment_density": float(
                assignment_support.detach().float().mean().item()
            ),
            "assignment_entropy_mean": mean_valid(assignment_entropy),
            "pair_match_mean": mean_valid(pair_match.squeeze(1)),
            "pair_similarity_mean": mean_valid(pair_similarity.squeeze(1)),
            "pair_message_norm": mean_valid(pair_message.norm(dim=1)),
            "h_gamma": float(self.h_gamma.detach().item()),
            "l_gamma": float(self.l_gamma.detach().item()),
            "pair_gamma": float(self.pair_gamma.detach().item()),
            "h_gate_mean": mean_valid(h_gate.mean(dim=1), h_valid),
            "l_gate_mean": mean_valid(l_gate.mean(dim=1), l_valid),
            "h_message_norm": mean_valid(h_message.norm(dim=1), h_valid),
            "l_message_norm": mean_valid(l_message.norm(dim=1), l_valid),
        }
        return updated_hsi, updated_lidar

    def project_pair_nodes(self):
        if self.pair_projection_assignment is None:
            raise RuntimeError(
                "Assignment graph third branch requires pair assignment."
            )
        if self.last_pair_context is None:
            raise RuntimeError(
                "Assignment graph pair context is not available before "
                "forward()."
            )
        pixel_pair_features = torch.sparse.mm(
            self.pair_projection_assignment,
            self.last_pair_context,
        )
        return self.graph_projection(pixel_pair_features)

    def diagnostics(self):
        return self.last_diagnostics


class BCQConsensusQuotientInteraction(nn.Module):
    """Shared-anchor membership consistency over sparse overlap support."""

    def __init__(
        self,
        channels,
        relation_data,
        real_class_count,
        anchor_ratio=4,
        anchor_topk=8,
        anchor_dk=32,
        conflict_alpha=2.0,
        gamma_init=0.0,
    ):
        super().__init__()
        if real_class_count <= 0:
            raise ValueError("BCQ requires a positive real class count.")
        anchor_count = int(anchor_ratio) * int(real_class_count)
        if anchor_count <= 0:
            raise ValueError("BCQ anchor count must be positive.")
        self.channels = int(channels)
        self.real_class_count = int(real_class_count)
        self.anchor_ratio = int(anchor_ratio)
        self.anchor_count = int(anchor_count)
        self.anchor_topk = int(anchor_topk)
        self.conflict_alpha = float(conflict_alpha)
        self.scale = float(anchor_dk) ** -0.5
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
        self.h_node_count = int(relation_data["h_node_count"])
        self.l_node_count = int(relation_data["l_node_count"])
        self.edge_count = int(relation_data["edge_count"])
        self.original_edge_count = int(
            relation_data.get("original_edge_count", self.edge_count)
        )
        self.retained_edge_fraction = float(
            relation_data.get("retained_edge_fraction", 1.0)
        )

        self.anchors = nn.Parameter(torch.empty(self.anchor_count, channels))
        nn.init.xavier_uniform_(self.anchors)
        self.h_node_key = nn.Linear(channels, anchor_dk, bias=False)
        self.l_node_key = nn.Linear(channels, anchor_dk, bias=False)
        self.h_anchor_key = nn.Linear(channels, anchor_dk, bias=False)
        self.l_anchor_key = nn.Linear(channels, anchor_dk, bias=False)
        self.h_anchor_value = nn.Linear(channels, channels, bias=False)
        self.l_anchor_value = nn.Linear(channels, channels, bias=False)
        self.h_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.l_gamma = nn.Parameter(torch.tensor(float(gamma_init)))
        self.last_diagnostics = None
        self.last_h_jsd = None
        self.last_l_jsd = None
        self.last_h_anchor_usage = None
        self.last_l_anchor_usage = None

    @staticmethod
    def _topk_softmax(logits, topk):
        if topk <= 0 or topk >= logits.shape[1]:
            return F.softmax(logits, dim=1)
        values, indices = torch.topk(logits, k=topk, dim=1)
        sparse_logits = torch.full_like(
            logits,
            torch.finfo(logits.dtype).min,
        )
        sparse_logits.scatter_(1, indices, values)
        return F.softmax(sparse_logits, dim=1)

    @staticmethod
    def _normalized_entropy(distribution):
        entropy = -torch.sum(
            distribution
            * torch.log(distribution.clamp_min(1e-12)),
            dim=1,
        )
        denominator = np.log(max(distribution.shape[1], 2))
        return entropy / denominator

    @staticmethod
    def _anchor_usage_entropy(usage):
        entropy = -torch.sum(
            usage * torch.log(usage.clamp_min(1e-12)),
            dim=0,
        )
        denominator = np.log(max(usage.numel(), 2))
        return entropy / denominator

    @staticmethod
    def _jsd(left, right):
        mixture = 0.5 * (left + right)
        kl_left = torch.sum(
            left
            * (
                torch.log(left.clamp_min(1e-12))
                - torch.log(mixture.clamp_min(1e-12))
            ),
            dim=1,
        )
        kl_right = torch.sum(
            right
            * (
                torch.log(right.clamp_min(1e-12))
                - torch.log(mixture.clamp_min(1e-12))
            ),
            dim=1,
        )
        return 0.5 * (kl_left + kl_right)

    def _memberships(self, hsi_nodes, lidar_nodes):
        h_logits = (
            self.h_node_key(hsi_nodes)
            @ self.h_anchor_key(self.anchors).transpose(0, 1)
        ) * self.scale
        l_logits = (
            self.l_node_key(lidar_nodes)
            @ self.l_anchor_key(self.anchors).transpose(0, 1)
        ) * self.scale
        topk = min(self.anchor_topk, self.anchor_count)
        return (
            self._topk_softmax(h_logits, topk),
            self._topk_softmax(l_logits, topk),
        )

    @staticmethod
    def _coverage_transport(
        source_membership,
        receiver_index,
        source_index,
        coverage,
        receiver_count,
        fallback_membership,
    ):
        denom = coverage.new_zeros(receiver_count)
        denom.index_add_(0, receiver_index, coverage)
        weights = coverage / denom.index_select(
            0,
            receiver_index,
        ).clamp_min(1e-8)
        transported = source_membership.new_zeros(
            (receiver_count, source_membership.shape[1])
        )
        transported.index_add_(
            0,
            receiver_index,
            source_membership.index_select(0, source_index)
            * weights.unsqueeze(1),
        )
        missing = denom <= 1e-8
        if bool(missing.any()):
            transported[missing] = fallback_membership[missing]
        return transported

    def forward(self, hsi_nodes, lidar_nodes):
        h_membership, l_membership = self._memberships(
            hsi_nodes,
            lidar_nodes,
        )
        h_expected = self._coverage_transport(
            l_membership,
            self.h_index,
            self.l_index,
            self.h_coverage,
            self.h_node_count,
            h_membership,
        )
        l_expected = self._coverage_transport(
            h_membership,
            self.l_index,
            self.h_index,
            self.l_coverage,
            self.l_node_count,
            l_membership,
        )
        h_jsd = self._jsd(h_membership, h_expected)
        l_jsd = self._jsd(l_membership, l_expected)
        h_gate = torch.exp(-self.conflict_alpha * h_jsd).unsqueeze(1)
        l_gate = torch.exp(-self.conflict_alpha * l_jsd).unsqueeze(1)
        h_anchor_features = self.h_anchor_value(self.anchors)
        l_anchor_features = self.l_anchor_value(self.anchors)
        h_target = h_membership @ h_anchor_features
        l_target = l_membership @ l_anchor_features
        updated_hsi = (
            hsi_nodes
            + self.h_gamma * h_gate * (h_target - hsi_nodes)
        )
        updated_lidar = (
            lidar_nodes
            + self.l_gamma * l_gate * (l_target - lidar_nodes)
        )
        h_entropy = self._normalized_entropy(h_membership)
        l_entropy = self._normalized_entropy(l_membership)
        h_anchor_usage = h_membership.detach().mean(dim=0)
        l_anchor_usage = l_membership.detach().mean(dim=0)
        self.last_h_jsd = h_jsd.detach()
        self.last_l_jsd = l_jsd.detach()
        self.last_h_anchor_usage = h_anchor_usage
        self.last_l_anchor_usage = l_anchor_usage
        self.last_diagnostics = {
            "mode": "bcq",
            "interaction": "post-gat2",
            "anchor_count": int(self.anchor_count),
            "anchor_ratio": int(self.anchor_ratio),
            "anchor_topk": int(self.anchor_topk),
            "conflict_alpha": float(self.conflict_alpha),
            "edge_count": int(self.edge_count),
            "original_edge_count": int(self.original_edge_count),
            "retained_edge_fraction": float(self.retained_edge_fraction),
            "h_gamma": float(self.h_gamma.detach().item()),
            "l_gamma": float(self.l_gamma.detach().item()),
            "h_jsd_mean": float(h_jsd.detach().mean().item()),
            "l_jsd_mean": float(l_jsd.detach().mean().item()),
            "h_jsd_max": float(h_jsd.detach().max().item()),
            "l_jsd_max": float(l_jsd.detach().max().item()),
            "h_gate_mean": float(h_gate.detach().mean().item()),
            "l_gate_mean": float(l_gate.detach().mean().item()),
            "h_entropy_mean": float(h_entropy.detach().mean().item()),
            "l_entropy_mean": float(l_entropy.detach().mean().item()),
            "h_anchor_usage_entropy": float(
                self._anchor_usage_entropy(h_anchor_usage).item()
            ),
            "l_anchor_usage_entropy": float(
                self._anchor_usage_entropy(l_anchor_usage).item()
            ),
            "h_anchor_usage_max": float(h_anchor_usage.max().item()),
            "l_anchor_usage_max": float(l_anchor_usage.max().item()),
            "h_target_delta_norm": float(
                (h_target - hsi_nodes).detach().norm(dim=1).mean().item()
            ),
            "l_target_delta_norm": float(
                (l_target - lidar_nodes)
                .detach()
                .norm(dim=1)
                .mean()
                .item()
            ),
        }
        return updated_hsi, updated_lidar

    def diagnostics(self):
        return self.last_diagnostics


class ConsensusTokenFusion(nn.Module):
    """Class-grouped consensus tokens with auxiliary supervision."""

    def __init__(
        self,
        channels,
        relation_data,
        real_class_count,
        tokens_per_class=4,
        num_heads=4,
        token_topk=0,
        tau=1.0,
        fusion_gate_init=0.0,
        eps=1e-8,
    ):
        super().__init__()
        if real_class_count <= 0:
            raise ValueError(
                "Consensus token fusion requires a positive class count."
            )
        token_count = int(tokens_per_class) * int(real_class_count)
        if token_count <= 0:
            raise ValueError("Consensus token count must be positive.")
        self.channels = int(channels)
        self.real_class_count = int(real_class_count)
        self.tokens_per_class = int(tokens_per_class)
        self.token_count = int(token_count)
        self.token_topk = int(token_topk)
        self.tau = float(tau)
        self.eps = float(eps)
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
        self.h_node_count = int(relation_data["h_node_count"])
        self.l_node_count = int(relation_data["l_node_count"])
        self.edge_count = int(relation_data["edge_count"])
        self.original_edge_count = int(
            relation_data.get("original_edge_count", self.edge_count)
        )
        self.retained_edge_fraction = float(
            relation_data.get("retained_edge_fraction", 1.0)
        )

        self.tokens = nn.Parameter(
            torch.randn(self.token_count, channels) * 0.02
        )
        self.h_attn = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.l_attn = nn.MultiheadAttention(
            channels,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.view_logit = nn.Parameter(torch.zeros(2))
        self.token_norm = nn.LayerNorm(channels)
        self.token_ffn = nn.Sequential(
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Linear(channels * 2, channels),
        )
        self.ffn_norm = nn.LayerNorm(channels)
        self.h_node_q = nn.Linear(channels, channels, bias=False)
        self.l_node_q = nn.Linear(channels, channels, bias=False)
        self.h_token_k = nn.Linear(channels, channels, bias=False)
        self.l_token_k = nn.Linear(channels, channels, bias=False)
        self.fusion_gate = nn.Parameter(
            torch.tensor(float(fusion_gate_init))
        )
        self.last_outputs = None
        self.last_diagnostics = None

    @staticmethod
    def _topk_softmax(logits, topk):
        if topk <= 0 or topk >= logits.shape[1]:
            return F.softmax(logits, dim=1)
        values, indices = torch.topk(logits, k=topk, dim=1)
        sparse_logits = torch.full_like(
            logits,
            torch.finfo(logits.dtype).min,
        )
        sparse_logits.scatter_(1, indices, values)
        return F.softmax(sparse_logits, dim=1)

    @staticmethod
    def _entropy(distribution):
        return -torch.sum(
            distribution
            * torch.log(distribution.clamp_min(1e-12)),
            dim=1,
        )

    @staticmethod
    def _kl(left, right):
        return torch.sum(
            left
            * (
                torch.log(left.clamp_min(1e-12))
                - torch.log(right.clamp_min(1e-12))
            ),
            dim=1,
        )

    def _update_tokens(self, hsi_nodes, lidar_nodes):
        tokens = self.tokens.unsqueeze(0)
        h_out, _ = self.h_attn(
            tokens,
            hsi_nodes.unsqueeze(0),
            hsi_nodes.unsqueeze(0),
            need_weights=False,
        )
        l_out, _ = self.l_attn(
            tokens,
            lidar_nodes.unsqueeze(0),
            lidar_nodes.unsqueeze(0),
            need_weights=False,
        )
        view_weights = F.softmax(self.view_logit, dim=0)
        tokens = self.token_norm(
            tokens
            + view_weights[0] * h_out
            + view_weights[1] * l_out
        )
        tokens = self.ffn_norm(tokens + self.token_ffn(tokens))
        return tokens.squeeze(0), view_weights

    def _membership(self, nodes, node_q, token_k, tokens):
        q = node_q(nodes)
        k = token_k(tokens)
        logits = q @ k.transpose(0, 1)
        logits = logits / ((q.shape[-1] ** 0.5) * self.tau)
        return self._topk_softmax(logits, self.token_topk)

    @staticmethod
    def _coverage_transport(
        source_membership,
        receiver_index,
        source_index,
        coverage,
        receiver_count,
        fallback_membership,
    ):
        mass = coverage.new_zeros(receiver_count)
        mass.index_add_(0, receiver_index, coverage)
        weights = coverage / mass.index_select(
            0,
            receiver_index,
        ).clamp_min(1e-8)
        transported = source_membership.new_zeros(
            (receiver_count, source_membership.shape[1])
        )
        transported.index_add_(
            0,
            receiver_index,
            source_membership.index_select(0, source_index)
            * weights.unsqueeze(1),
        )
        missing = mass <= 1e-8
        if bool(missing.any()):
            transported[missing] = fallback_membership[missing]
        return transported, mass

    def _agreement_loss(self, h_membership, l_membership):
        h_expected, h_mass = self._coverage_transport(
            l_membership,
            self.h_index,
            self.l_index,
            self.h_coverage,
            self.h_node_count,
            h_membership,
        )
        l_expected, l_mass = self._coverage_transport(
            h_membership,
            self.l_index,
            self.h_index,
            self.l_coverage,
            self.l_node_count,
            l_membership,
        )
        h_mix = 0.5 * (h_membership + h_expected)
        l_mix = 0.5 * (l_membership + l_expected)
        h_jsd = 0.5 * (
            self._kl(h_membership, h_mix)
            + self._kl(h_expected, h_mix)
        )
        l_jsd = 0.5 * (
            self._kl(l_membership, l_mix)
            + self._kl(l_expected, l_mix)
        )
        h_weight = h_mass / h_mass.sum().clamp_min(1e-8)
        l_weight = l_mass / l_mass.sum().clamp_min(1e-8)
        loss = 0.5 * (
            (h_weight * h_jsd).sum()
            + (l_weight * l_jsd).sum()
        )
        return loss, h_jsd, l_jsd

    def _usage_loss(self, h_membership, l_membership):
        usage = 0.5 * (
            h_membership.mean(dim=0)
            + l_membership.mean(dim=0)
        )
        entropy = -torch.sum(
            usage * torch.log(usage.clamp_min(self.eps)),
            dim=0,
        )
        return (
            torch.log(
                torch.tensor(
                    float(self.token_count),
                    device=usage.device,
                    dtype=usage.dtype,
                )
            )
            - entropy
        )

    def forward(self, hsi_nodes, lidar_nodes):
        tokens, view_weights = self._update_tokens(hsi_nodes, lidar_nodes)
        h_membership = self._membership(
            hsi_nodes,
            self.h_node_q,
            self.h_token_k,
            tokens,
        )
        l_membership = self._membership(
            lidar_nodes,
            self.l_node_q,
            self.l_token_k,
            tokens,
        )
        h_probability = h_membership.view(
            -1,
            self.real_class_count,
            self.tokens_per_class,
        ).sum(dim=2)
        l_probability = l_membership.view(
            -1,
            self.real_class_count,
            self.tokens_per_class,
        ).sum(dim=2)
        agreement_loss, h_jsd, l_jsd = self._agreement_loss(
            h_membership,
            l_membership,
        )
        usage_loss = self._usage_loss(h_membership, l_membership)
        h_entropy = self._entropy(h_membership) / np.log(
            max(self.token_count, 2)
        )
        l_entropy = self._entropy(l_membership) / np.log(
            max(self.token_count, 2)
        )
        usage = 0.5 * (
            h_membership.detach().mean(dim=0)
            + l_membership.detach().mean(dim=0)
        )
        self.last_outputs = {
            "S_h": h_membership,
            "S_l": l_membership,
            "p_h": h_probability,
            "p_l": l_probability,
            "tokens": tokens,
            "view_weights": view_weights,
            "agreement_loss": agreement_loss,
            "usage_loss": usage_loss,
            "h_jsd": h_jsd,
            "l_jsd": l_jsd,
        }
        self.last_diagnostics = {
            "mode": "consensus-token",
            "interaction": "post-gat2",
            "token_count": int(self.token_count),
            "tokens_per_class": int(self.tokens_per_class),
            "token_topk": int(self.token_topk),
            "edge_count": int(self.edge_count),
            "original_edge_count": int(self.original_edge_count),
            "retained_edge_fraction": float(self.retained_edge_fraction),
            "fusion_gate": float(self.fusion_gate.detach().item()),
            "view_weights": view_weights.detach().cpu().tolist(),
            "h_entropy_mean": float(h_entropy.detach().mean().item()),
            "l_entropy_mean": float(l_entropy.detach().mean().item()),
            "h_jsd_mean": float(h_jsd.detach().mean().item()),
            "l_jsd_mean": float(l_jsd.detach().mean().item()),
            "h_jsd_max": float(h_jsd.detach().max().item()),
            "l_jsd_max": float(l_jsd.detach().max().item()),
            "usage_entropy": float(
                (
                    -torch.sum(
                        usage * torch.log(usage.clamp_min(1e-12))
                    )
                    / np.log(max(self.token_count, 2))
                ).item()
            ),
            "usage_max": float(usage.max().item()),
        }
        return self.last_outputs

    def project_probabilities(self, h_projection, l_projection, outputs):
        h_pixel_probability = torch.sparse.mm(
            h_projection,
            outputs["p_h"],
        )
        l_pixel_probability = torch.sparse.mm(
            l_projection,
            outputs["p_l"],
        )
        return (
            h_pixel_probability.clamp_min(self.eps),
            l_pixel_probability.clamp_min(self.eps),
        )

    def fuse_logits(
        self,
        main_logits,
        h_pixel_probability,
        l_pixel_probability,
        view_weights,
    ):
        fused_probability = (
            view_weights[0] * h_pixel_probability
            + view_weights[1] * l_pixel_probability
        ).clamp_min(self.eps)
        output = main_logits.clone()
        output[:, : self.real_class_count] = (
            output[:, : self.real_class_count]
            + self.fusion_gate * torch.log(fused_probability)
        )
        return output

    def consensus_ce(self, pixel_probability, pixel_index, labels):
        return F.nll_loss(
            torch.log(
                pixel_probability.index_select(0, pixel_index).clamp_min(
                    self.eps
                )
            ),
            labels,
        )

    def diagnostics(self):
        return self.last_diagnostics


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
    ):
        graph_features = self.apply_gat2_nodes(
            first_graph_features,
            adjacency=adjacency,
            rebuild_graph=rebuild_graph,
        )
        return self.project_nodes(graph_features)

    def apply_gat2_nodes(
        self,
        first_graph_features,
        adjacency=None,
        rebuild_graph=False,
        return_intra=False,
    ):
        if rebuild_graph:
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
        dual_transport_arbitration="none",
        dta_d_k=64,
        dta_tau=0.1,
        dta_gamma_init=0.0,
        dta_variant="simple",
        dta_nonlocal_topk=10,
        dta_normalization="sinkhorn",
        dta_sinkhorn_iters=5,
        dta_film=False,
        dta_film_d_s=64,
        dta_film_gamma_init=0.0,
        assignment_graph="none",
        assignment_graph_topk=8,
        assignment_graph_gamma_init=0.0,
        assignment_graph_output="writeback",
        assignment_graph_weight=0.1,
        cross_overlap_relation="none",
        cross_overlap_stage="inter-gat",
        cross_overlap_message="qk-prior",
        cross_overlap_fusion="dual-channel",
        cross_overlap_prior_weight=1.0,
        cross_overlap_gamma_init=0.1,
        cross_overlap_second_gamma_init=0.0,
        cross_overlap_fragmentation_alpha=0.0,
        cross_overlap_data=None,
        overlap_distill_enabled=False,
        evidence_fusion="none",
        bcq_interaction="none",
        bcq_class_count=None,
        bcq_anchor_ratio=4,
        bcq_anchor_topk=8,
        bcq_anchor_dk=32,
        bcq_conflict_alpha=2.0,
        bcq_gamma_init=0.0,
        consensus_token_fusion="none",
        consensus_token_class_count=None,
        consensus_token_ratio=4,
        consensus_token_heads=4,
        consensus_token_topk=0,
        consensus_token_tau=1.0,
        consensus_token_fusion_gate_init=0.0,
        fdsm_scope="none",
        lidar_modulation="none",
        cnn_branch="original",
        cnn_layout="joint",
        cnn_share_weights=False,
        lidar_rag_adjacency=None,
        lidar_geometry_descriptors=None,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.graph_modality_lambda = graph_modality_lambda
        self.fusion_lambda = fusion_lambda
        self.cross_overlap_relation = cross_overlap_relation
        self.cross_overlap_stage = cross_overlap_stage
        self.dual_transport_arbitration = dual_transport_arbitration
        self.assignment_graph = assignment_graph
        self.assignment_graph_output = assignment_graph_output
        self.assignment_graph_weight = float(assignment_graph_weight)
        self.evidence_fusion = evidence_fusion
        self.bcq_interaction = bcq_interaction
        self.consensus_token_fusion = consensus_token_fusion
        self.last_dta_diagnostics = None
        self.last_dta_film_diagnostics = None
        self.last_assignment_graph_diagnostics = None
        self.last_cross_overlap_diagnostics = None
        self.last_bcq_diagnostics = None
        self.last_consensus_token_diagnostics = None
        self.last_consensus_token_outputs = None
        self.last_evidence_diagnostics = None
        self.cnn_branch_mode = cnn_branch
        self.cnn_layout = cnn_layout
        self.cnn_share_weights = cnn_share_weights

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
        self.overlap_distill_enabled = overlap_distill_enabled
        if evidence_fusion == "dirichlet":
            self.hsi_evidence_head = nn.Linear(hidden_dim, class_count)
            self.lidar_evidence_head = nn.Linear(hidden_dim, class_count)
        else:
            self.hsi_evidence_head = None
            self.lidar_evidence_head = None
        self.last_hsi_final_nodes = None
        self.last_lidar_final_nodes = None
        self.last_hsi_node_logits = None
        self.last_lidar_node_logits = None
        self.last_hsi_node_probabilities = None
        self.last_lidar_node_probabilities = None
        self.last_fused_alpha = None

        if evidence_fusion == "dirichlet" and cross_overlap_data is not None:
            self.register_buffer(
                "evidence_h_index",
                torch.as_tensor(
                    cross_overlap_data["h_index"],
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "evidence_l_index",
                torch.as_tensor(
                    cross_overlap_data["l_index"],
                    dtype=torch.long,
                ),
                persistent=False,
            )
            self.register_buffer(
                "evidence_iou",
                torch.as_tensor(
                    cross_overlap_data["iou"],
                    dtype=torch.float32,
                ),
                persistent=False,
            )
        else:
            self.evidence_h_index = None
            self.evidence_l_index = None
            self.evidence_iou = None

        if dual_transport_arbitration != "none":
            if cross_overlap_data is None:
                cross_overlap_data = build_sparse_overlap_relation_data(
                    hsi_assignment,
                    lidar_assignment,
                )
            self.dta_branch = DualTransportArbitration(
                hidden_dim,
                cross_overlap_data,
                attention_d_k=dta_d_k,
                tau=dta_tau,
                gamma_init=dta_gamma_init,
                variant=dta_variant,
                nonlocal_topk=dta_nonlocal_topk,
                normalization=dta_normalization,
                sinkhorn_iters=dta_sinkhorn_iters,
            )
        else:
            self.dta_branch = None
        if self.dta_branch is not None and dta_film:
            self.dta_film_branch = ConflictAwareFiLM(
                hidden_dim,
                d_s=dta_film_d_s,
                gamma_init=dta_film_gamma_init,
            )
        else:
            self.dta_film_branch = None

        if assignment_graph != "none":
            if cross_overlap_data is None:
                cross_overlap_data = build_sparse_overlap_relation_data(
                    hsi_assignment,
                    lidar_assignment,
                )
            pair_assignment = build_overlap_pair_assignment(
                hsi_assignment,
                lidar_assignment,
                cross_overlap_data,
            )
            self.assignment_graph_branch = AssignmentGraphInteraction(
                hidden_dim,
                cross_overlap_data,
                pair_assignment=pair_assignment,
                topk=assignment_graph_topk,
                gamma_init=assignment_graph_gamma_init,
                output=assignment_graph_output,
            )
        else:
            self.assignment_graph_branch = None

        if cross_overlap_relation == "sparse":
            if cross_overlap_data is None:
                cross_overlap_data = build_sparse_overlap_relation_data(
                    hsi_assignment,
                    lidar_assignment,
                )
            self.cross_overlap_branch = SparseOverlapCrossModalRelation(
                hidden_dim,
                cross_overlap_data,
                attention_d_k=dynamic_d_k,
                message=cross_overlap_message,
                fusion=cross_overlap_fusion,
                prior_weight=cross_overlap_prior_weight,
                gamma_init=cross_overlap_gamma_init,
                second_gamma_init=cross_overlap_second_gamma_init,
                fragmentation_alpha=cross_overlap_fragmentation_alpha,
            )
        else:
            self.cross_overlap_branch = None

        if bcq_interaction != "none":
            if cross_overlap_data is None:
                cross_overlap_data = build_sparse_overlap_relation_data(
                    hsi_assignment,
                    lidar_assignment,
                )
            bcq_rng_state = torch.get_rng_state()
            self.bcq_branch = BCQConsensusQuotientInteraction(
                hidden_dim,
                cross_overlap_data,
                real_class_count=(
                    int(bcq_class_count)
                    if bcq_class_count is not None
                    else int(class_count)
                ),
                anchor_ratio=bcq_anchor_ratio,
                anchor_topk=bcq_anchor_topk,
                anchor_dk=bcq_anchor_dk,
                conflict_alpha=bcq_conflict_alpha,
                gamma_init=bcq_gamma_init,
            )
            torch.set_rng_state(bcq_rng_state)
        else:
            self.bcq_branch = None

        if consensus_token_fusion != "none":
            if cross_overlap_data is None:
                cross_overlap_data = build_sparse_overlap_relation_data(
                    hsi_assignment,
                    lidar_assignment,
                )
            token_rng_state = torch.get_rng_state()
            self.consensus_token_branch = ConsensusTokenFusion(
                hidden_dim,
                cross_overlap_data,
                real_class_count=(
                    int(consensus_token_class_count)
                    if consensus_token_class_count is not None
                    else int(class_count)
                ),
                tokens_per_class=consensus_token_ratio,
                num_heads=consensus_token_heads,
                token_topk=consensus_token_topk,
                tau=consensus_token_tau,
                fusion_gate_init=consensus_token_fusion_gate_init,
            )
            torch.set_rng_state(token_rng_state)
        else:
            self.consensus_token_branch = None

    def _cache_branch_node_logits(self, hsi_nodes, lidar_nodes):
        should_cache = (
            self.overlap_distill_enabled
            or self.evidence_fusion == "dirichlet"
        )
        if not should_cache:
            self.last_hsi_final_nodes = None
            self.last_lidar_final_nodes = None
            self.last_hsi_node_logits = None
            self.last_lidar_node_logits = None
            self.last_hsi_node_probabilities = None
            self.last_lidar_node_probabilities = None
            return
        self.last_hsi_final_nodes = hsi_nodes
        self.last_lidar_final_nodes = lidar_nodes
        if self.evidence_fusion == "dirichlet":
            h_alpha = self._node_evidence_alpha(
                self.hsi_evidence_head,
                hsi_nodes,
            )
            l_alpha = self._node_evidence_alpha(
                self.lidar_evidence_head,
                lidar_nodes,
            )
            self.last_hsi_node_probabilities = (
                h_alpha / h_alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
            )
            self.last_lidar_node_probabilities = (
                l_alpha / l_alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
            )
            self.last_hsi_node_logits = None
            self.last_lidar_node_logits = None
        else:
            self.last_hsi_node_probabilities = None
            self.last_lidar_node_probabilities = None
            self.last_hsi_node_logits = self.classifier(hsi_nodes)
            self.last_lidar_node_logits = self.classifier(lidar_nodes)

    @staticmethod
    def _node_evidence_alpha(head, nodes):
        return F.softplus(head(nodes)) + 1.0

    def _project_branch_alpha_to_pixels(self, hsi_alpha, lidar_alpha):
        hsi_pixel_alpha = torch.sparse.mm(
            self.hsi_graph.projection_assignment,
            hsi_alpha,
        )
        lidar_pixel_alpha = torch.sparse.mm(
            self.lidar_graph.projection_assignment,
            lidar_alpha,
        )
        return hsi_pixel_alpha, lidar_pixel_alpha

    def _overlap_evidence_conflict_diagnostics(
        self,
        hsi_node_alpha,
        lidar_node_alpha,
    ):
        if self.evidence_h_index is None or self.evidence_iou is None:
            return {}
        h_evidence = (hsi_node_alpha - 1.0).clamp_min(0.0)
        l_evidence = (lidar_node_alpha - 1.0).clamp_min(0.0)
        h_strength = hsi_node_alpha.sum(dim=1, keepdim=True)
        l_strength = lidar_node_alpha.sum(dim=1, keepdim=True)
        h_belief = h_evidence / h_strength.clamp_min(1e-8)
        l_belief = l_evidence / l_strength.clamp_min(1e-8)
        edge_h_belief = h_belief.index_select(0, self.evidence_h_index)
        edge_l_belief = l_belief.index_select(0, self.evidence_l_index)
        agreement = (edge_h_belief * edge_l_belief).sum(
            dim=1,
            keepdim=False,
        )
        conflict = (
            edge_h_belief.sum(dim=1)
            * edge_l_belief.sum(dim=1)
            - agreement
        ).clamp_min(0.0)
        weight = self.evidence_iou
        weighted_conflict = (
            (weight * conflict).sum() / weight.sum().clamp_min(1e-8)
        )
        return {
            "overlap_conflict_mean": float(
                conflict.detach().mean().item()
            ),
            "overlap_conflict_iou_weighted": float(
                weighted_conflict.detach().item()
            ),
        }

    def _forward_evidence_fusion_from_cached_nodes(self):
        if (
            self.last_hsi_final_nodes is None
            or self.last_lidar_final_nodes is None
        ):
            raise RuntimeError(
                "Evidence fusion requires cached separate HSI/LiDAR "
                "graph nodes."
            )
        hsi_node_alpha = self._node_evidence_alpha(
            self.hsi_evidence_head,
            self.last_hsi_final_nodes,
        )
        lidar_node_alpha = self._node_evidence_alpha(
            self.lidar_evidence_head,
            self.last_lidar_final_nodes,
        )
        self.last_hsi_node_probabilities = (
            hsi_node_alpha
            / hsi_node_alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        self.last_lidar_node_probabilities = (
            lidar_node_alpha
            / lidar_node_alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        hsi_pixel_alpha, lidar_pixel_alpha = (
            self._project_branch_alpha_to_pixels(
                hsi_node_alpha,
                lidar_node_alpha,
            )
        )
        fused_alpha, diagnostics = dempster_combine_dirichlet(
            hsi_pixel_alpha,
            lidar_pixel_alpha,
        )
        diagnostics = {
            **diagnostics,
            **self._overlap_evidence_conflict_diagnostics(
                hsi_node_alpha,
                lidar_node_alpha,
            ),
        }
        self.last_fused_alpha = fused_alpha
        self.last_evidence_diagnostics = diagnostics
        fused_probability = (
            fused_alpha / fused_alpha.sum(dim=1, keepdim=True).clamp_min(1e-8)
        )
        return torch.log(fused_probability.clamp_min(1e-12))



    def _apply_private_gat1(self, hsi_nodes, lidar_nodes):
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
        if self.dta_branch is not None:
            hsi_final_nodes, lidar_final_nodes = self.dta_branch(
                hsi_final_nodes,
                lidar_final_nodes,
            )
            self.last_dta_diagnostics = self.dta_branch.diagnostics()
            if self.dta_film_branch is not None:
                hsi_final_nodes, lidar_final_nodes = self.dta_film_branch(
                    hsi_final_nodes,
                    lidar_final_nodes,
                    self.dta_branch.last_ctx_h,
                    self.dta_branch.last_ctx_l,
                    self.dta_branch.last_D_h,
                    self.dta_branch.last_D_l,
                )
                self.last_dta_film_diagnostics = (
                    self.dta_film_branch.diagnostics()
                )
        if self.assignment_graph_branch is not None:
            hsi_final_nodes, lidar_final_nodes = self.assignment_graph_branch(
                hsi_final_nodes,
                lidar_final_nodes,
                hsi_adjacency,
                lidar_adjacency,
            )
            self.last_assignment_graph_diagnostics = (
                self.assignment_graph_branch.diagnostics()
            )
        return hsi_final_nodes, lidar_final_nodes


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
        self._cache_branch_node_logits(hsi_final_nodes, lidar_final_nodes)
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

    def _forward_bcq_interaction(self, hsi, lidar):
        if self.bcq_interaction != "post-gat2":
            raise ValueError(
                f"Unsupported BCQ interaction slot: {self.bcq_interaction}"
            )
        (
            hsi_final_nodes,
            lidar_final_nodes,
        ) = self._encode_private_graph_nodes(
            hsi,
            lidar,
        )
        hsi_final_nodes, lidar_final_nodes = self.bcq_branch(
            hsi_final_nodes,
            lidar_final_nodes,
        )
        self._cache_branch_node_logits(hsi_final_nodes, lidar_final_nodes)
        hsi_graph_features = self.hsi_graph.project_nodes(hsi_final_nodes)
        lidar_graph_features = self.lidar_graph.project_nodes(
            lidar_final_nodes
        )
        self.last_bcq_diagnostics = self.bcq_branch.diagnostics()
        return (
            self.graph_modality_lambda * hsi_graph_features
            + (1.0 - self.graph_modality_lambda)
            * lidar_graph_features
        )

    def _forward_consensus_token_fusion(self, hsi, lidar):
        if self.consensus_token_fusion != "post-gat2":
            raise ValueError(
                "Unsupported consensus token fusion slot: "
                f"{self.consensus_token_fusion}"
            )
        (
            hsi_final_nodes,
            lidar_final_nodes,
        ) = self._encode_private_graph_nodes(
            hsi,
            lidar,
        )
        self._cache_branch_node_logits(hsi_final_nodes, lidar_final_nodes)
        hsi_graph_features = self.hsi_graph.project_nodes(hsi_final_nodes)
        lidar_graph_features = self.lidar_graph.project_nodes(
            lidar_final_nodes
        )
        token_outputs = self.consensus_token_branch(
            hsi_final_nodes,
            lidar_final_nodes,
        )
        h_pixel_probability, l_pixel_probability = (
            self.consensus_token_branch.project_probabilities(
                self.hsi_graph.projection_assignment,
                self.lidar_graph.projection_assignment,
                token_outputs,
            )
        )
        token_outputs = {
            **token_outputs,
            "p_h_pix": h_pixel_probability,
            "p_l_pix": l_pixel_probability,
        }
        self.last_consensus_token_outputs = token_outputs
        self.last_consensus_token_diagnostics = (
            self.consensus_token_branch.diagnostics()
        )
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
        self.last_dta_diagnostics = None
        self.last_dta_film_diagnostics = None
        self.last_assignment_graph_diagnostics = None
        self.last_cross_overlap_diagnostics = None
        self.last_bcq_diagnostics = None
        self.last_consensus_token_diagnostics = None
        self.last_consensus_token_outputs = None
        self.last_evidence_diagnostics = None
        self.last_hsi_final_nodes = None
        self.last_lidar_final_nodes = None
        self.last_hsi_node_logits = None
        self.last_lidar_node_logits = None
        self.last_hsi_node_probabilities = None
        self.last_lidar_node_probabilities = None
        self.last_fused_alpha = None
        if self.consensus_token_branch is not None:
            graph_features = self._forward_consensus_token_fusion(hsi, lidar)
        elif self.bcq_branch is not None:
            graph_features = self._forward_bcq_interaction(hsi, lidar)
        elif self.cross_overlap_branch is not None:
            graph_features = self._forward_cross_overlap_relation(hsi, lidar)
        else:
            if (
                self.dta_branch is not None
                or self.assignment_graph_branch is not None
                or self.overlap_distill_enabled
                or self.evidence_fusion == "dirichlet"
            ):
                (
                    hsi_final_nodes,
                    lidar_final_nodes,
                ) = self._encode_private_graph_nodes(hsi, lidar)
                self._cache_branch_node_logits(
                    hsi_final_nodes,
                    lidar_final_nodes,
                )
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
            if (
                self.assignment_graph_branch is not None
                and self.assignment_graph_output in ("third-branch", "both")
            ):
                assignment_graph_features = (
                    self.assignment_graph_branch.project_pair_nodes()
                )
                graph_features = (
                    (1.0 - self.assignment_graph_weight) * graph_features
                    + self.assignment_graph_weight
                    * assignment_graph_features
                )

        if self.evidence_fusion == "dirichlet":
            return self._forward_evidence_fusion_from_cached_nodes()

        cnn_features = self._forward_cnn_features(
            hsi,
            lidar,
            joint_input,
        )
        fused_features = (
            self.fusion_lambda * graph_features
            + (1.0 - self.fusion_lambda) * cnn_features
        )
        main_logits = self.classifier(fused_features)
        if self.consensus_token_branch is not None:
            token_outputs = self.last_consensus_token_outputs
            if token_outputs is None:
                raise RuntimeError(
                    "Consensus token fusion requires cached token outputs."
                )
            return self.consensus_token_branch.fuse_logits(
                main_logits,
                token_outputs["p_h_pix"],
                token_outputs["p_l_pix"],
                token_outputs["view_weights"],
            )
        return main_logits



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
    cell_data = None
    bridge_data = None
    cross_overlap_data = None
    needs_cross_overlap_data = (
        args.dual_transport_arbitration != "none"
        or args.assignment_graph != "none"
        or args.cross_overlap_relation == "sparse"
        or args.overlap_distill_weight > 0
        or args.evidence_fusion == "dirichlet"
        or args.bcq_interaction != "none"
        or args.consensus_token_fusion != "none"
    )
    if needs_cross_overlap_data:
        cross_overlap_data = build_sparse_overlap_relation_data(
            hsi_assignment,
            lidar_assignment,
            hsi_features=hsi,
            lidar_image=lidar,
            edge_attrs=args.cross_overlap_edge_attrs,
            min_coverage=args.cross_overlap_min_coverage,
            iou_topk=args.cross_overlap_iou_topk,
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
        cross_overlap_data,
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
    cross_overlap_data,
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
            dual_transport_arbitration=args.dual_transport_arbitration,
            dta_d_k=args.dta_dk,
            dta_tau=args.dta_tau,
            dta_gamma_init=args.dta_gamma_init,
            dta_variant=args.dta_variant,
            dta_nonlocal_topk=args.dta_nonlocal_topk,
            dta_normalization=args.dta_normalization,
            dta_sinkhorn_iters=args.dta_sinkhorn_iters,
            dta_film=args.dta_film,
            dta_film_d_s=args.dta_film_ds,
            dta_film_gamma_init=args.dta_film_gamma_init,
            assignment_graph=args.assignment_graph,
            assignment_graph_topk=args.assignment_graph_topk,
            assignment_graph_gamma_init=args.assignment_graph_gamma_init,
            assignment_graph_output=args.assignment_graph_output,
            assignment_graph_weight=args.assignment_graph_weight,
            cross_overlap_relation=args.cross_overlap_relation,
            cross_overlap_stage=args.cross_overlap_stage,
            cross_overlap_message=args.cross_overlap_message,
            cross_overlap_fusion=args.cross_overlap_fusion,
            cross_overlap_prior_weight=args.cross_overlap_prior_weight,
            cross_overlap_gamma_init=args.cross_overlap_gamma_init,
            cross_overlap_second_gamma_init=(
                args.cross_overlap_second_gamma_init
            ),
            cross_overlap_fragmentation_alpha=(
                args.cross_overlap_fragmentation_alpha
            ),
            cross_overlap_data=cross_overlap_data,
            overlap_distill_enabled=args.overlap_distill_weight > 0,
            evidence_fusion=args.evidence_fusion,
            bcq_interaction=args.bcq_interaction,
            bcq_class_count=class_count,
            bcq_anchor_ratio=args.bcq_anchor_ratio,
            bcq_anchor_topk=args.bcq_anchor_topk,
            bcq_anchor_dk=args.bcq_anchor_dk,
            bcq_conflict_alpha=args.bcq_conflict_alpha,
            bcq_gamma_init=args.bcq_gamma_init,
            consensus_token_fusion=args.consensus_token_fusion,
            consensus_token_class_count=class_count,
            consensus_token_ratio=args.consensus_token_ratio,
            consensus_token_heads=args.consensus_token_heads,
            consensus_token_topk=args.consensus_token_topk,
            consensus_token_tau=args.consensus_token_tau,
            consensus_token_fusion_gate_init=(
                args.consensus_token_fusion_gate_init
            ),
            fdsm_scope=args.fdsm_scope,
            cnn_branch=args.cnn_branch,
            cnn_layout=args.cnn_layout,
            cnn_share_weights=args.cnn_share_weights,
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
    overlap_distill_tensors = None
    if args.overlap_distill_weight > 0:
        overlap_distill_tensors = build_overlap_distill_tensors(
            cross_overlap_data,
            device,
        )
    best_loss = float("inf")
    best_state = None
    dta_diagnostics = []
    dta_film_diagnostics = []
    assignment_graph_diagnostics = []
    cross_overlap_diagnostics = []
    bcq_diagnostics = []
    consensus_token_diagnostics = []
    overlap_distill_diagnostics = []
    evidence_fusion_diagnostics = []
    start_time = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = forward_model()
        real_class_logits = logits[:, :class_count]
        evidence_record = None
        if args.evidence_fusion == "dirichlet":
            fused_alpha = getattr(model, "last_fused_alpha", None)
            if fused_alpha is None:
                raise RuntimeError(
                    "EDL requires fused Dirichlet alpha from the model."
                )
            classification_loss, edl_record = edl_classification_loss(
                fused_alpha.index_select(0, train_index),
                train_labels,
                classifier_output_dim,
                epoch,
                args.epochs,
                kl_weight=args.edl_kl_weight,
                anneal_ratio=args.edl_anneal_ratio,
            )
            evidence_record = {
                "epoch": epoch,
                "fusion": args.evidence_fusion,
                "kl_weight": float(args.edl_kl_weight),
                "anneal_ratio": float(args.edl_anneal_ratio),
                **edl_record,
                **(getattr(model, "last_evidence_diagnostics", None) or {}),
            }
        else:
            classification_loss = criterion(
                logits.index_select(0, train_index),
                train_labels,
            )
        loss = classification_loss
        dta_aux_record = None
        if (
            args.dual_transport_arbitration != "none"
            and args.dta_variant == "full"
        ):
            dta_branch = getattr(model, "dta_branch", None)
            dta_aux_losses = getattr(dta_branch, "last_aux_losses", {}) or {}
            if dta_aux_losses:
                dta_sparse_loss = dta_aux_losses.get(
                    "sparse",
                    loss.new_tensor(0.0),
                )
                dta_align_loss = dta_aux_losses.get(
                    "align",
                    loss.new_tensor(0.0),
                )
                dta_entropy_loss = dta_aux_losses.get(
                    "entropy",
                    loss.new_tensor(0.0),
                )
                weighted_dta_aux_loss = (
                    args.dta_aux_sparse_weight * dta_sparse_loss
                    + args.dta_aux_align_weight * dta_align_loss
                    + args.dta_aux_entropy_weight * dta_entropy_loss
                )
                loss = loss + weighted_dta_aux_loss
                dta_aux_record = {
                    "aux_sparse_weight": float(args.dta_aux_sparse_weight),
                    "aux_align_weight": float(args.dta_aux_align_weight),
                    "aux_entropy_weight": float(args.dta_aux_entropy_weight),
                    "aux_weighted_loss": float(
                        weighted_dta_aux_loss.detach().item()
                    ),
                    "aux_sparse_loss": float(
                        dta_sparse_loss.detach().item()
                    ),
                    "aux_align_loss": float(dta_align_loss.detach().item()),
                    "aux_entropy_loss": float(
                        dta_entropy_loss.detach().item()
                    ),
                }
        consensus_token_record = None
        if args.consensus_token_fusion != "none":
            token_branch = getattr(model, "consensus_token_branch", None)
            token_outputs = getattr(
                model,
                "last_consensus_token_outputs",
                None,
            )
            if token_branch is None or token_outputs is None:
                raise RuntimeError(
                    "Consensus token fusion requires cached token outputs."
                )
            h_token_ce = token_branch.consensus_ce(
                token_outputs["p_h_pix"],
                train_index,
                train_labels,
            )
            l_token_ce = token_branch.consensus_ce(
                token_outputs["p_l_pix"],
                train_index,
                train_labels,
            )
            token_ce_loss = 0.5 * (h_token_ce + l_token_ce)
            token_agreement_loss = token_outputs["agreement_loss"]
            token_usage_loss = token_outputs["usage_loss"]
            weighted_token_loss = (
                args.consensus_token_ce_weight * token_ce_loss
                + args.consensus_token_agreement_weight
                * token_agreement_loss
                + args.consensus_token_usage_weight * token_usage_loss
            )
            loss = loss + weighted_token_loss
            consensus_token_record = {
                "epoch": epoch,
                "fusion": args.consensus_token_fusion,
                "ce_weight": float(args.consensus_token_ce_weight),
                "agreement_weight": float(
                    args.consensus_token_agreement_weight
                ),
                "usage_weight": float(args.consensus_token_usage_weight),
                "weighted_loss": float(
                    weighted_token_loss.detach().item()
                ),
                "token_ce": float(token_ce_loss.detach().item()),
                "h_token_ce": float(h_token_ce.detach().item()),
                "l_token_ce": float(l_token_ce.detach().item()),
                "agreement_loss": float(
                    token_agreement_loss.detach().item()
                ),
                "usage_loss": float(token_usage_loss.detach().item()),
                **(
                    getattr(model, "last_consensus_token_diagnostics", None)
                    or {}
                ),
            }
        overlap_distill_record = None
        if args.overlap_distill_weight > 0:
            inputs_are_probabilities = args.evidence_fusion == "dirichlet"
            if inputs_are_probabilities:
                hsi_node_scores = getattr(
                    model,
                    "last_hsi_node_probabilities",
                    None,
                )
                lidar_node_scores = getattr(
                    model,
                    "last_lidar_node_probabilities",
                    None,
                )
            else:
                hsi_node_scores = getattr(model, "last_hsi_node_logits", None)
                lidar_node_scores = getattr(
                    model,
                    "last_lidar_node_logits",
                    None,
                )
            if hsi_node_scores is None or lidar_node_scores is None:
                raise RuntimeError(
                    "Overlap distillation requires separate HSI/LiDAR "
                    "node predictions. Use --graph-layout separate."
                )
            distill_loss, distill_record = (
                confidence_weighted_overlap_distill_loss(
                    hsi_node_scores,
                    lidar_node_scores,
                    overlap_distill_tensors,
                    class_count,
                    temperature=args.overlap_distill_temperature,
                    iou_threshold=args.overlap_distill_iou_threshold,
                    margin=args.overlap_distill_margin,
                    confidence_mode=args.overlap_distill_confidence,
                    inputs_are_probabilities=inputs_are_probabilities,
                )
            )
            distill_ramp = overlap_distill_ramp(
                epoch,
                args.epochs,
                args.overlap_distill_warmup_ratio,
            )
            weighted_distill_loss = (
                args.overlap_distill_weight
                * distill_ramp
                * distill_loss
            )
            loss = loss + weighted_distill_loss
            overlap_distill_record = {
                "epoch": epoch,
                "weight": float(args.overlap_distill_weight),
                "temperature": float(args.overlap_distill_temperature),
                "iou_threshold": float(args.overlap_distill_iou_threshold),
                "margin": float(args.overlap_distill_margin),
                "warmup_ratio": float(args.overlap_distill_warmup_ratio),
                "ramp": float(distill_ramp),
                "confidence": args.overlap_distill_confidence,
                "input": (
                    "expected-probability"
                    if inputs_are_probabilities
                    else "logits"
                ),
                "weighted_loss": float(
                    weighted_distill_loss.detach().item()
                ),
                **distill_record,
            }
        loss.backward()
        optimizer.step()
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
        if epoch == 1 or epoch % args.log_interval == 0:
            dta_record = getattr(model, "last_dta_diagnostics", None)
            if dta_record is not None:
                dta_record = {
                    "epoch": epoch,
                    **dta_record,
                }
                if dta_aux_record is not None:
                    dta_record.update(dta_aux_record)
                dta_diagnostics.append(dta_record)
            dta_film_record = getattr(
                model,
                "last_dta_film_diagnostics",
                None,
            )
            if dta_film_record is not None:
                dta_film_record = {
                    "epoch": epoch,
                    **dta_film_record,
                }
                dta_film_diagnostics.append(dta_film_record)
            assignment_graph_record = getattr(
                model,
                "last_assignment_graph_diagnostics",
                None,
            )
            if assignment_graph_record is not None:
                assignment_graph_record = {
                    "epoch": epoch,
                    **assignment_graph_record,
                }
                assignment_graph_diagnostics.append(assignment_graph_record)
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
            bcq_record = getattr(model, "last_bcq_diagnostics", None)
            if bcq_record is not None:
                bcq_record = {
                    "epoch": epoch,
                    **bcq_record,
                }
                bcq_diagnostics.append(bcq_record)
            if consensus_token_record is not None:
                consensus_token_diagnostics.append(consensus_token_record)
            if overlap_distill_record is not None:
                overlap_distill_diagnostics.append(
                    overlap_distill_record
                )
            if evidence_record is not None:
                evidence_fusion_diagnostics.append(evidence_record)
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
            if overlap_distill_record is not None:
                print(
                    "  Overlap distill: "
                    f"w={overlap_distill_record['weight']:.3g}, "
                    f"input={overlap_distill_record['input']}, "
                    f"ramp={overlap_distill_record['ramp']:.3f}, "
                    f"loss={overlap_distill_record['weighted_loss']:.6f}, "
                    f"edges={overlap_distill_record['selected_edges']}, "
                    "H->L/L->H="
                    f"{overlap_distill_record['h_to_l_edges']}/"
                    f"{overlap_distill_record['l_to_h_edges']}, "
                    "conf="
                    f"{overlap_distill_record['mean_conf_h']:.3f}/"
                    f"{overlap_distill_record['mean_conf_l']:.3f}, "
                    f"IoU={overlap_distill_record['mean_iou']:.3f}"
                )
            if evidence_record is not None:
                print(
                    "  Evidence fusion: "
                    f"EDL-ce={evidence_record['edl_ce']:.4f}, "
                    f"EDL-kl={evidence_record['edl_kl']:.4f}, "
                    f"anneal={evidence_record['edl_anneal']:.3f}, "
                    "uncert="
                    f"{evidence_record['h_uncertainty_mean']:.3f}/"
                    f"{evidence_record['l_uncertainty_mean']:.3f}/"
                    f"{evidence_record['fused_uncertainty_mean']:.3f}, "
                    "conflict="
                    f"{evidence_record['dempster_conflict_mean']:.3f}"
                )
            if dta_record is not None:
                print(
                    "  DTA: "
                    f"{dta_record['mode']}, "
                    f"edges={dta_record['edge_count']}, "
                    "gamma="
                    f"{dta_record['h_gamma']:.4f}/"
                    f"{dta_record['l_gamma']:.4f}, "
                    "D="
                    f"{dta_record['h_conflict_mean']:.4f}/"
                    f"{dta_record['l_conflict_mean']:.4f}, "
                    "gate="
                    f"{dta_record['h_gate_mean']:.4f}/"
                    f"{dta_record['l_gate_mean']:.4f}, "
                    "entropy geo/sem="
                    f"{dta_record['h_geo_entropy']:.3f}/"
                    f"{dta_record['h_sem_entropy']:.3f};"
                    f"{dta_record['l_geo_entropy']:.3f}/"
                    f"{dta_record['l_sem_entropy']:.3f}"
                )
                if "aux_weighted_loss" in dta_record:
                    print(
                        "  DTA full aux: "
                        f"loss={dta_record['aux_weighted_loss']:.6f}, "
                        f"sparse={dta_record['aux_sparse_loss']:.4f}, "
                        f"align={dta_record['aux_align_loss']:.4f}, "
                        f"ent={dta_record['aux_entropy_loss']:.4f}, "
                        "escape="
                        f"{dta_record['h_escape_mean']:.3f}/"
                        f"{dta_record['l_escape_mean']:.3f}"
                    )
            if dta_film_record is not None:
                print(
                    "  DTA-FiLM: "
                    f"ds={dta_film_record['shared_dim']}, "
                    "gamma="
                    f"{dta_film_record['gamma_h']:.4f}/"
                    f"{dta_film_record['gamma_l']:.4f}, "
                    "alpha="
                    f"{dta_film_record['alpha_h']:.3f}/"
                    f"{dta_film_record['alpha_l']:.3f}, "
                    "shrink="
                    f"{dta_film_record['shrink_h_mean']:.3f}/"
                    f"{dta_film_record['shrink_l_mean']:.3f}, "
                    "delta="
                    f"{dta_film_record['delta_h_norm']:.4f}/"
                    f"{dta_film_record['delta_l_norm']:.4f}"
                )
            if assignment_graph_record is not None:
                print(
                    "  Assignment graph: "
                    f"out={assignment_graph_record['output']}, "
                    f"nodes={assignment_graph_record['assignment_node_count']}, "
                    f"topk={assignment_graph_record['assignment_topk']}, "
                    "gamma="
                    f"{assignment_graph_record['h_gamma']:.4f}/"
                    f"{assignment_graph_record['l_gamma']:.4f}, "
                    f"pair-gamma={assignment_graph_record['pair_gamma']:.3f}, "
                    "gate="
                    f"{assignment_graph_record['h_gate_mean']:.3f}/"
                    f"{assignment_graph_record['l_gate_mean']:.3f}, "
                    f"match={assignment_graph_record['pair_match_mean']:.3f}, "
                    "adj="
                    f"dens{assignment_graph_record['assignment_density']:.4f}/"
                    f"ent{assignment_graph_record['assignment_entropy_mean']:.3f}"
                )
            if cross_overlap_record is not None:
                if (
                    cross_overlap_record["transport_fusion"]
                    == "dual-path"
                ):
                    sparse_suffix = (
                        "gamma-con/conf="
                        f"{cross_overlap_record['h_consensus_gamma']:.3f}/"
                        f"{cross_overlap_record['h_conflict_gamma']:.3f};"
                        f"{cross_overlap_record['l_consensus_gamma']:.3f}/"
                        f"{cross_overlap_record['l_conflict_gamma']:.3f}, "
                        "gate-con/conf="
                        f"{cross_overlap_record['h_consensus_gate_mean']:.3f}/"
                        f"{cross_overlap_record['h_conflict_gate_mean']:.3f};"
                        f"{cross_overlap_record['l_consensus_gate_mean']:.3f}/"
                        f"{cross_overlap_record['l_conflict_gate_mean']:.3f}"
                    )
                else:
                    sparse_suffix = (
                        "gamma="
                        f"{cross_overlap_record['h_transport_gamma']:.3f}/"
                        f"{cross_overlap_record['l_transport_gamma']:.3f}, "
                        "gate="
                        f"{cross_overlap_record['h_gate_mean']:.3f}/"
                        f"{cross_overlap_record['l_gate_mean']:.3f}"
                    )
                sparse_suffix = (
                    sparse_suffix
                    + ", frag="
                    f"a{cross_overlap_record.get('fragmentation_alpha', 0.0):.2g},"
                    "g"
                    f"{cross_overlap_record.get('h_fragmentation_gate_mean', 1.0):.3f}/"
                    f"{cross_overlap_record.get('l_fragmentation_gate_mean', 1.0):.3f}"
                )
                print(
                    "  Sparse overlap: "
                    f"stage={cross_overlap_record['stage']}, "
                    f"round={cross_overlap_record['transport_round']}, "
                    f"msg={cross_overlap_record['transport_message']}, "
                    f"fusion={cross_overlap_record['transport_fusion']}, "
                    f"edge-attrs={cross_overlap_record['edge_attribute_mode']}, "
                    f"edge-bias={cross_overlap_record.get('edge_bias_mean', 0.0) or 0.0:.3f}, "
                    f"edges={cross_overlap_record['edge_count']}/"
                    f"{cross_overlap_record.get('original_edge_count', cross_overlap_record['edge_count'])}"
                    f"({cross_overlap_record.get('retained_edge_fraction', 1.0):.2f}), "
                    f"{sparse_suffix}"
                    )
            if bcq_record is not None:
                print(
                    "  BCQ: "
                    f"slot={bcq_record['interaction']}, "
                    f"K={bcq_record['anchor_count']}, "
                    f"topk={bcq_record['anchor_topk']}, "
                    "gamma="
                    f"{bcq_record['h_gamma']:.4f}/"
                    f"{bcq_record['l_gamma']:.4f}, "
                    "jsd="
                    f"{bcq_record['h_jsd_mean']:.4f}/"
                    f"{bcq_record['l_jsd_mean']:.4f}, "
                    "gate="
                    f"{bcq_record['h_gate_mean']:.4f}/"
                    f"{bcq_record['l_gate_mean']:.4f}, "
                    "entropy="
                    f"{bcq_record['h_entropy_mean']:.4f}/"
                    f"{bcq_record['l_entropy_mean']:.4f}, "
                    "usage-ent="
                    f"{bcq_record['h_anchor_usage_entropy']:.4f}/"
                    f"{bcq_record['l_anchor_usage_entropy']:.4f}"
                )
            if consensus_token_record is not None:
                view_weights = np.asarray(
                    consensus_token_record["view_weights"]
                )
                print(
                    "  Consensus token: "
                    f"K={consensus_token_record['token_count']}, "
                    f"gate={consensus_token_record['fusion_gate']:.4f}, "
                    f"view={view_weights.round(2).tolist()}, "
                    f"loss={consensus_token_record['weighted_loss']:.4f}, "
                    f"ce={consensus_token_record['token_ce']:.4f}, "
                    f"agree={consensus_token_record['agreement_loss']:.4f}, "
                    f"usage={consensus_token_record['usage_loss']:.4f}, "
                    "jsd="
                    f"{consensus_token_record['h_jsd_mean']:.4f}/"
                    f"{consensus_token_record['l_jsd_mean']:.4f}, "
                    f"usage-ent={consensus_token_record['usage_entropy']:.4f}"
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
    bcq_final_node_diagnostics = None
    bcq_module = getattr(model, "bcq_branch", None)
    if bcq_module is not None:
        h_jsd = getattr(bcq_module, "last_h_jsd", None)
        l_jsd = getattr(bcq_module, "last_l_jsd", None)
        h_anchor_usage = getattr(bcq_module, "last_h_anchor_usage", None)
        l_anchor_usage = getattr(bcq_module, "last_l_anchor_usage", None)
        if h_jsd is not None and l_jsd is not None:
            bcq_final_node_diagnostics = {
                "node_order": (
                    "superpixel node indices in hsi_assignment and "
                    "lidar_assignment"
                ),
                "h_jsd": h_jsd.detach().cpu().tolist(),
                "l_jsd": l_jsd.detach().cpu().tolist(),
                "h_anchor_usage": (
                    h_anchor_usage.detach().cpu().tolist()
                    if h_anchor_usage is not None
                    else None
                ),
                "l_anchor_usage": (
                    l_anchor_usage.detach().cpu().tolist()
                    if l_anchor_usage is not None
                    else None
                ),
                "diagnostics": bcq_module.diagnostics(),
            }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_stem = (
        f"{args.dataset}_{args.train_samples_per_class}px_"
        f"{STAGE}_{args.graph_layout}_"
        f"lidar-{args.lidar_segmentation}_"
        f"prior-{args.lidar_graph_prior}_"
        f"{dta_configuration_tag(args)}_"
        f"{assignment_graph_configuration_tag(args)}_"
        f"{cross_overlap_configuration_tag(args)}_"
        f"{bcq_configuration_tag(args)}_"
        f"{consensus_token_configuration_tag(args)}_"
        f"{overlap_distill_configuration_tag(args)}_"
        f"{evidence_fusion_configuration_tag(args)}_"
        f"fdsm-{args.fdsm_scope}_"
        f"{cnn_configuration_tag(args)}_"
        f"lidarmod-{args.lidar_modulation}_"
        f"{seed_configuration_tag(args)}_"
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
        "dta_diagnostics": dta_diagnostics,
        "dta_film_diagnostics": dta_film_diagnostics,
        "assignment_graph_diagnostics": assignment_graph_diagnostics,
        "cross_overlap_diagnostics": cross_overlap_diagnostics,
        "bcq_diagnostics": bcq_diagnostics,
        "consensus_token_diagnostics": consensus_token_diagnostics,
        "bcq_final_node_diagnostics": bcq_final_node_diagnostics,
        "overlap_distill_diagnostics": overlap_distill_diagnostics,
        "evidence_fusion_diagnostics": evidence_fusion_diagnostics,
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
        "dual_transport_arbitration",
        "none",
        "--dual-transport-arbitration",
    )
    reset_if_needed("dta_dk", 64, "--dta-dk")
    reset_if_needed("dta_tau", 0.1, "--dta-tau")
    reset_if_needed("dta_gamma_init", 0.0, "--dta-gamma-init")
    reset_if_needed("dta_variant", "simple", "--dta-variant")
    reset_if_needed("dta_nonlocal_topk", 10, "--dta-nonlocal-topk")
    reset_if_needed("dta_normalization", "sinkhorn", "--dta-normalization")
    reset_if_needed("dta_sinkhorn_iters", 5, "--dta-sinkhorn-iters")
    reset_if_needed("dta_film", False, "--dta-film")
    reset_if_needed("dta_film_ds", 64, "--dta-film-ds")
    reset_if_needed(
        "dta_film_gamma_init",
        0.0,
        "--dta-film-gamma-init",
    )
    reset_if_needed(
        "dta_aux_sparse_weight",
        0.05,
        "--dta-aux-sparse-weight",
    )
    reset_if_needed(
        "dta_aux_align_weight",
        0.10,
        "--dta-aux-align-weight",
    )
    reset_if_needed(
        "dta_aux_entropy_weight",
        0.01,
        "--dta-aux-entropy-weight",
    )
    reset_if_needed("assignment_graph", "none", "--assignment-graph")
    reset_if_needed(
        "assignment_graph_topk",
        8,
        "--assignment-graph-topk",
    )
    reset_if_needed(
        "assignment_graph_gamma_init",
        0.0,
        "--assignment-graph-gamma-init",
    )
    reset_if_needed(
        "assignment_graph_output",
        "writeback",
        "--assignment-graph-output",
    )
    reset_if_needed(
        "assignment_graph_weight",
        0.1,
        "--assignment-graph-weight",
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
        "cross_overlap_edge_attrs",
        "none",
        "--cross-overlap-edge-attrs",
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
        "cross_overlap_min_coverage",
        0.0,
        "--cross-overlap-min-coverage",
    )
    reset_if_needed(
        "cross_overlap_iou_topk",
        0,
        "--cross-overlap-iou-topk",
    )
    reset_if_needed(
        "cross_overlap_fragmentation_alpha",
        0.0,
        "--cross-overlap-fragmentation-alpha",
    )
    reset_if_needed(
        "overlap_distill_weight",
        0.0,
        "--overlap-distill-weight",
    )
    reset_if_needed(
        "overlap_distill_temperature",
        2.0,
        "--overlap-distill-temperature",
    )
    reset_if_needed(
        "overlap_distill_iou_threshold",
        0.3,
        "--overlap-distill-iou-threshold",
    )
    reset_if_needed(
        "overlap_distill_margin",
        0.1,
        "--overlap-distill-margin",
    )
    reset_if_needed(
        "overlap_distill_warmup_ratio",
        0.2,
        "--overlap-distill-warmup-ratio",
    )
    reset_if_needed(
        "overlap_distill_confidence",
        "max-prob",
        "--overlap-distill-confidence",
    )
    reset_if_needed("evidence_fusion", "none", "--evidence-fusion")
    reset_if_needed("edl_kl_weight", 0.1, "--edl-kl-weight")
    reset_if_needed("edl_anneal_ratio", 0.5, "--edl-anneal-ratio")
    reset_if_needed("bcq_interaction", "none", "--bcq-interaction")
    reset_if_needed("bcq_anchor_ratio", 4, "--bcq-anchor-ratio")
    reset_if_needed("bcq_anchor_topk", 8, "--bcq-anchor-topk")
    reset_if_needed("bcq_anchor_dk", 32, "--bcq-anchor-dk")
    reset_if_needed("bcq_conflict_alpha", 2.0, "--bcq-conflict-alpha")
    reset_if_needed("bcq_gamma_init", 0.0, "--bcq-gamma-init")
    reset_if_needed(
        "consensus_token_fusion",
        "none",
        "--consensus-token-fusion",
    )
    reset_if_needed(
        "consensus_token_ratio",
        4,
        "--consensus-token-ratio",
    )
    reset_if_needed(
        "consensus_token_heads",
        4,
        "--consensus-token-heads",
    )
    reset_if_needed(
        "consensus_token_topk",
        0,
        "--consensus-token-topk",
    )
    reset_if_needed(
        "consensus_token_tau",
        1.0,
        "--consensus-token-tau",
    )
    reset_if_needed(
        "consensus_token_ce_weight",
        0.4,
        "--consensus-token-ce-weight",
    )
    reset_if_needed(
        "consensus_token_agreement_weight",
        0.1,
        "--consensus-token-agreement-weight",
    )
    reset_if_needed(
        "consensus_token_usage_weight",
        0.01,
        "--consensus-token-usage-weight",
    )
    reset_if_needed(
        "consensus_token_fusion_gate_init",
        0.0,
        "--consensus-token-fusion-gate-init",
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
    if args.dual_transport_arbitration in ("post-gat", "postgat"):
        args.dual_transport_arbitration = "post-gat2"
    if args.assignment_graph in ("post-gat", "postgat"):
        args.assignment_graph = "post-gat2"
    if args.bcq_interaction in ("post-gat", "postgat"):
        args.bcq_interaction = "post-gat2"
    if args.consensus_token_fusion in ("post-gat", "postgat"):
        args.consensus_token_fusion = "post-gat2"
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
    if args.dta_dk <= 0:
        raise ValueError("--dta-dk must be positive.")
    if args.dta_tau <= 0:
        raise ValueError("--dta-tau must be positive.")
    if args.dta_gamma_init < 0:
        raise ValueError("--dta-gamma-init must be nonnegative.")
    if args.dta_nonlocal_topk < 0:
        raise ValueError("--dta-nonlocal-topk must be nonnegative.")
    if args.dta_sinkhorn_iters <= 0:
        raise ValueError("--dta-sinkhorn-iters must be positive.")
    if args.dta_film_ds <= 0:
        raise ValueError("--dta-film-ds must be positive.")
    if args.dta_film_ds > args.hidden_dim:
        raise ValueError("--dta-film-ds must be <= --hidden-dim.")
    if args.dta_film_gamma_init < 0:
        raise ValueError("--dta-film-gamma-init must be nonnegative.")
    if args.dta_aux_sparse_weight < 0:
        raise ValueError("--dta-aux-sparse-weight must be nonnegative.")
    if args.dta_aux_align_weight < 0:
        raise ValueError("--dta-aux-align-weight must be nonnegative.")
    if args.dta_aux_entropy_weight < 0:
        raise ValueError("--dta-aux-entropy-weight must be nonnegative.")
    if args.assignment_graph_topk <= 0:
        raise ValueError("--assignment-graph-topk must be positive.")
    if args.assignment_graph_gamma_init < 0:
        raise ValueError(
            "--assignment-graph-gamma-init must be nonnegative."
        )
    if not 0.0 <= args.assignment_graph_weight <= 1.0:
        raise ValueError("--assignment-graph-weight must be in [0, 1].")
    if args.cross_overlap_min_coverage < 0:
        raise ValueError(
            "--cross-overlap-min-coverage must be nonnegative."
        )
    if args.cross_overlap_iou_topk < 0:
        raise ValueError("--cross-overlap-iou-topk must be nonnegative.")
    if args.cross_overlap_fragmentation_alpha < 0:
        raise ValueError(
            "--cross-overlap-fragmentation-alpha must be nonnegative."
        )
    if args.cross_overlap_relation != "none":
        if args.graph_layout != "separate":
            raise ValueError(
                "--cross-overlap-relation sparse requires "
                "--graph-layout separate."
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
        if (
            args.cross_overlap_edge_attrs != "none"
            and args.cross_overlap_message != "qk-prior"
        ):
            raise ValueError(
                "--cross-overlap-edge-attrs physical requires "
                "--cross-overlap-message qk-prior."
            )
    elif args.cross_overlap_edge_attrs != "none":
        raise ValueError(
            "--cross-overlap-edge-attrs physical requires "
            "--cross-overlap-relation sparse."
        )
    if args.dual_transport_arbitration != "none":
        if args.graph_layout != "separate":
            raise ValueError(
                "--dual-transport-arbitration requires "
                "--graph-layout separate."
            )
        if args.cross_overlap_relation != "none":
            raise ValueError(
                "--dual-transport-arbitration uses the same overlap "
                "support and is mutually exclusive with "
                "--cross-overlap-relation sparse."
            )
    elif args.dta_film:
        raise ValueError("--dta-film requires --dual-transport-arbitration.")
    if args.assignment_graph != "none":
        if args.graph_layout != "separate":
            raise ValueError(
                "--assignment-graph requires --graph-layout separate."
            )
        if args.dual_transport_arbitration != "none":
            raise ValueError(
                "--assignment-graph and --dual-transport-arbitration "
                "are separate post-GAT2 interaction ablations."
            )
        if args.cross_overlap_relation != "none":
            raise ValueError(
                "--assignment-graph and --cross-overlap-relation sparse "
                "are separate interaction ablations."
            )
        if args.bcq_interaction != "none":
            raise ValueError(
                "--assignment-graph and --bcq-interaction are separate "
                "post-GAT2 interaction ablations."
            )
        if args.consensus_token_fusion != "none":
            raise ValueError(
                "--assignment-graph and --consensus-token-fusion are "
                "separate post-GAT2 interaction ablations."
            )
    if args.overlap_distill_weight < 0:
        raise ValueError("--overlap-distill-weight must be nonnegative.")
    if args.overlap_distill_temperature <= 0:
        raise ValueError("--overlap-distill-temperature must be positive.")
    if args.overlap_distill_iou_threshold < 0:
        raise ValueError(
            "--overlap-distill-iou-threshold must be nonnegative."
        )
    if args.overlap_distill_margin < 0:
        raise ValueError("--overlap-distill-margin must be nonnegative.")
    if not 0.0 <= args.overlap_distill_warmup_ratio < 1.0:
        raise ValueError(
            "--overlap-distill-warmup-ratio must be in [0, 1)."
        )
    if args.overlap_distill_weight > 0 and args.graph_layout != "separate":
        raise ValueError(
            "--overlap-distill-weight requires --graph-layout separate."
        )
    if args.bcq_anchor_ratio <= 0:
        raise ValueError("--bcq-anchor-ratio must be positive.")
    if args.bcq_anchor_topk <= 0:
        raise ValueError("--bcq-anchor-topk must be positive.")
    if args.bcq_anchor_dk <= 0:
        raise ValueError("--bcq-anchor-dk must be positive.")
    if args.bcq_conflict_alpha < 0:
        raise ValueError("--bcq-conflict-alpha must be nonnegative.")
    if args.bcq_gamma_init < 0:
        raise ValueError("--bcq-gamma-init must be nonnegative.")
    if args.consensus_token_ratio <= 0:
        raise ValueError("--consensus-token-ratio must be positive.")
    if args.consensus_token_heads <= 0:
        raise ValueError("--consensus-token-heads must be positive.")
    if args.hidden_dim % args.consensus_token_heads != 0:
        raise ValueError(
            "--hidden-dim must be divisible by --consensus-token-heads."
        )
    if args.consensus_token_topk < 0:
        raise ValueError("--consensus-token-topk must be nonnegative.")
    if args.consensus_token_tau <= 0:
        raise ValueError("--consensus-token-tau must be positive.")
    if args.consensus_token_ce_weight < 0:
        raise ValueError(
            "--consensus-token-ce-weight must be nonnegative."
        )
    if args.consensus_token_agreement_weight < 0:
        raise ValueError(
            "--consensus-token-agreement-weight must be nonnegative."
        )
    if args.consensus_token_usage_weight < 0:
        raise ValueError(
            "--consensus-token-usage-weight must be nonnegative."
        )
    if args.bcq_interaction != "none":
        if args.graph_layout != "separate":
            raise ValueError(
                "--bcq-interaction requires --graph-layout separate."
            )
        if args.cross_overlap_relation != "none":
            raise ValueError(
                "--bcq-interaction is mutually exclusive with "
                "--cross-overlap-relation sparse."
            )
        if args.consensus_token_fusion != "none":
            raise ValueError(
                "--bcq-interaction and --consensus-token-fusion are "
                "separate ablations; enable only one."
            )
    if args.consensus_token_fusion != "none":
        if args.graph_layout != "separate":
            raise ValueError(
                "--consensus-token-fusion requires --graph-layout separate."
            )
        if args.cross_overlap_relation != "none":
            raise ValueError(
                "--consensus-token-fusion is mutually exclusive with "
                "--cross-overlap-relation sparse."
            )
        if args.evidence_fusion != "none":
            raise ValueError(
                "--consensus-token-fusion currently fuses logits and is "
                "mutually exclusive with --evidence-fusion."
            )
    if args.edl_kl_weight < 0:
        raise ValueError("--edl-kl-weight must be nonnegative.")
    if not 0.0 < args.edl_anneal_ratio <= 1.0:
        raise ValueError("--edl-anneal-ratio must be in (0, 1].")
    if args.evidence_fusion != "none" and args.graph_layout != "separate":
        raise ValueError(
            "--evidence-fusion dirichlet requires --graph-layout separate."
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
        cross_overlap_data,
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
        if args.dual_transport_arbitration != "none":
            edge_count = 0
            if cross_overlap_data is not None:
                edge_count = cross_overlap_data.get("edge_count", 0)
            print(
                "Dual transport arbitration: "
                f"{args.dual_transport_arbitration} | "
                f"variant={args.dta_variant} | "
                f"d_k={args.dta_dk} | "
                f"tau={args.dta_tau:g} | "
                f"gamma-init={args.dta_gamma_init:g} | "
                f"overlap-edges={edge_count} | "
                "geometry/semantic transport mixed before readout"
            )
            if args.dta_variant == "full":
                print(
                    "  DTA full: "
                    f"nonlocal-topk={args.dta_nonlocal_topk} | "
                    f"normalization={args.dta_normalization} | "
                    f"sinkhorn-iters={args.dta_sinkhorn_iters} | "
                    "aux weights="
                    f"{args.dta_aux_sparse_weight:g}/"
                    f"{args.dta_aux_align_weight:g}/"
                    f"{args.dta_aux_entropy_weight:g}"
                )
            if args.dta_film:
                print(
                    "  DTA-FiLM: "
                    f"shared-dim={args.dta_film_ds} | "
                    f"gamma-init={args.dta_film_gamma_init:g} | "
                    "conflict-aware shared-subspace modulation"
                )
        if args.assignment_graph != "none":
            edge_count = 0
            retained_fraction = 1.0
            if cross_overlap_data is not None:
                edge_count = cross_overlap_data.get("edge_count", 0)
                retained_fraction = cross_overlap_data.get(
                    "retained_edge_fraction",
                    1.0,
                )
            print(
                "Assignment graph interaction: "
                f"slot={args.assignment_graph} | "
                f"topk={args.assignment_graph_topk} | "
                f"gamma-init={args.assignment_graph_gamma_init:g} | "
                f"output={args.assignment_graph_output} | "
                f"weight={args.assignment_graph_weight:g} | "
                f"assignment-nodes={edge_count}({retained_fraction:.2%}) | "
                "overlap-restricted SEGMN-style pair graph after GAT2"
            )
        if args.cross_overlap_relation != "none":
            original_edges = 0
            retained_edges = 0
            retained_fraction = 1.0
            if cross_overlap_data is not None:
                original_edges = cross_overlap_data.get(
                    "original_edge_count",
                    cross_overlap_data.get("edge_count", 0),
                )
                retained_edges = cross_overlap_data.get("edge_count", 0)
                retained_fraction = cross_overlap_data.get(
                    "retained_edge_fraction",
                    1.0,
                )
            print(
                "Sparse overlap relation: "
                f"{args.cross_overlap_relation} | "
                f"stage={args.cross_overlap_stage} | "
                f"message={args.cross_overlap_message} | "
                f"fusion={args.cross_overlap_fusion} | "
                f"edge-attrs={args.cross_overlap_edge_attrs} | "
                f"prior-weight={args.cross_overlap_prior_weight:g} | "
                "gamma-init="
                f"{args.cross_overlap_gamma_init:g}/"
                f"{args.cross_overlap_second_gamma_init:g} | "
                "prune="
                f"coverage<{args.cross_overlap_min_coverage:g},"
                f"iou-topk={args.cross_overlap_iou_topk} | "
                f"frag-alpha={args.cross_overlap_fragmentation_alpha:g} | "
                f"edges={retained_edges}/{original_edges}"
                f"({retained_fraction:.2%}) | "
                "explicit C graph disabled"
            )
        if args.bcq_interaction != "none":
            edge_count = 0
            retained_fraction = 1.0
            if cross_overlap_data is not None:
                edge_count = cross_overlap_data.get("edge_count", 0)
                retained_fraction = cross_overlap_data.get(
                    "retained_edge_fraction",
                    1.0,
                )
            print(
                "BCQ interaction: "
                f"slot={args.bcq_interaction} | "
                f"anchors={args.bcq_anchor_ratio}xC | "
                f"topk={args.bcq_anchor_topk} | "
                f"d_k={args.bcq_anchor_dk} | "
                f"alpha={args.bcq_conflict_alpha:g} | "
                f"gamma-init={args.bcq_gamma_init:g} | "
                f"overlap-edges={edge_count}({retained_fraction:.2%}) | "
                "transports membership distributions, not features"
            )
        if args.consensus_token_fusion != "none":
            edge_count = 0
            retained_fraction = 1.0
            if cross_overlap_data is not None:
                edge_count = cross_overlap_data.get("edge_count", 0)
                retained_fraction = cross_overlap_data.get(
                    "retained_edge_fraction",
                    1.0,
                )
            print(
                "Consensus token fusion: "
                f"slot={args.consensus_token_fusion} | "
                f"tokens={args.consensus_token_ratio}xC | "
                f"heads={args.consensus_token_heads} | "
                f"topk={args.consensus_token_topk} | "
                f"tau={args.consensus_token_tau:g} | "
                "loss="
                f"ce{args.consensus_token_ce_weight:g}/"
                f"agree{args.consensus_token_agreement_weight:g}/"
                f"usage{args.consensus_token_usage_weight:g} | "
                f"gate-init={args.consensus_token_fusion_gate_init:g} | "
                f"overlap-edges={edge_count}({retained_fraction:.2%}) | "
                "logit fusion only, no node writeback"
            )
        if args.overlap_distill_weight > 0:
            edge_count = 0
            if cross_overlap_data is not None:
                edge_count = cross_overlap_data.get("edge_count", 0)
            print(
                "Overlap distillation: "
                f"weight={args.overlap_distill_weight:g} | "
                f"T={args.overlap_distill_temperature:g} | "
                f"IoU>{args.overlap_distill_iou_threshold:g} | "
                f"margin={args.overlap_distill_margin:g} | "
                f"warmup={args.overlap_distill_warmup_ratio:g} | "
                f"confidence={args.overlap_distill_confidence} | "
                f"edges={edge_count} | "
                "teacher=confidence winner with stop-grad"
            )
        if args.evidence_fusion != "none":
            edge_count = 0
            if cross_overlap_data is not None:
                edge_count = cross_overlap_data.get("edge_count", 0)
            print(
                "Evidence decision fusion: "
                f"{args.evidence_fusion} | "
                "heads=HSI/LiDAR fc->softplus evidence | "
                "pixel fusion=Dempster | "
                f"EDL-kl={args.edl_kl_weight:g} | "
                f"anneal-ratio={args.edl_anneal_ratio:g} | "
                f"overlap-conflict-edges={edge_count} | "
                "CNN classifier bypassed"
            )
        print(
            "Graph fusion H/L="
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
    if args.runs == 1:
        seed_text = str(args.seed)
    else:
        seed_text = f"{args.seed}..{args.seed + args.runs - 1}"
    print(f"Seed: base={args.seed} | run-seeds={seed_text}")
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
            cross_overlap_data,
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
        f"{dta_configuration_tag(args)}_"
        f"{assignment_graph_configuration_tag(args)}_"
        f"{cross_overlap_configuration_tag(args)}_"
        f"{bcq_configuration_tag(args)}_"
        f"{consensus_token_configuration_tag(args)}_"
        f"{overlap_distill_configuration_tag(args)}_"
        f"{evidence_fusion_configuration_tag(args)}_"
        f"fdsm-{args.fdsm_scope}_"
        f"{cnn_configuration_tag(args)}_"
        f"lidarmod-{args.lidar_modulation}_"
        f"{seed_configuration_tag(args)}_"
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
