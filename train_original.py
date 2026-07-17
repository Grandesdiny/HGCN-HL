import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

from train import (
    DATASET_CONFIG,
    OriginalSSConv,
    WMF,
    minmax_normalize,
    set_seed,
    split_fixed_samples_per_class,
)
from utils import (
    get_HSI_LiDAR_data,
    get_HSI_performance,
    obtain_H_from_HSI_with_LiDAR,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Train the original pixel-vertex HGCN-HL with a fixed "
            "number of training pixels per class."
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
        choices=("felzenszwalb", "slic"),
        default="felzenszwalb",
        help=(
            "Segmentation used to create LiDAR region hyperedges. "
            "Felzenszwalb reproduces the original HGCN-HL."
        ),
    )
    parser.add_argument("--fusion-lambda", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--dropout", type=float, default=0.4)
    parser.add_argument("--seed", type=int, default=0)#4242
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("model_original"),
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


def scipy_incidence_to_torch(incidence):
    coo = incidence.tocoo()
    indices = torch.from_numpy(
        np.vstack((coo.row, coo.col))
    ).long()
    values = torch.from_numpy(coo.data.astype(np.float32))
    with torch.sparse.check_sparse_tensor_invariants():
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=coo.shape,
        ).coalesce()


class FixedIncidenceHGCN(nn.Module):
    """Original normalized pixel -> hyperedge -> pixel propagation."""

    def __init__(self, in_channels, out_channels, incidence):
        super().__init__()
        incidence = incidence.coalesce()
        self.register_buffer(
            "incidence",
            incidence,
            persistent=False,
        )
        edge_degree = torch.sparse.sum(
            incidence,
            dim=0,
        ).to_dense().clamp_min(1.0).reciprocal()
        node_degree = torch.sparse.sum(
            incidence,
            dim=1,
        ).to_dense().clamp_min(1.0).pow(-0.5)
        self.register_buffer(
            "edge_degree",
            edge_degree,
            persistent=False,
        )
        self.register_buffer(
            "node_degree",
            node_degree,
            persistent=False,
        )
        self.linear = nn.Linear(
            in_channels,
            out_channels,
            bias=False,
        )
        self.hyperedge_weight = nn.Parameter(
            torch.ones(incidence.shape[1])
        )

    def forward(self, node_features):
        node_features = self.linear(node_features)
        node_features = (
            node_features * self.node_degree.unsqueeze(1)
        )
        hyperedge_features = torch.sparse.mm(
            self.incidence.transpose(0, 1),
            node_features,
        )
        hyperedge_features = (
            hyperedge_features
            * self.edge_degree.unsqueeze(1)
            * self.hyperedge_weight.unsqueeze(1)
        )
        outputs = torch.sparse.mm(
            self.incidence,
            hyperedge_features,
        )
        return outputs * self.node_degree.unsqueeze(1)


class OriginalHGCNHL(nn.Module):
    def __init__(
        self,
        height,
        width,
        input_channels,
        class_count,
        incidence,
        hidden_dim=128,
        fusion_lambda=0.5,
        dropout=0.4,
    ):
        super().__init__()
        self.height = height
        self.width = width
        self.fusion_lambda = fusion_lambda
        self.feature_mapping = nn.Sequential(
            WMF(input_channels, hidden_dim),
            WMF(hidden_dim, hidden_dim),
        )
        self.cnn_branch = nn.Sequential(
            OriginalSSConv(hidden_dim, hidden_dim, kernel_size=5),
            OriginalSSConv(hidden_dim, hidden_dim, kernel_size=5),
        )
        self.hgcn1 = FixedIncidenceHGCN(
            hidden_dim,
            hidden_dim,
            incidence,
        )
        self.hgcn2 = FixedIncidenceHGCN(
            hidden_dim,
            hidden_dim,
            incidence,
        )
        self.activation = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, class_count)

    def forward(self, inputs):
        mapped = self.feature_mapping(
            inputs.permute(2, 0, 1).unsqueeze(0)
        )
        cnn_features = (
            self.cnn_branch(mapped)
            .squeeze(0)
            .permute(1, 2, 0)
            .reshape(self.height * self.width, -1)
        )
        graph_features = (
            mapped.squeeze(0)
            .permute(1, 2, 0)
            .reshape(self.height * self.width, -1)
        )
        graph_features = self.activation(
            self.dropout(graph_features)
        )
        graph_features = self.activation(
            self.dropout(self.hgcn1(graph_features))
        )
        graph_features = self.activation(
            self.dropout(self.hgcn2(graph_features))
        )
        fused = (
            self.fusion_lambda * graph_features
            + (1.0 - self.fusion_lambda) * cnn_features
        )
        return self.classifier(fused)


def prepare_data(args, config):
    hsi, lidar, gt, class_count, _, _ = get_HSI_LiDAR_data(
        config["loader_name"],
        args.data_dir,
    )
    hsi = minmax_normalize(hsi)
    lidar = minmax_normalize(lidar)
    if lidar.ndim == 3:
        lidar = lidar[:, :, 0]

    incidence = obtain_H_from_HSI_with_LiDAR(
        hsi,
        lidar[:, :, np.newaxis],
        args.scales,
        lidar_segmentation=args.lidar_segmentation,
        return_separate=False,
        sparse_output=True,
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
    model_input = np.concatenate(
        (reduced_hsi, lidar[:, :, np.newaxis]),
        axis=2,
    ).astype(np.float32)
    return (
        model_input,
        gt,
        class_count,
        scipy_incidence_to_torch(incidence),
    )


def train_one_run(args, model_input, gt, class_count, incidence, run_index):
    run_seed = args.seed + run_index
    set_seed(run_seed)
    train_indices, test_indices = split_fixed_samples_per_class(
        gt,
        class_count,
        args.train_samples_per_class,
        run_seed,
    )
    if np.intersect1d(train_indices, test_indices).size:
        raise RuntimeError("Train/test overlap was detected.")

    device = torch.device(args.device)
    inputs = torch.from_numpy(model_input).to(device)
    flat_gt = gt.reshape(-1)
    train_index = torch.from_numpy(train_indices).long().to(device)
    test_index = torch.from_numpy(test_indices).long().to(device)
    train_labels = torch.from_numpy(
        flat_gt[train_indices] - 1
    ).long().to(device)
    test_labels = torch.from_numpy(
        flat_gt[test_indices] - 1
    ).long().to(device)

    model = OriginalHGCNHL(
        height=model_input.shape[0],
        width=model_input.shape[1],
        input_channels=model_input.shape[2],
        class_count=class_count,
        incidence=incidence.to(device),
        hidden_dim=args.hidden_dim,
        fusion_lambda=args.fusion_lambda,
        dropout=args.dropout,
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
        logits = model(inputs)
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
            predictions = (
                logits.index_select(0, train_index).argmax(dim=1)
            )
            train_oa = (
                predictions == train_labels
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
            model(inputs)
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
        f"original_hgcn_run{run_index + 1}.pt"
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


def main():
    args = parse_args()
    config = resolve_options(args)
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
    if not 0.0 <= args.dropout < 1.0:
        raise ValueError("--dropout must be in [0, 1).")

    set_seed(args.seed)
    model_input, gt, class_count, incidence = prepare_data(args, config)
    print(
        f"Dataset: {config['loader_name']} | input={model_input.shape} | "
        f"pixel-nodes={incidence.shape[0]} | "
        f"fixed-region-hyperedges={incidence.shape[1]}"
    )
    print(
        "Hyperedges: HSI SLIC + "
        f"LiDAR {args.lidar_segmentation} regions"
    )
    print(
        f"Split: exactly {args.train_samples_per_class} "
        "training pixels per class"
    )
    results = [
        train_one_run(
            args,
            model_input,
            gt,
            class_count,
            incidence,
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
        "original_hgcn_results.json"
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
