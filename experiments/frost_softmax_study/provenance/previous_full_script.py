from __future__ import annotations

import argparse
import json
import math
import random
import shutil
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw, ImageFont
from skimage.metrics import structural_similarity
from torch import nn


def set_seed(seed: int = 7) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(4, torch.get_num_threads()))


def fsm_stationary_basis(x: torch.Tensor, n_states: int) -> torch.Tensor:
    """Stationary state probabilities for the reflected N-state FSM.

    pi_i(x) is proportional to (x / (1-x))**i.  The log-domain form is
    numerically stable near the interval boundaries.
    """
    x = x.clamp(1e-5, 1.0 - 1e-5)
    log_ratio = torch.logit(x)
    indices = torch.arange(n_states, device=x.device, dtype=x.dtype)
    return torch.softmax(log_ratio.unsqueeze(-1) * indices, dim=-1)


class TTScalarOperator(nn.Module):
    """Analytical expected-output FROST operator for scalar mappings."""

    def __init__(self, input_dim: int, n_states: int = 4, rank: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.n_states = n_states
        self.rank = rank
        cores: list[nn.Parameter] = []
        cores.append(nn.Parameter(torch.randn(1, n_states, rank) * 0.15))
        for _ in range(1, input_dim - 1):
            core = torch.zeros(rank, n_states, rank)
            identity = torch.eye(rank)
            for state in range(n_states):
                core[:, state, :] = identity + 0.02 * torch.randn(rank, rank)
            cores.append(nn.Parameter(core))
        cores.append(nn.Parameter(torch.randn(rank, n_states, 1) * 0.15))
        self.cores = nn.ParameterList(cores)
        self.output_scale = nn.Parameter(torch.tensor(1.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = fsm_stationary_basis(x, self.n_states)
        vector = torch.einsum("bn,anr->br", probs[:, 0], self.cores[0])
        for mode in range(1, self.input_dim - 1):
            matrix = torch.einsum(
                "bn,anr->bar", probs[:, mode], self.cores[mode]
            )
            vector = torch.bmm(vector.unsqueeze(1), matrix).squeeze(1)
        final_matrix = torch.einsum(
            "bn,anr->bar", probs[:, -1], self.cores[-1]
        )
        raw = torch.bmm(vector.unsqueeze(1), final_matrix).squeeze(-1).squeeze(-1)
        return torch.sigmoid(self.output_scale * raw + self.output_bias)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TTAttentionOperator(nn.Module):
    """Multi-output FROST operator that maps a score row to an attention row."""

    def __init__(
        self,
        token_count: int,
        n_states: int = 4,
        rank: int = 12,
    ):
        super().__init__()
        self.token_count = token_count
        self.n_states = n_states
        self.rank = rank
        cores: list[nn.Parameter] = []
        cores.append(nn.Parameter(torch.randn(1, n_states, rank) * 0.15))
        for _ in range(1, token_count - 1):
            core = torch.zeros(rank, n_states, rank)
            identity = torch.eye(rank)
            for state in range(n_states):
                core[:, state, :] = identity + 0.02 * torch.randn(rank, rank)
            cores.append(nn.Parameter(core))
        cores.append(
            nn.Parameter(torch.randn(rank, n_states, token_count) * 0.15)
        )
        self.cores = nn.ParameterList(cores)
        self.output_bias = nn.Parameter(torch.zeros(token_count))
        self.temperature = nn.Parameter(torch.tensor(1.0))

    def expected_logits(self, x: torch.Tensor) -> torch.Tensor:
        probs = fsm_stationary_basis(x, self.n_states)
        vector = torch.einsum("bn,anr->br", probs[:, 0], self.cores[0])
        for mode in range(1, self.token_count - 1):
            matrix = torch.einsum(
                "bn,anr->bar", probs[:, mode], self.cores[mode]
            )
            vector = torch.bmm(vector.unsqueeze(1), matrix).squeeze(1)
        output_matrix = torch.einsum(
            "bn,ank->bak", probs[:, -1], self.cores[-1]
        )
        return torch.bmm(vector.unsqueeze(1), output_matrix).squeeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.expected_logits(x) + self.output_bias
        temperature = self.temperature.clamp(0.2, 5.0)
        return torch.softmax(logits / temperature, dim=-1)

    def selected_logits(self, states: torch.Tensor) -> torch.Tensor:
        """TT lookup for a batch of explicit FSM state codewords."""
        batch_size = states.shape[0]
        first = self.cores[0][0, states[:, 0], :]
        vector = first
        for mode in range(1, self.token_count - 1):
            core = self.cores[mode]
            matrix = core[:, states[:, mode], :].permute(1, 0, 2)
            vector = torch.bmm(vector.unsqueeze(1), matrix).squeeze(1)
        last = self.cores[-1]
        output_matrix = last[:, states[:, -1], :].permute(1, 0, 2)
        return torch.bmm(vector.unsqueeze(1), output_matrix).squeeze(1)

    @torch.no_grad()
    def stochastic_forward(
        self,
        x: torch.Tensor,
        bitstream_length: int,
        burn_in: int = 32,
    ) -> torch.Tensor:
        """Finite-window FSM simulation followed by the learned output calibration."""
        batch_size, dimensions = x.shape
        states = torch.full(
            (batch_size, dimensions),
            fill_value=(self.n_states - 1) // 2,
            dtype=torch.long,
            device=x.device,
        )
        logits_sum = torch.zeros(
            batch_size, self.token_count, dtype=x.dtype, device=x.device
        )
        total_cycles = burn_in + bitstream_length
        for cycle in range(total_cycles):
            bits = torch.rand_like(x) < x
            states = torch.where(
                bits,
                torch.clamp(states + 1, max=self.n_states - 1),
                torch.clamp(states - 1, min=0),
            )
            if cycle >= burn_in:
                logits_sum += self.selected_logits(states)
        mean_logits = logits_sum / bitstream_length + self.output_bias
        temperature = self.temperature.clamp(0.2, 5.0)
        return torch.softmax(mean_logits / temperature, dim=-1)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


@dataclass
class ScalarResult:
    function: str
    rank: int
    rmse: float
    mae: float
    parameters: int


@dataclass
class AttentionResult:
    rank: int
    kl_divergence: float
    rmse: float
    cosine_similarity: float
    top1_agreement: float
    parameters: int


def target_function(x: torch.Tensor, name: str) -> torch.Tensor:
    if name == "Softmax component":
        return torch.softmax(3.0 * (x - 0.5), dim=1)[:, 0]
    if name == "Gaussian radial":
        return torch.exp(-8.0 * torch.mean((x - 0.5) ** 2, dim=1))
    if name == "Multiplicative sinusoid":
        return torch.prod(0.5 + 0.5 * torch.sin(math.pi * x), dim=1)
    if name == "GELU mean":
        z = 4.0 * x - 2.0
        gelu = 0.5 * z * (1.0 + torch.erf(z / math.sqrt(2.0)))
        return ((gelu.mean(dim=1) + 0.16) / 1.96).clamp(0.0, 1.0)
    raise ValueError(f"Unknown function: {name}")


def train_scalar_operator(
    function_name: str,
    rank: int,
    input_dim: int,
    n_states: int,
    steps: int,
    seed: int,
) -> tuple[TTScalarOperator, list[float]]:
    set_seed(seed)
    model = TTScalarOperator(input_dim, n_states, rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-6)
    history: list[float] = []
    for step in range(steps):
        x = torch.rand(384, input_dim)
        target = target_function(x, function_name)
        prediction = model(x)
        loss = torch.mean((target - prediction) ** 2)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 10 == 0:
            history.append(float(torch.sqrt(loss).detach()))
    return model, history


def evaluate_scalar_operator(
    model: TTScalarOperator,
    function_name: str,
    sample_count: int = 6000,
) -> tuple[float, float]:
    with torch.no_grad():
        x = torch.rand(sample_count, model.input_dim)
        target = target_function(x, function_name)
        prediction = model(x)
        rmse = torch.sqrt(torch.mean((target - prediction) ** 2)).item()
        mae = torch.mean(torch.abs(target - prediction)).item()
    return rmse, mae


def make_score_batch(batch_size: int, token_count: int) -> torch.Tensor:
    scores = torch.randn(batch_size, token_count) * 1.30
    scores += torch.randn(batch_size, 1) * 0.35
    # Occasional sharper and flatter rows broaden the fitted temperature range.
    temperature = torch.empty(batch_size, 1).uniform_(0.65, 1.45)
    return scores / temperature


def scores_to_frost_input(scores: torch.Tensor, score_limit: float = 4.0) -> torch.Tensor:
    return ((scores + score_limit) / (2.0 * score_limit)).clamp(0.0, 1.0)


def train_attention_operator(
    token_count: int,
    rank: int,
    n_states: int,
    steps: int,
    seed: int,
) -> tuple[TTAttentionOperator, list[float]]:
    set_seed(seed)
    model = TTAttentionOperator(token_count, n_states, rank)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-6)
    history: list[float] = []
    for step in range(steps):
        scores = make_score_batch(512, token_count)
        target = torch.softmax(scores, dim=-1)
        prediction = model(scores_to_frost_input(scores))
        loss = -torch.sum(target * torch.log(prediction + 1e-8), dim=-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        if step % 10 == 0:
            with torch.no_grad():
                kl = torch.sum(
                    target
                    * (
                        torch.log(target + 1e-8)
                        - torch.log(prediction + 1e-8)
                    ),
                    dim=-1,
                ).mean()
            history.append(float(kl))
    return model, history


def evaluate_attention_operator(
    model: TTAttentionOperator,
    sample_count: int = 5000,
) -> AttentionResult:
    with torch.no_grad():
        scores = make_score_batch(sample_count, model.token_count)
        target = torch.softmax(scores, dim=-1)
        prediction = model(scores_to_frost_input(scores))
        kl = torch.sum(
            target
            * (torch.log(target + 1e-8) - torch.log(prediction + 1e-8)),
            dim=-1,
        ).mean()
        rmse = torch.sqrt(torch.mean((target - prediction) ** 2))
        cosine = torch.nn.functional.cosine_similarity(
            target, prediction, dim=-1
        ).mean()
        agreement = (target.argmax(dim=-1) == prediction.argmax(dim=-1)).float().mean()
    return AttentionResult(
        rank=model.rank,
        kl_divergence=float(kl),
        rmse=float(rmse),
        cosine_similarity=float(cosine),
        top1_agreement=float(agreement),
        parameters=model.parameter_count,
    )


def save_bar_chart(
    labels: list[str],
    groups: dict[str, list[float]],
    ylabel: str,
    output_path: Path,
) -> None:
    positions = np.arange(len(labels))
    width = 0.8 / max(1, len(groups))
    fig = plt.figure(figsize=(9.0, 5.6))
    ax = fig.add_axes([0.12, 0.20, 0.84, 0.74])
    for group_index, (name, values) in enumerate(groups.items()):
        offset = (group_index - (len(groups) - 1) / 2.0) * width
        ax.bar(positions + offset, values, width=width, label=name)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel(ylabel)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def save_line_chart(
    x: list[float] | np.ndarray,
    series: dict[str, list[float] | np.ndarray],
    xlabel: str,
    ylabel: str,
    output_path: Path,
    log_y: bool = False,
) -> None:
    fig = plt.figure(figsize=(7.6, 5.4))
    ax = fig.add_axes([0.15, 0.15, 0.80, 0.79])
    for name, values in series.items():
        ax.plot(x, values, marker="o", linewidth=2, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if log_y:
        ax.set_yscale("log")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def save_heatmap(matrix: np.ndarray, title: str, output_path: Path) -> None:
    fig = plt.figure(figsize=(6.2, 5.4))
    ax = fig.add_axes([0.14, 0.12, 0.72, 0.80])
    image = ax.imshow(matrix, aspect="auto")
    ax.set_xlabel("Key token")
    ax.set_ylabel("Query token")
    ax.set_title(title)
    colorbar_ax = fig.add_axes([0.88, 0.12, 0.03, 0.80])
    fig.colorbar(image, cax=colorbar_ax)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def save_scatter(
    target: np.ndarray,
    prediction: np.ndarray,
    output_path: Path,
    xlabel: str = "Reference",
    ylabel: str = "FROST",
) -> None:
    fig = plt.figure(figsize=(6.0, 5.6))
    ax = fig.add_axes([0.15, 0.14, 0.80, 0.80])
    ax.scatter(target, prediction, s=8, alpha=0.35)
    low = min(float(target.min()), float(prediction.min()))
    high = max(float(target.max()), float(prediction.max()))
    ax.plot([low, high], [low, high], linestyle="--", linewidth=1.5)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    fig.savefig(output_path, dpi=300)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def patchify(image: np.ndarray, patch_size: int = 8) -> np.ndarray:
    height, width, channels = image.shape
    patches = []
    for row in range(0, height, patch_size):
        for column in range(0, width, patch_size):
            patches.append(
                image[row : row + patch_size, column : column + patch_size].reshape(-1)
            )
    return np.stack(patches, axis=0)


def unpatchify(patches: np.ndarray, patch_size: int = 8) -> np.ndarray:
    token_count, flat_size = patches.shape
    channels = flat_size // (patch_size * patch_size)
    side = int(round(math.sqrt(token_count)))
    image = np.zeros((side * patch_size, side * patch_size, channels), dtype=np.float32)
    index = 0
    for row in range(side):
        for column in range(side):
            image[
                row * patch_size : (row + 1) * patch_size,
                column * patch_size : (column + 1) * patch_size,
            ] = patches[index].reshape(patch_size, patch_size, channels)
            index += 1
    return image


def image_attention_recovery(
    image: np.ndarray,
    model: TTAttentionOperator,
    projection: np.ndarray,
    patch_size: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float, float]:
    patches = patchify(image, patch_size)
    centered = patches - patches.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1, keepdims=True) + 1e-6
    normalized = centered / norms
    embeddings = normalized @ projection
    embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-6)
    scores = embeddings @ embeddings.T * 1.5
    exact_attention = torch.softmax(torch.tensor(scores, dtype=torch.float32), dim=-1)
    with torch.no_grad():
        frost_attention = model(
            scores_to_frost_input(torch.tensor(scores, dtype=torch.float32))
        )
    exact_patches = exact_attention.numpy() @ patches
    frost_patches = frost_attention.numpy() @ patches
    exact_image = np.clip(unpatchify(exact_patches, patch_size), 0.0, 1.0)
    frost_image = np.clip(unpatchify(frost_patches, patch_size), 0.0, 1.0)
    mse = float(np.mean((exact_image - frost_image) ** 2))
    psnr = float(10.0 * np.log10(1.0 / max(mse, 1e-12)))
    ssim = float(
        structural_similarity(
            exact_image,
            frost_image,
            channel_axis=2,
            data_range=1.0,
        )
    )
    attention_rmse = float(
        np.sqrt(np.mean((exact_attention.numpy() - frost_attention.numpy()) ** 2))
    )
    return (
        exact_image,
        frost_image,
        exact_attention.numpy(),
        frost_attention.numpy(),
        psnr,
        ssim,
        attention_rmse,
    )


def create_image_collage(
    rows: list[tuple[str, np.ndarray, np.ndarray]],
    metrics: list[tuple[float, float]],
    output_path: Path,
) -> None:
    scale = 5
    image_size = 32 * scale
    label_width = 150
    header_height = 42
    row_height = image_size + 48
    canvas_width = label_width + 3 * image_size + 4 * 20
    canvas_height = header_height + len(rows) * row_height + 20
    canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 18)
        small_font = ImageFont.truetype("DejaVuSans.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
        small_font = font
    headers = ["Input", "Exact attention", "FROST attention"]
    for column, header in enumerate(headers):
        x = label_width + 20 + column * (image_size + 20)
        draw.text((x, 10), header, fill="black", font=font)
    for row_index, ((label, original, exact, frost), (psnr, ssim)) in enumerate(
        zip(rows, metrics)
    ):
        y = header_height + row_index * row_height
        draw.text((10, y + 45), label, fill="black", font=font)
        draw.text(
            (10, y + 75),
            f"PSNR {psnr:.1f} dB\nSSIM {ssim:.3f}",
            fill="black",
            font=small_font,
        )
        for column, array in enumerate([original, exact, frost]):
            uint8_image = np.clip(array * 255.0, 0, 255).astype(np.uint8)
            tile = Image.fromarray(uint8_image).resize(
                (image_size, image_size), Image.Resampling.NEAREST
            )
            x = label_width + 20 + column * (image_size + 20)
            canvas.paste(tile, (x, y))
    canvas.save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--cifar-samples", type=Path, default=Path("cifar_samples"))
    parser.add_argument("--fast", action="store_true", help="Use shorter training runs")
    args = parser.parse_args()
    output_dir = args.output.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(7)

    input_dim = 8
    n_states = 4
    scalar_steps = 220 if args.fast else 360
    attention_steps = 450 if args.fast else 750

    scalar_results: list[ScalarResult] = []
    scalar_models: dict[tuple[str, int], TTScalarOperator] = {}
    scalar_histories: dict[tuple[str, int], list[float]] = {}
    functions = [
        "Softmax component",
        "Gaussian radial",
        "Multiplicative sinusoid",
        "GELU mean",
    ]
    # Main function benchmark at rank 4.
    for function_index, function_name in enumerate(functions):
        model, history = train_scalar_operator(
            function_name,
            rank=4,
            input_dim=input_dim,
            n_states=n_states,
            steps=scalar_steps,
            seed=20 + function_index,
        )
        rmse, mae = evaluate_scalar_operator(model, function_name)
        scalar_results.append(
            ScalarResult(function_name, 4, rmse, mae, model.parameter_count)
        )
        scalar_models[(function_name, 4)] = model
        scalar_histories[(function_name, 4)] = history

    # Rank ablation for the softmax-type multivariate function.
    for rank in (2, 8):
        model, history = train_scalar_operator(
            "Softmax component",
            rank=rank,
            input_dim=input_dim,
            n_states=n_states,
            steps=scalar_steps,
            seed=40 + rank,
        )
        rmse, mae = evaluate_scalar_operator(model, "Softmax component")
        scalar_results.append(
            ScalarResult("Softmax component", rank, rmse, mae, model.parameter_count)
        )
        scalar_models[("Softmax component", rank)] = model
        scalar_histories[("Softmax component", rank)] = history

    scalar_frame = pd.DataFrame([asdict(result) for result in scalar_results])
    scalar_frame.to_csv(output_dir / "scalar_function_metrics.csv", index=False)

    function_rank4 = scalar_frame[scalar_frame["rank"] == 4]
    save_bar_chart(
        function_rank4["function"].tolist(),
        {"FROST rank 4": function_rank4["rmse"].tolist()},
        "Test RMSE",
        output_dir / "fig1a_multivariate_function_rmse.png",
    )

    softmax_ablation = scalar_frame[
        scalar_frame["function"] == "Softmax component"
    ].sort_values("rank")
    save_line_chart(
        softmax_ablation["rank"].tolist(),
        {"Softmax component": softmax_ablation["rmse"].tolist()},
        "Tensor-train rank",
        "Test RMSE",
        output_dir / "fig1b_rank_ablation.png",
    )

    dimensions = np.array([2, 4, 8, 16, 32, 64])
    dense_parameters = n_states ** dimensions
    rank = 4
    tt_parameters = 2 * n_states * rank + (dimensions - 2) * n_states * rank**2
    save_line_chart(
        dimensions.tolist(),
        {
            "Dense SMURF table": dense_parameters.astype(float).tolist(),
            "FROST tensor train": tt_parameters.astype(float).tolist(),
        },
        "Input dimension",
        "Parameter count",
        output_dir / "fig1c_parameter_scaling.png",
        log_y=True,
    )

    with torch.no_grad():
        slice_x = torch.rand(1800, input_dim)
        slice_target = target_function(slice_x, "Softmax component")
        slice_prediction = scalar_models[("Softmax component", 8)](slice_x)
    save_scatter(
        slice_target.numpy(),
        slice_prediction.numpy(),
        output_dir / "fig1d_softmax_prediction_scatter.png",
    )

    token_count = 16
    attention_results: list[AttentionResult] = []
    attention_models: dict[int, TTAttentionOperator] = {}
    attention_histories: dict[int, list[float]] = {}
    for rank_index, rank in enumerate((4, 8, 12)):
        model, history = train_attention_operator(
            token_count=token_count,
            rank=rank,
            n_states=n_states,
            steps=attention_steps,
            seed=70 + rank_index,
        )
        result = evaluate_attention_operator(model)
        attention_results.append(result)
        attention_models[rank] = model
        attention_histories[rank] = history

    attention_frame = pd.DataFrame([asdict(result) for result in attention_results])
    attention_frame.to_csv(output_dir / "attention_fitting_metrics.csv", index=False)
    save_line_chart(
        attention_frame["rank"].tolist(),
        {"FROST attention": attention_frame["kl_divergence"].tolist()},
        "Tensor-train rank",
        "Mean KL divergence",
        output_dir / "fig2a_attention_kl_vs_rank.png",
    )

    best_model = attention_models[12]
    with torch.no_grad():
        example_scores = make_score_batch(token_count, token_count)
        exact_matrix = torch.softmax(example_scores, dim=-1)
        frost_matrix = best_model(scores_to_frost_input(example_scores))
    save_heatmap(
        exact_matrix.numpy(),
        "Reference softmax attention",
        output_dir / "fig2b_reference_attention_heatmap.png",
    )
    save_heatmap(
        frost_matrix.numpy(),
        "FROST-fitted attention",
        output_dir / "fig2c_frost_attention_heatmap.png",
    )
    save_heatmap(
        np.abs(exact_matrix.numpy() - frost_matrix.numpy()),
        "Absolute attention error",
        output_dir / "fig2d_attention_error_heatmap.png",
    )

    bitstream_lengths = [16, 32, 64, 128, 256]
    bitstream_rmse: list[float] = []
    with torch.no_grad():
        stochastic_scores = make_score_batch(128, token_count)
        stochastic_input = scores_to_frost_input(stochastic_scores)
        expected_output = best_model(stochastic_input)
        for length in bitstream_lengths:
            sampled_output = best_model.stochastic_forward(
                stochastic_input, bitstream_length=length, burn_in=32
            )
            error = torch.sqrt(torch.mean((expected_output - sampled_output) ** 2))
            bitstream_rmse.append(float(error))
    pd.DataFrame(
        {"bitstream_length": bitstream_lengths, "rmse_to_expected": bitstream_rmse}
    ).to_csv(output_dir / "bitstream_ablation_metrics.csv", index=False)
    save_line_chart(
        bitstream_lengths,
        {"Finite-window FROST": bitstream_rmse},
        "Bitstream length",
        "RMSE to expected FROST output",
        output_dir / "fig2e_bitstream_recovery.png",
    )

    sample_dir = args.cifar_samples.resolve()
    image_paths = sorted(sample_dir.glob("*1.png"))
    if not image_paths:
        raise FileNotFoundError(
            f"No CIFAR sample PNGs found in {sample_dir}. "
            "Place 32x32 RGB images there or pass --cifar-samples."
        )
    projection_rng = np.random.default_rng(123)
    projection = projection_rng.normal(size=(8 * 8 * 3, 16)).astype(np.float32)
    projection, _ = np.linalg.qr(projection)
    selected_paths = image_paths[:10]
    collage_rows: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    collage_metrics: list[tuple[float, float]] = []
    image_metric_rows: list[dict[str, float | str]] = []
    example_exact_attention = None
    example_frost_attention = None
    for image_path in selected_paths:
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.float32) / 255.0
        (
            exact_image,
            frost_image,
            exact_attention,
            frost_attention,
            psnr,
            ssim,
            attention_rmse,
        ) = image_attention_recovery(image, best_model, projection)
        label = image_path.stem.rstrip("0123456789")
        collage_rows.append((label, image, exact_image, frost_image))
        collage_metrics.append((psnr, ssim))
        image_metric_rows.append(
            {
                "image": image_path.name,
                "attention_rmse": attention_rmse,
                "attention_output_psnr_db": psnr,
                "attention_output_ssim": ssim,
            }
        )
        if example_exact_attention is None:
            example_exact_attention = exact_attention
            example_frost_attention = frost_attention

    image_metrics_frame = pd.DataFrame(image_metric_rows)
    image_metrics_frame.to_csv(output_dir / "cifar10_attention_recovery_metrics.csv", index=False)
    create_image_collage(
        collage_rows,
        collage_metrics,
        output_dir / "fig3a_cifar10_attention_recovery.png",
    )
    if example_exact_attention is not None and example_frost_attention is not None:
        save_heatmap(
            example_exact_attention,
            "CIFAR-10 reference patch attention",
            output_dir / "fig3b_cifar10_reference_patch_attention.png",
        )
        save_heatmap(
            example_frost_attention,
            "CIFAR-10 FROST patch attention",
            output_dir / "fig3c_cifar10_frost_patch_attention.png",
        )

    torch.save(
        {
            "model_state_dict": best_model.state_dict(),
            "token_count": token_count,
            "n_states": n_states,
            "rank": best_model.rank,
        },
        output_dir / "frost_attention_rank12.pt",
    )

    summary = {
        "experiment_1": {
            "input_dimension": input_dim,
            "states_per_fsm": n_states,
            "functions": functions,
            "scalar_steps": scalar_steps,
        },
        "experiment_2": {
            "token_count": token_count,
            "ranks": [4, 8, 12],
            "attention_steps": attention_steps,
        },
        "experiment_3": {
            "dataset": "Official CIFAR-10 sample images",
            "images_evaluated": len(selected_paths),
            "patch_size": 8,
            "token_count": token_count,
            "interpretation": (
                "Recovery is measured against the exact softmax-attention output, "
                "not against the untransformed input image."
            ),
        },
    }
    (output_dir / "experiment_configuration.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print("\nScalar function metrics\n", scalar_frame.to_string(index=False))
    print("\nAttention fitting metrics\n", attention_frame.to_string(index=False))
    print("\nCIFAR-10 attention recovery metrics\n", image_metrics_frame.to_string(index=False))
    print(f"\nResults written to: {output_dir}")


if __name__ == "__main__":
    main()
