import gc
import math
import os
import random
import warnings
from pathlib import Path
from typing import Any

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from safetensors.torch import load_file
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset
from transformers import AutoConfig

from chronos.chronos2.model import Chronos2Model


warnings.filterwarnings("ignore")
torch._dynamo.disable()
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass
    try:
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def cleanup() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true), eps)
    return float(np.mean(np.abs((y_true - y_pred) / denom)) * 100.0)


def smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    denom = np.maximum(np.abs(y_true) + np.abs(y_pred), eps)
    return float(np.mean(2.0 * np.abs(y_pred - y_true) / denom) * 100.0)


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    yt = y_true.reshape(-1)
    yp = y_pred.reshape(-1)
    err = yp - yt
    return {
        "mae": float(np.mean(np.abs(err))),
        "mape": safe_mape(yt, yp),
        "smape": smape(yt, yp),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
    }


def load_market_dataframe(cfg: dict[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    df = pd.read_csv(cfg["path"], index_col=0, parse_dates=True)
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, ~df.columns.str.startswith("Unnamed")]

    price_col = cfg["price"]
    exog_cols = cfg["exog"]
    required = [price_col] + exog_cols
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    rename_map = {price_col: "Price"}
    exog_names = []
    for idx, col in enumerate(exog_cols, start=1):
        name = f"Exog_{idx}"
        rename_map[col] = name
        exog_names.append(name)

    df = df[required].rename(columns=rename_map).copy()
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.resample("h").mean().interpolate().sort_index()
    if df.isna().any().any():
        raise ValueError("Input data still contains NaN values after preprocessing.")
    return df, exog_names


def split_by_days(df: pd.DataFrame, train_days: int, val_days: int, test_days: int):
    start = df.index.min().floor("D")
    val_start = start + pd.Timedelta(days=train_days)
    test_start = val_start + pd.Timedelta(days=val_days)
    test_end = test_start + pd.Timedelta(days=test_days) - pd.Timedelta(hours=1)

    train_df = df[(df.index >= start) & (df.index < val_start)].copy()
    val_df = df[(df.index >= val_start) & (df.index < test_start)].copy()
    test_df = df[(df.index >= test_start) & (df.index <= test_end)].copy()
    return train_df, val_df, test_df, val_start, test_start, test_end


def compute_diff_stats(price_values: np.ndarray):
    diff = np.concatenate(([0.0], np.diff(price_values))).astype(np.float32)
    diff_mean = float(np.mean(diff))
    diff_std = float(np.std(diff))
    if diff_std < 1e-6:
        diff_std = 1.0
    diff_norm = ((diff - diff_mean) / diff_std).astype(np.float32)
    return diff_mean, diff_std, diff_norm


def ordered_probs(model: GaussianMixture, values: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(values.reshape(-1, 1))
    weights = model.weights_
    first = int(np.argmax(weights))
    second = int(np.argmin(weights))
    third = [idx for idx in range(3) if idx not in (first, second)][0]
    return np.concatenate((probs[:, first:first + 1], probs[:, second:second + 1], probs[:, third:third + 1]), axis=1)


def ordered_segment_probs(model: GaussianMixture, segment: np.ndarray) -> np.ndarray:
    probs = model.predict_proba(segment.reshape(1, -1))
    weights = model.weights_
    first = int(np.argmin(weights))
    second = int(np.argmax(weights))
    third = [idx for idx in range(3) if idx not in (first, second)][0]
    return np.concatenate((probs[:, first:first + 1], probs[:, second:second + 1], probs[:, third:third + 1]), axis=1)[0]


def fit_clustering_models(train_df: pd.DataFrame, random_seed: int, segment_hours: int) -> dict[str, Any]:
    price = train_df["Price"].to_numpy(dtype=np.float32)
    diff_mean, diff_std, diff_norm = compute_diff_stats(price)

    gm3 = GaussianMixture(n_components=3, random_state=random_seed)
    gm3.fit(diff_norm.reshape(-1, 1))

    sample_rng = np.random.default_rng(random_seed + 17)
    sample_size = min(100000, len(diff_norm))
    sampled_idx = sample_rng.integers(0, len(diff_norm), size=sample_size)
    gmm0 = GaussianMixture(n_components=3, random_state=random_seed + 1)
    gmm0.fit(diff_norm[sampled_idx].reshape(-1, 1))

    segments = []
    total_days = len(price) // 24
    for day_idx in range(1, total_days):
        hour_start = day_idx * 24
        hist_start = max(0, hour_start - segment_hours)
        diff_window = diff_norm[hist_start:hour_start]
        if len(diff_window) < segment_hours:
            diff_window = np.concatenate((np.zeros(segment_hours - len(diff_window), dtype=np.float32), diff_window))
        probs = gm3.predict_proba(diff_window.reshape(-1, 1))
        weights = gm3.weights_
        prob_in = probs[:, 0] * weights[0] + probs[:, 1] * weights[1] + probs[:, 2] * weights[2]
        extreme_score = (1.0 - prob_in).astype(np.float32)
        segments.append(extreme_score)

    seg_gmm = GaussianMixture(n_components=3, random_state=random_seed + 2)
    seg_gmm.fit(np.stack(segments).astype(np.float32))
    return {
        "gm3": gm3,
        "gmm0": gmm0,
        "seg_gmm": seg_gmm,
        "diff_mean": diff_mean,
        "diff_std": diff_std,
    }


def build_router_inputs(price_history: np.ndarray, clustering: dict[str, Any], prediction_length: int, segment_hours: int, seq_weight: float):
    diff = np.concatenate(([0.0], np.diff(price_history))).astype(np.float32)
    diff_norm = ((diff - clustering["diff_mean"]) / clustering["diff_std"]).astype(np.float32)

    diff24 = diff_norm[-prediction_length:]
    if len(diff24) < prediction_length:
        diff24 = np.concatenate((np.zeros(prediction_length - len(diff24), dtype=np.float32), diff24))
    diff72 = diff_norm[-segment_hours:]
    if len(diff72) < segment_hours:
        diff72 = np.concatenate((np.zeros(segment_hours - len(diff72), dtype=np.float32), diff72))

    point_probs = ordered_probs(clustering["gmm0"], diff24).astype(np.float32)

    probs72 = clustering["gm3"].predict_proba(diff72.reshape(-1, 1))
    weights = clustering["gm3"].weights_
    prob_in72 = probs72[:, 0] * weights[0] + probs72[:, 1] * weights[1] + probs72[:, 2] * weights[2]
    extreme72 = (1.0 - prob_in72).astype(np.float32)
    segment_probs = ordered_segment_probs(clustering["seg_gmm"], extreme72.reshape(-1)).astype(np.float32)
    segment_tile = np.repeat(segment_probs.reshape(1, -1), prediction_length, axis=0).astype(np.float32)

    router_sequence = point_probs + seq_weight * segment_tile
    router_sequence = np.clip(router_sequence, 1e-6, None)
    router_sequence = (router_sequence / np.sum(router_sequence, axis=1, keepdims=True)).astype(np.float32)

    cluster_prior = router_sequence.mean(axis=0).astype(np.float32)
    cluster_prior = np.clip(cluster_prior, 1e-6, None)
    cluster_prior = (cluster_prior / np.sum(cluster_prior)).astype(np.float32)
    return router_sequence.astype(np.float32), cluster_prior


class ForecastDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        exog_names: list[str],
        clustering: dict[str, Any],
        context_length: int,
        prediction_length: int,
        segment_hours: int,
        seq_weight: float,
        origin_start: pd.Timestamp,
        origin_end: pd.Timestamp,
        stride: int,
    ):
        self.df = df
        self.exog_names = exog_names
        self.clustering = clustering
        self.context_length = context_length
        self.prediction_length = prediction_length
        self.segment_hours = segment_hours
        self.seq_weight = seq_weight
        self.y = df["Price"].values.astype(np.float32)
        self.x = df[exog_names].values.astype(np.float32)
        index = df.index

        min_origin = context_length
        max_origin = len(df) - prediction_length
        start_pos = max(index.searchsorted(origin_start), min_origin)
        end_pos = min(index.searchsorted(origin_end, side="right") - 1, max_origin)

        origins = np.arange(start_pos, end_pos + 1, stride, dtype=np.int64)
        origins = origins[(origins - context_length >= 0) & (origins + prediction_length <= len(df))]
        origins = origins[index[origins].hour == 0]
        if len(origins) == 0:
            raise ValueError("No valid daily origins found for the requested split.")
        self.origins = origins

    def __len__(self) -> int:
        return len(self.origins)

    def __getitem__(self, idx: int):
        origin = int(self.origins[idx])
        past_y = self.y[origin - self.context_length:origin]
        fut_x = self.x[origin:origin + self.prediction_length]
        router_feat, cluster_prior = build_router_inputs(
            past_y,
            self.clustering,
            self.prediction_length,
            self.segment_hours,
            self.seq_weight,
        )
        return {
            "past_y": torch.tensor(past_y, dtype=torch.float32).unsqueeze(0),
            "past_x": torch.tensor(self.x[origin - self.context_length:origin], dtype=torch.float32).T,
            "fut_y": torch.tensor(self.y[origin:origin + self.prediction_length], dtype=torch.float32).unsqueeze(0),
            "fut_x": torch.tensor(fut_x, dtype=torch.float32).T,
            "target_price": torch.tensor(self.y[origin:origin + self.prediction_length], dtype=torch.float32),
            "router_feat": torch.tensor(router_feat, dtype=torch.float32),
            "cluster_prior": torch.tensor(cluster_prior, dtype=torch.float32),
            "origin_ns": np.int64(self.df.index[origin].value),
        }


def collate_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    contexts, future_covariates, group_ids = [], [], []
    target_price, router_feat, cluster_prior = [], [], []
    n_vars = batch[0]["past_y"].shape[0] + batch[0]["past_x"].shape[0]

    for idx, item in enumerate(batch):
        contexts.append(torch.cat([item["past_y"], item["past_x"]], dim=0))
        future_covariates.append(torch.cat([torch.full_like(item["fut_y"], float("nan")), item["fut_x"]], dim=0))
        group_ids.append(torch.full((n_vars,), idx, dtype=torch.long))
        target_price.append(item["target_price"])
        router_feat.append(item["router_feat"])
        cluster_prior.append(item["cluster_prior"])

    return {
        "context": torch.cat(contexts, dim=0),
        "future_covariates": torch.cat(future_covariates, dim=0),
        "group_ids": torch.cat(group_ids, dim=0),
        "target_price": torch.stack(target_price, dim=0),
        "router_feat": torch.stack(router_feat, dim=0),
        "cluster_prior": torch.stack(cluster_prior, dim=0),
        "n_vars": n_vars,
        "origin_ns": np.array([item["origin_ns"] for item in batch], dtype=np.int64),
    }


def build_chronos_model(model_dir: Path, context_length: int) -> Chronos2Model:
    config = AutoConfig.from_pretrained(model_dir)
    if hasattr(config, "chronos_config"):
        config.chronos_config.pop("tokenizer_class", None)
        config.chronos_config["use_arcsinh"] = True
    model = Chronos2Model(config)
    state = load_file(model_dir / "model.safetensors")
    model.load_state_dict(state, strict=False)
    model = model.to(DEVICE)
    model = model.to(torch.float32)
    model.chronos_config.context_length = context_length
    return model


def select_lora_targets(model: nn.Module, scope: str) -> list[str]:
    selected = []
    for name, module in model.named_modules():
        lname = name.lower()
        is_linear = isinstance(module, nn.Linear)
        is_qkvo = any(lname.endswith(f".{x}") for x in ["q", "k", "v", "o"])
        in_encoder_block = "encoder.block" in lname
        in_time_or_group = (".layer.0." in lname) or (".layer.1." in lname)
        if is_linear and is_qkvo and in_encoder_block and in_time_or_group:
            selected.append(name)
            continue
        if scope in ["attn_plus_head", "attn_plus_head_ffn"]:
            if is_linear and "output_patch_embedding.output_layer" in lname:
                selected.append(name)
                continue
        if scope == "attn_plus_head_ffn":
            if is_linear and in_encoder_block and (lname.endswith(".wi") or lname.endswith(".wo")):
                selected.append(name)
                continue
    selected = sorted(set(selected))
    if len(selected) == 0:
        raise ValueError("No LoRA target modules were found.")
    return selected


def get_output_patch_size(model: nn.Module) -> int:
    for obj in [model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)]:
        if obj is None:
            continue
        cfg = getattr(obj, "chronos_config", None)
        if cfg is not None and hasattr(cfg, "output_patch_size"):
            return int(cfg.output_patch_size)
    raise AttributeError("Cannot read output_patch_size from the model.")


def get_quantiles_tensor(model: nn.Module) -> torch.Tensor:
    for obj in [model, getattr(model, "base_model", None), getattr(getattr(model, "base_model", None), "model", None)]:
        if obj is None:
            continue
        q = getattr(obj, "quantiles", None)
        if q is not None:
            return q
    raise AttributeError("Cannot read quantiles from the model.")


class AttentionRouter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, num_layers: int, prediction_length: int):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, prediction_length, hidden_dim))
        self.layers = nn.ModuleList(
            [
                nn.ModuleDict(
                    {
                        "norm1": nn.LayerNorm(hidden_dim),
                        "attn": nn.MultiheadAttention(hidden_dim, num_heads, dropout=0.1, batch_first=True),
                        "norm2": nn.LayerNorm(hidden_dim),
                        "ffn": nn.Sequential(
                            nn.Linear(hidden_dim, hidden_dim * 2),
                            nn.GELU(),
                            nn.Dropout(0.1),
                            nn.Linear(hidden_dim * 2, hidden_dim),
                        ),
                    }
                )
                for _ in range(num_layers)
            ]
        )
        self.out = nn.Linear(hidden_dim, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        for layer in self.layers:
            z = layer["norm1"](h)
            attn_out, _ = layer["attn"](z, z, z, need_weights=False)
            h = h + attn_out
            h = h + layer["ffn"](layer["norm2"](h))
        return torch.softmax(self.out(h), dim=-1)


class MarketModel(nn.Module):
    def __init__(self, model_dir: Path, params: dict[str, Any], router_input_dim: int, prediction_length: int):
        super().__init__()
        self.experts = nn.ModuleList()
        for _ in range(3):
            base_model = build_chronos_model(model_dir, params["context_length"])
            lora_cfg = LoraConfig(
                r=params["lora_r"],
                lora_alpha=int(round(params["lora_r"] * params["lora_alpha_ratio"])),
                target_modules=select_lora_targets(base_model, params["lora_scope"]),
                lora_dropout=params["lora_dropout"],
                bias="none",
            )
            expert = get_peft_model(base_model, lora_cfg)
            expert.to(DEVICE)
            self.experts.append(expert)

        self.router = AttentionRouter(
            input_dim=router_input_dim,
            hidden_dim=params["router_hidden_dim"],
            num_heads=params["router_num_heads"],
            num_layers=params["router_num_layers"],
            prediction_length=prediction_length,
        ).to(DEVICE)

    def forward(self, batch: dict[str, Any], num_output_patches: int, q50_idx: int, return_expert_preds: bool = False):
        weights = self.router(batch["router_feat"].to(DEVICE))
        preds = []
        for expert in self.experts:
            out = expert(
                context=batch["context"].to(DEVICE),
                future_covariates=batch["future_covariates"].to(DEVICE),
                group_ids=batch["group_ids"].to(DEVICE),
                num_output_patches=num_output_patches,
            )
            pred = out.quantile_preds[::batch["n_vars"], q50_idx, : batch["target_price"].shape[1]]
            preds.append(pred)
        pred_stack = torch.stack(preds, dim=2)
        final_pred = (pred_stack * weights).sum(dim=2)
        if return_expert_preds:
            return final_pred, weights, pred_stack
        return final_pred, weights


def build_dataloaders(
    df: pd.DataFrame,
    exog_names: list[str],
    settings: dict[str, Any],
    params: dict[str, Any],
    clustering: dict[str, Any],
):
    train_df, val_df, test_df, val_start, test_start, test_end = split_by_days(
        df, settings["train_days"], settings["val_days"], settings["test_days"]
    )
    scaler = StandardScaler()
    scaler.fit(train_df[exog_names])

    scaled = df.copy()
    scaled.loc[:, exog_names] = scaler.transform(scaled[exog_names])

    dataset_kwargs = {
        "df": scaled,
        "exog_names": exog_names,
        "clustering": clustering,
        "context_length": params["context_length"],
        "prediction_length": settings["prediction_length"],
        "segment_hours": settings["segment_hours"],
        "seq_weight": params["seq_weight"],
        "stride": settings["stride"],
    }
    train_dataset = ForecastDataset(
        origin_start=scaled.index[params["context_length"]],
        origin_end=val_start - pd.Timedelta(hours=1),
        **dataset_kwargs,
    )
    val_dataset = ForecastDataset(
        origin_start=val_start,
        origin_end=test_start - pd.Timedelta(hours=1),
        **dataset_kwargs,
    )
    test_dataset = ForecastDataset(
        origin_start=test_start,
        origin_end=test_end,
        **dataset_kwargs,
    )

    loader_kwargs = {
        "batch_size": params["batch_size"],
        "collate_fn": collate_batch,
        "num_workers": settings["num_workers"],
        "pin_memory": torch.cuda.is_available(),
        "persistent_workers": False,
    }
    return (
        DataLoader(train_dataset, shuffle=True, **loader_kwargs),
        DataLoader(val_dataset, shuffle=False, **loader_kwargs),
        DataLoader(test_dataset, shuffle=False, **loader_kwargs),
    )


@torch.no_grad()
def evaluate_model(model: MarketModel, loader: DataLoader, prediction_length: int):
    model.eval()
    preds, acts, expert_preds, router_weights, cluster_priors = [], [], [], [], []
    q_levels = get_quantiles_tensor(model.experts[0]).detach().cpu().numpy()
    q50_idx = int(np.argmin(np.abs(q_levels - 0.5)))
    num_output_patches = math.ceil(prediction_length / get_output_patch_size(model.experts[0]))

    for batch in loader:
        pred, weights, pred_stack = model(batch, num_output_patches, q50_idx, return_expert_preds=True)
        preds.append(pred.detach().cpu().numpy())
        acts.append(batch["target_price"].detach().cpu().numpy())
        expert_preds.append(pred_stack.detach().cpu().numpy())
        router_weights.append(weights.detach().cpu().numpy())
        cluster_priors.append(batch["cluster_prior"].detach().cpu().numpy())

    pred = np.concatenate(preds, axis=0)
    act = np.concatenate(acts, axis=0)
    expert_pred = np.concatenate(expert_preds, axis=0)
    router_w = np.concatenate(router_weights, axis=0)
    cluster_prior = np.concatenate(cluster_priors, axis=0)

    pairwise_corr = {}
    for i in range(expert_pred.shape[2]):
        for j in range(i + 1, expert_pred.shape[2]):
            a = expert_pred[:, :, i].reshape(-1)
            b = expert_pred[:, :, j].reshape(-1)
            pairwise_corr[f"expert{i}_expert{j}"] = float(np.corrcoef(a, b)[0, 1])

    router_day = np.mean(router_w, axis=1)
    argmax_counts = np.bincount(np.argmax(router_day, axis=1), minlength=3).astype(np.float32)
    argmax_share = (argmax_counts / max(float(np.sum(argmax_counts)), 1.0)).tolist()

    diagnostics = {
        "avg_router_weights": np.mean(router_w, axis=(0, 1)).reshape(-1).tolist(),
        "avg_cluster_prior": np.mean(cluster_prior, axis=0).reshape(-1).tolist(),
        "overall_avg_router_argmax_share": argmax_share,
        "pairwise_corr": pairwise_corr,
        "expert_metrics": [compute_metrics(act, expert_pred[:, :, i]) for i in range(expert_pred.shape[2])],
    }
    return {"metrics": compute_metrics(act, pred), "diagnostics": diagnostics}


def train_one_market(
    market_name: str,
    market_cfg: dict[str, Any],
    settings: dict[str, Any],
    params: dict[str, Any],
    model_dir: Path,
    log_fn,
    run_dir: Path | None = None,
):
    seed_everything(settings["random_seed"])
    cleanup()

    df, exog_names = load_market_dataframe(market_cfg)
    train_df, _, _, _, _, _ = split_by_days(df, settings["train_days"], settings["val_days"], settings["test_days"])
    clustering = fit_clustering_models(train_df, settings["random_seed"], settings["segment_hours"])
    train_loader, val_loader, test_loader = build_dataloaders(df, exog_names, settings, params, clustering)

    sample_batch = next(iter(train_loader))
    router_input_dim = sample_batch["router_feat"].shape[2]
    model = MarketModel(model_dir, params, router_input_dim, settings["prediction_length"])

    router_params = [p for p in model.router.parameters() if p.requires_grad]
    expert_params = []
    for expert in model.experts:
        expert_params.extend([p for p in expert.parameters() if p.requires_grad])
    optimizer = AdamW(
        [
            {"params": expert_params, "lr": params["expert_lr"]},
            {"params": router_params, "lr": params["router_lr"]},
        ]
    )
    scheduler = (
        ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=params["scheduler_factor"],
            patience=params["scheduler_patience"],
            min_lr=settings["min_lr"],
        )
        if params["use_scheduler"]
        else None
    )

    prediction_length = settings["prediction_length"]
    q_levels = get_quantiles_tensor(model.experts[0]).detach().cpu().numpy()
    q50_idx = int(np.argmin(np.abs(q_levels - 0.5)))
    num_output_patches = math.ceil(prediction_length / get_output_patch_size(model.experts[0]))

    best_val = float("inf")
    best_epoch = -1
    best_state = None
    wait = 0

    for epoch in range(settings["max_epochs"]):
        model.train()
        total_loss = 0.0
        num_batches = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            pred, weights, pred_stack = model(batch, num_output_patches, q50_idx, return_expert_preds=True)
            target = batch["target_price"].to(DEVICE)
            cluster_prior = batch["cluster_prior"].to(DEVICE)
            fusion_loss = nn.L1Loss()(pred, target)
            entropy = -(weights * torch.log(weights + 1e-8)).sum(dim=1).mean()
            mean_weights = weights.mean(dim=1)
            cluster_align = torch.sum(
                cluster_prior * (torch.log(cluster_prior + 1e-8) - torch.log(mean_weights + 1e-8)),
                dim=1,
            ).mean()
            expert_abs_err = torch.abs(pred_stack - target.unsqueeze(-1))
            expert_mae_per_sample = expert_abs_err.mean(dim=1)
            weighted_expert_loss = torch.sum(cluster_prior * expert_mae_per_sample, dim=1).mean()
            expert_weight = (
                params["warm_expert_weight"]
                if epoch < params["warm_expert_epochs"]
                else params["post_warm_expert_weight"]
            )
            loss = fusion_loss + 1e-3 * entropy + params["cluster_prior_weight"] * cluster_align + expert_weight * weighted_expert_loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"{market_name}: non-finite training loss")
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            num_batches += 1

        val_pack = evaluate_model(model, val_loader, prediction_length)
        val_mae = val_pack["metrics"]["mae"]
        if scheduler is not None:
            scheduler.step(val_mae)

        log_fn(
            f"[{market_name} E{epoch:02d}] loss={total_loss / max(num_batches, 1):.4f} "
            f"val_mae={val_mae:.4f} best_val={best_val:.4f}"
        )

        if val_mae < best_val:
            best_val = val_mae
            best_epoch = epoch
            wait = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= settings["patience"]:
                log_fn(f"[{market_name}] EARLY_STOP epoch={epoch}")
                break

    if best_state is None:
        raise RuntimeError(f"{market_name}: best checkpoint is empty")
    model.load_state_dict(best_state)

    checkpoint_path = ""
    if run_dir is not None:
        checkpoint_dir = run_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_file = checkpoint_dir / f"{market_name.lower()}_best.pt"
        torch.save(
            {
                "market": market_name,
                "best_epoch": best_epoch,
                "val_mae": best_val,
                "model_params": dict(params),
                "settings": dict(settings),
                "state_dict": best_state,
            },
            checkpoint_file,
        )
        checkpoint_path = str(checkpoint_file)

    test_pack = evaluate_model(model, test_loader, prediction_length)
    result = {
        "market": market_name,
        "best_epoch": best_epoch,
        "val_mae": best_val,
        "test_mae": test_pack["metrics"]["mae"],
        "test_mape": test_pack["metrics"]["mape"],
        "test_smape": test_pack["metrics"]["smape"],
        "test_rmse": test_pack["metrics"]["rmse"],
        "diagnostics": test_pack["diagnostics"],
        "checkpoint_path": checkpoint_path,
    }

    log_fn(
        f"[{market_name}] ROUTER avg_weights={[round(x, 4) for x in result['diagnostics']['avg_router_weights']]} "
        f"avg_cluster_prior={[round(x, 4) for x in result['diagnostics']['avg_cluster_prior']]} "
        f"argmax_share={[round(x, 4) for x in result['diagnostics']['overall_avg_router_argmax_share']]}"
    )
    log_fn(
        f"[{market_name}] EXPERT_CORR "
        + " ".join(f"{k}={v:.4f}" for k, v in result["diagnostics"]["pairwise_corr"].items())
    )

    del model, optimizer
    cleanup()
    return result
