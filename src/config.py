import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODEL_DIR = Path(os.environ.get("CHRONOS_MODEL_DIR", PROJECT_ROOT / "models" / "chronos-2-local"))
OUTPUT_DIR = Path(os.environ.get("CHRONOS_MCANN_OUTPUT_DIR", PROJECT_ROOT / "artifacts" / "runs"))


MARKETS = {
    "BE": {
        "path": DATA_DIR / "BE.csv",
        "price": "Prices",
        "exog": ["Generation forecast", "System load forecast"],
    },
    "DE": {
        "path": DATA_DIR / "DE.csv",
        "price": "Price",
        "exog": ["Ampirion Load Forecast", "PV+Wind Forecast"],
    },
    "FR": {
        "path": DATA_DIR / "FR.csv",
        "price": "Prices",
        "exog": ["Generation forecast", "System load forecast"],
    },
    "NP": {
        "path": DATA_DIR / "NP.csv",
        "price": "Price",
        "exog": ["Grid load forecast", "Wind power forecast"],
    },
    "PJM": {
        "path": DATA_DIR / "PJM.csv",
        "price": "Zonal COMED price",
        "exog": ["System load forecast", "Zonal COMED load foecast"],
    },
}


SETTINGS = {
    "train_days": 1162,
    "val_days": 294,
    "test_days": 728,
    "prediction_length": 24,
    "stride": 24,
    "max_epochs": 40,
    "patience": 8,
    "min_lr": 1e-6,
    "num_workers": 0,
    "random_seed": 42,
    "segment_hours": 72,
}


MODEL_PARAMS = {
    "context_length": 672,
    "expert_lr": 2.0e-4,
    "router_lr": 3.0e-5,
    "batch_size": 96,
    "lora_scope": "attn_plus_head_ffn",
    "lora_r": 16,
    "lora_alpha_ratio": 8.0,
    "lora_dropout": 0.0,
    "use_scheduler": True,
    "scheduler_factor": 0.7,
    "scheduler_patience": 1,
    "lambda_ent": 1.0e-3,
    "lambda_expert": 1.0,
    "seq_weight": 0.5,
    "warm_expert_epochs": 5,
    "warm_expert_weight": 0.3,
    "post_warm_expert_weight": 0.05,
    "router_hidden_dim": 64,
    "router_num_heads": 2,
    "router_num_layers": 1,
}
