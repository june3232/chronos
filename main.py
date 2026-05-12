from pathlib import Path

from src.config import MARKETS, MODEL_DIR, MODEL_PARAMS, OUTPUT_DIR, SETTINGS
from src.core import DEVICE, seed_everything, train_one_market

import pandas as pd


RUN_NAME = f"release_run_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}"
RUN_DIR = OUTPUT_DIR / RUN_NAME
LOG_PATH = RUN_DIR / "run.log"
SUMMARY_PATH = RUN_DIR / "summary.csv"


def log(message: str) -> None:
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    line = f"[{pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def main() -> None:
    seed_everything(SETTINGS["random_seed"])
    RUN_DIR.mkdir(parents=True, exist_ok=True)

    log(f"device={DEVICE}")
    log(f"model_dir={MODEL_DIR}")
    log(f"model_params={MODEL_PARAMS}")
    rows = []

    for market, cfg in MARKETS.items():
        log("=" * 80)
        log(f"RUN {market}: {cfg['path']}")
        result = train_one_market(market, cfg, SETTINGS, MODEL_PARAMS, MODEL_DIR, log, RUN_DIR)
        rows.append(result)
        log(
            f"[{market}] TEST "
            f"MAE={result['test_mae']:.4f} "
            f"MAPE={result['test_mape']:.4f}% "
            f"sMAPE={result['test_smape']:.4f}% "
            f"RMSE={result['test_rmse']:.4f} "
            f"best_epoch={result['best_epoch']}"
        )
        if result["checkpoint_path"]:
            log(f"[{market}] checkpoint={result['checkpoint_path']}")

    summary = pd.DataFrame(
        [
            {
                "market": r["market"],
                "best_epoch": r["best_epoch"],
                "val_mae": r["val_mae"],
                "test_mae": r["test_mae"],
                "test_mape": r["test_mape"],
                "test_smape": r["test_smape"],
                "test_rmse": r["test_rmse"],
                "checkpoint_path": r["checkpoint_path"],
            }
            for r in rows
        ]
    )
    summary.to_csv(SUMMARY_PATH, index=False)

    log("=" * 80)
    log("SUMMARY")
    for _, row in summary.iterrows():
        log(
            f"{row['market']}: "
            f"MAE={row['test_mae']:.4f}, "
            f"MAPE={row['test_mape']:.4f}%, "
            f"sMAPE={row['test_smape']:.4f}%, "
            f"RMSE={row['test_rmse']:.4f}"
        )
    log(f"summary_csv={SUMMARY_PATH}")


if __name__ == "__main__":
    main()
