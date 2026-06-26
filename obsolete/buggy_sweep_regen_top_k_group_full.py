import wandb

import os
import sys
import argparse
from datetime import datetime
from pathlib import Path

# Add parent directory to sys.path to find sweep_utils
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path = list(dict.fromkeys(sys.path + [current_dir, parent_dir]))

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
PROJECT_NAME = "Scratchpad"
CMD_ROOT = "/common/home/users/d/dmk.nguyen.2024/*projects/*neurips/dsrl_gen/group_run.py"
OUTPUT_FILE = f"dsrl_group_distilled_{datetime.now().strftime('%d-%m-%H-%M-%S')}.txt"

# Provided sweep IDs and names
SWEEP_LIST = [
    ("16luqtqt", "bulletgym_pilot"),
    ("b17cwld9", "metadrive_pilot"),
    ("iyz0zvfy", "safetygym_pilot"),
]

TUNABLE_PARAMS = ["learning_rate", "decay_rate"]
METRIC = "f1"

BASE_CONFIG = {
    "method": "grid", # Switch to grid for distilled params
    "metric": {"goal": "maximize", "name": "f1"},
    "parameters": {
        "num_iterations": {"values": [100000]},
        "decay_steps": {"values": [10]},
        "seed": {"values": [42]},
        "batch_size": {"values": [200]},
        "quantile": {"values": [0.85]},
        "safety_traj_num": {"values": [200]},
        "mode": {"values": ["full"]},
    },
    "command": [
        "${env}", "python", CMD_ROOT,
        "${args}"
    ]
}

ENV_SETTINGS = {
    "BulletGym": {},
    "Metadrive": {},
    "SafetyGym": {}
}

BASE_ABLATION = {"expert_traj_num": [2], "traj_num": [200]}

ABLATIONS = {
    "replicate": {
        "params": {param: {"values": val} for param, val in BASE_ABLATION.items()}
    },
    "one-expert": {
        "params": {param: {"values": val} for param, val in {**BASE_ABLATION, "expert_traj_num": [1]}.items()}
    },
    "expert-traj-num": {
        "params": {param: {"values": val} for param, val in {**BASE_ABLATION, "traj_num": [50, 100, 150]}.items()}
    },
}

# ==========================================
# 2. DISTILLATION HELPER
# ==========================================

from sweep_utils import initialize_distilled_sweeps

def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, default=10, help="Top-k hyperparams per sweep")
    parser.add_argument("--project", type=str, default="Scratchpad", help="Target WandB project")
    args = parser.parse_args()
    
    api = wandb.Api()
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    initialize_distilled_sweeps(
        api=api,
        sweep_list=SWEEP_LIST,
        project_name=args.project,
        metric=METRIC,
        tunable_params=TUNABLE_PARAMS,
        base_config=BASE_CONFIG,
        output_file=OUTPUT_FILE,
        k=args.k,
    )

    print("-" * 40 + f"\nDone. Distilled {len(SWEEP_LIST)} sweeps.")

if __name__ == "__main__":
    run()

# cd ~/*projects/*neurips/dsrl_gen
# python sweep_regen_top_k_group_full.py --k 5 --project Scratchpad
# https://stats.stackexchange.com/questions/160479/practical-hyperparameter-optimization-random-vs-grid-search/209409#209409
# TO-DO: Some hyperparameters are not being included in the distilled sweeps (e.g. decay_steps)