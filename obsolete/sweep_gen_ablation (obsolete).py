# NOTE: Either to tune for all env or tune a hyperparameter that works across all trajectories

import wandb
import copy
import os
import sys
from datetime import datetime
from collections import defaultdict
from pathlib import Path

# Add current and parent directories to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path = list(dict.fromkeys(sys.path + [current_dir, parent_dir]))

# ==========================================
# 1. GLOBAL CONFIGURATION
# ==========================================
PROJECT_NAME = "Scratchpad"
CMD_ROOT = "/common/home/users/d/dmk.nguyen.2024/*projects/*neurips/dsrl_gen/main.py"
# Added minutes and seconds to ensure unique output files
OUTPUT_FILE = f"dsrl_disc_ablation_{datetime.now().strftime('%d-%m-%H-%M-%S')}.txt"

BASE_CONFIG = {
    "name": "dsrl_disc_ablation_shared_hyperaram",
    "method": "grid",
    "metric": {"goal": "maximize", "name": "checkpoint/best_f1_is_expert"},
    "parameters": {
        "num_iterations": {"values": [10_000]},
        "decay_steps": {"values": [10]},
        "seed": {"values": [42]},
        "batch_size": {"values": [200]},
        "quantile": {"values": [0.85]},
        "safety_traj_num": {"values": [200]},
    },
    "command": [
        "${env}", "python", CMD_ROOT,
        "${args_no_hyphens}"
    ]
}

# ==========================================
# 2. ENVIRONMENT DEFINITIONS
# ==========================================

# Baseline hyperparameter templates per environment cluster
BULLET_HYPERPARAMETERS = {"learning_rate": 0.0001, "decay_rate": 0.85}
METADRIVE_HYPERPARAMETERS = {"learning_rate": 0.01, "decay_rate": 0.95}
SAFETY_HYPERPARAMETERS = {"learning_rate": 0.0003, "decay_rate": 0.8}

ENV_SETTINGS = {
    "BulletGym": {
        "envs": [
            "OfflineCarRun-v0",
            "OfflineBallCircle-v0", "OfflineAntRun-v0", 
            "OfflineCarCircle-v0", "OfflineBallRun-v0"
        ],
        **BULLET_HYPERPARAMETERS
    },
    "MetaDrive": {
        "envs": [
            "OfflineMetadrive-easymean-v0", "OfflineMetadrive-easydense-v0",
            "OfflineMetadrive-easysparse-v0", "OfflineMetadrive-mediummean-v0",
            "OfflineMetadrive-hardmean-v0", "OfflineMetadrive-hardsparse-v0",
            "OfflineMetadrive-mediumsparse-v0", "OfflineMetadrive-harddense-v0",
            "OfflineMetadrive-mediumdense-v0"
        ],
        **METADRIVE_HYPERPARAMETERS
    },
    "SafetyGym": {
        "envs": [
            "OfflinePointGoal2Gymnasium-v0", "OfflinePointCircle2Gymnasium-v0",
            "OfflinePointGoal1Gymnasium-v0", "OfflinePointCircle1Gymnasium-v0"
        ],
        **SAFETY_HYPERPARAMETERS
    }
}

# ==========================================
# 3. ABLATION TYPES
# ==========================================

# Baseline ablation parameters used as a foundation for merged sweeps
BASE_ABLATION = {"few_traj_num": [1], "expert_traj_num": [100]}

ABLATIONS = {
    "replicate": {
        "params": {param: {"values": val} for param, val in BASE_ABLATION.items()}
    },
    # [50, 100, 200] requires different quantile, tune quantile
    # "expert-traj-num": {
    #     "params": {param: {"values": val} for param, val in {**BASE_ABLATION, "suboptimal_traj_num": [50, 100, 200]}.items()}
    # },
}

# ==========================================
# 4. SWEEP GENERATOR
# ==========================================
from sweep_utils import initialize_task_sweeps

def sweep_task_generator(env_group, group_cfg, ablation_name, ablation_cfg):
    """Build a single DSRL sweep task dict."""
    env_prefix = ["${env}", "USE_GYMNASIUM=0"] if env_group == "MetaDrive" else ["${env}"]
    ablation_params = {k: v["values"] for k, v in ablation_cfg["params"].items()}
    
    cfg_copy = {k: v for k, v in group_cfg.items() if k != "envs"}
    return {
        "name": f"{BASE_CONFIG['name']}_{env_group.lower()}_{ablation_name}",
        "params": {
            "env_name": group_cfg["envs"],
            **cfg_copy,
            **ablation_params
        },
        "command": env_prefix + ["python", CMD_ROOT, "${args}"]
    }

def run():
    print(f"Generating sweeps and saving to {OUTPUT_FILE}...")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    
    tasks = [sweep_task_generator(eid, ecfg, aname, acfg) 
             for eid, ecfg in ENV_SETTINGS.items()
             for aname, acfg in ABLATIONS.items()]
             
    initialize_task_sweeps({"General": tasks}, {}, BASE_CONFIG, PROJECT_NAME, OUTPUT_FILE)

    print("-" * 40 + f"\nDone. Sweep IDs saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run()
    
# cd /common/home/users/d/dmk.nguyen.2024/*projects/*neurips/dsrl_gen/ && python sweep_gen_ablation.py