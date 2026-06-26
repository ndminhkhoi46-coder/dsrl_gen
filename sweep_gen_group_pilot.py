# CAUTION: MAKE SURE MODE IS ALWAYS PILOT, OTHERWISE WOULD BE EXTREMELY TIME-WASTING
# LESS HYPERPARAMETERS TO SEARCH IS MORE IN SEARCH QUALITY
import wandb
import copy
import os
import sys
from datetime import datetime
from collections import defaultdict
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
OUTPUT_FILE = f"dsrl_group_tuning_{datetime.now().strftime('%d-%m-%H-%M-%S')}.txt"

BASE_CONFIG = {
    "method": "bayes",
    "run_cap": 300, # https://datascience.stackexchange.com/questions/87905/is-there-a-rule-of-thumb-for-a-sufficient-number-of-trials-for-hyperparameter-se
    "metric": {"goal": "maximize", "name": "precision"},
    "early_terminate": {
        "type": "hyperband",
        "min_iter": 3,
        "eta": 2
    },
    "parameters": {
        "num_iterations": {"values": [50_000]}, # Note: train until convergence
        "decay_steps": {"values": [50]}, # Note: decay slowly in order to converge stably
        "seed": {"values": [42]},
        "quantile": {"values": [0.85]}, # optimal F1 for 200 trajs, decreases as traj_num decreases
        "learning_rate": {"values": [i * 10**-j for j in range(1, 6) for i in range(1, 10)]},
        "decay_rate": {"values": [i / 100 for i in range(70, 100, 5)] + [0.99]}, # Note: lower values avoid overfitting due to overtraining.
        # "mode": {"values": ["pilot"]}, # USE PILOT FOR TUNING
        "mode": {"values": ["full"]}, # USE FULL FOR FINAL RESULT
        "disable_model_and_data_saving": {"values": [True]},
    },
    "command": [
        "${env}", "python", CMD_ROOT,
        "${args}"
    ]
}

# ==========================================
# 2. GROUP DEFINITIONS
# ==========================================
ENV_SETTINGS = {
    # "BulletGym": {}, 
    # "MetaDrive": {
    #     "command": ["${env}", "USE_GYMNASIUM=0", "python", CMD_ROOT, "${args}"],
    # }, 
    # "SafetyGym": {},
    "all": {}
}

# ==========================================
# 3. ABLATION TYPES
# ==========================================
BASE_ABLATION = {"few_traj_num": [2], # Unsafe trajectories for discriminator F1, safe trajectories for baselines copmarison, so minimum is always 2
                "expert_traj_num": [200]}

ABLATIONS = {
    "main_result": {
        "params": {param: {"values": val} for param, val in {**BASE_ABLATION}.items()}
    },
    # "params": {param: {"values": val} for param, val in {**BASE_ABLATION, "suboptimal_traj_num": [50, 100, 150, 200]}.items()} #tune for quantile [0.85, 0.9, 0.95] later
}

# ==========================================
# 4. SWEEP GENERATOR
# ==========================================
from sweep_utils import generate_sweeps, initialize_tuning_sweeps

def run():
    print(f"Generating sweeps and saving to {OUTPUT_FILE}...")
    Path(OUTPUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    label_map = {"few_traj_num": "few", "expert_traj_num": "exp"}
    sweeps = generate_sweeps(ABLATIONS, ENV_SETTINGS, label_map)
    initialize_tuning_sweeps(sweeps, ENV_SETTINGS, BASE_CONFIG, PROJECT_NAME, OUTPUT_FILE)

    print("-" * 40 + f"\nDone. Saved to {OUTPUT_FILE}")

if __name__ == "__main__":
    run()
    
# python "/common/home/users/d/dmk.nguyen.2024/*projects/*neurips/dsrl_gen/sweep_gen_group_pilot.py"