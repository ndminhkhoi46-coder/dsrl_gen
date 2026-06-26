import json
import os

# Force deterministic hashing and single-thread BLAS to avoid order-dependent nondeterminism
import env_setup
env_setup.set_deterministic_env()

import sys
import time
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import wandb
from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict

# Fix system path for local imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path = list(dict.fromkeys(sys.path + [current_dir, parent_dir]))

import main as trainer
import utils
from sweep_utils import (
    FULL_ENV_GROUP_ENVS,
    DSRL_PILOT_ENV_GROUP_ENVS,
    run_group_experiment,
)

class DiscriminatorGroupConfig(trainer.Config):
    # Group execution fields
    env_group: str = "SafetyGym"
    mode: str = "full" 
    project: str = "Scratchpad"
    tag: str = "sync"
    top_k_params: Optional[str] = None
    use_wandb: bool = False
    disable_model_and_data_saving: bool = True 
    early_stop_threshold: float = 50.0
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    @classmethod
    def from_cli(cls):
        # Args containing -- so we have to clean this up
        raw_conf = OmegaConf.from_cli()
        conf_dict = OmegaConf.to_container(raw_conf, resolve=True)
        config = {k.lstrip('-'): v for k, v in conf_dict.items()}
        # Instantiate and run the top-k unpacker
        instance = cls(**config)
        return instance.unpack_top_k_params()

    def unpack_top_k_params(self) -> "DiscriminatorGroupConfig":
        if not self.top_k_params:
            return self
        overrides = json.loads(self.top_k_params)
        field_names = set(getattr(self, "model_fields", getattr(self, "__fields__", {})))
        updates = {k: v for k, v in overrides.items() if k in field_names}
        return self.model_copy(update=updates) if hasattr(self, "model_copy") else self.copy(update=updates)

def run_for_config(config: DiscriminatorGroupConfig):
    config.use_wandb = False
    print(f"Starting Discriminator training for {config.env_name}...")
    
    env_name = config.env_name
    seed = config.seed
    
    if env_name and "Metadrive" in env_name:
        import gym
    else:
        import gymnasium as gym
        
    env = gym.make(env_name)
    for space in [env.action_space, env.observation_space]:
        if hasattr(space, "seed"):
            space.seed(seed)
            
    master_key = utils.seed_everything(seed)
    
    utils.huggingface_login()
    
    few_data, mixture_data, safety_data = trainer.utils.generate_safe_IL_datasets(env, utils._config_to_dict(config))
    optim_tuple = (config.num_iterations, config.learning_rate, config.decay_steps, config.decay_rate, config.batch_size)
    mixture_data, few_data, best_state, info_dict, additional_data = trainer.train_discriminator(
        data_tuple=(few_data, mixture_data),
        optim_tuple=optim_tuple,
        config=utils._config_to_dict(config),
        rng_key=master_key,
        use_wandb=False,
        additional_data={'safety': safety_data},
        quantile=config.quantile,
    )
    env.close()
    return info_dict

def run_experiment(config: DiscriminatorGroupConfig):
    # Map early stop threshold directly based on env_group
    threshold_map = {
        "all": 50.0,
        "SafetyGym": 60.0,
        "BulletGym": 60.0,
        "MetaDrive": 50.0
    }
    config.early_stop_threshold = threshold_map.get(config.env_group, config.early_stop_threshold)

    utils.huggingface_login()
    run_group_experiment(
        config=config,
        env_map=DSRL_PILOT_ENV_GROUP_ENVS if config.mode == "pilot" else FULL_ENV_GROUP_ENVS,
        seed_list=[config.seed],
        run_func=run_for_config,
        name_prefix=f"disc-{config.tag}",
        early_stop_metric="precision",
        early_stop_threshold=config.early_stop_threshold
    )

if __name__ == "__main__":
    config = DiscriminatorGroupConfig.from_cli()
    run_experiment(config)

# python "/common/home/users/d/dmk.nguyen.2024/*projects/*neurips/dsrl_gen/group_run.py"