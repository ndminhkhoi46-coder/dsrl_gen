import os
import sys

# Force deterministic hashing and single-thread BLAS
import env_setup
env_setup.set_deterministic_env()

# Set DSRL dataset path before importing dsrl to avoid broken symlink issue
os.environ['DSRL_DATASET_DIR'] = os.path.expanduser('~/scratchDirectory/dataset/dsrl-dot-folder')

import dsrl
import numpy as np
import argparse

# Import refactored modules
from utils import set_global_seed, generate_safe_IL_datasets
from discriminator import JAXDiscriminatorTrainer

class Config:
    def __init__(self, **kwargs):
        self.few_traj_num = kwargs.get('few_traj_num', 5)
        self.expert_traj_num = kwargs.get('expert_traj_num', 200)
        self.suboptimal_traj_num = kwargs.get('suboptimal_traj_num', 200)
        self.medium = kwargs.get('medium', False)
        self.num_iterations = kwargs.get('num_iterations', 100)
        self.batch_size = kwargs.get('batch_size', 256)
        self.seed = kwargs.get('seed', 42)
        self.learning_rate = kwargs.get('learning_rate', 3e-4)

def generate_traj_indices(data_dict):
    terminals = data_dict['terminals'].flatten()
    timeouts = data_dict['timeouts'].flatten()
    ends = np.where(terminals | timeouts)[0]

    shifted_ends = np.append(-1, ends)
    traj_lengths = np.diff(shifted_ends)

    traj_indices = np.repeat(np.arange(len(traj_lengths), dtype=np.int32), traj_lengths)
    
    return traj_indices

def dimensionality_reduction(few_data_dict, mixture_data_dict, pca_dim=2):
    from utils import pca_fit, pca_pred
    pca_obs = pca_fit(few_data_dict['observations'], mixture_data_dict['observations'], n_components=pca_dim)
    pca_actions = pca_fit(few_data_dict['actions'], mixture_data_dict['actions'], n_components=pca_dim)

    combined_few = np.concatenate([pca_pred(few_data_dict['observations'], pca_obs, pca_dim), pca_pred(few_data_dict['actions'], pca_actions, pca_dim)], axis=-1)
    combined_mixture = np.concatenate([pca_pred(mixture_data_dict['observations'], pca_obs, pca_dim), pca_pred(mixture_data_dict['actions'], pca_actions, pca_dim)], axis=-1)
    
    return combined_few, combined_mixture

def main_jax():
    parser = argparse.ArgumentParser(description="Minimal BCE Discriminator Training with Generation")
    parser.add_argument("--env_name", type=str, default="OfflinePointCircle1Gymnasium-v0")
    parser.add_argument("--few_traj_num", type=int, default=1)
    parser.add_argument("--expert_traj_num", type=int, default=200)
    parser.add_argument("--suboptimal_traj_num", type=int, default=1000)
    parser.add_argument("--num_iterations", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--medium", action="store_true")
    args = parser.parse_args()

    master_key = set_global_seed(args.seed)

    if 'Metadrive' in args.env_name:
        import gym
    else:
        import gymnasium as gym

    env = gym.make(args.env_name)
    
    config = {
        'few_traj_num': args.few_traj_num,
        'expert_traj_num': args.expert_traj_num,
        'suboptimal_traj_num': args.suboptimal_traj_num,
        'medium': args.medium
    }

    print("Generating suboptimal dataset...")
    few_data_dict, mixture_data_dict, safety_data = generate_safe_IL_datasets(env, config)

    pca_dim = 2 
    combined_few, combined_mixture = dimensionality_reduction(few_data_dict, mixture_data_dict, pca_dim = pca_dim)
    combined_labels = np.concatenate([np.ones(len(combined_few), dtype=np.int32), mixture_data_dict['is_expert']])
    
    state_dim = min(pca_dim, env.observation_space.shape[0])
    action_dim = min(pca_dim, env.action_space.shape[0])

    trainer = JAXDiscriminatorTrainer(
        dim_tuple=(state_dim, action_dim, 256),
        optim_tuple=(3e-4, 10, 0.99),
        loss_tuple=(1.0, 10.0),
        use_wandb=False
    )
    
    print(f"Starting training on Few ({len(combined_few)}) and Mixture ({len(combined_mixture)}) data...")

    trainer.train(
        combined_few, combined_mixture, 
        offline_indices = generate_traj_indices(mixture_data_dict),
        combined_labels = combined_labels,
        num_iterations=args.num_iterations, batch_size=args.batch_size
    )

    print("Training complete.")

    # Prediction example
    print("Testing predictions...")
    all_data = np.concatenate([combined_few, combined_mixture], axis=0)
    rewards = trainer.predict_rewards(all_data)
    print(f"Predicted rewards range: [{rewards.min():.4f}, {rewards.max():.4f}]")
    print(f"Average predicted reward: {rewards.mean():.4f}")

if __name__ == "__main__":
    main_jax()