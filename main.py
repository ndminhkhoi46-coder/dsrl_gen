import os
import env_setup
env_setup.set_deterministic_env()

# Standard library
import random as pyrandom
from functools import partial
from typing import Any, Callable

# Third-party libraries
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import serialization
from flax.training import train_state
from jax import grad, jit, random, vmap
from omegaconf import OmegaConf

# Environment and Project Specific
import gymnasium as gym
# Set DSRL dataset path before importing dsrl to avoid broken symlink issue
os.environ['DSRL_DATASET_DIR'] = os.path.expanduser('~/scratchDirectory/dataset/dsrl-dot-folder')
import dsrl
import utils

# Import from OSRL after path setup
from osrl.common.dataset import _parse_trajectories

device = 'cuda:0'

import jax.nn.initializers as init
class DiscriminatorSA(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, input_data):
        
        state = input_data[..., :self.state_dim]
        action = input_data[..., self.state_dim:]
        
        h_s = nn.Dense(self.hidden_dim // 2, name='state_trunk')(state)
        h_a = nn.Dense(self.hidden_dim // 2, name='action_trunk')(action)
        h = jnp.concatenate([h_s, h_a], axis=-1)
        h = nn.tanh(h)

        h = nn.Dense(self.hidden_dim, name='trunk')(h)
        h = nn.tanh(h)
        
        h = nn.Dense(1, name='trunk2')(h)
        return h

def create_train_state(rng_key, model, input_shape, optim_tuple):
    """Create a training state for the discriminator with exponential learning rate decay."""
    learning_rate, decay_steps, decay_rate = optim_tuple
    params = model.init(rng_key, jnp.ones(input_shape))
    
    # cosine_decay_schedule performed poorly, don't know why
    schedule = optax.exponential_decay(
        init_value=learning_rate,
        transition_steps=decay_steps,
        decay_rate=decay_rate,
        staircase=True
    )

    tx = optax.adamw(learning_rate=schedule)
    
    return train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=tx)

@jit
def binary_cross_entropy_with_log_odds(log_odds, targets, mask=None):
    """JAX implementation of binary cross entropy with log_odds."""
    cross_entropy_pairwise = optax.sigmoid_binary_cross_entropy(log_odds, targets) 
    return jnp.mean(cross_entropy_pairwise) if mask is None else jnp.mean(cross_entropy_pairwise * mask)

def compute_gradient_penalty(state, few_data, mixture_data, lambda_gp, rng_key):
    """Compute gradient penalty for WGAN-GP."""
    batch_size = few_data.shape[0] # Undersampling majority class
    
    # Random interpolation
    rng_key, subkey = random.split(rng_key)
    alpha = random.uniform(subkey, (batch_size, 1))
    alpha = jnp.broadcast_to(alpha, few_data.shape)
    
    interpolated = alpha * few_data + (1 - alpha) * mixture_data
    
    def discriminator_fn(x):
        return state.apply_fn(state.params, x) # log(d^E/d^Mix)
    
    # Compute gradients
    grad_fn = grad(lambda x: jnp.sum(discriminator_fn(x)))
    gradients = grad_fn(interpolated)
    
    # Gradient penalty
    gradient_norm = jnp.linalg.norm(gradients, ord=2, axis=1)
    gradient_penalty = jnp.mean((gradient_norm - 1.0)**2)
    return lambda_gp * gradient_penalty

@jit
def update_discriminator(state, few_batch, mixture_batch, mixture_labels, rng_key):
    """Update discriminator parameters."""

    def loss_with_gp(params):
        few_log_odds = state.apply_fn(params, few_batch)
        mixture_log_odds = state.apply_fn(params, mixture_batch)

        # DEBUGGING CODE
        # Simple BCE
        few_risk = binary_cross_entropy_with_log_odds(few_log_odds, 1.0)
        mixture_risk =  binary_cross_entropy_with_log_odds(mixture_log_odds, 0.0) 

        # mixture_risk = binary_cross_entropy_with_log_odds(mixture_log_odds, mixture_labels)  # DEBUGGING CODE

        # L_u = binary_cross_entropy_with_log_odds(mixture_log_odds, mixture_labels)
        # L_p = binary_cross_entropy_with_log_odds(few_log_odds, 1.0)
        # L_p_neg = binary_cross_entropy_with_log_odds(few_log_odds, 0.0)

        # mixture_risk = L_u/EXPERT_PROPORTION
        # few_risk = L_p - L_p_neg
        # TODO: PU learning might improve safety, but might be more complicated to implement

        loss = 0.5 * few_risk + 0.5 * mixture_risk

        gradient_penalty = compute_gradient_penalty(state, few_batch, mixture_batch, lambda_gp=10.0, rng_key=rng_key)
        
        total_loss = loss + gradient_penalty
        return total_loss, (loss, few_risk, mixture_risk, gradient_penalty)


    grad_fn = grad(loss_with_gp, has_aux=True)
    grads, (loss, few_loss, mixture_loss, grad_pen) = grad_fn(state.params)
    
    state = state.apply_gradients(grads=grads)
    return state, loss, few_loss, mixture_loss, grad_pen
    
class JAXDiscriminatorTrainer:
    """JAX-based discriminator trainer."""
    
    def __init__(
        self,
        dim_tuple,
        optim_tuple,
        rng_key=None,
        use_wandb=True,
        quantile=0.85,
    ):
        self.state_dim, self.action_dim, self.hidden_dim = dim_tuple
        self.learning_rate, self.decay_steps, self.decay_rate = optim_tuple
        self.use_wandb = use_wandb
        
        self.expert_proportion = 1 - quantile
        self.threshold = 0.0
        
        # Initialize model and training state
        self.model = DiscriminatorSA(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )
        
        # Create training state with learning rate decay
        input_dim = self.state_dim + self.action_dim
        assert rng_key is not None, "RNG key must be provided"
        self.rng_key = rng_key
        self.state = create_train_state(
            rng_key, self.model, (1, input_dim), (self.learning_rate, self.decay_steps, self.decay_rate))
        
        # Cache for concatenated evaluation data (populated on first call)
        self._cached_is_expert_label = None
        self._cached_01_label = None
        
    def evaluate_preds(self, labels, preds, name='', verbose=False):
        from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix   
        prec = precision_score(labels, preds, zero_division=0)
        rec = recall_score(labels, preds, zero_division=0)
        f1 = f1_score(labels, preds, zero_division=0)
        acc = accuracy_score(labels, preds)
        cm = confusion_matrix(labels, preds)

        prec = utils.format_classification_metrics(prec)
        rec = utils.format_classification_metrics(rec)
        f1 = utils.format_classification_metrics(f1)
        acc = utils.format_classification_metrics(acc)
        
        if verbose:
            print(f"{name} | Recall: {rec}% | F1: {f1}% | Precision: {prec}% | Accuracy: {acc}%")
            print(f"Confusion Matrix:")
            print(f"TN: {cm[0][0]} | FP: {cm[0][1]}")
            print(f"FN: {cm[1][0]} | TP: {cm[1][1]}")

        return prec, rec, f1

    def evaluate_discriminator(self, data_tuple, data_dict_tuple):
        few_data, mixture_data = data_tuple
        few_data_dict, mixture_data_dict = data_dict_tuple

        self._cached_is_expert_label = self._cached_is_expert_label if self._cached_is_expert_label is not None else jnp.concatenate([few_data_dict['is_expert'], mixture_data_dict['is_expert']])

        exp_log_odds = self.get_few_log_odds_batch(few_data)
        mixture_log_odds = self.get_few_log_odds_batch(mixture_data)

        self.threshold = utils.get_discriminator_threshold(mixture_log_odds, 1 - self.expert_proportion)
        exp_preds = utils.threshold_log_odds(exp_log_odds, self.threshold)
        mixture_preds = utils.threshold_log_odds(mixture_log_odds, self.threshold)
        self.evaluate_preds(self._cached_is_expert_label, jnp.concatenate([np.ones_like(exp_preds), mixture_preds]), name='Evaluation', verbose=True)

        return mixture_preds

    def train_one_iteration(self, few_data, mixture_data, pred_mode, batch_size):
        # Batch preparation
        self.rng_key, exp_rng, off_rng = jax.random.split(self.rng_key, 3)
        few_batch = few_data[jax.random.choice(exp_rng, few_data.shape[0], shape=(batch_size,), replace=False)]
        mixture_batch_indices = jax.random.choice(off_rng, mixture_data.shape[0], shape=(batch_size,), replace=False)
        mixture_batch = mixture_data[mixture_batch_indices]
        self.state, batch_avg_loss, few_risk, mixture_risk, *_ = update_discriminator(self.state, few_batch, mixture_batch, pred_mode[mixture_batch_indices], self.rng_key)
        print(f"Batch Avg Loss: {batch_avg_loss:.2f} | Few Risk: {few_risk:.2f} | Mixture Risk: {mixture_risk:.2f}")
        return self.state, batch_avg_loss 

    def discriminator_training(self, data_tuple, eval_dict_tuple, num_iterations=1000, batch_size=256):
        (few_data, mixture_data) = data_tuple
        mixture_mode = jnp.zeros(mixture_data.shape[0])
        eval_iteration_num = max(num_iterations // 10, 1)
        losses = []

        # initial_expert_proportion = self.expert_proportion  # snapshot of 1 - args.quantile, used as upper bound
        for iteration in range(num_iterations):
            self.state, batch_avg_loss, *_ = self.train_one_iteration(few_data, mixture_data, mixture_mode, batch_size)
            losses.append(batch_avg_loss.item())
            if (iteration % eval_iteration_num == 0):
                avg_loss = np.mean(losses[-eval_iteration_num:])
                print(f"Iteration {iteration} | Loss: {avg_loss:.2f}")
                self.evaluate_discriminator(data_tuple=data_tuple, data_dict_tuple=eval_dict_tuple)

                if self.use_wandb:
                    wandb.log({
                        'expert_proportion': self.expert_proportion,
                        'avg_loss': avg_loss,
                        'losses': losses,
                    })

        print(f"Final estimated EXPERT_PROPORTION: {self.expert_proportion:.2f}")
        return self.state

    def train(self, few_data, mixture_data, is_expert, num_iterations=1000, batch_size=256,
              few_data_dict=None, mixture_data_dict=None):
        """Train the discriminator."""
        assert few_data.shape[0] > batch_size and mixture_data.shape[0] > batch_size, f"Batch size too large {batch_size}, few data: {few_data.shape[0]}, mixture data: {mixture_data.shape[0]}"

        print('Training discriminator')
        self.state = self.discriminator_training((few_data, mixture_data), (few_data_dict, mixture_data_dict), num_iterations=num_iterations, batch_size=batch_size)          
        # Compute final metrics using last model
        few_log_odds = self.get_few_log_odds_batch(few_data)
        mixture_log_odds = self.get_few_log_odds_batch(mixture_data)
        expert_log_odds_avg = np.concatenate([few_log_odds, mixture_log_odds[is_expert==1]]).mean()
        non_expert_log_odds_avg = mixture_log_odds[is_expert==0].mean()

        few_trajs_label = np.array(few_data_dict['is_expert'])
        mixture_trajs_label = np.array(mixture_data_dict['is_expert'])
        label = np.concatenate([few_trajs_label, mixture_trajs_label])
        
        self.threshold = utils.get_discriminator_threshold(mixture_log_odds, 1 - self.expert_proportion)
        utils.print_statistics(mixture_log_odds, 'mixture log_odds')
        exp_preds = utils.threshold_log_odds(few_log_odds, self.threshold)
        mixture_preds = utils.threshold_log_odds(mixture_log_odds, self.threshold)
        pred = np.concatenate([exp_preds, mixture_preds])

        precision, recall, f1 = self.evaluate_preds(label, pred, 'Mixed Dataset (Final iteration)', verbose=True)

        optimal_quantile = utils.format_classification_metrics(1 - np.mean(mixture_data_dict['is_expert']))
        mixture_non_expert_prediction_proportions = utils.format_classification_metrics(1 - mixture_preds.mean())

        print(f"Threshold: {self.threshold:.2f} | Quantile: {1 - self.expert_proportion} | Mixture Non-Expert Prediction Proportions: {mixture_non_expert_prediction_proportions}% | Optimal Quantile: {optimal_quantile}%")

        if self.use_wandb:
            log_odd_metrics = {
                "final/expert_log_odds_avg": expert_log_odds_avg,
                "final/non_expert_log_odds_avg": non_expert_log_odds_avg,
                "final/optimal_quantile": optimal_quantile,
                "final/checkpoint_quantile": 1 - self.expert_proportion,
            }
            classification_metrics = {
                "final/f1": f1,
                "final/recall": recall,
                "final/precision": precision,
            }
            wandb.log({**log_odd_metrics, **classification_metrics})
        print(f"Training Completed | Dataset F1 (MCR): {f1}%")
        info_dict = {"f1": float(f1), "precision": float(precision), "expert_proportion": float(self.expert_proportion)}
        return self.state, info_dict
    
    def get_few_log_odds_batch(self, input_data, batch_size=8192):
        """Predict log_odds for input data in batches to avoid OOM."""
        num_samples = input_data.shape[0]
        all_log_odds = []
        
        # Process in batches
        for i in range(0, num_samples, batch_size):
            batch = input_data[i:i + batch_size]
            batch_log_odds = self.state.apply_fn(self.state.params, batch) # dFew/dMixture
            all_log_odds.append(np.array(batch_log_odds).flatten())
        log_odds = np.concatenate(all_log_odds)
        return log_odds

def create_discriminator(pca_tuple, optim_tuple, rng_key, use_wandb, quantile):
    print("Creating JAX discriminator...")
    pca_obs, pca_act = pca_tuple
    _, learning_rate, decay_steps, decay_rate, _ = optim_tuple
    return JAXDiscriminatorTrainer(
        dim_tuple=(pca_obs.n_components_, pca_act.n_components_, 256),
        optim_tuple=(learning_rate, decay_steps, decay_rate),
        rng_key=rng_key,
        use_wandb=use_wandb,
        quantile=quantile,
    )

def report_metrics(jax_discriminator, few_input, mixture_input, transform_tuple, few_data, mixture_data, additional_data, info_dict):
    print(f"Combined Dataset F1: {info_dict['f1']}%")

    few_data['log_odds'] = jax_discriminator.get_few_log_odds_batch(few_input)
    mixture_data['log_odds'] = jax_discriminator.get_few_log_odds_batch(mixture_input)

    safety_data = additional_data['safety']
    safety_input = utils.transform_data((safety_data['observations'], safety_data['actions']), transform_tuple)
    safety_data['log_odds'] = jax_discriminator.get_few_log_odds_batch(safety_input)
    safety_preds = utils.threshold_log_odds(safety_data['log_odds'], jax_discriminator.threshold)

    prec, rec, f1 = jax_discriminator.evaluate_preds(safety_data['is_expert'], safety_preds, 'Safety Dataset', verbose=True)

    if jax_discriminator.use_wandb:
        wandb.log({
            "final_iteration/additional_safety_f1": f1,
            "final_iteration/additional_safety_recall": rec,
            "final_iteration/additional_safety_precision": prec,
        })

    utils.adversarial_validation(mixture_input, safety_input, label_name="Mixture vs. Safe")
    utils.adversarial_validation(mixture_input[mixture_data['is_expert']==0], safety_input[safety_data['is_expert']==0], label_name="Mixture-Suboptimal vs. Safe-Suboptimal")

    utils.print_statistics(mixture_data['rewards'][mixture_data['is_expert']==0], 'rewards mixture suboptimal')
    utils.print_statistics(safety_data['rewards'][safety_data['is_expert']==0], 'rewards safety suboptimal')

    print("JAX discriminator trainer created successfully!")

def train_discriminator(
    data_tuple,
    optim_tuple,
    config,
    rng_key=None,
    use_wandb=True,
    additional_data=None,
    quantile=0.85,
):
    few_data, mixture_data = data_tuple
    num_iterations, learning_rate, decay_steps, decay_rate, batch_size = optim_tuple

    cached_data = utils.load_discriminator(config) if not config['disable_model_and_data_saving'] else None
    if cached_data is not None:
        transform_tuple = cached_data['transform_tuple']
        few_input = utils.transform_data((few_data['observations'], few_data['actions']), transform_tuple)
        mixture_input = utils.transform_data((mixture_data['observations'], mixture_data['actions']), transform_tuple)
    else:
        few_input, mixture_input, transform_tuple = utils.dimension_reduction(few_data, mixture_data, use_wandb=use_wandb)
    
    pca_obs, pca_act, *_ = transform_tuple
    jax_discriminator = create_discriminator((pca_obs, pca_act), optim_tuple, rng_key, use_wandb, quantile)

    if cached_data is not None:
        jax_discriminator.state = jax_discriminator.state.replace(params=cached_data['params'])
        jax_discriminator.threshold = cached_data['threshold']
        info_dict = cached_data['info_dict']
        final_state = jax_discriminator.state
    else:
        print("Training JAX discriminator...")
        if decay_steps is not None: print(f"Using learning rate decay: initial_lr={learning_rate}, decay_steps={decay_steps}, decay_rate={decay_rate}")
        
        final_state, info_dict = jax_discriminator.train(
            few_data=few_input,
            mixture_data=mixture_input,
            is_expert=mixture_data['is_expert'],
            num_iterations=num_iterations,
            batch_size=batch_size,
            few_data_dict=few_data,
            mixture_data_dict=mixture_data,
        )

        if not config['disable_model_and_data_saving']:
            utils.save_discriminator({
                'params': final_state.params,
                'transform_tuple': transform_tuple,
                'info_dict': info_dict,
                'threshold': jax_discriminator.threshold
            }, config)

    report_metrics(jax_discriminator, few_input, mixture_input, transform_tuple, few_data, mixture_data, additional_data, info_dict)

    print("JAX discriminator trainer created successfully!")

    return mixture_data, few_data, final_state, info_dict, additional_data

from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any

class Config(BaseModel):
    model_config = ConfigDict(extra='forbid', populate_by_name=True)

    # Dataset & Environment
    few_traj_num: int = 2
    env_name: str = "OfflinePointCircle1Gymnasium-v0"
    num_iterations: int = 100
    medium: bool = False # Consider bool if you're not passing strings from CLI
    
    # Learning Rate & Optimizer
    learning_rate: float = 5e-4
    decay_steps: int = 10
    decay_rate: float = 0.80
    
    # Experiment Meta
    seed: int = 42
    expert_traj_num: int = 200
    suboptimal_traj_num: int = 1000
    log_dir_base: str = '~/scratchDirectory/logs'
    batch_size: int = 128
    
    # Algorithm Hyperparameters (Tunable)
    quantile: float = 0.85
    safety_traj_num: int = 200
    skip_existing: bool = False
    project: str = "Discriminator-Training"
    use_wandb: bool = True
    disable_model_and_data_saving: bool = False

def main_jax():
    """Modified main function using JAX discriminator."""    

    # Consistent with bcap.py pattern

    raw_conf = OmegaConf.from_cli()
    conf_dict = OmegaConf.to_container(raw_conf, resolve=True)
    clean_dict = {k.lstrip('-'): v for k, v in conf_dict.items()}
    args = Config(**clean_dict)

    # run run_group to check whether main.py printout still leaks
    utils.huggingface_login()
    if not args.disable_model_and_data_saving: time.sleep(60*5) # avoid rate limit
    if args.use_wandb:
        wandb_key = os.environ.get("WANDB_API_KEY", "d1f0e3a57d8d351605c85310bee971c1c532886d")
        wandb.login(key=wandb_key)
        wandb_dir = os.path.expanduser("~/scratchDirectory/wandb")
        os.makedirs(wandb_dir, exist_ok=True)
        wandb_name = f"{args.env_name}_{args.few_traj_num}_few_{args.num_iterations}_iterations_jax"
        wandb.init(
            project=args.project,
            group=str(args.env_name),
            name=wandb_name,
            dir=wandb_dir,
            config=utils._config_to_dict(args)
        )

    seed = args.seed
    env_name = args.env_name
    config = utils._config_to_dict(args)
    print(f"=== Processing Environment: {env_name} ===")

    if env_name and "Metadrive" in env_name:
        import gym
    else:
        import gymnasium as gym
        
    env = gym.make(env_name)
    print(f"Environment {env_name} created successfully.")
    
    master_key = utils.seed_everything(seed)
    few_data, mixture_data, safety_data = utils.generate_safe_IL_datasets(env, config)
    # Use JAX version instead of PyTorch
    optim_tuple = (args.num_iterations, args.learning_rate, args.decay_steps, args.decay_rate, args.batch_size)
    data_tuple = (few_data, mixture_data)
    
    mixture_data, few_data, final_state, info_dict, additional_data = train_discriminator(
        data_tuple=data_tuple,
        optim_tuple=optim_tuple,
        config=config,
        rng_key=master_key,
        additional_data={'safety': safety_data},
        use_wandb=args.use_wandb,
        quantile=args.quantile,
    )

    safety_data = additional_data['safety']
    
    dataset_keys = ['observations', 'actions', 'next_observations', 'rewards', 'costs', 'terminals', 'timeouts', 'is_expert', 'is_safe', 'log_odds', 'traj_index'] # Keep only relevant keys (e.g. Metadrive)
    mixture_data = {k: v for k, v in mixture_data.items() if k in dataset_keys}
    few_data = {k: v for k, v in few_data.items() if k in dataset_keys}
    safety_data = {k: v for k, v in safety_data.items() if k in dataset_keys}
    
    datasets = {"few": few_data, "mixture": mixture_data, "safety": safety_data}
    if not args.disable_model_and_data_saving:
        for name, ds in datasets.items():
            utils.save_dataset(ds, name, config, base_dir='~/scratchDirectory/dataset/dsrl-IL-data')

        data_config = {
            'suboptimal_traj_num': config['suboptimal_traj_num'],
            'env_name': config['env_name'],
            'few_traj_num': config['few_traj_num'],
            'expert_traj_num': config['expert_traj_num'],
            'medium': config.get('medium', False),
        }
        hash_data = utils.dict_hash_ignore_keys(data_config)
        base_path = os.path.expanduser("~/scratchDirectory/dataset/dsrl-IL-data/")
        
        metadata_path = os.path.join(base_path, "metadata.json")
        assert os.path.exists(metadata_path)
        metadata = utils.load_json(metadata_path)
        assert hash_data in metadata
        metadata[hash_data]["EXPERT_PROPORTION"] = float(info_dict['expert_proportion'])
        utils.save_json(metadata_path, metadata)

    env.close()

import time
import os
if __name__ == "__main__":
    start_time = time.time()
    main_jax()  # Use the JAX version instead of main()
    end_time = time.time()
    elapsed_time = end_time - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)
    seconds = int(elapsed_time % 60)
    print(f"Total execution time: {hours}h {minutes}m {seconds}s")