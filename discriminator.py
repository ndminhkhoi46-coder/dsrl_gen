# need to cache random

from sklearn.metrics import confusion_matrix
import jax
import jax.numpy as jnp
from jax import jit, random, lax
from functools import partial
import flax.linen as nn
import optax
from flax.training import train_state
from typing import Any, Dict, Tuple, List, Optional
import numpy as np
from sklearn.metrics import f1_score

class DiscriminatorSA(nn.Module):
    state_dim: int
    action_dim: int
    hidden_dim: int = 256
    
    @nn.compact
    def __call__(self, input_data):
        state = input_data[..., :self.state_dim]
        action = input_data[..., -self.action_dim:]
        
        h_s = nn.Dense(self.hidden_dim // 2, name='state_trunk')(state)
        h_a = nn.Dense(self.hidden_dim // 2, name='action_trunk')(action)
        h = jnp.concatenate([h_s, h_a], axis=-1)
        h = nn.tanh(h)

        h = nn.Dense(self.hidden_dim, name='hidden_trunk')(h)
        h = nn.tanh(h)
        
        h = nn.Dense(1, name='output_trunk')(h)
        return h

def create_train_state(rng_key, model, input_shape, optim_tuple):
    """Create a training state for the discriminator with exponential learning rate decay."""
    learning_rate, decay_steps, decay_rate = optim_tuple
    params = model.init(rng_key, jnp.ones(input_shape))
    
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
def generalized_bce(logits, target):
    """Binary cross entropy loss."""
    loss = optax.sigmoid_binary_cross_entropy(logits, target)
    return jnp.mean(loss)

@jit
def train_step(state, batch, target):
    """Update discriminator parameters with provided batch and target."""
    def loss_fn(params):
        logits = state.apply_fn(params, batch).flatten()        
        return generalized_bce(logits, target)

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    state = state.apply_gradients(grads=grads)
    return state, loss

@jit
def train_epoch(state, batches, targets):
    """Iterate over batches and targets using lax.scan."""
    def scan_fn(carry_state, step_tuple):
        batch, target = step_tuple
        new_state, loss = train_step(carry_state, batch, target)
        return new_state, loss

    final_state, losses = lax.scan(scan_fn, state, (batches, targets))
    return final_state, jnp.mean(losses)

@jit
def predict_batch(state, batch):
    """Jitted forward pass for rewards."""
    return state.apply_fn(state.params, batch).flatten()


class JAXDiscriminatorTrainer:
    """Optimized JAX-based discriminator trainer."""
    
    def __init__(self, dim_tuple, optim_tuple, rng_key=None, use_wandb=True):
        self.state_dim, self.action_dim, self.hidden_dim = dim_tuple
        self.learning_rate, self.decay_steps, self.decay_rate = optim_tuple
        self.use_wandb = use_wandb
        
        self.model = DiscriminatorSA(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
        )
        
        input_dim = self.state_dim + self.action_dim
        if rng_key is None:
            rng_key = random.PRNGKey(42)
        self.rng_key = rng_key
        self.state = create_train_state(
            rng_key, self.model, (1, input_dim), (self.learning_rate, self.decay_steps, self.decay_rate))
        
        # Initialize stats on device
        self.obs_mean = jnp.zeros(input_dim)
        self.obs_std = jnp.ones(input_dim)

    @partial(jit, static_argnums=(0,))
    def _compute_normalization_stats(self, data_jax):
        """Compute mean and std. Note: cannot modify self in-place inside @jit."""
        mean = jnp.mean(data_jax, axis=0)
        std = jnp.std(data_jax, axis=0) + 1e-8
        return mean, std

    @partial(jit, static_argnums=(0, 3))
    def pred_mode(self, preds_jax, traj_indices_jax, num_trajs):
        """Relabel data by taking the mode of predictions for each trajectory using JAX bincount."""
        counts = jnp.bincount(traj_indices_jax, length=num_trajs)
        sums = jnp.bincount(traj_indices_jax, weights=preds_jax, length=num_trajs)
        traj_means = sums / jnp.where(counts > 0, counts, 1.0)
        return (traj_means > 0.5).astype(jnp.float32)[traj_indices_jax]
        
    @partial(jit, static_argnums=(0,))
    def _normalize(self, data_jax):
        return (data_jax - self.obs_mean) / self.obs_std

    def evaluate(self, few_data, mixture_data, labels):
        """Compute evaluation metrics and return f1 and predictions."""
        few_logits = self.predict_rewards(few_data)
        mixture_logits = self.predict_rewards(mixture_data)
        logits = jnp.concatenate([few_logits, mixture_logits], axis=0)
        threshold = self.get_threshold(mixture_logits, 0.5)
        preds = (logits > threshold).astype(jnp.float32)
        
        f1 = f1_score(labels, np.array(preds).astype(int))
        cm = confusion_matrix(labels, np.array(preds).astype(int))
        print(f"Confusion matrix:\n{cm}")
        return f1, preds

    def train(self, few_data, mixture_data, mixture_indices, combined_labels, num_iterations=100, batch_size=256):
        # Initial stats computation
        few_data_jax = jnp.array(few_data)
        mixture_data_jax = jnp.array(mixture_data)
        
        num_few = few_data_jax.shape[0]
        num_mixture = mixture_data_jax.shape[0]
        num_samples = num_few + num_mixture
            
        mixture_indices_jax = jnp.array(mixture_indices, dtype=jnp.int32)
        num_mixture_trajs = int(mixture_indices[-1]) + 1
        
        # Jitted stats calculation
        mean, std = self._compute_normalization_stats(jnp.concatenate([few_data_jax, mixture_data_jax], axis=0))
        self.obs_mean, self.obs_std = mean, std
        
        few_data_normalized = self._normalize(few_data_jax)
        mixture_data_normalized = self._normalize(mixture_data_jax)
        
        # Combine few and mixture data for unified training
        combined_data = jnp.concatenate([few_data_normalized, mixture_data_normalized], axis=0)
        
        # Optimization: Pre-allocate combined_labels and use .at[].set() for updates
        combined_proxy_labels = jnp.ones(num_samples)
        combined_proxy_labels = combined_proxy_labels.at[-num_mixture:].set(0.0)
     
        for epoch in range(num_iterations):
            self.rng_key, shuffle_rng = jax.random.split(self.rng_key)
            indices = jax.random.permutation(shuffle_rng, num_samples)
            
            num_batches = num_samples // batch_size
            assert num_batches > 0, f"No batches to train on. num_samples: {num_samples}, batch_size: {batch_size}"

            # Prepare batched data and labels for unified training
            active_indices = indices[:num_batches * batch_size]
            batches = combined_data[active_indices].reshape(num_batches, batch_size, -1)
            proxy_labels_batched = combined_proxy_labels[active_indices].reshape(num_batches, batch_size)
            
            # Execute entire epoch on device in a single pass
            self.state, epoch_loss = train_epoch(self.state, batches, targets=proxy_labels_batched)
            
            if (epoch + 1) % 10 == 0:
                f1, preds = self.evaluate(few_data, mixture_data, combined_labels)
                print(f"Epoch {epoch+1}/{num_iterations} | Loss: {epoch_loss:.4f} | F1: {f1:.4f}")
                
                # # Relabel offline data: use only offline predictions
                # offline_preds = jnp.array(preds[num_expert:])
                # offline_data_mode = self.pred_mode(offline_preds, offline_indices_jax, num_offline_trajs)
                
                # # Update labels
                # combined_proxy_labels = combined_proxy_labels.at[num_expert:].set(offline_data_mode)
        
        # Final F1 computation
        f1, _ = self.evaluate(few_data, mixture_data, combined_labels)
        print(f"Training completed. Final F1: {f1:.4f}")

    def get_threshold(self, rewards, quantile=0.5):
        return np.quantile(rewards, quantile)
        # return 0.0

    def predict_rewards(self, input_data_jax, batch_size=8192):
        input_data_normalized = self._normalize(input_data_jax)
        num_samples = input_data_normalized.shape[0]
        all_rewards = []
        
        for i in range(0, num_samples, batch_size):
            batch = input_data_normalized[i:i + batch_size]
            logits = predict_batch(self.state, batch)
            all_rewards.append(logits)
            
        return np.array(jnp.concatenate(all_rewards))