import os
import time
import numpy as np
import pickle
import h5py
import json
import random as pyrandom
from typing import Any, Dict, Tuple, List, Optional
from pydantic import BaseModel
from osrl.common.dataset import _parse_trajectories

def save_h5(file_path: str, data_dict: Dict[str, np.ndarray]):
    """Save dataset to H5 file."""
    if os.path.dirname(file_path):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with h5py.File(file_path, 'w') as hf:
        for key, value in data_dict.items():
            hf.create_dataset(key, data=value, compression="gzip")

def load_h5(file_path: str):
    with h5py.File(file_path, 'r') as hf:
        data_dict = {key: hf[key][:] for key in hf.keys()}
    return data_dict

def load_json(file_path: str):
    return {} if not os.path.exists(file_path) else json.load(open(file_path, 'r'))

def save_json(file_path: str, data: dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)

def dict_hash_ignore_keys(d: dict, ignore_keys=None) -> str:
    import hashlib
    dict_filtered = {k: v for k, v in d.items() if k not in (ignore_keys or [])}
    hash_code = hashlib.sha256(json.dumps(dict_filtered, sort_keys=True, default=str).encode()).hexdigest()
    return hash_code

def seed_everything(seed=0):
    import jax
    pyrandom.seed(0)
    np.random.seed(0)
    return jax.random.PRNGKey(seed) # Only training is seeded

def trajs_indices_to_dataset(selected_items, data, mode='trajs'):
    indices = np.array([i for traj in selected_items for i in range(int(traj['start']), int(traj['end']) + 1)]) if mode == 'trajs' else np.array(selected_items)

    selected_data = {key: data[key][indices] for key in data.keys()}

    mask = np.ones(len(data['rewards']), dtype=bool)
    mask[indices] = False
    remaining_data = {key: data[key][mask] for key in data.keys()}

    # selected_trajs = _parse_trajectories(selected_data)
    
    # cost_lb = env.min_episode_cost
    # cost_ub = env.max_episode_cost
    # cost_bin_size = 10
    
    # costs = np.array([traj['total_cost'] for traj in selected_trajs])
    # bin_edges = np.arange(cost_lb, cost_ub + cost_bin_size, cost_bin_size)
    # hist, bin_edges = np.histogram(costs, bins=bin_edges)
    # print("Histogram of Expert Trajectory Costs:")
    # for i in range(len(hist)):
    #     bar = '*' * int(hist[i])
    #     print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {bar}")

    # rewards = np.array([traj['total_reward'] for traj in selected_trajs])
    # hist, bin_edges = np.histogram(rewards, bins=10)
    # print("Histogram of Expert Trajectory Rewards:")
    # for i in range(len(hist)):
    #     bar = '*' * int(hist[i])
    #     print(f"{bin_edges[i]:.2f} - {bin_edges[i+1]:.2f}: {bar}")

    return selected_data, remaining_data

def get_dataset_thresholds(trajs):
    """Calculates reward and cost thresholds (quartiles) for the entire dataset."""
    all_log_odds = [traj['total_reward'] for traj in trajs]
    reward_threshold = np.quantile(all_log_odds, 0.5)

    expert_trajs = [traj for traj in trajs if traj['total_reward'] >= reward_threshold]
    all_expert_costs = [traj['total_cost'] for traj in expert_trajs]
    cost_threshold = np.quantile(all_expert_costs, 0.5)

    return reward_threshold, cost_threshold

def indicator_function(trajectory, key, threshold):
    return trajectory[key] <= threshold

def add_extra_labels(trajs, dataset, reward_threshold, cost_threshold):
    dataset = label_expert(trajs, dataset, reward_threshold)
    dataset = label_safety(trajs, dataset, cost_threshold)
    dataset = label_traj_index(trajs, dataset)
    return dataset

def extract_median_trajs(safe_trajs, unsafe_trajs, expert_traj_num):
    safe_trajs = sorted(safe_trajs, key=lambda x: x['total_reward'])
    unsafe_trajs = sorted(unsafe_trajs, key=lambda x: x['total_reward'])
    
    n_unsafe = (expert_traj_num + 1) // 2
    n_safe = expert_traj_num - n_unsafe

    def get_medians(sorted_list, n):
        if n == 0: return []
        if n >= len(sorted_list): return sorted_list
        mid = len(sorted_list) // 2
        half_n = n // 2
        start = max(0, mid - half_n)
        return sorted_list[start:start+n]
        
    return get_medians(safe_trajs, n_safe) + get_medians(unsafe_trajs, n_unsafe)

def extract_extreme_trajs(safe_trajs, unsafe_trajs, expert_traj_num):
    safe_trajs = sorted(safe_trajs, key=lambda x: x['total_cost']) # minimum cost
    unsafe_trajs = sorted(unsafe_trajs, key=lambda x: x['total_reward'], reverse=True) # maximum reward
    
    n_unsafe = (expert_traj_num + 1) // 2
    n_safe = expert_traj_num - n_unsafe
    
    return safe_trajs[:n_safe] + unsafe_trajs[:n_unsafe]

def extract_random_trajs(safe_trajs, unsafe_trajs, expert_traj_num, seed=0):
    n_unsafe = (expert_traj_num + 1) // 2 
    n_safe = expert_traj_num - n_unsafe
    rng = np.random.default_rng(seed)
    return list(rng.choice(safe_trajs, n_safe, replace=False)) + \
           list(rng.choice(unsafe_trajs, n_unsafe, replace=False))

def generate_expert_dataset(data, trajs, expert_traj_num, threshold_tuple, sampling_method='random', return_remaining_data=False):
    reward_threshold, cost_threshold = threshold_tuple

    expert_trajs = [traj for traj in trajs if not indicator_function(traj, 'total_reward', reward_threshold)]
    safe_trajs = [traj for traj in expert_trajs if indicator_function(traj, 'total_cost', cost_threshold)]
    unsafe_trajs = [traj for traj in expert_trajs if not indicator_function(traj, 'total_cost', cost_threshold)]
    
    sampling_funcs = {
        'median': extract_median_trajs,
        'extreme': extract_extreme_trajs,
        'random': extract_random_trajs
    }

    if sampling_method not in sampling_funcs:
        raise ValueError(f"Unknown sampling method: {sampling_method}")

    selected_trajs = sampling_funcs[sampling_method](safe_trajs, unsafe_trajs, expert_traj_num)

    # NOTE: trajs_indices_to_dataset and add_extra_labels iterate over selected_trajs in the same order
    selected_data, remaining_data = trajs_indices_to_dataset(selected_trajs, data, mode = 'trajs')
    selected_data = add_extra_labels(selected_trajs, selected_data, reward_threshold, cost_threshold)
    remaining_trajs = [traj for traj in trajs if traj not in selected_trajs]
    
    if return_remaining_data:
        return selected_data, remaining_trajs, remaining_data
    return selected_data, remaining_trajs

def generate_safety_dataset(data, trajs, total_safe_traj_num, threshold_tuple):
    reward_threshold, cost_threshold = threshold_tuple

    rng = np.random.default_rng(0)

    num_unsafe_trajs = total_safe_traj_num // 2
    unsafe_trajs = list(rng.choice([traj for traj in trajs if not indicator_function(traj, 'total_cost', cost_threshold)], num_unsafe_trajs, replace=False))

    num_safe_trajs = total_safe_traj_num - num_unsafe_trajs
    safe_trajs = list(rng.choice([traj for traj in trajs if indicator_function(traj, 'total_cost', cost_threshold)], num_safe_trajs, replace=False))

    selected_trajs = safe_trajs + unsafe_trajs

    selected_data, remaining_data = trajs_indices_to_dataset(selected_trajs, data, mode = 'trajs')

    selected_data = add_extra_labels(selected_trajs, selected_data, reward_threshold, cost_threshold)

    return selected_data, remaining_data

def label_expert(trajs, dataset, reward_threshold):
    lengths = [traj['end'] - traj['start'] + 1 for traj in trajs]
    is_expert = [not indicator_function(traj, 'total_reward', reward_threshold) for traj in trajs]
    dataset['is_expert'] = np.repeat(is_expert, lengths)
    return dataset

def label_safety(trajs, dataset, cost_threshold):
    lengths = [traj['end'] - traj['start'] + 1 for traj in trajs]
    is_safe = [indicator_function(traj, 'total_cost', cost_threshold) for traj in trajs]
    dataset['is_safe'] = np.repeat(is_safe, lengths)
    return dataset

def label_traj_index(trajs, dataset):
    lengths = [traj['end'] - traj['start'] + 1 for traj in trajs]
    dataset['traj_index'] = np.repeat(np.arange(len(trajs)), lengths)
    return dataset

_hf_logged_in = False
def huggingface_login():
    global _hf_logged_in
    if not _hf_logged_in:
        from huggingface_hub import login
        hf_token = os.getenv("HF_TOKEN")
        if hf_token:
            login(token=hf_token)
            _hf_logged_in = True
        else:
            try:
                login()
                _hf_logged_in = True
            except Exception:
                print("Warning: HF_TOKEN environment variable not set. Hugging Face login may fail if credentials aren't cached.")

def load_dataset_from_hub(config):
    huggingface_login()
    from datasets import load_dataset
    env_name, medium = config['env_name'], 'medium' if config.get('medium', False) else 'random'
    hf_dataset = load_dataset(f"ndminhkhoi46/imitation-learning_dsrl_{env_name}_{medium}", download_mode="force_redownload")
    ds = {k: np.array(v) for k, v in hf_dataset['train'].to_dict().items()}
    return ds

def generate_offline_suboptimal_dataset(config, threshold_tuple, base_path):
    import os
    import numpy as np

    def subsampling(parent_dataset, parent_trajs, target_num, save_path):
        if os.path.exists(save_path):
            sampled_dataset = load_h5(save_path)
            sampled_trajs = _parse_trajectories(sampled_dataset)
            return sampled_dataset, sampled_trajs
        else:
            assert target_num <= len(parent_trajs), f"Sampling trajectories ({target_num}) must <= population trajectories count ({len(parent_trajs)})"
            rng = np.random.default_rng(0)
            selected_trajs = list(rng.choice(parent_trajs, target_num, replace=False))
            sampled_dataset, _ = trajs_indices_to_dataset(selected_trajs, parent_dataset, mode='trajs')
            if not config['disable_model_and_data_saving']: save_h5(save_path, sampled_dataset)
            return sampled_dataset, list(selected_trajs)

    def get_full_dataset():
        full_dataset = load_dataset_from_hub(config)
        full_trajs = _parse_trajectories(full_dataset)
        return full_dataset, full_trajs

    env_name = config['env_name']
    base_num = 1000
    base_file_path = os.path.join(base_path, f"offline_nonexpert_{env_name}_{base_num}.h5")
    
    full_dataset, full_trajs = (None, None) if os.path.exists(base_file_path) else get_full_dataset()
    base_sampled_dataset, base_sampled_trajs = subsampling(full_dataset, full_trajs, base_num, base_file_path)

    target_num = config['suboptimal_traj_num']
    target_file_path = os.path.join(base_path, f"offline_nonexpert_{env_name}_{target_num}.h5")
    target_subsample_dataset, target_subsample_trajs = subsampling(base_sampled_dataset, base_sampled_trajs, target_num, target_file_path)

    extra_label_keys = ['is_expert', 'is_safe', 'traj_index']
    if any(k not in target_subsample_dataset for k in extra_label_keys):
        target_subsample_dataset = {k: v for k, v in target_subsample_dataset.items() if k not in extra_label_keys} # Drop keys if exists
        reward_threshold, cost_threshold = threshold_tuple
        target_subsample_dataset = add_extra_labels(target_subsample_trajs, target_subsample_dataset, reward_threshold, cost_threshold)
    
    return target_subsample_dataset

def generate_safe_IL_datasets(env, config):
    data_config = {
        'suboptimal_traj_num': config['suboptimal_traj_num'],
        
        'env_name': env.spec.id,
        'few_traj_num': config['few_traj_num'],
        'expert_traj_num': config['expert_traj_num'],
        'medium': config.get('medium', False),
    }
    hash_data = dict_hash_ignore_keys(data_config)

    base_path = os.path.expanduser(f"~/scratchDirectory/dataset/dsrl-IL-data/")
    dsrl_trajs_path = os.path.join(base_path, f"dsrl_trajs_{hash_data}.json") 

    dsrl_data = env.get_dataset()
    # Cache the 2nd most time-consuming step in dataset generation
    if not config['disable_model_and_data_saving'] and os.path.exists(dsrl_trajs_path):
        dsrl_trajs = load_json(dsrl_trajs_path)
    else:
        dsrl_trajs = _parse_trajectories(dsrl_data)
        if not config['disable_model_and_data_saving']: save_json(dsrl_trajs_path, dsrl_trajs)

    threshold_tuple = get_dataset_thresholds(dsrl_trajs)
    mixture_expert, _ = generate_expert_dataset(dsrl_data, dsrl_trajs, config['expert_traj_num'] + config['few_traj_num'], threshold_tuple, sampling_method='random')

    mixture_expert_trajs = _parse_trajectories(mixture_expert)
    few_data, _, mixture_expert = generate_expert_dataset(
        mixture_expert, mixture_expert_trajs, config['few_traj_num'], threshold_tuple, 
        sampling_method='random', return_remaining_data=True
    )

    # Cache the most time-consuming step in dataset generation
    offline_suboptimal = generate_offline_suboptimal_dataset(config, threshold_tuple, base_path)
    safety_data, _ = generate_safety_dataset(dsrl_data, dsrl_trajs, config['safety_traj_num'], threshold_tuple)

    dataset_keys = ['observations', 'actions', 'next_observations', 'rewards', 'costs', 'terminals', 'timeouts', 'is_expert', 'is_safe', 'traj_index']
    mixture_data = {} 
    for k in dataset_keys:
        mixture_data[k] = np.concatenate([mixture_expert[k], offline_suboptimal[k]], axis=0)

    metadata_path = os.path.join(base_path, f"metadata.json")
    metadata = load_json(metadata_path)
    metadata[hash_data] = {'cost_threshold': float(threshold_tuple[1]), 'reward_threshold': float(threshold_tuple[0])}
    if not config['disable_model_and_data_saving']: save_json(metadata_path, metadata)
    return few_data, mixture_data, safety_data

def pca_fit(few_data, mixture_data, name):
    """
    Create and fit PCA on a subsample of concatenated data.
    """
    X = np.concatenate([few_data, mixture_data], axis=0)
    from sklearn.preprocessing import StandardScaler

    # Standardize first! Essential for high-dimensional IL
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    from sklearn.decomposition import PCA
    n_components = 0.8 # Pareto principle
    pca = PCA(n_components=n_components, svd_solver='full', random_state=0)
    pca.fit(X_scaled)
    print(f"PCA ({n_components}) keeps {pca.n_components_} dim for {name}.")
    return pca, scaler

def pca_transform(X, pca, scaler):
    """
    Transform input data (observations + actions) using separate PCAs
    """
    X_scaled = scaler.transform(X)
    X_pca = pca.transform(X_scaled)
    return X_pca[:,:pca.n_components_]


def load_unique_configs(file_path):
    if not os.path.exists(file_path):
        return {}
    unique_entries = {}
    with open(file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line: continue
            try:
                entry = json.loads(line)
                unique_entries[entry["hash"]] = entry
            except (json.JSONDecodeError, KeyError):
                pass
    return unique_entries

def save_configs(file_path, config_dict):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        for entry in config_dict.values():
            f.write(json.dumps(entry, default=str) + "\n")

def safe_reset(env, seed=None):
    env_name = env.spec.id
    is_metadrive = "Metadrive" in env_name
    if is_metadrive:
        return env.reset()
    else:
        return env.reset(seed=seed)

def reset_data():
    return {'observations': np.array([]),
            'actions': np.array([]),
            'next_observations': np.array([]),
            'rewards': np.array([]),
            'costs': np.array([]),
            'terminals': np.array([]),
            'timeouts': np.array([])
            }

def append_data(data, cmdp_tuple, done_tuple):
    s, a, s_next, r, c = cmdp_tuple
    terminal, timeout = done_tuple
    data['observations'] = np.append(data['observations'], s)
    data['actions'] = np.append(data['actions'], a)
    data['next_observations'] = np.append(data['next_observations'], s_next)
    data['rewards'] = np.append(data['rewards'], r)
    data['costs'] = np.append(data['costs'], c)
    data['terminals'] = np.append(data['terminals'], terminal)
    data['timeouts'] = np.append(data['timeouts'], timeout)
    return data

def adversarial_validation(X1, X2, label_name="Suboptimal-Expert vs. Safe-Expert"):
    """
    Train a classifier to distinguish between two datasets and report the AUC.
    """
    import numpy as np
    from xgboost import XGBClassifier
    from sklearn.model_selection import cross_val_score

    X_combined = np.concatenate([X1, X2], axis=0)
    y_combined = np.concatenate([np.zeros(len(X1)), np.ones(len(X2))], axis=0)
    
    clf = XGBClassifier(
        n_estimators=100,
        max_depth=3,
        learning_rate=0.1,
        random_state=0,
        eval_metric='logloss'
    )
    
    scores = cross_val_score(clf, X_combined, y_combined, cv=5, scoring='roc_auc')
    
    print(f"Mean Discriminator AUC ({label_name}): {np.mean(scores):.4f}")
    
    if np.mean(scores) > 0.70:
        print("Warning: Significant distribution shift detected.")

def _config_to_dict(cfg: BaseModel) -> dict:
    return cfg.model_dump() if hasattr(cfg, "model_dump") else cfg.dict()

def generate_traj_indices(data_dict):
    terminals = data_dict['terminals'].flatten()
    timeouts = data_dict['timeouts'].flatten()
    ends = np.where(terminals | timeouts)[0]

    shifted_ends = np.append(-1, ends)
    traj_lengths = np.diff(shifted_ends)

    traj_indices = np.repeat(np.arange(len(traj_lengths), dtype=np.int32), traj_lengths)
    
    return traj_indices

def transform_data(data_tuple, transform_tuple):
    obs, act = data_tuple
    pca_obs, pca_act, scaler_obs, scaler_act = transform_tuple
    transformed_input = np.concatenate([pca_transform(obs, pca_obs, scaler_obs), pca_transform(act, pca_act, scaler_act)], axis=1)
    return transformed_input
    
# JAX discriminator trainer
def dimension_reduction(few_data, mixture_data, use_wandb=False):
    pca_obs, scaler_obs = pca_fit(few_data['observations'], mixture_data['observations'], 'obs')
    pca_act, scaler_act = pca_fit(few_data['actions'], mixture_data['actions'], 'act')
    
    transform_tuple = (pca_obs, pca_act, scaler_obs, scaler_act)
    few_input = transform_data((few_data['observations'], few_data['actions']), transform_tuple)
    mixture_input = transform_data((mixture_data['observations'], mixture_data['actions']), transform_tuple)

    if use_wandb:
        # Check PCA result quality
        state_pca_var = np.sum(pca_obs.explained_variance_ratio_)
        action_pca_var = np.sum(pca_act.explained_variance_ratio_)
        print(f"PCA quality: State variance={state_pca_var:.2%}, Action variance={action_pca_var:.2%}")
    return few_input, mixture_input, transform_tuple

def get_discriminator_threshold(log_odds, quantile):
    xp = jnp if hasattr(log_odds, "__jax_array__") or "jax" in str(type(log_odds)).lower() else np
    return xp.quantile(log_odds, quantile)

def threshold_log_odds(log_odds, threshold):
    xp = jnp if hasattr(log_odds, "__jax_array__") or "jax" in str(type(log_odds)).lower() else np
    return (log_odds > threshold).astype(int)

def print_statistics(X, name):
    print(f"# STATS ({name}):")
    print(f"BOUNDARY: [Min: {np.min(X):.2f}, Max: {np.max(X):.2f}]")
    print(f"SHAPE: [Median: {np.median(X):.2f}, IQR: {np.percentile(X, 75) - np.percentile(X, 25):.2f}]")
    print(f"MEAN SPREAD: {np.mean(X):.2f} +- {np.std(X):.2f}")

def oversampling(X, y):
    import numpy as np
    from imblearn.over_sampling import RandomOverSampler
    from collections import Counter
    print("Original class distribution:", Counter(y))
    oversample = RandomOverSampler(sampling_strategy='minority', random_state=0)
    X_over, y_over = oversample.fit_resample(X, y)
    print("Oversampled class distribution:", Counter(y_over))
    return X_over, y_over

def undersampling(X, y):
    import numpy as np
    from imblearn.under_sampling import RandomUnderSampler
    from collections import Counter
    print("Original class distribution:", Counter(y))
    undersample = RandomUnderSampler(sampling_strategy='majority', random_state=0)
    X_under, y_under = undersample.fit_resample(X, y)
    print("Undersampled class distribution:", Counter(y_under))
    return X_under, y_under

def drop_keys(data_dict, keys):
    return {k: v for k, v in data_dict.items() if k not in keys}

def get_cache_path(config, name, ext, base_dir="~/scratchDirectory/dataset/dsrl-IL-data"):
    data_config = {
        'suboptimal_traj_num': config['suboptimal_traj_num'],
        'env_name': config['env_name'],
        'few_traj_num': config['few_traj_num'],
        'expert_traj_num': config['expert_traj_num'],
        'medium': config.get('medium', False),
    }
    if "safety" in name: data_config['safety_traj_num'] = config['safety_traj_num']
    hash_data = dict_hash_ignore_keys(data_config)
    dir_path = os.path.expanduser(base_dir)
    name_str = f"{name}_{config['safety_traj_num']}" if "safety" in name else name
    file_path = os.path.join(dir_path, f"{config['env_name']}_{name_str}_{hash_data}.{ext}")
    return file_path, dir_path

def save_dataset(data_dict, dataset_name, config, base_dir="~/scratchDirectory/dataset/dsrl-IL-data"):
    if config['disable_model_and_data_saving']: return None
    file_path, _ = get_cache_path(config, dataset_name, "h5", base_dir)
    save_h5(file_path, data_dict) 

def save_discriminator(data_dict, config, base_dir="~/scratchDirectory/dataset/dsrl-IL-data"):
    if config['disable_model_and_data_saving']: return None
    file_path, dir_path = get_cache_path(config, "discriminator", "pkl", base_dir)
    os.makedirs(dir_path, exist_ok=True)
    with open(file_path, 'wb') as f: pickle.dump(data_dict, f)
    print(f"Discriminator saved to {file_path}")

def load_discriminator(config, base_dir="~/scratchDirectory/dataset/dsrl-IL-data"):
    if config['disable_model_and_data_saving']: return None
    file_path, _ = get_cache_path(config, "discriminator", "pkl", base_dir)
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f: data_dict = pickle.load(f)
        print(f"Discriminator loaded from {file_path}")
        return data_dict
    return None

def format_classification_metrics(metric_scalar):
    return round(metric_scalar*100, 2)

def shapiro_test(data):
    from scipy import stats
    import numpy as np
    # Shapiro-Wilk test may have a limit of 5000 samples
    data = data if len(data) <= 5000 else np.random.choice(data, 5000, replace=False)
    stat, p = stats.shapiro(data)
    return p > 0.05

def calculate_ema(data, alpha=0.1):
    from scipy import signal
    x_coefs = [alpha]
    y_coefs = [1, -(1 - alpha)] 
    
    empty_prev_states = signal.lfilter_zi(x_coefs, y_coefs) 
    initial_states = empty_prev_states * data[0]

    ema_ls, _ = signal.lfilter(x_coefs, y_coefs, data, zi=initial_states)
    ema_val = ema_ls[-1]
    return round(ema_val, 4), ema_ls

    