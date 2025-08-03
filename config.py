"""
Configuration file for SAC Trading Agent
Contains hyperparameters and settings for training and environment.
"""

import numpy as np

# Training Configuration
TRAINING_CONFIG = {
    'episodes': 1000,
    'batch_size': 64,
    'buffer_capacity': 100000,
    'save_interval': 100,  # Save model every N episodes
    'eval_interval': 50,   # Evaluate model every N episodes
    'seed': 42,
}

# SAC Agent Configuration
SAC_CONFIG = {
    'learning_rate': 1e-4,
    'gamma': 0.99,          # Discount factor
    'tau': 0.005,           # Target network update rate
    'alpha': 0.2,           # Initial entropy coefficient
    'target_entropy_scale': 0.98,  # Target entropy = -log(1/action_dim) * scale
    'hidden_dim': 128,
    'sequence_length': 15,
}

# Environment Configuration
ENV_CONFIG = {
    'max_steps': 15,
    'lookback_window': 15,
    'initial_tp': 0.03,     # Initial take profit (3%)
    'initial_sl': -0.03,    # Initial stop loss (-3%)
    'normalization_alpha': 0.005,  # EMA normalization factor
}

# Action Space Configuration
ACTION_CONFIG = {
    'action_values': np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) / 100,
    'action_dim': 15,
}

# State Space Configuration
STATE_CONFIG = {
    'state_columns': ['sto_osc', 'macd', 'adx', 'obv', 'n_atr', 'log_ret', 'newsapi'],
    'state_dim': 7,
}

# Prioritized Experience Replay Configuration
PER_CONFIG = {
    'alpha': 0.6,           # Prioritization exponent
    'beta_start': 0.4,      # Initial importance sampling weight
    'beta_end': 1.0,        # Final importance sampling weight
}

# Reward Function Configuration
REWARD_CONFIG = {
    'time_reward_scale': 0.5,
    'shaped_reward_scale': 0.1,
    'bonus_multiplier': 1.0,
}

# Model Paths
MODEL_PATHS = {
    'model_dir': 'models',
    'results_dir': 'results',
    'logs_dir': 'logs',
    'checkpoint_prefix': 'sac_trading_',
}

# Data Configuration
DATA_CONFIG = {
    'required_columns': {
        'market_data': ['date', 'close', 'sto_osc', 'macd', 'adx', 'obv', 'n_atr', 'log_ret', 'newsapi'],
        'entry_signals': ['date'],
    },
    'date_format': '%Y-%m-%d',
}

# Logging Configuration
LOGGING_CONFIG = {
    'log_level': 'INFO',
    'log_format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'console_output': True,
    'file_output': True,
}

# Device Configuration
DEVICE_CONFIG = {
    'use_cuda': True,       # Use CUDA if available
    'cuda_deterministic': True,  # Make CUDA operations deterministic
}

# Evaluation Configuration
EVAL_CONFIG = {
    'eval_episodes': 5,
    'deterministic': True,  # Use deterministic actions during evaluation
    'save_trades': True,    # Save individual trade details
}

# Export all configs
__all__ = [
    'TRAINING_CONFIG',
    'SAC_CONFIG', 
    'ENV_CONFIG',
    'ACTION_CONFIG',
    'STATE_CONFIG',
    'PER_CONFIG',
    'REWARD_CONFIG',
    'MODEL_PATHS',
    'DATA_CONFIG',
    'LOGGING_CONFIG',
    'DEVICE_CONFIG',
    'EVAL_CONFIG',
]
