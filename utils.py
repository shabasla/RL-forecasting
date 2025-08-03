"""
Utility functions for the SAC Trading Agent.
Includes data validation, logging, visualization, and helper functions.
"""

import os
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging
from typing import List, Dict, Tuple, Optional


def setup_logging(log_dir: str = 'logs', level: str = 'INFO') -> logging.Logger:
    """
    Set up logging configuration.
    
    Args:
        log_dir (str): Directory to save log files
        level (str): Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
    
    Returns:
        logging.Logger: Configured logger
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('SAC_Trading')
    logger.setLevel(getattr(logging, level.upper()))
    
    # Create formatters
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    file_handler = logging.FileHandler(
        os.path.join(log_dir, f'training_{timestamp}.log')
    )
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, level.upper()))
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def validate_data_files(data_file: str, entry_points_file: str) -> Tuple[bool, List[str]]:
    """
    Validate the format and content of data files.
    
    Args:
        data_file (str): Path to market data CSV
        entry_points_file (str): Path to entry signals CSV
    
    Returns:
        Tuple[bool, List[str]]: (is_valid, error_messages)
    """
    errors = []
    
    # Check file existence
    if not os.path.exists(data_file):
        errors.append(f"Market data file not found: {data_file}")
    if not os.path.exists(entry_points_file):
        errors.append(f"Entry points file not found: {entry_points_file}")
    
    if errors:
        return False, errors
    
    try:
        # Load and validate market data
        market_data = pd.read_csv(data_file)
        required_market_cols = ['date', 'close', 'sto_osc', 'macd', 'adx', 'obv', 'n_atr', 'log_ret', 'newsapi']
        missing_market_cols = set(required_market_cols) - set(market_data.columns)
        if missing_market_cols:
            errors.append(f"Missing columns in market data: {missing_market_cols}")
        
        # Validate date format
        try:
            pd.to_datetime(market_data['date'])
        except Exception as e:
            errors.append(f"Invalid date format in market data: {e}")
        
        # Load and validate entry points
        entry_data = pd.read_csv(entry_points_file)
        if 'date' not in entry_data.columns:
            errors.append("Missing 'date' column in entry points file")
        
        try:
            pd.to_datetime(entry_data['date'])
        except Exception as e:
            errors.append(f"Invalid date format in entry points: {e}")
        
        # Check data alignment
        market_dates = set(pd.to_datetime(market_data['date']).dt.date)
        entry_dates = set(pd.to_datetime(entry_data['date']).dt.date)
        unmatched_entries = entry_dates - market_dates
        if unmatched_entries:
            errors.append(f"Entry dates not found in market data: {len(unmatched_entries)} dates")
    
    except Exception as e:
        errors.append(f"Error reading data files: {e}")
    
    return len(errors) == 0, errors


def set_random_seeds(seed: int = 42) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def create_directories(dirs: List[str]) -> None:
    """
    Create necessary directories if they don't exist.
    
    Args:
        dirs (List[str]): List of directory paths to create
    """
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)


def save_training_results(
    episode_rewards: List[float], 
    episode_returns: List[float],
    save_dir: str = 'results'
) -> None:
    """
    Save training results to CSV and create plots.
    
    Args:
        episode_rewards (List[float]): List of episode rewards
        episode_returns (List[float]): List of episode total returns
        save_dir (str): Directory to save results
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Save to CSV
    results_df = pd.DataFrame({
        'episode': range(1, len(episode_rewards) + 1),
        'reward': episode_rewards,
        'total_return': episode_returns
    })
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_path = os.path.join(save_dir, f'training_results_{timestamp}.csv')
    results_df.to_csv(csv_path, index=False)
    
    # Create plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot rewards
    ax1.plot(results_df['episode'], results_df['reward'], alpha=0.6)
    ax1.plot(results_df['episode'], results_df['reward'].rolling(50).mean(), 
             color='red', label='50-episode moving average')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Episode Reward')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot returns
    ax2.plot(results_df['episode'], results_df['total_return'], alpha=0.6)
    ax2.plot(results_df['episode'], results_df['total_return'].rolling(50).mean(), 
             color='red', label='50-episode moving average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Total Return')
    ax2.set_title('Training Progress: Episode Returns')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = os.path.join(save_dir, f'training_progress_{timestamp}.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Results saved to {csv_path}")
    print(f"Plots saved to {plot_path}")


def calculate_trading_metrics(trades: List[Dict]) -> Dict:
    """
    Calculate comprehensive trading performance metrics.
    
    Args:
        trades (List[Dict]): List of trade dictionaries
    
    Returns:
        Dict: Dictionary containing performance metrics
    """
    if not trades:
        return {}
    
    returns = [trade['final_return'] for trade in trades]
    steps = [trade['steps'] for trade in trades]
    
    # Basic metrics
    total_trades = len(trades)
    total_return = sum(returns)
    avg_return = np.mean(returns)
    std_return = np.std(returns)
    
    # Win/Loss metrics
    wins = [r for r in returns if r > 0]
    losses = [r for r in returns if r < 0]
    win_rate = len(wins) / total_trades if total_trades > 0 else 0
    
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean(losses) if losses else 0
    
    # Risk metrics
    sharpe_ratio = avg_return / std_return if std_return > 0 else 0
    max_return = max(returns) if returns else 0
    min_return = min(returns) if returns else 0
    
    # Profit factor
    total_wins = sum(wins) if wins else 0
    total_losses = abs(sum(losses)) if losses else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf')
    
    # Trade duration
    avg_duration = np.mean(steps)
    
    return {
        'total_trades': total_trades,
        'total_return': total_return,
        'avg_return': avg_return,
        'std_return': std_return,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'sharpe_ratio': sharpe_ratio,
        'max_return': max_return,
        'min_return': min_return,
        'profit_factor': profit_factor,
        'avg_duration': avg_duration,
    }


def plot_trade_analysis(trades: List[Dict], save_path: Optional[str] = None) -> None:
    """
    Create comprehensive trade analysis plots.
    
    Args:
        trades (List[Dict]): List of trade dictionaries
        save_path (Optional[str]): Path to save the plot
    """
    if not trades:
        print("No trades to analyze")
        return
    
    returns = [trade['final_return'] for trade in trades]
    steps = [trade['steps'] for trade in trades]
    reasons = [trade['reason'] for trade in trades]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Returns distribution
    axes[0, 0].hist(returns, bins=30, alpha=0.7, edgecolor='black')
    axes[0, 0].axvline(np.mean(returns), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(returns):.4f}')
    axes[0, 0].set_xlabel('Return')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].set_title('Distribution of Returns')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Returns over time
    axes[0, 1].plot(range(len(returns)), returns, alpha=0.6)
    axes[0, 1].axhline(0, color='black', linestyle='-', alpha=0.3)
    axes[0, 1].plot(range(len(returns)), np.cumsum(returns), 
                    color='red', label='Cumulative Return')
    axes[0, 1].set_xlabel('Trade Number')
    axes[0, 1].set_ylabel('Return')
    axes[0, 1].set_title('Returns Over Time')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Trade duration distribution
    axes[1, 0].hist(steps, bins=range(1, max(steps) + 2), alpha=0.7, edgecolor='black')
    axes[1, 0].axvline(np.mean(steps), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(steps):.1f}')
    axes[1, 0].set_xlabel('Trade Duration (Steps)')
    axes[1, 0].set_ylabel('Frequency')
    axes[1, 0].set_title('Distribution of Trade Durations')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Exit reasons
    reason_counts = pd.Series(reasons).value_counts()
    axes[1, 1].pie(reason_counts.values, labels=reason_counts.index, autopct='%1.1f%%')
    axes[1, 1].set_title('Trade Exit Reasons')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Trade analysis plot saved to {save_path}")
    
    plt.show()


def load_model_checkpoint(checkpoint_path: str, agent, device: str = 'cpu') -> Dict:
    """
    Load a complete model checkpoint.
    
    Args:
        checkpoint_path (str): Path to checkpoint file
        agent: SAC agent instance
        device (str): Device to load model on
    
    Returns:
        Dict: Checkpoint information
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model states
    agent.actor.load_state_dict(checkpoint['actor_state_dict'])
    agent.critic_1.load_state_dict(checkpoint['critic_1_state_dict'])
    agent.critic_2.load_state_dict(checkpoint['critic_2_state_dict'])
    agent.target_critic_1.load_state_dict(checkpoint['target_critic_1_state_dict'])
    agent.target_critic_2.load_state_dict(checkpoint['target_critic_2_state_dict'])
    
    # Load optimizer states
    agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
    agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    agent.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])
    
    # Load other parameters
    agent.log_alpha = checkpoint['log_alpha']
    
    print(f"Loaded checkpoint from episode {checkpoint['episode']}")
    
    return {
        'episode': checkpoint['episode'],
        'episode_rewards': checkpoint.get('episode_rewards', []),
        'episode_total_returns': checkpoint.get('episode_total_returns', []),
    }


def print_training_summary(episode_rewards: List[float], episode_returns: List[float]) -> None:
    """
    Print a comprehensive training summary.
    
    Args:
        episode_rewards (List[float]): List of episode rewards
        episode_returns (List[float]): List of episode total returns
    """
    if not episode_rewards or not episode_returns:
        print("No training data to summarize")
        return
    
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    
    # Basic statistics
    print(f"Total Episodes: {len(episode_rewards)}")
    print(f"Final Episode Reward: {episode_rewards[-1]:.4f}")
    print(f"Final Episode Return: {episode_returns[-1]:.4f}")
    
    # Reward statistics
    print(f"\nREWARD STATISTICS:")
    print(f"  Mean Reward: {np.mean(episode_rewards):.4f}")
    print(f"  Std Reward:  {np.std(episode_rewards):.4f}")
    print(f"  Max Reward:  {np.max(episode_rewards):.4f}")
    print(f"  Min Reward:  {np.min(episode_rewards):.4f}")
    
    # Return statistics
    print(f"\nRETURN STATISTICS:")
    print(f"  Mean Return: {np.mean(episode_returns):.4f}")
    print(f"  Std Return:  {np.std(episode_returns):.4f}")
    print(f"  Max Return:  {np.max(episode_returns):.4f}")
    print(f"  Min Return:  {np.min(episode_returns):.4f}")
    
    # Performance trends (last 20% of episodes)
    if len(episode_rewards) >= 10:
        recent_start = int(0.8 * len(episode_rewards))
        recent_rewards = episode_rewards[recent_start:]
        recent_returns = episode_returns[recent_start:]
        
        print(f"\nRECENT PERFORMANCE (Last {len(recent_rewards)} episodes):")
        print(f"  Mean Recent Reward: {np.mean(recent_rewards):.4f}")
        print(f"  Mean Recent Return: {np.mean(recent_returns):.4f}")
        
        # Compare with early performance
        early_rewards = episode_rewards[:len(recent_rewards)]
        early_returns = episode_returns[:len(recent_returns)]
        
        reward_improvement = np.mean(recent_rewards) - np.mean(early_rewards)
        return_improvement = np.mean(recent_returns) - np.mean(early_returns)
        
        print(f"  Reward Improvement: {reward_improvement:+.4f}")
        print(f"  Return Improvement: {return_improvement:+.4f}")
    
    print("="*80)


def format_time_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string.
    
    Args:
        seconds (float): Duration in seconds
    
    Returns:
        str: Formatted duration string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    
    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"


def validate_model_files(model_dir: str) -> Tuple[bool, List[str]]:
    """
    Validate that all required model files exist.
    
    Args:
        model_dir (str): Directory containing model files
    
    Returns:
        Tuple[bool, List[str]]: (all_files_exist, missing_files)
    """
    required_files = [
        'final_actor_weights.pth',
        'final_critic1_weights.pth',
        'final_critic2_weights.pth',
        'final_target_critic1_weights.pth',
        'final_target_critic2_weights.pth',
        'complete_model_checkpoint.pth'
    ]
    
    missing_files = []
    for filename in required_files:
        filepath = os.path.join(model_dir, filename)
        if not os.path.exists(filepath):
            missing_files.append(filename)
    
    return len(missing_files) == 0, missing_files


def export_trades_to_csv(trades: List[Dict], filepath: str) -> None:
    """
    Export trade details to CSV file.
    
    Args:
        trades (List[Dict]): List of trade dictionaries
        filepath (str): Output CSV file path
    """
    if not trades:
        print("No trades to export")
        return
    
    trades_df = pd.DataFrame(trades)
    trades_df.to_csv(filepath, index=False)
    print(f"Exported {len(trades)} trades to {filepath}")


def create_model_info_file(model_dir: str, config: Dict) -> None:
    """
    Create a text file with model information and configuration.
    
    Args:
        model_dir (str): Directory to save the info file
        config (Dict): Configuration dictionary
    """
    info_path = os.path.join(model_dir, 'model_info.txt')
    
    with open(info_path, 'w') as f:
        f.write("SAC Trading Agent Model Information\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"PyTorch Version: {torch.__version__}\n")
        f.write(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n\n")
        
        f.write("Configuration:\n")
        f.write("-" * 20 + "\n")
        for key, value in config.items():
            f.write(f"{key}: {value}\n")
        
        f.write("\nModel Files:\n")
        f.write("-" * 20 + "\n")
        f.write("- final_actor_weights.pth\n")
        f.write("- final_critic1_weights.pth\n")
        f.write("- final_critic2_weights.pth\n")
        f.write("- final_target_critic1_weights.pth\n")
        f.write("- final_target_critic2_weights.pth\n")
        f.write("- complete_model_checkpoint.pth\n")
    
    print(f"Model info saved to {info_path}")


# Export utility functions
__all__ = [
    'setup_logging',
    'validate_data_files',
    'set_random_seeds',
    'create_directories',
    'save_training_results',
    'calculate_trading_metrics',
    'plot_trade_analysis',
    'load_model_checkpoint',
    'print_training_summary',
    'format_time_duration',
    'validate_model_files',
    'export_trades_to_csv',
    'create_model_info_file',
]
