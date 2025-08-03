#!/usr/bin/env python3
"""
Reinforcement Learning Trading Agent using Soft Actor-Critic (SAC)
with CNN-LSTM architecture for dynamic TP/SL management.

Author: Your Name
Created: 2025
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
import os
from collections import deque
import random
from datetime import datetime, timedelta
import argparse


# Replay Buffer with PER
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = deque(maxlen=capacity)
        self.pos = 0
        
    def push(self, state, action, reward, next_state, done):
        max_priority = max(self.priorities) if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities.append(max_priority)
        self.pos = (self.pos + 1) % self.capacity
        
    def sample(self, batch_size, beta=0.4):
        priorities = np.array(self.priorities, dtype=np.float32)
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = zip(*samples)
        return (np.array(states), np.array(actions), np.array(rewards), 
                np.array(next_states), np.array(dones), indices, weights)
                
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            if idx < len(self.priorities):
                self.priorities[idx] = float(priority) + 1e-5
            
    def __len__(self):
        return len(self.buffer)


# CNN-LSTM Network
class CNNLSTMNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, sequence_length=15, hidden_dim=128):
        super(CNNLSTMNetwork, self).__init__()
        
        self.conv1 = nn.Conv1d(state_dim, 32, kernel_size=3, padding=1)
        self.ln1 = nn.LayerNorm([32, sequence_length])
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.ln2 = nn.LayerNorm([64, sequence_length])
        
        self.lstm = nn.LSTM(64, hidden_dim, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, x):
        # x shape: [batch, sequence, features]
        x = x.permute(0, 2, 1)  # [batch, features, sequence]
        x = F.relu(self.ln1(self.conv1(x)))
        x = F.relu(self.ln2(self.conv2(x)))
        x = x.permute(0, 2, 1)  # [batch, sequence, channels]
        x, _ = self.lstm(x)
        x = x[:, -1, :]  # Take last LSTM output
        x = F.relu(self.ln3(self.fc1(x)))
        return self.fc2(x)


# SAC Agent with Discrete Actions
class SACAgent:
    def __init__(self, state_dim, action_dim, sequence_length=15, hidden_dim=128, lr=1e-4, gamma=0.99, tau=0.005):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.action_values = np.array([-7, -6, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7]) / 100  # Discrete actions
        
        # Actor
        self.actor = CNNLSTMNetwork(state_dim, action_dim, sequence_length).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr)
        
        # Critics
        self.critic_1 = CNNLSTMNetwork(state_dim, action_dim, sequence_length).to(self.device)
        self.critic_2 = CNNLSTMNetwork(state_dim, action_dim, sequence_length).to(self.device)
        self.target_critic_1 = CNNLSTMNetwork(state_dim, action_dim, sequence_length).to(self.device)
        self.target_critic_2 = CNNLSTMNetwork(state_dim, action_dim, sequence_length).to(self.device)
        
        # Initialize target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(param.data)
            
        self.critic_optimizer = torch.optim.Adam(
            list(self.critic_1.parameters()) + list(self.critic_2.parameters()), lr=lr)
            
        self.gamma = gamma
        self.tau = tau
        self.action_dim = action_dim
        
        # Log entropy target
        self.target_entropy = -np.log(1.0 / action_dim) * 0.98
        self.log_alpha = torch.tensor(np.log(0.2), requires_grad=True, device=self.device)
        self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
        
    def select_action(self, state, deterministic=False):
        state = torch.FloatTensor(state).to(self.device).unsqueeze(0)
        logits = self.actor(state)
        
        if deterministic:
            action_idx = torch.argmax(logits, dim=1).item()
        else:
            dist = Categorical(logits=logits)
            action_idx = dist.sample().item()
            
        action = self.action_values[action_idx]
        return action, action_idx
        
    def update(self, replay_buffer, batch_size, beta):
        states, actions, rewards, next_states, dones, indices, weights = replay_buffer.sample(batch_size, beta)
        
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device).unsqueeze(1)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device).unsqueeze(1)
        weights = torch.FloatTensor(weights).to(self.device).unsqueeze(1)
        
        # Update critics
        with torch.no_grad():
            next_logits = self.actor(next_states)
            next_dist = Categorical(logits=next_logits)
            next_probs = next_dist.probs
            next_q1 = self.target_critic_1(next_states)
            next_q2 = self.target_critic_2(next_states)
            next_q = torch.min(next_q1, next_q2)
            next_q = (next_probs * (next_q - self.log_alpha.exp() * torch.log(next_probs + 1e-8))).sum(dim=1, keepdim=True)
            target_q = rewards + (1 - dones) * self.gamma * next_q
            
        current_q1 = self.critic_1(states).gather(1, actions.unsqueeze(1))
        current_q2 = self.critic_2(states).gather(1, actions.unsqueeze(1))
        
        critic_loss = (weights * (F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q))).mean()
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # Update actor
        logits = self.actor(states)
        dist = Categorical(logits=logits)
        probs = dist.probs
        q1 = self.critic_1(states)
        q2 = self.critic_2(states)
        q_values = torch.min(q1, q2)
        actor_loss = (probs * (self.log_alpha.exp() * torch.log(probs + 1e-8) - q_values)).sum(dim=1).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update alpha
        alpha_loss = -(self.log_alpha * (torch.log(probs + 1e-8) + self.target_entropy).detach()).mean()
        
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        
        # Update target networks
        for target_param, param in zip(self.target_critic_1.parameters(), self.critic_1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for target_param, param in zip(self.target_critic_2.parameters(), self.critic_2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            
        # Update priorities
        td_errors = torch.abs(current_q1 - target_q).detach().cpu().numpy().flatten()
        replay_buffer.update_priorities(indices, td_errors)


# Trading Environment
class TradingEnvironment:
    def __init__(self, data_file, entry_points_file, max_steps=15, lookback_window=15):
        self.data = pd.read_csv(data_file)
        self.entry_points = pd.read_csv(entry_points_file)
        self.state_cols = ['sto_osc', 'macd', 'adx', 'obv', 'n_atr', 'log_ret', 'newsapi']
        self.max_steps = max_steps
        self.lookback_window = lookback_window
        
        # Convert dates to datetime
        self.data['date'] = pd.to_datetime(self.data['date'])
        self.entry_points['date'] = pd.to_datetime(self.entry_points['date'])
        
        # Sort data by date
        self.data = self.data.sort_values('date').reset_index(drop=True)
        self.entry_points = self.entry_points.sort_values('date').reset_index(drop=True)
        
        # Initialize normalization parameters
        self.ema_means = {col: 0 for col in self.state_cols}
        self.ema_vars = {col: 1 for col in self.state_cols}
        self.alpha_norm = 0.005
        
        # Episode tracking
        self.total_entry_points = len(self.entry_points)
        self.delayed_exit = False
        self.pending_reward = 0
        self.pending_reason = None
        self.pending_return = 0

        print(f"Environment initialized with {len(self.data)} data points and {self.total_entry_points} entry points")
        
    def reset(self):
        """Reset environment for new episode - start from first entry point"""
        # Start from first entry point
        self.current_entry_idx = 0
        self.episode_trades = []
        self.episode_reward = 0
        self.trades_completed = 0

        self.delayed_exit = False
        self.pending_reward = 0
        self.pending_reason = None
        self.pending_return = 0
        
        # Initialize first trade
        self._start_new_trade()
        
        print(f"Episode reset: Starting with {self.total_entry_points} entry points")
        
        return self._get_state()
        
    def _start_new_trade(self):
        """Initialize a new trade at current entry point"""
        if self.current_entry_idx >= self.total_entry_points:
            self.position_open = False
            return
            
        # Get current entry point
        entry_point = self.entry_points.iloc[self.current_entry_idx]
        self.entry_date = entry_point['date']
        
        # Find entry date in market data
        entry_data_idx = self.data[self.data['date'] == self.entry_date].index
        if len(entry_data_idx) == 0:
            # Find closest date
            entry_data_idx = self.data[self.data['date'] <= self.entry_date].index
            if len(entry_data_idx) == 0:
                # Skip this entry point
                self.current_entry_idx += 1
                self._start_new_trade()
                return
            entry_data_idx = entry_data_idx[-1]
        else:
            entry_data_idx = entry_data_idx[0]
            
        self.entry_data_idx = entry_data_idx
        self.entry_price = self.data.iloc[entry_data_idx]['close']
        
        # Initialize position parameters - RESET for each new trade
        self.tp = 0.03  # 3%
        self.sl = -0.03  # -3%
        self.current_step = 0  # RESET step counter for new trade
        self.current_data_idx = entry_data_idx
        self.position_open = True
        
        print(f"  Trade {self.trades_completed + 1}: Entry date {self.entry_date}, Entry price {self.entry_price:.4f}")
        
    def _get_state(self):
        if not self.position_open:
            return np.zeros((self.lookback_window, len(self.state_cols)))
            
        # Get lookback window data ending at current_data_idx
        end_idx = self.current_data_idx
        start_idx = max(0, end_idx - self.lookback_window + 1)
        
        # Get state data
        state_data = self.data.iloc[start_idx:end_idx + 1][self.state_cols].values
        
        # Pad if insufficient data
        if len(state_data) < self.lookback_window:
            padded_state = np.zeros((self.lookback_window, len(self.state_cols)))
            padded_state[-len(state_data):] = state_data
            state_data = padded_state
        
        # Z-score normalization with EMA
        normalized_state = np.zeros_like(state_data)
        for i, col in enumerate(self.state_cols):
            for t in range(state_data.shape[0]):
                value = state_data[t, i]
                if not np.isnan(value):
                    self.ema_means[col] = (1 - self.alpha_norm) * self.ema_means[col] + self.alpha_norm * value
                    self.ema_vars[col] = (1 - self.alpha_norm) * self.ema_vars[col] + self.alpha_norm * (value - self.ema_means[col])**2
                    normalized_value = (value - self.ema_means[col]) / (np.sqrt(self.ema_vars[col]) + 1e-8)
                    normalized_state[t, i] = normalized_value
                    
        return normalized_state
        
    def step(self, action):
        if self.delayed_exit:
            self._close_trade(self.pending_return, self.pending_reason)
            self.delayed_exit = False
            return self._get_next_state_or_episode_end(self.pending_reward)

        if not self.position_open:
            # Episode is done, return terminal state
            return np.zeros((self.lookback_window, len(self.state_cols))), 0, True, {
                'trades_completed': self.trades_completed,
                'episode_trades': self.episode_trades,
                'reason': 'all_trades_completed'
            }
        
        # Action is a decimal value from action_values (e.g., 0.02 for 2%)
        action_value = action
        
        # Advance time by 1 day FIRST
        self.current_step += 1
        
        next_data_idx = self.entry_data_idx + self.current_step
        
        # Check if we have more data
        if next_data_idx >= len(self.data):
            # No more data, close position at current price
            current_price = self.data.iloc[self.current_data_idx]['close']
            ret = (current_price - self.entry_price) / self.entry_price
            reward = self._calculate_reward(ret, tp_sl_hit=False)
            self._close_trade(ret, 'no_more_data')
            return self._get_next_state_or_episode_end(reward)
        
        self.current_data_idx = next_data_idx
        current_price = self.data.iloc[self.current_data_idx]['close']
        
        # Calculate current return
        ret = (current_price - self.entry_price) / self.entry_price
        
        # Check if TP or SL hit BEFORE adjusting them
        trade_done = False
        tp_sl_hit = False
        reason = 'continuing'
        
        if ret >= self.tp:
            self.delayed_exit = True
            self.pending_return = ret
            self.pending_reason = 'tp_hit'
            self.pending_reward = self._calculate_reward(ret, tp_sl_hit=True)
            print(f"    TP hit (delayed exit)! Return: {ret:.4f}, TP: {self.tp:.4f}")
            return self._get_state(), 0, False, {
                'return': ret,
                'tp': self.tp,
                'sl': self.sl,
                'step': self.current_step,
                'delayed_exit': True
            }

        elif ret <= self.sl:
            self.delayed_exit = True
            self.pending_return = ret
            self.pending_reason = 'sl_hit'
            self.pending_reward = self._calculate_reward(ret, tp_sl_hit=True)
            print(f"    SL hit (delayed exit)! Return: {ret:.4f}, SL: {self.sl:.4f}")
            return self._get_state(), 0, False, {
                'return': ret,
                'tp': self.tp,
                'sl': self.sl,
                'step': self.current_step,
                'delayed_exit': True
            }

        elif self.current_step >= self.max_steps:
            trade_done = True
            reason = 'max_steps'
            print(f"    Max steps reached! Return: {ret:.4f}")
            
        # Calculate reward
        reward = self._calculate_reward(ret, tp_sl_hit)
        
        if trade_done:
            self._close_trade(ret, reason)
            return self._get_next_state_or_episode_end(reward)
        else:
            # ONLY if trade continues, THEN adjust TP and SL asymmetrically
            # Both TP and SL move by the same absolute amount (asymmetric window)
            center_displacement = action_value
            self.tp = center_displacement + 0.03
            self.sl = center_displacement - 0.03
            
            # Continue current trade
            next_state = self._get_state()
            return next_state, reward, False, {
                'return': ret, 
                'reason': reason, 
                'tp': self.tp, 
                'sl': self.sl,
                'trade_num': self.trades_completed + 1,
                'step': self.current_step
            }
            
    def _close_trade(self, final_return, reason):
        """Close current trade and record results"""
        trade_info = {   
            'entry_date': self.entry_date,
            'entry_price': self.entry_price,
            'final_return': final_return,
            'steps': self.current_step,
            'reason': reason
        }
        self.episode_trades.append(trade_info)
        self.trades_completed += 1
        
        print(f"    Trade {self.trades_completed} completed: Return {final_return:.4f}, Steps {self.current_step}, Reason: {reason}")
        
        # Move to next entry point
        self.current_entry_idx += 1
        self._start_new_trade()
        
    def _get_next_state_or_episode_end(self, reward):
        """Get next state or signal episode end"""
        episode_done = not self.position_open  # Episode done when no more positions
        
        if episode_done:
            # Calculate episode statistics
            total_return = sum(trade['final_return'] for trade in self.episode_trades)
            avg_return = total_return / len(self.episode_trades) if self.episode_trades else 0
            
            next_state = np.zeros((self.lookback_window, len(self.state_cols)))
            return next_state, reward, True, {
                'trades_completed': self.trades_completed,
                'total_return': total_return,
                'avg_return': avg_return,
                'episode_trades': self.episode_trades,
                'reason': 'all_trades_completed'
            }
        else:
            # Continue with next trade
            next_state = self._get_state()
            return next_state, reward, False, {
                'trade_num': self.trades_completed,
                'trades_completed': self.trades_completed,
                'continuing_to_next_trade': True
            }
            
    def _calculate_reward(self, ret, tp_sl_hit):
        """
        Reward function: shaped by exp(return/10) - 1,
        and adds +ret only when TP or SL is hit.
        """
        time_reward = ((15 - self.current_step)/15)*0.5
        shaped = (np.exp(ret) - 1)*0.1 + time_reward
        bonus = ret if tp_sl_hit else 0
        return shaped + bonus


# Training Loop
def train_sac(data_file, entry_points_file, episodes=1000, batch_size=64):
    """
    Train the SAC agent on trading data.
    
    Args:
        data_file (str): Path to market data CSV file
        entry_points_file (str): Path to entry signals CSV file
        episodes (int): Number of training episodes
        batch_size (int): Batch size for training updates
    """
    env = TradingEnvironment(data_file, entry_points_file)
    state_dim = len(env.state_cols)
    action_dim = 15  # Discrete actions: [-7%, -6%, ..., 0%, ..., +6%, +7%]
    
    agent = SACAgent(state_dim, action_dim)
    replay_buffer = PrioritizedReplayBuffer(capacity=100000)
    
    episode_rewards = []
    episode_total_returns = []
    
    for episode in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step_count = 0
        
        # Linearly anneal beta from 0.4 to 1.0
        beta = 0.4 + (1.0 - 0.4) * episode / episodes
        
        while not done:
            # Select action
            action, action_idx = agent.select_action(state)
            
            # Take step
            next_state, reward, done, info = env.step(action)
            
            # Store transition
            replay_buffer.push(state, action_idx, reward, next_state, done)
            
            # Update agent
            if len(replay_buffer) > batch_size: 
                agent.update(replay_buffer, batch_size, beta)
                
            state = next_state
            episode_reward += reward
            step_count += 1
            
            # Print step info if not episode end
            if not done and 'trade_num' in info:
                print(f"    Step {info.get('step', 0)}: Action {action:.3f}, Reward {reward:.4f}, Return {info.get('return', 0):.4f}, TP {info.get('tp', 0):.3f}, SL {info.get('sl', 0):.3f}")
            
        episode_rewards.append(episode_reward)
        
        # Episode summary
        total_return = info.get('total_return', 0)
        avg_return = info.get('avg_return', 0)
        trades_completed = info.get('trades_completed', 0)
        
        episode_total_returns.append(total_return)
        
        print(f"\nEpisode {episode + 1}/{episodes} Summary:")
        print(f"  Total Reward: {episode_reward:.4f}")
        print(f"  Trades Completed: {trades_completed}")
        print(f"  Total Return: {total_return:.4f}")
        print(f"  Average Return per Trade: {avg_return:.4f}")
        print(f"  Total Steps: {step_count}")
        
        # Print running averages
        if episode >= 4:
            avg_reward = np.mean(episode_rewards[-5:])
            avg_total_return = np.mean(episode_total_returns[-5:])
            print(f"  Last 5 episodes - Avg Reward: {avg_reward:.4f}, Avg Total Return: {avg_total_return:.4f}")
        
        print("=" * 100)
        
        # Save weights periodically
        if (episode + 1) % 100 == 0:
            os.makedirs('models', exist_ok=True)
            torch.save(agent.actor.state_dict(), f'models/actor_weights_ep{episode + 1}.pth')
            torch.save(agent.critic_1.state_dict(), f'models/critic1_weights_ep{episode + 1}.pth')
            torch.save(agent.critic_2.state_dict(), f'models/critic2_weights_ep{episode + 1}.pth')
            print(f"Saved model weights at episode {episode + 1}")
    
    # Save final weights after training completion
    os.makedirs('models', exist_ok=True)
    torch.save(agent.actor.state_dict(), f'models/final_actor_weights.pth')
    torch.save(agent.critic_1.state_dict(), f'models/final_critic1_weights.pth')
    torch.save(agent.critic_2.state_dict(), f'models/final_critic2_weights.pth')
    torch.save(agent.target_critic_1.state_dict(), f'models/final_target_critic1_weights.pth')
    torch.save(agent.target_critic_2.state_dict(), f'models/final_target_critic2_weights.pth')
    
    # Save complete model state (including optimizers and hyperparameters)
    torch.save({
        'episode': episodes,
        'actor_state_dict': agent.actor.state_dict(),
        'critic_1_state_dict': agent.critic_1.state_dict(),
        'critic_2_state_dict': agent.critic_2.state_dict(),
        'target_critic_1_state_dict': agent.target_critic_1.state_dict(),
        'target_critic_2_state_dict': agent.target_critic_2.state_dict(),
        'actor_optimizer_state_dict': agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': agent.critic_optimizer.state_dict(),
        'alpha_optimizer_state_dict': agent.alpha_optimizer.state_dict(),
        'log_alpha': agent.log_alpha,
        'episode_rewards': episode_rewards,
        'episode_total_returns': episode_total_returns,
    }, 'models/complete_model_checkpoint.pth')
    
    print(f"\nTraining completed! Final weights saved:")
    print(f"  - models/final_actor_weights.pth")
    print(f"  - models/final_critic1_weights.pth") 
    print(f"  - models/final_critic2_weights.pth")
    print(f"  - models/final_target_critic1_weights.pth")
    print(f"  - models/final_target_critic2_weights.pth")
    print(f"  - models/complete_model_checkpoint.pth (full checkpoint)")

    return episode_rewards, episode_total_returns


def main():
    """Main function with command line arguments."""
    parser = argparse.ArgumentParser(description='Train SAC Trading Agent')
    parser.add_argument('--data', type=str, required=True, 
                       help='Path to market data CSV file')
    parser.add_argument('--signals', type=str, required=True,
                       help='Path to entry signals CSV file')
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes (default: 1000)')
    parser.add_argument('--batch_size', type=int, default=64,
                       help='Batch size for training (default: 64)')
    
    args = parser.parse_args()
    
    print("Starting SAC Trading Agent Training...")
    print(f"Data file: {args.data}")
    print(f"Signals file: {args.signals}")
    print(f"Episodes: {args.episodes}")
    print(f"Batch size: {args.batch_size}")
    print(f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    print("=" * 100)
    
    train_sac(args.data, args.signals, args.episodes, args.batch_size)


if __name__ == "__main__":
    # Example usage (uncomment and modify paths as needed):
    # train_sac("data/market_data.csv", "data/entry_signals.csv", episodes=100)
    
    main()
