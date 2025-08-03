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

        if not
