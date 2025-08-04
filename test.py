#delayed output test code as per CHATGPT
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
    def __init__(self, state_dim, action_dim, sequence_length=15, hidden_dim=128, lr=3e-4, gamma=0.99, tau=0.005):
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
        probs = torch.softmax(logits, dim=1).squeeze().detach().cpu().numpy()

    
        # Define action classes based on their values
        pos_indices = [i for i, val in enumerate(self.action_values) if val > 0]
        zero_indices = [i for i, val in enumerate(self.action_values) if val == 0]
        neg_indices = [i for i, val in enumerate(self.action_values) if val < 0]
    
        # Compute total class probabilities
        pos_prob = sum(probs[i] for i in pos_indices)
        zero_prob = sum(probs[i] for i in zero_indices)
        neg_prob = sum(probs[i] for i in neg_indices)
    
        class_probs = [pos_prob, zero_prob, neg_prob]
        class_indices = [pos_indices, zero_indices, neg_indices]
    
        if deterministic:
            # Custom two-stage sampling
            if max(class_probs) > 0.5:
                # Deterministically choose the dominant class
                chosen_class_idx = np.argmax(class_probs)
            else:
                # Stochastically choose among classes
                total = sum(class_probs)
                if total == 0:  # handle edge case
                    chosen_class_idx = np.random.choice([0, 1, 2])
                else:
                    class_probs_normalized = np.array(class_probs) / total
                    chosen_class_idx = np.random.choice([0, 1, 2], p=class_probs_normalized)
    
            chosen_indices = class_indices[chosen_class_idx]
            if not chosen_indices:
                # fallback in case of empty class
                action_idx = np.argmax(probs)
            else:
             within_class_probs = np.array([probs[i] for i in chosen_indices])
             within_class_probs /= within_class_probs.sum()  # normalize
             action_idx = np.random.choice(chosen_indices, p=within_class_probs)
        else:
            # Standard stochastic policy for training
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

# ----- MODIFIED: TRADING ENVIRONMENT WITH REALISTIC TIME LOGIC -----
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

        self.delayed_exit = False
        self.pending_reward = 0
        self.pending_reason = None
        self.pending_return = 0

        
        self.total_entry_points = len(self.entry_points)
        print(f"Environment initialized with {len(self.data)} data points and {self.total_entry_points} entry points")
        
    def reset(self):
        """Reset environment for new episode, respecting realistic time flow."""
        # Initialize last exit date to a very early time to find the first trade.
        self.last_exit_date = pd.Timestamp.min
        self.episode_trades = []
        self.episode_reward = 0
        self.trades_completed = 0
        self.delayed_exit = False
        self.pending_reward = 0
        self.pending_reason = None
        self.pending_return = 0

        
        # Start the first trade
        self._start_new_trade()
        
        print(f"Episode reset: Starting with {self.total_entry_points} entry points, following time.")
        
        return self._get_state()
        
    def _start_new_trade(self):
        """Initialize a new trade by finding the next available entry point in time."""
        # Find the first entry point that occurs AFTER the last trade's exit.
        valid_entry_points = self.entry_points[self.entry_points['date'] > self.last_exit_date]
        
        if valid_entry_points.empty:
            # No more available entry points in the future. End of episode.
            self.position_open = False
            return
            
        # Get the next available entry point from the list
        entry_point = valid_entry_points.iloc[0]
        self.entry_date = entry_point['date']
        
        # Find the market data index for this entry date.
        # Use >= to find the first available trading day on or after the signal.
        entry_data_idx = self.data[self.data['date'] >= self.entry_date].index
        if len(entry_data_idx) == 0:
            # No data available on or after the entry point date. End of episode.
            self.position_open = False
            return
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
        
        action_value = action
        self.current_step += 1
        next_data_idx = self.entry_data_idx + self.current_step
        
        if next_data_idx >= len(self.data):
            current_price = self.data.iloc[self.current_data_idx]['close']
            ret = (current_price - self.entry_price) / self.entry_price
            reward = self._calculate_reward(ret, tp_sl_hit=False)
            self._close_trade(ret, 'no_more_data')
            return self._get_next_state_or_episode_end(reward)
        
        self.current_data_idx = next_data_idx
        current_price = self.data.iloc[self.current_data_idx]['close']
        ret = (current_price - self.entry_price) / self.entry_price
        
        trade_done, tp_sl_hit, reason = False, False, 'continuing'
        
        if ret >= self.tp:
            self.delayed_exit = True
            self.pending_return = ret
            self.pending_reason = 'tp_hit'
        
            print(f"    TP hit (delayed exit)! Return: {ret:.4f}, TP: {self.tp:.4f}")
            return self._get_state(), 0, False, {
        'return': ret, 'tp': self.tp, 'sl': self.sl,
        'step': self.current_step, 'delayed_exit': True
    }

        elif ret <= self.sl:
            self.delayed_exit = True
            self.pending_return = ret
            self.pending_reason = 'sl_hit'
            
            print(f"    SL hit (delayed exit)! Return: {ret:.4f}, SL: {self.sl:.4f}")
            return self._get_state(), 0, False, {
        'return': ret, 'tp': self.tp, 'sl': self.sl,
        'step': self.current_step, 'delayed_exit': True
    }

        elif self.current_step >= self.max_steps:
            trade_done, reason = True, 'max_steps'
            print(f"    Max steps reached! Return: {ret:.4f}")
            
        reward = self._calculate_reward(ret, tp_sl_hit)
        
        if trade_done:
            self._close_trade(ret, reason)
            return self._get_next_state_or_episode_end(reward)
        else:
            center_displacement = action_value
            self.tp = center_displacement + 0.03
            self.sl = center_displacement - 0.03
            next_state = self._get_state()
            return next_state, reward, False, {
                'return': ret, 'reason': reason, 'tp': self.tp, 'sl': self.sl,
                'trade_num': self.trades_completed + 1, 'step': self.current_step
            }
            
    def _close_trade(self, final_return, reason):
        """Close current trade, record results, and set the exit time for the next trade search."""
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
        
        # Record the exit date. The next trade must start AFTER this date.
        self.last_exit_date = self.data.iloc[self.current_data_idx]['date']
        
        # Look for the next trade in time
        self._start_new_trade()
        
    def _get_next_state_or_episode_end(self, reward):
        episode_done = not self.position_open
        if episode_done:
            total_return = sum(trade['final_return'] for trade in self.episode_trades)
            avg_return = total_return / len(self.episode_trades) if self.episode_trades else 0
            next_state = np.zeros((self.lookback_window, len(self.state_cols)))
            return next_state, reward, True, {
                'trades_completed': self.trades_completed, 'total_return': total_return,
                'avg_return': avg_return, 'episode_trades': self.episode_trades,
                'reason': 'all_trades_completed'
            }
        else:
            next_state = self._get_state()
            return next_state, reward, False, {
                'trade_num': self.trades_completed + 1, 'trades_completed': self.trades_completed,
                'continuing_to_next_trade': True
            }
        
    def _calculate_reward(self, ret, tp_sl_hit):
        reward = np.exp(ret/10) - 1 + ret
        return reward

# ----- PERFORMANCE METRICS CALCULATION -----
def calculate_and_print_performance_metrics(initial_portfolio, trades):
    """
    Calculates and prints performance metrics based on trade results.
    """
    if not trades:
        print("\nNo trades were made, cannot calculate performance metrics.")
        return

    portfolio_value = initial_portfolio
    portfolio_history = [initial_portfolio]
    trade_returns = []

    for trade in trades:
        final_return = trade['final_return']
        trade_returns.append(final_return)
        portfolio_value *= (1 + final_return)
        portfolio_history.append(portfolio_value)
    
    portfolio_history = np.array(portfolio_history)
    trade_returns = np.array(trade_returns)
    
    final_portfolio_value = portfolio_history[-1]
    total_return_pct = (final_portfolio_value - initial_portfolio) / initial_portfolio

    peaks = np.maximum.accumulate(portfolio_history)
    drawdowns = (peaks - portfolio_history) / peaks
    max_drawdown_pct = np.max(drawdowns)
    
    if len(trades) > 1:
        start_date = trades[0]['entry_date']
        end_date_last_trade = trades[-1]['entry_date'] + timedelta(days=trades[-1]['steps'])
        total_days = (end_date_last_trade - start_date).days
        annual_trading_freq = len(trades) / (total_days / 365.25) if total_days > 0 else 1.0
    else:
        annual_trading_freq = 1.0
        total_days = sum(t['steps'] for t in trades)

    if np.std(trade_returns) > 1e-8:
        sharpe_ratio = np.mean(trade_returns) / np.std(trade_returns)
        annualized_sharpe_ratio = sharpe_ratio * np.sqrt(annual_trading_freq)
    else:
        sharpe_ratio = float('inf') if np.mean(trade_returns) > 0 else 0
        annualized_sharpe_ratio = float('inf') if np.mean(trade_returns) > 0 else 0
        
    print("\n\n" + "=" * 60)
    print("PORTFOLIO PERFORMANCE METRICS")
    print("=" * 60)
    print(f"  Initial Portfolio Value: ${initial_portfolio:,.2f}")
    print(f"  Final Portfolio Value:   ${final_portfolio_value:,.2f}")
    print(f"  Total Portfolio Return:  {total_return_pct:.2%}")
    print(f"  Max Drawdown:            {max_drawdown_pct:.2%}")
    print(f"  Annualized Sharpe Ratio: {annualized_sharpe_ratio:.2f} (est. from {len(trades)} trades over ~{int(total_days)} days)")
    print("=" * 60)


# ----- TESTING SCRIPT -----
def test_sac(data_file, entry_points_file, actor_weights_path, initial_portfolio=10000.0):
    env = TradingEnvironment(data_file, entry_points_file)
    state_dim = len(env.state_cols)
    action_dim = 15
    agent = SACAgent(state_dim, action_dim)
    
    try:
        agent.actor.load_state_dict(torch.load(actor_weights_path, map_location=agent.device))
        agent.actor.eval()
        print(f"\nSuccessfully loaded actor weights from {actor_weights_path}")
    except FileNotFoundError:
        print(f"Error: Actor weights file not found at {actor_weights_path}")
        return
    except Exception as e:
        print(f"An error occurred while loading the weights: {e}")
        return

    state = env.reset()
    done = False
    step_count = 0
    
    print("\nStarting Testing Loop...")
    print("=" * 100)
    
    while not done:
        action, action_idx = agent.select_action(state, deterministic=True)
        next_state, reward, done, info = env.step(action)
        state = next_state
        step_count += 1
        
        if not done and 'trade_num' in info:
            print(f"    Step {info.get('step', 0)}: Action {action:.3f}, Return {info.get('return', 0):.4f}, TP {info.get('tp', 0):.3f}, SL {info.get('sl', 0):.3f}")

    print("\nTesting Finished!")
    print("=" * 100)
    
    trades_completed = info.get('trades_completed', 0)
    all_trades = info.get('episode_trades', [])
    
    print("\nTesting Summary:")
    print(f"  Trades Completed: {trades_completed}")
    print(f"  Total Steps: {step_count}")

    if all_trades:
        results_df = pd.DataFrame(all_trades)
        results_df.to_csv("test_results.csv", index=False)
        print("\nTest results saved to test_results.csv")
        print(results_df)
        calculate_and_print_performance_metrics(initial_portfolio, all_trades)

if __name__ == "__main__":
    TEST_DATA_FILE = "/kaggle/input/isthisforeal/test_data18alt.csv"
    TEST_ENTRY_POINTS_FILE = "/kaggle/input/againwego/minima_outputstrat.csv"
    ACTOR_WEIGHTS_FILE = "/kaggle/working/final_actor_weights.pth"
    INITIAL_PORTFOLIO_VALUE = 10000.0
    
    if not os.path.exists(TEST_DATA_FILE):
        print(f"Test data file not found: {TEST_DATA_FILE}")
    elif not os.path.exists(TEST_ENTRY_POINTS_FILE):
        print(f"Test entry points file not found: {TEST_ENTRY_POINTS_FILE}")
    elif not os.path.exists(ACTOR_WEIGHTS_FILE):
         print(f"Actor weights file not found: {ACTOR_WEIGHTS_FILE}")
    else:
        test_sac(
            data_file=TEST_DATA_FILE,
            entry_points_file=TEST_ENTRY_POINTS_FILE,
            actor_weights_path=ACTOR_WEIGHTS_FILE,
            initial_portfolio=INITIAL_PORTFOLIO_VALUE
        )
