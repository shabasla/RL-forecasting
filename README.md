# RL Trading Agent with Soft Actor-Critic (SAC)

A reinforcement learning trading system that uses **Soft Actor-Critic (SAC)** with a **CNN-LSTM architecture** for dynamic take-profit and stop-loss management in financial markets.

## 🚀 Features

- **Soft Actor-Critic (SAC)** algorithm for continuous control in discrete action space
- **CNN-LSTM hybrid network** for temporal pattern recognition
- **Prioritized Experience Replay (PER)** for efficient learning
- **Dynamic TP/SL adjustment** based on market conditions
- **Delayed reward mechanism** for realistic trading simulation
- **Z-score normalization** with exponential moving averages
- **Comprehensive trading metrics** and episode tracking

## 🏗️ Architecture

### Agent Architecture
- **Actor Network**: CNN-LSTM with LayerNorm for policy learning
- **Critic Networks**: Twin critics with target networks for stable Q-learning
- **Action Space**: 15 discrete actions representing percentage adjustments (-7% to +7%)
- **State Space**: Technical indicators over 15-day lookback window

### Technical Indicators Used
- Stochastic Oscillator (`sto_osc`)
- MACD (`macd`)
- Average Directional Index (`adx`)
- On-Balance Volume (`obv`)
- Normalized Average True Range (`n_atr`)
- Log Returns (`log_ret`)
- News API sentiment (`newsapi`)

## 📦 Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended)

### Dependencies
```bash
pip install -r requirements.txt
```

### Required Packages
- `torch>=1.9.0`
- `pandas>=1.3.0`
- `numpy>=1.21.0`
- `scikit-learn>=1.0.0`

## 🎯 Usage

### Basic Training
```python
from main import train_sac

# Train the agent
train_sac(
    data_file="path/to/market_data.csv",
    entry_points_file="path/to/entry_signals.csv",
    episodes=1000,
    batch_size=64
)
```

### Data Format Requirements

#### Market Data (`data_file`)
CSV file with columns:
- `date`: Date in YYYY-MM-DD format
- `close`: Closing prices
- `sto_osc`: Stochastic Oscillator values
- `macd`: MACD indicator values
- `adx`: ADX values
- `obv`: On-Balance Volume
- `n_atr`: Normalized ATR
- `log_ret`: Log returns
- `newsapi`: News sentiment scores

#### Entry Points (`entry_points_file`)
CSV file with columns:
- `date`: Entry signal dates in YYYY-MM-DD format

### Configuration
Modify hyperparameters in the training script:
```python
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    lr=1e-4,           # Learning rate
    gamma=0.99,        # Discount factor
    tau=0.005          # Target network update rate
)
```

## 🧠 Algorithm Details

### Soft Actor-Critic (SAC)
- **Entropy-regularized RL**: Balances exploration and exploitation
- **Actor-Critic architecture**: Separate policy and value networks
- **Twin Critics**: Reduces overestimation bias
- **Automatic temperature tuning**: Adaptive entropy coefficient

### Reward Function
```python
def _calculate_reward(self, ret, tp_sl_hit):
    time_reward = ((15 - self.current_step)/15) * 0.5
    shaped = (np.exp(ret) - 1) * 0.1 + time_reward
    bonus = ret if tp_sl_hit else 0
    return shaped + bonus
```

### Action Mechanism
- **Discrete Actions**: 15 percentage adjustments from -7% to +7%
- **Dynamic TP/SL**: Asymmetric window adjustment around action center
- **Delayed Exit**: Realistic market execution simulation

## 📊 Training Process

### Episode Structure
1. **Initialization**: Load entry points and market data
2. **Trade Execution**: Sequential processing of entry signals
3. **Dynamic Adjustment**: TP/SL modification based on agent actions
4. **Trade Closure**: Automatic exit on TP/SL hit or max steps
5. **Transition Storage**: Experience replay buffer updates
6. **Network Updates**: SAC learning with prioritized sampling

### Monitoring
- Episode rewards and returns
- Trade completion statistics
- Average performance metrics
- Model checkpointing every 100 episodes

## 🔧 Key Components

### `TradingEnvironment`
- Handles market data and entry point processing
- Implements realistic trading mechanics
- Provides normalized state representations
- Manages episode progression and trade tracking

### `SACAgent`
- Implements SAC algorithm with discrete actions
- CNN-LSTM networks for temporal pattern recognition
- Prioritized experience replay integration
- Automatic hyperparameter tuning

### `PrioritizedReplayBuffer`
- Importance sampling for experience replay
- Priority updates based on TD errors
- Efficient memory management

## 📈 Results and Monitoring

### Saved Outputs
- Model weights every 100 episodes
- Final trained models
- Complete checkpoint with training history
- Episode performance logs

### Metrics Tracked
- Episode rewards and total returns
- Individual trade performance
- Average returns per trade
- Training convergence statistics

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Built using PyTorch deep learning framework
- Inspired by SAC paper: "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
- CNN-LSTM architecture for financial time series analysis

## 📚 References

- [Soft Actor-Critic Paper](https://arxiv.org/abs/1801.01290)
- [Prioritized Experience Replay](https://arxiv.org/abs/1511.05952)
- [Deep Reinforcement Learning for Trading](https://arxiv.org/abs/2106.00123)

---

**Note**: This is a research project. Always validate trading strategies thoroughly before real-world application.# RL-forecasting
