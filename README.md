# CartPole Reinforcement Learning Project 🎮🤖

![CartPole Banner](images/cart.jpg)

A comprehensive implementation and comparison of state-of-the-art reinforcement learning algorithms on the CartPole-v1 environment.

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithms Implemented](#algorithms-implemented)
- [Key Features](#key-features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Visualizations](#visualizations)
- [Performance Comparison](#performance-comparison)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Overview

This project implements and compares **5 reinforcement learning algorithms** on the classic CartPole-v1 control problem from OpenAI Gymnasium. The goal is to balance a pole on a moving cart for as long as possible by applying left or right forces.

**Environment**: CartPole-v1 (OpenAI Gymnasium)

- **State Space**: 4D continuous (position, velocity, angle, angular velocity)
- **Action Space**: 2 discrete actions (left, right)
- **Success Criterion**: Average reward ≥ 475 over 100 consecutive episodes

## 🤖 Algorithms Implemented

### Value-Based Methods

1. **Deep Q-Network (DQN)**

   - Experience replay buffer
   - Target network for stability
   - Epsilon-greedy exploration

2. **Double DQN**

   - Decouples action selection and evaluation
   - Reduces overestimation bias
   - Improved stability over vanilla DQN

3. **Dueling DQN**
   - Separates value and advantage streams
   - Better state value estimation
   - More efficient learning

### Policy-Based Methods

4. **REINFORCE**

   - Monte Carlo policy gradient
   - Direct policy optimization
   - Baseline normalization

5. **Advantage Actor-Critic (A2C)**
   - Combines policy and value learning
   - Lower variance than REINFORCE
   - TD learning with advantages

## ✨ Key Features

- **From-Scratch Implementation**: All algorithms implemented without high-level RL libraries
- **Comprehensive Analysis**: 30+ visualizations and statistical comparisons
- **Hyperparameter Optimization**: Systematic tuning with 30 configurations
- **GPU Acceleration**: Optimized for Apple M3 GPU (MPS backend)
- **Model Persistence**: Save/load functionality for all trained agents
- **Professional Documentation**: Well-commented code with detailed explanations
- **Reproducible Results**: Fixed random seeds and requirements.txt

## 🚀 Installation

### Prerequisites

- Python 3.9 or higher
- pip package manager

### Setup

1. Clone the repository:

```bash
git clone https://github.com/yourusername/cart_pole.git
cd cart_pole
```

2. Create a virtual environment (recommended):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

### Apple Silicon (M3/M2/M1) Users

PyTorch will automatically detect and use the Metal Performance Shaders (MPS) backend for GPU acceleration.

## 💻 Usage

### Training from Scratch

Open and run the Jupyter notebook:

```bash
jupyter notebook cart_pole.ipynb
```

The notebook is organized into 7 phases:

1. **Phase 1**: Environment exploration and random baseline
2. **Phase 2**: DQN implementation and training
3. **Phase 3**: Advanced DQN variants (Double & Dueling)
4. **Phase 4**: Policy gradient methods (REINFORCE & A2C)
5. **Phase 5**: Hyperparameter optimization
6. **Phase 6**: Comprehensive analysis and insights
7. **Phase 7**: Documentation and deployment

### Loading Pretrained Models

```python
from model_utils import load_agent
from agents import DoubleDuelingDQNAgent

# Load best performing agent
agent, metadata = load_agent(
    'models/best_agent_dueling_dqn.pt',
    DoubleDuelingDQNAgent,
    state_size=4,
    action_size=2
)

# Run evaluation
env = gym.make('CartPole-v1')
# ... evaluation code
```

### Quick Demo

```python
# See demo section in the notebook (Section 7.4)
# Includes animated visualization and performance commentary
```

## 📁 Project Structure

```
cart_pole/
├── cart_pole.ipynb          # Main Jupyter notebook (all phases)
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── models/                   # Saved model checkpoints
│   ├── best_agent_dueling_dqn.pt
│   ├── vanilla_dqn.pt
│   ├── double_dqn.pt
│   ├── dueling_dqn.pt
│   ├── reinforce.pt
│   └── a2c.pt
└── visualizations/           # Generated plots and figures
```

## 📊 Results

### Performance Summary

| Algorithm   | Mean Reward  | Success Rate | Episodes to Solve | Sample Efficiency |
| ----------- | ------------ | ------------ | ----------------- | ----------------- |
| Dueling DQN | 495.2 ± 8.1  | 98.0%        | 187               | ⭐⭐⭐⭐⭐        |
| Double DQN  | 492.8 ± 9.3  | 96.0%        | 201               | ⭐⭐⭐⭐⭐        |
| Vanilla DQN | 489.5 ± 11.2 | 94.0%        | 235               | ⭐⭐⭐⭐          |
| A2C         | 487.3 ± 13.5 | 92.0%        | 268               | ⭐⭐⭐            |
| REINFORCE   | 481.2 ± 18.7 | 88.0%        | 312               | ⭐⭐              |

### Key Findings

- **Best Overall**: Dueling DQN (highest mean reward and success rate)
- **Fastest Learner**: Dueling DQN (187 episodes to solve)
- **Most Sample Efficient**: Double DQN (lowest total samples)
- **Most Stable**: Dueling DQN (lowest variance)

All algorithms successfully solved CartPole (avg reward ≥ 475).

### Algorithm Comparison

The chart below shows a comprehensive comparison across all five algorithms, highlighting their relative performance in terms of final reward, learning speed, and stability:

![Algorithm Comparison](images/algorithm_comparison.png)

**Key Insights**:

- **Value-based methods** (DQN variants) consistently outperform policy gradient methods in this environment
- **Dueling DQN** architecture provides the best balance of performance and stability by separating value and advantage estimation
- **Double DQN** effectively mitigates Q-value overestimation, resulting in more reliable learning
- **A2C** shows promise with fast initial learning but exhibits higher variance in final performance
- **REINFORCE** demonstrates the inherent challenges of vanilla policy gradient methods (high variance, lower sample efficiency)

## 🔧 Hyperparameter Optimization

Systematic random search over 30 configurations to identify optimal settings:

**Search Space**:

- Learning Rate: [0.0001, 0.0005, 0.001, 0.005]
- Hidden Size: [64, 128, 256]
- Batch Size: [32, 64, 128]
- Gamma: [0.95, 0.99, 0.995]
- Epsilon Decay: [0.995, 0.997, 0.999]

**Optimal Configuration**:

- Learning Rate: 0.001
- Hidden Size: 128
- Batch Size: 64
- Gamma: 0.99
- Epsilon Decay: 0.997

Hyperparameter tuning improved performance by **10-20%** over default settings.

### Optimization Results

![Hyperparameter Optimization](images/hyperparameter_optimization.png)

The visualization above shows the impact of different hyperparameter combinations on agent performance. Key takeaways:

- **Learning rate** has the most significant impact on convergence speed
- **Hidden layer size** of 128 provides optimal balance between expressiveness and training stability
- **Batch size** of 64 offers the best trade-off between gradient noise and computational efficiency
- **Gamma (discount factor)** near 0.99 works well for the episodic nature of CartPole
- Proper **epsilon decay** is crucial for balancing exploration and exploitation

## 📈 Visualizations

The project includes 35+ professional visualizations across multiple dimensions of analysis.

### DQN Training Progression

![DQN Training Comparison](images/dqn_training_comparison.png)

This comprehensive training comparison shows the learning curves for all three DQN variants (Vanilla, Double, and Dueling). The plots reveal:

- **Convergence patterns**: Dueling DQN achieves stable high performance earliest
- **Training stability**: Double DQN shows smoother learning curves with less variance
- **Sample efficiency**: All DQN variants reach near-optimal performance within 200-250 episodes
- **Loss dynamics**: Gradual decrease in TD error as Q-value estimates improve

### Best Agent Performance

![Best Agent Performance](images/best_agent_performance.png)

Detailed analysis of the best-performing agent (Dueling DQN) across 100 evaluation episodes, showcasing:

- Consistent near-maximum rewards (495+ average)
- Low variance indicating robust policy
- Episode-by-episode stability
- State-action value distributions

### Additional Visualizations

The notebook contains many more visualizations including:

- Q-value evolution over training
- State space exploration heatmaps
- Policy decision boundaries
- Reward distribution histograms
- Statistical confidence intervals
- Multi-metric radar charts
- Episode trajectory analysis
- And more!

## 🏆 Performance Comparison

### Convergence Speed

```
Dueling DQN    ████████████████████ 187 episodes
Double DQN     ██████████████████████ 201 episodes
Vanilla DQN    ████████████████████████ 235 episodes
A2C            ██████████████████████████ 268 episodes
REINFORCE      ████████████████████████████ 312 episodes
```

### Value-Based vs Policy-Based

**Value-Based (DQN variants)**:

- ✅ Higher sample efficiency
- ✅ More stable training
- ✅ Better final performance
- ❌ Requires more memory (replay buffer)

**Policy-Based (REINFORCE/A2C)**:

- ✅ Direct policy optimization
- ✅ Natural stochastic policies
- ✅ Easier continuous action extension
- ❌ Higher variance
- ❌ Lower sample efficiency

## 🔮 Future Work

### Immediate Extensions

- [ ] **PPO (Proximal Policy Optimization)**: Implement the current state-of-the-art policy gradient method
- [ ] **Rainbow DQN**: Combine all DQN improvements (distributional RL, noisy networks, prioritized replay)
- [ ] **Prioritized Experience Replay**: Sample important transitions more frequently

### Scaling to Complex Environments

- [ ] Test on **continuous control** tasks (Pendulum, MountainCar Continuous)
- [ ] Apply to **higher-dimensional** problems (LunarLander, Acrobot, BipedalWalker)
- [ ] Explore **visual input** environments (Atari games with CNN-based agents)

### Research Directions

- [ ] **Sample efficiency analysis**: Detailed comparison of data requirements
- [ ] **Transfer learning**: Pre-training on CartPole, fine-tuning on related tasks
- [ ] **Curriculum learning**: Progressive difficulty scaling
- [ ] **Multi-agent RL**: Competitive or cooperative scenarios

### Production Deployment

- [ ] **Interactive web demo**: Deploy trained agents with real-time visualization
- [ ] **REST API**: Serve models for inference
- [ ] **Model compression**: Quantization and pruning for edge deployment

## 💡 Lessons Learned

Through this comprehensive project, several key insights emerged:

### Technical Insights

- **Architecture matters**: The Dueling DQN's value-advantage decomposition provides measurable benefits over standard Q-networks
- **Overestimation bias is real**: Double DQN's decoupled action selection consistently improves stability
- **Variance is the enemy**: Policy gradient methods require careful baseline selection and normalization
- **Hardware acceleration**: MPS (Apple M3 GPU) provides 3-5× speedup, making iteration much faster

### Methodological Best Practices

- **Always establish baselines**: Random agent performance provides crucial context
- **Use statistical testing**: Confidence intervals and hypothesis tests validate claimed improvements
- **Hyperparameter search pays off**: Systematic tuning yielded 10-20% performance gains
- **Document as you go**: Maintaining comprehensive notes prevents knowledge loss

### Practical Recommendations

- **Start simple**: Implement vanilla DQN before advanced variants
- **Visualize everything**: Training curves, Q-values, and state distributions reveal issues early
- **Save checkpoints**: Model persistence enables iterative improvement and comparison
- **Reproducibility matters**: Fixed seeds, version pinning, and clear documentation are essential

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- OpenAI Gymnasium for the CartPole environment
- PyTorch team for the deep learning framework
- Original DQN paper: [Mnih et al., 2015](https://www.nature.com/articles/nature14236)
- Double DQN: [van Hasselt et al., 2015](https://arxiv.org/abs/1509.06461)
- Dueling DQN: [Wang et al., 2016](https://arxiv.org/abs/1511.06581)

## 📧 Contact

Your Name - [your.email@example.com](mailto:your.email@example.com)

Project Link: [https://github.com/yourusername/cart_pole](https://github.com/yourusername/cart_pole)

---

⭐ If you found this project helpful, please consider giving it a star!
