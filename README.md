# 6G-Enabled Autonomous Traffic Management with Multi-Agent Deep Reinforcement Learning

This project implements a comprehensive simulation and training framework for 6G-enabled autonomous vehicle traffic management using Multi-Agent Deep Reinforcement Learning (MARL). The system demonstrates how 6G's ultra-low latency communication enables signal-free intersection management through intelligent vehicle coordination.

## 🚗 Project Overview

### Key Features

- **Multi-Agent Environment**: Each vehicle is an independent RL agent with local observations
- **6G Communication Simulation**: Ultra-low latency (1ms) V2V and V2I communication with network slicing
- **Signal-Free Intersections**: Reservation-based intersection management without traffic lights
- **Advanced Visualization**: Real-time Pygame and static matplotlib visualizations
- **Multiple RL Algorithms**: Support for PPO, SAC, and DQN
- **Comprehensive Metrics**: Performance analysis including safety, efficiency, and network reliability

### Performance Benefits (Based on Research)
- 40-50% reduction in intersection wait times
- 50-70% fewer accidents
- 35% lower emissions compared to traditional signalized systems

## 📁 Project Structure

```
Multi-Agent Deep RL/
├── multi_agent_traffic_env.py    # Core multi-agent environment
├── train_marl.py                 # Training script with RLlib
├── simulate.py                   # Simulation and evaluation script
├── visualizer.py                 # Visualization components
├── requirements.txt              # Dependencies
├── README.md                     # This file
├── chatGPT_code.txt             # Original prototype code
└── results/                     # Training results and checkpoints
```

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher
- CUDA-compatible GPU (optional but recommended for training)

### Setup Instructions

1. **Clone or navigate to the project directory**
   ```bash
   cd "Multi-Agent Deep RL"
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Verify installation**
   ```bash
   python multi_agent_traffic_env.py
   ```

## 🚀 Quick Start

### 1. Test the Environment
```bash
python multi_agent_traffic_env.py
```

### 2. Train MARL Agents
```bash
# Basic training with PPO (recommended for beginners)
python train_marl.py --algorithm PPO --num-vehicles 4 --iterations 200

# Advanced training with more vehicles and longer duration
python train_marl.py --algorithm PPO --num-vehicles 8 --iterations 1000 --experiment-name advanced_training
```

### 3. Run Simulation with Trained Model
```bash
# Single episode with visualization
python simulate.py --checkpoint ./results/checkpoint_000100/checkpoint-100 --algorithm PPO

# Multiple episodes for evaluation
python simulate.py --checkpoint ./results/checkpoint_000100/checkpoint-100 --episodes 10 --visualization matplotlib
```

## 📚 Detailed Usage

### Training Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--algorithm` | RL algorithm to use | PPO | PPO, SAC, DQN |
| `--num-vehicles` | Number of vehicles | 4 | 2-20 |
| `--iterations` | Training iterations | 100 | 50-5000 |
| `--checkpoint-freq` | Checkpoint frequency | 100 | 10-500 |
| `--experiment-name` | Custom experiment name | Auto-generated | Any string |

### Simulation Parameters

| Parameter | Description | Default | Options |
|-----------|-------------|---------|---------|
| `--checkpoint` | Path to trained model | Required | File path |
| `--episodes` | Number of episodes | 1 | 1-100 |
| `--visualization` | Visualization type | pygame | pygame, matplotlib, none |
| `--no-render` | Disable rendering | False | Flag |

### Environment Configuration

The environment can be configured by modifying the config in the training/simulation scripts:

```python
env_config = {
    "num_vehicles": 4,              # Number of vehicles
    "max_episode_steps": 800,       # Episode length
    "dt": 0.1,                      # Simulation timestep (seconds)
    "intersection_positions": [(0.0, 0.0)],  # Intersection locations
    "vehicle_spawn_distance": 80.0  # Spawn distance from intersection
}
```

## 🎮 Visualization Controls

### Pygame Visualization Controls
- **WASD / Arrow Keys**: Move camera
- **C**: Toggle communication links
- **R**: Toggle reservation zones
- **M**: Toggle metrics panel
- **ESC**: Exit simulation

### Visualization Elements
- **Blue vehicles**: Normal state
- **Orange vehicles**: Have intersection reservation
- **Green vehicles**: Completed trip
- **Red vehicles**: Collision occurred
- **Yellow lines**: Communication links
- **White circles**: Intersection conflict zones

## 📊 Performance Metrics

The system tracks comprehensive metrics:

### Traffic Efficiency
- **Completion Rate**: Percentage of vehicles reaching their destination
- **Average Travel Time**: Mean time to complete trips
- **Average Waiting Time**: Time spent in slow/stopped state
- **Intersection Throughput**: Vehicles served per unit time

### Safety Metrics
- **Collision Rate**: Percentage of episodes with collisions
- **Near-miss Events**: Close encounters between vehicles
- **Safety Distance Violations**: Instances of inadequate spacing

### Network Performance
- **Communication Reliability**: Successful message delivery rate
- **Network Latency**: Average message delivery time
- **Message Volume**: Total communication overhead

## 🧪 Experiment Examples

### Basic Performance Comparison
```bash
# Train baseline model
python train_marl.py --algorithm PPO --num-vehicles 4 --iterations 500 --experiment-name baseline

# Train advanced model with more vehicles
python train_marl.py --algorithm PPO --num-vehicles 8 --iterations 500 --experiment-name advanced

# Evaluate both models
python simulate.py --checkpoint ./results/baseline/final_model --episodes 20 --visualization matplotlib
python simulate.py --checkpoint ./results/advanced/final_model --episodes 20 --visualization matplotlib
```

### Algorithm Comparison
```bash
# Train with different algorithms
python train_marl.py --algorithm PPO --iterations 300 --experiment-name ppo_test
python train_marl.py --algorithm SAC --iterations 300 --experiment-name sac_test

# Compare performance
python simulate.py --checkpoint ./results/ppo_test/final_model --episodes 10
python simulate.py --checkpoint ./results/sac_test/final_model --episodes 10
```

### Scalability Testing
```bash
# Test with increasing number of vehicles
for vehicles in 4 6 8 10; do
    python train_marl.py --algorithm PPO --num-vehicles $vehicles --iterations 200 --experiment-name "scale_${vehicles}v"
    python simulate.py --checkpoint "./results/scale_${vehicles}v/final_model" --episodes 5 --no-render
done
```

## 📈 Expected Results

### Training Progress
- **Initial Phase** (0-50 iterations): Agents learn basic collision avoidance
- **Learning Phase** (50-200 iterations): Coordination strategies emerge
- **Optimization Phase** (200+ iterations): Fine-tuning for efficiency

### Performance Targets
- **Completion Rate**: >90% (excellent performance)
- **Collision Rate**: <5% (acceptable safety level)
- **Network Reliability**: >99% (6G performance standard)

### Convergence Indicators
- Stable completion rates over 20+ episodes
- Decreasing collision rates
- Consistent travel times

## 🐛 Troubleshooting

### Common Issues

1. **CUDA/GPU Issues**
   ```bash
   # Force CPU training
   export CUDA_VISIBLE_DEVICES=""
   python train_marl.py --num-cpus 4 --num-gpus 0
   ```

2. **Memory Issues**
   ```bash
   # Reduce batch size and workers
   # Modify config in train_marl.py:
   # train_batch_size=1000, num_rollout_workers=2
   ```

3. **Pygame Display Issues**
   ```bash
   # Use matplotlib instead
   python simulate.py --visualization matplotlib
   ```

4. **Ray Initialization Errors**
   ```bash
   # Kill existing Ray processes
   ray stop
   python train_marl.py
   ```

### Performance Optimization

1. **Faster Training**
   - Use GPU acceleration
   - Increase number of workers
   - Reduce evaluation frequency

2. **Better Convergence**
   - Tune learning rate (3e-4 to 1e-3)
   - Adjust reward function weights
   - Use curriculum learning (start with fewer vehicles)

## 🔬 Research Applications

### Academic Use Cases
- Multi-agent coordination research
- 6G communication protocol evaluation
- Autonomous vehicle behavior analysis
- Urban traffic optimization studies

### Industry Applications
- Smart city traffic management
- Autonomous vehicle testing
- Communication network planning
- Traffic signal optimization

## 📝 Customization

### Adding New Vehicle Types
```python
# In multi_agent_traffic_env.py
class EmergencyVehicle(Vehicle):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.priority = "emergency"
        self.max_speed = 25.0  # Higher speed limit
```

### Custom Reward Functions
```python
def custom_reward(self, intersections, other_vehicles, dt):
    reward = 0.0
    # Add custom reward logic
    reward += self.get_efficiency_reward()
    reward += self.get_cooperation_reward()
    return reward
```

### Multiple Intersections
```python
env_config = {
    "intersection_positions": [
        (0.0, 0.0),      # Central intersection
        (100.0, 0.0),    # East intersection
        (0.0, 100.0),    # North intersection
        (100.0, 100.0)   # Northeast intersection
    ]
}
```

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@software{6g_traffic_marl,
  title={6G-Enabled Autonomous Traffic Management with Multi-Agent Deep Reinforcement Learning},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/6g-traffic-marl}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/new-feature`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/new-feature`)
5. Create a Pull Request

## 📧 Support

For questions and support:
- Create an issue in the repository
- Check the troubleshooting section
- Review the code documentation

## 🔮 Future Enhancements

- [ ] Integration with SUMO traffic simulator
- [ ] Real-world vehicle dynamics models
- [ ] Machine learning-based 6G network modeling
- [ ] Multi-intersection coordination
- [ ] Emergency vehicle prioritization
- [ ] Weather and environmental effects
- [ ] Real-time traffic data integration

---

**Note**: This simulation is designed for research and educational purposes. Real-world deployment would require extensive safety testing and regulatory approval. 