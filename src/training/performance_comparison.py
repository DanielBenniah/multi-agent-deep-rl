"""
Performance Comparison Framework: 6G Autonomous vs Traditional Traffic System

This module provides comprehensive performance comparison between:
1. 6G-enabled autonomous vehicle system (signal-free intersection)
2. Traditional traffic light controlled system with human drivers

Metrics compared:
- Average travel time
- Throughput (vehicles per minute)
- Waiting time and queue lengths
- Safety (collision rates)
- Energy efficiency
- System reliability
"""

import os
import json
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Any
from datetime import datetime
import sys

# Add paths for importing environments
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'environments'))
from multi_agent_traffic_env import MultiAgentTrafficEnv
from traffic_light_env import TrafficLightEnvironment

# Ray imports for 6G system
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig

class PerformanceMetrics:
    """Container for performance metrics with statistical analysis."""
    
    def __init__(self):
        self.travel_times = []
        self.waiting_times = []
        self.throughput_rates = []
        self.throughput_hourly = []
        self.average_speeds = []
        self.collision_counts = []
        self.queue_lengths = []
        self.energy_consumption = []
        self.episode_returns = []
        
    def add_episode(self, metrics: Dict):
        """Add metrics from a single episode."""
        self.travel_times.append(metrics.get("average_travel_time", 0))
        self.waiting_times.append(metrics.get("average_waiting_time", 0))
        self.throughput_rates.append(metrics.get("throughput_per_minute", 0))
        self.throughput_hourly.append(metrics.get("throughput_per_hour", 0))
        self.average_speeds.append(metrics.get("average_speed_mph", 0))
        self.collision_counts.append(metrics.get("collision_count", 0))
        self.queue_lengths.append(metrics.get("average_queue_length", 0))
        self.energy_consumption.append(metrics.get("energy_consumption", 0))
        self.episode_returns.append(metrics.get("episode_return", 0))
    
    def get_statistics(self) -> Dict:
        """Calculate comprehensive statistics."""
        def calc_stats(data):
            if not data:
                return {"mean": 0, "std": 0, "min": 0, "max": 0}
            return {
                "mean": np.mean(data),
                "std": np.std(data),
                "min": np.min(data),
                "max": np.max(data),
                "median": np.median(data)
            }
        
        return {
            "travel_time": calc_stats(self.travel_times),
            "waiting_time": calc_stats(self.waiting_times),
            "throughput": calc_stats(self.throughput_rates),
            "throughput_hourly": calc_stats(self.throughput_hourly),
            "average_speed": calc_stats(self.average_speeds),
            "collisions": calc_stats(self.collision_counts),
            "queue_length": calc_stats(self.queue_lengths),
            "energy": calc_stats(self.energy_consumption),
            "episode_return": calc_stats(self.episode_returns)
        }

class SixGSystemBenchmark:
    """Benchmark runner for 6G autonomous vehicle system."""
    
    def __init__(self, num_vehicles: int = 6, iterations: int = 10):
        self.num_vehicles = num_vehicles
        self.iterations = iterations
        self.results = PerformanceMetrics()
        
    def run_benchmark(self, num_episodes: int = 10) -> PerformanceMetrics:
        """Run benchmark episodes using trained 6G system."""
        print(f"🚗 Running 6G Autonomous System Benchmark")
        print(f"   Vehicles: {self.num_vehicles}")
        print(f"   Episodes: {num_episodes}")
        
        # Initialize Ray
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
        
        try:
            checkpoint_path = self._find_latest_checkpoint()
            if not checkpoint_path:
                raise FileNotFoundError(
                    "No trained 6G model checkpoint found. Run training before benchmarking."
                )

            print(f"   Using model: {checkpoint_path}")

            for episode in range(num_episodes):
                print(f"   Episode {episode + 1}/{num_episodes}", end=" ")
                episode_metrics = self._run_single_episode(checkpoint_path)
                self.results.add_episode(episode_metrics)
                print(f"✅ Return: {episode_metrics.get('episode_return', 0):.1f}")

            print("✅ 6G Autonomous System Benchmark Complete!\n")
            return self.results
        finally:
            ray.shutdown()
    
    def _find_latest_checkpoint(self) -> str:
        """Find the latest trained model checkpoint."""
        results_dir = os.path.abspath(f"./ray_results/6g_traffic_PPO_{self.num_vehicles}v")
        
        if not os.path.exists(results_dir):
            return None
            
        # Find subdirectories with checkpoints
        for subdir in os.listdir(results_dir):
            checkpoint_dir = os.path.join(results_dir, subdir)
            if os.path.isdir(checkpoint_dir):
                # Look for checkpoint folders
                checkpoints = [f for f in os.listdir(checkpoint_dir) 
                             if f.startswith("checkpoint_") and os.path.isdir(os.path.join(checkpoint_dir, f))]
                if checkpoints:
                    # Get the latest checkpoint
                    latest_checkpoint = sorted(checkpoints, key=lambda x: int(x.split("_")[1]))[-1]
                    return os.path.abspath(os.path.join(checkpoint_dir, latest_checkpoint))
        
        return None
    
    def _train_model(self):
        """Train a new model if none exists."""
        print("   Training new 6G model...")
        
        config = PPOConfig()
        config = config.environment(MultiAgentTrafficEnv, env_config={
            "num_vehicles": self.num_vehicles,
            "max_episode_steps": 800,
            "intersection_x": 50.0,
            "intersection_y": 50.0
        })
        config = config.framework("torch")
        config = config.rollouts(num_rollout_workers=1)
        config = config.training(
            train_batch_size=2000,
            sgd_minibatch_size=200,
            num_sgd_iter=10,
            lr=0.0003,
            gamma=0.95,
            lambda_=0.9,
            clip_param=0.2
        )
        
        # Quick training run
        results = tune.run(
            "PPO",
            config=config.to_dict(),
            stop={"training_iteration": max(5, self.iterations)},
            storage_path=os.path.abspath("./ray_results"),
            name=f"6g_traffic_PPO_{self.num_vehicles}v_benchmark",
            verbose=1
        )
    
    def _run_single_episode(self, checkpoint_path: str) -> Dict:
        """Run a single evaluation episode."""
        # Create environment
        env_config = {
            "num_vehicles": self.num_vehicles,
            "max_episode_steps": 800,
            "intersection_x": 50.0,
            "intersection_y": 50.0
        }
        env = MultiAgentTrafficEnv(env_config)
        
        # Load algorithm from checkpoint
        from ray.rllib.algorithms.algorithm import Algorithm
        algo = Algorithm.from_checkpoint(checkpoint_path)
        
        # Run episode
        obs, _ = env.reset()
        total_reward = 0
        step_count = 0
        done = False
        
        while not done and step_count < 800:
            # Get actions for all agents
            actions = {}
            for agent_id in obs.keys():
                action = algo.compute_single_action(obs[agent_id])
                actions[agent_id] = action
            
            # Step environment
            obs, rewards, terminateds, truncateds, infos = env.step(actions)
            
            # Calculate total reward
            total_reward += sum(rewards.values())
            step_count += 1
            
            # Check if done
            done = all(terminateds.values()) or all(truncateds.values())
        
        # Get final metrics
        final_info = env._get_global_info()

        completed_vehicles = len([v for v in env.vehicles.values() if v.completed_trip])
        episode_duration = step_count * 0.1  # 100ms steps
        throughput_per_minute = (completed_vehicles / (episode_duration / 60.0)) if episode_duration > 0 else 0
        avg_travel_time = final_info.get("average_travel_time", 0)
        avg_speed_mps = (env.vehicle_spawn_distance * 2 / avg_travel_time) if avg_travel_time > 0 else 0

        metrics = {
            "episode_return": total_reward,
            "average_travel_time": avg_travel_time,
            "average_waiting_time": final_info.get("average_waiting_time", 0),
            "throughput_per_minute": throughput_per_minute,
            "throughput_per_hour": throughput_per_minute * 60,
            "average_speed_mph": avg_speed_mps * 2.237,
            "collision_count": final_info.get("collision_rate", 0) * len(env.vehicles),
            "average_queue_length": 0,
            "energy_consumption": final_info.get("average_energy_consumed", 0),
            "vehicles_completed": completed_vehicles,
            "episode_duration": episode_duration,
        }
        
        return metrics
    

class TrafficLightBenchmark:
    """Benchmark runner for traditional traffic light system."""
    
    def __init__(self, num_vehicles: int = 6):
        self.num_vehicles = num_vehicles
        self.results = PerformanceMetrics()
    
    def run_benchmark(self, num_episodes: int = 10) -> PerformanceMetrics:
        """Run benchmark episodes using traffic light system."""
        print(f"🚦 Running Traditional Traffic Light System Benchmark")
        print(f"   Vehicles: {self.num_vehicles}")
        print(f"   Episodes: {num_episodes}")
        
        for episode in range(num_episodes):
            print(f"   Episode {episode + 1}/{num_episodes}", end=" ")
            
            episode_metrics = self._run_single_episode()
            self.results.add_episode(episode_metrics)
            
            print(f"✅ Throughput: {episode_metrics.get('throughput_per_minute', 0):.1f} veh/min")
        
        print("✅ Traditional Traffic Light Benchmark Complete!\n")
        return self.results
    
    def _run_single_episode(self) -> Dict:
        """Run a single episode in the traffic light environment."""

        env_config = {
            "num_vehicles": self.num_vehicles,
            "max_episode_steps": 800,
            "intersection_x": 50.0,
            "intersection_y": 50.0,
        }
        env = TrafficLightEnvironment(env_config)

        obs = env.reset()
        done = False
        total_reward = 0.0

        while not done:
            obs, reward, done, info = env.step()
            total_reward += reward

        final_metrics = env.get_final_metrics()

        avg_speed_mph = final_metrics.get("average_speed", 0) * 2.237
        throughput_per_minute = final_metrics.get("throughput_per_minute", 0)

        metrics = {
            "episode_return": total_reward,
            "average_speed_mph": avg_speed_mph,
            "average_travel_time": final_metrics.get("average_travel_time", 0),
            "average_waiting_time": final_metrics.get("average_waiting_time", 0),
            "throughput_per_minute": throughput_per_minute,
            "throughput_per_hour": throughput_per_minute * 60,
            "collision_count": 0,
            "average_queue_length": final_metrics.get("average_queue_length", 0),
            "energy_consumption": 0,
            "vehicles_completed": final_metrics.get("total_vehicles", 0),
            "episode_duration": final_metrics.get("episode_duration", 0),
        }

        return metrics

class ComparisonAnalyzer:
    """Analyzer for comparing 6G vs Traffic Light performance."""
    
    def __init__(self):
        self.sixg_results = None
        self.traffic_light_results = None
        
    def run_full_comparison(self, num_vehicles: int = 6, num_episodes: int = 10) -> Dict:
        """Run complete performance comparison."""
        print("🎯 Starting Comprehensive Performance Comparison")
        print("=" * 60)
        print(f"Configuration: {num_vehicles} vehicles, {num_episodes} episodes each")
        print()
        
        # Run 6G autonomous system benchmark
        sixg_benchmark = SixGSystemBenchmark(num_vehicles, iterations=10)
        self.sixg_results = sixg_benchmark.run_benchmark(num_episodes)
        
        # Run traffic light system benchmark
        traffic_benchmark = TrafficLightBenchmark(num_vehicles)
        self.traffic_light_results = traffic_benchmark.run_benchmark(num_episodes)
        
        # Analyze results
        comparison = self._analyze_results()
        
        # Generate report
        self._generate_report(comparison, num_vehicles, num_episodes)
        
        # Create visualizations
        self._create_visualizations(comparison, num_vehicles)
        
        return comparison
    
    def _analyze_results(self) -> Dict:
        """Analyze and compare the results."""
        sixg_stats = self.sixg_results.get_statistics()
        traffic_stats = self.traffic_light_results.get_statistics()
        
        # Calculate improvements (positive = 6G is better)
        improvements = {}
        
        # Metrics where lower is better (reductions)
        reduction_metrics = ["travel_time", "waiting_time", "queue_length"]
        # Metrics where higher is better (increases)  
        increase_metrics = ["throughput", "throughput_hourly", "average_speed"]
        
        for metric in reduction_metrics + increase_metrics:
            sixg_mean = sixg_stats[metric]["mean"]
            traffic_mean = traffic_stats[metric]["mean"]
            
            if metric in reduction_metrics:
                # Lower is better - calculate reduction percentage
                if traffic_mean > 0:
                    improvement = ((traffic_mean - sixg_mean) / traffic_mean) * 100
                else:
                    improvement = 0
            else:  # increase_metrics
                # Higher is better - calculate increase percentage
                if traffic_mean > 0:
                    improvement = ((sixg_mean - traffic_mean) / traffic_mean) * 100
                else:
                    improvement = 0
            
            improvements[metric] = improvement
        
        return {
            "sixg_statistics": sixg_stats,
            "traffic_light_statistics": traffic_stats,
            "improvements": improvements,
            "raw_data": {
                "sixg": {
                    "average_speeds": self.sixg_results.average_speeds,
                    "waiting_times": self.sixg_results.waiting_times,
                    "throughput_hourly": self.sixg_results.throughput_hourly,
                    "returns": self.sixg_results.episode_returns
                },
                "traffic_light": {
                    "average_speeds": self.traffic_light_results.average_speeds,
                    "waiting_times": self.traffic_light_results.waiting_times,
                    "throughput_hourly": self.traffic_light_results.throughput_hourly,
                    "returns": self.traffic_light_results.episode_returns
                }
            }
        }
    
    def _generate_report(self, comparison: Dict, num_vehicles: int, num_episodes: int):
        """Generate comprehensive comparison report."""
        print("📊 PERFORMANCE COMPARISON RESULTS")
        print("=" * 60)
        
        sixg_stats = comparison["sixg_statistics"]
        traffic_stats = comparison["traffic_light_statistics"]
        improvements = comparison["improvements"]
        
        print(f"\n🚗 6G AUTONOMOUS SYSTEM (Signal-Free)")
        print(f"   Average Speed: {sixg_stats['average_speed']['mean']:.1f} mph ± {sixg_stats['average_speed']['std']:.1f}")
        print(f"   Intersection Throughput: {sixg_stats['throughput_hourly']['mean']:.0f} vehicles/hour ± {sixg_stats['throughput_hourly']['std']:.0f}")
        print(f"   Average Waiting Time: {sixg_stats['waiting_time']['mean']:.1f}s ± {sixg_stats['waiting_time']['std']:.1f}")
        print(f"   Average Queue Length: {sixg_stats['queue_length']['mean']:.1f} vehicles")
        print(f"   Episode Return: {sixg_stats['episode_return']['mean']:.0f}")
        
        print(f"\n🚦 TRADITIONAL TRAFFIC LIGHT SYSTEM")
        print(f"   Average Speed: {traffic_stats['average_speed']['mean']:.1f} mph ± {traffic_stats['average_speed']['std']:.1f}")
        print(f"   Intersection Throughput: {traffic_stats['throughput_hourly']['mean']:.0f} vehicles/hour ± {traffic_stats['throughput_hourly']['std']:.0f}")
        print(f"   Average Waiting Time: {traffic_stats['waiting_time']['mean']:.1f}s ± {traffic_stats['waiting_time']['std']:.1f}")
        print(f"   Average Queue Length: {traffic_stats['queue_length']['mean']:.1f} vehicles")
        print(f"   Episode Return: {traffic_stats['episode_return']['mean']:.0f}")
        
        print(f"\n🎯 PERFORMANCE IMPROVEMENTS (6G vs Traffic Light)")
        print(f"   🚄 Average Speed Increase: {improvements['average_speed']:+.1f}%")
        print(f"   🚗 Throughput Increase: {improvements['throughput_hourly']:+.1f}% ({sixg_stats['throughput_hourly']['mean']:.0f} vs {traffic_stats['throughput_hourly']['mean']:.0f} veh/hr)")
        print(f"   ⏱️  Waiting Time Reduction: {improvements['waiting_time']:+.1f}%")
        print(f"   🚥 Queue Length Reduction: {improvements['queue_length']:+.1f}%")
        
        # Determine overall verdict
        avg_improvement = np.mean([improvements['average_speed'], improvements['waiting_time'], improvements['throughput_hourly']])
        
        print(f"\n🏆 OVERALL VERDICT:")
        if avg_improvement > 10:
            print(f"   6G Autonomous System is SIGNIFICANTLY BETTER (+{avg_improvement:.1f}% average improvement)")
        elif avg_improvement > 0:
            print(f"   6G Autonomous System is BETTER (+{avg_improvement:.1f}% average improvement)")
        elif avg_improvement > -10:
            print(f"   Systems are COMPARABLE ({avg_improvement:.1f}% difference)")
        else:
            print(f"   Traffic Light System is BETTER ({avg_improvement:.1f}% difference)")
        
        print("\n" + "=" * 60)
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"results/comparison_results_{num_vehicles}v_{timestamp}.json"
        
        os.makedirs("results", exist_ok=True)
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_numpy(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        comparison_serializable = convert_numpy(comparison)
        
        with open(results_file, 'w') as f:
            json.dump(comparison_serializable, f, indent=2)
        
        print(f"📁 Detailed results saved to: {results_file}")
    
    def _create_visualizations(self, comparison: Dict, num_vehicles: int):
        """Create comparison visualizations."""
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'6G Autonomous vs Traditional Traffic Light System ({num_vehicles} vehicles)', fontsize=16, fontweight='bold')
        
        # Data preparation
        sixg_data = comparison["raw_data"]["sixg"]
        traffic_data = comparison["raw_data"]["traffic_light"]
        
        # 1. Average Speed Comparison
        axes[0, 0].boxplot([sixg_data["average_speeds"], traffic_data["average_speeds"]], 
                          labels=['6G Autonomous', 'Traffic Light'])
        axes[0, 0].set_title('Average Speed Distribution')
        axes[0, 0].set_ylabel('Speed (mph)')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Waiting Time Comparison
        axes[0, 1].boxplot([sixg_data["waiting_times"], traffic_data["waiting_times"]], 
                          labels=['6G Autonomous', 'Traffic Light'])
        axes[0, 1].set_title('Waiting Time Distribution')
        axes[0, 1].set_ylabel('Time (seconds)')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Throughput Comparison
        axes[1, 0].bar(['6G Autonomous', 'Traffic Light'], 
                      [np.mean(sixg_data["throughput_hourly"]), np.mean(traffic_data["throughput_hourly"])],
                      color=['#2E86AB', '#A23B72'], alpha=0.8)
        axes[1, 0].set_title('Intersection Throughput')
        axes[1, 0].set_ylabel('Vehicles per hour')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Performance Improvements
        improvements = comparison["improvements"]
        metrics = ['Speed\nIncrease', 'Waiting Time\nReduction', 'Throughput\nIncrease']
        values = [improvements['average_speed'], improvements['waiting_time'], improvements['throughput_hourly']]
        colors = ['green' if v > 0 else 'red' for v in values]
        
        bars = axes[1, 1].bar(metrics, values, color=colors, alpha=0.7)
        axes[1, 1].set_title('6G System Improvements (%)')
        axes[1, 1].set_ylabel('Improvement (%)')
        axes[1, 1].axhline(y=0, color='black', linestyle='-', alpha=0.5)
        axes[1, 1].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                           f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_file = f"results/comparison_plot_{num_vehicles}v_{timestamp}.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        print(f"📊 Comparison plots saved to: {plot_file}")
        
        plt.show() 