"""
Simulation and Evaluation Script for 6G Traffic Management

This script loads trained MARL agents and runs simulation episodes
with real-time visualization and performance evaluation.
"""

import os
import ray
import time
import logging
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional
import json
from datetime import datetime

from ray.rllib.algorithms.ppo import PPO
from ray.rllib.algorithms.sac import SAC
from ray.rllib.algorithms.dqn import DQN
from ray.tune.registry import register_env

from multi_agent_traffic_env import MultiAgentTrafficEnv
from visualizer import PygameVisualizer, MatplotlibVisualizer
from train_marl import setup_training_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Simulation")

class SimulationRunner:
    """
    Manages simulation episodes with trained agents.
    """
    
    def __init__(self, checkpoint_path: str, algorithm: str = "PPO", 
                 num_vehicles: int = 4, visualization: str = "pygame"):
        self.checkpoint_path = checkpoint_path
        self.algorithm = algorithm
        self.num_vehicles = num_vehicles
        self.visualization = visualization
        
        # Initialize components
        self.env = None
        self.agent = None
        self.visualizer = None
        
        # Simulation results
        self.episode_results = []
        self.current_episode = 0
        
    def setup_environment(self):
        """Setup the simulation environment."""
        def env_creator(env_config):
            return MultiAgentTrafficEnv(env_config)
        
        register_env("traffic_env", env_creator)
        
        # Create environment instance for simulation
        env_config = {
            "num_vehicles": self.num_vehicles,
            "max_episode_steps": 1000,
            "dt": 0.1,
            "intersection_positions": [(0.0, 0.0)],
            "vehicle_spawn_distance": 80.0
        }
        
        self.env = MultiAgentTrafficEnv(env_config)
        logger.info("Environment setup completed")
    
    def load_trained_agent(self):
        """Load the trained agent from checkpoint."""
        if not os.path.exists(self.checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {self.checkpoint_path}")
        
        # Setup algorithm configuration
        config = setup_training_config(
            algorithm=self.algorithm,
            num_vehicles=self.num_vehicles
        )
        
        # Create and restore agent
        if self.algorithm == "PPO":
            self.agent = PPO(config=config)
        elif self.algorithm == "SAC":
            self.agent = SAC(config=config)
        elif self.algorithm == "DQN":
            self.agent = DQN(config=config)
        else:
            raise ValueError(f"Unsupported algorithm: {self.algorithm}")
        
        self.agent.restore(self.checkpoint_path)
        logger.info(f"Loaded trained {self.algorithm} agent from {self.checkpoint_path}")
    
    def setup_visualization(self):
        """Setup visualization system."""
        if self.visualization == "pygame":
            self.visualizer = PygameVisualizer(width=1400, height=900, scale=3.0)
            logger.info("Pygame visualization initialized")
        elif self.visualization == "matplotlib":
            self.visualizer = MatplotlibVisualizer()
            logger.info("Matplotlib visualization initialized")
        elif self.visualization == "none":
            self.visualizer = None
            logger.info("No visualization selected")
        else:
            raise ValueError(f"Unknown visualization type: {self.visualization}")
    
    def run_episode(self, episode_num: int, max_steps: int = 1000, 
                   render: bool = True, record_trajectory: bool = True):
        """Run a single simulation episode."""
        logger.info(f"Starting episode {episode_num}")
        
        # Reset environment
        obs, info = self.env.reset()
        episode_metrics = {
            "episode": episode_num,
            "steps": 0,
            "total_reward": 0.0,
            "individual_rewards": {agent_id: 0.0 for agent_id in obs.keys()},
            "completion_times": {},
            "collision_count": 0,
            "trajectory": [] if record_trajectory else None
        }
        
        step = 0
        done = False
        
        while not done and step < max_steps:
            # Get actions from trained agent
            actions = {}
            for agent_id, agent_obs in obs.items():
                if agent_id in self.env.vehicles and not self.env.vehicles[agent_id].completed_trip:
                    action = self.agent.compute_single_action(agent_obs, policy_id="shared_policy")
                    actions[agent_id] = action
            
            # Step environment
            obs, rewards, dones, truncated, infos = self.env.step(actions)
            
            # Update metrics
            step += 1
            episode_metrics["steps"] = step
            
            for agent_id, reward in rewards.items():
                episode_metrics["individual_rewards"][agent_id] += reward
                episode_metrics["total_reward"] += reward
            
            # Record trajectory
            if record_trajectory:
                frame_data = {
                    "step": step,
                    "vehicles": {},
                    "intersections": [],
                    "network_stats": self.env.network.get_network_stats()
                }
                
                for agent_id, vehicle in self.env.vehicles.items():
                    frame_data["vehicles"][agent_id] = {
                        "x": vehicle.x,
                        "y": vehicle.y,
                        "vx": vehicle.vx,
                        "vy": vehicle.vy,
                        "has_reservation": vehicle.has_reservation,
                        "completed_trip": vehicle.completed_trip,
                        "collision_occurred": vehicle.collision_occurred
                    }
                
                for intersection in self.env.intersections:
                    frame_data["intersections"].append({
                        "id": intersection.id,
                        "x": intersection.x,
                        "y": intersection.y,
                        "active_reservations": len(intersection.reservations),
                        "vehicles_served": intersection.vehicles_served
                    })
                
                episode_metrics["trajectory"].append(frame_data)
            
            # Check for completion and collisions
            for agent_id, vehicle in self.env.vehicles.items():
                if vehicle.completed_trip and agent_id not in episode_metrics["completion_times"]:
                    episode_metrics["completion_times"][agent_id] = step * self.env.dt
                
                if vehicle.collision_occurred:
                    episode_metrics["collision_count"] += 1
            
            # Visualization
            if render and self.visualizer and isinstance(self.visualizer, PygameVisualizer):
                env_info = infos.get("__all__", {})
                network_stats = self.env.network.get_network_stats()
                
                # Render frame
                continue_simulation = self.visualizer.render_frame(
                    self.env, self.env.vehicles, self.env.intersections, 
                    env_info, network_stats
                )
                
                if not continue_simulation:
                    logger.info("Simulation stopped by user")
                    break
            
            # Check if episode is done
            done = dones.get("__all__", False)
            
            # Small delay for real-time visualization
            if render and self.visualization == "pygame":
                time.sleep(0.05)  # 20 FPS for smooth visualization
        
        # Final episode statistics
        global_info = infos.get("__all__", {})
        episode_metrics.update({
            "completion_rate": global_info.get("completion_rate", 0),
            "collision_rate": global_info.get("collision_rate", 0),
            "average_travel_time": global_info.get("average_travel_time", 0),
            "average_waiting_time": global_info.get("average_waiting_time", 0),
            "network_reliability": global_info.get("network_reliability", 0),
            "total_messages": global_info.get("network_total_messages", 0)
        })
        
        logger.info(f"Episode {episode_num} completed: "
                   f"Steps={step}, "
                   f"Completion={episode_metrics['completion_rate']:.2%}, "
                   f"Collisions={episode_metrics['collision_rate']:.2%}")
        
        return episode_metrics
    
    def run_multiple_episodes(self, num_episodes: int = 10, render: bool = True, 
                            save_results: bool = True):
        """Run multiple simulation episodes for evaluation."""
        logger.info(f"Running {num_episodes} episodes for evaluation")
        
        all_results = []
        
        for episode in range(num_episodes):
            try:
                result = self.run_episode(
                    episode_num=episode,
                    render=render and (episode == 0),  # Only render first episode
                    record_trajectory=(episode == 0)   # Only record first episode trajectory
                )
                all_results.append(result)
                
            except KeyboardInterrupt:
                logger.info("Evaluation interrupted by user")
                break
            except Exception as e:
                logger.error(f"Episode {episode} failed: {e}")
                continue
        
        # Calculate aggregate statistics
        if all_results:
            aggregate_stats = self.calculate_aggregate_statistics(all_results)
            
            if save_results:
                self.save_evaluation_results(all_results, aggregate_stats)
            
            self.print_evaluation_summary(aggregate_stats)
            
            # Create performance plots
            if self.visualization in ["matplotlib", "both"]:
                self.create_evaluation_plots(all_results)
        
        return all_results
    
    def calculate_aggregate_statistics(self, results: List[Dict]) -> Dict:
        """Calculate aggregate statistics across episodes."""
        stats = {
            "num_episodes": len(results),
            "mean_completion_rate": np.mean([r["completion_rate"] for r in results]),
            "std_completion_rate": np.std([r["completion_rate"] for r in results]),
            "mean_collision_rate": np.mean([r["collision_rate"] for r in results]),
            "std_collision_rate": np.std([r["collision_rate"] for r in results]),
            "mean_travel_time": np.mean([r["average_travel_time"] for r in results]),
            "std_travel_time": np.std([r["average_travel_time"] for r in results]),
            "mean_waiting_time": np.mean([r["average_waiting_time"] for r in results]),
            "std_waiting_time": np.std([r["average_waiting_time"] for r in results]),
            "mean_episode_length": np.mean([r["steps"] for r in results]),
            "std_episode_length": np.std([r["steps"] for r in results]),
            "mean_network_reliability": np.mean([r["network_reliability"] for r in results]),
            "total_messages": sum([r["total_messages"] for r in results])
        }
        
        return stats
    
    def save_evaluation_results(self, results: List[Dict], stats: Dict):
        """Save evaluation results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_dir = f"simulation_results_{timestamp}"
        os.makedirs(results_dir, exist_ok=True)
        
        # Save detailed results
        with open(f"{results_dir}/episode_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Save aggregate statistics
        with open(f"{results_dir}/aggregate_stats.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Results saved to {results_dir}")
    
    def print_evaluation_summary(self, stats: Dict):
        """Print evaluation summary to console."""
        print("\n" + "="*60)
        print("SIMULATION EVALUATION SUMMARY")
        print("="*60)
        print(f"Episodes Completed: {stats['num_episodes']}")
        print(f"Algorithm: {self.algorithm}")
        print(f"Number of Vehicles: {self.num_vehicles}")
        print("-"*60)
        print("PERFORMANCE METRICS:")
        print(f"  Completion Rate: {stats['mean_completion_rate']:.2%} ± {stats['std_completion_rate']:.2%}")
        print(f"  Collision Rate:  {stats['mean_collision_rate']:.2%} ± {stats['std_collision_rate']:.2%}")
        print(f"  Travel Time:     {stats['mean_travel_time']:.1f}s ± {stats['std_travel_time']:.1f}s")
        print(f"  Waiting Time:    {stats['mean_waiting_time']:.1f}s ± {stats['std_waiting_time']:.1f}s")
        print(f"  Episode Length:  {stats['mean_episode_length']:.1f} ± {stats['std_episode_length']:.1f} steps")
        print("-"*60)
        print("NETWORK PERFORMANCE:")
        print(f"  Avg Reliability: {stats['mean_network_reliability']:.2%}")
        print(f"  Total Messages:  {stats['total_messages']}")
        print("="*60)
    
    def create_evaluation_plots(self, results: List[Dict]):
        """Create evaluation performance plots."""
        if not results:
            return
        
        # Create subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Simulation Evaluation Results - {self.algorithm}', fontsize=16)
        
        episodes = range(len(results))
        
        # Plot 1: Completion rates
        completion_rates = [r["completion_rate"] for r in results]
        axes[0, 0].plot(episodes, completion_rates, 'g-o', markersize=4)
        axes[0, 0].set_title('Completion Rate per Episode')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Completion Rate')
        axes[0, 0].set_ylim(0, 1)
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=np.mean(completion_rates), color='g', linestyle='--', alpha=0.7)
        
        # Plot 2: Collision rates
        collision_rates = [r["collision_rate"] for r in results]
        axes[0, 1].plot(episodes, collision_rates, 'r-o', markersize=4)
        axes[0, 1].set_title('Collision Rate per Episode')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Collision Rate')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].axhline(y=np.mean(collision_rates), color='r', linestyle='--', alpha=0.7)
        
        # Plot 3: Travel times
        travel_times = [r["average_travel_time"] for r in results]
        axes[0, 2].plot(episodes, travel_times, 'b-o', markersize=4)
        axes[0, 2].set_title('Average Travel Time per Episode')
        axes[0, 2].set_xlabel('Episode')
        axes[0, 2].set_ylabel('Travel Time (s)')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].axhline(y=np.mean(travel_times), color='b', linestyle='--', alpha=0.7)
        
        # Plot 4: Episode lengths
        episode_lengths = [r["steps"] for r in results]
        axes[1, 0].plot(episodes, episode_lengths, 'purple', marker='o', markersize=4)
        axes[1, 0].set_title('Episode Length')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].axhline(y=np.mean(episode_lengths), color='purple', linestyle='--', alpha=0.7)
        
        # Plot 5: Performance distribution
        metrics = ['Completion\nRate', 'Safety\n(1-Collision)', 'Efficiency\n(1/Travel Time)']
        values = [
            np.mean(completion_rates),
            1 - np.mean(collision_rates),
            1 / max(np.mean(travel_times), 1)
        ]
        
        bars = axes[1, 1].bar(metrics, values, color=['green', 'red', 'blue'], alpha=0.7)
        axes[1, 1].set_title('Overall Performance Metrics')
        axes[1, 1].set_ylabel('Normalized Score')
        axes[1, 1].set_ylim(0, 1)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
        
        # Plot 6: Network reliability
        network_reliability = [r["network_reliability"] for r in results]
        axes[1, 2].plot(episodes, network_reliability, 'orange', marker='o', markersize=4)
        axes[1, 2].set_title('Network Reliability')
        axes[1, 2].set_xlabel('Episode')
        axes[1, 2].set_ylabel('Reliability')
        axes[1, 2].set_ylim(0, 1)
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].axhline(y=np.mean(network_reliability), color='orange', linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = f"evaluation_plots_{timestamp}.png"
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.show()
        
        logger.info(f"Evaluation plots saved: {plot_filename}")

def main():
    """Main function for simulation."""
    parser = argparse.ArgumentParser(description="Run 6G traffic management simulation")
    parser.add_argument("--checkpoint", type=str, required=True,
                       help="Path to trained model checkpoint")
    parser.add_argument("--algorithm", type=str, default="PPO",
                       choices=["PPO", "SAC", "DQN"],
                       help="RL algorithm used")
    parser.add_argument("--num-vehicles", type=int, default=4,
                       help="Number of vehicles in simulation")
    parser.add_argument("--episodes", type=int, default=1,
                       help="Number of episodes to run")
    parser.add_argument("--visualization", type=str, default="pygame",
                       choices=["pygame", "matplotlib", "none"],
                       help="Visualization type")
    parser.add_argument("--no-render", action="store_true",
                       help="Disable rendering for faster evaluation")
    
    args = parser.parse_args()
    
    # Initialize Ray if not already running
    if not ray.is_initialized():
        ray.init(local_mode=True)
    
    try:
        # Create simulation runner
        runner = SimulationRunner(
            checkpoint_path=args.checkpoint,
            algorithm=args.algorithm,
            num_vehicles=args.num_vehicles,
            visualization=args.visualization
        )
        
        # Setup components
        runner.setup_environment()
        runner.load_trained_agent()
        
        if not args.no_render:
            runner.setup_visualization()
        
        # Run simulation
        if args.episodes == 1:
            # Single episode with full visualization
            result = runner.run_episode(
                episode_num=0,
                render=not args.no_render,
                record_trajectory=True
            )
            print(f"Episode completed with {result['completion_rate']:.2%} completion rate")
        else:
            # Multiple episodes for evaluation
            results = runner.run_multiple_episodes(
                num_episodes=args.episodes,
                render=not args.no_render,
                save_results=True
            )
        
        logger.info("Simulation completed successfully!")
        
    except FileNotFoundError as e:
        logger.error(f"Checkpoint not found: {e}")
    except Exception as e:
        logger.error(f"Simulation failed: {e}")
        raise
    finally:
        # Cleanup
        if ray.is_initialized():
            ray.shutdown()

if __name__ == "__main__":
    main() 