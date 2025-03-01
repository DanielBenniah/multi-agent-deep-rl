"""
Multi-Agent Reinforcement Learning Training Script for 6G Traffic Management
"""

import os
import ray
import logging
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, Any
import argparse
import json

from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.tune.registry import register_env

from multi_agent_traffic_env import MultiAgentTrafficEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MARL_Training")

def setup_training_config(algorithm="PPO", num_vehicles=4):
    """Setup training configuration for RL algorithms."""
    
    # Environment configuration
    env_config = {
        "num_vehicles": num_vehicles,
        "max_episode_steps": 800,
        "dt": 0.1,
        "intersection_positions": [(0.0, 0.0)],
        "vehicle_spawn_distance": 80.0
    }
    
    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return "shared_policy"
    
    policies = {
        "shared_policy": PolicySpec(
            policy_class=None,
            observation_space=None,
            action_space=None,
            config={}
        )
    }
    
    config = PPOConfig()
    config = config.environment(env="traffic_env", env_config=env_config)
    config = config.multi_agent(
        policies=policies,
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["shared_policy"]
    )
    config = config.rollouts(num_rollout_workers=0, rollout_fragment_length=200)  # 0 workers for local mode
    config = config.training(
        train_batch_size=1000,  # Reduced for faster training
        sgd_minibatch_size=64,   # Smaller batches
        num_sgd_iter=5,          # Fewer iterations per update
        lr=3e-4,
        gamma=0.99
    )
    config = config.framework("torch")
    
    return config

def main():
    """Main function to run training."""
    parser = argparse.ArgumentParser(description="Train MARL agents for 6G traffic management")
    parser.add_argument("--num-vehicles", type=int, default=4, help="Number of vehicles")
    parser.add_argument("--iterations", type=int, default=100, help="Number of training iterations")
    
    args = parser.parse_args()
    
    # Initialize Ray in local mode (more reliable on Windows)
    if not ray.is_initialized():
        try:
            # Try local mode first (single process, more stable)
            ray.init(local_mode=True, ignore_reinit_error=True)
            logger.info("Ray initialized in local mode")
        except Exception as e:
            logger.warning(f"Local mode failed: {e}, trying regular mode with reduced resources...")
            try:
                ray.init(num_cpus=2, object_store_memory=1000000000, ignore_reinit_error=True)
                logger.info("Ray initialized in regular mode")
            except Exception as e2:
                logger.error(f"Failed to initialize Ray: {e2}")
                raise
    
    try:
        # Register environment
        def env_creator(env_config):
            return MultiAgentTrafficEnv(env_config)
        
        register_env("traffic_env", env_creator)
        
        # Setup configuration
        config = setup_training_config(num_vehicles=args.num_vehicles)
        
        # Create algorithm instance
        algo = PPO(config=config)
        
        logger.info("Starting training...")
        
        # Training loop
        for iteration in range(args.iterations):
            result = algo.train()
            
            episode_reward = result.get("episode_reward_mean", 0)
            episode_length = result.get("episode_len_mean", 0)
            
            if iteration % 10 == 0:
                logger.info(f"Iteration {iteration}: Reward={episode_reward:.2f}, Length={episode_length:.1f}")
        
        # Save final model
        checkpoint_path = algo.save("./results")
        logger.info(f"Model saved: {checkpoint_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    
    finally:
        ray.shutdown()

if __name__ == "__main__":
    main() 