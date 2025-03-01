"""
Demo Script for 6G Traffic Management Simulation

This script demonstrates the basic functionality of the simulation
without requiring trained models. Uses simple rule-based agents
for quick demonstration.
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from multi_agent_traffic_env import MultiAgentTrafficEnv
from visualizer import PygameVisualizer, MatplotlibVisualizer

class RuleBasedAgent:
    """
    Simple rule-based agent for demonstration purposes.
    """
    
    def __init__(self):
        self.name = "RuleBasedAgent"
    
    def get_action(self, observation, vehicle, env):
        """
        Simple rule-based decision making.
        """
        # Extract vehicle state
        x, y = vehicle.x, vehicle.y
        vx, vy = vehicle.vx, vehicle.vy
        current_speed = np.sqrt(vx*vx + vy*vy)
        
        # Check if already completed trip
        if vehicle.completed_trip:
            return np.array([0.0])
        
        # Calculate distance to target
        dist_to_target = np.sqrt((vehicle.target_x - x)**2 + (vehicle.target_y - y)**2)
        
        # If very close to target, slow down
        if dist_to_target < 15.0:
            if current_speed > 5.0:
                return np.array([-1.0])  # Slow down
            else:
                return np.array([0.0])   # Maintain low speed
        
        # Find nearest intersection
        nearest_intersection = None
        min_distance = float('inf')
        
        for intersection in env.intersections:
            dx = intersection.x - x
            dy = intersection.y - y
            distance = np.sqrt(dx*dx + dy*dy)
            
            if distance < min_distance:
                min_distance = distance
                nearest_intersection = intersection
        
        # Decision logic
        if nearest_intersection:
            # Distance to intersection
            distance = min_distance
            
            # If approaching intersection
            if distance < 30.0:
                # Check if intersection is occupied
                if nearest_intersection.current_vehicle and nearest_intersection.current_vehicle.id != vehicle.id:
                    # Another vehicle in intersection - brake hard
                    return np.array([-2.5])  # Brake
                elif vehicle.has_reservation:
                    # Have reservation - proceed with caution
                    if distance < 10.0:
                        return np.array([0.0])   # Maintain speed through intersection
                    else:
                        return np.array([0.5])   # Slight acceleration to reach intersection
                elif distance < 15.0:
                    # Too close without reservation - brake hard
                    return np.array([-3.0])  # Hard brake
                else:
                    # Approaching - slow down and wait for reservation
                    return np.array([-1.0])  # Moderate braking
            else:
                # Far from intersection - normal driving
                target_speed = vehicle.max_speed * 0.7  # More conservative speed
                
                if current_speed < target_speed:
                    return np.array([1.5])   # Accelerate towards target speed
                elif current_speed > target_speed:
                    return np.array([-0.5])  # Slow down if too fast
                else:
                    return np.array([0.0])   # Maintain speed
        
        # Default: accelerate towards target speed
        target_speed = vehicle.max_speed * 0.7
        if current_speed < target_speed:
            return np.array([1.0])   # Accelerate
        else:
            return np.array([0.0])   # Maintain speed

def run_demo_episode(env, agent, max_steps=300, use_visualization=True):
    """
    Run a single demo episode with rule-based agents.
    """
    print("Starting demo episode...")
    
    # Setup visualization
    visualizer = None
    if use_visualization:
        try:
            visualizer = PygameVisualizer(width=1200, height=800, scale=2.5)
            print("Pygame visualization initialized")
        except Exception as e:
            print(f"Pygame failed, using matplotlib: {e}")
            visualizer = MatplotlibVisualizer()
    
    # Reset environment
    obs, info = env.reset()
    
    episode_data = {
        'steps': [],
        'completion_rates': [],
        'collision_rates': [],
        'vehicles_data': []
    }
    
    for step in range(max_steps):
        # Get actions from rule-based agent
        actions = {}
        for agent_id, observation in obs.items():
            if agent_id in env.vehicles:
                vehicle = env.vehicles[agent_id]
                if not vehicle.completed_trip:
                    action = agent.get_action(observation, vehicle, env)
                    actions[agent_id] = action
        
        # Step environment
        obs, rewards, dones, truncated, infos = env.step(actions)
        
        # Collect data
        global_info = infos.get("__all__", {})
        episode_data['steps'].append(step)
        episode_data['completion_rates'].append(global_info.get('completion_rate', 0))
        episode_data['collision_rates'].append(global_info.get('collision_rate', 0))
        
        # Vehicle positions for trajectory
        vehicle_positions = {}
        for agent_id, vehicle in env.vehicles.items():
            vehicle_positions[agent_id] = {
                'x': vehicle.x, 'y': vehicle.y,
                'completed': vehicle.completed_trip,
                'collision': vehicle.collision_occurred,
                'reserved': vehicle.has_reservation
            }
        episode_data['vehicles_data'].append(vehicle_positions)
        
        # Visualization
        if use_visualization and visualizer:
            if isinstance(visualizer, PygameVisualizer):
                network_stats = env.network.get_network_stats()
                continue_sim = visualizer.render_frame(
                    env, env.vehicles, env.intersections, global_info, network_stats
                )
                if not continue_sim:
                    print("Demo stopped by user")
                    break
                time.sleep(0.1)  # Slow down for visibility
        
        # Check if episode is done
        if dones.get("__all__", False):
            print(f"Episode completed at step {step}")
            break
        
        # Print progress
        if step % 50 == 0:
            completion_rate = global_info.get('completion_rate', 0)
            collision_rate = global_info.get('collision_rate', 0)
            print(f"Step {step}: Completion {completion_rate:.2%}, Collisions {collision_rate:.2%}")
            
            # Debug vehicle positions
            for agent_id, vehicle in env.vehicles.items():
                dist_to_target = np.sqrt((vehicle.target_x - vehicle.x)**2 + (vehicle.target_y - vehicle.y)**2)
                speed = np.sqrt(vehicle.vx**2 + vehicle.vy**2)
                print(f"  {agent_id}: pos=({vehicle.x:.1f},{vehicle.y:.1f}), "
                      f"target=({vehicle.target_x:.1f},{vehicle.target_y:.1f}), "
                      f"dist={dist_to_target:.1f}, speed={speed:.1f}, "
                      f"completed={vehicle.completed_trip}")
    
    return episode_data

def create_demo_plots(episode_data):
    """
    Create plots showing the demo results.
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('6G Traffic Management Demo Results', fontsize=16)
    
    steps = episode_data['steps']
    
    # Plot 1: Completion rate over time
    axes[0, 0].plot(steps, episode_data['completion_rates'], 'g-', linewidth=2)
    axes[0, 0].set_title('Trip Completion Rate')
    axes[0, 0].set_xlabel('Simulation Step')
    axes[0, 0].set_ylabel('Completion Rate')
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Collision rate over time
    axes[0, 1].plot(steps, episode_data['collision_rates'], 'r-', linewidth=2)
    axes[0, 1].set_title('Collision Rate')
    axes[0, 1].set_xlabel('Simulation Step')
    axes[0, 1].set_ylabel('Collision Rate')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Vehicle trajectories
    axes[1, 0].set_title('Vehicle Trajectories')
    axes[1, 0].set_xlabel('X Position (m)')
    axes[1, 0].set_ylabel('Y Position (m)')
    
    # Plot trajectories for each vehicle
    colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
    for step_data in episode_data['vehicles_data'][::10]:  # Sample every 10 steps
        for i, (agent_id, vehicle_data) in enumerate(step_data.items()):
            color = colors[i % len(colors)]
            if vehicle_data['completed']:
                marker = 'o'
            elif vehicle_data['collision']:
                marker = 'x'
            elif vehicle_data['reserved']:
                marker = 's'
            else:
                marker = '.'
            
            axes[1, 0].plot(vehicle_data['x'], vehicle_data['y'], 
                          marker=marker, color=color, markersize=4, alpha=0.7)
    
    # Draw intersection
    axes[1, 0].plot(0, 0, 'rs', markersize=15, label='Intersection')
    axes[1, 0].set_aspect('equal')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Final statistics
    axes[1, 1].axis('off')
    
    final_completion = episode_data['completion_rates'][-1] if episode_data['completion_rates'] else 0
    final_collision = episode_data['collision_rates'][-1] if episode_data['collision_rates'] else 0
    
    stats_text = f"""
    Demo Results Summary:
    
    Final Completion Rate: {final_completion:.1%}
    Final Collision Rate: {final_collision:.1%}
    
    Simulation Steps: {len(steps)}
    
    System Features Demonstrated:
    ✓ Multi-agent coordination
    ✓ 6G communication simulation
    ✓ Signal-free intersection management
    ✓ Real-time visualization
    ✓ Performance metrics tracking
    
    Note: This demo uses simple rule-based
    agents. Train with MARL for optimal
    performance!
    """
    
    axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes,
                    fontsize=11, verticalalignment='top', fontfamily='monospace',
                    bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    plt.show()
    
    return fig

def main():
    """
    Main demo function.
    """
    print("="*60)
    print("6G Traffic Management Simulation Demo")
    print("="*60)
    print()
    print("This demo shows the basic functionality using rule-based agents.")
    print("For optimal performance, train with reinforcement learning!")
    print()
    
    # Setup environment
    env_config = {
        "num_vehicles": 4,
        "max_episode_steps": 600,
        "dt": 0.1,
        "intersection_positions": [(0.0, 0.0)],
        "vehicle_spawn_distance": 60.0
    }
    
    env = MultiAgentTrafficEnv(env_config)
    agent = RuleBasedAgent()
    
    print("Environment created with:")
    print(f"  - {env_config['num_vehicles']} vehicles")
    print(f"  - {len(env_config['intersection_positions'])} intersection(s)")
    print(f"  - 6G network with {env.network.latency*1000:.1f}ms latency")
    print()
    
    try:
        # Run demo episode
        episode_data = run_demo_episode(env, agent, max_steps=400, use_visualization=True)
        
        print("\nDemo completed!")
        print(f"Final completion rate: {episode_data['completion_rates'][-1]:.1%}")
        print(f"Final collision rate: {episode_data['collision_rates'][-1]:.1%}")
        
        # Create plots
        print("\nGenerating demo plots...")
        create_demo_plots(episode_data)
        
        print("\nDemo finished successfully!")
        print("\nNext steps:")
        print("1. Install dependencies: pip install -r requirements.txt")
        print("2. Train MARL agents: python train_marl.py")
        print("3. Run simulation: python simulate.py --checkpoint [path]")
        
    except KeyboardInterrupt:
        print("\nDemo interrupted by user")
    except Exception as e:
        print(f"\nDemo failed: {e}")
        print("Try running without visualization:")
        print("python demo.py --no-viz")

if __name__ == "__main__":
    import sys
    
    # Check for no-visualization flag
    use_viz = "--no-viz" not in sys.argv
    
    if not use_viz:
        print("Running demo without visualization...")
        
        # Run simple demo
        env_config = {
            "num_vehicles": 4,
            "max_episode_steps": 200,
            "dt": 0.1,
            "intersection_positions": [(0.0, 0.0)],
            "vehicle_spawn_distance": 60.0
        }
        
        env = MultiAgentTrafficEnv(env_config)
        agent = RuleBasedAgent()
        
        episode_data = run_demo_episode(env, agent, max_steps=200, use_visualization=False)
        create_demo_plots(episode_data)
    else:
        main() 