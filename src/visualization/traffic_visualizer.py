#!/usr/bin/env python3
"""
Traffic Flow Visualization for 6G Multi-Agent Deep RL System
Phase 2: Real-time visualization of vehicle coordination, 6G communication, and traffic patterns
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle, Rectangle, FancyBboxPatch, Arrow
from matplotlib.collections import LineCollection
import seaborn as sns
from typing import Dict, List, Tuple, Optional
import time
from collections import deque
import threading
import queue
import json

# Set style for better visuals
try:
    plt.style.use('seaborn-v0_8')
except:
    plt.style.use('seaborn')
sns.set_palette("husl")

class TrafficFlowVisualizer:
    """
    Real-time traffic flow visualization for 6G-enabled autonomous vehicles.
    Shows vehicle movements, intersection coordination, and 6G communication patterns.
    """
    
    def __init__(self, figsize=(16, 12), max_trail_length=50):
        """Initialize the traffic visualizer."""
        self.figsize = figsize
        self.max_trail_length = max_trail_length
        
        # Create figure and subplots
        self.fig = plt.figure(figsize=figsize)
        gs = self.fig.add_gridspec(3, 3, height_ratios=[2, 1, 1], width_ratios=[2, 1, 1])
        
        # Main traffic flow plot
        self.ax_main = self.fig.add_subplot(gs[0, :2])
        
        # Metrics plots
        self.ax_rewards = self.fig.add_subplot(gs[0, 2])
        self.ax_collisions = self.fig.add_subplot(gs[1, 0])
        self.ax_network = self.fig.add_subplot(gs[1, 1])
        self.ax_speeds = self.fig.add_subplot(gs[1, 2])
        self.ax_queue = self.fig.add_subplot(gs[2, :])
        
        # Data storage
        self.vehicle_trails = {}  # Vehicle movement trails
        self.vehicle_positions = {}
        self.vehicle_states = {}
        self.messages = []  # 6G messages
        self.collision_events = []
        self.reservation_status = {}
        
        # Metrics history
        self.time_history = deque(maxlen=200)
        self.reward_history = deque(maxlen=200)
        self.collision_history = deque(maxlen=200)
        self.speed_history = deque(maxlen=200)
        self.network_history = deque(maxlen=200)
        self.queue_history = deque(maxlen=200)
        
        # Animation objects
        self.vehicle_patches = {}
        self.message_lines = []
        self.collision_patches = []
        
        # Setup plots
        self._setup_main_plot()
        self._setup_metric_plots()
        
        # Data queue for thread-safe updates
        self.data_queue = queue.Queue()
        self.is_running = False
        
    def _setup_main_plot(self):
        """Setup the main traffic flow visualization."""
        self.ax_main.set_xlim(-150, 150)
        self.ax_main.set_ylim(-150, 150)
        self.ax_main.set_aspect('equal')
        self.ax_main.set_title('6G Traffic Flow - Real-Time Vehicle Coordination', fontsize=14, fontweight='bold')
        self.ax_main.set_xlabel('Distance (meters)')
        self.ax_main.set_ylabel('Distance (meters)')
        self.ax_main.grid(True, alpha=0.3)
        
        # Draw intersection
        intersection_size = 30
        intersection = Rectangle((-intersection_size/2, -intersection_size/2), 
                               intersection_size, intersection_size, 
                               facecolor='lightgray', edgecolor='black', linewidth=2, alpha=0.7)
        self.ax_main.add_patch(intersection)
        
        # Draw roads
        road_width = 20
        # Horizontal road
        horizontal_road = Rectangle((-150, -road_width/2), 300, road_width, 
                                  facecolor='darkgray', alpha=0.5)
        self.ax_main.add_patch(horizontal_road)
        
        # Vertical road  
        vertical_road = Rectangle((-road_width/2, -150), road_width, 300, 
                                facecolor='darkgray', alpha=0.5)
        self.ax_main.add_patch(vertical_road)
        
        # Add road markings
        for x in range(-150, 151, 20):
            if abs(x) > 20:  # Skip intersection area
                self.ax_main.plot([x, x+10], [0, 0], 'w--', linewidth=1, alpha=0.7)
        for y in range(-150, 151, 20):
            if abs(y) > 20:  # Skip intersection area
                self.ax_main.plot([0, 0], [y, y+10], 'w--', linewidth=1, alpha=0.7)
        
        # Legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='blue', markersize=8, label='Active Vehicle'),
            plt.Line2D([0], [0], marker='s', color='red', markersize=8, label='Collision'),
            plt.Line2D([0], [0], color='green', linewidth=2, label='6G Communication'),
            plt.Line2D([0], [0], color='orange', linewidth=1, alpha=0.7, label='Vehicle Trail')
        ]
        self.ax_main.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1, 1))
        
    def _setup_metric_plots(self):
        """Setup metric visualization plots."""
        # Rewards plot
        self.ax_rewards.set_title('Episode Returns', fontsize=10, fontweight='bold')
        self.ax_rewards.set_ylabel('Return')
        self.ax_rewards.grid(True, alpha=0.3)
        
        # Collisions plot
        self.ax_collisions.set_title('Collision Rate', fontsize=10, fontweight='bold')
        self.ax_collisions.set_ylabel('Collisions/Episode')
        self.ax_collisions.grid(True, alpha=0.3)
        
        # Network metrics
        self.ax_network.set_title('6G Network Status', fontsize=10, fontweight='bold')
        self.ax_network.set_ylabel('Reliability')
        self.ax_network.grid(True, alpha=0.3)
        
        # Speed metrics
        self.ax_speeds.set_title('Average Speed', fontsize=10, fontweight='bold')
        self.ax_speeds.set_ylabel('Speed (m/s)')
        self.ax_speeds.grid(True, alpha=0.3)
        
        # Queue analysis
        self.ax_queue.set_title('M/M/c Queue Analysis - Traffic Flow', fontsize=10, fontweight='bold')
        self.ax_queue.set_xlabel('Time')
        self.ax_queue.set_ylabel('Queue Length')
        self.ax_queue.grid(True, alpha=0.3)
        
    def update_data(self, env_data: Dict):
        """Update visualization with new environment data."""
        if not self.is_running:
            return
            
        # Add data to queue for thread-safe processing
        self.data_queue.put(env_data)
        
    def _process_data_update(self, env_data: Dict):
        """Process environment data update."""
        current_time = env_data.get('current_time', 0)
        
        # Update vehicle data
        vehicles = env_data.get('vehicles', {})
        for vehicle_id, vehicle_data in vehicles.items():
            pos = (vehicle_data['x'], vehicle_data['y'])
            
            # Update position
            self.vehicle_positions[vehicle_id] = pos
            self.vehicle_states[vehicle_id] = vehicle_data
            
            # Update trail
            if vehicle_id not in self.vehicle_trails:
                self.vehicle_trails[vehicle_id] = deque(maxlen=self.max_trail_length)
            self.vehicle_trails[vehicle_id].append(pos)
        
        # Update messages
        self.messages = env_data.get('messages', [])
        
        # Update collision events
        if env_data.get('collision_detected', False):
            self.collision_events.append({
                'time': current_time,
                'position': env_data.get('collision_position', (0, 0))
            })
        
        # Update reservation status
        self.reservation_status = env_data.get('reservations', {})
        
        # Update metrics history
        self.time_history.append(current_time)
        self.reward_history.append(env_data.get('total_reward', 0))
        self.collision_history.append(env_data.get('collision_count', 0))
        
        # Calculate average speed
        if vehicles:
            avg_speed = np.mean([v.get('speed', 0) for v in vehicles.values()])
        else:
            avg_speed = 0
        self.speed_history.append(avg_speed)
        
        # Network metrics
        network_data = env_data.get('network_metrics', {})
        self.network_history.append(network_data.get('reliability', 1.0))
        
        # Queue metrics
        queue_data = env_data.get('queue_metrics', {})
        self.queue_history.append(queue_data.get('queue_length', 0))
        
    def _animate(self, frame):
        """Animation update function."""
        # Process any pending data updates
        try:
            while True:
                env_data = self.data_queue.get_nowait()
                self._process_data_update(env_data)
        except queue.Empty:
            pass
        
        # Clear previous dynamic elements
        for patch in self.collision_patches:
            patch.remove()
        self.collision_patches.clear()
        
        for line in self.message_lines:
            line.remove()
        self.message_lines.clear()
        
        # Update vehicles
        for vehicle_id, position in self.vehicle_positions.items():
            vehicle_state = self.vehicle_states.get(vehicle_id, {})
            
            # Create or update vehicle patch
            if vehicle_id not in self.vehicle_patches:
                # Vehicle represented as circle
                color = 'blue'
                if vehicle_state.get('collision_occurred', False):
                    color = 'red'
                elif vehicle_state.get('trip_completed', False):
                    color = 'green'
                elif vehicle_id in self.reservation_status:
                    color = 'orange'
                    
                vehicle_patch = Circle(position, 3, facecolor=color, edgecolor='black', linewidth=1)
                self.ax_main.add_patch(vehicle_patch)
                self.vehicle_patches[vehicle_id] = vehicle_patch
            else:
                # Update existing patch
                vehicle_patch = self.vehicle_patches[vehicle_id]
                vehicle_patch.center = position
                
                # Update color based on state
                color = 'blue'
                if vehicle_state.get('collision_occurred', False):
                    color = 'red'
                elif vehicle_state.get('trip_completed', False):
                    color = 'green'
                elif vehicle_id in self.reservation_status:
                    color = 'orange'
                vehicle_patch.set_facecolor(color)
            
            # Draw vehicle trail
            if vehicle_id in self.vehicle_trails and len(self.vehicle_trails[vehicle_id]) > 1:
                trail_points = list(self.vehicle_trails[vehicle_id])
                trail_x = [p[0] for p in trail_points]
                trail_y = [p[1] for p in trail_points]
                trail_line = self.ax_main.plot(trail_x, trail_y, 'orange', alpha=0.5, linewidth=1)[0]
                self.message_lines.append(trail_line)
            
            # Add vehicle ID label
            label = self.ax_main.text(position[0], position[1] + 5, f'V{vehicle_id[-1]}', 
                                    ha='center', va='bottom', fontsize=8, fontweight='bold')
            self.message_lines.append(label)
        
        # Draw 6G communications
        for msg in self.messages:
            sender_pos = msg.get('sender_pos')
            receiver_pos = msg.get('receiver_pos')
            if sender_pos and receiver_pos:
                comm_line = self.ax_main.plot([sender_pos[0], receiver_pos[0]], 
                                            [sender_pos[1], receiver_pos[1]], 
                                            'green', linewidth=2, alpha=0.7)[0]
                self.message_lines.append(comm_line)
        
        # Draw recent collisions
        current_time = max(self.time_history) if self.time_history else 0
        for collision in self.collision_events:
            if current_time - collision['time'] < 2.0:  # Show for 2 seconds
                pos = collision['position']
                collision_patch = Circle(pos, 5, facecolor='red', alpha=0.8, edgecolor='darkred', linewidth=2)
                self.ax_main.add_patch(collision_patch)
                self.collision_patches.append(collision_patch)
        
        # Update metric plots
        self._update_metric_plots()
        
        return []
    
    def _update_metric_plots(self):
        """Update all metric plots."""
        if len(self.time_history) < 2:
            return
            
        time_data = list(self.time_history)
        
        # Clear and update rewards
        self.ax_rewards.clear()
        self.ax_rewards.plot(time_data, list(self.reward_history), 'b-', linewidth=2)
        self.ax_rewards.set_title('Episode Returns', fontsize=10, fontweight='bold')
        self.ax_rewards.set_ylabel('Return')
        self.ax_rewards.grid(True, alpha=0.3)
        
        # Clear and update collisions
        self.ax_collisions.clear()
        self.ax_collisions.plot(time_data, list(self.collision_history), 'r-', linewidth=2)
        self.ax_collisions.set_title('Collision Rate', fontsize=10, fontweight='bold')
        self.ax_collisions.set_ylabel('Collisions')
        self.ax_collisions.grid(True, alpha=0.3)
        
        # Clear and update network
        self.ax_network.clear()
        self.ax_network.plot(time_data, list(self.network_history), 'g-', linewidth=2)
        self.ax_network.set_title('6G Network Status', fontsize=10, fontweight='bold')
        self.ax_network.set_ylabel('Reliability')
        self.ax_network.set_ylim(0, 1.1)
        self.ax_network.grid(True, alpha=0.3)
        
        # Clear and update speeds
        self.ax_speeds.clear()
        self.ax_speeds.plot(time_data, list(self.speed_history), 'm-', linewidth=2)
        self.ax_speeds.set_title('Average Speed', fontsize=10, fontweight='bold')
        self.ax_speeds.set_ylabel('Speed (m/s)')
        self.ax_speeds.grid(True, alpha=0.3)
        
        # Clear and update queue
        self.ax_queue.clear()
        self.ax_queue.plot(time_data, list(self.queue_history), 'orange', linewidth=2)
        self.ax_queue.set_title('M/M/c Queue Analysis', fontsize=10, fontweight='bold')
        self.ax_queue.set_xlabel('Time (s)')
        self.ax_queue.set_ylabel('Queue Length')
        self.ax_queue.grid(True, alpha=0.3)
        
    def start_visualization(self, interval=100):
        """Start the real-time visualization."""
        self.is_running = True
        self.ani = animation.FuncAnimation(self.fig, self._animate, interval=interval, 
                                         blit=False, cache_frame_data=False)
        plt.tight_layout()
        plt.show(block=True)
        
    def stop_visualization(self):
        """Stop the visualization."""
        self.is_running = False
        if hasattr(self, 'ani'):
            self.ani.event_source.stop()
            
    def save_frame(self, filename):
        """Save current visualization frame."""
        self.fig.savefig(filename, dpi=300, bbox_inches='tight')

class TrafficDataLogger:
    """
    Data logger that extracts visualization data from the environment.
    """
    
    def __init__(self, visualizer: TrafficFlowVisualizer):
        self.visualizer = visualizer
        
    def log_environment_state(self, env, additional_info=None):
        """Extract and log environment state for visualization."""
        if not hasattr(env, 'vehicles'):
            return
            
        # Extract vehicle data
        vehicles_data = {}
        messages_data = []
        
        for vehicle_id, vehicle in env.vehicles.items():
            vehicles_data[vehicle_id] = {
                'x': vehicle.x,
                'y': vehicle.y,
                'vx': vehicle.vx,
                'vy': vehicle.vy,
                'speed': np.sqrt(vehicle.vx**2 + vehicle.vy**2),
                'trip_completed': vehicle.trip_completed,
                'collision_occurred': vehicle.collision_occurred,
                'total_travel_time': vehicle.total_travel_time
            }
        
        # Extract network messages (simplified for visualization)
        if hasattr(env, 'network') and hasattr(env.network, 'message_queue'):
            for msg_entry in env.network.message_queue[-10:]:  # Last 10 messages
                try:
                    if len(msg_entry) >= 4:
                        deliver_time, receiver, message, sender = msg_entry[:4]
                        if hasattr(sender, 'x') and hasattr(receiver, 'x'):
                            messages_data.append({
                                'sender_pos': (sender.x, sender.y),
                                'receiver_pos': (receiver.x, receiver.y),
                                'type': message.get('type', 'unknown') if isinstance(message, dict) else 'message'
                            })
                except (ValueError, AttributeError) as e:
                    # Skip malformed message entries
                    continue
        
        # Get intersection reservations
        reservations = {}
        if hasattr(env, 'intersection'):
            reservations = env.intersection.reservations
        
        # Get metrics
        network_metrics = {}
        queue_metrics = {}
        if hasattr(env, 'network'):
            network_metrics = env.network.get_network_metrics()
        if hasattr(env, 'intersection'):
            intersection_metrics = env.intersection.get_metrics()
            queue_metrics = intersection_metrics.get('mmc_analysis', {}).get('empirical', {})
        
        # Compile data
        env_data = {
            'current_time': getattr(env, 'current_time', 0),
            'vehicles': vehicles_data,
            'messages': messages_data,
            'reservations': reservations,
            'network_metrics': network_metrics,
            'queue_metrics': queue_metrics,
            'collision_detected': env.total_collisions > getattr(self, '_last_collision_count', 0),
            'collision_count': env.total_collisions,
            'total_reward': sum([v.get('total_travel_time', 0) for v in vehicles_data.values()])
        }
        
        self._last_collision_count = env.total_collisions
        
        # Update additional info
        if additional_info:
            env_data.update(additional_info)
            
        # Send to visualizer
        self.visualizer.update_data(env_data)

def create_training_visualizer():
    """Create a traffic flow visualizer for training monitoring."""
    visualizer = TrafficFlowVisualizer(figsize=(20, 12))
    logger = TrafficDataLogger(visualizer)
    return visualizer, logger

if __name__ == "__main__":
    # Example usage
    print("🎯 Phase 2: Traffic Flow Visualization System")
    print("=====================================")
    print("Features:")
    print("✅ Real-time vehicle movement visualization")
    print("✅ 6G communication link display")
    print("✅ Intersection coordination monitoring")
    print("✅ Collision detection and visualization")
    print("✅ Training metrics dashboard")
    print("✅ M/M/c queue analysis plots")
    print("\nIntegration: Use with TrafficDataLogger during training")
    
    # Create demo visualization
    visualizer = TrafficFlowVisualizer()
    
    # Simulate some demo data
    demo_data = {
        'current_time': 10.5,
        'vehicles': {
            'vehicle_0': {'x': -50, 'y': 0, 'vx': 10, 'vy': 0, 'speed': 10, 
                         'trip_completed': False, 'collision_occurred': False, 'total_travel_time': 5.0},
            'vehicle_1': {'x': 0, 'y': -30, 'vx': 0, 'vy': 8, 'speed': 8,
                         'trip_completed': False, 'collision_occurred': False, 'total_travel_time': 3.0}
        },
        'messages': [
            {'sender_pos': (-50, 0), 'receiver_pos': (0, 0), 'type': 'reservation_request'}
        ],
        'network_metrics': {'reliability': 0.99},
        'queue_metrics': {'queue_length': 2},
        'collision_detected': False,
        'collision_count': 0,
        'total_reward': 100
    }
    
    visualizer.update_data(demo_data)
    print("\n🚀 Starting demo visualization...")
    print("Close the window to exit.")
    
    # Start visualization
    try:
        visualizer.start_visualization(interval=50)
    except KeyboardInterrupt:
        print("\nVisualization stopped.")
    finally:
        visualizer.stop_visualization() 