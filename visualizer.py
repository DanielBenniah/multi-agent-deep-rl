"""
Visualization Module for 6G-Enabled Autonomous Traffic Management

This module provides real-time visualization of the traffic simulation
including vehicles, intersections, communication networks, and performance metrics.
"""

import pygame
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import math
import time
from collections import deque, defaultdict

# Pygame setup
pygame.init()

# Color definitions
COLORS = {
    'road': (50, 50, 50),
    'intersection': (80, 80, 80),
    'vehicle_normal': (0, 150, 255),
    'vehicle_reserved': (255, 150, 0),
    'vehicle_completed': (0, 255, 0),
    'vehicle_collision': (255, 0, 0),
    'communication': (255, 255, 0),
    'reservation_zone': (255, 255, 255, 50),
    'text': (255, 255, 255),
    'background': (30, 30, 30),
    'grid': (70, 70, 70)
}

class PygameVisualizer:
    """
    Real-time Pygame visualization for the traffic simulation.
    """
    
    def __init__(self, width=1200, height=800, scale=2.0):
        self.width = width
        self.height = height
        self.scale = scale  # pixels per meter
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("6G Traffic Management Simulation")
        
        # Fonts
        self.font_small = pygame.font.Font(None, 20)
        self.font_medium = pygame.font.Font(None, 24)
        self.font_large = pygame.font.Font(None, 32)
        
        # Visualization state
        self.camera_x = 0
        self.camera_y = 0
        self.show_communication = True
        self.show_reservations = True
        self.show_metrics = True
        
        # Performance tracking
        self.fps_target = 30
        self.clock = pygame.time.Clock()
        
        # Data tracking
        self.metrics_history = defaultdict(deque)
        self.max_history = 100
        
    def world_to_screen(self, x, y):
        """Convert world coordinates to screen coordinates."""
        screen_x = int((x - self.camera_x) * self.scale + self.width // 2)
        screen_y = int((y - self.camera_y) * self.scale + self.height // 2)
        return screen_x, screen_y
    
    def draw_grid(self):
        """Draw background grid for reference."""
        grid_size = 20  # meters
        
        # Calculate grid bounds in world space
        left = self.camera_x - self.width // (2 * self.scale)
        right = self.camera_x + self.width // (2 * self.scale)
        top = self.camera_y - self.height // (2 * self.scale)
        bottom = self.camera_y + self.height // (2 * self.scale)
        
        # Draw vertical lines
        start_x = int(left // grid_size) * grid_size
        for world_x in range(start_x, int(right) + grid_size, grid_size):
            screen_x, _ = self.world_to_screen(world_x, 0)
            if 0 <= screen_x <= self.width:
                pygame.draw.line(self.screen, COLORS['grid'], 
                               (screen_x, 0), (screen_x, self.height))
        
        # Draw horizontal lines
        start_y = int(top // grid_size) * grid_size
        for world_y in range(start_y, int(bottom) + grid_size, grid_size):
            _, screen_y = self.world_to_screen(0, world_y)
            if 0 <= screen_y <= self.height:
                pygame.draw.line(self.screen, COLORS['grid'], 
                               (0, screen_y), (self.width, screen_y))
    
    def draw_roads(self, intersections):
        """Draw road network."""
        road_width = int(8 * self.scale)  # 8 meters wide road
        
        for intersection in intersections:
            center_x, center_y = self.world_to_screen(intersection.x, intersection.y)
            
            # Draw intersection
            intersection_rect = pygame.Rect(
                center_x - road_width//2, center_y - road_width//2,
                road_width, road_width
            )
            pygame.draw.rect(self.screen, COLORS['intersection'], intersection_rect)
            
            # Draw roads extending from intersection
            # Horizontal road
            pygame.draw.rect(self.screen, COLORS['road'], 
                           (0, center_y - road_width//2, self.width, road_width))
            
            # Vertical road
            pygame.draw.rect(self.screen, COLORS['road'], 
                           (center_x - road_width//2, 0, road_width, self.height))
            
            # Draw intersection conflict zone (if enabled)
            if self.show_reservations:
                conflict_radius = int(intersection.conflict_radius * self.scale)
                # Create surface with alpha for transparency
                conflict_surface = pygame.Surface((conflict_radius * 2, conflict_radius * 2))
                conflict_surface.set_alpha(50)
                conflict_surface.fill(COLORS['reservation_zone'][:3])
                self.screen.blit(conflict_surface, 
                               (center_x - conflict_radius, center_y - conflict_radius))
    
    def draw_vehicles(self, vehicles):
        """Draw all vehicles with their current status."""
        for vehicle in vehicles.values():
            self.draw_vehicle(vehicle)
    
    def draw_vehicle(self, vehicle):
        """Draw a single vehicle."""
        # Determine vehicle color based on status
        if vehicle.collision_occurred:
            color = COLORS['vehicle_collision']
        elif vehicle.completed_trip:
            color = COLORS['vehicle_completed']
        elif vehicle.has_reservation:
            color = COLORS['vehicle_reserved']
        else:
            color = COLORS['vehicle_normal']
        
        # Vehicle position and size
        center_x, center_y = self.world_to_screen(vehicle.x, vehicle.y)
        vehicle_length = int(vehicle.length * self.scale)
        vehicle_width = int(vehicle.width * self.scale)
        
        # Draw vehicle rectangle
        vehicle_rect = pygame.Rect(
            center_x - vehicle_width//2, center_y - vehicle_length//2,
            vehicle_width, vehicle_length
        )
        pygame.draw.rect(self.screen, color, vehicle_rect)
        pygame.draw.rect(self.screen, (255, 255, 255), vehicle_rect, 1)
        
        # Draw vehicle ID
        id_text = self.font_small.render(str(vehicle.id), True, COLORS['text'])
        self.screen.blit(id_text, (center_x - 10, center_y - 10))
        
        # Draw velocity vector
        if hasattr(vehicle, 'vx') and hasattr(vehicle, 'vy'):
            velocity_scale = 5
            end_x = center_x + int(vehicle.vx * velocity_scale)
            end_y = center_y + int(vehicle.vy * velocity_scale)
            pygame.draw.line(self.screen, color, (center_x, center_y), (end_x, end_y), 2)
            pygame.draw.circle(self.screen, color, (end_x, end_y), 3)
    
    def draw_communication_links(self, vehicles, intersections):
        """Draw communication links between vehicles and intersections."""
        if not self.show_communication:
            return
        
        # Vehicle-to-intersection communication
        for vehicle in vehicles.values():
            if vehicle.has_reservation and vehicle.reservation_intersection:
                # Find the intersection this vehicle has a reservation with
                for intersection in intersections:
                    if intersection.id == vehicle.reservation_intersection:
                        v_pos = self.world_to_screen(vehicle.x, vehicle.y)
                        i_pos = self.world_to_screen(intersection.x, intersection.y)
                        
                        # Draw communication link
                        pygame.draw.line(self.screen, COLORS['communication'], v_pos, i_pos, 1)
                        break
        
        # Vehicle-to-vehicle communication (nearby vehicles)
        for vehicle in vehicles.values():
            v_pos = self.world_to_screen(vehicle.x, vehicle.y)
            
            for other_vehicle in vehicles.values():
                if other_vehicle.id != vehicle.id:
                    distance = math.hypot(other_vehicle.x - vehicle.x, other_vehicle.y - vehicle.y)
                    if distance < 30.0:  # Communication range
                        other_pos = self.world_to_screen(other_vehicle.x, other_vehicle.y)
                        # Draw faint communication link
                        pygame.draw.line(self.screen, (*COLORS['communication'], 100), v_pos, other_pos, 1)
    
    def draw_metrics_panel(self, env_info, network_stats):
        """Draw real-time metrics panel."""
        if not self.show_metrics:
            return
        
        # Panel background
        panel_width = 300
        panel_height = 200
        panel_rect = pygame.Rect(self.width - panel_width - 10, 10, panel_width, panel_height)
        panel_surface = pygame.Surface((panel_width, panel_height))
        panel_surface.set_alpha(200)
        panel_surface.fill((0, 0, 0))
        self.screen.blit(panel_surface, panel_rect)
        
        # Draw metrics text
        y_offset = 20
        line_height = 20
        
        metrics = [
            f"Time: {env_info.get('simulation_time', 0):.1f}s",
            f"Completion Rate: {env_info.get('completion_rate', 0):.2%}",
            f"Collision Rate: {env_info.get('collision_rate', 0):.2%}",
            f"Avg Travel Time: {env_info.get('average_travel_time', 0):.1f}s",
            f"Avg Waiting Time: {env_info.get('average_waiting_time', 0):.1f}s",
            f"Network Reliability: {network_stats.get('reliability', 0):.2%}",
            f"Total Messages: {network_stats.get('total_messages', 0)}",
            f"Dropped Messages: {network_stats.get('dropped_messages', 0)}"
        ]
        
        for i, metric in enumerate(metrics):
            text = self.font_small.render(metric, True, COLORS['text'])
            self.screen.blit(text, (self.width - panel_width, y_offset + i * line_height))
    
    def draw_legend(self):
        """Draw color legend for vehicle states."""
        legend_items = [
            ("Normal Vehicle", COLORS['vehicle_normal']),
            ("Reserved Vehicle", COLORS['vehicle_reserved']),
            ("Completed Trip", COLORS['vehicle_completed']),
            ("Collision", COLORS['vehicle_collision'])
        ]
        
        x = 10
        y = self.height - 100
        
        for i, (label, color) in enumerate(legend_items):
            # Draw color box
            pygame.draw.rect(self.screen, color, (x, y + i * 20, 15, 15))
            # Draw label
            text = self.font_small.render(label, True, COLORS['text'])
            self.screen.blit(text, (x + 20, y + i * 20))
    
    def handle_events(self):
        """Handle pygame events."""
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_c:
                    self.show_communication = not self.show_communication
                elif event.key == pygame.K_r:
                    self.show_reservations = not self.show_reservations
                elif event.key == pygame.K_m:
                    self.show_metrics = not self.show_metrics
                elif event.key == pygame.K_ESCAPE:
                    return False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # Camera control with mouse
                pass
        
        # Keyboard camera controls
        keys = pygame.key.get_pressed()
        camera_speed = 2.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.camera_x -= camera_speed
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.camera_x += camera_speed
        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.camera_y -= camera_speed
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.camera_y += camera_speed
        
        return True
    
    def render_frame(self, env, vehicles, intersections, env_info, network_stats):
        """Render a complete frame."""
        # Clear screen
        self.screen.fill(COLORS['background'])
        
        # Draw scene elements
        self.draw_grid()
        self.draw_roads(intersections)
        self.draw_communication_links(vehicles, intersections)
        self.draw_vehicles(vehicles)
        self.draw_metrics_panel(env_info, network_stats)
        self.draw_legend()
        
        # Update display
        pygame.display.flip()
        self.clock.tick(self.fps_target)
        
        return self.handle_events()

class MatplotlibVisualizer:
    """
    Matplotlib-based visualization for detailed analysis and static plots.
    """
    
    def __init__(self, figsize=(12, 8)):
        self.figsize = figsize
        self.fig = None
        self.axes = None
        
    def plot_simulation_state(self, vehicles, intersections, env_info, network_stats):
        """Create a static plot of the current simulation state."""
        self.fig, self.axes = plt.subplots(2, 2, figsize=self.figsize)
        self.fig.suptitle('6G Traffic Management Simulation State', fontsize=16)
        
        # Plot 1: Vehicle positions and status
        ax1 = self.axes[0, 0]
        self._plot_traffic_scene(ax1, vehicles, intersections)
        
        # Plot 2: Performance metrics
        ax2 = self.axes[0, 1]
        self._plot_performance_metrics(ax2, env_info)
        
        # Plot 3: Network statistics
        ax3 = self.axes[1, 0]
        self._plot_network_stats(ax3, network_stats)
        
        # Plot 4: Vehicle trajectories (if available)
        ax4 = self.axes[1, 1]
        self._plot_vehicle_trajectories(ax4, vehicles)
        
        plt.tight_layout()
        return self.fig
    
    def _plot_traffic_scene(self, ax, vehicles, intersections):
        """Plot the traffic scene with vehicles and intersections."""
        ax.set_title('Traffic Scene')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        
        # Plot intersections
        for intersection in intersections:
            circle = plt.Circle((intersection.x, intersection.y), 
                              intersection.conflict_radius, 
                              fill=False, color='red', linestyle='--', alpha=0.5)
            ax.add_patch(circle)
            ax.plot(intersection.x, intersection.y, 'rs', markersize=10, label='Intersection')
        
        # Plot vehicles with different colors based on status
        for vehicle in vehicles.values():
            if vehicle.collision_occurred:
                color, marker = 'red', 'x'
            elif vehicle.completed_trip:
                color, marker = 'green', 'o'
            elif vehicle.has_reservation:
                color, marker = 'orange', 's'
            else:
                color, marker = 'blue', 'o'
            
            ax.plot(vehicle.x, vehicle.y, marker, color=color, markersize=8)
            ax.annotate(f'V{vehicle.id}', (vehicle.x, vehicle.y), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
            
            # Draw velocity vector
            if hasattr(vehicle, 'vx') and hasattr(vehicle, 'vy'):
                ax.arrow(vehicle.x, vehicle.y, vehicle.vx * 2, vehicle.vy * 2,
                        head_width=1, head_length=1, fc=color, ec=color, alpha=0.7)
        
        ax.set_aspect('equal')
        ax.legend()
    
    def _plot_performance_metrics(self, ax, env_info):
        """Plot performance metrics as a bar chart."""
        ax.set_title('Performance Metrics')
        
        metrics = {
            'Completion Rate': env_info.get('completion_rate', 0),
            'Collision Rate': env_info.get('collision_rate', 0),
            'Avg Travel Time': env_info.get('average_travel_time', 0) / 100,  # Normalize
            'Avg Waiting Time': env_info.get('average_waiting_time', 0) / 100  # Normalize
        }
        
        bars = ax.bar(metrics.keys(), metrics.values())
        
        # Color code bars
        for i, bar in enumerate(bars):
            if i == 0:  # Completion rate - higher is better
                bar.set_color('green')
            elif i == 1:  # Collision rate - lower is better
                bar.set_color('red')
            else:
                bar.set_color('blue')
        
        ax.set_ylabel('Value')
        ax.tick_params(axis='x', rotation=45)
    
    def _plot_network_stats(self, ax, network_stats):
        """Plot network communication statistics."""
        ax.set_title('6G Network Statistics')
        
        # Pie chart of message delivery
        labels = ['Delivered', 'Dropped']
        delivered = network_stats.get('total_messages', 0) - network_stats.get('dropped_messages', 0)
        dropped = network_stats.get('dropped_messages', 0)
        sizes = [delivered, dropped]
        
        if sum(sizes) > 0:
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        else:
            ax.text(0.5, 0.5, 'No Data', transform=ax.transAxes, 
                   ha='center', va='center', fontsize=12)
    
    def _plot_vehicle_trajectories(self, ax, vehicles):
        """Plot vehicle trajectories if available."""
        ax.set_title('Vehicle Trajectories')
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Y Position (m)')
        ax.grid(True, alpha=0.3)
        
        # This would require storing trajectory history
        # For now, just plot current positions
        for vehicle in vehicles.values():
            ax.plot(vehicle.x, vehicle.y, 'o', markersize=6, alpha=0.7)
            ax.annotate(f'V{vehicle.id}', (vehicle.x, vehicle.y), 
                       xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        ax.set_aspect('equal')

def create_performance_dashboard(training_results):
    """Create a comprehensive performance dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('6G Traffic Management - Training Performance Dashboard', fontsize=16)
    
    # Extract metrics from training results
    iterations = [r.get("training_iteration", i) for i, r in enumerate(training_results)]
    rewards = [r.get("episode_reward_mean", 0) for r in training_results]
    
    # Plot 1: Training rewards
    axes[0, 0].plot(iterations, rewards, 'b-', linewidth=2)
    axes[0, 0].set_title('Training Rewards')
    axes[0, 0].set_xlabel('Iteration')
    axes[0, 0].set_ylabel('Mean Episode Reward')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Completion rates
    completion_rates = []
    for result in training_results:
        custom_metrics = result.get("custom_metrics", {})
        completion_rates.append(custom_metrics.get("completion_rate_mean", 0))
    
    axes[0, 1].plot(iterations, completion_rates, 'g-', linewidth=2)
    axes[0, 1].set_title('Trip Completion Rate')
    axes[0, 1].set_xlabel('Iteration')
    axes[0, 1].set_ylabel('Completion Rate')
    axes[0, 1].set_ylim(0, 1)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Collision rates
    collision_rates = []
    for result in training_results:
        custom_metrics = result.get("custom_metrics", {})
        collision_rates.append(custom_metrics.get("collision_rate_mean", 0))
    
    axes[0, 2].plot(iterations, collision_rates, 'r-', linewidth=2)
    axes[0, 2].set_title('Collision Rate')
    axes[0, 2].set_xlabel('Iteration')
    axes[0, 2].set_ylabel('Collision Rate')
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Episode lengths
    episode_lengths = [r.get("episode_len_mean", 0) for r in training_results]
    axes[1, 0].plot(iterations, episode_lengths, 'purple', linewidth=2)
    axes[1, 0].set_title('Episode Length')
    axes[1, 0].set_xlabel('Iteration')
    axes[1, 0].set_ylabel('Mean Steps per Episode')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Learning progress heatmap
    if len(training_results) > 10:
        # Create a heatmap of recent performance
        recent_results = training_results[-20:]  # Last 20 iterations
        metrics_matrix = []
        
        for result in recent_results:
            custom_metrics = result.get("custom_metrics", {})
            row = [
                result.get("episode_reward_mean", 0),
                custom_metrics.get("completion_rate_mean", 0),
                1 - custom_metrics.get("collision_rate_mean", 0),  # Invert collision rate
                1 / max(result.get("episode_len_mean", 1), 1)  # Shorter episodes are better
            ]
            metrics_matrix.append(row)
        
        metrics_matrix = np.array(metrics_matrix).T
        sns.heatmap(metrics_matrix, ax=axes[1, 1], cmap='RdYlGn', 
                   yticklabels=['Reward', 'Completion', 'Safety', 'Efficiency'],
                   xticklabels=[f'Iter {len(training_results)-20+i}' for i in range(20)])
        axes[1, 1].set_title('Recent Performance Heatmap')
    
    # Plot 6: Summary statistics
    axes[1, 2].axis('off')
    if training_results:
        final_result = training_results[-1]
        final_custom = final_result.get("custom_metrics", {})
        
        summary_text = f"""
        Final Performance Summary:
        
        Mean Reward: {final_result.get('episode_reward_mean', 0):.2f}
        Completion Rate: {final_custom.get('completion_rate_mean', 0):.2%}
        Collision Rate: {final_custom.get('collision_rate_mean', 0):.2%}
        Mean Episode Length: {final_result.get('episode_len_mean', 0):.1f}
        
        Best Completion Rate: {max(completion_rates):.2%}
        Lowest Collision Rate: {min(collision_rates):.2%}
        
        Training Iterations: {len(training_results)}
        """
        
        axes[1, 2].text(0.1, 0.9, summary_text, transform=axes[1, 2].transAxes,
                        fontsize=12, verticalalignment='top', fontfamily='monospace',
                        bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    
    plt.tight_layout()
    return fig

# Test visualization
if __name__ == "__main__":
    # Simple test of visualization
    print("Testing visualization components...")
    
    # Test Pygame visualization
    try:
        visualizer = PygameVisualizer()
        print("Pygame visualizer created successfully!")
    except Exception as e:
        print(f"Pygame visualization test failed: {e}")
    
    # Test Matplotlib visualization
    try:
        plt_viz = MatplotlibVisualizer()
        print("Matplotlib visualizer created successfully!")
    except Exception as e:
        print(f"Matplotlib visualization test failed: {e}")
    
    print("Visualization tests completed!") 