#!/usr/bin/env python3
"""
Demo Traffic Flow Visualization - Creates GIF for README
Shows 6G-enabled autonomous vehicles with real-time coordination
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'visualization'))

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from traffic_visualizer import TrafficFlowVisualizer
import time

def create_demo_gif():
    """Create a demo GIF showing 6G traffic coordination."""
    print("🎯 Starting Traffic Flow Visualization Demo")
    print("=========================================")
    print("📊 Simulating 60 seconds of traffic")
    print("🚗 4 vehicles approaching intersection") 
    print("📡 6G communication and coordination")
    print("🚦 Signal-free intersection management")
    print("🎬 Creating GIF for README...")
    
    # Create visualizer
    visualizer = TrafficFlowVisualizer(figsize=(12, 8))
    
    # Storage for frames
    frame_data = []
    
    # Simulate traffic scenario
    dt = 0.1
    total_steps = 600  # 60 seconds at 0.1s timesteps
    
    # Initialize 4 vehicles approaching intersection
    vehicles = {
        'vehicle_0': {'x': -100, 'y': 0, 'vx': 15, 'vy': 0, 'target_speed': 15, 'completed': False},
        'vehicle_1': {'x': 0, 'y': -80, 'vx': 0, 'vy': 12, 'target_speed': 12, 'completed': False},
        'vehicle_2': {'x': 90, 'y': 0, 'vx': -10, 'vy': 0, 'target_speed': 10, 'completed': False},
        'vehicle_3': {'x': 0, 'y': 100, 'vx': 0, 'vy': -14, 'target_speed': 14, 'completed': False}
    }
    
    reservations = {}
    collision_count = 0
    
    for step in range(total_steps):
        current_time = step * dt
        
        # Update vehicle positions with smart coordination
        messages = []
        
        for vid, vehicle in vehicles.items():
            if vehicle['completed']:
                continue
                
            # Calculate distance to intersection
            dist_to_intersection = np.sqrt(vehicle['x']**2 + vehicle['y']**2)
            
            # Smart coordination: request reservation when approaching
            if 20 < dist_to_intersection < 60 and vid not in reservations:
                # Send reservation request
                messages.append({
                    'sender_pos': (vehicle['x'], vehicle['y']),
                    'receiver_pos': (0, 0),
                    'type': 'reservation_request'
                })
                
                # Grant reservation (simulate successful coordination)
                eta = current_time + dist_to_intersection / (vehicle['target_speed'] + 0.1)
                reservations[vid] = eta
            
            # Collision avoidance with other vehicles
            should_slow = False
            for other_vid, other_vehicle in vehicles.items():
                if other_vid != vid and not other_vehicle['completed']:
                    dx = vehicle['x'] - other_vehicle['x']
                    dy = vehicle['y'] - other_vehicle['y']
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < 20:  # Too close, coordinate speeds
                        should_slow = True
                        break
            
            # Adjust speed based on coordination
            if should_slow:
                speed_factor = 0.4  # Slow down for coordination
            elif dist_to_intersection < 25 and vid not in reservations:
                speed_factor = 0.6  # Cautious approach without reservation
            else:
                speed_factor = 1.0  # Normal speed with reservation
            
            # Update velocity with smooth coordination
            current_speed = np.sqrt(vehicle['vx']**2 + vehicle['vy']**2)
            if current_speed > 0.1:
                direction_x = vehicle['vx'] / current_speed
                direction_y = vehicle['vy'] / current_speed
                
                target_speed = vehicle['target_speed'] * speed_factor
                vehicle['vx'] = 0.8 * vehicle['vx'] + 0.2 * direction_x * target_speed
                vehicle['vy'] = 0.8 * vehicle['vy'] + 0.2 * direction_y * target_speed
            
            # Update position
            vehicle['x'] += vehicle['vx'] * dt
            vehicle['y'] += vehicle['vy'] * dt
            
            # Check if vehicle completed trip (passed intersection)
            if abs(vehicle['x']) > 120 or abs(vehicle['y']) > 120:
                vehicle['completed'] = True
                if vid in reservations:
                    del reservations[vid]
        
        # Create 6G communication visualization
        active_vehicles = {k: v for k, v in vehicles.items() if not v['completed']}
        
        # Add inter-vehicle communication for nearby vehicles
        for vid1, v1 in active_vehicles.items():
            for vid2, v2 in active_vehicles.items():
                if vid1 != vid2:
                    distance = np.sqrt((v1['x'] - v2['x'])**2 + (v1['y'] - v2['y'])**2)
                    if distance < 50:  # 6G communication range
                        messages.append({
                            'sender_pos': (v1['x'], v1['y']),
                            'receiver_pos': (v2['x'], v2['y']),
                            'type': 'coordination'
                        })
        
        # Prepare frame data
        vehicles_data = {}
        for vid, vehicle in vehicles.items():
            if not vehicle['completed']:
                vehicles_data[vid] = {
                    'x': vehicle['x'],
                    'y': vehicle['y'],
                    'vx': vehicle['vx'],
                    'vy': vehicle['vy'],
                    'speed': np.sqrt(vehicle['vx']**2 + vehicle['vy']**2),
                    'trip_completed': False,
                    'collision_occurred': False,
                    'total_travel_time': current_time
                }
        
        # Calculate metrics
        active_count = len(vehicles_data)
        avg_speed = np.mean([v['speed'] for v in vehicles_data.values()]) if vehicles_data else 0
        queue_length = len([v for v in vehicles_data.values() 
                           if np.sqrt(v['x']**2 + v['y']**2) < 40])
        
        env_data = {
            'current_time': current_time,
            'vehicles': vehicles_data,
            'messages': messages[-20:],  # Show recent messages only
            'network_metrics': {'reliability': 0.99},
            'queue_metrics': {'queue_length': queue_length},
            'collision_detected': False,
            'collision_count': collision_count,
            'total_reward': 100 + 50 * np.sin(current_time * 0.1),
            'reservations': reservations
        }
        
        frame_data.append(env_data)
        
        # Print progress
        if step % 100 == 0:
            print(f"⏰ Simulation time: {current_time:.1f}s")
            print(f"🚗 Active vehicles: {active_count}")
            print(f"📡 Reservations: {len(reservations)}")
        
        # Stop if all vehicles completed
        if all(v['completed'] for v in vehicles.values()):
            print(f"✅ All vehicles completed trips at {current_time:.1f}s")
            break
    
    print("\n✅ Demo data collected!")
    print("🎨 Creating animated GIF...")
    
    # Create animated GIF
    gif_path, png_path = create_gif_from_data(visualizer, frame_data)
    
    print("📊 Traffic simulation finished successfully")
    return gif_path, png_path

def create_gif_from_data(visualizer, frame_data):
    """Create GIF from simulation data."""
    
    def animate_frame(frame_num):
        """Animation function for creating GIF."""
        if frame_num >= len(frame_data):
            return []
        
        # Update visualizer with frame data
        visualizer.update_data(frame_data[frame_num])
        
        # Force animation update
        visualizer._animate(frame_num)
        
        return []
    
    # Set up animation
    visualizer.is_running = True
    
    # Create animation - sample every 3rd frame for smaller GIF
    sample_frames = frame_data[::3]  # Take every 3rd frame
    
    ani = animation.FuncAnimation(
        visualizer.fig, 
        lambda frame: animate_frame(frame * 3),  # Adjust frame index
        frames=len(sample_frames),
        interval=150,  # 150ms between frames for smooth playback
        blit=False,
        repeat=True
    )
    
    # Save as GIF
    gif_path = "6g_traffic_demo.gif"
    png_path = "6g_traffic_demo.png"
    
    print(f"💾 Saving GIF to: {gif_path}")
    
    try:
        # Try to use pillow writer for GIF
        ani.save(gif_path, writer='pillow', fps=8, dpi=80)
        print(f"✅ GIF saved successfully: {gif_path}")
        
    except Exception as e:
        print(f"⚠️ Error saving GIF with pillow: {e}")
        
        try:
            # Fallback to imagemagick
            ani.save(gif_path, writer='imagemagick', fps=8, dpi=80)
            print(f"✅ GIF saved with imagemagick: {gif_path}")
        except Exception as e2:
            print(f"⚠️ Error saving GIF with imagemagick: {e2}")
            gif_path = None
    
    # Always save a high-quality PNG frame
    try:
        # Use a representative frame (middle of simulation)
        mid_frame = len(frame_data) // 2
        visualizer.update_data(frame_data[mid_frame])
        visualizer._animate(0)
        visualizer.fig.savefig(png_path, dpi=150, bbox_inches='tight')
        print(f"✅ PNG saved successfully: {png_path}")
    except Exception as e:
        print(f"⚠️ Error saving PNG: {e}")
        png_path = None
    
    return gif_path, png_path

if __name__ == "__main__":
    create_demo_gif() 