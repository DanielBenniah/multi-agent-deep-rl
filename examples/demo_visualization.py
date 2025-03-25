#!/usr/bin/env python3
"""
Demo script for testing the traffic flow visualization system
Simulates realistic traffic patterns without needing full RL training
"""

import numpy as np
import time
import threading
import sys
import os
import matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for better macOS compatibility
import matplotlib.pyplot as plt

# Add path to visualization module
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src', 'visualization'))
from traffic_visualizer import TrafficFlowVisualizer

class TrafficSimulationDemo:
    """
    Simple traffic simulation for demonstrating the visualization system.
    """
    
    def __init__(self):
        self.visualizer = TrafficFlowVisualizer(figsize=(16, 10))
        self.running = False
        self.current_time = 0.0
        self.dt = 0.1
        
        # Initialize demo vehicles
        self.vehicles = {
            'vehicle_0': {'x': -80, 'y': 0, 'vx': 12, 'vy': 0, 'target_speed': 12},
            'vehicle_1': {'x': 0, 'y': -70, 'vx': 0, 'vy': 10, 'target_speed': 10},
            'vehicle_2': {'x': 60, 'y': 0, 'vx': -8, 'vy': 0, 'target_speed': 8},
            'vehicle_3': {'x': 0, 'y': 80, 'vx': 0, 'vy': -11, 'target_speed': 11}
        }
        
        # Simulation state
        self.reservations = {}
        self.collision_count = 0
        self.message_history = []
        
    def simulate_6g_communication(self):
        """Simulate 6G message exchanges between vehicles and intersection."""
        messages = []
        
        # Each vehicle near intersection sends reservation request
        for vid, vehicle in self.vehicles.items():
            dist_to_intersection = np.sqrt(vehicle['x']**2 + vehicle['y']**2)
            
            if 20 < dist_to_intersection < 50 and vid not in self.reservations:
                # Send reservation request
                messages.append({
                    'sender_pos': (vehicle['x'], vehicle['y']),
                    'receiver_pos': (0, 0),  # Intersection at origin
                    'type': 'reservation_request'
                })
                
                # Simulate granting reservation
                eta = self.current_time + dist_to_intersection / np.sqrt(vehicle['vx']**2 + vehicle['vy']**2 + 0.1)
                self.reservations[vid] = eta
                
        return messages
    
    def simulate_vehicle_dynamics(self):
        """Simulate realistic vehicle movement with collision avoidance."""
        for vid, vehicle in self.vehicles.items():
            # Check distance to intersection
            dist_to_intersection = np.sqrt(vehicle['x']**2 + vehicle['y']**2)
            
            # Simple collision avoidance: slow down if another vehicle is close
            should_slow = False
            for other_vid, other_vehicle in self.vehicles.items():
                if other_vid != vid:
                    dx = vehicle['x'] - other_vehicle['x']
                    dy = vehicle['y'] - other_vehicle['y']
                    distance = np.sqrt(dx**2 + dy**2)
                    
                    if distance < 15:  # Too close, slow down
                        should_slow = True
                        break
            
            # Adjust speed based on conditions
            if should_slow:
                # Slow down
                speed_factor = 0.3
            elif dist_to_intersection < 20 and vid not in self.reservations:
                # Approaching intersection without reservation, slow down
                speed_factor = 0.5
            else:
                # Normal speed
                speed_factor = 1.0
            
            # Update velocity
            target_vx = vehicle['vx'] / max(np.sqrt(vehicle['vx']**2 + vehicle['vy']**2), 0.1) * vehicle['target_speed'] * speed_factor
            target_vy = vehicle['vy'] / max(np.sqrt(vehicle['vx']**2 + vehicle['vy']**2), 0.1) * vehicle['target_speed'] * speed_factor
            
            vehicle['vx'] = 0.9 * vehicle['vx'] + 0.1 * target_vx
            vehicle['vy'] = 0.9 * vehicle['vy'] + 0.1 * target_vy
            
            # Update position
            vehicle['x'] += vehicle['vx'] * self.dt
            vehicle['y'] += vehicle['vy'] * self.dt
            
            # Reset vehicles that have passed through
            if abs(vehicle['x']) > 120 or abs(vehicle['y']) > 120:
                if vid == 'vehicle_0':
                    vehicle['x'], vehicle['y'] = -80, 0
                elif vid == 'vehicle_1':
                    vehicle['x'], vehicle['y'] = 0, -70
                elif vid == 'vehicle_2':
                    vehicle['x'], vehicle['y'] = 60, 0
                elif vid == 'vehicle_3':
                    vehicle['x'], vehicle['y'] = 0, 80
                
                # Clear reservation
                if vid in self.reservations:
                    del self.reservations[vid]
    
    def generate_demo_data(self):
        """Generate simulation data for visualization."""
        # Simulate 6G communications
        messages = self.simulate_6g_communication()
        
        # Update vehicle dynamics
        self.simulate_vehicle_dynamics()
        
        # Prepare vehicle data for visualization
        vehicles_data = {}
        for vid, vehicle in self.vehicles.items():
            vehicles_data[vid] = {
                'x': vehicle['x'],
                'y': vehicle['y'],
                'vx': vehicle['vx'],
                'vy': vehicle['vy'],
                'speed': np.sqrt(vehicle['vx']**2 + vehicle['vy']**2),
                'trip_completed': False,
                'collision_occurred': False,
                'total_travel_time': self.current_time
            }
        
        # Calculate metrics
        avg_speed = np.mean([v['speed'] for v in vehicles_data.values()])
        queue_length = len([v for v in self.vehicles.values() 
                           if np.sqrt(v['x']**2 + v['y']**2) < 30])
        
        # Create environment data
        env_data = {
            'current_time': self.current_time,
            'vehicles': vehicles_data,
            'messages': messages,
            'reservations': self.reservations,
            'network_metrics': {'reliability': 0.98 + 0.02 * np.sin(self.current_time * 0.1)},
            'queue_metrics': {'queue_length': queue_length},
            'collision_detected': False,
            'collision_count': self.collision_count,
            'total_reward': 50 + 30 * np.sin(self.current_time * 0.05)
        }
        
        return env_data
    
    def run_simulation(self, duration=60):
        """Run the demo simulation."""
        print("🎯 Starting Traffic Flow Visualization Demo")
        print("=========================================")
        print(f"📊 Simulating {duration} seconds of traffic")
        print("🚗 4 vehicles approaching intersection")
        print("📡 6G communication and coordination")
        print("🚦 Signal-free intersection management")
        
        # Run simulation in background thread, visualization in main thread
        def simulation_loop():
            self.running = True
            while self.running and self.current_time < duration:
                # Generate and send data to visualizer
                env_data = self.generate_demo_data()
                self.visualizer.update_data(env_data)
                
                # Update simulation time
                self.current_time += self.dt
                
                # Control simulation speed
                time.sleep(self.dt / 5)  # Run 5x real-time for demo
                
                # Print progress every 10 seconds
                if int(self.current_time) % 10 == 0 and int(self.current_time) != int(self.current_time - self.dt):
                    print(f"⏰ Simulation time: {self.current_time:.1f}s")
                    print(f"🚗 Active vehicles: {len(self.vehicles)}")
                    print(f"📡 Reservations: {len(self.reservations)}")
        
        # Start simulation thread
        sim_thread = threading.Thread(target=simulation_loop)
        sim_thread.daemon = True
        sim_thread.start()
        
        # Start visualization in main thread (required by matplotlib)
        try:
            print("🎬 Opening visualization window...")
            print("   You should see vehicles moving toward intersection!")
            print("   🔵 Blue = vehicles, 🟠 = reserved, 🟢 = communication")
            print("   Close window or press Ctrl+C to stop")
            self.visualizer.start_visualization(interval=50)
        except KeyboardInterrupt:
            print("\n🛑 Demo interrupted by user")
        except Exception as e:
            print(f"\n❌ Visualization window failed: {e}")
            print("🔄 Running text-only mode instead...")
            # Run simple text demo as fallback
            self.running = True
            while self.running and self.current_time < 30:
                env_data = self.generate_demo_data()
                self.current_time += 1
                if int(self.current_time) % 5 == 0:
                    print(f"⏰ Time: {self.current_time:.0f}s | Vehicles at intersection: {env_data['queue_metrics']['queue_length']}")
                time.sleep(0.2)
        finally:
            self.running = False
            print("\n✅ Demo completed!")
            print("📊 Traffic simulation finished successfully")

def main():
    """Run the visualization demo."""
    demo = TrafficSimulationDemo()
    demo.run_simulation(duration=120)  # 2 minutes of simulation

if __name__ == "__main__":
    main() 