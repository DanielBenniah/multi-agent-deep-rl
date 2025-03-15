"""
6G-Enabled Autonomous Traffic Management with RLlib Multi-Agent Deep RL
Implementation following the comprehensive architecture from chatGPT_code.txt

Key Features:
- 6G Communication Module with ultra-low latency simulation
- Multi-Agent Deep Reinforcement Learning using RLlib
- Signal-Free Intersection Management with reservation system  
- Traffic Flow Modeling with M/M/c queues
- Proper OpenAI Gym and RLlib MultiAgentEnv compatibility
"""

import math
import random
import logging
from collections import deque, defaultdict
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import threading
import time

import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from ray.rllib.utils.typing import MultiAgentDict
import ray
from ray import tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.sac import SACConfig

# Optional visualization imports (graceful fallback if not available)
try:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'visualization'))
    from traffic_visualizer import create_training_visualizer
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("ℹ️  Visualization not available. Install: pip install matplotlib seaborn")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("6G_Traffic_System")

# Constants for 6G network simulation
DEFAULT_LATENCY_S = 0.001  # 1 ms ultra-low latency
DEFAULT_RANGE = 1000.0     # 6G communication range in meters
DEFAULT_BANDWIDTH = 10e9   # 10 Gbps theoretical 6G bandwidth
THz_FREQUENCY = 1e12       # 1 THz frequency band

class MMcQueueAnalyzer:
    """
    M/M/c Queue analysis for traffic flow validation (as specified in paper).
    Used alongside RL for theoretical baseline and performance validation.
    
    Paper Quote: "integrating the M/M/c model provides an analytical baseline 
    to compare against simulation results and against traditional systems"
    """
    
    def __init__(self):
        self.arrival_data = deque(maxlen=1000)  # Track vehicle arrivals
        self.service_data = deque(maxlen=1000)  # Track service times
        self.queue_lengths = deque(maxlen=1000)  # Track queue lengths
        
    def record_arrival(self, timestamp: float, intersection_id: str):
        """Record vehicle arrival for Poisson process analysis."""
        self.arrival_data.append(timestamp)
        
    def record_service(self, service_time: float, intersection_id: str):
        """Record service time (intersection crossing time)."""
        self.service_data.append(service_time)
        
    def record_queue_length(self, queue_length: int):
        """Record current queue length."""
        self.queue_lengths.append(queue_length)
        
    def calculate_arrival_rate(self) -> float:
        """Calculate λ (lambda) - arrival rate per second."""
        if len(self.arrival_data) < 2:
            return 0.0
        
        time_span = self.arrival_data[-1] - self.arrival_data[0]
        if time_span <= 0:
            return 0.0
            
        return len(self.arrival_data) / time_span
        
    def calculate_service_rate(self) -> float:
        """Calculate μ (mu) - service rate per second."""
        if not self.service_data:
            return 0.0
            
        avg_service_time = sum(self.service_data) / len(self.service_data)
        return 1.0 / avg_service_time if avg_service_time > 0 else 0.0
        
    def calculate_utilization(self, num_servers: int = 1) -> float:
        """Calculate ρ (rho) = λ/(c*μ) - system utilization."""
        lambda_rate = self.calculate_arrival_rate()
        mu_rate = self.calculate_service_rate()
        
        if mu_rate <= 0 or num_servers <= 0:
            return 0.0
            
        return lambda_rate / (num_servers * mu_rate)
        
    def calculate_theoretical_metrics(self, num_servers: int = 1) -> Dict[str, float]:
        """
        Calculate theoretical M/M/c queue metrics for comparison with RL results.
        Paper uses this for validation of signal-free vs traditional systems.
        """
        lambda_rate = self.calculate_arrival_rate()
        mu_rate = self.calculate_service_rate()
        rho = self.calculate_utilization(num_servers)
        
        if rho >= 1.0:  # Unstable system
            return {
                'lambda': lambda_rate,
                'mu': mu_rate,
                'rho': rho,
                'avg_queue_length': float('inf'),
                'avg_waiting_time': float('inf'),
                'stability': 'unstable'
            }
        
        # M/M/c queue formulas (simplified for c=1, can extend for c>1)
        if num_servers == 1:  # M/M/1 queue
            if mu_rate > 0 and rho < 1:
                avg_queue_length = rho / (1 - rho)
                avg_waiting_time = rho / (mu_rate * (1 - rho))
            else:
                avg_queue_length = 0.0
                avg_waiting_time = 0.0
        else:  # M/M/c queue (more complex, simplified here)
            if rho < 1 and lambda_rate > 0:
                avg_queue_length = rho / (1 - rho)
                avg_waiting_time = avg_queue_length / lambda_rate
            else:
                avg_queue_length = 0.0
                avg_waiting_time = 0.0
            
        return {
            'lambda': lambda_rate,
            'mu': mu_rate,
            'rho': rho,
            'avg_queue_length': avg_queue_length,
            'avg_waiting_time': avg_waiting_time,
            'stability': 'stable' if rho < 1 else 'unstable',
            'num_servers': num_servers
        }
        
    def get_empirical_metrics(self) -> Dict[str, float]:
        """Get empirical metrics from simulation for comparison."""
        if not self.queue_lengths:
            return {'avg_queue_length': 0.0, 'max_queue_length': 0}
            
        return {
            'avg_queue_length': sum(self.queue_lengths) / len(self.queue_lengths),
            'max_queue_length': max(self.queue_lengths),
            'min_queue_length': min(self.queue_lengths)
        }

class SixGNetwork:
    """
    Advanced 6G Network simulation with terahertz-band communication,
    network slicing, and ultra-low latency V2V/V2I links.
    """
    
    def __init__(self, latency=DEFAULT_LATENCY_S, comm_range=DEFAULT_RANGE):
        self.latency = latency
        self.range = comm_range
        self.bandwidth = DEFAULT_BANDWIDTH
        
        # Message queue: (deliver_time, receiver, message, sender, priority)
        self.message_queue = []
        
        # Network slicing simulation
        self.slices = {
            'URLLC': {'latency': 0.0005, 'reliability': 0.99999},  # Ultra-reliable low-latency
            'eMBB': {'latency': 0.001, 'reliability': 0.999},      # Enhanced mobile broadband
            'mMTC': {'latency': 0.01, 'reliability': 0.99}         # Massive machine-type communications
        }
        
        # Performance metrics
        self.total_messages = 0
        self.dropped_messages = 0
        self.average_latency = 0.0
        
    def send_message(self, sender, receiver, message, current_time, priority='eMBB'):
        """Send message with 6G network characteristics."""
        self.total_messages += 1
        
        # Calculate distance if both sender and receiver have positions
        dist = 0.0
        if hasattr(sender, 'x') and hasattr(receiver, 'x'):
            dx = sender.x - receiver.x
            dy = sender.y - receiver.y
            dist = math.sqrt(dx*dx + dy*dy)
        
        # Check range and apply path loss for THz frequencies
        if dist > self.range:
            self.dropped_messages += 1
            logger.debug(f"Message dropped: out of range ({dist:.1f}m > {self.range}m)")
            return False
        
        # Network slice characteristics
        slice_config = self.slices.get(priority, self.slices['eMBB'])
        actual_latency = slice_config['latency']
        reliability = slice_config['reliability']
        
        # Simulate message loss based on reliability
        if random.random() > reliability:
            self.dropped_messages += 1
            logger.debug(f"Message dropped: reliability failure")
            return False
        
        # Schedule message delivery
        deliver_time = current_time + actual_latency
        self.message_queue.append((deliver_time, receiver, message, sender, priority))
        
        logger.debug(f"6G message scheduled: {sender} -> {receiver} at t={deliver_time:.4f}s")
        return True
    
    def deliver_messages(self, current_time):
        """Deliver messages that have reached their delivery time."""
        delivered = []
        remaining = []
        
        for msg_data in self.message_queue:
            deliver_time, receiver, message, sender, priority = msg_data
            if deliver_time <= current_time:
                try:
                    receiver.receive_message(message, sender)
                    delivered.append(msg_data)
                    logger.debug(f"6G message delivered: {sender} -> {receiver}")
                except Exception as e:
                    logger.warning(f"Message delivery failed: {e}")
            else:
                remaining.append(msg_data)
        
        self.message_queue = remaining
        return len(delivered)
    
    def get_network_metrics(self):
        """Get network performance metrics."""
        reliability = 1.0 - (self.dropped_messages / max(self.total_messages, 1))
        return {
            'total_messages': self.total_messages,
            'dropped_messages': self.dropped_messages,
            'reliability': reliability,
            'queue_length': len(self.message_queue)
        }

class Vehicle:
    """
    Autonomous vehicle with 6G communication capabilities and RL agent interface.
    """
    
    def __init__(self, vehicle_id: str, x: float, y: float, vx: float = 0.0, vy: float = 0.0):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.vx = vx  # velocity in m/s
        self.vy = vy
        
        # Physical properties
        self.length = 5.0  # meters
        self.width = 2.0   # meters
        self.max_speed = 30.0  # m/s (108 km/h)
        self.max_accel = 3.0   # m/s²
        self.max_decel = -8.0  # m/s² (emergency braking)
        
        # Control
        self.acceleration = 0.0
        self.target_speed = 15.0  # m/s default
        
        # Communication and coordination
        self.network = None
        self.nearby_vehicles = []
        self.intersection_reservations = {}
        
        # State tracking
        self.trip_completed = False
        self.collision_occurred = False
        self.total_travel_time = 0.0
        self.total_waiting_time = 0.0
        self.last_speed = 0.0
        
        # Route (simplified - could be expanded)
        self.destination = None
        self.route_waypoints = []
        
    def set_network(self, network: SixGNetwork):
        """Connect vehicle to 6G network."""
        self.network = network
        
    def get_observation(self, intersections: List, other_vehicles: List) -> np.ndarray:
        """
        Construct observation vector for RL agent.
        Based on the comprehensive state representation described in the paper.
        """
        obs = []
        
        # Own vehicle state
        obs.extend([
            self.x / 100.0,  # normalized position
            self.y / 100.0,
            self.vx / 30.0,  # normalized velocity
            self.vy / 30.0,
            self.acceleration / 10.0,  # normalized acceleration
        ])
        
        # Nearby vehicles (up to 5 closest)
        nearby = sorted(other_vehicles, key=lambda v: self.distance_to(v))[:5]
        for i in range(5):
            if i < len(nearby):
                v = nearby[i]
                rel_x = (v.x - self.x) / 100.0
                rel_y = (v.y - self.y) / 100.0
                rel_vx = (v.vx - self.vx) / 30.0
                rel_vy = (v.vy - self.vy) / 30.0
                obs.extend([rel_x, rel_y, rel_vx, rel_vy])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])  # padding
        
        # Intersection information (closest intersection)
        if intersections:
            closest_intersection = min(intersections, key=lambda i: self.distance_to_point(i.x, i.y))
            dist_to_intersection = self.distance_to_point(closest_intersection.x, closest_intersection.y)
            obs.extend([
                (closest_intersection.x - self.x) / 100.0,  # relative position
                (closest_intersection.y - self.y) / 100.0,
                dist_to_intersection / 100.0,  # normalized distance
                1.0 if self.id in closest_intersection.reservations else 0.0,  # has reservation
            ])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0])
        
        # Traffic density (local)
        vehicles_nearby = len([v for v in other_vehicles if self.distance_to(v) < 50.0])
        obs.append(vehicles_nearby / 10.0)  # normalized traffic density
        
        # Network status
        if self.network:
            metrics = self.network.get_network_metrics()
            obs.append(metrics['reliability'])
            obs.append(min(metrics['queue_length'] / 100.0, 1.0))  # normalized queue length
        else:
            obs.extend([0.0, 0.0])
        
        # Add one more value to match the 33-dimensional space (time step)
        obs.append(min(self.total_travel_time / 100.0, 1.0))  # normalized travel time
        
        # Ensure we have exactly 33 dimensions first
        while len(obs) < 33:
            obs.append(0.0)
        obs = obs[:33]  # Truncate if too long
        
        obs_array = np.array(obs, dtype=np.float32)
        # Ensure the observation is within bounds
        obs_array = np.clip(obs_array, -10.0, 10.0)
        
        return obs_array
    
    def apply_action(self, action: float, dt: float):
        """Apply RL action (acceleration command)."""
        # Action is normalized acceleration command [-1, 1]
        target_accel = action * self.max_accel
        
        # Smooth acceleration change
        self.acceleration = np.clip(target_accel, self.max_decel, self.max_accel)
        
        # Update velocity
        self.vx += self.acceleration * dt
        self.vy += self.acceleration * dt  # simplified 1D motion
        
        # Clamp to max speed
        speed = math.sqrt(self.vx*self.vx + self.vy*self.vy)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            self.vx *= scale
            self.vy *= scale
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Track metrics
        self.total_travel_time += dt
        if speed < 2.0:  # considered waiting if very slow
            self.total_waiting_time += dt
        
        self.last_speed = speed
    
    def request_intersection_reservation(self, intersection, current_time):
        """Request reservation for intersection crossing."""
        if self.network and intersection:
            # Calculate ETA to intersection
            dist = self.distance_to_point(intersection.x, intersection.y)
            speed = max(math.sqrt(self.vx*self.vx + self.vy*self.vy), 1.0)
            eta = current_time + dist / speed
            
            # Create reservation request
            message = {
                'type': 'reservation_request',
                'vehicle_id': self.id,
                'eta': eta,
                'position': (self.x, self.y),
                'velocity': (self.vx, self.vy),
                'priority': 'normal'  # could be 'emergency' for emergency vehicles
            }
            
            # Send via 6G network with URLLC slice for safety-critical messages
            success = self.network.send_message(
                self, intersection, message, current_time, priority='URLLC'
            )
            
            if success:
                logger.debug(f"Vehicle {self.id} requested intersection reservation (ETA: {eta:.2f})")
            
            return success
        return False
    
    def receive_message(self, message: Dict, sender):
        """Handle incoming 6G messages."""
        msg_type = message.get('type')
        
        if msg_type == 'reservation_granted':
            self.intersection_reservations[sender.id] = {
                'time_slot': message['time_slot'],
                'granted_at': message['granted_at']
            }
            logger.debug(f"Vehicle {self.id} received reservation for intersection {sender.id}")
            
        elif msg_type == 'reservation_denied':
            logger.debug(f"Vehicle {self.id} reservation denied for intersection {sender.id}")
            
        elif msg_type == 'vehicle_state_broadcast':
            # Update nearby vehicles information
            vehicle_data = message['vehicle_state']
            self.nearby_vehicles = [v for v in self.nearby_vehicles if v['id'] != vehicle_data['id']]
            self.nearby_vehicles.append(vehicle_data)
            
        elif msg_type == 'emergency_brake':
            # Safety override
            self.acceleration = self.max_decel
            logger.warning(f"Vehicle {self.id} received emergency brake command")
    
    def distance_to(self, other_vehicle) -> float:
        """Calculate distance to another vehicle."""
        return self.distance_to_point(other_vehicle.x, other_vehicle.y)
    
    def distance_to_point(self, x: float, y: float) -> float:
        """Calculate distance to a point."""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)
    
    def get_reward(self, intersections: List, other_vehicles: List, dt: float) -> float:
        """
        Calculate reward for RL training following paper's specification:
        "weighted combination of travel time, safety, energy efficiency, and fairness"
        
        Paper quote: "each time step an agent might receive -1 for each second of travel,
        an additional penalty of -100 for a collision, and smaller penalties for high fuel consumption"
        """
        reward = 0.0
        
        # 1. Travel Time/Throughput Component (negative waiting time)
        # Paper: "-1 for each second of travel (to minimize time)"
        reward -= 1.0 * dt  # base time penalty to encourage faster completion
        
        speed = math.sqrt(self.vx*self.vx + self.vy*self.vy)
        
        # Reward forward progress (throughput)
        if speed > 0.5:  # moving forward
            reward += speed * 0.2  # reward proportional to speed
        
        # Heavy penalty for waiting/stopping (negative waiting time)
        if speed < 2.0:
            reward -= 10.0 * dt  # significant penalty for being slow/stopped
        
        # 2. Safety Component (penalty for collisions or near-misses)
        # Paper: "additional penalty of -100 for a collision"
        if self.collision_occurred:
            reward -= 100.0  # exact penalty from paper
            
        # Near-miss penalties (distance-based safety)
        min_safe_distance = 8.0  # meters
        for other in other_vehicles:
            if other.id != self.id:
                dist = self.distance_to(other)
                if dist < min_safe_distance:
                    # Exponential penalty for getting too close
                    proximity_penalty = (min_safe_distance - dist) ** 2
                    reward -= proximity_penalty * 2.0
        
        # 3. Energy Efficiency Component (penalize excessive acceleration/braking)
        # Paper: "smaller penalties for high fuel consumption or discomfort"
        fuel_consumption = abs(self.acceleration) * dt
        reward -= fuel_consumption * 1.0  # energy efficiency penalty
        
        # Comfort penalty for excessive jerk (change in acceleration)
        if hasattr(self, 'prev_acceleration'):
            jerk = abs(self.acceleration - self.prev_acceleration) / dt
            reward -= jerk * 0.5  # comfort penalty
        self.prev_acceleration = self.acceleration
        
        # 4. Fairness Component (intersection cooperation)
        # Reward cooperative behavior at intersections
        for intersection in intersections:
            dist_to_intersection = self.distance_to_point(intersection.x, intersection.y)
            
            if dist_to_intersection < 30.0:  # approaching intersection
                if self.id in intersection.reservations:
                    reward += 3.0  # bonus for having reservation (cooperation)
                elif dist_to_intersection < 15.0:
                    reward -= 5.0  # penalty for not requesting reservation when close
                    
        # 5. Trip Completion Bonus
        if self.trip_completed:
            # Efficiency bonus based on travel time
            time_efficiency = max(0, 100 - self.total_travel_time)
            reward += time_efficiency  # bonus for completing trip efficiently
        
        # 6. Network Efficiency (6G communication bonus)
        if self.network:
            network_metrics = self.network.get_network_metrics()
            if network_metrics['reliability'] > 0.99:
                reward += 1.0  # bonus for maintaining high network reliability
        
        return reward
    
    def check_collision(self, other_vehicles: List) -> bool:
        """Check for collision with other vehicles."""
        for other in other_vehicles:
            if other.id != self.id:
                dist = self.distance_to(other)
                # More conservative collision detection - vehicles must actually overlap
                collision_threshold = (self.length + other.length) / 2 - 0.5  # Tighter threshold
                if dist < collision_threshold:
                    self.collision_occurred = True
                    other.collision_occurred = True
                    return True
        return False
    
    def __repr__(self):
        return f"Vehicle({self.id})"

class IntersectionManager:
    """
    Signal-free intersection management using reservation-based protocol.
    Implements the architecture described in the paper.
    """
    
    def __init__(self, intersection_id: str, x: float, y: float, conflict_radius: float = 15.0):
        self.id = intersection_id
        self.x = x
        self.y = y
        self.conflict_radius = conflict_radius
        
        # Reservation system
        self.reservations = {}  # vehicle_id -> reservation_data
        self.reservation_slots = []  # time-ordered list of reservations
        self.current_occupant = None
        
        # Performance metrics
        self.vehicles_served = 0
        self.total_wait_time = 0.0
        self.conflicts_resolved = 0
        
        # Network reference
        self.network = None
        
        # M/M/c Queue Analysis (for validation alongside RL)
        self.queue_analyzer = MMcQueueAnalyzer()
        self.entering_vehicles = []  # Track vehicles entering intersection
        self.service_start_times = {}  # vehicle_id -> start_time
        
    def set_network(self, network: SixGNetwork):
        """Connect intersection to 6G network."""
        self.network = network
        
    def receive_message(self, message: Dict, sender):
        """Process messages from vehicles."""
        msg_type = message.get('type')
        
        if msg_type == 'reservation_request':
            self.process_reservation_request(message, sender)
        
    def process_reservation_request(self, request: Dict, vehicle):
        """
        Process vehicle reservation request using conflict detection.
        Implements the reservation-based protocol from the paper.
        """
        vehicle_id = request['vehicle_id']
        requested_eta = request['eta']
        priority = request.get('priority', 'normal')
        
        # Check for conflicts with existing reservations
        conflict_found = False
        conflict_window = 2.0  # seconds - time needed to clear intersection
        
        for slot_time, reserved_vehicle_id in self.reservation_slots:
            if abs(slot_time - requested_eta) < conflict_window:
                if priority != 'emergency':  # emergency vehicles override
                    conflict_found = True
                    break
                else:
                    # Emergency vehicle preempts existing reservation
                    self.cancel_reservation(reserved_vehicle_id)
                    self.conflicts_resolved += 1
        
        if not conflict_found:
            # Grant reservation
            self.grant_reservation(vehicle_id, requested_eta, vehicle)
        else:
            # Deny reservation
            self.deny_reservation(vehicle_id, vehicle)
    
    def grant_reservation(self, vehicle_id: str, time_slot: float, vehicle):
        """Grant intersection reservation to vehicle."""
        self.reservations[vehicle_id] = {
            'time_slot': time_slot,
            'granted_at': time_slot,
            'vehicle': vehicle
        }
        
        # Add to time-ordered slots
        self.reservation_slots.append((time_slot, vehicle_id))
        self.reservation_slots.sort()
        
        # Send confirmation via 6G
        if self.network:
            response = {
                'type': 'reservation_granted',
                'time_slot': time_slot,
                'granted_at': time_slot,
                'intersection_id': self.id
            }
            
            self.network.send_message(
                self, vehicle, response, time_slot, priority='URLLC'
            )
        
        logger.debug(f"Intersection {self.id}: Granted reservation to {vehicle_id} for t={time_slot:.2f}")
    
    def deny_reservation(self, vehicle_id: str, vehicle):
        """Deny intersection reservation to vehicle."""
        if self.network:
            response = {
                'type': 'reservation_denied',
                'reason': 'time_conflict',
                'intersection_id': self.id
            }
            
            self.network.send_message(
                self, vehicle, response, 0, priority='URLLC'
            )
        
        logger.debug(f"Intersection {self.id}: Denied reservation to {vehicle_id}")
    
    def cancel_reservation(self, vehicle_id: str):
        """Cancel existing reservation (e.g., for emergency vehicle)."""
        if vehicle_id in self.reservations:
            del self.reservations[vehicle_id]
            self.reservation_slots = [(t, v) for t, v in self.reservation_slots if v != vehicle_id]
            logger.debug(f"Intersection {self.id}: Cancelled reservation for {vehicle_id}")
    
    def update(self, vehicles: List[Vehicle], current_time: float):
        """Update intersection state each simulation step and collect M/M/c data."""
        # Clean up old reservations
        current_reservations = {}
        current_slots = []
        
        for vehicle_id, res_data in self.reservations.items():
            if res_data['time_slot'] > current_time - 5.0:  # keep recent reservations
                current_reservations[vehicle_id] = res_data
                current_slots.append((res_data['time_slot'], vehicle_id))
        
        self.reservations = current_reservations
        self.reservation_slots = sorted(current_slots)
        
        # M/M/c Analysis: Track vehicles approaching intersection
        vehicles_in_intersection = []
        vehicles_approaching = []
        
        for vehicle in vehicles:
            dist = math.sqrt((vehicle.x - self.x)**2 + (vehicle.y - self.y)**2)
            if dist < self.conflict_radius:
                vehicles_in_intersection.append(vehicle)
                
                # Record arrival for M/M/c analysis
                if vehicle.id not in self.entering_vehicles:
                    self.queue_analyzer.record_arrival(current_time, self.id)
                    self.entering_vehicles.append(vehicle.id)
                    self.service_start_times[vehicle.id] = current_time
                    logger.debug(f"M/M/c: Vehicle {vehicle.id} entered intersection at t={current_time:.2f}")
                    
            elif dist < 50.0:  # approaching within 50m
                vehicles_approaching.append(vehicle)
        
        # Record queue length for M/M/c analysis (vehicles waiting to enter)
        queue_length = len(vehicles_approaching)
        self.queue_analyzer.record_queue_length(queue_length)
        
        # Track service completion (vehicles leaving intersection)
        if self.current_occupant is not None:
            current_occupant_id = self.current_occupant.id
            if current_occupant_id not in [v.id for v in vehicles_in_intersection]:
                # Vehicle just left intersection - record service time
                if current_occupant_id in self.service_start_times:
                    service_time = current_time - self.service_start_times[current_occupant_id]
                    self.queue_analyzer.record_service(service_time, self.id)
                    logger.debug(f"M/M/c: Vehicle {current_occupant_id} service time: {service_time:.2f}s")
                    
                    # Clean up tracking
                    del self.service_start_times[current_occupant_id]
                    if current_occupant_id in self.entering_vehicles:
                        self.entering_vehicles.remove(current_occupant_id)
                
                self.vehicles_served += 1
                self.current_occupant = None
        
        # Update current occupant
        if vehicles_in_intersection:
            self.current_occupant = vehicles_in_intersection[0]  # assume one at a time
    
    def get_metrics(self) -> Dict:
        """Get intersection performance metrics including M/M/c analysis."""
        # Basic intersection metrics
        basic_metrics = {
            'vehicles_served': self.vehicles_served,
            'active_reservations': len(self.reservations),
            'conflicts_resolved': self.conflicts_resolved,
            'current_occupant': self.current_occupant.id if self.current_occupant else None
        }
        
        # M/M/c Queue Analysis Metrics (as specified in paper)
        # Paper: "verify that the simulation's measured average delay aligns with M/M/c predictions"
        
        # Calculate theoretical metrics (M/M/1 for single intersection)
        theoretical_metrics = self.queue_analyzer.calculate_theoretical_metrics(num_servers=1)
        
        # Get empirical metrics from simulation
        empirical_metrics = self.queue_analyzer.get_empirical_metrics()
        
        # Combine all metrics
        combined_metrics = {
            **basic_metrics,
            'mmc_analysis': {
                'theoretical': theoretical_metrics,
                'empirical': empirical_metrics,
                'comparison': {
                    'queue_length_diff': abs(
                        theoretical_metrics.get('avg_queue_length', 0) - 
                        empirical_metrics.get('avg_queue_length', 0)
                    ),
                    'system_stability': theoretical_metrics.get('stability', 'unknown'),
                    'utilization_ratio': theoretical_metrics.get('rho', 0.0)
                }
            }
        }
        
        return combined_metrics
    
    def __repr__(self):
        return f"Intersection({self.id})"

class SixGTrafficEnv(MultiAgentEnv):
    """
    Multi-Agent Environment for 6G-enabled traffic management.
    Compatible with RLlib's MultiAgentEnv interface.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        # Configuration
        self.config = config or {}
        self.num_vehicles = self.config.get('num_vehicles', 4)
        self.max_episode_steps = self.config.get('max_episode_steps', 1000)
        self.dt = self.config.get('dt', 0.1)  # simulation timestep
        
        # Create 6G network
        self.network = SixGNetwork(
            latency=self.config.get('latency', 0.001),
            comm_range=self.config.get('comm_range', 500.0)
        )
        
        # Create intersection
        self.intersection = IntersectionManager('intersection_0', 0.0, 0.0)
        self.intersection.set_network(self.network)
        
        # Create vehicles
        self.vehicles = {}
        self.agents = []
        
        for i in range(self.num_vehicles):
            vehicle_id = f"vehicle_{i}"
            
            # Spawn vehicles from different directions with safe distances
            if i % 4 == 0:  # from west
                x, y = -100.0, 0.0
                vx, vy = 12.0, 0.0
            elif i % 4 == 1:  # from east  
                x, y = 100.0, 0.0
                vx, vy = -12.0, 0.0
            elif i % 4 == 2:  # from south
                x, y = 0.0, -100.0
                vx, vy = 0.0, 12.0
            else:  # from north
                x, y = 0.0, 100.0
                vx, vy = 0.0, -12.0
            
            vehicle = Vehicle(vehicle_id, x, y, vx, vy)
            vehicle.set_network(self.network)
            
            self.vehicles[vehicle_id] = vehicle
            self.agents.append(vehicle_id)
        
        # Define observation and action spaces for multi-agent
        # Observation space: 33-dimensional (as mentioned in the paper)
        obs_dim = 33
        single_obs_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )
        
        # Multi-agent observation and action spaces
        self.observation_space = spaces.Dict({
            agent_id: single_obs_space for agent_id in self.agents
        })
        self.action_space = spaces.Dict({
            agent_id: single_action_space for agent_id in self.agents
        })
        
        # Environment state
        self.current_time = 0.0
        self.episode_step = 0
        
        # Metrics
        self.total_vehicles_completed = 0
        self.total_collisions = 0
        self.total_delay_time = 0.0
        
        # RLlib compatibility
        self._agent_ids = set(self.agents)
        self.possible_agents = self.agents.copy()
    
    def reset(self, *, seed=None, options=None) -> Tuple[MultiAgentDict, MultiAgentDict]:
        """Reset environment to initial state with gymnasium API."""
        self.current_time = 0.0
        self.episode_step = 0
        
        # Reset vehicles to initial positions with staggered spacing
        spacing_offset = 20.0  # Additional spacing between vehicles from same direction
        for i, (vehicle_id, vehicle) in enumerate(self.vehicles.items()):
            if i % 4 == 0:  # from west
                extra_offset = (i // 4) * spacing_offset
                vehicle.x, vehicle.y = -100.0 - extra_offset, 0.0
                vehicle.vx, vehicle.vy = 12.0, 0.0
            elif i % 4 == 1:  # from east
                extra_offset = (i // 4) * spacing_offset
                vehicle.x, vehicle.y = 100.0 + extra_offset, 0.0
                vehicle.vx, vehicle.vy = -12.0, 0.0
            elif i % 4 == 2:  # from south
                extra_offset = (i // 4) * spacing_offset
                vehicle.x, vehicle.y = 0.0, -100.0 - extra_offset
                vehicle.vx, vehicle.vy = 0.0, 12.0
            else:  # from north
                extra_offset = (i // 4) * spacing_offset
                vehicle.x, vehicle.y = 0.0, 100.0 + extra_offset
                vehicle.vx, vehicle.vy = 0.0, -12.0
            
            vehicle.acceleration = 0.0
            vehicle.trip_completed = False
            vehicle.collision_occurred = False
            vehicle.total_travel_time = 0.0
            vehicle.total_waiting_time = 0.0
            vehicle.intersection_reservations = {}
        
        # Reset intersection
        self.intersection.reservations = {}
        self.intersection.reservation_slots = []
        self.intersection.current_occupant = None
        
        # Reset network
        self.network.message_queue = []
        
        # Reset metrics
        self.total_vehicles_completed = 0
        self.total_collisions = 0
        self.total_delay_time = 0.0
        
        # Return initial observations and infos
        observations = self._get_observations()
        infos = self._get_infos()
        return observations, infos
    
    def step(self, action_dict: MultiAgentDict) -> Tuple[MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict]:
        """Execute one step of the environment with gymnasium API."""
        self.episode_step += 1
        
        # Apply actions to all vehicles in action_dict (RLlib sends actions for all agents)
        for vehicle_id, action in action_dict.items():
            if vehicle_id in self.vehicles:
                vehicle = self.vehicles[vehicle_id]
                # Only apply actions to active agents, ignore for completed/crashed
                if not vehicle.trip_completed and not vehicle.collision_occurred:
                    vehicle.apply_action(action[0], self.dt)
        
        # Handle 6G communications
        self._handle_communications()
        
        # Update intersection
        vehicle_list = list(self.vehicles.values())
        self.intersection.update(vehicle_list, self.current_time)
        
        # Check for collisions
        self._check_collisions()
        
        # Check trip completions
        self._check_trip_completions()
        
        # Deliver network messages
        self.network.deliver_messages(self.current_time)
        
        # Update time
        self.current_time += self.dt
        
        # Get observations, rewards, terminated, truncated, and info flags
        observations = self._get_observations()
        rewards = self._get_rewards()
        terminateds = self._get_terminateds()
        truncateds = self._get_truncateds()
        infos = self._get_infos()
        
        return observations, rewards, terminateds, truncateds, infos
    
    def _handle_communications(self):
        """Handle V2V and V2I communications."""
        for vehicle in self.vehicles.values():
            if vehicle.trip_completed or vehicle.collision_occurred:
                continue
                
            # Check if vehicle needs intersection reservation
            dist_to_intersection = vehicle.distance_to_point(
                self.intersection.x, self.intersection.y
            )
            
            if (dist_to_intersection < 50.0 and 
                self.intersection.id not in vehicle.intersection_reservations):
                vehicle.request_intersection_reservation(self.intersection, self.current_time)
            
            # Broadcast vehicle state to nearby vehicles (V2V)
            for other_vehicle in self.vehicles.values():
                if (other_vehicle.id != vehicle.id and 
                    vehicle.distance_to(other_vehicle) < 100.0):
                    
                    state_message = {
                        'type': 'vehicle_state_broadcast',
                        'vehicle_state': {
                            'id': vehicle.id,
                            'position': (vehicle.x, vehicle.y),
                            'velocity': (vehicle.vx, vehicle.vy),
                            'acceleration': vehicle.acceleration
                        }
                    }
                    
                    self.network.send_message(
                        vehicle, other_vehicle, state_message, 
                        self.current_time, priority='eMBB'
                    )
    
    def _check_collisions(self):
        """Check for vehicle collisions."""
        vehicles_list = list(self.vehicles.values())
        for vehicle in vehicles_list:
            if vehicle.check_collision(vehicles_list):
                self.total_collisions += 1
                logger.warning(f"Collision detected involving vehicle {vehicle.id}")
    
    def _check_trip_completions(self):
        """Check if vehicles have completed their trips."""
        for vehicle in self.vehicles.values():
            if not vehicle.trip_completed:
                # Check if vehicle has passed through intersection and traveled beyond it
                # For vehicles coming from west/east, check x coordinate
                # For vehicles coming from north/south, check y coordinate
                passed_intersection = False
                
                if abs(vehicle.vx) > abs(vehicle.vy):  # Horizontal movement
                    if vehicle.vx > 0 and vehicle.x > 120.0:  # Moving east, passed far right
                        passed_intersection = True
                    elif vehicle.vx < 0 and vehicle.x < -120.0:  # Moving west, passed far left  
                        passed_intersection = True
                else:  # Vertical movement
                    if vehicle.vy > 0 and vehicle.y > 120.0:  # Moving north, passed far up
                        passed_intersection = True
                    elif vehicle.vy < 0 and vehicle.y < -120.0:  # Moving south, passed far down
                        passed_intersection = True
                
                if passed_intersection:
                    vehicle.trip_completed = True
                    self.total_vehicles_completed += 1
                    logger.info(f"Vehicle {vehicle.id} completed trip")
    
    def _get_observations(self) -> MultiAgentDict:
        """Get observations for ACTIVE agents only (RLlib new API requirement)."""
        observations = {}
        vehicle_list = list(self.vehicles.values())
        intersections = [self.intersection]
        
        for vehicle_id, vehicle in self.vehicles.items():
            # Only include observations for agents that are NOT done
            if not (vehicle.trip_completed or vehicle.collision_occurred):
                other_vehicles = [v for v in vehicle_list if v.id != vehicle_id]
                observations[vehicle_id] = vehicle.get_observation(intersections, other_vehicles)
        
        return observations
    
    def _get_rewards(self) -> MultiAgentDict:
        """Get rewards for ACTIVE agents only (RLlib new API requirement)."""
        rewards = {}
        vehicle_list = list(self.vehicles.values())
        intersections = [self.intersection]
        
        for vehicle_id, vehicle in self.vehicles.items():
            # Only include rewards for agents that are NOT done
            if not (vehicle.trip_completed or vehicle.collision_occurred):
                other_vehicles = [v for v in vehicle_list if v.id != vehicle_id]
                rewards[vehicle_id] = vehicle.get_reward(intersections, other_vehicles, self.dt)
        
        return rewards
    
    def _get_terminateds(self) -> MultiAgentDict:
        """Get terminated flags for all agents (episode truly ended)."""
        terminateds = {}
        
        for vehicle_id, vehicle in self.vehicles.items():
            # Agent is terminated if trip completed or collision occurred
            is_done = vehicle.trip_completed or vehicle.collision_occurred
            terminateds[vehicle_id] = is_done
        
        # Global terminated flag - check if all agents are done
        all_done = all(v.trip_completed or v.collision_occurred for v in self.vehicles.values())
        terminateds['__all__'] = all_done
        
        return terminateds
    
    def _get_truncateds(self) -> MultiAgentDict:
        """Get truncated flags for all agents (episode ended due to time limit)."""
        truncateds = {}
        
        # Episode is truncated if max steps reached but agents haven't naturally terminated
        time_limit_reached = self.episode_step >= self.max_episode_steps
        
        for vehicle_id, vehicle in self.vehicles.items():
            # Agent is truncated if time limit reached but not naturally terminated
            is_done = vehicle.trip_completed or vehicle.collision_occurred
            truncateds[vehicle_id] = time_limit_reached and not is_done
        
        # Global truncated flag
        any_active = any(not (v.trip_completed or v.collision_occurred) for v in self.vehicles.values())
        truncateds['__all__'] = time_limit_reached and any_active
        
        return truncateds
    
    def _get_infos(self) -> MultiAgentDict:
        """Get info dictionaries for ACTIVE agents only (RLlib new API requirement)."""
        infos = {}
        
        # Network metrics
        network_metrics = self.network.get_network_metrics()
        intersection_metrics = self.intersection.get_metrics()
        
        for vehicle_id, vehicle in self.vehicles.items():
            # Only include info for agents that are NOT done
            if not (vehicle.trip_completed or vehicle.collision_occurred):
                infos[vehicle_id] = {
                    'trip_completed': vehicle.trip_completed,
                    'collision_occurred': vehicle.collision_occurred,
                    'total_travel_time': vehicle.total_travel_time,
                    'total_waiting_time': vehicle.total_waiting_time,
                    'current_speed': vehicle.last_speed,
                    'distance_to_intersection': vehicle.distance_to_point(
                        self.intersection.x, self.intersection.y
                    ),
                    'has_reservation': self.intersection.id in vehicle.intersection_reservations
                }
        
        # Global info (always included)
        infos['__common__'] = {
            'episode_step': self.episode_step,
            'current_time': self.current_time,
            'total_vehicles_completed': self.total_vehicles_completed,
            'total_collisions': self.total_collisions,
            'network_reliability': network_metrics['reliability'],
            'intersection_throughput': intersection_metrics['vehicles_served'],
            'mmc_analysis': intersection_metrics.get('mmc_analysis', {}),
            'active_agents': len([v for v in self.vehicles.values() 
                                if not (v.trip_completed or v.collision_occurred)])
        }
        
        return infos
    
    def render(self, mode='human'):
        """Render the environment state including M/M/c analysis."""
        if mode == 'human':
            print(f"\n=== 6G Traffic Management System - Time: {self.current_time:.1f}s, Step: {self.episode_step} ===")
            print(f"Vehicles completed: {self.total_vehicles_completed}")
            print(f"Collisions: {self.total_collisions}")
            
            print("\nVehicle Status:")
            for vehicle_id, vehicle in self.vehicles.items():
                status = "ACTIVE"
                if vehicle.trip_completed:
                    status = "COMPLETED"
                elif vehicle.collision_occurred:
                    status = "COLLISION"
                
                has_reservation = self.intersection.id in vehicle.intersection_reservations
                print(f"  {vehicle_id}: pos=({vehicle.x:.1f}, {vehicle.y:.1f}), "
                      f"speed={vehicle.last_speed:.1f}, status={status}, "
                      f"reservation={'YES' if has_reservation else 'NO'}")
            
            # Network metrics
            network_metrics = self.network.get_network_metrics()
            print(f"\n6G Network: reliability={network_metrics['reliability']:.3f}, "
                  f"messages={network_metrics['total_messages']}")
            
            # M/M/c Queue Analysis (Paper Integration)
            intersection_metrics = self.intersection.get_metrics()
            mmc_analysis = intersection_metrics.get('mmc_analysis', {})
            
            if mmc_analysis:
                theoretical = mmc_analysis.get('theoretical', {})
                empirical = mmc_analysis.get('empirical', {})
                comparison = mmc_analysis.get('comparison', {})
                
                print(f"\n📊 M/M/c Queue Analysis (Paper Validation):")
                print(f"  Theoretical (λ={theoretical.get('lambda', 0):.3f}, μ={theoretical.get('mu', 0):.3f}, ρ={theoretical.get('rho', 0):.3f}):")
                print(f"    Avg Queue Length: {theoretical.get('avg_queue_length', 0):.2f}")
                print(f"    Avg Waiting Time: {theoretical.get('avg_waiting_time', 0):.2f}s")
                print(f"    System Stability: {theoretical.get('stability', 'unknown')}")
                
                print(f"  Empirical (RL Simulation):")
                print(f"    Avg Queue Length: {empirical.get('avg_queue_length', 0):.2f}")
                print(f"    Max Queue Length: {empirical.get('max_queue_length', 0)}")
                
                print(f"  Validation:")
                print(f"    Queue Length Difference: {comparison.get('queue_length_diff', 0):.2f}")
                print(f"    Utilization Ratio: {comparison.get('utilization_ratio', 0):.3f}")
                
                # Performance comparison note
                if comparison.get('system_stability') == 'stable':
                    print(f"  ✅ System is stable - RL performance can be compared to theory")
                else:
                    print(f"  ⚠️  System unstable - high traffic load, RL coordination critical")

def create_rllib_config(algorithm='PPO', num_vehicles=4, num_workers=2):
    """Create RLlib configuration for training."""
    
    # Environment configuration
    env_config = {
        'num_vehicles': num_vehicles,
        'max_episode_steps': 800,
        'dt': 0.1,
        'latency': 0.001,  # 1ms 6G latency
        'comm_range': 500.0
    }
    
    # Policy configuration - Updated for newer RLlib API
    def policy_mapping_fn(agent_id, episode, **kwargs):
        return "shared_policy"
    
    policies = {
        "shared_policy": (
            None,  # policy class (auto-detected)
            None,  # observation space (auto-detected)
            None,  # action space (auto-detected)
            {}     # policy config
        )
    }
    
    if algorithm == 'PPO':
        config = (
            PPOConfig()
            .environment(
                env=SixGTrafficEnv,
                env_config=env_config
            )
            .framework("torch")
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .env_runners(
                num_env_runners=num_workers,
                num_envs_per_env_runner=1,
                rollout_fragment_length=200,
            )
            .training(
                train_batch_size=2000,
                minibatch_size=128,
                num_epochs=10,
                lr=3e-4,
                gamma=0.99,
                lambda_=0.95,
                clip_param=0.2,
                vf_clip_param=10.0,
                entropy_coeff=0.01,
            )
            .debugging(
                log_level="INFO"
            )
            .resources(
                num_gpus=0,  # Set to 1 if GPU available
                num_cpus_per_worker=1,
            )
        )
    
    elif algorithm == 'SAC':
        config = (
            SACConfig()
            .environment(
                env=SixGTrafficEnv,
                env_config=env_config
            )
            .framework("torch")
            .multi_agent(
                policies=policies,
                policy_mapping_fn=policy_mapping_fn,
            )
            .env_runners(
                num_env_runners=num_workers,
                num_envs_per_env_runner=1,
            )
            .training(
                train_batch_size=1000,
                replay_buffer_config={
                    "type": "MultiAgentReplayBuffer",
                    "capacity": 100000,
                },
                lr=3e-4,
                gamma=0.99,
                tau=0.005,
                target_entropy="auto",
            )
            .debugging(
                log_level="INFO"
            )
            .resources(
                num_gpus=0,
                num_cpus_per_worker=1,
            )
        )
    
    return config

def main():
    """Main training function with optional visualization."""
    import argparse
    
    parser = argparse.ArgumentParser(description="6G Traffic Management with RLlib")
    parser.add_argument("--algorithm", type=str, default="PPO", 
                       choices=["PPO", "SAC"], help="RL algorithm")
    parser.add_argument("--num-vehicles", type=int, default=4,
                       help="Number of vehicles")
    parser.add_argument("--num-workers", type=int, default=2,
                       help="Number of rollout workers")
    parser.add_argument("--iterations", type=int, default=100,
                       help="Number of training iterations")
    parser.add_argument("--checkpoint-freq", type=int, default=10,
                       help="Checkpoint frequency")
    parser.add_argument("--visualize", action="store_true",
                       help="Enable real-time traffic flow visualization")
    parser.add_argument("--save-videos", action="store_true",
                       help="Save visualization frames")
    
    args = parser.parse_args()
    
    # Initialize visualization if requested
    visualizer = None
    data_logger = None
    visualization_thread = None
    
    if args.visualize and VISUALIZATION_AVAILABLE:
        print("🎯 Visualization requested - switching to single-worker mode for compatibility")
        visualizer, data_logger = create_training_visualizer()
        
        # Override worker settings for visualization compatibility
        args.num_workers = 0  # Single worker mode
        print("✅ Visualization initialized")
        
    elif args.visualize and not VISUALIZATION_AVAILABLE:
        print("⚠️  Visualization requested but not available. Install: pip install matplotlib seaborn")
        return

    # Initialize Ray
    ray.init(ignore_reinit_error=True)
    
    try:
        # Create monitored environment if visualization is enabled
        if visualizer and data_logger:
            # Create a monitored version that logs to visualizer
            class MonitoredSixGTrafficEnv(SixGTrafficEnv):
                def __init__(self, config):
                    super().__init__(config)
                    # Note: We can't store visualizer directly due to serialization
                    # Instead, we'll use a global reference approach
                    
                def step(self, action_dict):
                    obs, rewards, terminateds, truncateds, infos = super().step(action_dict)
                    
                    # Log to global visualizer every few steps
                    if hasattr(self, '_step_count'):
                        self._step_count += 1
                    else:
                        self._step_count = 0
                        
                    if self._step_count % 10 == 0:
                        try:
                            # Use global data_logger reference
                            global _global_data_logger
                            if '_global_data_logger' in globals() and _global_data_logger:
                                additional_info = {
                                    'episode_returns': sum(rewards.values()) if rewards else 0,
                                    'step_count': self._step_count
                                }
                                _global_data_logger.log_environment_state(self, additional_info)
                        except Exception as e:
                            pass  # Silently ignore visualization errors
                    
                    return obs, rewards, terminateds, truncateds, infos
                    
                def reset(self, **kwargs):
                    obs, infos = super().reset(**kwargs)
                    self._step_count = 0
                    return obs, infos
            
            # Set global reference for the environment to use
            global _global_data_logger
            _global_data_logger = data_logger
            
            env_name = MonitoredSixGTrafficEnv
        else:
            env_name = SixGTrafficEnv
        
        # Create configuration
        config = create_rllib_config(
            algorithm=args.algorithm,
            num_vehicles=args.num_vehicles,
            num_workers=args.num_workers
        )
        
        print(f"🚀 Starting 6G Traffic Management Training")
        print(f"📊 Algorithm: {args.algorithm}")
        print(f"🚗 Vehicles: {args.num_vehicles}")
        print(f"🔄 Workers: {args.num_workers}")
        print(f"📈 Iterations: {args.iterations}")
        print(f"📺 Visualization: {'ON' if args.visualize and VISUALIZATION_AVAILABLE else 'OFF'}")
        
        # Note: We'll start visualization after training due to GUI threading constraints on macOS
        if visualizer:
            print("🎬 Visualization will start after training completes")
        
        # Run training
        import os
        storage_path = os.path.abspath("./ray_results")
        
        results = tune.run(
            args.algorithm,
            config=config.to_dict(),
            stop={"training_iteration": args.iterations},
            checkpoint_freq=args.checkpoint_freq,
            storage_path=storage_path,
            name=f"6g_traffic_{args.algorithm}_{args.num_vehicles}v{'_vis' if args.visualize else ''}",
            verbose=1
        )
        
        print("\n✅ Training completed successfully!")
        print(f"📁 Results saved to: {storage_path}")
        
        # Save final visualization frame if requested
        if visualizer and args.save_videos:
            final_frame_path = f"{storage_path}/6g_traffic_{args.algorithm}_{args.num_vehicles}v_final_visualization.png"
            try:
                visualizer.save_frame(final_frame_path)
                print(f"🖼️  Final visualization saved: {final_frame_path}")
            except Exception as e:
                print(f"⚠️  Could not save visualization frame: {e}")
        
        if visualizer:
            print("\n🎬 Starting post-training visualization...")
            try:
                # Load training data and show visualization
                visualizer.start_visualization(interval=100)
            except Exception as e:
                print(f"⚠️ Visualization error: {e}")
                print("📊 Training data logged for later analysis")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup visualization
        if visualizer:
            try:
                visualizer.stop_visualization()
                print("🛑 Visualization stopped")
            except:
                pass
        
        ray.shutdown()

if __name__ == "__main__":
    # Test environment first
    print("🧪 Testing 6G Traffic Environment...")
    
    env = SixGTrafficEnv({'num_vehicles': 2, 'max_episode_steps': 100})
    obs, infos = env.reset()
    
    print(f"✅ Environment created successfully")
    print(f"🚗 Agents: {env.agents}")
    print(f"📊 Observation space: {env.observation_space}")
    print(f"🎮 Action space: {env.action_space}")
    
    # Test a few steps
    for step in range(5):
        actions = {agent_id: env.action_space.spaces[agent_id].sample() for agent_id in obs.keys()}
        obs, rewards, terminateds, truncateds, infos = env.step(actions)
        env.render()
        
        if terminateds.get('__all__', False) or truncateds.get('__all__', False):
            break
    
    print("\n✅ Environment test completed successfully!")
    print("\n🚀 Starting RLlib training...")
    
    # Run full training
    main() 