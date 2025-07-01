"""
Multi-Agent 6G-Enabled Autonomous Traffic Management Environment

This module implements a proper multi-agent reinforcement learning environment
for 6G-enabled autonomous vehicle traffic management with signal-free intersections.
Each vehicle is an independent agent with local observations and individual rewards.
"""

import math
import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque, defaultdict
import gymnasium as gym
from gymnasium import spaces
from ray.rllib.env.multi_agent_env import MultiAgentEnv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("MARL_Traffic")

# Constants for 6G network simulation
DEFAULT_6G_LATENCY = 0.001  # 1 ms latency
DEFAULT_6G_RANGE = 1000.0   # 1 km communication range
DEFAULT_6G_BANDWIDTH = 1e9  # 1 Gbps
DEFAULT_6G_SNR = 10.0       # Linear signal-to-noise ratio
DEFAULT_THz_PATH_LOSS_DB = 0.0  # Path loss in decibels for THz channel

class SixGNetwork:
    """
    Advanced 6G network simulation with ultra-low latency and high reliability.
    Supports V2V and V2I communication with network slicing.
    """
    
    def __init__(self, latency=DEFAULT_6G_LATENCY, comm_range=DEFAULT_6G_RANGE,
                 bandwidth=DEFAULT_6G_BANDWIDTH, snr=DEFAULT_6G_SNR,
                 path_loss_db=DEFAULT_THz_PATH_LOSS_DB):
        self.latency = latency
        self.range = comm_range
        self.bandwidth = bandwidth
        self.snr = snr
        self.path_loss_db = path_loss_db

        self.message_queue = []
        self.network_load = 0
        self.total_messages = 0
        self.dropped_messages = 0

    def calculate_capacity(self) -> float:
        """Calculate channel capacity accounting for path loss."""
        effective_snr = self.snr * (10 ** (-self.path_loss_db / 10))
        return self.bandwidth * math.log2(1 + effective_snr)
        
    def send_message(self, sender, receiver, message, priority="normal", current_time=0):
        """Send message with network slicing support for different priorities."""
        # Calculate distance if both sender and receiver have positions
        if hasattr(sender, 'x') and hasattr(receiver, 'x'):
            dx = sender.x - receiver.x
            dy = sender.y - receiver.y
            dist = math.hypot(dx, dy)
            
            if dist > self.range:
                self.dropped_messages += 1
                logger.debug(f"Message dropped: out of range ({dist:.1f}m > {self.range}m)")
                return False
        
        # Adjust latency based on priority (network slicing)
        if priority == "emergency":
            actual_latency = self.latency * 0.5  # URLLC slice
        elif priority == "safety":
            actual_latency = self.latency * 0.7
        else:
            actual_latency = self.latency
            
        # Transmission delay based on capacity
        message_size_bits = len(str(message).encode("utf-8")) * 8
        capacity = self.calculate_capacity()
        transmission_delay = message_size_bits / max(capacity, 1e-9)

        deliver_time = current_time + actual_latency + transmission_delay
        self.message_queue.append((deliver_time, receiver, message, sender))
        self.total_messages += 1
        return True
    
    def deliver_messages(self, current_time):
        """Deliver all due messages."""
        delivered = []
        remaining = []
        
        for msg_tuple in self.message_queue:
            deliver_time, receiver, message, sender = msg_tuple
            if deliver_time <= current_time:
                try:
                    receiver.receive_message(message, sender)
                    delivered.append(msg_tuple)
                except Exception as e:
                    logger.warning(f"Failed to deliver message: {e}")
            else:
                remaining.append(msg_tuple)
        
        self.message_queue = remaining
        return len(delivered)
    
    def get_network_stats(self):
        """Get network performance statistics."""
        if self.total_messages > 0:
            reliability = (self.total_messages - self.dropped_messages) / self.total_messages
        else:
            reliability = 1.0
        
        return {
            "total_messages": self.total_messages,
            "dropped_messages": self.dropped_messages,
            "reliability": reliability,
            "queue_length": len(self.message_queue),
            "average_latency": self.latency,
            "capacity_bps": self.calculate_capacity()
        }

class Vehicle:
    """
    Autonomous vehicle agent with 6G communication capabilities.
    Each vehicle is an independent RL agent.
    """
    
    def __init__(self, vid: int, x: float, y: float, target_x: float, target_y: float, 
                 max_speed: float = 15.0, approach_direction: str = "east"):
        self.id = vid
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.approach_direction = approach_direction
        
        # Physical properties
        self.length = 5.0
        self.width = 2.0
        self.max_speed = max_speed
        
        # Set initial velocity based on approach direction
        if approach_direction == "east":
            self.vx = max_speed
            self.vy = 0.0
        elif approach_direction == "west":
            self.vx = -max_speed  # Negative for westward movement
            self.vy = 0.0
        elif approach_direction == "north":
            self.vx = 0.0
            self.vy = max_speed
        elif approach_direction == "south":
            self.vx = 0.0
            self.vy = -max_speed  # Negative for southward movement
        else:
            self.vx = 0.0
            self.vy = 0.0
            
        self.acceleration = 0.0
        self.max_acceleration = 3.0
        self.max_deceleration = -5.0
        
        # Communication and coordination
        self.network = None
        self.nearby_vehicles = []
        self.intersection_reservations = {}
        self.has_reservation = False
        self.reservation_time = None
        self.reservation_intersection = None
        
        # State tracking
        self.total_travel_time = 0.0
        self.waiting_time = 0.0
        self.energy_consumed = 0.0
        self.completed_trip = False
        self.collision_occurred = False
        
        # Agent memory for better decision making
        self.state_history = deque(maxlen=10)
        self.action_history = deque(maxlen=5)
        
    def set_network(self, network: SixGNetwork):
        """Connect vehicle to 6G network."""
        self.network = network
        
    def get_local_observation(self, intersections, other_vehicles, max_vehicles=5):
        """
        Get local observation for this vehicle agent.
        Includes own state, nearby vehicles, and intersection status.
        """
        obs = []
        
        # Own state (normalized)
        obs.extend([
            self.x / 100.0,  # Normalize position
            self.y / 100.0,
            self.vx / self.max_speed,  # Normalize velocity
            self.vy / self.max_speed,
            self.acceleration / self.max_acceleration,
            1.0 if self.has_reservation else 0.0,
            (self.target_x - self.x) / 100.0,  # Distance to target
            (self.target_y - self.y) / 100.0
        ])
        
        # Nearest intersection info
        nearest_intersection = self._find_nearest_intersection(intersections)
        if nearest_intersection:
            dist_to_intersection = math.hypot(
                nearest_intersection.x - self.x,
                nearest_intersection.y - self.y
            )
            obs.extend([
                (nearest_intersection.x - self.x) / 100.0,
                (nearest_intersection.y - self.y) / 100.0,
                dist_to_intersection / 100.0,
                len(nearest_intersection.reservations) / 10.0,  # Normalize queue length
                1.0 if nearest_intersection.current_vehicle else 0.0
            ])
        else:
            obs.extend([0.0, 0.0, 0.0, 0.0, 0.0])
        
        # Nearby vehicles (up to max_vehicles)
        nearby = self._get_nearby_vehicles(other_vehicles, max_vehicles)
        for i in range(max_vehicles):
            if i < len(nearby):
                other = nearby[i]
                relative_x = (other.x - self.x) / 100.0
                relative_y = (other.y - self.y) / 100.0
                relative_vx = (other.vx - self.vx) / self.max_speed
                relative_vy = (other.vy - self.vy) / self.max_speed
                obs.extend([relative_x, relative_y, relative_vx, relative_vy])
            else:
                obs.extend([0.0, 0.0, 0.0, 0.0])  # Padding for missing vehicles
        
        return np.array(obs, dtype=np.float32)
    
    def _find_nearest_intersection(self, intersections):
        """Find the nearest intersection ahead in the vehicle's path."""
        nearest = None
        min_distance = float('inf')
        
        for intersection in intersections:
            # Only consider intersections ahead in the direction of travel
            if self.approach_direction == "east" and intersection.x > self.x:
                distance = abs(intersection.x - self.x) + abs(intersection.y - self.y)
            elif self.approach_direction == "west" and intersection.x < self.x:
                distance = abs(intersection.x - self.x) + abs(intersection.y - self.y)
            elif self.approach_direction == "north" and intersection.y > self.y:
                distance = abs(intersection.x - self.x) + abs(intersection.y - self.y)
            elif self.approach_direction == "south" and intersection.y < self.y:
                distance = abs(intersection.x - self.x) + abs(intersection.y - self.y)
            else:
                continue
                
            if distance < min_distance:
                min_distance = distance
                nearest = intersection
                
        return nearest
    
    def _get_nearby_vehicles(self, other_vehicles, max_count):
        """Get nearby vehicles sorted by distance."""
        distances = []
        for other in other_vehicles:
            if other.id != self.id:
                dist = math.hypot(other.x - self.x, other.y - self.y)
                if dist < 50.0:  # Only consider vehicles within 50m
                    distances.append((dist, other))
        
        distances.sort(key=lambda x: x[0])
        return [vehicle for _, vehicle in distances[:max_count]]
    
    def apply_action(self, action: float):
        """Apply acceleration action."""
        self.acceleration = np.clip(action, self.max_deceleration, self.max_acceleration)
        self.action_history.append(action)
    
    def update_physics(self, dt: float):
        """Update vehicle physics."""
        # Update velocity based on approach direction
        if self.approach_direction == "east":
            self.vx += self.acceleration * dt
            self.vx = np.clip(self.vx, 0.0, self.max_speed)  # Can't go backwards
        elif self.approach_direction == "west":
            self.vx += self.acceleration * dt
            self.vx = np.clip(self.vx, -self.max_speed, 0.0)  # Can't go forwards (positive)
        elif self.approach_direction == "north":
            self.vy += self.acceleration * dt
            self.vy = np.clip(self.vy, 0.0, self.max_speed)  # Can't go backwards
        elif self.approach_direction == "south":
            self.vy += self.acceleration * dt
            self.vy = np.clip(self.vy, -self.max_speed, 0.0)  # Can't go forwards (positive)
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Update metrics
        self.total_travel_time += dt
        speed = math.hypot(self.vx, self.vy)
        if speed < 1.0:  # Consider as waiting if moving very slowly
            self.waiting_time += dt
        
        # Energy consumption (simplified model)
        self.energy_consumed += abs(self.acceleration) * dt + speed * dt * 0.1
        
        # Check if reached target - make completion easier
        dist_to_target = math.hypot(self.target_x - self.x, self.target_y - self.y)
        if dist_to_target < 10.0:  # Increased from 5.0 to make completion easier
            self.completed_trip = True
    
    def request_intersection_reservation(self, intersection, current_time):
        """Request reservation at intersection."""
        if not self.network or self.has_reservation:
            return False
            
        # Calculate ETA to intersection
        dist = math.hypot(intersection.x - self.x, intersection.y - self.y)
        speed = math.hypot(self.vx, self.vy)
        
        if speed < 0.1:  # Avoid division by zero
            eta = current_time + 10.0
        else:
            eta = current_time + dist / speed
        
        message = {
            "type": "reservation_request",
            "eta": eta,
            "vehicle_id": self.id,
            "approach_direction": self.approach_direction,
            "priority": "normal"
        }
        
        return self.network.send_message(self, intersection, message, "normal", current_time)
    
    def receive_message(self, message: Dict, sender):
        """Handle incoming network messages."""
        if message.get("type") == "reservation_confirmation":
            self.has_reservation = True
            self.reservation_time = message.get("slot")
            self.reservation_intersection = sender.id
            logger.debug(f"Vehicle {self.id} received reservation for t={self.reservation_time:.2f}")
            
        elif message.get("type") == "reservation_denied":
            self.has_reservation = False
            logger.debug(f"Vehicle {self.id} reservation denied")
            
        elif message.get("type") == "vehicle_state_broadcast":
            # Update nearby vehicles list
            state = message.get("state")
            if state:
                self.nearby_vehicles = state.get("nearby_vehicles", [])
    
    def get_individual_reward(self, intersections, other_vehicles, dt):
        """Calculate individual reward for this vehicle."""
        reward = 0.0
        
        # Progress reward (moving towards target)
        dist_to_target = math.hypot(self.target_x - self.x, self.target_y - self.y)
        prev_dist = getattr(self, '_prev_dist_to_target', dist_to_target)
        progress = prev_dist - dist_to_target
        reward += progress * 10.0  # Reward progress towards target
        self._prev_dist_to_target = dist_to_target
        
        # Speed reward (encourage maintaining reasonable speed)
        speed = math.hypot(self.vx, self.vy)
        optimal_speed = self.max_speed * 0.8
        speed_reward = -abs(speed - optimal_speed) * 0.1
        reward += speed_reward
        
        # Efficiency reward (penalize excessive acceleration)
        acceleration_penalty = -abs(self.acceleration) * 0.05
        reward += acceleration_penalty
        
        # Safety reward (avoid collisions)
        for other in other_vehicles:
            if other.id != self.id:
                dist = math.hypot(other.x - self.x, other.y - self.y)
                if dist < 3.0:  # Very close vehicles
                    reward -= 50.0  # Heavy penalty for near collision
                elif dist < 8.0:  # Close vehicles
                    reward -= 5.0   # Moderate penalty
        
        # Intersection cooperation reward
        nearest_intersection = self._find_nearest_intersection(intersections)
        if nearest_intersection:
            dist_to_intersection = math.hypot(
                nearest_intersection.x - self.x,
                nearest_intersection.y - self.y
            )
            
            # Reward for having reservation when close to intersection
            if dist_to_intersection < 30.0 and self.has_reservation:
                reward += 2.0
            elif dist_to_intersection < 15.0 and not self.has_reservation:
                reward -= 3.0  # Penalty for not having reservation when close
        
        # Completion reward
        if self.completed_trip:
            # Bonus based on efficiency
            time_bonus = max(0, 100 - self.total_travel_time)
            energy_bonus = max(0, 50 - self.energy_consumed / 10)
            reward += time_bonus + energy_bonus
        
        # Penalty for excessive waiting
        if self.waiting_time > 0:
            reward -= self.waiting_time * 2.0
        
        return reward

class IntersectionManager:
    """
    Intelligent intersection manager for signal-free coordination.
    Uses 6G communication to coordinate vehicle crossings.
    """
    
    def __init__(self, iid: str, x: float, y: float, conflict_radius: float = 12.0):
        self.id = iid
        self.x = x
        self.y = y
        self.conflict_radius = conflict_radius
        self.reservations = []  # (time_slot, vehicle_id, direction)
        self.current_vehicle = None
        self.vehicles_served = 0
        self.total_wait_time = 0.0
        self.rejection_count = 0
        self.network = None
        
    def set_network(self, network: SixGNetwork):
        """Connect intersection to 6G network."""
        self.network = network
        
    def receive_message(self, message: Dict, sender):
        """Handle reservation requests from vehicles."""
        if message.get("type") != "reservation_request":
            return
            
        vehicle_id = message.get("vehicle_id")
        eta = message.get("eta")
        direction = message.get("approach_direction")
        priority = message.get("priority", "normal")
        
        # Check for conflicts
        if self._check_reservation_conflict(eta, direction):
            # Send denial
            if self.network:
                denial_msg = {
                    "type": "reservation_denied",
                    "reason": "time_conflict",
                    "suggested_eta": eta + 2.0
                }
                self.network.send_message(self, sender, denial_msg, "normal")
                self.rejection_count += 1
                logger.debug(f"Intersection {self.id}: denied reservation for vehicle {vehicle_id}")
        else:
            # Grant reservation
            self.reservations.append((eta, vehicle_id, direction))
            self.reservations.sort(key=lambda x: x[0])  # Sort by time
            
            if self.network:
                confirmation_msg = {
                    "type": "reservation_confirmation",
                    "slot": eta,
                    "intersection_id": self.id
                }
                self.network.send_message(self, sender, confirmation_msg, "safety")
                logger.debug(f"Intersection {self.id}: granted reservation to vehicle {vehicle_id} at t={eta:.2f}")
    
    def _check_reservation_conflict(self, requested_eta: float, direction: str) -> bool:
        """Check if requested time conflicts with existing reservations."""
        buffer_time = 1.5  # Minimum time between vehicles (seconds)
        
        for eta, _, existing_direction in self.reservations:
            time_diff = abs(eta - requested_eta)
            
            # Check for time conflicts
            if time_diff < buffer_time:
                # Additional check for crossing conflicts
                if self._directions_conflict(direction, existing_direction):
                    return True
                elif time_diff < buffer_time * 0.5:  # Same direction needs smaller buffer
                    return True
        
        return False
    
    def _directions_conflict(self, dir1: str, dir2: str) -> bool:
        """Check if two approach directions conflict."""
        crossing_conflicts = {
            ("east", "north"), ("east", "south"),
            ("west", "north"), ("west", "south"),
            ("north", "east"), ("north", "west"),
            ("south", "east"), ("south", "west")
        }
        return (dir1, dir2) in crossing_conflicts or (dir2, dir1) in crossing_conflicts
    
    def update(self, vehicles: List[Vehicle], current_time: float):
        """Update intersection state."""
        # Remove old reservations
        self.reservations = [(t, v, d) for t, v, d in self.reservations 
                           if t > current_time - 5.0]
        
        # Check which vehicles are in intersection
        vehicles_in_intersection = []
        for vehicle in vehicles:
            dist = math.hypot(vehicle.x - self.x, vehicle.y - self.y)
            if dist < self.conflict_radius:
                vehicles_in_intersection.append(vehicle)
        
        # Update current vehicle
        if vehicles_in_intersection:
            self.current_vehicle = vehicles_in_intersection[0]
        else:
            if self.current_vehicle:
                self.vehicles_served += 1
            self.current_vehicle = None
    
    def get_intersection_metrics(self):
        """Get intersection performance metrics."""
        return {
            "vehicles_served": self.vehicles_served,
            "active_reservations": len(self.reservations),
            "rejection_count": self.rejection_count,
            "efficiency": self.vehicles_served / max(1, self.vehicles_served + self.rejection_count)
        }

class MultiAgentTrafficEnv(MultiAgentEnv):
    """
    Multi-Agent Traffic Environment for 6G-enabled autonomous vehicles.
    Each vehicle is an independent agent with local observations.
    """
    
    def __init__(self, config: Dict = None):
        super().__init__()
        
        # Configuration
        config = config or {}
        self.num_vehicles = config.get("num_vehicles", 4)
        self.max_episode_steps = config.get("max_episode_steps", 1000)
        self.dt = config.get("dt", 0.1)
        self.intersection_positions = config.get("intersection_positions", [(0.0, 0.0)])
        self.vehicle_spawn_distance = config.get("vehicle_spawn_distance", 80.0)
        
        # Environment entities
        self.network = SixGNetwork(
            latency=config.get("6g_latency", DEFAULT_6G_LATENCY),
            comm_range=config.get("6g_range", DEFAULT_6G_RANGE),
            bandwidth=config.get("6g_bandwidth", DEFAULT_6G_BANDWIDTH),
            snr=config.get("6g_snr", DEFAULT_6G_SNR),
            path_loss_db=config.get("terahertz_path_loss_db", DEFAULT_THz_PATH_LOSS_DB),
        )
        self.intersections = []
        self.vehicles = {}
        self.time = 0.0
        self.episode_steps = 0
        
        # Initialize intersections
        for i, pos in enumerate(self.intersection_positions):
            intersection = IntersectionManager(f"I{i}", pos[0], pos[1])
            intersection.set_network(self.network)
            self.intersections.append(intersection)
        
        # Agent IDs
        self.agent_ids = set()
        
        # PettingZoo-style agent tracking for RLlib
        self.agents = [f"vehicle_{i}" for i in range(self.num_vehicles)]
        self.possible_agents = [f"vehicle_{i}" for i in range(self.num_vehicles)]
        
        # Observation and action spaces
        # Observation: own state + nearest intersection + nearby vehicles
        obs_dim = 8 + 5 + (5 * 4)  # vehicle state + intersection + 5 nearby vehicles * 4 features
        single_obs_space = spaces.Box(
            low=-10.0, high=10.0, shape=(obs_dim,), dtype=np.float32
        )
        
        # Action: continuous acceleration
        single_action_space = spaces.Box(
            low=-5.0, high=3.0, shape=(1,), dtype=np.float32
        )
        
        # Multi-agent spaces
        self.observation_space = {agent_id: single_obs_space for agent_id in self.possible_agents}
        self.action_space = {agent_id: single_action_space for agent_id in self.possible_agents}
        
        # Metrics tracking
        self.episode_metrics = defaultdict(list)
        self.global_metrics = defaultdict(float)
        
    def reset(self, *, seed=None, options=None):
        """Reset environment to initial state."""
        self.time = 0.0
        self.episode_steps = 0
        self.vehicles = {}
        self.agent_ids = set()
        
        # Clear intersection reservations
        for intersection in self.intersections:
            intersection.reservations = []
            intersection.current_vehicle = None
            intersection.vehicles_served = 0
            intersection.rejection_count = 0
        
        # Clear network
        self.network.message_queue = []
        self.network.total_messages = 0
        self.network.dropped_messages = 0
        
        # Spawn vehicles
        spawn_configs = [
            {"direction": "east", "start_x": -self.vehicle_spawn_distance, "start_y": 0, 
             "target_x": self.vehicle_spawn_distance, "target_y": 0},
            {"direction": "west", "start_x": self.vehicle_spawn_distance, "start_y": 0, 
             "target_x": -self.vehicle_spawn_distance, "target_y": 0},
            {"direction": "north", "start_x": 0, "start_y": -self.vehicle_spawn_distance, 
             "target_x": 0, "target_y": self.vehicle_spawn_distance},
            {"direction": "south", "start_x": 0, "start_y": self.vehicle_spawn_distance, 
             "target_x": 0, "target_y": -self.vehicle_spawn_distance}
        ]
        
        for i in range(self.num_vehicles):
            config = spawn_configs[i % len(spawn_configs)]
            
            # Add some randomness to spawn positions
            noise_x = random.uniform(-5, 5)
            noise_y = random.uniform(-5, 5)
            
            vehicle = Vehicle(
                vid=i,
                x=config["start_x"] + noise_x,
                y=config["start_y"] + noise_y,
                target_x=config["target_x"],
                target_y=config["target_y"],
                approach_direction=config["direction"]
            )
            vehicle.set_network(self.network)
            
            agent_id = f"vehicle_{i}"
            self.vehicles[agent_id] = vehicle
            self.agent_ids.add(agent_id)
        
        # Update agents list for RLlib
        self.agents = list(self.agent_ids)
        
        # Reset metrics
        self.episode_metrics = defaultdict(list)
        self.global_metrics = defaultdict(float)
        
        # Get initial observations
        observations = {}
        for agent_id, vehicle in self.vehicles.items():
            observations[agent_id] = vehicle.get_local_observation(
                self.intersections, list(self.vehicles.values())
            )
        
        infos = {agent_id: {} for agent_id in self.agent_ids}
        
        return observations, infos
    
    def step(self, actions: Dict[str, np.ndarray]):
        """Execute one environment step."""
        self.episode_steps += 1
        
        # Apply actions
        for agent_id, action in actions.items():
            if agent_id in self.vehicles:
                self.vehicles[agent_id].apply_action(action[0])
        
        # Vehicle-intersection communication
        for agent_id, vehicle in self.vehicles.items():
            if not vehicle.completed_trip:
                nearest_intersection = vehicle._find_nearest_intersection(self.intersections)
                if nearest_intersection:
                    dist = math.hypot(nearest_intersection.x - vehicle.x, 
                                    nearest_intersection.y - vehicle.y)
                    
                    # Request reservation when approaching intersection
                    if dist < 40.0 and not vehicle.has_reservation:
                        vehicle.request_intersection_reservation(nearest_intersection, self.time)
        
        # Deliver network messages
        self.network.deliver_messages(self.time + self.dt)
        
        # Update intersections
        for intersection in self.intersections:
            intersection.update(list(self.vehicles.values()), self.time)
        
        # Update vehicle physics and safety checks
        for agent_id, vehicle in self.vehicles.items():
            if not vehicle.completed_trip:
                # Safety check before movement
                self._apply_safety_constraints(vehicle)
                
                # Update physics
                vehicle.update_physics(self.dt)
        
        # Update time
        self.time += self.dt
        
        # Get observations, rewards, and dones
        observations = {}
        rewards = {}
        dones = {}
        infos = {}
        
        for agent_id, vehicle in self.vehicles.items():
            observations[agent_id] = vehicle.get_local_observation(
                self.intersections, list(self.vehicles.values())
            )
            
            rewards[agent_id] = vehicle.get_individual_reward(
                self.intersections, list(self.vehicles.values()), self.dt
            )
            
            # Check if agent is done
            dones[agent_id] = (
                vehicle.completed_trip or 
                self.episode_steps >= self.max_episode_steps or
                vehicle.collision_occurred
            )
            
            infos[agent_id] = {
                "total_travel_time": vehicle.total_travel_time,
                "waiting_time": vehicle.waiting_time,
                "energy_consumed": vehicle.energy_consumed,
                "completed_trip": vehicle.completed_trip,
                "collision": vehicle.collision_occurred
            }
        
        # Global termination condition - more predictable for RLlib
        episode_timeout = self.episode_steps >= self.max_episode_steps
        agents_completed = sum(1 for v in self.vehicles.values() if v.completed_trip)
        completion_rate = agents_completed / len(self.vehicles)
        
        # End episode if timeout, high completion rate, or all agents done
        all_done = episode_timeout or completion_rate >= 0.75 or all(dones.values())
        
        if all_done:
            # Mark all remaining agents as done for consistent episode ending
            for agent_id in self.agent_ids:
                if agent_id not in dones:
                    dones[agent_id] = True
            dones["__all__"] = True
            infos["__common__"] = self._get_global_info()
        else:
            dones["__all__"] = False
        
        return observations, rewards, dones, dones, infos
    
    def _apply_safety_constraints(self, vehicle: Vehicle):
        """Apply safety constraints to prevent collisions."""
        for other_vehicle in self.vehicles.values():
            if other_vehicle.id != vehicle.id and not other_vehicle.completed_trip:
                # Calculate distance and relative velocity
                dx = other_vehicle.x - vehicle.x
                dy = other_vehicle.y - vehicle.y
                distance = math.hypot(dx, dy)
                
                # If vehicles are too close, apply emergency braking
                if distance < 8.0:  # Safety threshold
                    # Calculate if they're on collision course
                    relative_vx = vehicle.vx - other_vehicle.vx
                    relative_vy = vehicle.vy - other_vehicle.vy
                    
                    # Project relative position and velocity
                    if distance > 0:
                        closing_speed = (dx * relative_vx + dy * relative_vy) / distance
                        
                        if closing_speed > 0:  # Vehicles are approaching each other
                            # Apply emergency braking
                            vehicle.acceleration = min(vehicle.acceleration, -3.0)
                            logger.debug(f"Emergency braking applied to vehicle {vehicle.id}")
                            
                            # Flag potential collision
                            if distance < 3.0:
                                vehicle.collision_occurred = True
                                other_vehicle.collision_occurred = True
    
    def _get_global_info(self):
        """Get global environment information."""
        total_travel_time = sum(v.total_travel_time for v in self.vehicles.values())
        total_waiting_time = sum(v.waiting_time for v in self.vehicles.values())
        total_energy = sum(v.energy_consumed for v in self.vehicles.values())
        completed_trips = sum(1 for v in self.vehicles.values() if v.completed_trip)
        collisions = sum(1 for v in self.vehicles.values() if v.collision_occurred)
        
        network_stats = self.network.get_network_stats()
        
        intersection_metrics = {}
        for i, intersection in enumerate(self.intersections):
            intersection_metrics[f"intersection_{i}"] = intersection.get_intersection_metrics()
        
        return {
            "total_travel_time": total_travel_time,
            "average_travel_time": total_travel_time / max(1, len(self.vehicles)),
            "total_waiting_time": total_waiting_time,
            "average_waiting_time": total_waiting_time / max(1, len(self.vehicles)),
            "total_energy_consumed": total_energy,
            "average_energy_consumed": total_energy / max(1, len(self.vehicles)),
            "completion_rate": completed_trips / len(self.vehicles),
            "collision_rate": collisions / len(self.vehicles),
            "network_reliability": network_stats["reliability"],
            "network_total_messages": network_stats["total_messages"],
            "intersection_metrics": intersection_metrics,
            "episode_steps": self.episode_steps,
            "simulation_time": self.time
        }
    
    def render(self, mode="human"):
        """Render the environment (placeholder for visualization)."""
        if mode == "human":
            print(f"\n=== Time: {self.time:.1f}s, Step: {self.episode_steps} ===")
            for agent_id, vehicle in self.vehicles.items():
                status = []
                if vehicle.completed_trip:
                    status.append("COMPLETED")
                if vehicle.has_reservation:
                    status.append(f"RESERVED@{vehicle.reservation_time:.1f}")
                if vehicle.collision_occurred:
                    status.append("COLLISION")
                
                status_str = " [" + ", ".join(status) + "]" if status else ""
                print(f"{agent_id}: pos=({vehicle.x:.1f}, {vehicle.y:.1f}), "
                      f"speed={math.hypot(vehicle.vx, vehicle.vy):.1f}m/s{status_str}")
        
        return None
    
    def get_agent_ids(self):
        """Return current agent IDs."""
        return set(self.agents)
    
    def get_observation_space(self, agent_id):
        """Get observation space for a specific agent."""
        return self.observation_space[agent_id]
    
    def get_action_space(self, agent_id):
        """Get action space for a specific agent."""
        return self.action_space[agent_id]

# Test function
if __name__ == "__main__":
    # Simple test of the environment
    env = MultiAgentTrafficEnv({"num_vehicles": 4, "max_episode_steps": 100})
    
    obs, info = env.reset()
    print("Environment created successfully!")
    print(f"Number of agents: {len(obs)}")
    print(f"Observation space shape (per agent): {list(env.observation_space.values())[0].shape}")
    print(f"Action space shape (per agent): {list(env.action_space.values())[0].shape}")
    
    # Run a few random steps
    for step in range(5):
        actions = {}
        for agent_id in obs.keys():
            actions[agent_id] = env.action_space[agent_id].sample()
        
        obs, rewards, dones, truncated, infos = env.step(actions)
        env.render()
        
        if dones.get("__all__", False):
            break
    
    print("\nTest completed successfully!") 