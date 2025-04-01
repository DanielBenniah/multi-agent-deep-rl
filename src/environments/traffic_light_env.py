"""
Traditional Traffic Light Controlled Environment with Human Drivers

This module implements a traditional traffic system with traffic lights and
realistic human driving behavior for performance comparison against the
6G-enabled autonomous vehicle system.

Key Features:
- Realistic traffic light timing and phases
- Human driver behavior modeling (speed up, slow down, queuing)
- Performance metrics for comparison with autonomous system
- Queue formation and dissipation modeling
"""

import math
import random
import logging
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
from collections import deque
import gymnasium as gym
from gymnasium import spaces
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrafficLight_Env")

class TrafficLightState(Enum):
    """Traffic light states with realistic timing."""
    RED = "red"
    YELLOW = "yellow" 
    GREEN = "green"

class TrafficLight:
    """
    Realistic traffic light controller with configurable timing.
    Supports multi-direction intersection control.
    """
    
    def __init__(self, intersection_id: str, x: float, y: float):
        self.id = intersection_id
        self.x = x
        self.y = y
        
        # Traffic light timing (seconds) - based on real-world values
        self.green_time = 30.0      # Green light duration
        self.yellow_time = 4.0      # Yellow light duration (3-5 seconds typical)
        self.red_time = 34.0        # Red light duration (green + yellow for other direction)
        self.all_red_time = 2.0     # All-red clearance time
        
        # Current state for each direction
        self.states = {
            "east_west": TrafficLightState.GREEN,
            "north_south": TrafficLightState.RED
        }
        
        # Timing tracking
        self.state_start_time = 0.0
        self.cycle_time = 0.0
        self.total_cycle_time = (self.green_time + self.yellow_time) * 2 + self.all_red_time * 2
        
        # Performance metrics
        self.vehicles_passed = {"east_west": 0, "north_south": 0}
        self.total_waiting_time = 0.0
        self.queue_lengths = {"east_west": [], "north_south": []}
        
    def update(self, dt: float, current_time: float):
        """Update traffic light state based on timing."""
        self.cycle_time += dt
        time_in_state = current_time - self.state_start_time
        
        # East-West direction logic
        if self.states["east_west"] == TrafficLightState.GREEN:
            if time_in_state >= self.green_time:
                self.states["east_west"] = TrafficLightState.YELLOW
                self.state_start_time = current_time
                
        elif self.states["east_west"] == TrafficLightState.YELLOW:
            if time_in_state >= self.yellow_time:
                self.states["east_west"] = TrafficLightState.RED
                self.states["north_south"] = TrafficLightState.GREEN  # Switch to N-S green
                self.state_start_time = current_time
                
        elif self.states["east_west"] == TrafficLightState.RED:
            # Check if N-S has completed its green+yellow cycle
            if self.states["north_south"] == TrafficLightState.RED:
                if time_in_state >= self.all_red_time:
                    self.states["east_west"] = TrafficLightState.GREEN
                    self.state_start_time = current_time
        
        # North-South direction logic
        if self.states["north_south"] == TrafficLightState.GREEN:
            if time_in_state >= self.green_time:
                self.states["north_south"] = TrafficLightState.YELLOW
                self.state_start_time = current_time
                
        elif self.states["north_south"] == TrafficLightState.YELLOW:
            if time_in_state >= self.yellow_time:
                self.states["north_south"] = TrafficLightState.RED
                self.state_start_time = current_time
    
    def get_light_state(self, approach_direction: str) -> TrafficLightState:
        """Get current light state for a specific approach direction."""
        if approach_direction in ["east", "west"]:
            return self.states["east_west"]
        elif approach_direction in ["north", "south"]:
            return self.states["north_south"]
        else:
            return TrafficLightState.RED  # Default to red for safety
    
    def get_time_until_green(self, approach_direction: str, current_time: float) -> float:
        """Calculate time until green light for given direction."""
        current_state = self.get_light_state(approach_direction)
        time_in_state = current_time - self.state_start_time
        
        if current_state == TrafficLightState.GREEN:
            return 0.0
        elif current_state == TrafficLightState.YELLOW:
            # Red time + other direction's green + yellow + all red
            return self.yellow_time - time_in_state + self.red_time
        else:  # RED
            # Time remaining in red
            remaining_red = self.red_time - time_in_state
            return max(0.0, remaining_red)

class HumanDriver:
    """
    Realistic human driver behavior model.
    Implements real-world driving patterns including reaction times,
    acceleration/deceleration curves, and traffic light responses.
    """
    
    def __init__(self, vehicle_id: int, x: float, y: float, target_x: float, target_y: float,
                 approach_direction: str = "east", driver_type: str = "average"):
        self.id = vehicle_id
        self.x = x
        self.y = y
        self.target_x = target_x
        self.target_y = target_y
        self.approach_direction = approach_direction
        
        # Physical properties
        self.length = 5.0
        self.width = 2.0
        
        # Driver characteristics (varies by driver_type)
        self.driver_type = driver_type
        self._set_driver_characteristics()
        
        # Velocity and acceleration
        self.vx = 0.0
        self.vy = 0.0
        self.speed = 0.0
        self.acceleration = 0.0
        
        # Set initial velocity based on approach direction
        initial_speed = random.uniform(8.0, 12.0)  # Start with varied speeds
        if approach_direction == "east":
            self.vx = initial_speed
            self.vy = 0.0
        elif approach_direction == "west":
            self.vx = -initial_speed
            self.vy = 0.0
        elif approach_direction == "north":
            self.vx = 0.0
            self.vy = initial_speed
        elif approach_direction == "south":
            self.vx = 0.0
            self.vy = -initial_speed
            
        # State tracking
        self.state = "free_driving"  # free_driving, approaching_light, stopped, accelerating
        self.total_travel_time = 0.0
        self.waiting_time = 0.0
        self.stop_time = 0.0
        self.completed_trip = False
        
        # Reaction and decision making
        self.reaction_time = random.uniform(0.8, 1.5)  # Human reaction time
        self.last_decision_time = 0.0
        self.decision_distance = None  # Distance at which driver made stop/go decision
        
    def _set_driver_characteristics(self):
        """Set driver characteristics based on driver type."""
        if self.driver_type == "aggressive":
            self.max_speed = random.uniform(18.0, 22.0)
            self.max_acceleration = random.uniform(3.5, 4.5)
            self.max_deceleration = random.uniform(-6.0, -4.5)
            self.following_distance = random.uniform(2.0, 4.0)
            self.yellow_light_aggressiveness = 0.8  # More likely to run yellow
            
        elif self.driver_type == "conservative":
            self.max_speed = random.uniform(12.0, 16.0)
            self.max_acceleration = random.uniform(2.0, 3.0)
            self.max_deceleration = random.uniform(-4.0, -3.0)
            self.following_distance = random.uniform(6.0, 10.0)
            self.yellow_light_aggressiveness = 0.2  # More likely to stop on yellow
            
        else:  # average driver
            self.max_speed = random.uniform(14.0, 18.0)
            self.max_acceleration = random.uniform(2.5, 3.5)
            self.max_deceleration = random.uniform(-5.0, -3.5)
            self.following_distance = random.uniform(4.0, 7.0)
            self.yellow_light_aggressiveness = 0.5  # 50/50 decision on yellow
    
    def update_behavior(self, dt: float, traffic_light: TrafficLight, other_vehicles: List, current_time: float):
        """Update human driver behavior based on traffic conditions."""
        self.total_travel_time += dt
        
        # Calculate current speed
        self.speed = math.hypot(self.vx, self.vy)
        
        # Find distance to intersection
        dist_to_intersection = math.hypot(traffic_light.x - self.x, traffic_light.y - self.y)
        
        # Get current traffic light state
        light_state = traffic_light.get_light_state(self.approach_direction)
        
        # Decision making with reaction time
        if current_time - self.last_decision_time >= self.reaction_time:
            self._make_driving_decision(light_state, dist_to_intersection, other_vehicles, current_time)
            self.last_decision_time = current_time
        
        # Apply behavior based on current state
        if self.state == "free_driving":
            self._free_driving_behavior(dt)
        elif self.state == "approaching_light":
            self._approaching_light_behavior(light_state, dist_to_intersection, dt)
        elif self.state == "stopped":
            self._stopped_behavior(light_state, dist_to_intersection, dt)
        elif self.state == "accelerating":
            self._accelerating_behavior(dt)
        
        # Update position
        self.x += self.vx * dt
        self.y += self.vy * dt
        
        # Track waiting time
        if self.speed < 1.0:
            self.waiting_time += dt
            if self.state == "stopped":
                self.stop_time += dt
        
        # Check completion
        dist_to_target = math.hypot(self.target_x - self.x, self.target_y - self.y)
        if dist_to_target < 8.0:
            self.completed_trip = True
    
    def _make_driving_decision(self, light_state: TrafficLightState, dist_to_intersection: float, 
                             other_vehicles: List, current_time: float):
        """Make driving decisions based on current conditions."""
        
        # Check for vehicles ahead
        vehicle_ahead = self._get_vehicle_ahead(other_vehicles)
        
        if vehicle_ahead:
            dist_to_vehicle = self._get_distance_to_vehicle(vehicle_ahead)
            if dist_to_vehicle < self.following_distance + self.speed * 0.5:
                self.state = "stopped" if self.speed < 1.0 else "approaching_light"
                return
        
        # Traffic light decision logic
        if dist_to_intersection > 100.0:
            self.state = "free_driving"
        elif dist_to_intersection > 50.0:
            self.state = "approaching_light"
        else:
            # Close to intersection - need to decide
            if light_state == TrafficLightState.GREEN:
                self.state = "accelerating"
            elif light_state == TrafficLightState.RED:
                self.state = "stopped"
            elif light_state == TrafficLightState.YELLOW:
                # Yellow light decision - realistic human behavior
                stopping_distance = self._calculate_stopping_distance()
                
                if dist_to_intersection <= stopping_distance:
                    # Too close to stop safely - continue
                    self.state = "accelerating"
                else:
                    # Can stop safely - decision based on driver aggressiveness
                    if random.random() < self.yellow_light_aggressiveness:
                        self.state = "accelerating"  # Run the yellow
                    else:
                        self.state = "stopped"  # Stop for yellow
    
    def _free_driving_behavior(self, dt: float):
        """Free driving behavior - maintain desired speed."""
        target_speed = self.max_speed * random.uniform(0.85, 0.95)  # Slight speed variation
        
        if self.speed < target_speed:
            self.acceleration = self.max_acceleration * random.uniform(0.6, 0.9)
        else:
            self.acceleration = -0.5  # Gentle deceleration
        
        self._update_velocity(dt)
    
    def _approaching_light_behavior(self, light_state: TrafficLightState, dist_to_intersection: float, dt: float):
        """Behavior when approaching traffic light."""
        if light_state == TrafficLightState.GREEN:
            # Maintain or slightly increase speed
            target_speed = self.max_speed * 0.9
            if self.speed < target_speed:
                self.acceleration = self.max_acceleration * 0.7
            else:
                self.acceleration = 0.0
        
        elif light_state == TrafficLightState.RED:
            # Decelerate to stop
            self._decelerate_to_stop(dist_to_intersection, dt)
        
        elif light_state == TrafficLightState.YELLOW:
            # Decision already made in _make_driving_decision
            if self.state == "stopped":
                self._decelerate_to_stop(dist_to_intersection, dt)
            else:
                # Continue through yellow
                self.acceleration = self.max_acceleration * 0.5
        
        self._update_velocity(dt)
    
    def _stopped_behavior(self, light_state: TrafficLightState, dist_to_intersection: float, dt: float):
        """Behavior when stopped at intersection."""
        self.acceleration = self.max_deceleration * 0.3  # Gentle braking
        
        # Check if light turned green
        if light_state == TrafficLightState.GREEN:
            # Reaction time before starting
            if random.random() < 0.3:  # 30% chance to start moving this frame
                self.state = "accelerating"
        
        self._update_velocity(dt)
        
        # Ensure vehicle doesn't move backwards
        if self.approach_direction == "east" and self.vx < 0:
            self.vx = 0
        elif self.approach_direction == "west" and self.vx > 0:
            self.vx = 0
        elif self.approach_direction == "north" and self.vy < 0:
            self.vy = 0
        elif self.approach_direction == "south" and self.vy > 0:
            self.vy = 0
    
    def _accelerating_behavior(self, dt: float):
        """Behavior when accelerating from stop or through intersection."""
        # Progressive acceleration - humans don't floor it immediately
        if self.speed < 5.0:
            self.acceleration = self.max_acceleration * 0.6  # Gentle start
        elif self.speed < 10.0:
            self.acceleration = self.max_acceleration * 0.8  # Build up speed
        else:
            self.acceleration = self.max_acceleration * 0.4  # Approach target speed
        
        # Return to free driving once at reasonable speed
        if self.speed > self.max_speed * 0.7:
            self.state = "free_driving"
        
        self._update_velocity(dt)
    
    def _update_velocity(self, dt: float):
        """Update velocity based on current acceleration and constraints."""
        # Update velocity based on approach direction
        if self.approach_direction == "east":
            self.vx += self.acceleration * dt
            self.vx = np.clip(self.vx, 0.0, self.max_speed)
        elif self.approach_direction == "west":
            self.vx += self.acceleration * dt
            self.vx = np.clip(self.vx, -self.max_speed, 0.0)
        elif self.approach_direction == "north":
            self.vy += self.acceleration * dt
            self.vy = np.clip(self.vy, 0.0, self.max_speed)
        elif self.approach_direction == "south":
            self.vy += self.acceleration * dt
            self.vy = np.clip(self.vy, -self.max_speed, 0.0)
    
    def _calculate_stopping_distance(self) -> float:
        """Calculate realistic stopping distance based on current speed."""
        if self.speed <= 0:
            return 0.0
        
        # Physics: d = v²/(2μg) + reaction_distance
        # μ (friction coefficient) ≈ 0.7 for dry pavement
        # g = 9.81 m/s²
        
        reaction_distance = self.speed * self.reaction_time
        braking_distance = (self.speed ** 2) / (2 * 0.7 * 9.81)
        safety_margin = 3.0  # Extra buffer
        
        return reaction_distance + braking_distance + safety_margin
    
    def _decelerate_to_stop(self, dist_to_intersection: float, dt: float):
        """Calculate appropriate deceleration to stop before intersection."""
        if dist_to_intersection <= 3.0:  # Very close - hard brake
            self.acceleration = self.max_deceleration
        elif self.speed > 0:
            # Calculate required deceleration
            required_decel = (self.speed ** 2) / (2 * max(1.0, dist_to_intersection - 2.0))
            self.acceleration = -min(required_decel, abs(self.max_deceleration))
        else:
            self.acceleration = 0.0
    
    def _get_vehicle_ahead(self, other_vehicles: List) -> Optional['HumanDriver']:
        """Find the nearest vehicle ahead in the same direction."""
        ahead_vehicle = None
        min_distance = float('inf')
        
        for other in other_vehicles:
            if other.id == self.id:
                continue
                
            # Check if vehicle is ahead in the same direction
            if self.approach_direction == "east" and other.x > self.x and abs(other.y - self.y) < 5:
                distance = other.x - self.x
            elif self.approach_direction == "west" and other.x < self.x and abs(other.y - self.y) < 5:
                distance = self.x - other.x
            elif self.approach_direction == "north" and other.y > self.y and abs(other.x - self.x) < 5:
                distance = other.y - self.y
            elif self.approach_direction == "south" and other.y < self.y and abs(other.x - self.x) < 5:
                distance = self.y - other.y
            else:
                continue
            
            if distance > 0 and distance < min_distance:
                min_distance = distance
                ahead_vehicle = other
        
        return ahead_vehicle
    
    def _get_distance_to_vehicle(self, other_vehicle: 'HumanDriver') -> float:
        """Calculate distance to another vehicle."""
        return math.hypot(other_vehicle.x - self.x, other_vehicle.y - self.y)

class TrafficLightEnvironment:
    """
    Traditional traffic light controlled intersection environment.
    Simulates realistic human driving behavior for performance comparison.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        # Environment parameters
        self.num_vehicles = self.config.get("num_vehicles", 4)
        self.max_episode_steps = self.config.get("max_episode_steps", 800)
        self.intersection_x = 50.0
        self.intersection_y = 50.0
        
        # Create traffic light
        self.traffic_light = TrafficLight("main_intersection", self.intersection_x, self.intersection_y)
        
        # Vehicle management
        self.vehicles = []
        self.completed_vehicles = []
        self.total_vehicles_spawned = 0
        
        # Metrics tracking
        self.episode_metrics = {
            "total_travel_time": 0.0,
            "total_waiting_time": 0.0,
            "total_stop_time": 0.0,
            "throughput": 0,
            "average_speed": 0.0,
            "fuel_consumption": 0.0,
            "queue_length_history": [],
            "light_cycle_count": 0
        }
        
        # Timing
        self.current_time = 0.0
        self.step_count = 0
        self.dt = 0.1  # 100ms time steps
        
    def reset(self):
        """Reset environment for new episode."""
        self.current_time = 0.0
        self.step_count = 0
        self.vehicles.clear()
        self.completed_vehicles.clear()
        self.total_vehicles_spawned = 0
        
        # Reset metrics
        self.episode_metrics = {
            "total_travel_time": 0.0,
            "total_waiting_time": 0.0,
            "total_stop_time": 0.0,
            "throughput": 0,
            "average_speed": 0.0,
            "fuel_consumption": 0.0,
            "queue_length_history": [],
            "light_cycle_count": 0
        }
        
        # Reset traffic light
        self.traffic_light = TrafficLight("main_intersection", self.intersection_x, self.intersection_y)
        
        # Spawn initial vehicles
        self._spawn_initial_vehicles()
        
        return self._get_observation()
    
    def step(self):
        """Execute one simulation step."""
        self.step_count += 1
        self.current_time += self.dt
        
        # Update traffic light
        self.traffic_light.update(self.dt, self.current_time)
        
        # Update all vehicles
        for vehicle in self.vehicles:
            vehicle.update_behavior(self.dt, self.traffic_light, self.vehicles, self.current_time)
        
        # Check for completed vehicles
        self._check_completions()
        
        # Spawn new vehicles periodically
        if self.step_count % 100 == 0 and len(self.vehicles) < self.num_vehicles * 2:
            self._spawn_new_vehicle()
        
        # Update metrics
        self._update_metrics()
        
        # Check if episode is done
        done = self.step_count >= self.max_episode_steps
        
        return self._get_observation(), self._calculate_reward(), done, self._get_info()
    
    def _spawn_initial_vehicles(self):
        """Spawn initial vehicles from different directions."""
        directions = ["east", "west", "north", "south"]
        driver_types = ["aggressive", "average", "conservative"]
        
        for i in range(self.num_vehicles):
            direction = directions[i % len(directions)]
            driver_type = random.choice(driver_types)
            
            # Calculate spawn position and target based on direction
            if direction == "east":
                x, y = -50, self.intersection_y + random.uniform(-2, 2)
                target_x, target_y = 150, y
            elif direction == "west":
                x, y = 150, self.intersection_y + random.uniform(-2, 2)
                target_x, target_y = -50, y
            elif direction == "north":
                x, y = self.intersection_x + random.uniform(-2, 2), -50
                target_x, target_y = x, 150
            else:  # south
                x, y = self.intersection_x + random.uniform(-2, 2), 150
                target_x, target_y = x, -50
            
            vehicle = HumanDriver(
                vehicle_id=self.total_vehicles_spawned,
                x=x, y=y,
                target_x=target_x, target_y=target_y,
                approach_direction=direction,
                driver_type=driver_type
            )
            
            self.vehicles.append(vehicle)
            self.total_vehicles_spawned += 1
    
    def _spawn_new_vehicle(self):
        """Spawn a new vehicle during simulation."""
        directions = ["east", "west", "north", "south"]
        direction = random.choice(directions)
        driver_type = random.choice(["aggressive", "average", "conservative"])
        
        # Calculate spawn position (further back to avoid conflicts)
        if direction == "east":
            x, y = -80, self.intersection_y + random.uniform(-2, 2)
            target_x, target_y = 180, y
        elif direction == "west":
            x, y = 180, self.intersection_y + random.uniform(-2, 2)
            target_x, target_y = -80, y
        elif direction == "north":
            x, y = self.intersection_x + random.uniform(-2, 2), -80
            target_x, target_y = x, 180
        else:  # south
            x, y = self.intersection_x + random.uniform(-2, 2), 180
            target_x, target_y = x, -80
        
        vehicle = HumanDriver(
            vehicle_id=self.total_vehicles_spawned,
            x=x, y=y,
            target_x=target_x, target_y=target_y,
            approach_direction=direction,
            driver_type=driver_type
        )
        
        self.vehicles.append(vehicle)
        self.total_vehicles_spawned += 1
    
    def _check_completions(self):
        """Check for vehicles that have completed their trips."""
        completed = []
        for vehicle in self.vehicles:
            if vehicle.completed_trip:
                completed.append(vehicle)
        
        for vehicle in completed:
            self.vehicles.remove(vehicle)
            self.completed_vehicles.append(vehicle)
            self.episode_metrics["throughput"] += 1
    
    def _update_metrics(self):
        """Update episode performance metrics."""
        if not self.vehicles:
            return
        
        # Calculate queue lengths by direction
        queues = {"east": 0, "west": 0, "north": 0, "south": 0}
        total_speed = 0.0
        
        for vehicle in self.vehicles:
            # Count vehicles in queue (stopped or very slow near intersection)
            dist_to_intersection = math.hypot(
                self.traffic_light.x - vehicle.x,
                self.traffic_light.y - vehicle.y
            )
            
            if dist_to_intersection < 80 and vehicle.speed < 2.0:
                queues[vehicle.approach_direction] += 1
            
            total_speed += vehicle.speed
            
            # Accumulate individual metrics
            self.episode_metrics["total_travel_time"] += vehicle.total_travel_time * self.dt
            self.episode_metrics["total_waiting_time"] += vehicle.waiting_time * self.dt
            self.episode_metrics["total_stop_time"] += vehicle.stop_time * self.dt
        
        # Store queue length snapshot
        self.episode_metrics["queue_length_history"].append({
            "time": self.current_time,
            "queues": queues.copy(),
            "total_queue": sum(queues.values())
        })
        
        # Average speed
        if len(self.vehicles) > 0:
            self.episode_metrics["average_speed"] = total_speed / len(self.vehicles)
    
    def _get_observation(self):
        """Get current state observation (for monitoring)."""
        return {
            "vehicles": len(self.vehicles),
            "completed": len(self.completed_vehicles),
            "traffic_light_state": {
                "east_west": self.traffic_light.states["east_west"].value,
                "north_south": self.traffic_light.states["north_south"].value
            },
            "current_time": self.current_time
        }
    
    def _calculate_reward(self):
        """Calculate episode reward (negative total system cost)."""
        # Simple reward based on system efficiency
        throughput_reward = len(self.completed_vehicles) * 100
        waiting_penalty = -sum(v.waiting_time for v in self.vehicles + self.completed_vehicles)
        
        return throughput_reward + waiting_penalty
    
    def _get_info(self):
        """Get additional episode information."""
        return {
            "metrics": self.episode_metrics.copy(),
            "step_count": self.step_count,
            "vehicles_active": len(self.vehicles),
            "vehicles_completed": len(self.completed_vehicles)
        }
    
    def get_final_metrics(self):
        """Get comprehensive final episode metrics."""
        total_vehicles = len(self.completed_vehicles)
        
        if total_vehicles == 0:
            return self.episode_metrics
        
        # Calculate averages for completed vehicles
        avg_travel_time = sum(v.total_travel_time for v in self.completed_vehicles) / total_vehicles
        avg_waiting_time = sum(v.waiting_time for v in self.completed_vehicles) / total_vehicles
        avg_stop_time = sum(v.stop_time for v in self.completed_vehicles) / total_vehicles
        
        # Calculate throughput (vehicles per minute)
        throughput_per_minute = (total_vehicles / (self.current_time / 60.0)) if self.current_time > 0 else 0
        
        # Calculate queue statistics
        if self.episode_metrics["queue_length_history"]:
            max_queue = max(q["total_queue"] for q in self.episode_metrics["queue_length_history"])
            avg_queue = sum(q["total_queue"] for q in self.episode_metrics["queue_length_history"]) / len(self.episode_metrics["queue_length_history"])
        else:
            max_queue = avg_queue = 0
        
        final_metrics = {
            "total_vehicles": total_vehicles,
            "episode_duration": self.current_time,
            "average_travel_time": avg_travel_time,
            "average_waiting_time": avg_waiting_time,
            "average_stop_time": avg_stop_time,
            "throughput_per_minute": throughput_per_minute,
            "average_speed": self.episode_metrics["average_speed"],
            "max_queue_length": max_queue,
            "average_queue_length": avg_queue,
            "traffic_light_cycles": self.current_time / self.traffic_light.total_cycle_time
        }
        
        return final_metrics 