import math
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))
from environments.multi_agent_traffic_env import Vehicle, IntersectionManager, SixGNetwork

def test_priority_conflict_resolution():
    net = SixGNetwork()
    inter = IntersectionManager("I0", 0, 0)
    inter.set_network(net)
    v1 = Vehicle(1, -20, 0, 20, 0, max_speed=10, approach_direction="east", priority="high")
    v2 = Vehicle(2, 0, -20, 0, 20, max_speed=10, approach_direction="north", priority="normal")
    v1.set_network(net)
    v2.set_network(net)
    msg1 = {"type": "reservation_request", "eta": 5.0, "vehicle_id": 1, "approach_direction": "east", "priority": "high", "request_time":0}
    msg2 = {"type": "reservation_request", "eta": 5.0, "vehicle_id": 2, "approach_direction": "north", "priority": "normal", "request_time":0}
    inter.receive_message(msg1, v1)
    inter.receive_message(msg2, v2)
    assert len(inter.reservations) == 2
    inter.reservations.sort(key=lambda r: r["entry_time"])
    first, second = inter.reservations
    assert first["vehicle_id"] == 1
    assert second["entry_time"] >= first["exit_time"]

def test_trajectory_feasibility():
    inter = IntersectionManager("I0", 0, 0)
    v = Vehicle(1, -10, 0, 20, 0, max_speed=5, approach_direction="east")
    eta = 1.0
    feasible = inter._find_feasible_time(eta, "east", "normal", v, 0.0)
    assert feasible is not None
    dt = feasible - 0.0
    dist = math.hypot(v.x - inter.x, v.y - inter.y)
    speed = math.hypot(v.vx, v.vy)
    req_acc = 2*(dist - speed*dt)/(dt**2)
    assert v.max_deceleration <= req_acc <= v.max_acceleration

from src.environments.multi_agent_traffic_env import MultiAgentTrafficEnv

def test_safe_crossing_env():
    env = MultiAgentTrafficEnv({"num_vehicles": 2, "max_episode_steps": 100, "dt":0.1, "vehicle_spawn_distance":20})
    obs, _ = env.reset()
    inter = env.intersections[0]
    v0 = env.vehicles["vehicle_0"]
    v1 = env.vehicles["vehicle_1"]
    v0.request_intersection_reservation(inter, env.time)
    v1.request_intersection_reservation(inter, env.time)
    env.network.deliver_messages(env.time + 0.1)
    r0 = inter.get_vehicle_reservation(v0.id)
    r1 = inter.get_vehicle_reservation(v1.id)
    assert r0 and r1
    cross = {v0.id: None, v1.id: None}
    for _ in range(80):
        actions = {aid: [0.0] for aid in env.agents}
        obs, rewards, dones, _, info = env.step(actions)
        for veh in [v0, v1]:
            if cross[veh.id] is None:
                d = math.hypot(veh.x - inter.x, veh.y - inter.y)
                if d < inter.conflict_radius:
                    cross[veh.id] = env.time
        if dones.get("__all__"):
            break
    assert cross[v0.id] >= r0["entry_time"]
    assert cross[v1.id] >= r1["entry_time"]
    assert not v0.collision_occurred and not v1.collision_occurred
    env.close() if hasattr(env, 'close') else None
