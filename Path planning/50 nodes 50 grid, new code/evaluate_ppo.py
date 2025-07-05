import matplotlib.pyplot as plt
import numpy as np
from uav_env import UAVPathPlanningEnv
from stable_baselines3 import PPO

# ✅ Load trained model
env = UAVPathPlanningEnv()
model = PPO.load("ppo_uav_path_planning")

# ✅ Reset environment
obs, _ = env.reset()
done = False
step_count = 0
MAX_EVAL_STEPS = 1000

# ✅ UAV Path and Stops Tracking
uav_path = [env.uav_position.tolist()]
stop_locations = set()
stop_data_transfer = {}  # Tracks data transmission per stop

# ✅ Battery and Flight Metrics
battery_remaining = 100  # Start with full battery (percentage)
battery_per_distance = 0.2  # Battery drain per unit distance
battery_per_time = 0.2  # Battery drain per time unit spent at a stop
speed = 1.0  # UAV speed in units per time
low_battery_threshold = 40  # Below this, speed decreases

# ✅ Additional tracking variables
total_flight_time = 0
total_data_collected = 0
travel_times = []
stop_times = []
battery_usage = []
last_stop = None
stopped_at_base = False  # Prevents infinite loop at (0,0)

while not done and step_count < MAX_EVAL_STEPS:
    action, _ = model.predict(obs)
    obs, reward, done, _, _ = env.step(action)

    uav_position = tuple(env.uav_position)

    # ✅ Prevent stopping at the same location multiple times
    if uav_position != last_stop:
        uav_path.append(list(uav_position))

        # ✅ Compute travel distance
        if len(uav_path) > 1:
            prev_location = np.array(uav_path[-2])
            current_location = np.array(uav_path[-1])
            travel_distance = np.linalg.norm(prev_location - current_location)

            # ✅ Adjust travel time based on speed
            adjusted_speed = speed * 0.5 if battery_remaining < low_battery_threshold else speed
            travel_time = travel_distance / adjusted_speed
            total_flight_time += travel_time
            travel_times.append((uav_position, travel_time))

            # 🔋 Battery Consumption - Travel
            battery_used = travel_distance * battery_per_distance
            battery_remaining -= battery_used
            battery_usage.append((uav_position, battery_used))

        # ✅ If UAV reaches base after visiting all stops, terminate
        if uav_position == (0, 0) and len(stop_locations) == len(env.stop_locations):
            print(f"🏁 UAV has returned to base at {uav_position}. Ending simulation.")
            break  # ✅ Stop simulation after completing all stops

        # 🚨 **Fix: Don't treat (0,0) as a stop**
        if uav_position != (0, 0):  # ✅ Avoid treating (0,0) as a stop
            stop_locations.add(uav_position)
            last_stop = uav_position  

            # ✅ Get nodes assigned to this stop from `cluster_mapping`
            nodes_at_stop = env.cluster_mapping.get(uav_position, [])

            found_node_ids = []
            total_data_at_stop = 0  

            for node_pos, data_size in nodes_at_stop:
                node_id = next((node[0] for node in env.nodes if (node[1], node[2]) == tuple(node_pos)), None)
                if node_id is not None:
                    found_node_ids.append((node_id, data_size))
                    total_data_at_stop += data_size

            print(f"\n📡 At Stop {uav_position}, Found Nodes: {found_node_ids}")

            # ✅ Compute stop time based on total data collected
            stop_time = total_data_at_stop / env.data_transfer_rate  # ✅ Time required to transmit all data
            stop_times.append((uav_position, stop_time))
            total_data_collected += total_data_at_stop  

            # 🔋 Battery Consumption - Stop Time
            battery_used_stop = stop_time * battery_per_time
            battery_remaining -= battery_used_stop
            battery_usage.append((uav_position, battery_used_stop))

            # ✅ Store updated transmission sequence
            stop_data_transfer[uav_position] = found_node_ids

    step_count += 1

# ✅ Convert to numpy arrays
uav_path = np.array(uav_path)
stop_locations = np.array(list(stop_locations)) if stop_locations else np.array([])

# ✅ Print all nodes that should be collected
all_node_ids = set(node[0] for node in env.nodes)
collected_node_ids = set()

# ✅ Track which nodes were actually collected
for stop, sequence in stop_data_transfer.items():
    for node_id, data_size in sequence:
        collected_node_ids.add(node_id)

# ✅ Print missing nodes
missing_nodes = all_node_ids - collected_node_ids
if missing_nodes:
    print("\n🚨 **Missing Nodes:**", missing_nodes)
else:
    print("\n✅ **All Nodes Were Collected!**")

# ✅ Display results
print(f"\n🚀 **Total Flight Time:** {total_flight_time:.2f} time units")
print(f"🔋 **Battery Remaining:** {battery_remaining:.2f}%")
print(f"📦 **Total Data Collected:** {total_data_collected} KB")

print("\n🔽 **Stop Time at Each Stop Location:**")
for stop, time in stop_times:
    battery_used = next((b[1] for b in battery_usage if b[0] == stop), 0)
    print(f"📍 Stop at {stop}: ⏳ {time:.2f} time units 🔋 Battery Used: {battery_used:.2f}%")

print("\n📡 **Data Transmission Order at Each Stop:**")
for stop, sequence in stop_data_transfer.items():
    print(f"📍 Stop at {stop}:")
    for node_id, data_size in sequence:
        print(f"    🔹 Node {node_id} → {data_size} KB ({data_size / 5:.2f} time units)")

print("\n🛫 **Travel Time Between Stops:**")
for stop, travel_time in travel_times:
    print(f"🚀 Travel to {stop}: ⏳ {travel_time:.2f} time units")

# ✅ Plot UAV Path
plt.figure(figsize=(8, 8))
plt.grid(True)
plt.xlim(0, env.grid_size)
plt.ylim(0, env.grid_size)

# ✅ Sensor Nodes
node_positions = [(node[1], node[2]) for node in env.nodes]
plt.scatter(*zip(*node_positions), marker="o", color="red", label="Sensor Nodes")

# ✅ UAV Path
plt.plot(uav_path[:, 1], uav_path[:, 0], marker="s", color="blue", label="UAV Path", linestyle="-")

# ✅ Stop Locations
if len(stop_locations) > 0:
    plt.scatter(stop_locations[:, 1], stop_locations[:, 0], marker="x", color="purple", s=100, label="Stops") 

# ✅ Base Station
plt.scatter(env.base_location[1], env.base_location[0], marker="*", color="green", s=150, label="Base")

plt.legend()
plt.title("Optimized UAV Path")
plt.xlabel("X Coordinate")
plt.ylabel("Y Coordinate")
plt.show()
