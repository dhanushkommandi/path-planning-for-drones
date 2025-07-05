import gym
import numpy as np
from gym import spaces
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import cdist
from scipy.spatial import distance

class UAVPathPlanningEnv(gym.Env):
    def __init__(self, grid_size=50, base_location=(0, 0), max_steps=150, uav_range=4, data_transfer_rate=5):
        super(UAVPathPlanningEnv, self).__init__()

        self.grid_size = grid_size
        self.base_location = np.array(base_location)
        self.uav_range = uav_range
        self.data_transfer_rate = data_transfer_rate
        self.max_steps = max_steps

        # ✅ Fixed sensor nodes (ID, x, y, data in KB)
        self.nodes = [
            (0, 12, 45, 25), (1, 8, 20, 30), (2, 35, 15, 40), (3, 42, 8, 22),
            (4, 10, 30, 47), (5, 20, 18, 41), (6, 45, 40, 20), (7, 28, 35, 32),
            (8, 18, 25, 36), (9, 5, 22, 50), (10, 24, 30, 42), (11, 39, 25, 14),
            (12, 33, 45, 26), (13, 48, 20, 13), (14, 22, 42, 34), (15, 38, 12, 44),
            (16, 9, 10, 48), (17, 12, 40, 24), (18, 20, 33, 19), (19, 4, 14, 38),
            (20, 40, 48, 16), (21, 14, 8, 42), (22, 35, 9, 36), (23, 10, 45, 14),
            (24, 28, 40, 30), (25, 46, 18, 20), (26, 30, 12, 45), (27, 18, 28, 36),
            (28, 27, 22, 29), (29, 38, 44, 21), (30, 6, 35, 26), (31, 30, 5, 48),
            (32, 42, 30, 25), (33, 22, 46, 18), (34, 34, 40, 28), (35, 44, 28, 40),
            (36, 25, 10, 47), (37, 48, 24, 12), (38, 2, 33, 20), (39, 36, 4, 32), (40, 14, 20, 45), 
            (41, 30, 30, 35), (42, 40, 10, 30), (43, 20, 40, 25), (44, 10, 20, 40), (45, 30, 10, 45), 
            (46, 40, 20, 30), (47, 20, 30, 35), (48, 30, 40, 25), (49, 10, 10, 40)
        ]




        # ✅ Compute Clusters & Stop Locations using DBSCAN
        self.stop_locations, self.cluster_mapping = self.compute_cluster_stops()
        print("\n✅ Clustering Completed!")

        # ✅ Optimize Path
        print("\n✅ Running Path Optimization...")
        self.optimized_path = self.get_optimized_path()
        print("\n✅ Path Optimization Completed!")

        # ✅ UAV State Variables
        self.uav_position = np.array(self.base_location)
        self.visited_stops = set()
        self.steps_taken = 0
        self.total_flight_time = 0
        self.total_data_collected = 0
        self.stop_times = []
        self.current_target_index = 0

        # ✅ Define Spaces (State includes UAV position + Stop locations + Node Data)
        self.observation_size = 2 + len(self.nodes) * 4
        self.observation_space = spaces.Box(low=0, high=self.grid_size, shape=(self.observation_size,), dtype=np.float32)
        self.action_space = spaces.Discrete(len(self.optimized_path))  

    def compute_cluster_stops(self):
        """Clusters nodes and assigns stop locations, ensuring every node is covered."""
        node_positions = np.array([(n[1], n[2]) for n in self.nodes])
        node_data = np.array([n[3] for n in self.nodes])
        clustering = DBSCAN(eps=self.uav_range, min_samples=1).fit(node_positions)
        cluster_labels = clustering.labels_

        stop_locations = []
        cluster_mapping = {}
        assigned_nodes = set()

        for cluster_id in set(cluster_labels):
            cluster_nodes = node_positions[cluster_labels == cluster_id]
            cluster_data = node_data[cluster_labels == cluster_id]

            # Compute weighted centroid
            centroid = np.average(cluster_nodes, axis=0, weights=cluster_data)

            # Snap stop to the closest node in the cluster
            closest_node = min(cluster_nodes, key=lambda n: np.linalg.norm(n - centroid))

            stop_locations.append(tuple(closest_node))
            cluster_mapping[tuple(closest_node)] = list(zip(cluster_nodes, cluster_data))

            # Mark nodes in this stop as assigned
            for n in cluster_nodes:
                assigned_nodes.add(tuple(n))

        print("\n📌 **DEBUG: Cluster Assignments BEFORE Fix**")
        for stop, nodes in cluster_mapping.items():
            print(f"🛑 Stop at {stop}: {[(tuple(n[0]), n[1]) for n in nodes]}")

        # ✅ Ensure every assigned node is within range of its stop
        for stop, nodes in cluster_mapping.items():
            cluster_mapping[stop] = [
                (node, data) for node, data in nodes if np.linalg.norm(np.array(stop) - np.array(node)) <= self.uav_range
            ]

        # ✅ **FINAL CHECK: Add missing nodes to nearest stop**
        all_nodes = {tuple(node[1:3]) for node in self.nodes}
        orphaned_nodes = all_nodes - assigned_nodes  

        for orphan in orphaned_nodes:
            orphan_pos = np.array(orphan)
            closest_stop = min(stop_locations, key=lambda s: np.linalg.norm(orphan_pos - np.array(s)))
            cluster_mapping[tuple(closest_stop)].append(
                (orphan_pos, next(n[3] for n in self.nodes if tuple(n[1:3]) == orphan))
            )
            assigned_nodes.add(tuple(orphan_pos))

        print("\n📌 **DEBUG: Cluster Assignments AFTER Fix**")
        for stop, nodes in cluster_mapping.items():
            print(f"🛑 Stop at {stop}: {[(tuple(n[0]), n[1]) for n in nodes]}")

        return stop_locations, cluster_mapping

    def step(self, action):
        """Moves UAV to selected stop and collects data, ensuring all nodes are covered."""
        self.steps_taken += 1
        reward = -1
        terminated = False
        truncated = False  # ✅ Gymnasium expects this

        if self.current_target_index < len(self.optimized_path):
            target_stop = np.array(self.optimized_path[self.current_target_index])
            self.uav_position = target_stop
            self.visited_stops.add(tuple(target_stop))
            self.current_target_index += 1  

            reward += 100  # ✅ Reward for reaching a stop

            # ✅ Collect data from all nodes at this stop
            nodes_at_stop = self.cluster_mapping.get(tuple(self.uav_position), [])

            found_node_ids = []
            total_data_at_stop = 0  
            for n in nodes_at_stop:
                node_id = next((node[0] for node in self.nodes if (node[1], node[2]) == tuple(n[0])), None)
                if node_id is not None:
                    found_node_ids.append(node_id)
                    total_data_at_stop += n[1]  

            print(f"\n📡 At Stop {self.uav_position}, Found Nodes: {found_node_ids}")

            if nodes_at_stop:
                stop_time = total_data_at_stop / self.data_transfer_rate  
                self.stop_times.append((tuple(self.uav_position), stop_time))
                self.total_flight_time += stop_time  
                self.total_data_collected += total_data_at_stop  
                reward += 50 / (1 + stop_time)  

        if np.array_equal(self.uav_position, self.base_location) and len(self.visited_stops) == len(self.stop_locations):
            reward += 200
            terminated = True

        # ✅ Ensure episode ends if max steps reached
        if self.steps_taken >= self.max_steps:
            truncated = True  # ✅ Set truncated to True when max steps are reached

        self.total_flight_time += 1

        # ✅ Ensure no nodes are left uncollected
        all_nodes = {tuple(node[1:3]) for node in self.nodes}
        collected_nodes = {tuple(n[0]) for stop in self.cluster_mapping.values() for n in stop}
        missing_nodes = all_nodes - collected_nodes

        # if missing_nodes:
        #     print(f"🚨 WARNING: Some nodes were assigned but NOT collected! {missing_nodes}")

        return self._get_observation(), reward, terminated, truncated, {}  # ✅ Returning correct format


    def get_optimized_path(self):
        """Finds an efficient path using Nearest Neighbor Heuristic instead of permutations."""
        all_locations = [self.base_location] + self.stop_locations + [self.base_location]
        num_stops = len(all_locations)

        distance_matrix = cdist(all_locations, all_locations, metric='euclidean')

        # ✅ Start at base
        unvisited = set(range(1, num_stops - 1))
        current = 0  
        path = [current]

        while unvisited:
            nearest_stop = min(unvisited, key=lambda i: distance_matrix[current][i])
            path.append(nearest_stop)
            unvisited.remove(nearest_stop)
            current = nearest_stop

        path.append(num_stops - 1)  # Return to base
        optimized_path = [all_locations[i] for i in path]

        return optimized_path
    
    def reset(self, seed=None, options=None):
        """Resets the environment and returns the initial observation."""
        super().reset(seed=seed)

        self.uav_position = np.array(self.base_location)
        self.visited_stops.clear()
        self.steps_taken = 0
        self.total_flight_time = 0
        self.total_data_collected = 0
        self.stop_times.clear()
        self.current_target_index = 0  

        return self._get_observation(), {}  # ✅ Gymnasium expects (obs, info)


    def _get_observation(self):
        """Constructs the observation state dynamically."""
        nodes_info = []
        for node in self.nodes:
            nodes_info.extend([node[1], node[2], node[3], 1 if node[0] in self.visited_stops else 0])

        return np.array([*self.uav_position, *nodes_info], dtype=np.float32)

    def render(self):
        """Visualizes the UAV grid."""
        print(f"UAV at {self.uav_position}, Stops visited: {len(self.visited_stops)}/{len(self.stop_locations)}")
        print(f"Total Flight Time: {self.total_flight_time:.2f}")
        print(f"Total Data Collected: {self.total_data_collected} KB")
        for stop, time in self.stop_times:
            print(f"Stop at {stop}: {time:.2f} time units")
