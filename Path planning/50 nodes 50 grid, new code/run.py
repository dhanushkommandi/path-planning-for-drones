from uav_env import UAVPathPlanningEnv

print("✅ Creating UAVPathPlanningEnv instance...")
env = UAVPathPlanningEnv()

print("\n✅ Running Clustering Test...")
stop_locations, cluster_mapping = env.compute_cluster_stops()

print("\n✅ Cluster Assignments:")
for stop, nodes in cluster_mapping.items():
    print(f"🛑 Stop at {stop}: {[(tuple(n[0]), n[1]) for n in nodes]}")

print("\n✅ Clustering Works!")
