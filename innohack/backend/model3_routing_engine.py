"""
Model 3 — Predictive Routing Engine 🗺️
======================================
Implements A* / Dijkstra routing algorithms.
Key modification: Edge weights are NOT distance. 
Edge Cost = Predicted Travel Time (Base Time + Model 1 Congestion Forecast).
"""

import sys
import heapq
from typing import Dict, List, Tuple, Any

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Import the OSM Graph from Module 3
from module3_osm_graph import build_sample_madrid_osm_graph, OSMRoadGraph


class PredictiveRoutingEngine:
    def __init__(self, graph: OSMRoadGraph):
        self.graph = graph

    def get_model1_predicted_congestion(self, road_id: str, forecast_horizon_min: int = 15) -> float:
        """
        Simulates querying Model 1 (Forecasting Engine) for a specific road.
        Returns a Congestion Multiplier (1.0 = free flow, >1.0 = congested).
        
        In the real system, this calls: Model1.predict(road_id, time=now + 15m)
        """
        # For demonstration: We inject a predicted heavy traffic jam on the main highway (R_M30_02)
        if road_id == "R_M30_02":
            return 3.5  # 3.5x slower due to predicted heavy congestion in 15 mins!
        elif road_id == "R_M30_01":
            return 1.5
        elif road_id == "R_CITY_02":
            return 1.8
        else:
            return 1.0  # Free flow on other roads

    def calculate_predicted_travel_time(self, u: str, v: str, use_predictions: bool = True) -> float:
        """
        Calculates edge cost (travel time in seconds).
        If use_predictions is True, applies Model 1 congestion multipliers.
        """
        edge = self.graph.get_edge_attributes(u, v)
        if not edge:
            return float('inf')
        
        # Base free-flow time (Length / MaxSpeed)
        speed_m_s = edge["maxspeed"] / 3.6
        base_time_sec = edge["length"] / speed_m_s

        if not use_predictions:
            return base_time_sec

        # Apply Model 1 predicted congestion multiplier
        congestion_multiplier = self.get_model1_predicted_congestion(edge["road_id"])
        
        # Predicted Travel Time
        return base_time_sec * congestion_multiplier

    def find_optimal_path(self, start_node: str, end_node: str, use_predictions: bool = True) -> Dict[str, Any]:
        """
        Standard Dijkstra's algorithm, but the cost function is dynamically modified
        by the Predicted Travel Time.
        """
        # Priority queue: (cumulative_cost_time, current_node, path_taken)
        queue = [(0.0, start_node, [start_node])]
        visited = set()
        
        # Track best times to each node to avoid redundant paths
        min_times = {start_node: 0.0}

        while queue:
            current_time, current_node, path = heapq.heappop(queue)

            if current_node == end_node:
                # Calculate physical distance of this path
                total_distance = 0.0
                for i in range(len(path)-1):
                    u, v = path[i], path[i+1]
                    total_distance += self.graph.get_edge_attributes(u, v)["length"]
                
                return {
                    "path": path,
                    "travel_time_sec": current_time,
                    "travel_time_min": round(current_time / 60.0, 1),
                    "distance_km": round(total_distance / 1000.0, 2)
                }

            if current_node in visited:
                continue
            visited.add(current_node)

            # Explore neighbors
            neighbors = self.graph.edges.get(current_node, {})
            for neighbor_node in neighbors:
                if neighbor_node in visited:
                    continue

                # Calculate custom cost (Predicted Travel Time)
                edge_cost = self.calculate_predicted_travel_time(current_node, neighbor_node, use_predictions)
                new_time = current_time + edge_cost

                if new_time < min_times.get(neighbor_node, float('inf')):
                    min_times[neighbor_node] = new_time
                    heapq.heappush(queue, (new_time, neighbor_node, path + [neighbor_node]))

        return {"error": "No path found"}


if __name__ == "__main__":
    print("==================================================================")
    print("🗺️ MODEL 3: PREDICTIVE ROUTING ENGINE (DIJKSTRA / A*)")
    print("==================================================================")
    
    # 1. Load OSM Graph (Module 3)
    graph = build_sample_madrid_osm_graph()
    router = PredictiveRoutingEngine(graph)
    
    start, end = "N1", "N5"
    print(f"[*] Task: Find the fastest route from {start} to {end}")
    
    # Scenario A: Standard Routing (Static Map, No AI Predictions)
    print("\n[ SCENARIO A: Standard Map Routing (Distance / Free-Flow) ]")
    print("    Assumes road is empty. Ignores future traffic.")
    standard_route = router.find_optimal_path(start, end, use_predictions=False)
    
    print(f"    -> Route Taken:  {' ➔ '.join(standard_route['path'])}")
    print(f"    -> Distance:     {standard_route['distance_km']} km")
    print(f"    -> ETA:          {standard_route['travel_time_min']} minutes ({round(standard_route['travel_time_sec'])} sec)")
    
    # Scenario B: Predictive Routing (Powered by Model 1 Forecasts)
    print("\n[ SCENARIO B: AI Predictive Routing (Model 1 Integrated) ]")
    print("    Model 1 predicts a massive traffic jam forming on highway segment R_M30_02 in 15 mins!")
    print("    Routing algorithm dynamically adjusts edge costs based on predicted travel time.")
    predictive_route = router.find_optimal_path(start, end, use_predictions=True)
    
    print(f"    -> Route Taken:  {' ➔ '.join(predictive_route['path'])}")
    print(f"    -> Distance:     {predictive_route['distance_km']} km")
    print(f"    -> ETA:          {predictive_route['travel_time_min']} minutes ({round(predictive_route['travel_time_sec'])} sec)")
    
    print("\n[ 📊 ANALYSIS ]")
    if standard_route['path'] != predictive_route['path']:
        print("    SUCCESS! Model 3 successfully detected the predicted congestion on the M-30 highway")
        print("    and re-routed the driver through the city arterials (N1 ➔ N4 ➔ N6 ➔ N5).")
        print("    Although the physical distance is slightly longer, it saves the driver from getting stuck!")
    else:
        print("    Same path taken.")
    print("==================================================================")
