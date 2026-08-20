"""
Module 3 — OpenStreetMap (OSM) Road Graph Database
==================================================
Converts road networks into a Directed Graph Database.
Nodes: Intersections (lat, lon)
Edges: Road Segments with static attributes:
       - road_id, length, lanes, maxspeed, oneway, road_type

This graph serves as the foundational map for Model 3 (Routing Engine).
"""

import sys
import json
import math
import heapq
from typing import Dict, List, Tuple, Any, Optional

# Safe encoding for Windows console
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

class OSMRoadGraph:
    def __init__(self):
        # Nodes: {node_id: {"lat": float, "lon": float}}
        self.nodes: Dict[str, Dict[str, float]] = {}
        
        # Edges (Adjacency List): {u: {v: {attributes...}}}
        # Directed graph: u -> v
        self.edges: Dict[str, Dict[str, Dict[str, Any]]] = {}
        
        # Map flat road_ids back to (u, v) pairs for quick updates from Model 1
        self.road_index: Dict[str, Tuple[str, str]] = {}

    def add_node(self, node_id: str, lat: float, lon: float):
        """Adds a physical intersection/node to the graph."""
        self.nodes[node_id] = {"lat": lat, "lon": lon}
        if node_id not in self.edges:
            self.edges[node_id] = {}

    def add_edge(self, u: str, v: str, road_id: str, length: float, lanes: int, 
                 maxspeed: float, oneway: bool, road_type: str):
        """
        Adds a road segment (edge) between node u and node v.
        """
        if u not in self.nodes or v not in self.nodes:
            raise ValueError(f"Nodes {u} and {v} must be added before creating an edge.")

        # Add directed edge u -> v
        self.edges[u][v] = {
            "road_id": road_id,
            "length": length,        # in meters
            "lanes": lanes,          # integer count
            "maxspeed": maxspeed,    # in km/h
            "oneway": oneway,        # boolean
            "road_type": road_type   # e.g., 'motorway', 'primary', 'residential'
        }
        self.road_index[road_id] = (u, v)

        # If it's a two-way street, add the reverse edge v -> u
        if not oneway:
            if v not in self.edges:
                self.edges[v] = {}
            reverse_road_id = f"{road_id}_rev"
            self.edges[v][u] = {
                "road_id": reverse_road_id,
                "length": length,
                "lanes": lanes,
                "maxspeed": maxspeed,
                "oneway": False,
                "road_type": road_type
            }
            self.road_index[reverse_road_id] = (v, u)

    def get_edge_attributes(self, u: str, v: str) -> Optional[Dict[str, Any]]:
        """Returns the static OSM attributes for a given road segment."""
        return self.edges.get(u, {}).get(v)

    def calculate_free_flow_time(self, u: str, v: str) -> float:
        """
        Calculates the baseline time to traverse the edge WITHOUT traffic.
        Returns time in seconds.
        """
        edge = self.get_edge_attributes(u, v)
        if not edge:
            return float('inf')
        
        # length (m) / maxspeed (m/s)
        # maxspeed km/h -> m/s = maxspeed / 3.6
        speed_m_s = edge["maxspeed"] / 3.6
        return edge["length"] / speed_m_s

    def export_to_json(self, filepath: str):
        """Exports the graph to JSON format."""
        data = {
            "nodes": self.nodes,
            "edges": self.edges
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def get_graph_stats(self) -> Dict[str, Any]:
        """Returns statistics about the Graph Database."""
        total_length_m = sum(attr["length"] for u, neighbors in self.edges.items() for v, attr in neighbors.items())
        return {
            "total_nodes": len(self.nodes),
            "total_directed_edges": sum(len(neighbors) for neighbors in self.edges.values()),
            "total_road_length_km": round(total_length_m / 1000.0, 2)
        }


# =====================================================================
# Build a sample OSM Graph representative of Madrid / Target City
# =====================================================================
def build_sample_madrid_osm_graph() -> OSMRoadGraph:
    """Builds an OSM Graph using typical structural data."""
    graph = OSMRoadGraph()
    
    # 1. Add Nodes (Intersections)
    # Using real coordinates roughly around Madrid center/M-30
    nodes = [
        ("N1", 40.4168, -3.7038), ("N2", 40.4180, -3.7010),
        ("N3", 40.4195, -3.6985), ("N4", 40.4150, -3.7050),
        ("N5", 40.4200, -3.6900), ("N6", 40.4140, -3.6950)
    ]
    for n_id, lat, lon in nodes:
        graph.add_node(n_id, lat, lon)

    # 2. Add Edges (Roads)
    # Attributes: road_id, length(m), lanes, maxspeed(km/h), oneway, road_type
    roads = [
        ("N1", "N2", "R_M30_01", 850.0, 3, 90.0, True, "motorway"),
        ("N2", "N3", "R_M30_02", 1200.0, 3, 90.0, True, "motorway"),
        ("N3", "N5", "R_M30_03", 950.0, 4, 90.0, True, "motorway"),
        ("N1", "N4", "R_CITY_01", 400.0, 2, 50.0, False, "primary"),
        ("N4", "N6", "R_CITY_02", 600.0, 2, 50.0, False, "primary"),
        ("N6", "N5", "R_CITY_03", 1100.0, 2, 60.0, False, "primary"),
        ("N2", "N6", "R_CROSS_01", 550.0, 1, 40.0, True, "residential"),
    ]
    
    for u, v, r_id, length, lanes, maxsp, onew, r_type in roads:
        graph.add_edge(u, v, r_id, length, lanes, maxsp, onew, r_type)
        
    return graph


if __name__ == "__main__":
    print("=========================================================")
    print("🗺️ MODULE 3: OSM ROAD GRAPH DATABASE INITIALIZATION")
    print("=========================================================")
    
    osm_graph = build_sample_madrid_osm_graph()
    stats = osm_graph.get_graph_stats()
    
    print(f"[*] Nodes (Intersections) Inserted: {stats['total_nodes']}")
    print(f"[*] Edges (Road Segments) Inserted: {stats['total_directed_edges']}")
    print(f"[*] Total Road Network Length:      {stats['total_road_length_km']} km")
    
    print("\n[*] Inspecting Sample Edge (N1 -> N2):")
    sample_edge = osm_graph.get_edge_attributes("N1", "N2")
    for key, val in sample_edge.items():
        print(f"    - {key}: {val}")
        
    free_flow_time = osm_graph.calculate_free_flow_time("N1", "N2")
    print(f"    - Base Free-Flow Time: {free_flow_time:.1f} seconds")
    
    import os
    base_models = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "models")
    os.makedirs(base_models, exist_ok=True)
    output_path = os.path.join(base_models, "module3_osm_graph.json")
    osm_graph.export_to_json(output_path)
    print(f"\n[+] Successfully exported OSM Graph Database to: {output_path}")
    print("=========================================================")
