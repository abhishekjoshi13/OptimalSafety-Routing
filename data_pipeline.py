import joblib
import pandas as pd
import networkx as nx
import osmnx as osmx

# --- OSMNX CONFIGURATION FOR STABILITY ---
osmx.settings.timeout = 300  # API ko data fetch karne ke liye 5 mins ka time do
osmx.settings.use_cache = True  # Ek baar data aa jaye toh save kar lega
# ---------------------------------------

def build_dynamic_safety_graph(city_name: str, model_path: str) -> nx.MultiDiGraph:
    """
    Dynamically fetches city road network, extracts CPTED features, 
    runs RF inference, and stores metrics for tradeoff analysis.
    """
    # Fetch graph dynamically via OSM API
    graph = osmx.graph_from_place(city_name, network_type="walk")
    model = joblib.load(model_path)
    
    edges_data = []
    edges_order = []
    
    # Feature extraction (CPTED Heuristics)
    for u, v, k, edge_attr in graph.edges(keys=True, data=True):
        road_length = float(edge_attr.get("length", 100.0))
        highway = str(edge_attr.get("highway", "unclassified"))
        
        if highway in ["primary", "secondary", "trunk"]:
            lux, traffic, shops, police = 85.0, 200, 8, 1.0
        elif highway in ["tertiary", "residential"]:
            lux, traffic, shops, police = 45.0, 50, 2, 3.0
        else:
            lux, traffic, shops, police = 15.0, 10, 0, 5.0
            
        edges_data.append({
            "length": road_length, 
            "laneCount": 1, 
            "luxLevel": lux, 
            "pedestrianTraffic": traffic, 
            "shopCount": shops, 
            "policeDist": police
        })
        edges_order.append((u, v, k))
        
    # Batch ML inference
    features_df = pd.DataFrame(edges_data)
    predictions = model.predict(features_df)
    
    # Apply dynamic risk penalties and store metrics for UI Analysis
    for i, (u, v, k) in enumerate(edges_order):
        prediction = predictions[i]
        multiplier = 1.0
        
        if prediction == "Moderate":
            multiplier = 2.5
        elif prediction == "High Risk":
            multiplier = 6.0
            
        graph[u][v][k]["safety_cost"] = edges_data[i]["length"] * multiplier
        graph[u][v][k]["risk_class"] = prediction
        
        # Saving raw features for the Tradeoff Analytics Engine
        graph[u][v][k]["lux_level"] = edges_data[i]["luxLevel"]
        graph[u][v][k]["police_dist"] = edges_data[i]["policeDist"]
        
    return graph

def extract_path_metrics(graph: nx.MultiDiGraph, path: list) -> dict:
    """Extracts aggregated CPTED metrics across a given path."""
    total_dist = 0
    high_risk_segments = 0
    total_lux = 0
    edge_count = 0
    
    for u, v in zip(path[:-1], path[1:]):
        edge = graph[u][v][0]
        total_dist += edge["length"]
        total_lux += edge["lux_level"]
        if edge["risk_class"] == "High Risk":
            high_risk_segments += 1
        edge_count += 1
        
    avg_lux = total_lux / edge_count if edge_count > 0 else 0
    
    return {
        "distance": total_dist,
        "high_risk_count": high_risk_segments,
        "avg_lux": avg_lux
    }
       
       
