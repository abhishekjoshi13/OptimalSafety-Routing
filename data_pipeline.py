import os
import gzip
import shutil
import joblib
import pandas as pd
import networkx as nx
import osmnx as osmx

def decompress_graph_data(gz_path: str, target_path: str) -> None:
    """Decompresses the GraphML file if not already present in the environment."""
    if not os.path.exists(target_path):
        with gzip.open(gz_path, "rb") as f_in:
            with open(target_path, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)

def build_safety_graph(graph_path: str, model_path: str) -> nx.MultiDiGraph:
    """
    Loads OSMnx graph, extracts CPTED features, runs Random Forest inference, 
    and applies dynamic risk penalties to edge weights.
    """
    decompress_graph_data(f"{graph_path}.gz", graph_path)
    
    graph = osmx.load_graphml(graph_path)
    model = joblib.load(model_path)
    
    edges_data = []
    edges_order = []
    
    # Extract structural features for the ML model
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
    
    # Optimization Layer: Apply danger multipliers to graph edge weights
    for i, (u, v, k) in enumerate(edges_order):
        prediction = predictions[i]
        multiplier = 1.0
        
        if prediction == "Moderate":
            multiplier = 2.5
        elif prediction == "High Risk":
            multiplier = 6.0
            
        graph[u][v][k]["safety_cost"] = edges_data[i]["length"] * multiplier
        graph[u][v][k]["risk_class"] = prediction
        
    return graph
