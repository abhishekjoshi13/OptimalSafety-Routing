import streamlit as st
import osmnx as osmx
import networkx as nx
import joblib
import pandas as pd
import gzip
import shutil
import os
from streamlit_folium import st_folium
st.set_page_config(page_title="Optimal Safety Routing", layout="wide")
st.title("Optimal Safety Routing Engine")
st.subheader("AI-Driven Pedestrian Navigation")
@st.cache_resource
def load_system():
    if not os.path.exists("city_map.graphml"):
        with gzip.open("city_map.graphml.gz", "rb") as f_in:
            with open("city_map.graphml", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
    graph = osmx.load_graphml("city_map.graphml")
    model = joblib.load("safety_model.joblib")
    edges_data = []
    edges_order = []
    for u, v, k, edge_attr in graph.edges(keys=True, data=True):
        road_length = float(edge_attr.get("length", 100.0))
        highway = str(edge_attr.get("highway", "unclassified"))
        if highway in ["primary", "secondary", "trunk"]:
            lux, traffic, shops, police = 85.0, 200, 8, 1.0
        elif highway in ["tertiary", "residential"]:
            lux, traffic, shops, police = 45.0, 50, 2, 3.0
        else:
            lux, traffic, shops, police = 15.0, 10, 0, 5.0
        edges_data.append({"length": road_length, "laneCount": 1, "luxLevel": lux, "pedestrianTraffic": traffic, "shopCount": shops, "policeDist": police})
        edges_order.append((u, v, k))
    features_df = pd.DataFrame(edges_data)
    predictions = model.predict(features_df)
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
city_graph = load_system()
nodes = list(city_graph.nodes())
st.sidebar.header("Navigation Parameters")
start_index = st.sidebar.number_input("Start Node Index", min_value=0, max_value=len(nodes)-1, value=10)
end_index = st.sidebar.number_input("Destination Node Index", min_value=0, max_value=len(nodes)-1, value=150)
if st.sidebar.button("Generate Routes"):
    start_node = nodes[start_index]
    end_node = nodes[end_index]
    fastest_path = nx.shortest_path(city_graph, start_node, end_node, weight="length")
    safest_path = nx.shortest_path(city_graph, start_node, end_node, weight="safety_cost")
    fastest_len = sum(city_graph[u][v][0]["length"] for u, v in zip(fastest_path[:-1], fastest_path[1:]))
    safest_len = sum(city_graph[u][v][0]["length"] for u, v in zip(safest_path[:-1], safest_path[1:]))
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Shortest Route Distance", f"{fastest_len:.1f} meters")
    with col2:
        st.metric("Safest Route Distance", f"{safest_len:.1f} meters", delta=f"+{safest_len - fastest_len:.1f}m Safety Detour", delta_color="inverse")
    fastest_edges = osmx.routing.route_to_gdf(city_graph, fastest_path, weight="length")
    safest_edges = osmx.routing.route_to_gdf(city_graph, safest_path, weight="safety_cost")
    map_visual = fastest_edges.explore(color="red", style_kwds={"weight": 4, "opacity": 0.6})
    map_visual = safest_edges.explore(m=map_visual, color="green", style_kwds={"weight": 6, "opacity": 0.9})
    st_folium(map_visual, width=1100, height=500)
    st.info("Red: Shortest Path | Green: AI Safest Path")
st.markdown("---")
st.caption("Legal Disclaimer: This routing engine provides AI-based predictions utilizing physical infrastructure data. Real-world conditions are inherently unpredictable, and this tool does not guarantee absolute safety. Users must always exercise personal judgment and situational awareness. The developers assume no legal liability for incidents occurring on generated routes.")
