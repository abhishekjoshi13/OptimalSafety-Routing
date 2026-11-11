import streamlit as st
import networkx as nx
import osmnx as osmx
from streamlit_folium import st_folium
from pydantic import ValidationError
import logging

# Importing our custom backend modules
from data_pipeline import build_safety_graph
from schemas import RouteRequestSchema

st.set_page_config(page_title="Optimal Safety Routing", layout="wide")
st.title("Optimal Safety Routing Engine")
st.subheader("AI-Driven Pedestrian Navigation")

@st.cache_resource
def load_backend_graph():
    return build_safety_graph("city_map.graphml", "safety_model.joblib")

try:
    city_graph = load_backend_graph()
    nodes = list(city_graph.nodes())
except Exception as e:
    st.error(f"Failed to boot routing engine: {str(e)}")
    st.stop()

st.sidebar.header("Navigation Parameters")
start_index = st.sidebar.number_input("Start Node Index", min_value=0, value=10)
end_index = st.sidebar.number_input("Destination Node Index", min_value=0, value=150)

if "route_generated" not in st.session_state:
    st.session_state.route_generated = False

if st.sidebar.button("Generate Routes"):
    st.session_state.route_generated = True

if st.session_state.route_generated:
    try:
        # Strict Input Sanitization using Pydantic
        validated_request = RouteRequestSchema(
            start_node_idx=start_index,
            end_node_idx=end_index,
            total_nodes=len(nodes)
        )
        
        start_node = nodes[validated_request.start_node_idx]
        end_node = nodes[validated_request.end_node_idx]
        
        # Calculate optimal paths via Dijkstra
        fastest_path = nx.shortest_path(city_graph, start_node, end_node, weight="length")
        safest_path = nx.shortest_path(city_graph, start_node, end_node, weight="safety_cost")
        
        fastest_len = sum(city_graph[u][v][0]["length"] for u, v in zip(fastest_path[:-1], fastest_path[1:]))
        safest_len = sum(city_graph[u][v][0]["length"] for u, v in zip(safest_path[:-1], safest_path[1:]))
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Shortest Route Distance", f"{fastest_len:.1f} meters")
        with col2:
            st.metric("Safest Route Distance", f"{safest_len:.1f} meters", 
                      delta=f"+{safest_len - fastest_len:.1f}m Safety Detour", delta_color="inverse")
            
        fastest_edges = osmx.routing.route_to_gdf(city_graph, fastest_path, weight="length")
        safest_edges = osmx.routing.route_to_gdf(city_graph, safest_path, weight="safety_cost")
        
        map_visual = fastest_edges.explore(color="red", style_kwds={"weight": 4, "opacity": 0.6})
        map_visual = safest_edges.explore(m=map_visual, color="green", style_kwds={"weight": 6, "opacity": 0.9})
        
        st_folium(map_visual, width=1100, height=500)
        st.info("Red: Shortest Path (Optimized for Distance) | Green: Safest Path (Optimized via AI Penalty Multipliers)")

    except ValidationError as ve:
        logging.error(f"Input validation failed: {ve}")
        st.sidebar.error("Security Alert: Invalid node indices. Potential out-of-bound traversal blocked.")
    except nx.NetworkXNoPath:
        st.sidebar.error("No valid path exists between the selected locations.")

st.markdown("---")
st.caption("Legal Disclaimer: This routing engine provides AI-based predictions utilizing physical infrastructure data. Real-world conditions are inherently unpredictable, and this tool does not guarantee absolute safety. Users must always exercise personal judgment and situational awareness. The developers assume no legal liability for incidents occurring on generated routes.")
        
