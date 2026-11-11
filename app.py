import streamlit as st
import networkx as nx
import osmnx as osmx
import folium
from streamlit_folium import st_folium
from pydantic import ValidationError

from data_pipeline import build_dynamic_safety_graph, extract_path_metrics
from schemas import LocationQuerySchema

st.set_page_config(page_title="Optimal Safety Routing", layout="wide")
st.title("Optimal Safety Routing Engine")
st.subheader("AI-Driven Pedestrian Navigation & CPTED Tradeoff Analytics")

# --- UI State Management ---
if "city_graph" not in st.session_state:
    st.session_state.city_graph = None
if "current_city" not in st.session_state:
    st.session_state.current_city = ""

# --- Sidebar: Location Parameters ---
st.sidebar.header("1. Initialize City Region")
city_input = st.sidebar.text_input("Enter City (e.g., Ranchi, India)", value="Ranchi, India")

if st.sidebar.button("Load City Graph"):
    with st.spinner(f"Fetching mapping data for {city_input} and running ML inference..."):
        try:
            st.session_state.city_graph = build_dynamic_safety_graph(city_input, "safety_model.joblib")
            st.session_state.current_city = city_input
            st.sidebar.success(f"{city_input} loaded and optimized!")
        except Exception as e:
            st.sidebar.error("Failed to fetch city. Please check spelling or network.")

st.sidebar.markdown("---")
st.sidebar.header("2. Navigation Points")
start_input = st.sidebar.text_input("Start Location (e.g., BIT Mesra)")
end_input = st.sidebar.text_input("Destination (e.g., Lalpur)")

# --- Routing Execution ---
if st.sidebar.button("Generate Secure Route"):
    if st.session_state.city_graph is None:
        st.warning("Please load a city graph first from the sidebar.")
        st.stop()
        
    try:
        # 1. Input Validation
        query = LocationQuerySchema(
            city_name=st.session_state.current_city,
            start_address=start_input,
            end_address=end_input
        )
        
        with st.spinner("Geocoding addresses to coordinates..."):
            # 2. Geocoding (Text to Lat/Lng)
            start_coords = osmx.geocode(f"{query.start_address}, {query.city_name}")
            end_coords = osmx.geocode(f"{query.end_address}, {query.city_name}")
            
            # 3. Find Nearest Graph Nodes
            start_node = osmx.distance.nearest_nodes(st.session_state.city_graph, X=start_coords[1], Y=start_coords[0])
            end_node = osmx.distance.nearest_nodes(st.session_state.city_graph, X=end_coords[1], Y=end_coords[0])
            
        with st.spinner("Calculating optimal paths via Dijkstra..."):
            # 4. Pathfinding
            fastest_path = nx.shortest_path(st.session_state.city_graph, start_node, end_node, weight="length")
            safest_path = nx.shortest_path(st.session_state.city_graph, start_node, end_node, weight="safety_cost")
            
            # 5. Extract Analytics
            fast_metrics = extract_path_metrics(st.session_state.city_graph, fastest_path)
            safe_metrics = extract_path_metrics(st.session_state.city_graph, safest_path)
            
        # --- UI Rendering: Tradeoff Analytics ---
        st.markdown("### Route Trade-off Analysis")
        col1, col2, col3 = st.columns(3)
        
        dist_diff = safe_metrics['distance'] - fast_metrics['distance']
        lux_diff = safe_metrics['avg_lux'] - fast_metrics['avg_lux']
        risk_diff = safe_metrics['high_risk_count'] - fast_metrics['high_risk_count']
        
        with col1:
            st.metric("Safe Route Distance", f"{safe_metrics['distance']:.1f} m", 
                      delta=f"{dist_diff:.1f} m vs Shortest", delta_color="inverse")
        with col2:
            st.metric("Avg Street Illumination", f"{safe_metrics['avg_lux']:.1f} Lux", 
                      delta=f"{lux_diff:.1f} Lux vs Shortest", delta_color="normal")
        with col3:
            st.metric("High-Risk Segments", f"{safe_metrics['high_risk_count']}", 
                      delta=f"{risk_diff} vs Shortest", delta_color="inverse")
            
        # Justification Text
        st.info(f"**Analytics Engine Recommendation:** The AI recommends the Safe Route. Although it adds **{dist_diff:.1f} meters** to your walk, it completely avoids/reduces High-Risk unlit zones by **{abs(risk_diff)} segments** and improves overall visibility by **{abs(lux_diff):.1f} Lux**, aligning with optimal CPTED safety standards.")

        # --- Interactive Map Generation ---
        fastest_edges = osmx.routing.route_to_gdf(st.session_state.city_graph, fastest_path, weight="length")
        safest_edges = osmx.routing.route_to_gdf(st.session_state.city_graph, safest_path, weight="safety_cost")
        
        route_map = fastest_edges.explore(color="red", style_kwds={"weight": 4, "opacity": 0.6}, name="Shortest Path")
        route_map = safest_edges.explore(m=route_map, color="green", style_kwds={"weight": 6, "opacity": 0.9}, name="Safest Path")
        
        # Add G-Maps style Origin and Destination Markers
        folium.Marker(location=start_coords, popup="Origin", icon=folium.Icon(color="blue", icon="play")).add_to(route_map)
        folium.Marker(location=end_coords, popup="Destination", icon=folium.Icon(color="red", icon="stop")).add_to(route_map)
        
        st_folium(route_map, width=1100, height=500)
        
    except ValidationError:
        st.error("Invalid text input. Please enter valid locations.")
    except Exception as e:
        st.error(f"Routing failed: Please ensure locations are specific to {st.session_state.current_city}.")

st.markdown("---")
st.caption("Legal Disclaimer: AI predictions are based on physical infrastructure data. Users must exercise personal judgment.")
