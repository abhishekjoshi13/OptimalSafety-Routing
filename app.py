import streamlit as st
import networkx as nx
import osmnx as osmx
import folium
from streamlit_folium import st_folium
from pydantic import ValidationError

# Correctly importing the updated function names
from data_pipeline import build_dynamic_safety_graph, extract_path_metrics
from schemas import LocationQuerySchema

st.set_page_config(page_title="Optimal Safety Routing", layout="wide")
st.title("Optimal Safety Routing Engine")
st.subheader("AI-Driven Pedestrian Navigation & CPTED Tradeoff Analytics")

if "city_graph" not in st.session_state:
    st.session_state.city_graph = None
if "current_city" not in st.session_state:
    st.session_state.current_city = ""
if "route_generated" not in st.session_state:
    st.session_state.route_generated = False

st.sidebar.header("1. Initialize City Region")
city_input = st.sidebar.text_input("Enter City Area (e.g., Lalpur, Ranchi)", value="Lalpur, Ranchi")

if st.sidebar.button("Load Graph (1.5km Radius)"):
    with st.spinner(f"Fetching mapping data for {city_input} and running ML inference..."):
        try:
            st.session_state.city_graph = build_dynamic_safety_graph(city_input, "safety_model.joblib")
            st.session_state.current_city = city_input
            st.session_state.route_generated = False 
            st.sidebar.success(f"{city_input} loaded and optimized!")
        except Exception as e:
            st.sidebar.error(f"Failed to fetch area. Error Details: {str(e)}")

st.sidebar.markdown("---")
st.sidebar.header("2. Navigation Points")
start_input = st.sidebar.text_input("Start Location (e.g., Nucleus Mall)")
end_input = st.sidebar.text_input("Destination (e.g., Lalpur Chowk)")

if st.sidebar.button("Generate Secure Route"):
    st.session_state.route_generated = True

if st.session_state.route_generated:
    if st.session_state.city_graph is None:
        st.warning("Please load a city graph first from the sidebar.")
        st.stop()
        
    try:
        query = LocationQuerySchema(
            city_name=st.session_state.current_city,
            start_address=start_input,
            end_address=end_input
        )
        
        with st.spinner("Geocoding addresses to coordinates..."):
            start_coords = osmx.geocode(f"{query.start_address}, {query.city_name}")
            end_coords = osmx.geocode(f"{query.end_address}, {query.city_name}")
            
            start_node = osmx.distance.nearest_nodes(st.session_state.city_graph, X=start_coords[1], Y=start_coords[0])
            end_node = osmx.distance.nearest_nodes(st.session_state.city_graph, X=end_coords[1], Y=end_coords[0])
            
        with st.spinner("Calculating optimal paths via Dijkstra..."):
            fastest_path = nx.shortest_path(st.session_state.city_graph, start_node, end_node, weight="length")
            safest_path = nx.shortest_path(st.session_state.city_graph, start_node, end_node, weight="safety_cost")
            
            fast_metrics = extract_path_metrics(st.session_state.city_graph, fastest_path)
            safe_metrics = extract_path_metrics(st.session_state.city_graph, safest_path)
            
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
            
        st.info(f"**Analytics Engine Recommendation:** The AI recommends the Safe Route. Although it adds **{dist_diff:.1f} meters** to your walk, it avoids/reduces High-Risk zones by **{abs(risk_diff)} segments** and improves overall visibility by **{abs(lux_diff):.1f} Lux**, aligning with optimal CPTED safety standards.")

        fastest_edges = osmx.routing.route_to_gdf(st.session_state.city_graph, fastest_path, weight="length")
        safest_edges = osmx.routing.route_to_gdf(st.session_state.city_graph, safest_path, weight="safety_cost")
        
        route_map = fastest_edges.explore(color="red", style_kwds={"weight": 4, "opacity": 0.6}, name="Shortest Path")
        route_map = safest_edges.explore(m=route_map, color="green", style_kwds={"weight": 6, "opacity": 0.9}, name="Safest Path")
        
        folium.Marker(location=start_coords, popup="Origin", icon=folium.Icon(color="blue", icon="play")).add_to(route_map)
        folium.Marker(location=end_coords, popup="Destination", icon=folium.Icon(color="red", icon="stop")).add_to(route_map)
        
        st_folium(route_map, width=1100, height=500)
        
    except ValidationError:
        st.error("Invalid text input. Please enter valid locations.")
    except nx.NetworkXNoPath:
        st.error("Routing failed: No walkable path found between these two locations.")
    except Exception as e:
        st.error(f"Routing Error: {str(e)}. Please ensure the locations exist within {st.session_state.current_city}.")

st.markdown("---")
st.caption("Legal Disclaimer: AI predictions are based on physical infrastructure data. Users must exercise personal judgment.")

       
        
