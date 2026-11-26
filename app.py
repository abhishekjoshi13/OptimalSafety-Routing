
import logging
import math
from pathlib import Path

import folium
import networkx as nx
import osmnx as ox
import streamlit as st
from pydantic import ValidationError
from streamlit_folium import st_folium

from data_pipeline import build_dynamic_safety_graph, extract_path_metrics
from schemas import LocationQuerySchema


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "safety_model.joblib"


def validate_coordinates(coords):
    if not isinstance(coords, (tuple, list)) or len(coords) != 2:
        raise ValueError("Geocoding did not return valid coordinates")

    latitude, longitude = coords

    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        raise ValueError("Geocoding returned invalid coordinate values")

    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise ValueError("Geocoding returned out-of-range coordinates")

    return float(latitude), float(longitude)


st.set_page_config(
    page_title="Optimal Safety Routing",
    layout="wide"
)

st.title("Optimal Safety Routing Engine")
st.subheader("AI-Driven Pedestrian Navigation & CPTED Trade-off Analytics")

if "city_graph" not in st.session_state:
    st.session_state.city_graph = None

if "current_city" not in st.session_state:
    st.session_state.current_city = ""

if "route_generated" not in st.session_state:
    st.session_state.route_generated = False


st.sidebar.header("1. Initialize City Region")

city_input = st.sidebar.text_input(
    "Enter City Area (e.g., Lalpur, Ranchi)",
    value="Lalpur, Ranchi"
)

if st.sidebar.button("Load Graph (1.5km Radius)"):
    city_input = city_input.strip()

    if len(city_input) < 2 or len(city_input) > 120:
        st.sidebar.error("Please enter a valid city or area name.")
    else:
        with st.spinner(
            "Fetching mapping data and running ML inference..."
        ):
            try:
                graph = build_dynamic_safety_graph(
                    city_input,
                    str(MODEL_PATH)
                )

                st.session_state.city_graph = graph
                st.session_state.current_city = city_input
                st.session_state.route_generated = False

                st.sidebar.success(
                    f"{city_input} loaded successfully!"
                )

            except Exception:
                logger.exception("City graph initialization failed")
                st.sidebar.error(
                    "Unable to load this area. Please verify the "
                    "location and try again."
                )


st.sidebar.markdown("---")
st.sidebar.header("2. Navigation Points")

start_input = st.sidebar.text_input(
    "Start Location (e.g., Nucleus Mall)"
)

end_input = st.sidebar.text_input(
    "Destination (e.g., Lalpur Chowk)"
)

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

        with st.spinner("Geocoding addresses..."):

            start_coords = ox.geocode(
                f"{query.start_address}, {query.city_name}"
            )

            end_coords = ox.geocode(
                f"{query.end_address}, {query.city_name}"
            )

            start_coords = validate_coordinates(start_coords)
            end_coords = validate_coordinates(end_coords)

            graph = st.session_state.city_graph

            start_node = ox.distance.nearest_nodes(
                graph,
                X=start_coords[1],
                Y=start_coords[0]
            )

            end_node = ox.distance.nearest_nodes(
                graph,
                X=end_coords[1],
                Y=end_coords[0]
            )

        if start_node == end_node:
            st.warning(
                "The selected start and destination resolve to "
                "the same road-network node. Please choose "
                "different locations."
            )
            st.stop()

        with st.spinner("Calculating optimal routes..."):

            fastest_path = nx.shortest_path(
                graph,
                start_node,
                end_node,
                weight="length"
            )

            safest_path = nx.shortest_path(
                graph,
                start_node,
                end_node,
                weight="safety_cost"
            )

            fast_metrics = extract_path_metrics(
                graph,
                fastest_path,
                route_weight="length"
            )

            safe_metrics = extract_path_metrics(
                graph,
                safest_path,
                route_weight="safety_cost"
            )

        st.markdown("### Route Trade-off Analysis")

        col1, col2, col3 = st.columns(3)

        distance_difference = (
            safe_metrics["distance"] - fast_metrics["distance"]
        )

        lighting_difference = (
            safe_metrics["avg_lux"] - fast_metrics["avg_lux"]
        )

        risk_difference = (
            safe_metrics["high_risk_count"]
            - fast_metrics["high_risk_count"]
        )

        with col1:
            st.metric(
                "Safety-Weighted Route Distance",
                f"{safe_metrics['distance']:.1f} m",
                delta=f"{distance_difference:+.1f} m vs Shortest",
                delta_color="inverse"
            )

        with col2:
            st.metric(
                "Average Lighting Proxy",
                f"{safe_metrics['avg_lux']:.1f}",
                delta=f"{lighting_difference:+.1f} vs Shortest",
                delta_color="normal"
            )

        with col3:
            st.metric(
                "High-Risk Segments",
                safe_metrics["high_risk_count"],
                delta=f"{risk_difference:+d} vs Shortest",
                delta_color="inverse"
            )

        st.info(
            "The safety-weighted route is calculated by minimizing "
            "the model-based safety cost assigned to road segments. "
            f"It is {distance_difference:+.1f} metres different in "
            "length from the shortest route and has "
            f"{risk_difference:+d} high-risk segments compared "
            "with that route."
        )

        st.caption(
            "Important: Lighting values and other CPTED features "
            "are rule-based proxies derived from road classifications. "
            "They are not verified real-world measurements or "
            "guarantees of personal safety."
        )

        fastest_edges = ox.routing.route_to_gdf(
            graph,
            fastest_path,
            weight="length"
        )

        safest_edges = ox.routing.route_to_gdf(
            graph,
            safest_path,
            weight="safety_cost"
        )

        route_map = fastest_edges.explore(
            color="red",
            style_kwds={
                "weight": 4,
                "opacity": 0.6
            },
            name="Shortest Path"
        )

        route_map = safest_edges.explore(
            m=route_map,
            color="green",
            style_kwds={
                "weight": 6,
                "opacity": 0.9
            },
            name="Safety-Weighted Path"
        )

        folium.Marker(
            location=start_coords,
            popup="Origin",
            icon=folium.Icon(
                color="blue",
                icon="play"
            )
        ).add_to(route_map)

        folium.Marker(
            location=end_coords,
            popup="Destination",
            icon=folium.Icon(
                color="red",
                icon="stop"
            )
        ).add_to(route_map)

        folium.LayerControl().add_to(route_map)

        st.markdown("### Route Visualization")

        st_folium(
            route_map,
            width=1100,
            height=500
        )

    except ValidationError:
        st.error(
            "Invalid input. Please provide valid start, destination "
            "and city names."
        )

    except nx.NetworkXNoPath:
        st.error(
            "No walkable path was found between these locations."
        )

    except nx.NodeNotFound:
        st.error(
            "A selected location could not be matched to the "
            "loaded road network."
        )

    except Exception:
        logger.exception("Route generation failed")
        st.error(
            "Route generation failed. Please check the locations "
            "and ensure they belong to the loaded area."
        )


st.markdown("---")

st.caption(
    "Legal Disclaimer: Route predictions are model-based estimates "
    "using available road-network data. They do not guarantee safety. "
    "Users should exercise personal judgment."
)
