
import logging
from collections import Counter
from itertools import combinations
from pathlib import Path

import folium
import networkx as nx
import osmnx as ox
import streamlit as st
from pydantic import ValidationError
from streamlit_folium import st_folium

from data_pipeline import build_dynamic_safety_graph
from routing_helpers import (
    MAX_SNAP_DISTANCE_METERS,
    LocationOutsideGraphError,
    LocationTooFarFromRoadError,
    calculate_routes,
    create_route_map,
    snap_location_to_graph,
    validate_coordinates,
)
from schemas import LocationQuerySchema


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "safety_model.joblib"


def _clean_location_text(value):
    return " ".join(value.strip().split())


def geocode_location(address, city_name):
    address = _clean_location_text(address)
    city_name = _clean_location_text(city_name)

    city_lower = city_name.lower()
    address_lower = address.lower()

    candidates = []

    if "ranchi" in city_lower or "ranchi" in address_lower:
        candidates.extend([
            f"{address}, Lalpur, Ranchi, Jharkhand, India",
            f"{address}, Ranchi, Jharkhand, India",
            f"{address}, {city_name}",
            address
        ])
    else:
        candidates.extend([
            f"{address}, {city_name}, India",
            f"{address}, {city_name}",
            address
        ])

    unique_candidates = []
    seen = set()

    for candidate in candidates:
        candidate = _clean_location_text(candidate)
        normalized = candidate.lower()

        if normalized not in seen:
            seen.add(normalized)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        try:
            logger.info(
                "Trying geocoding query: %s",
                candidate
            )

            coordinates = ox.geocode(candidate)
            coordinates = validate_coordinates(coordinates)

            logger.info(
                "Geocoded '%s' to coordinates: latitude=%s, longitude=%s",
                candidate,
                coordinates[0],
                coordinates[1]
            )

            return coordinates

        except Exception as exc:
            logger.warning(
                "Geocoding attempt failed for '%s': %s",
                candidate,
                exc
            )

    raise ValueError(
        f"Could not locate '{address}'. "
        "Please try a more specific location name or nearby landmark."
    )


def find_different_route_pair(graph, max_sample_nodes=30):
    """
    Inspect a bounded, geographically distributed sample of graph nodes.
    Return a pair for which shortest and safety-weighted paths differ.
    """

    if graph is None or graph.number_of_nodes() < 2:
        return None

    valid_nodes = []

    for node, data in graph.nodes(data=True):
        try:
            latitude = float(data["y"])
            longitude = float(data["x"])
        except (KeyError, TypeError, ValueError):
            continue

        valid_nodes.append(
            (node, latitude, longitude)
        )

    if len(valid_nodes) < 2:
        return None

    valid_nodes.sort(
        key=lambda item: (item[1], item[2])
    )

    sample_count = min(
        max_sample_nodes,
        len(valid_nodes)
    )

    if sample_count < len(valid_nodes):
        indices = [
            round(
                index * (len(valid_nodes) - 1)
                / (sample_count - 1)
            )
            for index in range(sample_count)
        ]
        sampled_nodes = [
            valid_nodes[index]
            for index in indices
        ]
    else:
        sampled_nodes = valid_nodes

    best_candidate = None
    best_score = None

    for start_info, end_info in combinations(sampled_nodes, 2):
        start_node = start_info[0]
        end_node = end_info[0]

        try:
            routes = calculate_routes(
                graph,
                start_node,
                end_node
            )
        except (
            nx.NetworkXNoPath,
            nx.NodeNotFound,
            ValueError
        ):
            continue

        shortest_path = routes["shortest_path"]
        safety_path = routes["safety_path"]

        if shortest_path == safety_path:
            continue

        shortest_nodes = set(shortest_path)
        safety_nodes = set(safety_path)

        path_difference = len(
            shortest_nodes.symmetric_difference(safety_nodes)
        )

        risk_difference = abs(
            routes["shortest_metrics"]["high_risk_count"]
            - routes["safety_metrics"]["high_risk_count"]
        )

        shortest_distance = routes["shortest_metrics"]["distance"]
        safety_distance = routes["safety_metrics"]["distance"]

        extra_distance = (
            safety_distance - shortest_distance
        )

        score = (
            path_difference,
            risk_difference,
            -extra_distance
        )

        if best_score is None or score > best_score:
            best_score = score

            best_candidate = {
                **routes,
                "start_node": start_node,
                "end_node": end_node,
                "start_coords": (
                    start_info[1],
                    start_info[2]
                ),
                "end_coords": (
                    end_info[1],
                    end_info[2]
                )
            }

    return best_candidate


st.set_page_config(
    page_title="Optimal Safety Routing",
    layout="wide"
)

st.title("Optimal Safety Routing Engine")

st.subheader(
    "AI-Driven Pedestrian Navigation & CPTED Trade-off Analytics"
)


# Initialize application state

session_defaults = {
    "city_graph": None,
    "current_city": "",
    "route_result": None,
    "route_error": None,
    "diagnostic_message": None
}

for key, default_value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_value


# Sidebar: Load city graph

st.sidebar.header("1. Initialize City Region")

city_input = st.sidebar.text_input(
    "Enter City Area (e.g., Lalpur, Ranchi)",
    value="Lalpur, Ranchi"
)

if st.sidebar.button("Load Graph (1.5km Radius)"):

    st.session_state.city_graph = None
    st.session_state.current_city = ""
    st.session_state.route_result = None
    st.session_state.route_error = None
    st.session_state.diagnostic_message = None

    city_input = _clean_location_text(city_input)

    if not 2 <= len(city_input) <= 120:
        st.sidebar.error(
            "Please enter a valid city or area name."
        )

    else:
        with st.spinner(
            "Fetching mapping data and running ML inference..."
        ):
            try:
                graph = build_dynamic_safety_graph(
                    city_input,
                    str(MODEL_PATH)
                )

                if (
                    graph is None
                    or graph.number_of_nodes() == 0
                    or graph.number_of_edges() == 0
                ):
                    raise ValueError(
                        "The generated road graph is empty."
                    )

                from routing_helpers import get_graph_bounds

                get_graph_bounds(graph)

                st.session_state.city_graph = graph
                st.session_state.current_city = city_input

            except Exception:
                logger.exception(
                    "City graph initialization failed"
                )

                st.sidebar.error(
                    "Unable to load this area. Please verify "
                    "the location and try again."
                )


if st.session_state.city_graph is not None:

    st.sidebar.success(
        f"Loaded region: {st.session_state.current_city}"
    )

    st.sidebar.caption(
        f"Maximum location snapping distance: "
        f"{MAX_SNAP_DISTANCE_METERS:.0f} metres"
    )


# Sidebar: Route inputs

st.sidebar.markdown("---")
st.sidebar.header("2. Navigation Points")

if st.session_state.city_graph is not None:

    with st.sidebar.form("route_form"):

        start_input = st.text_input(
            "Start Location",
            value="Nucleus Mall"
        )

        end_input = st.text_input(
            "Destination",
            value="Lalpur Chowk"
        )

        generate_route = st.form_submit_button(
            "Generate Routes"
        )

    if generate_route:

        st.session_state.route_result = None
        st.session_state.route_error = None
        st.session_state.diagnostic_message = None

        try:
            query = LocationQuerySchema(
                city_name=st.session_state.current_city,
                start_address=start_input,
                end_address=end_input
            )

            with st.spinner("Finding both locations..."):

                start_coords = geocode_location(
                    query.start_address,
                    query.city_name
                )

                end_coords = geocode_location(
                    query.end_address,
                    query.city_name
                )

                graph = st.session_state.city_graph

                start_node, start_snap_distance = (
                    snap_location_to_graph(
                        graph,
                        start_coords,
                        "Starting location"
                    )
                )

                end_node, end_snap_distance = (
                    snap_location_to_graph(
                        graph,
                        end_coords,
                        "Destination"
                    )
                )

            if start_node == end_node:
                raise ValueError(
                    "The start and destination resolve to the "
                    "same road-network node. Please choose "
                    "different locations."
                )

            with st.spinner("Calculating optimal routes..."):

                calculated_routes = calculate_routes(
                    graph,
                    start_node,
                    end_node
                )

            st.session_state.route_result = {
                **calculated_routes,
                "start_coords": start_coords,
                "end_coords": end_coords,
                "start_address": query.start_address,
                "end_address": query.end_address,
                "city_name": query.city_name,
                "start_snap_distance": start_snap_distance,
                "end_snap_distance": end_snap_distance
            }

        except ValidationError:
            st.session_state.route_error = (
                "Invalid input. Please provide valid start, "
                "destination and city names."
            )

        except LocationOutsideGraphError as exc:
            st.session_state.route_error = str(exc)

        except LocationTooFarFromRoadError as exc:
            st.session_state.route_error = str(exc)

        except nx.NetworkXNoPath:
            st.session_state.route_error = (
                "No walkable path was found between these locations."
            )

        except nx.NodeNotFound:
            st.session_state.route_error = (
                "A selected location could not be matched to "
                "the loaded road network."
            )

        except ValueError as exc:
            st.session_state.route_error = str(exc)

        except Exception:
            logger.exception("Route generation failed")

            st.session_state.route_error = (
                "Route generation failed due to an unexpected error. "
                "Please check the application logs."
            )


else:
    st.sidebar.info(
        "Load a city graph before generating routes."
    )


# Route diagnostics

st.markdown("---")
st.markdown("## 3. Route Testing / Diagnostics")

if st.session_state.city_graph is not None:

    graph = st.session_state.city_graph

    st.write(
        "Automatically inspect selected points inside the loaded "
        "road network and search for a pair where the shortest "
        "and safety-weighted paths are different."
    )

    risk_counts = Counter(
        str(data.get("risk_class", "Unknown"))
        for _, _, _, data in graph.edges(
            keys=True,
            data=True
        )
    )

    st.markdown("**Model risk-class distribution in the loaded graph**")

    risk_columns = st.columns(max(1, len(risk_counts)))

    for index, (risk_class, count) in enumerate(
        sorted(risk_counts.items())
    ):
        with risk_columns[index]:
            st.metric(
                risk_class,
                count
            )

    if st.button("Find a Route with Different Paths"):

        st.session_state.route_result = None
        st.session_state.route_error = None
        st.session_state.diagnostic_message = None

        with st.spinner(
            "Inspecting the loaded graph and comparing candidate routes..."
        ):
            try:
                candidate = find_different_route_pair(graph)

                if candidate is None:
                    st.session_state.diagnostic_message = (
                        "No different shortest and safety-weighted "
                        "paths were found among the sampled nodes. "
                        "This may mean the current graph has similar "
                        "risk costs across its roads, or the sampled "
                        "pairs do not have useful alternative paths."
                    )

                else:
                    st.session_state.route_result = {
                        **candidate,
                        "start_address": (
                            f"Diagnostic graph node "
                            f"{candidate['start_node']}"
                        ),
                        "end_address": (
                            f"Diagnostic graph node "
                            f"{candidate['end_node']}"
                        ),
                        "city_name": st.session_state.current_city,
                        "start_snap_distance": 0.0,
                        "end_snap_distance": 0.0
                    }

                    st.session_state.diagnostic_message = (
                        "A pair with different shortest and "
                        "safety-weighted paths was found."
                    )

            except Exception:
                logger.exception(
                    "Route diagnostics failed"
                )

                st.session_state.diagnostic_message = (
                    "Diagnostics could not complete. Please check "
                    "the application logs."
                )

    if st.session_state.diagnostic_message:
        if st.session_state.route_result is not None:
            st.success(
                st.session_state.diagnostic_message
            )
        else:
            st.warning(
                st.session_state.diagnostic_message
            )

else:
    st.info(
        "Load a city graph first to use route diagnostics."
    )


# Main page: Display errors

if st.session_state.route_error:
    st.error(st.session_state.route_error)


# Main page: Display generated routes

route_result = st.session_state.route_result

if (
    route_result is not None
    and route_result["city_name"] == st.session_state.current_city
    and st.session_state.city_graph is not None
):

    shortest_metrics = route_result["shortest_metrics"]
    safety_metrics = route_result["safety_metrics"]

    st.markdown("### Route Trade-off Analysis")

    st.caption(
        f"Route: {route_result['start_address']} → "
        f"{route_result['end_address']} "
        f"({route_result['city_name']})"
    )

    col1, col2, col3 = st.columns(3)

    distance_difference = (
        safety_metrics["distance"]
        - shortest_metrics["distance"]
    )

    lighting_difference = (
        safety_metrics["avg_lux"]
        - shortest_metrics["avg_lux"]
    )

    risk_difference = (
        safety_metrics["high_risk_count"]
        - shortest_metrics["high_risk_count"]
    )

    with col1:
        st.metric(
            "Shortest Route Distance",
            f"{shortest_metrics['distance']:.1f} m"
        )

        st.metric(
            "Safety-Weighted Route Distance",
            f"{safety_metrics['distance']:.1f} m",
            delta=f"{distance_difference:+.1f} m vs Shortest",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            "Shortest Route Lighting Proxy",
            f"{shortest_metrics['avg_lux']:.1f}"
        )

        st.metric(
            "Safety Route Lighting Proxy",
            f"{safety_metrics['avg_lux']:.1f}",
            delta=f"{lighting_difference:+.1f} vs Shortest",
            delta_color="normal"
        )

    with col3:
        st.metric(
            "Shortest Route High-Risk Segments",
            shortest_metrics["high_risk_count"]
        )

        st.metric(
            "Safety Route High-Risk Segments",
            safety_metrics["high_risk_count"],
            delta=f"{risk_difference:+d} vs Shortest",
            delta_color="inverse"
        )

    st.info(
        "The safety-weighted route minimizes the model-based "
        "safety cost assigned to road segments. "
        f"Its distance differs by {distance_difference:+.1f} metres "
        "from the shortest route, and it has "
        f"{risk_difference:+d} high-risk segments compared "
        "with that route."
    )

    st.caption(
        "Location snapping distances: "
        f"Start: {route_result['start_snap_distance']:.1f} m | "
        f"Destination: {route_result['end_snap_distance']:.1f} m"
    )

    st.caption(
        "Important: Lighting values and other CPTED features "
        "are rule-based proxies derived from road classifications. "
        "They are not verified real-world measurements or "
        "guarantees of personal safety."
    )

    st.markdown("### Route Visualization")

    try:
        route_map = create_route_map(
            st.session_state.city_graph,
            route_result
        )

        st_folium(
            route_map,
            width=1100,
            height=500
        )

    except Exception:
        logger.exception("Route map rendering failed")

        st.warning(
            "Routes were calculated successfully, but the map "
            "could not be displayed. Please try again."
        )


st.markdown("---")

st.caption(
    "Legal Disclaimer: Route predictions are model-based estimates "
    "using available road-network data. They do not guarantee safety. "
    "Users should exercise personal judgment."
)





   

  

    









          

           
      
