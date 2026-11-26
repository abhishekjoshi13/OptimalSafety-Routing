
import logging
import math
from numbers import Real
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

MAX_SNAP_DISTANCE_METERS = 250.0
EARTH_RADIUS_METERS = 6371000.0


class LocationOutsideGraphError(ValueError):
    pass


class LocationTooFarFromRoadError(ValueError):
    pass


def validate_coordinates(coords):
    if not isinstance(coords, (tuple, list)) or len(coords) != 2:
        raise ValueError("Geocoding did not return valid coordinates")

    latitude, longitude = coords

    if not isinstance(latitude, Real) or not isinstance(longitude, Real):
        raise ValueError("Geocoding returned invalid coordinate values")

    latitude = float(latitude)
    longitude = float(longitude)

    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise ValueError("Geocoding returned out-of-range coordinates")

    return latitude, longitude


def haversine_distance(lat1, lon1, lat2, lon2):
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    a = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    a = min(1.0, max(0.0, a))

    return (
        2
        * EARTH_RADIUS_METERS
        * math.asin(math.sqrt(a))
    )


def get_graph_bounds(graph):
    if graph is None or graph.number_of_nodes() == 0:
        raise ValueError("The loaded road graph is empty")

    latitudes = []
    longitudes = []

    for _, node_data in graph.nodes(data=True):
        try:
            latitude = float(node_data["y"])
            longitude = float(node_data["x"])
        except (KeyError, TypeError, ValueError):
            continue

        if (
            math.isfinite(latitude)
            and math.isfinite(longitude)
            and -90 <= latitude <= 90
            and -180 <= longitude <= 180
        ):
            latitudes.append(latitude)
            longitudes.append(longitude)

    if not latitudes or not longitudes:
        raise ValueError("The road graph has invalid geographic coordinates")

    return (
        min(latitudes),
        max(latitudes),
        min(longitudes),
        max(longitudes)
    )


def validate_location_in_graph(graph, coords):
    latitude, longitude = coords

    min_lat, max_lat, min_lon, max_lon = get_graph_bounds(graph)

    latitude_margin = MAX_SNAP_DISTANCE_METERS / 111320.0

    middle_latitude = math.radians(
        (min_lat + max_lat) / 2
    )

    longitude_scale = max(
        abs(math.cos(middle_latitude)),
        0.000001
    )

    longitude_margin = (
        MAX_SNAP_DISTANCE_METERS
        / (111320.0 * longitude_scale)
    )

    if not (
        min_lat - latitude_margin
        <= latitude
        <= max_lat + latitude_margin
        and min_lon - longitude_margin
        <= longitude
        <= max_lon + longitude_margin
    ):
        raise LocationOutsideGraphError(
            "This location is outside the loaded road-network region."
        )


def snap_location_to_graph(graph, coords, location_name):
    latitude, longitude = coords

    validate_location_in_graph(graph, coords)

    try:
        nearest_node = ox.distance.nearest_nodes(
            graph,
            X=longitude,
            Y=latitude
        )
    except Exception as exc:
        logger.exception(
            "Nearest road node lookup failed for %s",
            location_name
        )
        raise ValueError(
            f"Could not find a nearby road node for {location_name}."
        ) from exc

    node_data = graph.nodes[nearest_node]

    try:
        node_latitude = float(node_data["y"])
        node_longitude = float(node_data["x"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"The matched road node for {location_name} has invalid coordinates."
        ) from exc

    snap_distance = haversine_distance(
        latitude,
        longitude,
        node_latitude,
        node_longitude
    )

    if snap_distance > MAX_SNAP_DISTANCE_METERS:
        raise LocationTooFarFromRoadError(
            f"{location_name} is approximately "
            f"{snap_distance:.0f} metres from the nearest road node. "
            f"The maximum allowed distance is "
            f"{MAX_SNAP_DISTANCE_METERS:.0f} metres."
        )

    return nearest_node, snap_distance


def calculate_routes(graph, start_node, end_node):
    shortest_path = nx.shortest_path(
        graph,
        start_node,
        end_node,
        weight="length"
    )

    safety_path = nx.shortest_path(
        graph,
        start_node,
        end_node,
        weight="safety_cost"
    )

    shortest_metrics = extract_path_metrics(
        graph,
        shortest_path,
        route_weight="length"
    )

    safety_metrics = extract_path_metrics(
        graph,
        safety_path,
        route_weight="safety_cost"
    )

    return {
        "shortest_path": shortest_path,
        "safety_path": safety_path,
        "shortest_metrics": shortest_metrics,
        "safety_metrics": safety_metrics
    }


def create_route_map(graph, route_result):
    shortest_path = route_result["shortest_path"]
    safety_path = route_result["safety_path"]

    start_coords = route_result["start_coords"]
    end_coords = route_result["end_coords"]

    shortest_edges = ox.routing.route_to_gdf(
        graph,
        shortest_path,
        weight="length"
    )

    safety_edges = ox.routing.route_to_gdf(
        graph,
        safety_path,
        weight="safety_cost"
    )

    route_map = shortest_edges.explore(
        color="red",
        style_kwds={
            "weight": 4,
            "opacity": 0.6
        },
        name="Shortest Path"
    )

    safety_edges.explore(
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

    return route_map


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
    "route_error": None
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

    # Clear the previous graph before attempting a new load.
    st.session_state.city_graph = None
    st.session_state.current_city = ""
    st.session_state.route_result = None
    st.session_state.route_error = None

    city_input = city_input.strip()

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

                get_graph_bounds(graph)

                st.session_state.city_graph = graph
                st.session_state.current_city = city_input

                st.sidebar.success(
                    f"{city_input} loaded successfully!"
                )

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
            "Start Location (e.g., Nucleus Mall)"
        )

        end_input = st.text_input(
            "Destination (e.g., Lalpur Chowk)"
        )

        generate_route = st.form_submit_button(
            "Generate Routes"
        )

    if generate_route:

        st.session_state.route_result = None
        st.session_state.route_error = None

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

                start_coords = validate_coordinates(
                    start_coords
                )

                end_coords = validate_coordinates(
                    end_coords
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

        except Exception:
            logger.exception("Route generation failed")

            st.session_state.route_error = (
                "Route generation failed. Please check the "
                "locations and try again."
            )


else:
    st.sidebar.info(
        "Load a city graph before generating routes."
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
            "Safety-Weighted Route Distance",
            f"{safety_metrics['distance']:.1f} m",
            delta=f"{distance_difference:+.1f} m vs Shortest",
            delta_color="inverse"
        )

    with col2:
        st.metric(
            "Average Lighting Proxy",
            f"{safety_metrics['avg_lux']:.1f}",
            delta=f"{lighting_difference:+.1f} vs Shortest",
            delta_color="normal"
        )

    with col3:
        st.metric(
            "High-Risk Segments",
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
