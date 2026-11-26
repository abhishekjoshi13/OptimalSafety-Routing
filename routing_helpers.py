import folium
import logging
import math
from numbers import Real

import networkx as nx
import osmnx as ox

from data_pipeline import extract_path_metrics


logger = logging.getLogger(__name__)

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

    if (
        isinstance(latitude, bool)
        or isinstance(longitude, bool)
        or not isinstance(latitude, Real)
        or not isinstance(longitude, Real)
    ):
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
