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

RISK_COLORS = {
    "Safe": "#16803c",
    "Moderate": "#e69b19",
    "High Risk": "#c62828",
    "Unknown": "#777777",
}


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

    return 2 * EARTH_RADIUS_METERS * math.asin(math.sqrt(a))


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

    return min(latitudes), max(latitudes), min(longitudes), max(longitudes)


def validate_location_in_graph(graph, coords):
    latitude, longitude = coords
    min_lat, max_lat, min_lon, max_lon = get_graph_bounds(graph)

    latitude_margin = MAX_SNAP_DISTANCE_METERS / 111320.0
    middle_latitude = math.radians((min_lat + max_lat) / 2)
    longitude_scale = max(abs(math.cos(middle_latitude)), 0.000001)
    longitude_margin = MAX_SNAP_DISTANCE_METERS / (111320.0 * longitude_scale)

    if not (
        min_lat - latitude_margin <= latitude <= max_lat + latitude_margin
        and min_lon - longitude_margin <= longitude <= max_lon + longitude_margin
    ):
        raise LocationOutsideGraphError(
            "This location is outside the loaded road-network region."
        )


def snap_location_to_graph(graph, coords, location_name):
    latitude, longitude = coords
    validate_location_in_graph(graph, coords)

    try:
        nearest_node = ox.distance.nearest_nodes(
            graph, X=longitude, Y=latitude
        )
    except Exception as exc:
        logger.exception("Nearest road node lookup failed for %s", location_name)
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
        latitude, longitude, node_latitude, node_longitude
    )

    if snap_distance > MAX_SNAP_DISTANCE_METERS:
        raise LocationTooFarFromRoadError(
            f"{location_name} is approximately {snap_distance:.0f} metres "
            f"from the nearest road node. The maximum allowed distance is "
            f"{MAX_SNAP_DISTANCE_METERS:.0f} metres."
        )

    return nearest_node, snap_distance


def select_path_edges(graph, path, route_weight="safety_cost"):
    """
    Select the exact parallel edge used for each step of a route.

    For a shortest route, choose the parallel edge with minimum length.
    For a safety route, choose the parallel edge with minimum safety_cost.
    The returned records are also used by route metrics and map rendering.
    """
    if route_weight not in ("length", "safety_cost"):
        raise ValueError("route_weight must be 'length' or 'safety_cost'")

    if graph is None or not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("A NetworkX MultiDiGraph is required")

    if not isinstance(path, (list, tuple)) or len(path) < 1:
        raise ValueError("A non-empty route path is required")

    selected_edges = []

    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            raise ValueError(
                "The supplied path contains an edge not present in the graph"
            )

        parallel_edges = graph[u][v]
        selected_key, selected_data = min(
            parallel_edges.items(),
            key=lambda item: float(item[1].get(route_weight, 1))
        )

        edge_data = dict(selected_data)
        edge_data.update({"u": u, "v": v, "key": selected_key})
        selected_edges.append(edge_data)

    return selected_edges


def _metrics_from_selected_edges(selected_edges):
    total_dist = 0.0
    total_safety_cost = 0.0
    total_lux = 0.0
    high_risk_segments = 0

    for edge in selected_edges:
        edge_length = float(edge.get("length", 0.0))
        edge_safety_cost = float(edge.get("safety_cost", 0.0))
        edge_lux = float(edge.get("lux_level", 0.0))

        if not all(
            math.isfinite(value) and value >= 0
            for value in (edge_length, edge_safety_cost, edge_lux)
        ):
            raise ValueError("Route contains invalid edge metrics")

        total_dist += edge_length
        total_safety_cost += edge_safety_cost
        total_lux += edge_lux

        if edge.get("risk_class") == "High Risk":
            high_risk_segments += 1

    edge_count = len(selected_edges)
    average_lux = total_lux / edge_count if edge_count else 0.0

    return {
        "distance": total_dist,
        "safety_cost": total_safety_cost,
        "high_risk_count": high_risk_segments,
        "avg_lux": average_lux,
        "edge_count": edge_count,
    }


def calculate_routes(graph, start_node, end_node):
    shortest_path = nx.shortest_path(
        graph, start_node, end_node, weight="length"
    )
    safety_path = nx.shortest_path(
        graph, start_node, end_node, weight="safety_cost"
    )

    shortest_edges = select_path_edges(
        graph, shortest_path, route_weight="length"
    )
    safety_edges = select_path_edges(
        graph, safety_path, route_weight="safety_cost"
    )

    shortest_metrics = _metrics_from_selected_edges(shortest_edges)
    safety_metrics = _metrics_from_selected_edges(safety_edges)

    return {
        "shortest_path": shortest_path,
        "safety_path": safety_path,
        "shortest_edges": shortest_edges,
        "safety_edges": safety_edges,
        "shortest_metrics": shortest_metrics,
        "safety_metrics": safety_metrics,
    }


def _edge_coordinates(graph, edge):
    geometry = edge.get("geometry")

    if geometry is not None and not getattr(geometry, "is_empty", True):
        try:
            if geometry.geom_type == "LineString":
                coordinate_parts = [list(geometry.coords)]
            elif geometry.geom_type == "MultiLineString":
                coordinate_parts = [
                    list(part.coords) for part in geometry.geoms
                ]
            else:
                coordinate_parts = []
        except (AttributeError, TypeError):
            coordinate_parts = []

        if coordinate_parts:
            return [
                [(float(lat), float(lon)) for lon, lat in part]
                for part in coordinate_parts
                if len(part) >= 2
            ]

    start = graph.nodes[edge["u"]]
    end = graph.nodes[edge["v"]]

    return [[
        (float(start["y"]), float(start["x"])),
        (float(end["y"]), float(end["x"])),
    ]]


def _add_route_edges_to_map(
    route_map, graph, edges, layer_name, color, show_risk_colors=False
):
    layer = folium.FeatureGroup(name=layer_name, show=True)

    for index, edge in enumerate(edges, start=1):
        risk_class = str(edge.get("risk_class", "Unknown"))
        segment_color = (
            RISK_COLORS.get(risk_class, RISK_COLORS["Unknown"])
            if show_risk_colors
            else color
        )

        road_name = edge.get("name", "Unnamed road")
        if isinstance(road_name, list):
            road_name = ", ".join(str(name) for name in road_name)

        popup = (
            f"{layer_name} | Segment {index}<br>"
            f"Road: {road_name}<br>"
            f"Predicted class: {risk_class}<br>"
            f"Length: {float(edge.get('length', 0.0)):.1f} m"
        )

        for coordinates in _edge_coordinates(graph, edge):
            folium.PolyLine(
                locations=coordinates,
                color=segment_color,
                weight=6 if show_risk_colors else 5,
                opacity=0.9,
                popup=popup,
                tooltip=f"{risk_class}: {road_name}",
            ).add_to(layer)

    layer.add_to(route_map)


def create_route_map(graph, route_result):
    start_coords = route_result["start_coords"]
    end_coords = route_result["end_coords"]

    center = (
        (start_coords[0] + end_coords[0]) / 2,
        (start_coords[1] + end_coords[1]) / 2,
    )

    route_map = folium.Map(location=center, zoom_start=15)

    shortest_edges = route_result.get("shortest_edges")
    safety_edges = route_result.get("safety_edges")

    if shortest_edges is None:
        shortest_edges = select_path_edges(
            graph, route_result["shortest_path"], "length"
        )

    if safety_edges is None:
        safety_edges = select_path_edges(
            graph, route_result["safety_path"], "safety_cost"
        )

    _add_route_edges_to_map(
        route_map,
        graph,
        shortest_edges,
        "Shortest Path (red)",
        "#d62728",
    )

    _add_route_edges_to_map(
        route_map,
        graph,
        safety_edges,
        "Safety Route — predicted risk colors",
        "#16803c",
        show_risk_colors=True,
    )

    folium.Marker(
        location=start_coords,
        popup="Origin",
        icon=folium.Icon(color="blue", icon="play"),
    ).add_to(route_map)

    folium.Marker(
        location=end_coords,
        popup="Destination",
        icon=folium.Icon(color="red", icon="stop"),
    ).add_to(route_map)

    legend_html = """
    <div style="
        position: fixed; bottom: 25px; left: 25px; z-index: 9999;
        background: white; padding: 10px 12px; border: 1px solid #888;
        border-radius: 5px; font-size: 13px;">
      <b>Safety-route predicted classes</b><br>
      <span style="color:#16803c;">●</span> Safe<br>
      <span style="color:#e69b19;">●</span> Moderate<br>
      <span style="color:#c62828;">●</span> High Risk<br>
      <span style="color:#777777;">●</span> Unknown<br>
      <span style="color:#d62728;">━</span> Shortest route
    </div>
    """
    route_map.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl(collapsed=False).add_to(route_map)
    return route_map
