
import ast
import logging
import math
import os
from pathlib import Path

import joblib
import networkx as nx
import osmnx as osmx
import pandas as pd


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
BACKUP_GRAPH_PATH = BASE_DIR / "city_map.graphml.gz"

GRAPH_RADIUS_METERS = 1500

osmx.settings.timeout = 300
osmx.settings.requests_timeout = 60
osmx.settings.use_cache = True
osmx.settings.overpass_rate_limit = False

osmx.settings.overpass_url = (
    "https://overpass.private.coffee/api"
)

osmx.settings.http_user_agent = (
    "OptimalSafetyRouting/1.0 "
    "(https://github.com/abhishekjoshi13/OptimalSafety-Routing)"
)


FEATURE_COLUMNS = [
    "length",
    "laneCount",
    "luxLevel",
    "pedestrianTraffic",
    "shopCount",
    "policeDist"
]


def _get_road_class(highway):
    if isinstance(highway, list):
        return str(highway[0]) if highway else "unclassified"

    if highway is None:
        return "unclassified"

    if isinstance(highway, str) and highway.startswith("["):
        try:
            parsed = ast.literal_eval(highway)

            if isinstance(parsed, list) and parsed:
                return str(parsed[0])
        except (ValueError, SyntaxError):
            pass

    return str(highway)


def _build_cpted_features(edge_attr):
    road_length = float(edge_attr.get("length", 100.0))

    if not math.isfinite(road_length) or road_length < 0:
        road_length = 100.0

    highway = _get_road_class(edge_attr.get("highway"))

    if highway in ["primary", "secondary", "trunk"]:
        lux = 85.0
        traffic = 200
        shops = 8
        police = 1.0

    elif highway in ["tertiary", "residential"]:
        lux = 45.0
        traffic = 50
        shops = 2
        police = 3.0

    else:
        lux = 15.0
        traffic = 10
        shops = 0
        police = 5.0

    return {
        "length": road_length,
        "laneCount": 1,
        "luxLevel": lux,
        "pedestrianTraffic": traffic,
        "shopCount": shops,
        "policeDist": police
    }


def _haversine_distance(lat1, lon1, lat2, lon2):
    earth_radius = 6371000.0

    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)

    delta_lat = lat2 - lat1
    delta_lon = lon2 - lon1

    value = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1)
        * math.cos(lat2)
        * math.sin(delta_lon / 2) ** 2
    )

    value = min(1.0, max(0.0, value))

    return (
        2
        * earth_radius
        * math.asin(math.sqrt(value))
    )


def _load_backup_graph(center_point, backup_graph_path):
    backup_graph_path = Path(backup_graph_path)

    if not backup_graph_path.is_file():
        raise FileNotFoundError(
            f"Backup road graph not found: {backup_graph_path}"
        )

    logger.warning(
        "Loading backup GraphML road network: %s",
        backup_graph_path.name
    )

    graph = osmx.load_graphml(
        filepath=backup_graph_path
    )

    if (
        graph is None
        or not isinstance(graph, nx.MultiDiGraph)
        or graph.number_of_nodes() == 0
        or graph.number_of_edges() == 0
    ):
        raise ValueError(
            "The backup GraphML file does not contain a valid road graph"
        )

    center_lat, center_lon = center_point

    nearby_nodes = []

    for node, data in graph.nodes(data=True):
        try:
            latitude = float(data["y"])
            longitude = float(data["x"])
        except (KeyError, TypeError, ValueError):
            continue

        if not (
            math.isfinite(latitude)
            and math.isfinite(longitude)
        ):
            continue

        distance = _haversine_distance(
            center_lat,
            center_lon,
            latitude,
            longitude
        )

        if distance <= GRAPH_RADIUS_METERS:
            nearby_nodes.append(node)

    if len(nearby_nodes) < 2:
        raise ValueError(
            "The backup road graph does not cover the requested "
            "location within the configured 1.5 km radius"
        )

    local_graph = graph.subgraph(nearby_nodes).copy()

    if local_graph.number_of_edges() == 0:
        raise ValueError(
            "No road connections were found around the requested location"
        )

    components = list(
        nx.weakly_connected_components(local_graph)
    )

    largest_component = max(
        components,
        key=len
    )

    local_graph = local_graph.subgraph(
        largest_component
    ).copy()

    if (
        local_graph.number_of_nodes() < 2
        or local_graph.number_of_edges() == 0
    ):
        raise ValueError(
            "The backup road network has no usable connected component"
        )

    for node, data in local_graph.nodes(data=True):
        data["x"] = float(data["x"])
        data["y"] = float(data["y"])

    for u, v, key, data in local_graph.edges(
        keys=True,
        data=True
    ):
        try:
            edge_length = float(data["length"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                "The backup graph contains an edge without valid length"
            ) from exc

        if not math.isfinite(edge_length) or edge_length < 0:
            raise ValueError(
                "The backup graph contains an invalid road length"
            )

        data["length"] = edge_length

    logger.info(
        "Backup graph loaded: %s nodes and %s edges",
        local_graph.number_of_nodes(),
        local_graph.number_of_edges()
    )

    return local_graph


def _add_model_predictions(graph, model_path):
    model = joblib.load(model_path)

    edges_data = []
    edges_order = []

    for u, v, key, edge_attr in graph.edges(
        keys=True,
        data=True
    ):
        features = _build_cpted_features(edge_attr)

        edges_data.append(features)
        edges_order.append((u, v, key))

    if not edges_data:
        raise ValueError(
            "The road graph contains no edges for model prediction"
        )

    features_df = pd.DataFrame(
        edges_data,
        columns=FEATURE_COLUMNS
    )

    predictions = model.predict(features_df)

    if len(predictions) != len(edges_order):
        raise ValueError(
            "The model returned an unexpected number of predictions"
        )

    for index, (u, v, key) in enumerate(edges_order):
        prediction = str(predictions[index])

        if prediction == "Moderate":
            multiplier = 2.5

        elif prediction == "High Risk":
            multiplier = 6.0

        else:
            multiplier = 1.0

        edge_features = edges_data[index]

        graph[u][v][key]["safety_cost"] = (
            edge_features["length"] * multiplier
        )

        graph[u][v][key]["risk_class"] = prediction

        graph[u][v][key]["lux_level"] = (
            edge_features["luxLevel"]
        )

        graph[u][v][key]["police_dist"] = (
            edge_features["policeDist"]
        )

    logger.info(
        "ML inference completed for %s road segments",
        len(edges_order)
    )

    return graph


def build_dynamic_safety_graph(
    city_name: str,
    model_path: str,
    backup_graph_path=None
) -> nx.MultiDiGraph:

    if not isinstance(city_name, str) or not city_name.strip():
        raise ValueError(
            "A valid city or area name is required"
        )

    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError(
            "A valid model file path is required"
        )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "The configured safety model was not found"
        )

    if backup_graph_path is None:
        backup_graph_path = BACKUP_GRAPH_PATH

    center_point = osmx.geocode(city_name.strip())

    if (
        not isinstance(center_point, (tuple, list))
        or len(center_point) != 2
    ):
        raise ValueError(
            "Geocoding did not return valid coordinates"
        )

    latitude, longitude = center_point

    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise ValueError(
            "Geocoding returned out-of-range coordinates"
        )

    try:
        logger.info(
            "Attempting live OpenStreetMap graph download for %s",
            city_name
        )

        graph = osmx.graph_from_point(
            center_point,
            dist=GRAPH_RADIUS_METERS,
            network_type="walk"
        )

        if (
            graph is None
            or graph.number_of_nodes() == 0
            or graph.number_of_edges() == 0
        ):
            raise ValueError(
                "Live OpenStreetMap returned an empty road graph"
            )

        logger.info(
            "Live graph downloaded successfully: %s nodes, %s edges",
            graph.number_of_nodes(),
            graph.number_of_edges()
        )

    except Exception as live_error:
        logger.warning(
            "Live OpenStreetMap download failed. "
            "Trying the local GraphML backup. Reason: %s",
            live_error
        )

        try:
            graph = _load_backup_graph(
                center_point,
                backup_graph_path
            )

        except Exception as backup_error:
            logger.exception(
                "Both live graph download and backup graph loading failed"
            )

            raise RuntimeError(
                "Unable to obtain a road network. "
                f"Live download failed: {live_error}. "
                f"Backup loading failed: {backup_error}."
            ) from backup_error

    graph = _add_model_predictions(
        graph,
        model_path
    )

    return graph


def extract_path_metrics(
    graph: nx.MultiDiGraph,
    path: list,
    route_weight: str = "safety_cost"
) -> dict:

    if route_weight not in ("length", "safety_cost"):
        raise ValueError(
            "route_weight must be 'length' or 'safety_cost'"
        )

    if graph is None or not isinstance(graph, nx.MultiDiGraph):
        raise TypeError(
            "A NetworkX MultiDiGraph is required"
        )

    if not isinstance(path, (list, tuple)) or len(path) < 1:
        raise ValueError(
            "A non-empty route path is required"
        )

    total_dist = 0.0
    total_safety_cost = 0.0
    total_lux = 0.0
    high_risk_segments = 0
    edge_count = 0

    for u, v in zip(path[:-1], path[1:]):
        if not graph.has_edge(u, v):
            raise ValueError(
                "The supplied path contains an edge not present in the graph"
            )

        parallel_edges = graph[u][v]

        selected_key, selected_edge = min(
            parallel_edges.items(),
            key=lambda item: float(
                item[1].get(route_weight, 1)
            )
        )

        edge_length = float(
            selected_edge.get("length", 0.0)
        )

        edge_safety_cost = float(
            selected_edge.get("safety_cost", 0.0)
        )

        edge_lux = float(
            selected_edge.get("lux_level", 0.0)
        )

        if not all(
            math.isfinite(value) and value >= 0
            for value in (
                edge_length,
                edge_safety_cost,
                edge_lux
            )
        ):
            raise ValueError(
                "Route contains invalid edge metrics"
            )

        total_dist += edge_length
        total_safety_cost += edge_safety_cost
        total_lux += edge_lux

        if selected_edge.get("risk_class") == "High Risk":
            high_risk_segments += 1

        edge_count += 1

    average_lux = (
        total_lux / edge_count
        if edge_count > 0
        else 0.0
    )

    return {
        "distance": total_dist,
        "safety_cost": total_safety_cost,
        "high_risk_count": high_risk_segments,
        "avg_lux": average_lux,
        "edge_count": edge_count
    }
