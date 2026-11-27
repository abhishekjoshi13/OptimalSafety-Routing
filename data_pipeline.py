
import ast
import logging
import math
import os
from pathlib import Path

import geopandas as gpd
import joblib
import networkx as nx
import osmnx as osmx
import pandas as pd


logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent
BACKUP_GRAPH_PATH = BASE_DIR / "city_map.graphml.gz"

GRAPH_RADIUS_METERS = 1500

LALPUR_RANCHI_CENTER = (23.3721, 85.3383)

LIVE_OVERPASS_ENDPOINTS = [
    "https://overpass.private.coffee/api",
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
]

osmx.settings.timeout = 180
osmx.settings.requests_timeout = 60
osmx.settings.use_cache = True
osmx.settings.overpass_rate_limit = False
osmx.settings.overpass_url = LIVE_OVERPASS_ENDPOINTS[0]

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
    "policeDist",
]


def _get_city_boundary(city_name):
    if not isinstance(city_name, str) or not city_name.strip():
        raise ValueError("A valid city name is required")

    normalized_city = city_name.strip()

    logger.info(
        "Retrieving geographic boundary for: %s",
        normalized_city,
    )

    try:
        boundary_gdf = osmx.geocode_to_gdf(
            normalized_city,
            which_result=1,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not retrieve the geographic boundary for "
            f"'{normalized_city}'. Check the city name and geocoding "
            "service availability."
        ) from exc

    if (
        boundary_gdf is None
        or boundary_gdf.empty
        or "geometry" not in boundary_gdf.columns
    ):
        raise ValueError(
            f"No geographic boundary was found for '{normalized_city}'"
        )

    if boundary_gdf.crs is None:
        raise ValueError(
            "The city boundary does not have a defined coordinate system"
        )

    boundary_gdf = boundary_gdf.to_crs(epsg=4326)

    geometries = [
        geometry
        for geometry in boundary_gdf.geometry
        if geometry is not None
        and not geometry.is_empty
        and geometry.is_valid
        and geometry.geom_type in ("Polygon", "MultiPolygon")
    ]

    if not geometries:
        raise ValueError(
            f"No valid city polygon was found for '{normalized_city}'"
        )

    city_polygon = gpd.GeoSeries(
        geometries,
        crs="EPSG:4326",
    ).union_all()

    if city_polygon.is_empty or not city_polygon.is_valid:
        raise ValueError(
            "The retrieved city boundary is invalid"
        )

    if city_polygon.geom_type not in ("Polygon", "MultiPolygon"):
        raise ValueError(
            "The retrieved boundary is not a usable city polygon"
        )

    if city_polygon.area <= 0:
        raise ValueError(
            "The retrieved city boundary has no usable area"
        )

    logger.info(
        "City boundary retrieved successfully. Geometry type: %s",
        city_polygon.geom_type,
    )

    return city_polygon


def _get_graph_center(city_name):
    normalized_city = city_name.lower()

    if "lalpur" in normalized_city and "ranchi" in normalized_city:
        center_point = LALPUR_RANCHI_CENTER
    else:
        city_polygon = _get_city_boundary(city_name)
        representative_point = city_polygon.representative_point()

        center_point = (
            float(representative_point.y),
            float(representative_point.x),
        )

    latitude, longitude = center_point

    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise ValueError(
            "The city boundary produced invalid centre coordinates"
        )

    return latitude, longitude


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
        "policeDist": police,
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


def _validate_road_graph(graph, source_name):
    if (
        graph is None
        or not isinstance(graph, nx.MultiDiGraph)
        or graph.number_of_nodes() == 0
        or graph.number_of_edges() == 0
    ):
        raise ValueError(
            f"{source_name} did not provide a valid road graph"
        )

    return graph


def _download_live_graph(city_polygon, city_name):
    original_endpoint = osmx.settings.overpass_url
    endpoint_errors = []

    try:
        for endpoint in LIVE_OVERPASS_ENDPOINTS:
            try:
                logger.info(
                    "Downloading complete city road network for '%s' "
                    "using %s",
                    city_name,
                    endpoint,
                )

                osmx.settings.overpass_url = endpoint

                graph = osmx.graph_from_polygon(
                    city_polygon,
                    network_type="walk",
                    simplify=True,
                    retain_all=False,
                    truncate_by_edge=True,
                )

                graph = _validate_road_graph(
                    graph,
                    f"Live OSM endpoint {endpoint}",
                )

                graph.graph["data_source"] = "OpenStreetMap"
                graph.graph["overpass_endpoint"] = endpoint
                graph.graph["requested_location"] = city_name
                graph.graph["coverage_type"] = "city_boundary"

                logger.info(
                    "City-wide graph downloaded: %s nodes, %s edges",
                    graph.number_of_nodes(),
                    graph.number_of_edges(),
                )

                return graph

            except Exception as exc:
                endpoint_errors.append(
                    f"{endpoint}: {exc}"
                )

                logger.warning(
                    "OSM endpoint failed: %s. Reason: %s",
                    endpoint,
                    exc,
                )

    finally:
        osmx.settings.overpass_url = original_endpoint

    raise RuntimeError(
        "All configured OpenStreetMap endpoints failed. "
        + " | ".join(endpoint_errors)
    )


def _load_backup_graph(center_point, backup_graph_path):
    backup_graph_path = Path(backup_graph_path)

    if not backup_graph_path.is_file():
        raise FileNotFoundError(
            "No local backup road graph is available"
        )

    graph = osmx.load_graphml(
        filepath=backup_graph_path
    )

    graph = _validate_road_graph(
        graph,
        "The local backup GraphML file",
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
            longitude,
        )

        if distance <= GRAPH_RADIUS_METERS:
            nearby_nodes.append(node)

    if len(nearby_nodes) < 2:
        raise ValueError(
            "The available backup graph does not cover this location "
            "within its configured 1.5 km radius."
        )

    local_graph = graph.subgraph(nearby_nodes).copy()

    if local_graph.number_of_edges() == 0:
        raise ValueError(
            "The matching backup area contains no road connections"
        )

    components = list(
        nx.weakly_connected_components(local_graph)
    )

    largest_component = max(
        components,
        key=len,
    )

    local_graph = local_graph.subgraph(
        largest_component
    ).copy()

    if (
        local_graph.number_of_nodes() < 2
        or local_graph.number_of_edges() == 0
    ):
        raise ValueError(
            "The matching backup area has no usable connected road network"
        )

    for node, data in local_graph.nodes(data=True):
        data["x"] = float(data["x"])
        data["y"] = float(data["y"])

    for u, v, key, data in local_graph.edges(
        keys=True,
        data=True,
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

    local_graph.graph["data_source"] = "Local backup GraphML"
    local_graph.graph["coverage_type"] = "local_backup"

    return local_graph


def _add_model_predictions(graph, model_path):
    model = joblib.load(model_path)

    edges_data = []
    edges_order = []

    for u, v, key, edge_attr in graph.edges(
        keys=True,
        data=True,
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
        columns=FEATURE_COLUMNS,
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
        len(edges_order),
    )

    return graph


def build_dynamic_safety_graph(
    city_name: str,
    model_path: str,
    backup_graph_path=None,
) -> nx.MultiDiGraph:

    if not isinstance(city_name, str) or not city_name.strip():
        raise ValueError(
            "A valid city name is required"
        )

    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError(
            "A valid model file path is required"
        )

    if not os.path.isfile(model_path):
        raise FileNotFoundError(
            "The configured safety model was not found"
        )

    city_name = city_name.strip()

    if backup_graph_path is None:
        backup_graph_path = BACKUP_GRAPH_PATH

    city_polygon = _get_city_boundary(city_name)

    representative_point = city_polygon.representative_point()

    center_point = (
        float(representative_point.y),
        float(representative_point.x),
    )

    live_error = None

    try:
        graph = _download_live_graph(
            city_polygon,
            city_name,
        )

    except Exception as exc:
        live_error = exc

        logger.warning(
            "Live city-wide OSM download failed for '%s': %s",
            city_name,
            exc,
        )

        try:
            graph = _load_backup_graph(
                center_point,
                backup_graph_path,
            )

        except Exception as backup_error:
            logger.exception(
                "No geographically matching backup was found"
            )

            raise RuntimeError(
                f"Could not load the road network for '{city_name}'. "
                "The complete city-wide OpenStreetMap download failed, "
                "and the available local backup does not cover the "
                "requested city. "
                f"Live OSM error: {live_error}. "
                f"Backup error: {backup_error}."
            ) from backup_error

    graph = _add_model_predictions(
        graph,
        model_path,
    )

    logger.info(
        "Road graph ready for '%s'. Source: %s",
        city_name,
        graph.graph.get("data_source", "Unknown"),
    )

    return graph


def extract_path_metrics(
    graph: nx.MultiDiGraph,
    path: list,
    route_weight: str = "safety_cost",
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
            ),
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
                edge_lux,
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
        "edge_count": edge_count,
    }
