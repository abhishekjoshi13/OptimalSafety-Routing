
import os
import math

import joblib
import pandas as pd
import networkx as nx
import osmnx as osmx


osmx.settings.timeout = 300
osmx.settings.use_cache = True


FEATURE_COLUMNS = [
    "length",
    "laneCount",
    "luxLevel",
    "pedestrianTraffic",
    "shopCount",
    "policeDist"
]


def _get_road_class(highway):
    """
    Normalize the OSM highway attribute because it may be
    either a string or a list of road classifications.
    """
    if isinstance(highway, list):
        return str(highway[0]) if highway else "unclassified"

    if highway is None:
        return "unclassified"

    return str(highway)


def _build_cpted_features(edge_attr):
    """
    Build the feature values currently used by this project.

    These are rule-based proxy values derived from road class.
    They are not live measurements of street lighting,
    pedestrian activity, shops, or police proximity.
    """
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


def build_dynamic_safety_graph(
    city_name: str,
    model_path: str
) -> nx.MultiDiGraph:
    """
    Build a walkable street graph around a geocoded location,
    predict risk classes for its edges, and attach routing costs.

    The graph radius is 1500 metres around the geocoded center.
    """

    if not isinstance(city_name, str) or not city_name.strip():
        raise ValueError("A valid city or area name is required")

    if not isinstance(model_path, str) or not model_path.strip():
        raise ValueError("A valid model file path is required")

    if not os.path.isfile(model_path):
        raise FileNotFoundError("The configured safety model was not found")

    center_point = osmx.geocode(city_name.strip())

    if (
        not isinstance(center_point, (tuple, list))
        or len(center_point) != 2
    ):
        raise ValueError("Geocoding did not return valid coordinates")

    latitude, longitude = center_point

    if not (
        math.isfinite(latitude)
        and math.isfinite(longitude)
        and -90 <= latitude <= 90
        and -180 <= longitude <= 180
    ):
        raise ValueError("Geocoding returned out-of-range coordinates")

    graph = osmx.graph_from_point(
        center_point,
        dist=1500,
        network_type="walk"
    )

    if graph is None or graph.number_of_nodes() == 0:
        raise ValueError("No walkable road network was found for this area")

    if graph.number_of_edges() == 0:
        raise ValueError("The downloaded road network contains no edges")

    # Only load model artifacts that are trusted and controlled
    # by the project owner. Joblib files can execute code on loading.
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

    return graph


def extract_path_metrics(
    graph: nx.MultiDiGraph,
    path: list,
    route_weight: str = "safety_cost"
) -> dict:
    """
    Calculate route metrics using the same edge-selection
    rule used by NetworkX for weighted MultiDiGraph paths.

    route_weight should be either 'length' or 'safety_cost'.
    """

    if route_weight not in ("length", "safety_cost"):
        raise ValueError(
            "route_weight must be 'length' or 'safety_cost'"
        )

    if graph is None or not isinstance(graph, nx.MultiDiGraph):
        raise TypeError("A NetworkX MultiDiGraph is required")

    if not isinstance(path, (list, tuple)) or len(path) < 1:
        raise ValueError("A non-empty route path is required")

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

        # NetworkX's default MultiGraph weight handling selects
        # the minimum weight among parallel edges.
        selected_key, selected_edge = min(
            parallel_edges.items(),
            key=lambda item: float(
                item[1].get(route_weight, 1)
            )
        )

        edge_length = float(selected_edge.get("length", 0.0))
        edge_safety_cost = float(
            selected_edge.get("safety_cost", 0.0)
        )
        edge_lux = float(selected_edge.get("lux_level", 0.0))

        if not all(
            math.isfinite(value) and value >= 0
            for value in (
                edge_length,
                edge_safety_cost,
                edge_lux
            )
        ):
            raise ValueError("Route contains invalid edge metrics")

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
