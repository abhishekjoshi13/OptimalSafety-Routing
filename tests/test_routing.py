
import math
from pathlib import Path

import geopandas as gpd
import joblib
import networkx as nx
import pandas as pd
import pytest
from pydantic import ValidationError
from shapely.geometry import Polygon

import data_pipeline
import routing_helpers

from data_pipeline import (
    FEATURE_COLUMNS,
    _build_cpted_features,
    build_dynamic_safety_graph,
    extract_path_metrics,
)

from schemas import LocationQuerySchema

from routing_helpers import (
    MAX_SNAP_DISTANCE_METERS,
    LocationOutsideGraphError,
    LocationTooFarFromRoadError,
    calculate_routes,
    get_graph_bounds,
    haversine_distance,
    snap_location_to_graph,
    validate_coordinates,
    validate_location_in_graph,
)


class FakeModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.received_features = None

    def predict(self, features):
        self.received_features = features
        return self.predictions


def create_test_graph():
    graph = nx.MultiDiGraph()

    graph.add_node(1, x=85.3000, y=23.3000)
    graph.add_node(2, x=85.3010, y=23.3000)
    graph.add_node(3, x=85.3020, y=23.3000)

    graph.add_edge(
        1,
        2,
        length=100.0,
        safety_cost=500.0,
        lux_level=15.0,
        risk_class="High Risk",
    )

    graph.add_edge(
        2,
        3,
        length=100.0,
        safety_cost=500.0,
        lux_level=15.0,
        risk_class="High Risk",
    )

    graph.add_edge(
        1,
        3,
        length=300.0,
        safety_cost=100.0,
        lux_level=85.0,
        risk_class="Safe",
    )

    return graph


def test_location_schema_strips_whitespace():
    query = LocationQuerySchema(
        city_name="  Ranchi  ",
        start_address="  Main Road  ",
        end_address="  Lalpur Chowk  ",
    )

    assert query.city_name == "Ranchi"
    assert query.start_address == "Main Road"
    assert query.end_address == "Lalpur Chowk"


@pytest.mark.parametrize(
    "field_values",
    [
        {
            "city_name": "  ",
            "start_address": "Main Road",
            "end_address": "Lalpur Chowk",
        },
        {
            "city_name": "Ranchi",
            "start_address": "  ",
            "end_address": "Lalpur Chowk",
        },
        {
            "city_name": "Ranchi",
            "start_address": "Main Road",
            "end_address": "  ",
        },
    ],
)
def test_location_schema_rejects_blank_values(field_values):
    with pytest.raises(ValidationError):
        LocationQuerySchema(**field_values)


def test_location_schema_rejects_oversized_city_name():
    with pytest.raises(ValidationError):
        LocationQuerySchema(
            city_name="R" * 121,
            start_address="Main Road",
            end_address="Lalpur Chowk",
        )


def test_primary_road_features_use_expected_proxies():
    features = _build_cpted_features(
        {
            "length": 250.0,
            "highway": ["primary", "secondary"],
        }
    )

    assert features == {
        "length": 250.0,
        "laneCount": 1,
        "luxLevel": 85.0,
        "pedestrianTraffic": 200,
        "shopCount": 8,
        "policeDist": 1.0,
    }


def test_residential_road_features_use_expected_proxies():
    features = _build_cpted_features(
        {
            "length": 120.0,
            "highway": "residential",
        }
    )

    assert features["luxLevel"] == 45.0
    assert features["pedestrianTraffic"] == 50
    assert features["shopCount"] == 2
    assert features["policeDist"] == 3.0


def test_unknown_road_class_uses_fallback_proxies():
    features = _build_cpted_features(
        {
            "length": 80.0,
            "highway": None,
        }
    )

    assert features["luxLevel"] == 15.0
    assert features["pedestrianTraffic"] == 10
    assert features["shopCount"] == 0
    assert features["policeDist"] == 5.0


def test_build_dynamic_safety_graph_adds_predictions(
    monkeypatch,
    tmp_path,
):
    graph = nx.MultiDiGraph()

    graph.add_node(1, x=85.3, y=23.3)
    graph.add_node(2, x=85.4, y=23.4)

    graph.add_edge(
        1,
        2,
        key=0,
        length=100.0,
        highway="primary",
    )

    graph.add_edge(
        2,
        1,
        key=0,
        length=200.0,
        highway="residential",
    )

    model = FakeModel(["Moderate", "High Risk"])

    model_file = tmp_path / "trusted_model.joblib"
    model_file.touch()

    boundary = gpd.GeoDataFrame(
        {
            "geometry": [
                Polygon([
                    (85.2, 23.2),
                    (85.5, 23.2),
                    (85.5, 23.5),
                    (85.2, 23.5),
                    (85.2, 23.2),
                ])
            ]
        },
        crs="EPSG:4326",
    )

    monkeypatch.setattr(
        data_pipeline.osmx,
        "geocode_to_gdf",
        lambda *args, **kwargs: boundary,
    )

    monkeypatch.setattr(
        data_pipeline.osmx,
        "graph_from_polygon",
        lambda *args, **kwargs: graph.copy(),
    )

    monkeypatch.setattr(
        data_pipeline.joblib,
        "load",
        lambda path: model,
    )

    result = build_dynamic_safety_graph(
        "Ranchi",
        str(model_file),
    )

    assert result.number_of_edges() == 2

    assert result[1][2][0]["risk_class"] == "Moderate"
    assert result[1][2][0]["safety_cost"] == 250.0

    assert result[2][1][0]["risk_class"] == "High Risk"
    assert result[2][1][0]["safety_cost"] == 1200.0

    assert isinstance(model.received_features, pd.DataFrame)
    assert list(model.received_features.columns) == FEATURE_COLUMNS
    assert len(model.received_features) == 2


def test_build_dynamic_safety_graph_rejects_missing_model():
    with pytest.raises(FileNotFoundError):
        build_dynamic_safety_graph(
            "Ranchi",
            "missing_model_file.joblib",
        )


def test_route_metrics_select_edge_using_length_weight():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        1,
        2,
        key=0,
        length=100.0,
        safety_cost=600.0,
        lux_level=15.0,
        risk_class="High Risk",
    )

    graph.add_edge(
        1,
        2,
        key=1,
        length=120.0,
        safety_cost=120.0,
        lux_level=85.0,
        risk_class="Safe",
    )

    metrics = extract_path_metrics(
        graph,
        [1, 2],
        route_weight="length",
    )

    assert metrics["distance"] == 100.0
    assert metrics["safety_cost"] == 600.0
    assert metrics["high_risk_count"] == 1
    assert metrics["avg_lux"] == 15.0
    assert metrics["edge_count"] == 1


def test_route_metrics_select_edge_using_safety_cost():
    graph = nx.MultiDiGraph()

    graph.add_edge(
        1,
        2,
        key=0,
        length=100.0,
        safety_cost=600.0,
        lux_level=15.0,
        risk_class="High Risk",
    )

    graph.add_edge(
        1,
        2,
        key=1,
        length=120.0,
        safety_cost=120.0,
        lux_level=85.0,
        risk_class="Safe",
    )

    metrics = extract_path_metrics(
        graph,
        [1, 2],
        route_weight="safety_cost",
    )

    assert metrics["distance"] == 120.0
    assert metrics["safety_cost"] == 120.0
    assert metrics["high_risk_count"] == 0
    assert metrics["avg_lux"] == 85.0
    assert metrics["edge_count"] == 1


def test_route_metrics_reject_invalid_weight():
    graph = nx.MultiDiGraph()
    graph.add_edge(1, 2, length=100.0, safety_cost=100.0)

    with pytest.raises(ValueError):
        extract_path_metrics(
            graph,
            [1, 2],
            route_weight="invalid",
        )


def test_route_metrics_reject_missing_path_edge():
    graph = nx.MultiDiGraph()
    graph.add_node(1)
    graph.add_node(2)

    with pytest.raises(ValueError):
        extract_path_metrics(
            graph,
            [1, 2],
            route_weight="length",
        )


# Coordinate validation tests


@pytest.mark.parametrize(
    "coords, expected",
    [
        ((23.3, 85.3), (23.3, 85.3)),
        ([0, 0], (0.0, 0.0)),
        ((90, 180), (90.0, 180.0)),
        ((-90, -180), (-90.0, -180.0)),
    ],
)
def test_validate_coordinates_accepts_valid_values(coords, expected):
    assert validate_coordinates(coords) == expected


@pytest.mark.parametrize(
    "coords",
    [
        None,
        "23.3,85.3",
        (23.3,),
        (23.3, 85.3, 10),
        ("23.3", 85.3),
        (True, 85.3),
        (23.3, False),
        (float("nan"), 85.3),
        (23.3, float("inf")),
        (91, 85.3),
        (23.3, 181),
    ],
)
def test_validate_coordinates_rejects_invalid_values(coords):
    with pytest.raises(ValueError):
        validate_coordinates(coords)


# Haversine distance tests


def test_haversine_distance_same_point_is_zero():
    distance = haversine_distance(
        23.3,
        85.3,
        23.3,
        85.3,
    )

    assert distance == pytest.approx(0.0)


def test_haversine_distance_for_one_degree_latitude():
    distance = haversine_distance(
        0,
        0,
        1,
        0,
    )

    assert distance == pytest.approx(111194.9, abs=2.0)


def test_haversine_distance_is_symmetric():
    first_distance = haversine_distance(
        23.3,
        85.3,
        23.31,
        85.31,
    )

    reverse_distance = haversine_distance(
        23.31,
        85.31,
        23.3,
        85.3,
    )

    assert first_distance == pytest.approx(reverse_distance)


# Graph bounds tests


def test_get_graph_bounds_returns_correct_boundaries():
    graph = nx.MultiDiGraph()

    graph.add_node(1, x=85.3, y=23.3)
    graph.add_node(2, x=85.5, y=23.5)
    graph.add_node(3, x=85.4, y=23.4)

    bounds = get_graph_bounds(graph)

    assert bounds == (
        23.3,
        23.5,
        85.3,
        85.5,
    )


def test_get_graph_bounds_ignores_invalid_node_coordinates():
    graph = nx.MultiDiGraph()

    graph.add_node(1, x=85.3, y=23.3)
    graph.add_node(2, x=float("nan"), y=23.4)
    graph.add_node(3, x=85.4)

    bounds = get_graph_bounds(graph)

    assert bounds == (
        23.3,
        23.3,
        85.3,
        85.3,
    )


@pytest.mark.parametrize(
    "graph",
    [
        None,
        nx.MultiDiGraph(),
    ],
)
def test_get_graph_bounds_rejects_empty_graph(graph):
    with pytest.raises(ValueError):
        get_graph_bounds(graph)


def test_get_graph_bounds_rejects_graph_without_valid_coordinates():
    graph = nx.MultiDiGraph()
    graph.add_node(1, x=float("nan"), y=float("inf"))

    with pytest.raises(ValueError):
        get_graph_bounds(graph)


# Location inside graph tests


def test_validate_location_in_graph_accepts_location_inside_bounds():
    graph = create_test_graph()

    validate_location_in_graph(
        graph,
        (23.3000, 85.3010),
    )


def test_validate_location_in_graph_rejects_far_location():
    graph = create_test_graph()

    with pytest.raises(LocationOutsideGraphError):
        validate_location_in_graph(
            graph,
            (24.0, 86.0),
        )


def test_validate_location_in_graph_rejects_empty_graph():
    graph = nx.MultiDiGraph()

    with pytest.raises(ValueError):
        validate_location_in_graph(
            graph,
            (23.3, 85.3),
        )


# Nearest road node snapping tests


def test_snap_location_to_graph_returns_nearest_node(
    monkeypatch,
):
    graph = create_test_graph()

    monkeypatch.setattr(
        routing_helpers.ox.distance,
        "nearest_nodes",
        lambda graph, X, Y: 1,
    )

    node, distance = snap_location_to_graph(
        graph,
        (23.3000, 85.3000),
        "Starting location",
    )

    assert node == 1
    assert distance == pytest.approx(0.0)


def test_snap_location_to_graph_rejects_location_outside_graph(
    monkeypatch,
):
    graph = create_test_graph()

    def unexpected_lookup(*args, **kwargs):
        pytest.fail("Nearest-node lookup should not be called")

    monkeypatch.setattr(
        routing_helpers.ox.distance,
        "nearest_nodes",
        unexpected_lookup,
    )

    with pytest.raises(LocationOutsideGraphError):
        snap_location_to_graph(
            graph,
            (24.0, 86.0),
            "Starting location",
        )


def test_snap_location_to_graph_rejects_node_too_far(
    monkeypatch,
):
    graph = nx.MultiDiGraph()

    graph.add_node(1, x=85.3, y=23.3000)
    graph.add_node(2, x=85.3, y=23.3100)

    monkeypatch.setattr(
        routing_helpers.ox.distance,
        "nearest_nodes",
        lambda graph, X, Y: 1,
    )

    with pytest.raises(LocationTooFarFromRoadError):
        snap_location_to_graph(
            graph,
            (23.3050, 85.3),
            "Starting location",
        )


def test_snap_location_to_graph_handles_lookup_failure(
    monkeypatch,
):
    graph = create_test_graph()

    def failed_lookup(*args, **kwargs):
        raise RuntimeError("Simulated lookup failure")

    monkeypatch.setattr(
        routing_helpers.ox.distance,
        "nearest_nodes",
        failed_lookup,
    )

    with pytest.raises(ValueError, match="Could not find"):
        snap_location_to_graph(
            graph,
            (23.3000, 85.3000),
            "Starting location",
        )


# Route calculation tests


def test_calculate_routes_finds_different_optimal_paths():
    graph = create_test_graph()

    result = calculate_routes(
        graph,
        1,
        3,
    )

    assert result["shortest_path"] == [1, 2, 3]
    assert result["safety_path"] == [1, 3]

    assert result["shortest_metrics"]["distance"] == 200.0
    assert result["shortest_metrics"]["safety_cost"] == 1000.0

    assert result["safety_metrics"]["distance"] == 300.0
    assert result["safety_metrics"]["safety_cost"] == 100.0


def test_calculate_routes_rejects_missing_path():
    graph = nx.MultiDiGraph()

    graph.add_node(1)
    graph.add_node(2)

    with pytest.raises(nx.NetworkXNoPath):
        calculate_routes(
            graph,
            1,
            2,
        )


# Real model artifact compatibility test


def test_real_safety_model_artifact_loads_and_predicts():
    model_path = (
        Path(__file__).resolve().parents[1]
        / "safety_model.joblib"
    )

    assert model_path.is_file()

    model = joblib.load(model_path)

    sample_features = pd.DataFrame(
        [
            [100.0, 1, 85.0, 200, 8, 1.0],
            [150.0, 1, 45.0, 50, 2, 3.0],
            [80.0, 1, 15.0, 10, 0, 5.0],
        ],
        columns=FEATURE_COLUMNS,
    )

    assert list(model.feature_names_in_) == FEATURE_COLUMNS

    predictions = model.predict(sample_features)

    assert len(predictions) == 3

    assert set(predictions).issubset(
        {"Safe", "Moderate", "High Risk"}
    )
