
import networkx as nx
import pandas as pd
import pytest
from pydantic import ValidationError

import data_pipeline
from data_pipeline import (
    FEATURE_COLUMNS,
    _build_cpted_features,
    build_dynamic_safety_graph,
    extract_path_metrics,
)
from schemas import LocationQuerySchema


class FakeModel:
    def __init__(self, predictions):
        self.predictions = predictions
        self.received_features = None

    def predict(self, features):
        self.received_features = features
        return self.predictions


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

    monkeypatch.setattr(
        data_pipeline.osmx,
        "geocode",
        lambda city: (23.3, 85.3),
    )

    monkeypatch.setattr(
        data_pipeline.osmx,
        "graph_from_point",
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
