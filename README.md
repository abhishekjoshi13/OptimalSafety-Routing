
# Optimal Safety Routing Engine

An experimental geospatial pedestrian-routing application that compares the shortest route with a model-weighted alternative using OpenStreetMap road-network data and machine-learning risk classifications.

## Live Demo

[Open the Streamlit Application](https://optimalsafety-routing-9cgrcknqxxsgyy5nrnheqf.streamlit.app/)

## Project Overview

Conventional navigation generally prioritizes distance or travel time.

This project explores another question:

**Can pedestrian routes be compared using both distance and estimated environmental risk?**

The application:

1. Retrieves a walkable street network around a selected area.
2. Generates road-segment features.
3. Predicts risk categories using a trained machine-learning model.
4. Assigns risk-weighted costs to road segments.
5. Calculates shortest and safety-weighted routes.
6. Displays route metrics and an interactive map.

The application is an experimental decision-support tool. It does not guarantee personal safety or replace local knowledge and judgment.

## Technical Stack

| Component | Technology |
|---|---|
| Application interface | Streamlit |
| Street-network data | OpenStreetMap |
| Geospatial network extraction | OSMnx |
| Graph representation and routing | NetworkX |
| Machine learning | Scikit-learn |
| Data processing | Pandas, NumPy |
| Interactive maps | Folium, GeoPandas, Streamlit-Folium |
| Input validation | Pydantic |
| Model serialization | Joblib |
| Automated testing | Pytest |
| Continuous integration | GitHub Actions |

## How It Works

### 1. Location Input and Validation

Users provide a city or area, a starting location and a destination.

Pydantic validates and normalizes the input strings. OSMnx geocodes the locations into geographic coordinates.

The application validates the returned coordinates before attempting to match them to the road network.

### 2. Walkable Network Extraction

The data pipeline retrieves a walkable OpenStreetMap network within a 1,500-metre radius around the selected area.

The resulting network is represented as a NetworkX directed multigraph.

### 3. Feature Engineering and Risk Prediction

The application generates six model input features:

- Road length
- Lane count
- Lighting-level proxy
- Pedestrian-traffic proxy
- Commercial-activity proxy
- Police-distance proxy

The saved Scikit-learn model (`safety_model.joblib`) predicts a risk category for each road segment.

**Important limitation:** The current lighting, pedestrian-traffic, shop-count and police-distance values are rule-based proxies derived from road classifications. They are not live measurements.

### 4. Risk-Weighted Graph Construction

The predicted category is converted into a cost multiplier:

| Predicted category | Cost multiplier |
|---|---:|
| Other / lower-risk category | 1.0 |
| Moderate | 2.5 |
| High Risk | 6.0 |

The application calculates:

`Safety Cost = Road Length × Risk Multiplier`

Each road segment receives a `safety_cost` attribute.

### 5. Route Optimization

NetworkX calculates two routes:

- **Shortest Route:** Minimizes total road-network distance using `length`.
- **Safety-Weighted Route:** Minimizes total model-based cost using `safety_cost`.

The application calculates route distance, safety cost, average lighting proxy and high-risk segment count.

### 6. Visualization

The Streamlit interface displays route comparisons and an interactive Folium map with separate layers for the two routes.

## Repository Structure

```text
OptimalSafety-Routing/
│
├── .github/
│   └── workflows/
│       └── tests.yml
│
├── tests/
│   └── test_routing.py
│
├── app.py
├── data_pipeline.py
├── routing_helpers.py
├── schemas.py
│
├── safety_model.joblib
├── city_map.graphml.gz
├── OptimalSafety_Routing.ipynb
│
├── requirements.txt
├── requirements-dev.txt
├── requirements-lock.txt
└── README.md
```

## Local Setup

### Prerequisites

- Python 3.11
- pip
- Internet connectivity for OpenStreetMap geocoding and network retrieval

### Installation

Clone the repository:

```bash
git clone https://github.com/abhishekjoshi13/OptimalSafety-Routing.git
cd OptimalSafety-Routing
```

Install application dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit.

## Testing

The project uses Pytest for automated testing.

Install the development dependencies:

```bash
pip install -r requirements-dev.txt
```

Run the test suite:

```bash
pytest -v
```

The development requirements use `requirements-lock.txt` to reproduce the dependency versions used by the CI environment.

### Automated CI

GitHub Actions runs the automated test suite on pushes and pull requests targeting `main`.

The workflow:

1. Checks out the repository.
2. Sets up Python 3.11.
3. Installs locked development dependencies.
4. Executes `pytest -v`.

The current verified baseline contains 47 passing test cases.

## Model and Data Considerations

The application loads `safety_model.joblib` using Joblib.

Only load model artifacts from trusted sources. Joblib uses Python serialization mechanisms, and loading an untrusted artifact can execute malicious code.

The application retrieves street-network information dynamically through OSMnx. Data availability depends on OpenStreetMap coverage, network access and geocoding results.

The repository also contains a GraphML network file and a Jupyter notebook. The current Streamlit application builds its road graph dynamically rather than using the GraphML file as its primary graph source.

## Limitations

- CPTED feature values are rule-based proxies, not verified real-world measurements.
- The model was trained using synthetic feature values and incident labels generated from predefined assumptions.
- Recorded test accuracy and F1 scores measure performance on a synthetic held-out dataset, not real-world crime prediction performance.
- Runtime proxy features may differ from the randomized feature distributions used during model training.
- The application does not establish that predicted risk categories correspond to actual crime incidence.
- A 1,500-metre extraction radius may not cover every requested start and destination.
- Geocoding and OpenStreetMap coverage can affect route generation.
- The 250-metre snapping threshold is an application-level configuration, not a guarantee that a location is correctly represented.

Model outputs should therefore be treated as experimental estimates, not validated real-world safety assessments.

## Potential Future Improvements

- Integrating verified street-lighting and pedestrian-activity datasets.
- Using reliable, geographically representative training data.
- Evaluating model performance using independent real-world datasets.
- Improving route coverage and network retrieval strategies.
- Adding further application-level integration tests.
- Evaluating alternative risk-weighting strategies.

## Disclaimer

This project is an educational and experimental geospatial machine-learning application.

Its predictions and route recommendations are estimates based on available map data and model assumptions. They do not guarantee personal safety. Users should exercise independent judgment when choosing a route.
