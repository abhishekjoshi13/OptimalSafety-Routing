
# Optimal Safety Routing Engine

An experimental geospatial routing application that compares the shortest pedestrian route with a model-weighted alternative using OpenStreetMap road-network data and machine-learning risk classifications.

## Live Demo

[Open the Streamlit Application](https://optimalsafety-routing-9cgrcknqxxsgyy5nrnheqf.streamlit.app/)

## Project Overview

Conventional navigation generally prioritizes distance or travel time. This project explores a different question:

**Can a pedestrian route be evaluated using both distance and estimated environmental risk?**

The application downloads a walkable street network around a selected area, assigns model-predicted risk categories to road segments, and compares:

- **Shortest Route:** Minimizes road-network distance.
- **Safety-Weighted Route:** Minimizes a weighted cost that penalizes segments classified as higher risk.

The result is a route comparison with distance, risk-segment counts, proxy lighting values, and an interactive map.

This is an experimental decision-support tool. It does not guarantee personal safety or replace local knowledge and judgment.

## Technical Stack

| Component | Technology |
|---|---|
| Application interface | Streamlit |
| Street-network data | OpenStreetMap |
| Geospatial network extraction | OSMnx |
| Graph representation and routing | NetworkX |
| Machine learning inference | Scikit-learn |
| Data processing | Pandas, NumPy |
| Interactive maps | Folium, GeoPandas, Streamlit-Folium |
| Input validation | Pydantic |
| Model serialization | Joblib |

## How It Works

### 1. Location Input and Validation

The user provides a city or area, a starting location, and a destination.

Pydantic validates and normalizes the location strings, including whitespace trimming and length constraints.

OSMnx geocodes the locations into geographic coordinates. The application validates returned coordinates before using them to identify nearby graph nodes.

### 2. Walkable Network Extraction

The data pipeline geocodes the selected city or area and requests a walkable OpenStreetMap network within a 1,500-metre radius.

The resulting road network is represented as a NetworkX directed multigraph.

### 3. Feature Engineering and Risk Prediction

For each road segment, the pipeline constructs the feature columns expected by the trained model:

- Road length
- Lane count
- Lighting-level proxy
- Pedestrian-traffic proxy
- Commercial-activity proxy
- Police-distance proxy

The current implementation derives these proxy values from OpenStreetMap road classifications using fixed rule-based assignments.

The saved Scikit-learn model (`safety_model.joblib`) predicts a risk category for each segment.

**Important limitation:** These proxy values are not live measurements of illumination, pedestrian volume, shop density, or police-station distance. Their quality depends on the assumptions used to construct the features and train the model.

### 4. Risk-Weighted Graph Construction

The predicted risk category is converted into a multiplier:

| Predicted category | Cost multiplier |
|---|---:|
| Other / lower-risk category | 1.0 |
| Moderate | 2.5 |
| High Risk | 6.0 |

The pipeline calculates:

`Safety Cost = Road Length × Risk Multiplier`

Each road segment receives a `safety_cost` attribute, which is used by the routing algorithm.

### 5. Route Optimization

NetworkX computes two routes between the selected graph nodes:

- Shortest route using `length` as the edge weight.
- Safety-weighted route using `safety_cost` as the edge weight.

The application then calculates route metrics using the edge selected by the corresponding routing weight, including total distance, average lighting proxy, and high-risk segment count.

### 6. Trade-off Analysis and Visualization

The interface compares the two routes and displays:

- Safety-weighted route distance
- Difference in distance compared with the shortest route
- Average lighting proxy
- Difference in high-risk segment counts

Folium visualizes the shortest route and safety-weighted route with separate map layers.

## Repository Structure

```text
OptimalSafety-Routing/
│
├── app.py
├── data_pipeline.py
├── schemas.py
├── safety_model.joblib
├── city_map.graphml.gz
├── OptimalSafety_Routing.ipynb
├── requirements.txt
└── README.md
```

## Local Setup

### Prerequisites

- Python 3.10 or a compatible Python version for the installed dependencies
- pip
- Internet connectivity for OpenStreetMap geocoding and network retrieval

### Installation

Clone the repository:

```bash
git clone https://github.com/abhishekjoshi13/OptimalSafety-Routing.git
cd OptimalSafety-Routing
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

Open the local URL displayed by Streamlit in your browser.

## Model and Data Considerations

The application loads `safety_model.joblib` using Joblib.

Only load model artifacts from trusted sources. Joblib uses Python serialization mechanisms and loading an untrusted artifact can execute malicious code.

The application retrieves street-network information dynamically through OSMnx. Availability and completeness depend on OpenStreetMap coverage, network access, and geocoding results.

The repository also contains a GraphML network file and a Jupyter notebook. The Streamlit application currently builds its graph dynamically rather than loading the GraphML file as its primary graph source.

## Limitations and Future Improvements

Current limitations include:

- CPTED feature values are rule-based proxies rather than verified real-world measurements.
- The quality of model predictions depends on the training data and feature assumptions.
- The application does not establish that a predicted risk category corresponds to actual crime incidence.
- A 1,500-metre network extraction radius may not cover every requested start and destination.
- Geocoding and OpenStreetMap data availability can affect route generation.
- The model-based route is an experimental estimate, not a guarantee of safety.

Potential future improvements include:

- Integrating verified street-lighting and pedestrian-activity datasets.
- Using reliable, geographically representative training data.
- Evaluating model performance with documented metrics and independent test data.
- Adding automated tests for feature generation, graph construction, routing, and metric calculations.
- Improving route coverage and handling locations outside the downloaded network.

## Disclaimer

This project is an educational and experimental geospatial machine-learning application. Its predictions and route recommendations are estimates based on available map data and model assumptions. Users should exercise independent judgment when choosing a route.

   ```
