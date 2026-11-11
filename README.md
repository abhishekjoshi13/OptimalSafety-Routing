# Optimal Safety Routing Engine

Live Demo: [https://optimalsafety-routing-9cgrcknqxxsgyy5nrnheqf.streamlit.app/](https://optimalsafety-routing-9cgrcknqxxsgyy5nrnheqf.streamlit.app/)

## Overview
Standard navigation applications optimize strictly for distance or travel time. However, for pedestrians navigating an urban environment, the shortest path is not always the safest. This project is a hybrid geospatial and machine learning routing engine that balances travel distance with environmental safety metrics.

To avoid the algorithmic bias introduced by using historical police arrest records (which often measure patrol density rather than actual crime rates), this system evaluates road networks using Crime Prevention Through Environmental Design (CPTED) principles. The model predicts localized risk based on objective physical infrastructure elements such as street illumination, foot traffic density, and commercial activity.

## Technical Architecture
- **Data Engineering:** Pandas, NumPy
- **Geospatial Processing:** OSMnx, NetworkX, GeoPandas
- **Machine Learning:** Scikit-Learn (Random Forest)
- **Security & Validation:** Pydantic
- **Frontend/Deployment:** Streamlit, Folium, Mapclassify

## Core Modules & Methodology

### 1. Security & Validation (`schemas.py`)
To prevent injection vulnerabilities and SSRF (Server-Side Request Forgery) attacks during graph traversal, all user inputs are strictly validated using Pydantic schemas. This ensures that node index parameters remain strictly within memory bounds before any processing occurs.

### 2. Geospatial ETL & Inference (`data_pipeline.py`)
This decoupled layer handles dynamic data extraction and machine learning inference:
- **Dynamic Geocoding:** Converts user string addresses into latitude/longitude coordinates via the Nominatim API and extracts a 1.5km walkable network radius using OSMnx.
- **Feature Engineering:** Synthesizes CPTED features using stochastic distributions applied to road classifications.
- **Batch Inference:** Evaluates risk classification via a Random Forest Classifier trained on 50,000+ street segments. The model utilizes balanced class weights to handle severe minority class imbalance, pruned to a max depth of 15 to force generalized pattern recognition and compress the binary.

### 3. Routing & Analytics Engine (`app.py`)
The presentation and optimization layer:
- **Optimization:** The pre-evaluated machine learning risk attributes are translated into dynamic penalty multipliers. These weights are fed into a modified Dijkstra pathfinding algorithm via NetworkX, calculating an optimal path that minimizes danger exposure.
- **Trade-off Analytics:** Compares the shortest path versus the safest path in real-time, calculating detour distance penalties against visibility improvements (Lux) and bypassed high-risk segments to provide transparent, explainable AI recommendations.

### 4. Custom Pathfinding Fallback (`routing_engine.cpp`)
A hardened, custom C++ Dijkstra implementation built from scratch for high-performance edge tracing and L4 network state machine tracing.

## Local Installation
To run this project on your local machine:

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/optimal-safety-routing.git
   cd optimal-safety-routing
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit application:
   ```bash
   streamlit run app.py
   ```
