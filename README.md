# Optimal Safety Routing Engine

Live Demo: [https://optimalsafety-routing-wsow5uyairz4fqjrojwmzn.streamlit.app/]

## Overview
Standard navigation applications optimize strictly for distance or travel time. However, for pedestrians navigating an urban environment, the shortest path is not always the safest. This project is a hybrid geospatial and machine learning routing engine that balances travel distance with environmental safety metrics.

To avoid the algorithmic bias introduced by using historical police arrest records (which often measure patrol density rather than actual crime rates), this system evaluates road networks using Crime Prevention Through Environmental Design (CPTED) principles. The model predicts localized risk based on objective physical infrastructure elements such as street illumination, foot traffic density, and commercial activity.

## Technical Architecture
* **Data Engineering:** Pandas, NumPy
* **Geospatial Processing:** OSMnx, NetworkX, GeoPandas
* **Machine Learning:** Scikit-Learn (Random Forest)
* **Frontend/Deployment:** Streamlit, Folium, Mapclassify

## Methodology

### 1. Data Pipeline and Feature Engineering
The system extracts physical street network data directly from OpenStreetMap. CPTED features are synthesized using stochastic distributions applied to road classifications. Primary arterial roads receive higher baseline illumination and traffic scores, while tertiary residential alleys receive lower baseline metrics. 

### 2. Machine Learning Classifier
Risk classification is handled by a Random Forest Classifier trained on a generated dataset of over 50,000 street segments. The model utilizes balanced class weights to account for severe class imbalance (the statistical rarity of high-risk streets). The trees are pruned with a max depth of 15 to force generalized pattern recognition, prevent overfitting, and compress the deployment binary to under 5MB.

### 3. Routing Algorithm
The engine decouples the machine learning inference from the graph traversal. The model pre-evaluates edge risk attributes, which are converted into dynamic penalty multipliers. These weights are fed into a modified Dijkstra pathfinding algorithm via NetworkX, calculating an optimal path that minimizes danger exposure while keeping the walking distance reasonable.

## Repository Structure
* `app.py`: The main Streamlit application script containing the frontend UI, session state management, and Geopandas rendering logic.
* `city_map.graphml.gz`: The compressed OpenStreetMap graph data containing the street network nodes and edges.
* `safety_model.joblib`: The trained and pruned Random Forest binary file.
* `requirements.txt`: Deployment dependencies.

## Local Installation
To run this project on your local machine:

1. Clone the repository:
   ```bash
   git clone [https://github.com/your-username/optimal-safety-routing.git](https://github.com/your-username/optimal-safety-routing.git)
   cd optimal-safety-routing
