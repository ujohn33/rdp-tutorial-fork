# EV Charging Session Prediction Platform

## About

This repository provides a complete platform for predicting EV charging session duration and energy demand using the [Rapid Deployment Platform](https://ait-rdp.github.io/) architecture. The system simulates real-time EV charging data and provides machine learning-based predictions through interactive dashboards.

## Features

### Data Processing
- **EV Data Simulator**: Replays historical charging session data in real-time
- **Session Analytics**: Processes energy consumption, duration, and temporal patterns
- **Time Feature Engineering**: Includes sin/cos encodings for cyclical patterns (hour, weekday, month)

### Machine Learning Predictions
- **Duration Prediction**: Forecasts individual session durations
- **Energy Demand Prediction**: Predicts kWh consumption per session
- **Temporal Forecasting**: Hourly and daily aggregated predictions
- **Real-time Model Updates**: Continuously learns from new session data

### Interactive Dashboards
1. **Forecast Overview Panel**
   - Total forecasted kWh per day (line chart)
   - Total forecasted session duration per day
   - Prediction intervals for probabilistic forecasting

2. **Temporal Patterns**
   - Hourly average kWh and session duration (bar charts)
   - Day-of-week trends in energy demand and duration
   - Monthly seasonality patterns

3. **Session Distribution Analysis**
   - Histogram of session durations
   - Histogram of energy usage per session
   - Statistical summaries

4. **Forecast vs Actual (for live monitoring)**
   - Actual vs Forecasted kWh comparison
   - Actual vs Forecasted Duration comparison
   - Error metrics (MAE, RMSE) as single-value panels

## Architecture

The platform uses a microservices architecture with:
- **Redis Streams**: Real-time data communication
- **TimescaleDB**: Time series data storage
- **Grafana**: Interactive visualization dashboards
- **Docker Compose**: Service orchestration
- **Machine Learning**: Scikit-learn based prediction models

### Services

- **ev-data-service**: Simulates real-time EV charging session data from CSV
- **ev-prediction-service**: Machine learning service for demand forecasting
- **data-sync-service**: Syncs data from Redis streams to TimescaleDB
- **grafana**: Visualization and dashboard service
- **redis**: Stream processing and caching
- **timescale**: Time series database

## Data Format

The system expects EV charging session data with the following features:
- **Energy consumption**: kWh per session
- **Session duration**: seconds, minutes, and hours
- **Start times**: with temporal features
- **Cyclical encodings**: sin/cos transformations for time features
- **Location information**: charging point details

## Prerequisites

You need to have [Docker](https://docs.docker.com/) and [Docker Compose](https://docs.docker.com/compose/) installed.

## Setup

1. **Clone the repository**:
   ```bash
   git clone <your-repository-url>
   cd rdp-tutorial-fork
   ```

2. **Copy environment variables**:
   ```bash
   cp .env.example .env
   ```
   Edit `.env` file with your desired passwords and configurations.

3. **Ensure your EV charging data CSV is in the correct location**:
   ```bash
   # Your CSV should be at: ./data/data_ev_parking_duration.csv
   # Make sure it has the required columns for EV session data
   ```

## Usage

Build the services and deploy the setup:
```bash
docker compose build
docker compose up
```

**Access the services**:
- **Grafana Dashboard**: http://localhost:3000 (admin/your_password)
- **Redis Insight**: http://localhost:5540 (for debugging Redis streams)

## Configuration

### EV Data Service
- Modify `ev-data-service/config.yml` to adjust replay speed and batch size
- The service reads from `/data/data_ev_parking_duration.csv` by default

### Prediction Service  
- Configure prediction horizon and update frequency in `ev-prediction-service/config.yml`
- Model parameters can be tuned in the EVPredictionModel class

### Dashboard Customization
- Grafana dashboards are automatically provisioned
- Customize panels in `grafana/dashboards/RDP Tutorial Dashboard/`

## Monitoring

- **Redis Streams**: Use Redis Insight to monitor data flow
- **Database**: Connect to TimescaleDB for direct data access
- **Logs**: Use `docker compose logs <service-name>` to view service logs
- **Metrics**: Grafana provides comprehensive analytics and forecasting views

## Development

To extend the platform:

1. **Add new prediction models**: Extend the `EVPredictionModel` class
2. **Create new dashboard panels**: Add JSON configurations to Grafana
3. **Modify data processing**: Update the data sync service configuration
4. **Scale services**: Adjust Docker Compose for multiple instances
