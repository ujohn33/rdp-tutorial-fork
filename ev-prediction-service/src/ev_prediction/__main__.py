import json
import logging
import pathlib
import redis
import os
import time
import yaml
import numpy as np
import pandas as pd
from .models import EVPredictionModel

Logger = logging.getLogger('ev_prediction')


def load_redis_connection_pool(redis_config: dict) -> redis.ConnectionPool:
    """Parse configuration and create Redis connection pool."""
    host = redis_config['host']
    port = redis_config['port']
    db = redis_config['db']
    # Read password from environment variable
    pwd = os.getenv('REDIS_PASSWORD')
    if not pwd:
        pwd = redis_config.get('password')  # Fallback to config file
    
    Logger.info(f'Configure redis connection to {host}:{port} using db {db}')

    if pwd:
        pool = redis.ConnectionPool(
            host=host, port=port, db=db, decode_responses=True, password=pwd
        )
    else:
        pool = redis.ConnectionPool(
            host=host, port=port, db=db, decode_responses=True
        )

    client = redis.Redis(connection_pool=pool)
    client.ping()
    Logger.info(f'Redis connection to {host}:{port} using db {db} is alive.')

    return pool


def main():
    """Main application entry point."""
    # Read config
    config_file_path = pathlib.Path('/app/config.yml').resolve(strict=True)
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Redis config
    redis_config = config['redis']
    redis_pool = load_redis_connection_pool(redis_config=redis_config)

    ev_config = config['ev_prediction']
    model = EVPredictionModel()

    try:
        while True:
            with redis.StrictRedis(connection_pool=redis_pool) as r:
                # Read new EV charging sessions
                session_data = r.xread(
                    streams={ev_config['input_stream']: '$'},
                    count=1, block=5000  # 5 second timeout
                )

            if session_data:
                # Parse session data - new format with arrays
                data = session_data[0][1][-1][1]
                
                # Extract arrays from the new format
                session_times = json.loads(data['session_times'])
                kwh_values = json.loads(data['kwh_values'])
                duration_values = json.loads(data['duration_values'])
                user_ids = json.loads(data.get('user_ids', '[]'))

                # Reconstruct session objects for the model
                sessions = []
                for i in range(len(session_times)):
                    # Parse the timestamp to extract temporal features
                    start_time = pd.to_datetime(session_times[i])
                    
                    # Create session object with required features
                    session = {
                        'start_time': session_times[i],
                        'kwh': kwh_values[i],
                        'duration_minutes': duration_values[i] * 60,  # hrs
                        'start_hour_sin': np.sin(
                            2 * np.pi * start_time.hour / 24
                        ),
                        'start_hour_cos': np.cos(
                            2 * np.pi * start_time.hour / 24
                        ),
                        'start_weekday_sin': np.sin(
                            2 * np.pi * start_time.weekday() / 7
                        ),
                        'start_weekday_cos': np.cos(
                            2 * np.pi * start_time.weekday() / 7
                        ),
                        'start_month_sin': np.sin(
                            2 * np.pi * start_time.month / 12
                        ),
                        'start_month_cos': np.cos(
                            2 * np.pi * start_time.month / 12
                        )
                    }
                    sessions.append(session)

                Logger.info(f"Received {len(sessions)} new sessions")
                
                # Debug: Log session structure
                if sessions:
                    keys = list(sessions[0].keys())
                    Logger.info(f"Sample session keys: {keys}")

                # Add sessions to the model's historical database
                model.add_sessions(sessions)
                
                # Debug: Log model state
                total_sessions = (
                    len(model.sessions_df)
                    if not model.sessions_df.empty else 0
                )
                Logger.info(
                    f"Model trained: {model.is_trained}, "
                    f"Total sessions: {total_sessions}"
                )

                # Generate predictions if model is trained
                if model.is_trained:
                    # Get model performance metrics
                    metrics = model.get_model_metrics()
                    if metrics:
                        Logger.info(f"Model metrics: {metrics}")

                    # Generate predictions for future arrivals
                    # Simulate new arrivals by predicting for sessions
                    predictions = []
                    current_time = pd.to_datetime(session_times[-1])
                    
                    # Generate predictions for next sessions (arrivals)
                    for i in range(5):  # Predict for next 5 potential arrivals
                        # Simulate arrival times (every 30-120 minutes)
                        minutes = np.random.randint(30, 120)
                        arrival_offset = pd.Timedelta(minutes=minutes)
                        arrival_time = current_time + arrival_offset * (i + 1)
                        user_id = f"user_{np.random.randint(1000, 9999)}"
                        
                        # Predict for this simulated arrival
                        prediction = model.predict_session_on_arrival(
                            arrival_time.isoformat(),
                            user_id
                        )
                        predictions.append(prediction)
                    
                    # Send predictions to Redis stream
                    if predictions:
                        with redis.StrictRedis(
                            connection_pool=redis_pool
                        ) as r:
                            for pred in predictions:
                                pred_data = {
                                    'timestamp': data['timestamp'],
                                    'location': data['location'],
                                    'data_provider': 'EV_Prediction_Model',
                                    'arrival_time': pred['arrival_time'],
                                    'user_id': pred['user_id'],
                                    'predicted_kwh': pred['predicted_kwh'],
                                    'predicted_duration_minutes': pred[
                                        'predicted_duration_minutes'
                                    ],
                                    'prediction_runtime': pred[
                                        'prediction_runtime'
                                    ]
                                }
                                r.xadd(ev_config['output_stream'], pred_data)
                        
                        Logger.info(
                            f"Sent {len(predictions)} session predictions "
                            f"to Redis stream"
                        )

            # Update frequency control
            time.sleep(ev_config.get('update_frequency_minutes', 15) * 60)

    except KeyboardInterrupt:
        Logger.info('Stopping EV prediction service...')
    finally:
        Logger.info('EV prediction service stopped')


if __name__ == '__main__':
    main()
