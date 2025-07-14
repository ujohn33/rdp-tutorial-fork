import json
import logging
import pathlib
import redis
import time
import yaml
from .models import EVPredictionModel

Logger = logging.getLogger('ev_prediction')


def load_redis_connection_pool(redis_config: dict) -> redis.ConnectionPool:
    """Parse configuration and create Redis connection pool."""
    host = redis_config['host']
    port = redis_config['port']
    db = redis_config['db']
    pwd = redis_config.get('password')
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
                # Parse session data
                data = session_data[0][1][-1][1]
                sessions = json.loads(data['sessions'])

                Logger.info(f"Received {len(sessions)} new sessions")

                # Add sessions to the model's historical database
                model.add_sessions(sessions)

                # Generate predictions if model is trained
                if model.is_trained:
                    # Get model performance metrics
                    metrics = model.get_model_metrics()
                    if metrics:
                        Logger.info(f"Model metrics: {metrics}")

                    # Hourly predictions
                    timestamps, energy_pred, duration_pred = \
                        model.predict_hourly_demand(24)

                    # Daily predictions
                    daily_pred = model.predict_daily_totals(7)

                    # Send predictions to Redis
                    with redis.StrictRedis(connection_pool=redis_pool) as r:
                        pred_data = {
                            'timestamp': data['timestamp'],
                            'location': data['location'],
                            'data_provider': 'EV_Prediction_Model',
                            'hourly_timestamps': json.dumps(timestamps),
                            'hourly_energy_kwh': json.dumps(energy_pred),
                            'hourly_duration_min': json.dumps(duration_pred),
                            'daily_timestamps': json.dumps(daily_pred['timestamps']),
                            'daily_energy_kwh': json.dumps(daily_pred['energy_kwh']),
                            'daily_duration_min': json.dumps(daily_pred['duration_minutes'])
                        }
                        r.xadd(ev_config['output_stream'], pred_data)
                        Logger.info("Sent predictions to Redis stream")

            # Update frequency control
            time.sleep(ev_config.get('update_frequency_minutes', 15) * 60)

    except KeyboardInterrupt:
        Logger.info('Stopping EV prediction service...')
    finally:
        Logger.info('EV prediction service stopped')


if __name__ == '__main__':
    main()
