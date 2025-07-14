import json
import logging
import pathlib
import redis
import time
import yaml
import os
from .simulator import EVDataSimulator

Logger = logging.getLogger('ev_data')


def load_redis_connection_pool(redis_config: dict) -> redis.ConnectionPool:
    """Parse configuration and create Redis connection pool."""
    host = redis_config['host']
    port = redis_config['port']
    db = redis_config['db']
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
    client.ping()  # Will raise exception if connection fails
    Logger.info(f'Redis connection to {host}:{port} using db {db} is alive.')

    return pool


def main():
    """Main application entry point."""
    # Read config file
    config_file_path = pathlib.Path('/app/config.yml').resolve(strict=True)
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logging.basicConfig(level=logging.INFO)

    # Redis config
    redis_config = config['redis']
    redis_pool = load_redis_connection_pool(redis_config=redis_config)

    # Initialize EV data simulator
    simulator = EVDataSimulator(config)

    try:
        while True:
            # Get next batch of EV sessions
            batch_data = simulator.get_next_batch()

            # Send to Redis stream
            with redis.StrictRedis(connection_pool=redis_pool) as r:
                # Convert data to Redis stream format - using new array format
                stream_data = {
                    'timestamp': batch_data['timestamp'],
                    'location': batch_data['location'],
                    'data_provider': batch_data['data_provider'],
                    'session_times': json.dumps(batch_data['session_times']),
                    'kwh_values': json.dumps(batch_data['kwh_values']),
                    'duration_values': json.dumps(
                        batch_data['duration_values']
                    )
                }
                r.xadd(simulator.output_stream, stream_data)
                num_points = len(batch_data['session_times'])
                Logger.info(
                    f"Sent {num_points} data points to stream "
                    f"(kWh and duration metrics)"
                )

            # Wait before next batch (simulate real-time)
            time.sleep(60 / simulator.replay_speed)  # Adjust for replay speed

    except KeyboardInterrupt:
        Logger.info('Stopping EV data service...')
    finally:
        Logger.info('EV data service stopped')


if __name__ == '__main__':
    main()
