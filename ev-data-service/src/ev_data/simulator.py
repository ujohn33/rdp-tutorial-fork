import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any

Logger = logging.getLogger('ev_data')


class EVDataSimulator:
    """Simulates real-time EV charging session data."""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config['ev_data']
        self.data_file = self.config['data_file']
        self.output_stream = self.config['output_stream']
        self.location = self.config['location']
        # hours per minute
        self.replay_speed = self.config.get('replay_speed', 24)
        self.batch_size = self.config.get('batch_size', 10)
        
        # Load and prepare data
        self.df = self._load_and_prepare_data()
        self.current_index = 0
        
    def _load_and_prepare_data(self) -> pd.DataFrame:
        """Load CSV data and prepare for streaming."""
        Logger.info(f"Loading EV charging data from {self.data_file}")
        
        df = pd.read_csv(self.data_file)
        
        # Convert timestamps
        df['Start time'] = pd.to_datetime(df['Start time'])
        df['End time'] = pd.to_datetime(df['End time'])
        
        # Clean and extract relevant columns
        df['kwh'] = df['KWh'].str.replace(',', '.').astype(float)
        df['duration_seconds'] = df['Duration_hours'] * 3600
        df['duration_minutes'] = df['Duration_hours'] * 60
        
        # Sort by start time
        df = df.sort_values('Start time').reset_index(drop=True)
        
        # Select relevant columns for streaming
        columns = [
            'Start time', 'End time', 'kwh', 'duration_seconds',
            'duration_minutes', 'Duration_hours', 'Start_time_Month',
            'Start_time_Hour', 'Start_time_Weekday', 'Start_time_Month_x',
            'Start_time_Month_y', 'Start_time_Hour_x', 'Start_time_Hour_y',
            'Start_time_Weekday_x', 'Start_time_Weekday_y',
            'Location', 'Charging point'
        ]
        
        return df[columns].copy()
        
    def get_next_batch(self) -> Dict[str, Any]:
        """Get next batch of EV charging sessions."""
        if self.current_index >= len(self.df):
            self.current_index = 0  # Loop back to beginning
            
        end_index = min(self.current_index + self.batch_size, len(self.df))
        batch = self.df.iloc[self.current_index:end_index]
        
        # Convert to streaming format
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'location': self.location,
            'data_provider': 'EV_Data_Simulator',
            'sessions': []
        }
        
        for _, row in batch.iterrows():
            session = {
                'start_time': row['Start time'].isoformat(),
                'end_time': row['End time'].isoformat(),
                'kwh': float(row['kwh']),
                'duration_seconds': float(row['duration_seconds']),
                'duration_minutes': float(row['duration_minutes']),
                'duration_hours': float(row['Duration_hours']),
                'charging_point': str(row['Charging point']),
                'start_month': int(row['Start_time_Month']),
                'start_hour': int(row['Start_time_Hour']),
                'start_weekday': int(row['Start_time_Weekday']),
                'start_month_sin': float(row['Start_time_Month_x']),
                'start_month_cos': float(row['Start_time_Month_y']),
                'start_hour_sin': float(row['Start_time_Hour_x']),
                'start_hour_cos': float(row['Start_time_Hour_y']),
                'start_weekday_sin': float(row['Start_time_Weekday_x']),
                'start_weekday_cos': float(row['Start_time_Weekday_y'])
            }
            session_data['sessions'].append(session)
            
        self.current_index = end_index
        Logger.info(f"Generated batch of {len(batch)} EV charging sessions")
        
        return session_data
