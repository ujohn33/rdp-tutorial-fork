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
        
        # Extract data for each session
        session_times = []
        kwh_values = []
        duration_values = []
        
        for _, row in batch.iterrows():
            session_times.append(row['Start time'].isoformat())
            kwh_values.append(float(row['kwh']))
            duration_values.append(float(row['Duration_hours']))
        
        # Convert to streaming format compatible with RedSQL
        session_data = {
            'timestamp': datetime.now().isoformat(),
            'location': self.location,
            'data_provider': 'EV_Data_Simulator',
            'session_times': session_times,
            'kwh_values': kwh_values,
            'duration_values': duration_values
        }
            
        self.current_index = end_index
        Logger.info(f"Generated batch of {len(batch)} EV charging sessions")
        
        return session_data
