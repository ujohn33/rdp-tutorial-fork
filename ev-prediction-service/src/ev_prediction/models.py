import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import mean_absolute_error
from typing import Dict, List, Tuple

Logger = logging.getLogger('ev_prediction')


class EVPredictionModel:
    """Similar session-based model for predicting EV charging sessions."""

    def __init__(self, n_similar=5):
        self.sessions_df = pd.DataFrame()
        self.is_trained = False
        self.n_similar = n_similar
        self.feature_columns = [
            'start_hour_sin', 'start_hour_cos',
            'start_weekday_sin', 'start_weekday_cos',
            'start_month_sin', 'start_month_cos'
        ]
        self.target_columns = ['kwh', 'duration_minutes']

    def add_sessions(self, sessions: List[Dict]) -> None:
        """Add new sessions to the historical database."""
        if not sessions:
            return

        df = pd.DataFrame(sessions)
        
        # Add timestamp for ordering
        df['timestamp'] = pd.to_datetime(df['start_time'])
        
        # Keep only required columns
        required_cols = self.feature_columns + self.target_columns + ['timestamp']
        available_cols = [col for col in required_cols if col in df.columns]
        
        if len(available_cols) < len(self.feature_columns) + len(self.target_columns):
            Logger.warning("Missing required columns in session data")
            return
            
        new_sessions = df[available_cols].copy()
        
        # Concatenate with existing sessions
        if self.sessions_df.empty:
            self.sessions_df = new_sessions
        else:
            self.sessions_df = pd.concat([self.sessions_df, new_sessions], 
                                       ignore_index=True)
        
        # Sort by timestamp and keep recent sessions only (last 10000)
        self.sessions_df = self.sessions_df.sort_values('timestamp')
        if len(self.sessions_df) > 10000:
            self.sessions_df = self.sessions_df.tail(10000).reset_index(drop=True)
        
        # Mark as trained if we have enough data
        if len(self.sessions_df) >= 10:
            self.is_trained = True
            Logger.info(f"Model ready with {len(self.sessions_df)} sessions")

    def get_similar_sessions(self, target_features: np.ndarray, 
                           target_col: str) -> Tuple[float, float]:
        """Find similar sessions and return mean prediction and runtime."""
        if not self.is_trained or self.sessions_df.empty:
            return 0.0, 0.0
            
        start_time = pd.Timestamp.now()
        
        # Get feature matrix from historical sessions
        feature_matrix = self.sessions_df[self.feature_columns].values
        
        # Calculate cosine similarity
        similarities = cosine_similarity(
            feature_matrix, 
            target_features.reshape(1, -1)
        ).flatten()
        
        # Find top N similar sessions
        top_indices = np.argsort(similarities)[-self.n_similar:]
        similar_values = self.sessions_df.iloc[top_indices][target_col].values
        
        # Return mean of similar sessions
        prediction = np.mean(similar_values)
        runtime = (pd.Timestamp.now() - start_time).total_seconds()
        
        return prediction, runtime

    def predict_single_session(self, session_features: Dict) -> Dict[str, float]:
        """Predict energy and duration for a single session."""
        if not self.is_trained:
            return {'kwh': 0.0, 'duration_minutes': 0.0, 'runtime': 0.0}
        
        # Extract features
        features = np.array([session_features.get(col, 0.0) 
                           for col in self.feature_columns])
        
        # Predict energy
        energy_pred, energy_runtime = self.get_similar_sessions(
            features, 'kwh'
        )
        
        # Predict duration
        duration_pred, duration_runtime = self.get_similar_sessions(
            features, 'duration_minutes'
        )
        
        return {
            'kwh': energy_pred,
            'duration_minutes': duration_pred,
            'runtime': energy_runtime + duration_runtime
        }

    def predict_session_on_arrival(
        self,
        arrival_time: str,
        user_id: str = None
    ) -> Dict[str, float]:
        """Predict energy and duration for a session based on arrival time."""
        if not self.is_trained:
            Logger.warning("Model not trained yet")
            return {
                'kwh': 0.0,
                'duration_minutes': 0.0,
                'arrival_time': arrival_time
            }

        # Parse arrival time
        arrival_dt = pd.to_datetime(arrival_time)
        
        # Create cyclical features for arrival time
        hour_rad = 2 * np.pi * arrival_dt.hour / 24
        weekday_rad = 2 * np.pi * arrival_dt.weekday() / 7
        month_rad = 2 * np.pi * arrival_dt.month / 12

        session_features = {
            'start_hour_sin': np.sin(hour_rad),
            'start_hour_cos': np.cos(hour_rad),
            'start_weekday_sin': np.sin(weekday_rad),
            'start_weekday_cos': np.cos(weekday_rad),
            'start_month_sin': np.sin(month_rad),
            'start_month_cos': np.cos(month_rad)
        }

        # Predict for this session
        prediction = self.predict_single_session(session_features)
        
        return {
            'arrival_time': arrival_time,
            'user_id': user_id or 'unknown',
            'predicted_kwh': prediction['kwh'],
            'predicted_duration_minutes': prediction['duration_minutes'],
            'prediction_runtime': prediction['runtime']
        }

    def get_model_metrics(self) -> Dict[str, float]:
        """Calculate model performance metrics using recent data."""
        if not self.is_trained or len(self.sessions_df) < 100:
            return {}
            
        # Use last 20% of data for testing
        test_size = int(0.2 * len(self.sessions_df))
        train_data = self.sessions_df.iloc[:-test_size]
        test_data = self.sessions_df.iloc[-test_size:]
        
        # Store original data
        original_df = self.sessions_df.copy()
        
        # Temporarily use only training data
        self.sessions_df = train_data
        
        energy_predictions = []
        duration_predictions = []
        
        for _, row in test_data.iterrows():
            features = {col: row[col] for col in self.feature_columns}
            pred = self.predict_single_session(features)
            energy_predictions.append(pred['kwh'])
            duration_predictions.append(pred['duration_minutes'])
        
        # Restore original data
        self.sessions_df = original_df
        
        # Calculate metrics
        energy_mae = mean_absolute_error(test_data['kwh'], energy_predictions)
        duration_mae = mean_absolute_error(
            test_data['duration_minutes'],
            duration_predictions
        )
        
        return {
            'energy_mae': energy_mae,
            'duration_mae': duration_mae,
            'test_samples': len(test_data),
            'train_samples': len(train_data)
        }
