"""
Data Validator - Handles validation of mood entries and ensures data quality
"""

import json
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from jsonschema import validate, ValidationError

logger = logging.getLogger(__name__)

class DataValidator:
    """
    Handles validation of mood entries and ensures data quality
    """
    
    def __init__(self):
        self.mood_entry_schema = {
            "type": "object",
            "properties": {
                "timestamp": {"type": "string", "format": "date-time"},
                "mood_score": {"type": "integer", "minimum": 1, "maximum": 10},
                "emotions": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 5
                },
                "sleep_hours": {"type": "number", "minimum": 0, "maximum": 24},
                "exercise_minutes": {"type": "integer", "minimum": 0, "maximum": 480},
                "social_interactions": {"type": "integer", "minimum": 0, "maximum": 50},
                "stress_level": {"type": "integer", "minimum": 1, "maximum": 10},
                "weather": {"type": "string"},
                "notes": {"type": "string"},
                "triggers": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "entry_id": {"type": "string"}
            },
            "required": ["timestamp", "mood_score", "emotions", "sleep_hours", 
                        "exercise_minutes", "social_interactions", "stress_level", 
                        "weather", "entry_id"]
        }
        
        self.valid_emotions = {
            'happy', 'sad', 'anxious', 'excited', 'angry', 'calm', 'frustrated',
            'content', 'worried', 'energetic', 'depressed', 'hopeful', 'irritable',
            'peaceful', 'stressed', 'joyful', 'lonely', 'confident', 'overwhelmed'
        }
        
        self.valid_triggers = {
            'work_stress', 'family_issues', 'health_concerns', 'financial_worry',
            'relationship_problems', 'sleep_deprivation', 'social_isolation',
            'weather_change', 'exercise_lack', 'good_news', 'achievement',
            'social_interaction', 'relaxation', 'travel', 'creative_activity'
        }
        
        self.valid_weather = {
            'sunny', 'cloudy', 'rainy', 'snowy', 'windy', 'foggy', 'stormy'
        }
        
    def validate_single_entry(self, entry: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """
        Validate a single mood entry
        
        Args:
            entry: Mood entry dictionary
            
        Returns:
            Tuple of (is_valid, error_messages)
        """
        errors = []
        
        try:
            # Schema validation
            validate(entry, self.mood_entry_schema)
        except ValidationError as e:
            errors.append(f"Schema validation failed: {e.message}")
        
        # Additional custom validations
        errors.extend(self._validate_emotions(entry.get('emotions', [])))
        errors.extend(self._validate_triggers(entry.get('triggers', [])))
        errors.extend(self._validate_weather(entry.get('weather', '')))
        errors.extend(self._validate_timestamp(entry.get('timestamp', '')))
        errors.extend(self._validate_correlations(entry))
        
        is_valid = len(errors) == 0
        
        if not is_valid:
            logger.warning(f"Validation failed for entry {entry.get('entry_id', 'unknown')}: {errors}")
        
        return is_valid, errors
    
    def _validate_emotions(self, emotions: List[str]) -> List[str]:
        """Validate emotions list"""
        errors = []
        
        if not emotions:
            errors.append("At least one emotion must be specified")
        
        for emotion in emotions:
            if emotion not in self.valid_emotions:
                errors.append(f"Invalid emotion: {emotion}")
        
        return errors
    
    def _validate_triggers(self, triggers: List[str]) -> List[str]:
        """Validate triggers list"""
        errors = []
        
        for trigger in triggers:
            if trigger not in self.valid_triggers:
                errors.append(f"Invalid trigger: {trigger}")
        
        return errors
    
    def _validate_weather(self, weather: str) -> List[str]:
        """Validate weather condition"""
        errors = []
        
        if weather not in self.valid_weather:
            errors.append(f"Invalid weather condition: {weather}")
        
        return errors
    
    def _validate_timestamp(self, timestamp: str) -> List[str]:
        """Validate timestamp format and range"""
        errors = []
        
        try:
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            
            # Check if timestamp is not in the future
            if dt > datetime.now():
                errors.append("Timestamp cannot be in the future")
            
            # Check if timestamp is not too old (more than 5 years)
            if dt < datetime.now() - timedelta(days=5*365):
                errors.append("Timestamp is too old (more than 5 years)")
                
        except ValueError:
            errors.append("Invalid timestamp format")
        
        return errors
    
    def _validate_correlations(self, entry: Dict[str, Any]) -> List[str]:
        """Validate logical correlations between data points"""
        errors = []
        
        mood_score = entry.get('mood_score', 5)
        sleep_hours = entry.get('sleep_hours', 8)
        stress_level = entry.get('stress_level', 5)
        emotions = entry.get('emotions', [])
        
        # Check for logical inconsistencies
        if mood_score >= 8 and 'depressed' in emotions:
            errors.append("High mood score inconsistent with depression emotion")
        
        if mood_score <= 3 and 'happy' in emotions:
            errors.append("Low mood score inconsistent with happy emotion")
        
        if sleep_hours < 4 and mood_score >= 8:
            errors.append("Very low sleep with high mood may indicate mania")
        
        if stress_level >= 8 and mood_score >= 8:
            errors.append("High stress with high mood may indicate inconsistency")
        
        return errors
    
    def validate(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Validate a list of mood entries
        
        Args:
            data: List of mood entries
            
        Returns:
            List of valid mood entries
        """
        valid_entries = []
        invalid_count = 0
        
        logger.info(f"Starting validation of {len(data)} mood entries")
        
        for entry in data:
            is_valid, errors = self.validate_single_entry(entry)
            
            if is_valid:
                valid_entries.append(entry)
            else:
                invalid_count += 1
                logger.warning(f"Invalid entry {entry.get('entry_id', 'unknown')}: {errors}")
        
        logger.info(f"Validation complete: {len(valid_entries)} valid, {invalid_count} invalid")
        
        return valid_entries
    
    def detect_missing_entries(self, data: List[Dict[str, Any]]) -> List[datetime]:
        """
        Detect missing entries in the mood data
        
        Args:
            data: List of mood entries
            
        Returns:
            List of dates where entries are missing
        """
        if not data:
            return []
        
        # Sort data by timestamp
        sorted_data = sorted(data, key=lambda x: x['timestamp'])
        
        # Get date range
        start_date = datetime.fromisoformat(sorted_data[0]['timestamp'].replace('Z', '+00:00')).date()
        end_date = datetime.fromisoformat(sorted_data[-1]['timestamp'].replace('Z', '+00:00')).date()
        
        # Get all dates with entries
        entry_dates = set()
        for entry in sorted_data:
            entry_date = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')).date()
            entry_dates.add(entry_date)
        
        # Find missing dates
        missing_dates = []
        current_date = start_date
        
        while current_date <= end_date:
            if current_date not in entry_dates:
                missing_dates.append(current_date)
            current_date += timedelta(days=1)
        
        logger.info(f"Found {len(missing_dates)} missing entries")
        return missing_dates
    
    def handle_missing_entries(self, data: List[Dict[str, Any]], 
                             method: str = "interpolate") -> List[Dict[str, Any]]:
        """
        Handle missing entries using various methods
        
        Args:
            data: List of mood entries
            method: Method to handle missing entries ('interpolate', 'average', 'flag')
            
        Returns:
            List of mood entries with missing entries handled
        """
        missing_dates = self.detect_missing_entries(data)
        
        if not missing_dates:
            return data
        
        logger.info(f"Handling {len(missing_dates)} missing entries using method: {method}")
        
        if method == "interpolate":
            return self._interpolate_missing_entries(data, missing_dates)
        elif method == "average":
            return self._average_missing_entries(data, missing_dates)
        elif method == "flag":
            return self._flag_missing_entries(data, missing_dates)
        else:
            logger.warning(f"Unknown method {method}, using interpolation")
            return self._interpolate_missing_entries(data, missing_dates)
    
    def _interpolate_missing_entries(self, data: List[Dict[str, Any]], 
                                   missing_dates: List[datetime]) -> List[Dict[str, Any]]:
        """Interpolate missing entries using linear interpolation"""
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        
        # Create complete date range
        start_date = df['date'].min()
        end_date = df['date'].max()
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # Group by date and take mean for multiple entries per day
        daily_data = df.groupby('date').agg({
            'mood_score': 'mean',
            'sleep_hours': 'mean',
            'exercise_minutes': 'mean',
            'social_interactions': 'mean',
            'stress_level': 'mean'
        }).reset_index()
        
        # Reindex to include missing dates
        daily_data = daily_data.set_index('date').reindex(date_range.date).reset_index()
        daily_data.columns = ['date', 'mood_score', 'sleep_hours', 'exercise_minutes', 
                             'social_interactions', 'stress_level']
        
        # Interpolate missing values
        daily_data = daily_data.interpolate(method='linear')
        
        # Convert back to original format
        interpolated_data = []
        for _, row in daily_data.iterrows():
            if row['date'] in [d.date() for d in missing_dates]:
                entry = {
                    'timestamp': datetime.combine(row['date'], datetime.min.time()).isoformat(),
                    'mood_score': int(round(row['mood_score'])),
                    'emotions': ['interpolated'],
                    'sleep_hours': round(row['sleep_hours'], 1),
                    'exercise_minutes': int(round(row['exercise_minutes'])),
                    'social_interactions': int(round(row['social_interactions'])),
                    'stress_level': int(round(row['stress_level'])),
                    'weather': 'unknown',
                    'notes': 'Interpolated entry for missing data',
                    'triggers': [],
                    'entry_id': f"interpolated_{row['date'].strftime('%Y%m%d')}"
                }
                interpolated_data.append(entry)
        
        # Combine with original data
        result_data = data + interpolated_data
        logger.info(f"Interpolated {len(interpolated_data)} missing entries")
        return result_data
    
    def _average_missing_entries(self, data: List[Dict[str, Any]], 
                               missing_dates: List[datetime]) -> List[Dict[str, Any]]:
        """Fill missing entries with average values"""
        df = pd.DataFrame(data)
        
        # Calculate averages
        averages = {
            'mood_score': int(round(df['mood_score'].mean())),
            'sleep_hours': round(df['sleep_hours'].mean(), 1),
            'exercise_minutes': int(round(df['exercise_minutes'].mean())),
            'social_interactions': int(round(df['social_interactions'].mean())),
            'stress_level': int(round(df['stress_level'].mean()))
        }
        
        # Create entries for missing dates
        average_entries = []
        for missing_date in missing_dates:
            entry = {
                'timestamp': datetime.combine(missing_date, datetime.min.time()).isoformat(),
                'mood_score': averages['mood_score'],
                'emotions': ['average'],
                'sleep_hours': averages['sleep_hours'],
                'exercise_minutes': averages['exercise_minutes'],
                'social_interactions': averages['social_interactions'],
                'stress_level': averages['stress_level'],
                'weather': 'unknown',
                'notes': 'Average entry for missing data',
                'triggers': [],
                'entry_id': f"average_{missing_date.strftime('%Y%m%d')}"
            }
            average_entries.append(entry)
        
        # Combine with original data
        result_data = data + average_entries
        logger.info(f"Created {len(average_entries)} average entries for missing data")
        return result_data
    
    def _flag_missing_entries(self, data: List[Dict[str, Any]], 
                            missing_dates: List[datetime]) -> List[Dict[str, Any]]:
        """Flag missing entries for manual review"""
        logger.info(f"Flagged {len(missing_dates)} missing entries for manual review")
        
        # Add metadata about missing entries
        for entry in data:
            entry['has_missing_neighbors'] = False
            entry_date = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')).date()
            
            # Check if there are missing entries within 3 days
            for missing_date in missing_dates:
                if abs((entry_date - missing_date).days) <= 3:
                    entry['has_missing_neighbors'] = True
                    break
        
        return data
    
    def get_data_quality_report(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate a data quality report
        
        Args:
            data: List of mood entries
            
        Returns:
            Dictionary containing data quality metrics
        """
        if not data:
            return {"error": "No data to analyze"}
        
        df = pd.DataFrame(data)
        
        # Basic statistics
        total_entries = len(data)
        date_range = self._get_date_range(data)
        missing_dates = self.detect_missing_entries(data)
        
        # Data completeness
        completeness = 1 - (len(missing_dates) / date_range) if date_range > 0 else 0
        
        # Validation statistics
        valid_entries = 0
        invalid_entries = 0
        
        for entry in data:
            is_valid, _ = self.validate_single_entry(entry)
            if is_valid:
                valid_entries += 1
            else:
                invalid_entries += 1
        
        # Value distributions
        mood_stats = df['mood_score'].describe().to_dict()
        sleep_stats = df['sleep_hours'].describe().to_dict()
        
        report = {
            'total_entries': total_entries,
            'date_range_days': date_range,
            'missing_entries': len(missing_dates),
            'data_completeness': round(completeness * 100, 2),
            'valid_entries': valid_entries,
            'invalid_entries': invalid_entries,
            'validation_rate': round((valid_entries / total_entries) * 100, 2),
            'mood_statistics': mood_stats,
            'sleep_statistics': sleep_stats,
            'most_common_emotions': self._get_most_common_emotions(data),
            'most_common_triggers': self._get_most_common_triggers(data)
        }
        
        logger.info(f"Data quality report generated: {report['validation_rate']}% valid entries")
        return report
    
    def _get_date_range(self, data: List[Dict[str, Any]]) -> int:
        """Get the date range of the data in days"""
        if not data:
            return 0
        
        dates = []
        for entry in data:
            date = datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')).date()
            dates.append(date)
        
        return (max(dates) - min(dates)).days + 1
    
    def _get_most_common_emotions(self, data: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Get the most common emotions"""
        emotion_counts = {}
        
        for entry in data:
            for emotion in entry.get('emotions', []):
                emotion_counts[emotion] = emotion_counts.get(emotion, 0) + 1
        
        return sorted(emotion_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    def _get_most_common_triggers(self, data: List[Dict[str, Any]]) -> List[Tuple[str, int]]:
        """Get the most common triggers"""
        trigger_counts = {}
        
        for entry in data:
            for trigger in entry.get('triggers', []):
                trigger_counts[trigger] = trigger_counts.get(trigger, 0) + 1
        
        return sorted(trigger_counts.items(), key=lambda x: x[1], reverse=True)[:5] 