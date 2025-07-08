"""
Storage Manager - Handles data storage, retrieval, and export functionality
"""

import json
import os
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

class StorageManager:
    """
    Handles data storage, retrieval, and export functionality
    """
    
    def __init__(self, storage_dir: str = "data"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        self.json_dir = self.storage_dir / "json"
        self.excel_dir = self.storage_dir / "excel"
        self.backup_dir = self.storage_dir / "backups"
        
        self.json_dir.mkdir(exist_ok=True)
        self.excel_dir.mkdir(exist_ok=True)
        self.backup_dir.mkdir(exist_ok=True)
        
        logger.info(f"Storage manager initialized with directory: {self.storage_dir}")
    
    def save_data(self, data: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Save mood data to JSON file
        
        Args:
            data: List of mood entries
            filename: Optional filename, defaults to timestamped filename
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mood_data_{timestamp}.json"
        
        filepath = self.json_dir / filename
        
        try:
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Saved {len(data)} mood entries to {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Failed to save data to {filepath}: {e}")
            raise
    
    def load_data(self, filename: str = None) -> List[Dict[str, Any]]:
        """
        Load mood data from JSON file
        
        Args:
            filename: Optional filename, defaults to most recent file
            
        Returns:
            List of mood entries
        """
        if filename is None:
            filename = self._get_most_recent_file()
        
        filepath = self.json_dir / filename
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            logger.info(f"Loaded {len(data)} mood entries from {filepath}")
            return data
        
        except FileNotFoundError:
            logger.warning(f"File not found: {filepath}")
            return []
        except Exception as e:
            logger.error(f"Failed to load data from {filepath}: {e}")
            raise
    
    def export_to_excel(self, data: List[Dict[str, Any]], filename: str = None) -> str:
        """
        Export mood data to Excel file
        
        Args:
            data: List of mood entries
            filename: Optional filename, defaults to timestamped filename
            
        Returns:
            Path to exported Excel file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"mood_data_{timestamp}.xlsx"
        
        filepath = self.excel_dir / filename
        
        try:
            # Convert to DataFrame
            df = self._prepare_dataframe(data)
            
            # Create Excel writer with multiple sheets
            with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
                # Main data sheet
                df.to_excel(writer, sheet_name='Mood Data', index=False)
                
                # Summary statistics sheet
                summary_df = self._create_summary_sheet(df)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Emotions analysis sheet
                emotions_df = self._create_emotions_sheet(data)
                emotions_df.to_excel(writer, sheet_name='Emotions Analysis', index=False)
                
                # Triggers analysis sheet
                triggers_df = self._create_triggers_sheet(data)
                triggers_df.to_excel(writer, sheet_name='Triggers Analysis', index=False)
                
                # Monthly trends sheet
                monthly_df = self._create_monthly_trends_sheet(df)
                monthly_df.to_excel(writer, sheet_name='Monthly Trends', index=False)
            
            logger.info(f"Exported {len(data)} mood entries to {filepath}")
            return str(filepath)
        
        except Exception as e:
            logger.error(f"Failed to export data to Excel: {e}")
            raise
    
    def _prepare_dataframe(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Prepare DataFrame from mood data"""
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['month'] = df['timestamp'].dt.month_name()
        df['week'] = df['timestamp'].dt.isocalendar().week
        
        # Convert lists to strings for Excel compatibility
        df['emotions_str'] = df['emotions'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        df['triggers_str'] = df['triggers'].apply(lambda x: ', '.join(x) if isinstance(x, list) else str(x))
        
        # Reorder columns for better readability
        column_order = [
            'entry_id', 'timestamp', 'date', 'time', 'day_of_week', 'month', 'week',
            'mood_score', 'emotions_str', 'sleep_hours', 'exercise_minutes',
            'social_interactions', 'stress_level', 'weather', 'triggers_str', 'notes'
        ]
        
        # Only include columns that exist in the DataFrame
        available_columns = [col for col in column_order if col in df.columns]
        df = df[available_columns]
        
        return df
    
    def _create_summary_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create summary statistics sheet"""
        summary_data = []
        
        # Basic statistics
        summary_data.append(['Total Entries', len(df)])
        summary_data.append(['Date Range', f"{df['date'].min()} to {df['date'].max()}"])
        summary_data.append(['Average Mood Score', round(df['mood_score'].mean(), 2)])
        summary_data.append(['Median Mood Score', df['mood_score'].median()])
        summary_data.append(['Mood Score Std Dev', round(df['mood_score'].std(), 2)])
        summary_data.append(['Min Mood Score', df['mood_score'].min()])
        summary_data.append(['Max Mood Score', df['mood_score'].max()])
        
        # Sleep statistics
        summary_data.append(['Average Sleep Hours', round(df['sleep_hours'].mean(), 2)])
        summary_data.append(['Average Exercise Minutes', round(df['exercise_minutes'].mean(), 2)])
        summary_data.append(['Average Social Interactions', round(df['social_interactions'].mean(), 2)])
        summary_data.append(['Average Stress Level', round(df['stress_level'].mean(), 2)])
        
        # Day of week analysis
        summary_data.append(['', ''])  # Empty row
        summary_data.append(['Day of Week Analysis', ''])
        
        for day in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']:
            day_data = df[df['day_of_week'] == day]
            if len(day_data) > 0:
                avg_mood = round(day_data['mood_score'].mean(), 2)
                summary_data.append([f'Average Mood - {day}', avg_mood])
        
        return pd.DataFrame(summary_data, columns=['Metric', 'Value'])
    
    def _create_emotions_sheet(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create emotions analysis sheet"""
        emotion_counts = {}
        emotion_mood_totals = {}
        
        for entry in data:
            mood_score = entry.get('mood_score', 5)
            emotions = entry.get('emotions', [])
            
            for emotion in emotions:
                if emotion not in emotion_counts:
                    emotion_counts[emotion] = 0
                    emotion_mood_totals[emotion] = 0
                
                emotion_counts[emotion] += 1
                emotion_mood_totals[emotion] += mood_score
        
        emotions_data = []
        for emotion, count in emotion_counts.items():
            avg_mood = round(emotion_mood_totals[emotion] / count, 2) if count > 0 else 0
            percentage = round((count / len(data)) * 100, 2)
            
            emotions_data.append({
                'Emotion': emotion,
                'Frequency': count,
                'Percentage': percentage,
                'Average Mood When Present': avg_mood
            })
        
        # Sort by frequency
        emotions_data.sort(key=lambda x: x['Frequency'], reverse=True)
        
        return pd.DataFrame(emotions_data)
    
    def _create_triggers_sheet(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """Create triggers analysis sheet"""
        trigger_counts = {}
        trigger_mood_totals = {}
        
        for entry in data:
            mood_score = entry.get('mood_score', 5)
            triggers = entry.get('triggers', [])
            
            for trigger in triggers:
                if trigger not in trigger_counts:
                    trigger_counts[trigger] = 0
                    trigger_mood_totals[trigger] = 0
                
                trigger_counts[trigger] += 1
                trigger_mood_totals[trigger] += mood_score
        
        triggers_data = []
        for trigger, count in trigger_counts.items():
            avg_mood = round(trigger_mood_totals[trigger] / count, 2) if count > 0 else 0
            percentage = round((count / len(data)) * 100, 2)
            
            triggers_data.append({
                'Trigger': trigger,
                'Frequency': count,
                'Percentage': percentage,
                'Average Mood When Present': avg_mood
            })
        
        # Sort by frequency
        triggers_data.sort(key=lambda x: x['Frequency'], reverse=True)
        
        return pd.DataFrame(triggers_data)
    
    def _create_monthly_trends_sheet(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create monthly trends sheet"""
        monthly_data = df.groupby('month').agg({
            'mood_score': ['mean', 'std', 'min', 'max', 'count'],
            'sleep_hours': 'mean',
            'exercise_minutes': 'mean',
            'social_interactions': 'mean',
            'stress_level': 'mean'
        }).round(2)
        
        # Flatten column names
        monthly_data.columns = ['_'.join(col).strip() for col in monthly_data.columns.values]
        monthly_data = monthly_data.reset_index()
        
        return monthly_data
    
    def backup_data(self, data: List[Dict[str, Any]]) -> str:
        """
        Create a backup of the data
        
        Args:
            data: List of mood entries
            
        Returns:
            Path to backup file
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_filename = f"mood_data_backup_{timestamp}.json"
        backup_filepath = self.backup_dir / backup_filename
        
        try:
            with open(backup_filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
            
            logger.info(f"Created backup: {backup_filepath}")
            return str(backup_filepath)
        
        except Exception as e:
            logger.error(f"Failed to create backup: {e}")
            raise
    
    def _get_most_recent_file(self) -> str:
        """Get the most recent JSON file"""
        json_files = list(self.json_dir.glob("*.json"))
        
        if not json_files:
            raise FileNotFoundError("No JSON files found in storage directory")
        
        # Sort by modification time
        most_recent = max(json_files, key=lambda f: f.stat().st_mtime)
        return most_recent.name
    
    def list_files(self) -> Dict[str, List[str]]:
        """
        List all files in storage directories
        
        Returns:
            Dictionary with file types as keys and file lists as values
        """
        return {
            'json_files': [f.name for f in self.json_dir.glob("*.json")],
            'excel_files': [f.name for f in self.excel_dir.glob("*.xlsx")],
            'backup_files': [f.name for f in self.backup_dir.glob("*.json")]
        }
    
    def get_storage_info(self) -> Dict[str, Any]:
        """
        Get storage information
        
        Returns:
            Dictionary with storage statistics
        """
        files = self.list_files()
        
        total_files = len(files['json_files']) + len(files['excel_files']) + len(files['backup_files'])
        
        # Calculate directory sizes
        json_size = sum(f.stat().st_size for f in self.json_dir.glob("*.json"))
        excel_size = sum(f.stat().st_size for f in self.excel_dir.glob("*.xlsx"))
        backup_size = sum(f.stat().st_size for f in self.backup_dir.glob("*.json"))
        
        return {
            'storage_directory': str(self.storage_dir),
            'total_files': total_files,
            'json_files_count': len(files['json_files']),
            'excel_files_count': len(files['excel_files']),
            'backup_files_count': len(files['backup_files']),
            'json_size_bytes': json_size,
            'excel_size_bytes': excel_size,
            'backup_size_bytes': backup_size,
            'total_size_bytes': json_size + excel_size + backup_size
        }
    
    def cleanup_old_files(self, days: int = 30) -> int:
        """
        Clean up files older than specified days
        
        Args:
            days: Number of days to keep files
            
        Returns:
            Number of files deleted
        """
        cutoff_date = datetime.now() - timedelta(days=days)
        deleted_count = 0
        
        # Only clean up backup files to be safe
        for backup_file in self.backup_dir.glob("*.json"):
            if datetime.fromtimestamp(backup_file.stat().st_mtime) < cutoff_date:
                try:
                    backup_file.unlink()
                    deleted_count += 1
                    logger.info(f"Deleted old backup file: {backup_file}")
                except Exception as e:
                    logger.error(f"Failed to delete {backup_file}: {e}")
        
        logger.info(f"Cleanup complete: {deleted_count} files deleted")
        return deleted_count 