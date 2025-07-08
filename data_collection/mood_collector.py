"""
Mood Collector - Handles mood entry collection and sample data generation
"""

import json
import random
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging

logger = logging.getLogger(__name__)

class MoodCollector:
    """
    Handles mood entry collection and sample data generation
    """
    
    def __init__(self):
        self.emotions = [
            'happy', 'sad', 'anxious', 'excited', 'angry', 'calm', 'frustrated',
            'content', 'worried', 'energetic', 'depressed', 'hopeful', 'irritable',
            'peaceful', 'stressed', 'joyful', 'lonely', 'confident', 'overwhelmed'
        ]
        
        self.triggers = [
            'work_stress', 'family_issues', 'health_concerns', 'financial_worry',
            'relationship_problems', 'sleep_deprivation', 'social_isolation',
            'weather_change', 'exercise_lack', 'good_news', 'achievement',
            'social_interaction', 'relaxation', 'travel', 'creative_activity'
        ]
        
        self.weather_conditions = [
            'sunny', 'cloudy', 'rainy', 'snowy', 'windy', 'foggy', 'stormy'
        ]
        
    def collect_mood_entry(self, 
                          mood_score: int,
                          emotions: List[str],
                          sleep_hours: float,
                          exercise_minutes: int,
                          social_interactions: int,
                          stress_level: int,
                          weather: str,
                          notes: str = "",
                          triggers: List[str] = None,
                          timestamp: datetime = None) -> Dict[str, Any]:
        """
        Collect a single mood entry
        
        Args:
            mood_score: Mood rating from 1-10
            emotions: List of emotions experienced
            sleep_hours: Hours of sleep
            exercise_minutes: Minutes of exercise
            social_interactions: Number of social interactions
            stress_level: Stress level from 1-10
            weather: Weather condition
            notes: Optional notes
            triggers: List of triggers
            timestamp: Entry timestamp
            
        Returns:
            Dictionary containing mood entry data
        """
        if timestamp is None:
            timestamp = datetime.now()
            
        if triggers is None:
            triggers = []
            
        entry = {
            'timestamp': timestamp.isoformat(),
            'mood_score': mood_score,
            'emotions': emotions,
            'sleep_hours': sleep_hours,
            'exercise_minutes': exercise_minutes,
            'social_interactions': social_interactions,
            'stress_level': stress_level,
            'weather': weather,
            'notes': notes,
            'triggers': triggers,
            'entry_id': f"mood_{timestamp.strftime('%Y%m%d_%H%M%S')}"
        }
        
        logger.info(f"Mood entry collected: {entry['entry_id']}")
        return entry
    
    def generate_realistic_mood_pattern(self, days: int = 90) -> List[Dict[str, Any]]:
        """
        Generate realistic mood patterns including:
        - Seasonal patterns
        - Weekly patterns
        - Mental health indicators
        - Concerning trends
        """
        mood_entries = []
        start_date = datetime.now() - timedelta(days=days)
        
        # Base mood level with some personality variation
        base_mood = random.uniform(5.0, 7.0)
        
        # Seasonal effect (simulate seasonal affective patterns)
        seasonal_factor = 0.0
        
        # Weekly pattern (weekend effect)
        weekend_boost = 0.5
        
        # Simulate different mental health scenarios
        scenario = random.choice([
            'stable', 'mild_depression', 'anxiety_pattern', 'seasonal_depression',
            'bipolar_pattern', 'stress_period', 'recovery_pattern'
        ])
        
        logger.info(f"Generating mood pattern: {scenario}")
        
        for i in range(days):
            current_date = start_date + timedelta(days=i)
            
            # Base mood for this day
            daily_mood = base_mood
            
            # Apply scenario-specific patterns
            if scenario == 'mild_depression':
                daily_mood = self._apply_depression_pattern(daily_mood, i, days)
            elif scenario == 'anxiety_pattern':
                daily_mood = self._apply_anxiety_pattern(daily_mood, i)
            elif scenario == 'seasonal_depression':
                daily_mood = self._apply_seasonal_pattern(daily_mood, current_date)
            elif scenario == 'bipolar_pattern':
                daily_mood = self._apply_bipolar_pattern(daily_mood, i)
            elif scenario == 'stress_period':
                daily_mood = self._apply_stress_pattern(daily_mood, i, days)
            elif scenario == 'recovery_pattern':
                daily_mood = self._apply_recovery_pattern(daily_mood, i, days)
            
            # Weekend effect
            if current_date.weekday() >= 5:  # Saturday or Sunday
                daily_mood += weekend_boost
            
            # Add random variation
            daily_mood += random.uniform(-0.5, 0.5)
            
            # Ensure mood stays in valid range
            daily_mood = max(1.0, min(10.0, daily_mood))
            
            # Generate corresponding emotions based on mood
            emotions = self._generate_emotions_for_mood(daily_mood)
            
            # Generate contextual data
            sleep_hours = self._generate_sleep_hours(daily_mood)
            exercise_minutes = self._generate_exercise_minutes(daily_mood)
            social_interactions = self._generate_social_interactions(daily_mood, current_date)
            stress_level = self._generate_stress_level(daily_mood)
            weather = random.choice(self.weather_conditions)
            triggers = self._generate_triggers(daily_mood, scenario)
            
            # Create mood entry
            entry = self.collect_mood_entry(
                mood_score=int(round(daily_mood)),
                emotions=emotions,
                sleep_hours=sleep_hours,
                exercise_minutes=exercise_minutes,
                social_interactions=social_interactions,
                stress_level=stress_level,
                weather=weather,
                notes=f"Day {i+1} - {scenario} pattern",
                triggers=triggers,
                timestamp=current_date
            )
            
            mood_entries.append(entry)
        
        logger.info(f"Generated {len(mood_entries)} mood entries with {scenario} pattern")
        return mood_entries
    
    def _apply_depression_pattern(self, base_mood: float, day: int, total_days: int) -> float:
        """Apply depression pattern - sustained low mood with occasional ups"""
        # Overall downward trend
        depression_factor = -1.5 + (day / total_days) * 0.5  # Slight improvement over time
        
        # Occasional good days
        if random.random() < 0.15:  # 15% chance of better day
            depression_factor += 1.0
        
        return base_mood + depression_factor
    
    def _apply_anxiety_pattern(self, base_mood: float, day: int) -> float:
        """Apply anxiety pattern - more volatile mood with stress spikes"""
        anxiety_volatility = random.uniform(-1.0, 1.0) * 1.5
        
        # Stress spikes
        if random.random() < 0.3:  # 30% chance of stress spike
            anxiety_volatility -= 1.5
        
        return base_mood + anxiety_volatility
    
    def _apply_seasonal_pattern(self, base_mood: float, date: datetime) -> float:
        """Apply seasonal pattern - lower mood in winter months"""
        month = date.month
        if month in [11, 12, 1, 2]:  # Winter months
            seasonal_factor = -1.0
        elif month in [3, 4]:  # Spring recovery
            seasonal_factor = 0.5
        else:
            seasonal_factor = 0.0
        
        return base_mood + seasonal_factor
    
    def _apply_bipolar_pattern(self, base_mood: float, day: int) -> float:
        """Apply bipolar pattern - cycling between high and low periods"""
        cycle_length = 14  # 2-week cycles
        cycle_position = (day % cycle_length) / cycle_length
        
        if cycle_position < 0.3:  # Depressive phase
            bipolar_factor = -2.0
        elif cycle_position < 0.7:  # Stable phase
            bipolar_factor = 0.0
        else:  # Manic/hypomanic phase
            bipolar_factor = 2.0
        
        return base_mood + bipolar_factor
    
    def _apply_stress_pattern(self, base_mood: float, day: int, total_days: int) -> float:
        """Apply stress pattern - declining mood during stress period"""
        stress_peak = total_days * 0.6  # Peak stress around 60% through period
        
        if day < stress_peak:
            stress_factor = -(day / stress_peak) * 1.5
        else:
            stress_factor = -1.5 + ((day - stress_peak) / (total_days - stress_peak)) * 1.5
        
        return base_mood + stress_factor
    
    def _apply_recovery_pattern(self, base_mood: float, day: int, total_days: int) -> float:
        """Apply recovery pattern - gradual improvement over time"""
        recovery_factor = (day / total_days) * 2.0 - 1.0
        return base_mood + recovery_factor
    
    def _generate_emotions_for_mood(self, mood_score: float) -> List[str]:
        """Generate appropriate emotions based on mood score"""
        if mood_score >= 8:
            return random.sample(['happy', 'excited', 'joyful', 'confident', 'energetic'], 
                               random.randint(1, 3))
        elif mood_score >= 6:
            return random.sample(['content', 'calm', 'peaceful', 'hopeful'], 
                               random.randint(1, 2))
        elif mood_score >= 4:
            return random.sample(['worried', 'stressed', 'frustrated', 'irritable'], 
                               random.randint(1, 2))
        else:
            return random.sample(['sad', 'anxious', 'depressed', 'lonely', 'overwhelmed'], 
                               random.randint(1, 3))
    
    def _generate_sleep_hours(self, mood_score: float) -> float:
        """Generate sleep hours correlated with mood"""
        if mood_score >= 7:
            return random.uniform(7.0, 9.0)
        elif mood_score >= 5:
            return random.uniform(6.0, 8.0)
        else:
            return random.uniform(4.0, 7.0)
    
    def _generate_exercise_minutes(self, mood_score: float) -> int:
        """Generate exercise minutes correlated with mood"""
        if mood_score >= 7:
            return random.randint(20, 60)
        elif mood_score >= 5:
            return random.randint(0, 30)
        else:
            return random.randint(0, 15)
    
    def _generate_social_interactions(self, mood_score: float, date: datetime) -> int:
        """Generate social interactions correlated with mood and day of week"""
        base_interactions = 3 if date.weekday() >= 5 else 2
        
        if mood_score >= 7:
            return base_interactions + random.randint(1, 3)
        elif mood_score >= 5:
            return base_interactions + random.randint(0, 2)
        else:
            return max(0, base_interactions - random.randint(1, 2))
    
    def _generate_stress_level(self, mood_score: float) -> int:
        """Generate stress level inversely correlated with mood"""
        if mood_score >= 7:
            return random.randint(1, 4)
        elif mood_score >= 5:
            return random.randint(3, 6)
        else:
            return random.randint(6, 10)
    
    def _generate_triggers(self, mood_score: float, scenario: str) -> List[str]:
        """Generate triggers based on mood and scenario"""
        if mood_score >= 7:
            positive_triggers = ['good_news', 'achievement', 'social_interaction', 
                               'relaxation', 'creative_activity']
            return random.sample(positive_triggers, random.randint(0, 2))
        else:
            negative_triggers = ['work_stress', 'family_issues', 'health_concerns', 
                               'financial_worry', 'relationship_problems', 'sleep_deprivation']
            
            # Add scenario-specific triggers
            if scenario == 'anxiety_pattern':
                negative_triggers.extend(['social_isolation', 'weather_change'])
            elif scenario == 'stress_period':
                negative_triggers.extend(['work_stress', 'financial_worry'])
            
            return random.sample(negative_triggers, random.randint(1, 3))
    
    def collect_sample_data(self, days: int = 90) -> List[Dict[str, Any]]:
        """
        Collect sample mood data for demonstration
        """
        logger.info(f"Generating sample mood data for {days} days")
        return self.generate_realistic_mood_pattern(days)
    
    def save_to_json(self, data: List[Dict[str, Any]], filename: str = "mood_data.json"):
        """Save mood data to JSON file"""
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Mood data saved to {filename}")
    
    def load_from_json(self, filename: str = "mood_data.json") -> List[Dict[str, Any]]:
        """Load mood data from JSON file"""
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            logger.info(f"Mood data loaded from {filename}")
            return data
        except FileNotFoundError:
            logger.warning(f"File {filename} not found")
            return [] 