"""
Mental Health Mood Tracker with Insights
Complete system integrating all modules for comprehensive mood analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import json
import os
import sys
from typing import Dict, List, Any, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mental_health_tracker.log'),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import all modules
try:
    from data_collection.mood_collector import MoodCollector
    from data_collection.data_validator import DataValidator
    from data_collection.storage_manager import StorageManager
    
    from trend_analysis.time_series_analyzer import TimeSeriesAnalyzer
    from trend_analysis.trend_visualizer import TrendVisualizer
    from trend_analysis.trend_analysis_module import TrendAnalysisModule
    from trend_analysis.trend_detector import TrendDetector
    from trend_analysis.seasonal_analyzer import SeasonalAnalyzer
    
    from pattern_detection.anomaly_detector import AnomalyDetector
    from pattern_detection.pattern_recognizer import PatternRecognizer
    from pattern_detection.mental_health_indicators import MentalHealthIndicators
    
    from recommendations.insight_generator import InsightGenerator
    from recommendations.care_recommender import CareRecommender
    from recommendations.crisis_detector import CrisisDetector
    
    logger.info("All modules imported successfully")
except ImportError as e:
    logger.error(f"Import error: {e}")
    logger.info("Some modules may not be available. Continuing with available modules.")

class MentalHealthMoodTracker:
    """
    Complete Mental Health Mood Tracker System
    
    Integrates all modules to provide comprehensive mood analysis,
    pattern detection, and actionable insights with care recommendations.
    """
    
    def __init__(self):
        """Initialize the Mental Health Mood Tracker system"""
        logger.info("Initializing Mental Health Mood Tracker System")
        
        # Initialize all components
        self.mood_collector = MoodCollector()
        self.data_validator = DataValidator()
        self.storage_manager = StorageManager()
        
        self.time_series_analyzer = TimeSeriesAnalyzer()
        self.trend_visualizer = TrendVisualizer()
        self.trend_analysis_module = TrendAnalysisModule()
        self.trend_detector = TrendDetector()
        self.seasonal_analyzer = SeasonalAnalyzer()
        
        self.anomaly_detector = AnomalyDetector()
        self.pattern_recognizer = PatternRecognizer()
        self.mental_health_indicators = MentalHealthIndicators()
        
        self.insight_generator = InsightGenerator()
        self.care_recommender = CareRecommender()
        self.crisis_detector = CrisisDetector()
        
        # System state
        self.current_data = []
        self.analysis_results = {}
        self.last_analysis_date = None
        
        logger.info("Mental Health Mood Tracker System initialized successfully")
    
    def collect_mood_entry(self, 
                          mood_score: int,
                          stress_level: int,
                          sleep_hours: float,
                          exercise_minutes: int,
                          social_interactions: int,
                          emotions: List[str] = None,
                          notes: str = "",
                          timestamp: datetime = None) -> Dict[str, Any]:
        """
        Collect a new mood entry
        
        Args:
            mood_score: Mood rating 1-10
            stress_level: Stress level 1-10
            sleep_hours: Hours of sleep
            exercise_minutes: Minutes of exercise
            social_interactions: Number of social interactions
            emotions: List of emotions experienced
            notes: Additional notes
            timestamp: Entry timestamp (defaults to now)
            
        Returns:
            Dictionary with entry result and validation status
        """
        logger.info("Collecting new mood entry")
        
        try:
            # Collect the entry
            entry_result = self.mood_collector.collect_mood_entry(
                mood_score=mood_score,
                emotions=emotions or [],
                sleep_hours=sleep_hours,
                exercise_minutes=exercise_minutes,
                social_interactions=social_interactions,
                stress_level=stress_level,
                weather='unknown',  # Default weather
                notes=notes,
                timestamp=timestamp
            )
            
            # Validate the entry
            validation_result = self.data_validator.validate_entry(entry_result)
            
            if validation_result.get('valid'):
                # Load existing data
                existing_data = self.storage_manager.load_data()
                
                # Add new entry
                existing_data.append(entry_result)
                
                # Store updated data
                storage_result = self.storage_manager.save_data(existing_data)
                
                # Update current data
                self.current_data = existing_data
                
                # Check for immediate crisis indicators
                crisis_check = self._quick_crisis_check(entry_result)
                
                return {
                    'success': True,
                    'entry_id': entry_result['entry_id'],
                    'message': 'Mood entry collected and stored successfully',
                    'crisis_alert': crisis_check.get('crisis_detected', False),
                    'immediate_actions': crisis_check.get('immediate_actions', [])
                }
            else:
                return {
                    'success': False,
                    'error': 'Entry validation failed',
                    'validation_errors': validation_result.get('errors', [])
                }
        except Exception as e:
            logger.error(f"Error collecting mood entry: {e}")
            return {
                'success': False,
                'error': f'System error: {str(e)}'
            }
    
    def run_comprehensive_analysis(self, days_back: int = 30) -> Dict[str, Any]:
        """
        Run comprehensive analysis of mood data
        
        Args:
            days_back: Number of days to analyze (default 30)
            
        Returns:
            Complete analysis results
        """
        logger.info(f"Running comprehensive analysis for last {days_back} days")
        
        try:
            # Get recent data
            if not self.current_data:
                self.current_data = self.storage_manager.load_data()
            
            if len(self.current_data) < 7:
                return {
                    'success': False,
                    'error': 'Insufficient data for analysis (minimum 7 entries required)',
                    'current_entries': len(self.current_data)
                }
            
            # Filter data for analysis period
            cutoff_date = datetime.now() - timedelta(days=days_back)
            recent_data = [
                entry for entry in self.current_data 
                if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None) >= cutoff_date
            ]
            
            logger.info(f"Analyzing {len(recent_data)} entries from last {days_back} days")
            
            # 1. Trend Analysis
            logger.info("Running trend analysis...")
            trend_results = self._run_trend_analysis(recent_data)
            
            # 2. Pattern Detection
            logger.info("Running pattern detection...")
            pattern_results = self._run_pattern_detection(recent_data)
            
            # 3. Anomaly Detection
            logger.info("Running anomaly detection...")
            anomaly_results = self._run_anomaly_detection(recent_data)
            
            # 4. Mental Health Indicators
            logger.info("Analyzing mental health indicators...")
            mental_health_results = self._run_mental_health_analysis(recent_data)
            
            # 5. Crisis Detection
            logger.info("Running crisis detection...")
            crisis_results = self._run_crisis_detection(recent_data, anomaly_results, mental_health_results)
            
            # 6. Generate Insights
            logger.info("Generating insights...")
            insights_results = self._generate_insights(
                trend_results, pattern_results, anomaly_results, mental_health_results
            )
            
            # 7. Care Recommendations
            logger.info("Generating care recommendations...")
            care_results = self._generate_care_recommendations(
                mental_health_results, anomaly_results, pattern_results
            )
            
            # Compile comprehensive results
            analysis_results = {
                'analysis_metadata': {
                    'analysis_date': datetime.now().isoformat(),
                    'data_points_analyzed': len(recent_data),
                    'analysis_period_days': days_back,
                    'date_range': {
                        'start': min(entry['timestamp'] for entry in recent_data),
                        'end': max(entry['timestamp'] for entry in recent_data)
                    }
                },
                'trend_analysis': trend_results,
                'pattern_detection': pattern_results,
                'anomaly_detection': anomaly_results,
                'mental_health_indicators': mental_health_results,
                'crisis_detection': crisis_results,
                'insights': insights_results,
                'care_recommendations': care_results,
                'overall_summary': self._generate_overall_summary(
                    trend_results, pattern_results, anomaly_results, 
                    mental_health_results, crisis_results, insights_results, care_results
                )
            }
            
            # Store analysis results
            self.analysis_results = analysis_results
            self.last_analysis_date = datetime.now()
            
            # Save analysis to file
            self._save_analysis_results(analysis_results)
            
            logger.info("Comprehensive analysis completed successfully")
            
            return {
                'success': True,
                'analysis_results': analysis_results
            }
        
        except Exception as e:
            logger.error(f"Error running comprehensive analysis: {e}")
            return {
                'success': False,
                'error': f'Analysis error: {str(e)}'
            }
    
    def get_daily_insights(self) -> Dict[str, Any]:
        """Get daily insights and recommendations"""
        logger.info("Generating daily insights")
        
        try:
            if not self.current_data:
                self.current_data = self.storage_manager.load_data()
            
            if len(self.current_data) == 0:
                return {
                    'success': False,
                    'message': 'No data available for insights'
                }
            
            # Get recent data (last 7 days)
            recent_data = self.current_data[-7:] if len(self.current_data) >= 7 else self.current_data
            
            # Quick analysis for daily insights
            daily_insights = {
                'current_mood_trend': self._analyze_recent_mood_trend(recent_data),
                'stress_pattern': self._analyze_stress_pattern(recent_data),
                'sleep_quality': self._analyze_sleep_quality(recent_data),
                'daily_recommendations': self._generate_daily_recommendations(recent_data),
                'warning_signs': self._check_warning_signs(recent_data),
                'positive_patterns': self._identify_positive_patterns(recent_data)
            }
            
            return {
                'success': True,
                'daily_insights': daily_insights,
                'data_points': len(recent_data)
            }
        
        except Exception as e:
            logger.error(f"Error generating daily insights: {e}")
            return {
                'success': False,
                'error': f'Daily insights error: {str(e)}'
            }
    
    def get_crisis_status(self) -> Dict[str, Any]:
        """Check current crisis status"""
        logger.info("Checking crisis status")
        
        try:
            if not self.current_data:
                self.current_data = self.storage_manager.load_data()
            
            if len(self.current_data) == 0:
                return {
                    'crisis_level': 'unknown',
                    'message': 'No data available for crisis assessment'
                }
            
            # Get recent entries (last 3 days)
            recent_entries = self.current_data[-3:] if len(self.current_data) >= 3 else self.current_data
            
            # Quick crisis assessment
            crisis_status = self.crisis_detector.detect_crisis(
                recent_entries,
                anomaly_results=None,
                mental_health_results=None
            )
            
            return {
                'success': True,
                'crisis_status': crisis_status
            }
        
        except Exception as e:
            logger.error(f"Error checking crisis status: {e}")
            return {
                'success': False,
                'error': f'Crisis status error: {str(e)}'
            }
    
    def export_data(self, format: str = 'excel', filename: str = None) -> Dict[str, Any]:
        """
        Export mood tracking data
        
        Args:
            format: Export format ('excel', 'json', 'csv')
            filename: Custom filename (optional)
            
        Returns:
            Export result
        """
        logger.info(f"Exporting data in {format} format")
        
        try:
            if not self.current_data:
                self.current_data = self.storage_manager.load_data()
            
            if format.lower() == 'excel':
                result = self.storage_manager.export_to_excel(self.current_data, filename)
                return {
                    'success': True,
                    'filename': result
                }
            elif format.lower() == 'json':
                result = self.storage_manager.save_data(self.current_data, filename)
                return {
                    'success': True,
                    'filename': result
                }
            else:
                return {
                    'success': False,
                    'error': f'Unsupported export format: {format}'
                }
        
        except Exception as e:
            logger.error(f"Error exporting data: {e}")
            return {
                'success': False,
                'error': f'Export error: {str(e)}'
            }
    
    def generate_report(self, report_type: str = 'comprehensive') -> Dict[str, Any]:
        """
        Generate different types of reports
        
        Args:
            report_type: Type of report ('comprehensive', 'summary', 'crisis', 'trends')
            
        Returns:
            Generated report
        """
        logger.info(f"Generating {report_type} report")
        
        try:
            if report_type == 'comprehensive':
                return self._generate_comprehensive_report()
            elif report_type == 'summary':
                return self._generate_summary_report()
            elif report_type == 'crisis':
                return self._generate_crisis_report()
            elif report_type == 'trends':
                return self._generate_trends_report()
            else:
                return {
                    'success': False,
                    'error': f'Unsupported report type: {report_type}'
                }
        
        except Exception as e:
            logger.error(f"Error generating report: {e}")
            return {
                'success': False,
                'error': f'Report generation error: {str(e)}'
            }
    
    # Private helper methods
    def _quick_crisis_check(self, entry: Dict[str, Any]) -> Dict[str, Any]:
        """Quick crisis check for new entries"""
        crisis_indicators = []
        
        if entry.get('mood_score', 5) <= 2:
            crisis_indicators.append('Severe low mood detected')
        
        if entry.get('stress_level', 5) >= 9:
            crisis_indicators.append('Extreme stress level detected')
        
        if entry.get('sleep_hours', 7) < 3:
            crisis_indicators.append('Severe sleep deprivation detected')
        
        emotions = entry.get('emotions', [])
        crisis_emotions = ['suicidal', 'hopeless', 'worthless', 'trapped']
        if any(emotion in crisis_emotions for emotion in emotions):
            crisis_indicators.append('Concerning emotional state detected')
        
        crisis_detected = len(crisis_indicators) > 0
        
        immediate_actions = []
        if crisis_detected:
            immediate_actions = [
                'Consider contacting crisis hotline (988)',
                'Reach out to trusted friend or family member',
                'Contact mental health professional if available',
                'Use emergency coping strategies'
            ]
        
        return {
            'crisis_detected': crisis_detected,
            'crisis_indicators': crisis_indicators,
            'immediate_actions': immediate_actions
        }
    
    def _run_trend_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run trend analysis on data"""
        try:
            return self.trend_analysis_module.analyze_trends(data)
        except Exception as e:
            logger.error(f"Trend analysis error: {e}")
            return {'error': str(e)}
    
    def _run_pattern_detection(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run pattern detection on data"""
        try:
            return self.pattern_recognizer.recognize_patterns(data)
        except Exception as e:
            logger.error(f"Pattern detection error: {e}")
            return {'error': str(e)}
    
    def _run_anomaly_detection(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run anomaly detection on data"""
        try:
            return self.anomaly_detector.detect_anomalies(data)
        except Exception as e:
            logger.error(f"Anomaly detection error: {e}")
            return {'error': str(e)}
    
    def _run_mental_health_analysis(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Run mental health indicators analysis"""
        try:
            return self.mental_health_indicators.analyze_mental_health_indicators(data)
        except Exception as e:
            logger.error(f"Mental health analysis error: {e}")
            return {'error': str(e)}
    
    def _run_crisis_detection(self, data: List[Dict[str, Any]], 
                            anomaly_results: Dict[str, Any],
                            mental_health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Run crisis detection"""
        try:
            return self.crisis_detector.detect_crisis(data, anomaly_results, mental_health_results)
        except Exception as e:
            logger.error(f"Crisis detection error: {e}")
            return {'error': str(e)}
    
    def _generate_insights(self, trend_results: Dict[str, Any],
                          pattern_results: Dict[str, Any],
                          anomaly_results: Dict[str, Any],
                          mental_health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from analysis results"""
        try:
            return self.insight_generator.generate_insights(
                trend_results, pattern_results, anomaly_results, mental_health_results
            )
        except Exception as e:
            logger.error(f"Insights generation error: {e}")
            return {'error': str(e)}
    
    def _generate_care_recommendations(self, mental_health_results: Dict[str, Any],
                                     anomaly_results: Dict[str, Any],
                                     pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate care recommendations"""
        try:
            return self.care_recommender.generate_care_recommendations(
                mental_health_results, anomaly_results, pattern_results
            )
        except Exception as e:
            logger.error(f"Care recommendations error: {e}")
            return {'error': str(e)}
    
    def _generate_overall_summary(self, *args) -> Dict[str, Any]:
        """Generate overall summary of all analysis results"""
        trend_results, pattern_results, anomaly_results, mental_health_results, crisis_results, insights_results, care_results = args
        
        summary = {
            'overall_status': 'stable',
            'key_findings': [],
            'priority_recommendations': [],
            'crisis_level': crisis_results.get('crisis_level', 'low'),
            'mental_health_risk': mental_health_results.get('overall_risk_assessment', {}).get('overall_risk_level', 'low'),
            'immediate_actions_needed': False
        }
        
        # Determine overall status
        crisis_level = crisis_results.get('crisis_level', 'low')
        mental_health_risk = mental_health_results.get('overall_risk_assessment', {}).get('overall_risk_level', 'low')
        
        if crisis_level in ['critical', 'high'] or mental_health_risk == 'high':
            summary['overall_status'] = 'concerning'
            summary['immediate_actions_needed'] = True
        elif crisis_level == 'moderate' or mental_health_risk == 'moderate':
            summary['overall_status'] = 'needs_attention'
        
        # Extract key findings
        if 'mood_insights' in insights_results:
            for insight in insights_results['mood_insights'][:3]:  # Top 3 insights
                summary['key_findings'].append(insight.get('title', 'Mood insight detected'))
        
        # Extract priority recommendations
        if 'priority_actions' in insights_results:
            for action in insights_results['priority_actions'][:3]:  # Top 3 actions
                summary['priority_recommendations'].append(action.get('action', 'Priority action needed'))
        
        return summary
    
    def _save_analysis_results(self, results: Dict[str, Any]) -> None:
        """Save analysis results to file"""
        try:
            os.makedirs('analysis_results', exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"analysis_results/mental_health_analysis_{timestamp}.json"
            
            with open(filename, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            
            logger.info(f"Analysis results saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving analysis results: {e}")
    
    def _analyze_recent_mood_trend(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze recent mood trend"""
        if len(data) < 3:
            return {'trend': 'insufficient_data'}
        
        moods = [entry.get('mood_score', 5) for entry in data[-3:]]
        
        if moods[-1] > moods[0] + 1:
            return {'trend': 'improving', 'description': 'Mood has been improving recently'}
        elif moods[-1] < moods[0] - 1:
            return {'trend': 'declining', 'description': 'Mood has been declining recently'}
        else:
            return {'trend': 'stable', 'description': 'Mood has been relatively stable'}
    
    def _analyze_stress_pattern(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze stress patterns"""
        if not data:
            return {'pattern': 'no_data'}
        
        stress_levels = [entry.get('stress_level', 5) for entry in data]
        avg_stress = sum(stress_levels) / len(stress_levels)
        
        if avg_stress >= 7:
            return {'pattern': 'high_stress', 'average': avg_stress}
        elif avg_stress <= 3:
            return {'pattern': 'low_stress', 'average': avg_stress}
        else:
            return {'pattern': 'moderate_stress', 'average': avg_stress}
    
    def _analyze_sleep_quality(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze sleep quality"""
        if not data:
            return {'quality': 'no_data'}
        
        sleep_hours = [entry.get('sleep_hours', 7) for entry in data]
        avg_sleep = sum(sleep_hours) / len(sleep_hours)
        
        if 7 <= avg_sleep <= 9:
            return {'quality': 'good', 'average_hours': avg_sleep}
        elif avg_sleep < 6:
            return {'quality': 'insufficient', 'average_hours': avg_sleep}
        else:
            return {'quality': 'excessive', 'average_hours': avg_sleep}
    
    def _generate_daily_recommendations(self, data: List[Dict[str, Any]]) -> List[str]:
        """Generate daily recommendations"""
        recommendations = []
        
        if not data:
            return ['Start tracking your mood daily for personalized insights']
        
        latest_entry = data[-1]
        
        # Mood-based recommendations
        mood = latest_entry.get('mood_score', 5)
        if mood <= 4:
            recommendations.append('Consider practicing self-care activities today')
            recommendations.append('Reach out to a friend or family member')
        elif mood >= 8:
            recommendations.append('Great mood! Consider planning something enjoyable')
        
        # Sleep-based recommendations
        sleep = latest_entry.get('sleep_hours', 7)
        if sleep < 6:
            recommendations.append('Prioritize getting adequate sleep tonight')
        elif sleep > 10:
            recommendations.append('Try to maintain a consistent sleep schedule')
        
        # Exercise recommendations
        exercise = latest_entry.get('exercise_minutes', 30)
        if exercise < 15:
            recommendations.append('Add some physical activity to your day, even 10 minutes helps')
        
        # Stress management
        stress = latest_entry.get('stress_level', 5)
        if stress >= 7:
            recommendations.append('Practice stress reduction techniques today')
        
        return recommendations[:3]  # Limit to top 3 recommendations
    
    def _check_warning_signs(self, data: List[Dict[str, Any]]) -> List[str]:
        """Check for warning signs"""
        warnings = []
        
        if not data:
            return warnings
        
        # Check recent trends
        if len(data) >= 3:
            recent_moods = [entry.get('mood_score', 5) for entry in data[-3:]]
            if all(mood <= 4 for mood in recent_moods):
                warnings.append('Persistent low mood for 3+ days')
        
        # Check latest entry
        latest = data[-1]
        if latest.get('mood_score', 5) <= 2:
            warnings.append('Severe low mood detected')
        if latest.get('stress_level', 5) >= 9:
            warnings.append('Extreme stress level detected')
        if latest.get('sleep_hours', 7) < 4:
            warnings.append('Significant sleep deprivation')
        
        return warnings
    
    def _identify_positive_patterns(self, data: List[Dict[str, Any]]) -> List[str]:
        """Identify positive patterns"""
        positives = []
        
        if not data:
            return positives
        
        # Check for good mood days
        good_mood_days = sum(1 for entry in data if entry.get('mood_score', 5) >= 7)
        if good_mood_days >= len(data) * 0.6:
            positives.append(f'Good mood {good_mood_days}/{len(data)} days recently')
        
        # Check exercise consistency
        exercise_days = sum(1 for entry in data if entry.get('exercise_minutes', 0) > 0)
        if exercise_days >= len(data) * 0.7:
            positives.append(f'Consistent exercise {exercise_days}/{len(data)} days')
        
        # Check sleep quality
        good_sleep_days = sum(1 for entry in data if 7 <= entry.get('sleep_hours', 7) <= 9)
        if good_sleep_days >= len(data) * 0.7:
            positives.append(f'Good sleep {good_sleep_days}/{len(data)} days')
        
        return positives
    
    def _generate_comprehensive_report(self) -> Dict[str, Any]:
        """Generate comprehensive report"""
        if not self.analysis_results:
            return {
                'success': False,
                'message': 'No analysis results available. Run comprehensive analysis first.'
            }
        
        return {
            'success': True,
            'report_type': 'comprehensive',
            'generated_at': datetime.now().isoformat(),
            'report': self.analysis_results
        }
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report"""
        if not self.analysis_results:
            return self.get_daily_insights()
        
        summary = self.analysis_results.get('overall_summary', {})
        key_insights = self.analysis_results.get('insights', {})
        
        return {
            'success': True,
            'report_type': 'summary',
            'generated_at': datetime.now().isoformat(),
            'summary': summary,
            'key_insights': key_insights.get('mood_insights', [])[:3],
            'recommendations': key_insights.get('personalized_recommendations', [])[:3]
        }
    
    def _generate_crisis_report(self) -> Dict[str, Any]:
        """Generate crisis report"""
        crisis_status = self.get_crisis_status()
        
        if not crisis_status.get('success'):
            return crisis_status
        
        return {
            'success': True,
            'report_type': 'crisis',
            'generated_at': datetime.now().isoformat(),
            'crisis_assessment': crisis_status['crisis_status']
        }
    
    def _generate_trends_report(self) -> Dict[str, Any]:
        """Generate trends report"""
        if not self.analysis_results:
            return {
                'success': False,
                'message': 'No analysis results available. Run comprehensive analysis first.'
            }
        
        return {
            'success': True,
            'report_type': 'trends',
            'generated_at': datetime.now().isoformat(),
            'trend_analysis': self.analysis_results.get('trend_analysis', {}),
            'pattern_detection': self.analysis_results.get('pattern_detection', {})
        }


def main():
    """
    Main function to demonstrate the Mental Health Mood Tracker system
    """
    print("🧠 Mental Health Mood Tracker with Insights")
    print("=" * 60)
    
    # Initialize the system
    tracker = MentalHealthMoodTracker()
    
    # Sample usage demonstration
    print("\n📝 Collecting sample mood entries...")
    
    # Sample entries
    sample_entries = [
        {
            'mood_score': 7,
            'stress_level': 4,
            'sleep_hours': 8.0,
            'exercise_minutes': 30,
            'social_interactions': 3,
            'emotions': ['content', 'motivated'],
            'notes': 'Good day overall'
        },
        {
            'mood_score': 5,
            'stress_level': 6,
            'sleep_hours': 6.5,
            'exercise_minutes': 0,
            'social_interactions': 1,
            'emotions': ['tired', 'stressed'],
            'notes': 'Busy day at work'
        },
        {
            'mood_score': 8,
            'stress_level': 3,
            'sleep_hours': 8.5,
            'exercise_minutes': 45,
            'social_interactions': 4,
            'emotions': ['happy', 'energetic'],
            'notes': 'Great workout and social time'
        }
    ]
    
    # Collect sample entries
    for i, entry in enumerate(sample_entries):
        # Use different timestamps for each entry
        entry_time = datetime.now() - timedelta(days=len(sample_entries)-i-1)
        result = tracker.collect_mood_entry(timestamp=entry_time, **entry)
        
        if result['success']:
            print(f"✅ Entry {i+1} collected successfully")
            if result.get('crisis_alert'):
                print(f"⚠️  Crisis alert: {result['immediate_actions']}")
        else:
            print(f"❌ Failed to collect entry {i+1}: {result['error']}")
    
    print(f"\n📊 Running comprehensive analysis...")
    
    # Run comprehensive analysis
    analysis_result = tracker.run_comprehensive_analysis(days_back=30)
    
    if analysis_result['success']:
        print("✅ Comprehensive analysis completed successfully")
        
        # Display summary
        summary = analysis_result['analysis_results']['overall_summary']
        print(f"\n📋 Overall Status: {summary['overall_status'].upper()}")
        print(f"🎯 Crisis Level: {summary['crisis_level']}")
        print(f"🧠 Mental Health Risk: {summary['mental_health_risk']}")
        
        if summary['immediate_actions_needed']:
            print("🚨 IMMEDIATE ACTIONS NEEDED")
        
        print(f"\n🔍 Key Findings:")
        for finding in summary['key_findings'][:3]:
            print(f"  • {finding}")
        
        print(f"\n💡 Priority Recommendations:")
        for rec in summary['priority_recommendations'][:3]:
            print(f"  • {rec}")
    else:
        print(f"❌ Analysis failed: {analysis_result['error']}")
    
    print(f"\n🌅 Getting daily insights...")
    
    # Get daily insights
    daily_insights = tracker.get_daily_insights()
    
    if daily_insights['success']:
        insights = daily_insights['daily_insights']
        print(f"✅ Daily insights generated")
        print(f"📈 Recent mood trend: {insights['current_mood_trend']['trend']}")
        print(f"😰 Stress pattern: {insights['stress_pattern']['pattern']}")
        print(f"💤 Sleep quality: {insights['sleep_quality']['quality']}")
        
        if insights['warning_signs']:
            print(f"\n⚠️  Warning signs:")
            for warning in insights['warning_signs']:
                print(f"  • {warning}")
        
        if insights['positive_patterns']:
            print(f"\n✨ Positive patterns:")
            for positive in insights['positive_patterns']:
                print(f"  • {positive}")
        
        print(f"\n📝 Daily recommendations:")
        for rec in insights['daily_recommendations']:
            print(f"  • {rec}")
    else:
        print(f"❌ Daily insights failed: {daily_insights['error']}")
    
    print(f"\n🛡️  Checking crisis status...")
    
    # Check crisis status
    crisis_status = tracker.get_crisis_status()
    
    if crisis_status['success']:
        crisis_info = crisis_status['crisis_status']
        print(f"✅ Crisis assessment completed")
        print(f"🚨 Crisis level: {crisis_info['crisis_level']}")
        
        if crisis_info.get('immediate_action_plan'):
            action_plan = crisis_info['immediate_action_plan']
            if action_plan['immediate_steps']:
                print(f"\n🚨 Immediate steps needed:")
                for step in action_plan['immediate_steps'][:3]:
                    print(f"  • {step}")
    else:
        print(f"❌ Crisis status check failed: {crisis_status['error']}")
    
    print(f"\n💾 Exporting data...")
    
    # Export data
    export_result = tracker.export_data('excel')
    
    if export_result['success']:
        print(f"✅ Data exported to: {export_result['filename']}")
    else:
        print(f"❌ Export failed: {export_result['error']}")
    
    print(f"\n🎉 Mental Health Mood Tracker demonstration completed!")
    print(f"📁 Check the generated files for detailed analysis results.")
    print(f"\n💡 This system provides:")
    print(f"  • Comprehensive mood tracking and validation")
    print(f"  • Advanced trend and pattern analysis")
    print(f"  • Mental health indicators and risk assessment")
    print(f"  • Crisis detection and intervention recommendations")
    print(f"  • Personalized insights and care recommendations")
    print(f"  • Professional-grade analysis with actionable insights")


if __name__ == "__main__":
    main() 