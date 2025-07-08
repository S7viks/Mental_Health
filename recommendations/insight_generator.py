"""
Insight Generator - Generate actionable insights from mood tracking analysis
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class InsightGenerator:
    """
    Generate actionable insights from mood tracking analysis results
    """
    
    def __init__(self):
        self.insights_results = {}
        self.personalized_insights = {}
        
    def generate_insights(self, 
                         trend_results: Dict[str, Any] = None,
                         pattern_results: Dict[str, Any] = None,
                         anomaly_results: Dict[str, Any] = None,
                         mental_health_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive insights from all analysis results
        
        Args:
            trend_results: Results from trend analysis
            pattern_results: Results from pattern recognition
            anomaly_results: Results from anomaly detection
            mental_health_results: Results from mental health indicators
            
        Returns:
            Dictionary with comprehensive insights
        """
        logger.info("Starting insight generation")
        
        insights = {
            'mood_insights': self._generate_mood_insights(trend_results, pattern_results),
            'behavioral_insights': self._generate_behavioral_insights(pattern_results, anomaly_results),
            'temporal_insights': self._generate_temporal_insights(pattern_results),
            'health_insights': self._generate_health_insights(mental_health_results),
            'predictive_insights': self._generate_predictive_insights(pattern_results, trend_results),
            'personalized_recommendations': self._generate_personalized_recommendations(
                trend_results, pattern_results, anomaly_results, mental_health_results
            ),
            'priority_actions': self._identify_priority_actions(
                trend_results, pattern_results, anomaly_results, mental_health_results
            ),
            'progress_insights': self._generate_progress_insights(trend_results),
            'success_patterns': self._identify_success_patterns(pattern_results, trend_results)
        }
        
        self.insights_results = insights
        logger.info("Insight generation completed")
        
        return insights
    
    def _generate_mood_insights(self, trend_results: Dict[str, Any], pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about mood patterns and trends"""
        insights = []
        
        # Trend insights
        if trend_results and 'mood_score' in trend_results:
            mood_trend = trend_results['mood_score'].get('overall_trend', {})
            
            if mood_trend.get('direction') == 'increasing':
                insights.append({
                    'type': 'positive_trend',
                    'title': 'Improving Mood Trend',
                    'description': f"Your mood has been improving over time with an R² of {mood_trend.get('r_squared', 0):.3f}",
                    'actionable': 'Continue your current strategies as they appear to be working well',
                    'confidence': 'high' if mood_trend.get('r_squared', 0) > 0.5 else 'medium'
                })
            elif mood_trend.get('direction') == 'decreasing':
                insights.append({
                    'type': 'concerning_trend',
                    'title': 'Declining Mood Trend',
                    'description': f"Your mood has been declining over time with an R² of {mood_trend.get('r_squared', 0):.3f}",
                    'actionable': 'Consider implementing mood-boosting activities and evaluate recent life changes',
                    'confidence': 'high' if mood_trend.get('r_squared', 0) > 0.5 else 'medium'
                })
        
        # Pattern insights
        if pattern_results and 'mood_patterns' in pattern_results:
            mood_patterns = pattern_results['mood_patterns']
            
            # Mood cycles
            if 'mood_cycles' in mood_patterns:
                cycles = mood_patterns['mood_cycles']
                if cycles.get('average_cycle_length', 0) > 0:
                    insights.append({
                        'type': 'pattern_discovery',
                        'title': 'Mood Cycle Pattern',
                        'description': f"Your mood follows a cycle of approximately {cycles['average_cycle_length']:.1f} days",
                        'actionable': 'Plan extra self-care during predicted low periods in your cycle',
                        'confidence': 'medium'
                    })
            
            # Mood stability
            if 'stability_patterns' in mood_patterns:
                stability = mood_patterns['stability_patterns']
                cv = stability.get('coefficient_of_variation', 0)
                
                if cv < 0.2:
                    insights.append({
                        'type': 'positive_pattern',
                        'title': 'Excellent Mood Stability',
                        'description': 'Your mood is very stable with minimal day-to-day variation',
                        'actionable': 'Continue maintaining your current routine and stress management',
                        'confidence': 'high'
                    })
                elif cv > 0.4:
                    insights.append({
                        'type': 'improvement_opportunity',
                        'title': 'High Mood Variability',
                        'description': 'Your mood shows significant day-to-day variation',
                        'actionable': 'Focus on consistent sleep, exercise, and stress management routines',
                        'confidence': 'high'
                    })
        
        return insights
    
    def _generate_behavioral_insights(self, pattern_results: Dict[str, Any], anomaly_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about behavioral patterns"""
        insights = []
        
        # Sleep insights
        if pattern_results and 'behavioral_patterns' in pattern_results:
            behavioral = pattern_results['behavioral_patterns']
            
            if 'sleep_patterns' in behavioral:
                sleep = behavioral['sleep_patterns']
                duration = sleep.get('duration_stats', {})
                
                if duration.get('optimal_percentage', 0) > 80:
                    insights.append({
                        'type': 'positive_behavior',
                        'title': 'Excellent Sleep Habits',
                        'description': f"You get optimal sleep (7-9 hours) {duration['optimal_percentage']:.1f}% of the time",
                        'actionable': 'Keep up your excellent sleep routine - it\'s supporting your mental health',
                        'confidence': 'high'
                    })
                elif duration.get('optimal_percentage', 0) < 50:
                    insights.append({
                        'type': 'improvement_opportunity',
                        'title': 'Sleep Duration Needs Attention',
                        'description': f"You only get optimal sleep {duration['optimal_percentage']:.1f}% of the time",
                        'actionable': 'Prioritize sleep hygiene: consistent bedtime, dark room, no screens before bed',
                        'confidence': 'high'
                    })
            
            # Exercise insights
            if 'exercise_patterns' in behavioral:
                exercise = behavioral['exercise_patterns']
                frequency = exercise.get('frequency', {})
                
                if frequency.get('percentage_active_days', 0) > 70:
                    insights.append({
                        'type': 'positive_behavior',
                        'title': 'Great Exercise Consistency',
                        'description': f"You exercise {frequency['percentage_active_days']:.1f}% of days",
                        'actionable': 'Your exercise routine is excellent for mental health - maintain it!',
                        'confidence': 'high'
                    })
                elif frequency.get('percentage_active_days', 0) < 30:
                    insights.append({
                        'type': 'improvement_opportunity',
                        'title': 'Low Exercise Frequency',
                        'description': f"You only exercise {frequency['percentage_active_days']:.1f}% of days",
                        'actionable': 'Start with 10-15 minutes of daily movement - even walking helps mood',
                        'confidence': 'high'
                    })
        
        # Anomaly-based behavioral insights
        if anomaly_results and 'behavioral_anomalies' in anomaly_results:
            behavioral_anomalies = anomaly_results['behavioral_anomalies']
            
            if 'social_isolation' in behavioral_anomalies:
                isolation = behavioral_anomalies['social_isolation']
                if isolation.get('severity') == 'high':
                    insights.append({
                        'type': 'concerning_behavior',
                        'title': 'Social Isolation Detected',
                        'description': f"Extended periods of social isolation detected ({isolation['count']} instances)",
                        'actionable': 'Schedule regular social activities, even brief check-ins with friends',
                        'confidence': 'high'
                    })
        
        return insights
    
    def _generate_temporal_insights(self, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about temporal patterns"""
        insights = []
        
        if not pattern_results or 'temporal_patterns' not in pattern_results:
            return insights
        
        temporal = pattern_results['temporal_patterns']
        
        # Day of week insights
        if 'day_of_week_patterns' in temporal:
            dow = temporal['day_of_week_patterns']
            best_day = dow.get('best_day')
            worst_day = dow.get('worst_day')
            
            if best_day and worst_day and best_day != worst_day:
                insights.append({
                    'type': 'temporal_pattern',
                    'title': 'Weekly Mood Pattern',
                    'description': f"Your mood is typically best on {best_day}s and lowest on {worst_day}s",
                    'actionable': f"Plan challenging tasks for {best_day}s and extra self-care for {worst_day}s",
                    'confidence': 'medium'
                })
        
        # Time of day insights
        if 'time_of_day_patterns' in temporal:
            tod = temporal['time_of_day_patterns']
            peak_hour = tod.get('peak_hour')
            lowest_hour = tod.get('lowest_hour')
            
            if peak_hour is not None and lowest_hour is not None:
                insights.append({
                    'type': 'temporal_pattern',
                    'title': 'Daily Energy Rhythm',
                    'description': f"Your mood peaks around {peak_hour}:00 and is lowest around {lowest_hour}:00",
                    'actionable': f"Schedule important tasks around {peak_hour}:00 and practice self-care around {lowest_hour}:00",
                    'confidence': 'medium'
                })
        
        # Seasonal insights
        if 'seasonal_patterns' in temporal:
            seasonal = temporal['seasonal_patterns']
            best_season = seasonal.get('best_season')
            worst_season = seasonal.get('worst_season')
            
            if best_season and worst_season and best_season != worst_season:
                insights.append({
                    'type': 'seasonal_pattern',
                    'title': 'Seasonal Mood Variation',
                    'description': f"Your mood is typically best in {best_season} and lowest in {worst_season}",
                    'actionable': f"Prepare extra support strategies for {worst_season} and leverage {best_season} energy",
                    'confidence': 'medium'
                })
        
        return insights
    
    def _generate_health_insights(self, mental_health_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate health-related insights"""
        insights = []
        
        if not mental_health_results:
            return insights
        
        # Overall risk assessment
        if 'overall_risk_assessment' in mental_health_results:
            risk = mental_health_results['overall_risk_assessment']
            risk_level = risk.get('overall_risk_level', 'low')
            
            if risk_level == 'high':
                insights.append({
                    'type': 'health_concern',
                    'title': 'High Mental Health Risk Detected',
                    'description': 'Multiple indicators suggest you may benefit from professional support',
                    'actionable': 'Consider scheduling an appointment with a mental health professional',
                    'confidence': 'high'
                })
            elif risk_level == 'moderate':
                insights.append({
                    'type': 'health_awareness',
                    'title': 'Moderate Mental Health Risk',
                    'description': 'Some indicators suggest monitoring your mental health more closely',
                    'actionable': 'Implement stress management techniques and consider professional consultation',
                    'confidence': 'medium'
                })
        
        # Specific condition insights
        conditions = ['depression_indicators', 'anxiety_indicators', 'stress_indicators']
        
        for condition in conditions:
            if condition in mental_health_results:
                condition_data = mental_health_results[condition]
                risk_score_key = condition.replace('_indicators', '_risk_score')
                
                if risk_score_key in condition_data:
                    risk_info = condition_data[risk_score_key]
                    if risk_info.get('risk_level') == 'high':
                        condition_name = condition.split('_')[0].title()
                        insights.append({
                            'type': 'health_indicator',
                            'title': f'{condition_name} Indicators Present',
                            'description': f'Pattern analysis suggests elevated {condition_name.lower()} indicators',
                            'actionable': f'Consider {condition_name.lower()}-specific interventions and professional evaluation',
                            'confidence': 'medium'
                        })
        
        return insights
    
    def _generate_predictive_insights(self, pattern_results: Dict[str, Any], trend_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate predictive insights about future mood patterns"""
        insights = []
        
        # Leading indicators
        if pattern_results and 'predictive_patterns' in pattern_results:
            predictive = pattern_results['predictive_patterns']
            
            if 'leading_indicators' in predictive:
                indicators = predictive['leading_indicators']
                
                for variable, info in indicators.items():
                    correlation = info.get('correlation', 0)
                    if abs(correlation) > 0.4:
                        direction = 'positively' if correlation > 0 else 'negatively'
                        var_name = variable.replace('_', ' ').title()
                        
                        insights.append({
                            'type': 'predictive_pattern',
                            'title': f'{var_name} Predicts Tomorrow\'s Mood',
                            'description': f'Your {variable.replace("_", " ")} today {direction} predicts tomorrow\'s mood',
                            'actionable': f'Pay attention to your {variable.replace("_", " ")} to anticipate mood changes',
                            'confidence': 'medium'
                        })
        
        # Trend forecasting
        if trend_results and 'mood_score' in trend_results:
            if 'forecast' in trend_results['mood_score']:
                forecast = trend_results['mood_score']['forecast']
                
                # Simple forecast interpretation
                if 'forecast' in forecast:
                    forecast_values = forecast['forecast']
                    if len(forecast_values) > 0:
                        trend_direction = 'improving' if forecast_values[-1] > forecast_values[0] else 'declining'
                        
                        insights.append({
                            'type': 'forecast',
                            'title': f'Mood Forecast: {trend_direction.title()}',
                            'description': f'Based on current patterns, your mood trend appears to be {trend_direction}',
                            'actionable': 'Use this forecast to plan appropriate self-care strategies',
                            'confidence': 'low'
                        })
        
        return insights
    
    def _generate_personalized_recommendations(self, 
                                             trend_results: Dict[str, Any],
                                             pattern_results: Dict[str, Any],
                                             anomaly_results: Dict[str, Any],
                                             mental_health_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate personalized recommendations based on individual patterns"""
        recommendations = []
        
        # Sleep recommendations
        sleep_rec = self._generate_sleep_recommendations(pattern_results, anomaly_results)
        recommendations.extend(sleep_rec)
        
        # Exercise recommendations
        exercise_rec = self._generate_exercise_recommendations(pattern_results, trend_results)
        recommendations.extend(exercise_rec)
        
        # Stress management recommendations
        stress_rec = self._generate_stress_recommendations(mental_health_results, pattern_results)
        recommendations.extend(stress_rec)
        
        # Social recommendations
        social_rec = self._generate_social_recommendations(pattern_results, anomaly_results)
        recommendations.extend(social_rec)
        
        # Temporal recommendations
        temporal_rec = self._generate_temporal_recommendations(pattern_results)
        recommendations.extend(temporal_rec)
        
        return recommendations
    
    def _generate_sleep_recommendations(self, pattern_results: Dict[str, Any], anomaly_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate sleep-specific recommendations"""
        recommendations = []
        
        # Check sleep patterns
        if pattern_results and 'behavioral_patterns' in pattern_results:
            sleep_patterns = pattern_results['behavioral_patterns'].get('sleep_patterns', {})
            duration_stats = sleep_patterns.get('duration_stats', {})
            
            if duration_stats.get('optimal_percentage', 0) < 70:
                recommendations.append({
                    'category': 'sleep',
                    'priority': 'high',
                    'title': 'Improve Sleep Duration',
                    'description': 'Focus on getting 7-9 hours of sleep consistently',
                    'specific_actions': [
                        'Set a consistent bedtime and wake time',
                        'Create a relaxing bedtime routine',
                        'Avoid screens 1 hour before bed',
                        'Keep bedroom cool, dark, and quiet'
                    ],
                    'expected_benefit': 'Better mood stability and emotional regulation'
                })
        
        # Check for sleep anomalies
        if anomaly_results and 'behavioral_anomalies' in anomaly_results:
            sleep_disruption = anomaly_results['behavioral_anomalies'].get('sleep_disruption', {})
            
            if sleep_disruption.get('severity') in ['high', 'medium']:
                recommendations.append({
                    'category': 'sleep',
                    'priority': 'high',
                    'title': 'Address Sleep Disruptions',
                    'description': 'Your sleep patterns show significant disruptions',
                    'specific_actions': [
                        'Track what causes poor sleep nights',
                        'Consider sleep hygiene assessment',
                        'Limit caffeine after 2 PM',
                        'Consider professional sleep evaluation'
                    ],
                    'expected_benefit': 'More consistent mood and better stress resilience'
                })
        
        return recommendations
    
    def _generate_exercise_recommendations(self, pattern_results: Dict[str, Any], trend_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate exercise-specific recommendations"""
        recommendations = []
        
        if pattern_results and 'behavioral_patterns' in pattern_results:
            exercise_patterns = pattern_results['behavioral_patterns'].get('exercise_patterns', {})
            frequency = exercise_patterns.get('frequency', {})
            
            if frequency.get('percentage_active_days', 0) < 50:
                recommendations.append({
                    'category': 'exercise',
                    'priority': 'medium',
                    'title': 'Increase Physical Activity',
                    'description': 'Regular exercise can significantly improve mood',
                    'specific_actions': [
                        'Start with 10-15 minutes of daily walking',
                        'Try activities you enjoy (dancing, swimming, hiking)',
                        'Set realistic weekly exercise goals',
                        'Use movement as a mood booster during low periods'
                    ],
                    'expected_benefit': 'Improved mood, better sleep, reduced stress'
                })
        
        return recommendations
    
    def _generate_stress_recommendations(self, mental_health_results: Dict[str, Any], pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate stress management recommendations"""
        recommendations = []
        
        if mental_health_results and 'stress_indicators' in mental_health_results:
            stress_indicators = mental_health_results['stress_indicators']
            risk_score = stress_indicators.get('stress_risk_score', {})
            
            if risk_score.get('risk_level') in ['high', 'moderate']:
                recommendations.append({
                    'category': 'stress_management',
                    'priority': 'high',
                    'title': 'Implement Stress Management Strategies',
                    'description': 'Your stress levels are impacting your mental health',
                    'specific_actions': [
                        'Practice daily mindfulness or meditation (even 5 minutes helps)',
                        'Learn deep breathing techniques for acute stress',
                        'Identify and address major stressors in your life',
                        'Consider stress management counseling'
                    ],
                    'expected_benefit': 'Better emotional regulation and improved overall wellbeing'
                })
        
        return recommendations
    
    def _generate_social_recommendations(self, pattern_results: Dict[str, Any], anomaly_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate social interaction recommendations"""
        recommendations = []
        
        # Check for social isolation
        if anomaly_results and 'behavioral_anomalies' in anomaly_results:
            social_isolation = anomaly_results['behavioral_anomalies'].get('social_isolation', {})
            
            if social_isolation.get('severity') in ['high', 'medium']:
                recommendations.append({
                    'category': 'social',
                    'priority': 'medium',
                    'title': 'Increase Social Connections',
                    'description': 'Social support is crucial for mental health',
                    'specific_actions': [
                        'Schedule regular check-ins with friends or family',
                        'Join groups or activities aligned with your interests',
                        'Consider volunteering in your community',
                        'Reach out when you\'re feeling isolated'
                    ],
                    'expected_benefit': 'Better mood support and reduced feelings of loneliness'
                })
        
        return recommendations
    
    def _generate_temporal_recommendations(self, pattern_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate time-based recommendations"""
        recommendations = []
        
        if pattern_results and 'temporal_patterns' in pattern_results:
            temporal = pattern_results['temporal_patterns']
            
            # Day of week recommendations
            if 'day_of_week_patterns' in temporal:
                dow = temporal['day_of_week_patterns']
                worst_day = dow.get('worst_day')
                
                if worst_day:
                    recommendations.append({
                        'category': 'temporal',
                        'priority': 'low',
                        'title': f'Plan Extra Support for {worst_day}s',
                        'description': f'Your mood tends to be lowest on {worst_day}s',
                        'specific_actions': [
                            f'Schedule lighter workload on {worst_day}s when possible',
                            f'Plan enjoyable activities for {worst_day} evenings',
                            f'Practice extra self-care on {worst_day}s',
                            'Be aware of this pattern to normalize lower mood days'
                        ],
                        'expected_benefit': 'Better preparation for challenging days'
                    })
        
        return recommendations
    
    def _identify_priority_actions(self, 
                                 trend_results: Dict[str, Any],
                                 pattern_results: Dict[str, Any],
                                 anomaly_results: Dict[str, Any],
                                 mental_health_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify highest priority actions based on all analyses"""
        priority_actions = []
        
        # Crisis indicators (highest priority)
        if anomaly_results and 'crisis_indicators' in anomaly_results:
            crisis = anomaly_results['crisis_indicators']
            
            for indicator, details in crisis.items():
                if details.get('severity') == 'critical':
                    priority_actions.append({
                        'priority': 'critical',
                        'action': 'Seek immediate professional help',
                        'reason': f'Critical indicator detected: {indicator}',
                        'urgency': 'immediate'
                    })
        
        # High mental health risks
        if mental_health_results and 'overall_risk_assessment' in mental_health_results:
            risk = mental_health_results['overall_risk_assessment']
            if risk.get('overall_risk_level') == 'high':
                priority_actions.append({
                    'priority': 'high',
                    'action': 'Schedule mental health professional consultation',
                    'reason': 'Multiple mental health risk indicators present',
                    'urgency': 'within_week'
                })
        
        # Declining trends
        if trend_results and 'mood_score' in trend_results:
            trend = trend_results['mood_score'].get('overall_trend', {})
            if trend.get('direction') == 'decreasing' and trend.get('r_squared', 0) > 0.3:
                priority_actions.append({
                    'priority': 'high',
                    'action': 'Address declining mood trend',
                    'reason': 'Significant downward mood trend detected',
                    'urgency': 'within_week'
                })
        
        # Major behavioral disruptions
        if anomaly_results and 'behavioral_anomalies' in anomaly_results:
            behavioral = anomaly_results['behavioral_anomalies']
            
            high_severity_behaviors = [key for key, value in behavioral.items() 
                                     if isinstance(value, dict) and value.get('severity') == 'high']
            
            if len(high_severity_behaviors) >= 2:
                priority_actions.append({
                    'priority': 'medium',
                    'action': 'Focus on behavioral stability',
                    'reason': f'Multiple behavioral disruptions: {", ".join(high_severity_behaviors)}',
                    'urgency': 'within_month'
                })
        
        return sorted(priority_actions, key=lambda x: {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}[x['priority']])
    
    def _generate_progress_insights(self, trend_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate insights about progress and improvement"""
        insights = []
        
        if not trend_results:
            return insights
        
        # Check trends for different metrics
        metrics = ['mood_score', 'stress_level', 'sleep_hours']
        
        for metric in metrics:
            if metric in trend_results:
                trend = trend_results[metric].get('overall_trend', {})
                
                if trend.get('direction') == 'increasing':
                    if metric == 'mood_score':
                        insights.append({
                            'type': 'progress',
                            'title': 'Mood Improvement Trend',
                            'description': 'Your mood has been steadily improving',
                            'celebration': 'Great job! Your efforts are paying off',
                            'maintain': 'Keep doing what you\'re doing'
                        })
                    elif metric == 'sleep_hours':
                        insights.append({
                            'type': 'progress',
                            'title': 'Sleep Improvement',
                            'description': 'Your sleep duration has been increasing',
                            'celebration': 'Excellent progress on sleep habits!',
                            'maintain': 'Continue prioritizing sleep hygiene'
                        })
                elif trend.get('direction') == 'decreasing':
                    if metric == 'stress_level':
                        insights.append({
                            'type': 'progress',
                            'title': 'Stress Reduction Success',
                            'description': 'Your stress levels have been decreasing',
                            'celebration': 'You\'re successfully managing stress!',
                            'maintain': 'Continue your stress management strategies'
                        })
        
        return insights
    
    def _identify_success_patterns(self, pattern_results: Dict[str, Any], trend_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify patterns associated with your best days"""
        success_patterns = []
        
        # This would need actual data analysis to identify what factors correlate with highest mood days
        # For now, provide general success pattern identification framework
        
        success_patterns.append({
            'type': 'success_factor',
            'title': 'Identify Your Success Formula',
            'description': 'Track what factors are present on your best mood days',
            'factors_to_track': [
                'Sleep duration on good mood days',
                'Exercise activity before good mood days',
                'Social interactions on good mood days',
                'Stress levels on good mood days',
                'Weather conditions on good mood days'
            ],
            'actionable': 'Replicate the conditions that lead to your best days'
        })
        
        return success_patterns
    
    def get_key_insights_summary(self) -> Dict[str, Any]:
        """Get a summary of the most important insights"""
        if not self.insights_results:
            return {'error': 'No insights generated. Run generate_insights() first.'}
        
        # Extract top insights by priority and type
        key_insights = {
            'most_critical': [],
            'biggest_opportunities': [],
            'success_factors': [],
            'immediate_actions': []
        }
        
        # Get priority actions
        priority_actions = self.insights_results.get('priority_actions', [])
        key_insights['immediate_actions'] = priority_actions[:3]  # Top 3 priority actions
        
        # Get improvement opportunities
        all_insights = []
        for category in ['mood_insights', 'behavioral_insights', 'health_insights']:
            if category in self.insights_results:
                all_insights.extend(self.insights_results[category])
        
        # Filter by type
        key_insights['most_critical'] = [i for i in all_insights if i.get('type') in ['concerning_trend', 'health_concern']][:2]
        key_insights['biggest_opportunities'] = [i for i in all_insights if i.get('type') == 'improvement_opportunity'][:3]
        key_insights['success_factors'] = [i for i in all_insights if i.get('type') in ['positive_trend', 'positive_behavior']][:2]
        
        return key_insights
    
    def get_insights_report(self) -> Dict[str, Any]:
        """Get comprehensive insights report"""
        if not self.insights_results:
            return {'error': 'No insights available. Run generate_insights() first.'}
        
        return {
            'summary': self.get_key_insights_summary(),
            'personalized_recommendations': self.insights_results.get('personalized_recommendations', []),
            'priority_actions': self.insights_results.get('priority_actions', []),
            'progress_insights': self.insights_results.get('progress_insights', []),
            'all_insights': {
                'mood': self.insights_results.get('mood_insights', []),
                'behavioral': self.insights_results.get('behavioral_insights', []),
                'temporal': self.insights_results.get('temporal_insights', []),
                'health': self.insights_results.get('health_insights', [])
            }
        } 