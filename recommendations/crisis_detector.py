"""
Crisis Detector - Identify mental health crises and provide immediate intervention recommendations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CrisisDetector:
    """
    Detect mental health crises and provide immediate intervention recommendations
    """
    
    def __init__(self):
        self.crisis_results = {}
        self.risk_levels = {
            'low': {'score': 0.0, 'color': 'green', 'action': 'continue_monitoring'},
            'moderate': {'score': 0.3, 'color': 'yellow', 'action': 'increased_support'},
            'high': {'score': 0.6, 'color': 'orange', 'action': 'professional_help'},
            'critical': {'score': 0.8, 'color': 'red', 'action': 'immediate_intervention'}
        }
        
    def detect_crisis(self, 
                     data: List[Dict[str, Any]],
                     anomaly_results: Dict[str, Any] = None,
                     mental_health_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Comprehensive crisis detection
        
        Args:
            data: List of mood tracking entries
            anomaly_results: Anomaly detection results
            mental_health_results: Mental health indicators results
            
        Returns:
            Dictionary with crisis detection results
        """
        logger.info("Starting crisis detection")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        crisis_analysis = {
            'immediate_risk_assessment': self._assess_immediate_risk(df, anomaly_results, mental_health_results),
            'suicidal_ideation_indicators': self._detect_suicidal_ideation(df),
            'severe_depression_indicators': self._detect_severe_depression(df),
            'acute_anxiety_crisis': self._detect_acute_anxiety(df),
            'psychotic_symptoms': self._detect_psychotic_symptoms(df),
            'substance_abuse_crisis': self._detect_substance_abuse_crisis(df),
            'self_harm_indicators': self._detect_self_harm_indicators(df),
            'functional_impairment': self._assess_functional_impairment(df),
            'social_safety_net': self._assess_social_safety_net(df),
            'crisis_timeline': self._create_crisis_timeline(df),
            'intervention_recommendations': self._generate_intervention_recommendations(df),
            'safety_plan_urgency': self._assess_safety_plan_urgency(df),
            'professional_contact_priority': self._prioritize_professional_contacts(df, mental_health_results)
        }
        
        # Calculate overall crisis score
        crisis_score = self._calculate_crisis_score(crisis_analysis)
        crisis_analysis['overall_crisis_score'] = crisis_score
        
        # Determine crisis level
        crisis_level = self._determine_crisis_level(crisis_score)
        crisis_analysis['crisis_level'] = crisis_level
        
        # Generate immediate action plan
        action_plan = self._generate_immediate_action_plan(crisis_analysis)
        crisis_analysis['immediate_action_plan'] = action_plan
        
        self.crisis_results = crisis_analysis
        logger.info("Crisis detection completed")
        
        return crisis_analysis
    
    def _assess_immediate_risk(self, df: pd.DataFrame, anomaly_results: Dict[str, Any], mental_health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess immediate risk factors"""
        risk_factors = {
            'severe_mood_episodes': [],
            'extreme_stress_periods': [],
            'sleep_deprivation_crisis': [],
            'complete_isolation': [],
            'rapid_deterioration': [],
            'total_risk_score': 0.0
        }
        
        # Severe mood episodes (score ≤ 2)
        severe_mood = df[df['mood_score'] <= 2]
        if len(severe_mood) > 0:
            risk_factors['severe_mood_episodes'] = {
                'count': len(severe_mood),
                'dates': severe_mood.index.tolist(),
                'scores': severe_mood['mood_score'].tolist(),
                'risk_contribution': 0.3
            }
            risk_factors['total_risk_score'] += 0.3
        
        # Extreme stress (score ≥ 9)
        extreme_stress = df[df['stress_level'] >= 9]
        if len(extreme_stress) > 0:
            risk_factors['extreme_stress_periods'] = {
                'count': len(extreme_stress),
                'dates': extreme_stress.index.tolist(),
                'scores': extreme_stress['stress_level'].tolist(),
                'risk_contribution': 0.2
            }
            risk_factors['total_risk_score'] += 0.2
        
        # Severe sleep deprivation (< 3 hours)
        sleep_deprivation = df[df['sleep_hours'] < 3]
        if len(sleep_deprivation) > 0:
            risk_factors['sleep_deprivation_crisis'] = {
                'count': len(sleep_deprivation),
                'dates': sleep_deprivation.index.tolist(),
                'hours': sleep_deprivation['sleep_hours'].tolist(),
                'risk_contribution': 0.15
            }
            risk_factors['total_risk_score'] += 0.15
        
        # Complete social isolation (0 interactions for 7+ days)
        isolation_streak = 0
        isolation_periods = []
        
        for date, interactions in df['social_interactions'].items():
            if interactions == 0:
                isolation_streak += 1
            else:
                if isolation_streak >= 7:
                    isolation_periods.append({
                        'end_date': date,
                        'duration': isolation_streak
                    })
                isolation_streak = 0
        
        if isolation_periods:
            risk_factors['complete_isolation'] = {
                'periods': isolation_periods,
                'longest_period': max(p['duration'] for p in isolation_periods),
                'risk_contribution': 0.1
            }
            risk_factors['total_risk_score'] += 0.1
        
        # Rapid deterioration (mood dropping 3+ points in 3 days)
        mood_changes = df['mood_score'].diff()
        rapid_declines = []
        
        for i in range(2, len(mood_changes)):
            three_day_change = mood_changes.iloc[i-2:i+1].sum()
            if three_day_change <= -3:
                rapid_declines.append({
                    'date': df.index[i],
                    'change': three_day_change
                })
        
        if rapid_declines:
            risk_factors['rapid_deterioration'] = {
                'incidents': rapid_declines,
                'count': len(rapid_declines),
                'risk_contribution': 0.2
            }
            risk_factors['total_risk_score'] += 0.2
        
        return risk_factors
    
    def _detect_suicidal_ideation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect indicators of suicidal ideation"""
        indicators = {
            'direct_indicators': [],
            'indirect_indicators': [],
            'risk_factors': [],
            'protective_factors': [],
            'urgency_level': 'low'
        }
        
        # Direct indicators from emotions (if available)
        if 'emotions' in df.columns:
            concerning_emotions = ['suicidal', 'hopeless', 'worthless', 'trapped', 'burden']
            
            for date, emotions in df['emotions'].items():
                if isinstance(emotions, list):
                    found_concerning = [e for e in emotions if e in concerning_emotions]
                    if found_concerning:
                        indicators['direct_indicators'].append({
                            'date': date,
                            'emotions': found_concerning,
                            'severity': 'critical' if 'suicidal' in found_concerning else 'high'
                        })
        
        # Indirect indicators
        # Persistent low mood + high stress + isolation
        concerning_combinations = df[
            (df['mood_score'] <= 3) & 
            (df['stress_level'] >= 8) & 
            (df['social_interactions'] == 0)
        ]
        
        if len(concerning_combinations) > 0:
            indicators['indirect_indicators'] = {
                'concerning_combinations': len(concerning_combinations),
                'dates': concerning_combinations.index.tolist(),
                'description': 'Combination of low mood, high stress, and isolation'
            }
        
        # Risk factors
        risk_factors = []
        
        # Prolonged low mood
        low_mood_streak = (df['mood_score'] <= 4).sum()
        if low_mood_streak >= 14:
            risk_factors.append(f'Prolonged low mood ({low_mood_streak} days)')
        
        # Sleep disruption
        sleep_issues = ((df['sleep_hours'] < 4) | (df['sleep_hours'] > 10)).sum()
        if sleep_issues >= 7:
            risk_factors.append(f'Significant sleep disruption ({sleep_issues} days)')
        
        # Social isolation
        isolation_days = (df['social_interactions'] == 0).sum()
        if isolation_days >= 10:
            risk_factors.append(f'Extended social isolation ({isolation_days} days)')
        
        indicators['risk_factors'] = risk_factors
        
        # Protective factors
        protective_factors = []
        
        # Recent positive mood days
        recent_good_days = (df.tail(7)['mood_score'] >= 7).sum()
        if recent_good_days >= 3:
            protective_factors.append(f'Recent positive mood days ({recent_good_days} in last week)')
        
        # Social connections
        recent_social = df.tail(7)['social_interactions'].sum()
        if recent_social >= 5:
            protective_factors.append(f'Recent social connections ({recent_social} interactions in last week)')
        
        indicators['protective_factors'] = protective_factors
        
        # Determine urgency level
        if len(indicators['direct_indicators']) > 0:
            indicators['urgency_level'] = 'critical'
        elif len(indicators['indirect_indicators']) > 0 and len(risk_factors) >= 2:
            indicators['urgency_level'] = 'high'
        elif len(risk_factors) >= 2:
            indicators['urgency_level'] = 'moderate'
        
        return indicators
    
    def _detect_severe_depression(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect severe depression indicators"""
        indicators = {
            'persistent_low_mood': False,
            'anhedonia_signs': [],
            'cognitive_symptoms': [],
            'physical_symptoms': [],
            'severity_score': 0.0,
            'duration_weeks': 0
        }
        
        # Persistent low mood (mood ≤ 4 for 14+ days)
        low_mood_days = (df['mood_score'] <= 4).sum()
        if low_mood_days >= 14:
            indicators['persistent_low_mood'] = True
            indicators['duration_weeks'] = low_mood_days // 7
            indicators['severity_score'] += 0.3
        
        # Anhedonia signs (lack of positive emotions)
        if 'emotions' in df.columns:
            positive_emotions = ['happy', 'excited', 'joyful', 'content', 'hopeful']
            days_with_positive = 0
            
            for emotions in df['emotions']:
                if isinstance(emotions, list):
                    if any(e in positive_emotions for e in emotions):
                        days_with_positive += 1
            
            if days_with_positive < len(df) * 0.2:  # Less than 20% of days
                indicators['anhedonia_signs'] = {
                    'days_with_positive_emotions': days_with_positive,
                    'percentage': days_with_positive / len(df) * 100,
                    'severity': 'severe' if days_with_positive == 0 else 'moderate'
                }
                indicators['severity_score'] += 0.2
        
        # Physical symptoms
        physical_symptoms = []
        
        # Sleep disturbances
        sleep_issues = ((df['sleep_hours'] < 5) | (df['sleep_hours'] > 9)).sum()
        if sleep_issues >= len(df) * 0.5:
            physical_symptoms.append('Sleep disturbances')
        
        # Fatigue (low exercise as proxy)
        low_energy_days = (df['exercise_minutes'] < 10).sum()
        if low_energy_days >= len(df) * 0.7:
            physical_symptoms.append('Low energy/fatigue')
        
        indicators['physical_symptoms'] = physical_symptoms
        if len(physical_symptoms) >= 1:
            indicators['severity_score'] += 0.1
        
        # Cognitive symptoms (inferred from mood patterns)
        cognitive_symptoms = []
        
        # Concentration issues (high mood variability as proxy)
        mood_variability = df['mood_score'].std()
        if mood_variability > 2:
            cognitive_symptoms.append('Concentration difficulties (inferred from mood instability)')
        
        # Decision-making difficulties (extended periods of low mood)
        if low_mood_days >= 21:
            cognitive_symptoms.append('Decision-making difficulties (inferred from prolonged low mood)')
        
        indicators['cognitive_symptoms'] = cognitive_symptoms
        if len(cognitive_symptoms) >= 1:
            indicators['severity_score'] += 0.1
        
        return indicators
    
    def _detect_acute_anxiety(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect acute anxiety crisis indicators"""
        indicators = {
            'panic_episodes': [],
            'extreme_stress_periods': [],
            'anxiety_symptoms': [],
            'crisis_level': 'low'
        }
        
        # Extreme stress as proxy for panic
        extreme_stress = df[df['stress_level'] >= 9]
        if len(extreme_stress) > 0:
            indicators['panic_episodes'] = {
                'count': len(extreme_stress),
                'dates': extreme_stress.index.tolist(),
                'stress_levels': extreme_stress['stress_level'].tolist()
            }
        
        # Sustained high stress
        high_stress_periods = df[df['stress_level'] >= 7]
        if len(high_stress_periods) >= 7:  # 7+ days of high stress
            indicators['extreme_stress_periods'] = {
                'duration': len(high_stress_periods),
                'dates': high_stress_periods.index.tolist(),
                'average_stress': high_stress_periods['stress_level'].mean()
            }
        
        # Anxiety symptoms (from emotions if available)
        if 'emotions' in df.columns:
            anxiety_emotions = ['anxious', 'panicked', 'overwhelmed', 'terrified', 'worried']
            anxiety_days = []
            
            for date, emotions in df['emotions'].items():
                if isinstance(emotions, list):
                    found_anxiety = [e for e in emotions if e in anxiety_emotions]
                    if found_anxiety:
                        anxiety_days.append({
                            'date': date,
                            'emotions': found_anxiety
                        })
            
            indicators['anxiety_symptoms'] = anxiety_days
        
        # Sleep disruption from anxiety
        sleep_disruption = df[df['sleep_hours'] < 4]
        if len(sleep_disruption) >= 3:
            indicators['sleep_disruption'] = {
                'nights': len(sleep_disruption),
                'dates': sleep_disruption.index.tolist(),
                'hours': sleep_disruption['sleep_hours'].tolist()
            }
        
        # Determine crisis level
        crisis_factors = 0
        if len(extreme_stress) > 0:
            crisis_factors += 1
        if len(high_stress_periods) >= 7:
            crisis_factors += 1
        if 'anxiety_symptoms' in indicators and len(indicators['anxiety_symptoms']) >= 3:
            crisis_factors += 1
        if 'sleep_disruption' in indicators:
            crisis_factors += 1
        
        if crisis_factors >= 3:
            indicators['crisis_level'] = 'high'
        elif crisis_factors >= 2:
            indicators['crisis_level'] = 'moderate'
        
        return indicators
    
    def _detect_psychotic_symptoms(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential psychotic symptoms"""
        indicators = {
            'reality_distortion_signs': [],
            'extreme_mood_episodes': [],
            'severe_sleep_disruption': [],
            'risk_level': 'low'
        }
        
        # Extreme mood episodes (could indicate mania/psychosis)
        extreme_high_mood = df[df['mood_score'] >= 9]
        extreme_low_mood = df[df['mood_score'] <= 2]
        
        if len(extreme_high_mood) > 0:
            indicators['extreme_mood_episodes'].append({
                'type': 'extreme_high',
                'count': len(extreme_high_mood),
                'dates': extreme_high_mood.index.tolist(),
                'scores': extreme_high_mood['mood_score'].tolist()
            })
        
        if len(extreme_low_mood) > 0:
            indicators['extreme_mood_episodes'].append({
                'type': 'extreme_low',
                'count': len(extreme_low_mood),
                'dates': extreme_low_mood.index.tolist(),
                'scores': extreme_low_mood['mood_score'].tolist()
            })
        
        # Severe sleep disruption (< 2 hours or > 12 hours)
        severe_sleep_issues = df[(df['sleep_hours'] < 2) | (df['sleep_hours'] > 12)]
        if len(severe_sleep_issues) > 0:
            indicators['severe_sleep_disruption'] = {
                'count': len(severe_sleep_issues),
                'dates': severe_sleep_issues.index.tolist(),
                'hours': severe_sleep_issues['sleep_hours'].tolist()
            }
        
        # Reality distortion signs (from emotions if available)
        if 'emotions' in df.columns:
            psychotic_emotions = ['confused', 'disconnected', 'unreal', 'paranoid']
            
            for date, emotions in df['emotions'].items():
                if isinstance(emotions, list):
                    found_psychotic = [e for e in emotions if e in psychotic_emotions]
                    if found_psychotic:
                        indicators['reality_distortion_signs'].append({
                            'date': date,
                            'emotions': found_psychotic
                        })
        
        # Determine risk level
        risk_factors = len(indicators['extreme_mood_episodes'])
        if 'severe_sleep_disruption' in indicators:
            risk_factors += 1
        if len(indicators['reality_distortion_signs']) > 0:
            risk_factors += 2  # Higher weight for reality distortion
        
        if risk_factors >= 3:
            indicators['risk_level'] = 'high'
        elif risk_factors >= 2:
            indicators['risk_level'] = 'moderate'
        
        return indicators
    
    def _detect_substance_abuse_crisis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect substance abuse crisis indicators"""
        indicators = {
            'substance_use_patterns': [],
            'crisis_indicators': [],
            'risk_level': 'low'
        }
        
        # This would need specific substance use data
        # For now, infer from patterns that might indicate substance abuse
        
        # Erratic mood patterns
        mood_volatility = df['mood_score'].std()
        if mood_volatility > 3:
            indicators['substance_use_patterns'].append({
                'pattern': 'extreme_mood_volatility',
                'description': 'Highly erratic mood patterns may indicate substance use',
                'volatility_score': mood_volatility
            })
        
        # Extreme sleep patterns
        sleep_volatility = df['sleep_hours'].std()
        if sleep_volatility > 3:
            indicators['substance_use_patterns'].append({
                'pattern': 'erratic_sleep_patterns',
                'description': 'Highly inconsistent sleep patterns',
                'volatility_score': sleep_volatility
            })
        
        # Note: This is a simplified inference model
        # Real substance abuse detection would require specific data
        
        return indicators
    
    def _detect_self_harm_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect self-harm indicators"""
        indicators = {
            'emotional_indicators': [],
            'behavioral_indicators': [],
            'risk_level': 'low'
        }
        
        # Emotional indicators
        if 'emotions' in df.columns:
            self_harm_emotions = ['angry', 'frustrated', 'numb', 'empty', 'worthless']
            
            for date, emotions in df['emotions'].items():
                if isinstance(emotions, list):
                    found_indicators = [e for e in emotions if e in self_harm_emotions]
                    if len(found_indicators) >= 2:  # Multiple concerning emotions
                        indicators['emotional_indicators'].append({
                            'date': date,
                            'emotions': found_indicators
                        })
        
        # Behavioral indicators
        behavioral_indicators = []
        
        # Sudden isolation after social periods
        social_drops = []
        for i in range(1, len(df)):
            if df['social_interactions'].iloc[i-1] >= 3 and df['social_interactions'].iloc[i] == 0:
                social_drops.append(df.index[i])
        
        if len(social_drops) >= 3:
            behavioral_indicators.append({
                'pattern': 'sudden_social_withdrawal',
                'dates': social_drops,
                'description': 'Sudden withdrawal from social activities'
            })
        
        # Extreme mood swings
        mood_swings = []
        for i in range(1, len(df)):
            mood_change = abs(df['mood_score'].iloc[i] - df['mood_score'].iloc[i-1])
            if mood_change >= 4:
                mood_swings.append({
                    'date': df.index[i],
                    'change': mood_change
                })
        
        if len(mood_swings) >= 5:
            behavioral_indicators.append({
                'pattern': 'extreme_mood_swings',
                'incidents': mood_swings,
                'description': 'Frequent extreme mood changes'
            })
        
        indicators['behavioral_indicators'] = behavioral_indicators
        
        # Determine risk level
        risk_factors = len(indicators['emotional_indicators']) + len(behavioral_indicators)
        if risk_factors >= 3:
            indicators['risk_level'] = 'high'
        elif risk_factors >= 2:
            indicators['risk_level'] = 'moderate'
        
        return indicators
    
    def _assess_functional_impairment(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess functional impairment levels"""
        impairment = {
            'sleep_function': 'normal',
            'social_function': 'normal',
            'self_care_function': 'normal',
            'overall_impairment': 'none'
        }
        
        # Sleep function impairment
        sleep_issues = ((df['sleep_hours'] < 4) | (df['sleep_hours'] > 10)).sum()
        sleep_percentage = sleep_issues / len(df) * 100
        
        if sleep_percentage >= 50:
            impairment['sleep_function'] = 'severe'
        elif sleep_percentage >= 25:
            impairment['sleep_function'] = 'moderate'
        elif sleep_percentage >= 10:
            impairment['sleep_function'] = 'mild'
        
        # Social function impairment
        isolation_days = (df['social_interactions'] == 0).sum()
        isolation_percentage = isolation_days / len(df) * 100
        
        if isolation_percentage >= 70:
            impairment['social_function'] = 'severe'
        elif isolation_percentage >= 40:
            impairment['social_function'] = 'moderate'
        elif isolation_percentage >= 20:
            impairment['social_function'] = 'mild'
        
        # Self-care function (inferred from exercise as proxy)
        no_exercise_days = (df['exercise_minutes'] == 0).sum()
        no_exercise_percentage = no_exercise_days / len(df) * 100
        
        if no_exercise_percentage >= 80:
            impairment['self_care_function'] = 'severe'
        elif no_exercise_percentage >= 60:
            impairment['self_care_function'] = 'moderate'
        elif no_exercise_percentage >= 40:
            impairment['self_care_function'] = 'mild'
        
        # Overall impairment
        impairment_scores = {
            'none': 0, 'normal': 0, 'mild': 1, 'moderate': 2, 'severe': 3
        }
        
        total_impairment = sum(impairment_scores.get(level, 0) for level in 
                             [impairment['sleep_function'], impairment['social_function'], 
                              impairment['self_care_function']])
        
        if total_impairment >= 7:
            impairment['overall_impairment'] = 'severe'
        elif total_impairment >= 5:
            impairment['overall_impairment'] = 'moderate'
        elif total_impairment >= 3:
            impairment['overall_impairment'] = 'mild'
        
        return impairment
    
    def _assess_social_safety_net(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess social safety net strength"""
        safety_net = {
            'average_daily_interactions': df['social_interactions'].mean(),
            'isolation_risk': 'low',
            'support_consistency': 'stable',
            'safety_net_strength': 'strong'
        }
        
        # Isolation risk
        isolation_days = (df['social_interactions'] == 0).sum()
        isolation_percentage = isolation_days / len(df) * 100
        
        if isolation_percentage >= 50:
            safety_net['isolation_risk'] = 'high'
        elif isolation_percentage >= 25:
            safety_net['isolation_risk'] = 'moderate'
        
        # Support consistency
        social_std = df['social_interactions'].std()
        if social_std > 3:
            safety_net['support_consistency'] = 'unstable'
        elif social_std > 1.5:
            safety_net['support_consistency'] = 'variable'
        
        # Overall safety net strength
        if (safety_net['isolation_risk'] == 'high' or 
            safety_net['average_daily_interactions'] < 1):
            safety_net['safety_net_strength'] = 'weak'
        elif (safety_net['isolation_risk'] == 'moderate' or 
              safety_net['average_daily_interactions'] < 2):
            safety_net['safety_net_strength'] = 'moderate'
        
        return safety_net
    
    def _create_crisis_timeline(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Create timeline of crisis development"""
        timeline = {
            'crisis_events': [],
            'escalation_pattern': 'stable',
            'recent_deterioration': False
        }
        
        # Identify crisis events
        crisis_events = []
        
        # Severe mood episodes
        severe_mood_days = df[df['mood_score'] <= 2]
        for date, row in severe_mood_days.iterrows():
            crisis_events.append({
                'date': date,
                'type': 'severe_mood',
                'severity': 'high',
                'details': f"Mood score: {row['mood_score']}"
            })
        
        # Extreme stress episodes
        extreme_stress_days = df[df['stress_level'] >= 9]
        for date, row in extreme_stress_days.iterrows():
            crisis_events.append({
                'date': date,
                'type': 'extreme_stress',
                'severity': 'high',
                'details': f"Stress level: {row['stress_level']}"
            })
        
        # Sleep crisis
        sleep_crisis_days = df[df['sleep_hours'] < 3]
        for date, row in sleep_crisis_days.iterrows():
            crisis_events.append({
                'date': date,
                'type': 'sleep_crisis',
                'severity': 'medium',
                'details': f"Sleep hours: {row['sleep_hours']}"
            })
        
        # Sort events by date
        crisis_events.sort(key=lambda x: x['date'])
        timeline['crisis_events'] = crisis_events
        
        # Analyze escalation pattern
        if len(crisis_events) >= 3:
            recent_events = [e for e in crisis_events if e['date'] >= df.index[-14]]  # Last 2 weeks
            if len(recent_events) >= 2:
                timeline['escalation_pattern'] = 'escalating'
        
        # Recent deterioration
        if len(df) >= 7:
            recent_avg = df.tail(7)['mood_score'].mean()
            earlier_avg = df.head(7)['mood_score'].mean()
            if recent_avg < earlier_avg - 2:
                timeline['recent_deterioration'] = True
        
        return timeline
    
    def _generate_intervention_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate immediate intervention recommendations"""
        recommendations = {
            'immediate_actions': [],
            'within_24_hours': [],
            'within_week': [],
            'ongoing_support': []
        }
        
        # Immediate actions (if severe crisis detected)
        if (df['mood_score'] <= 2).any():
            recommendations['immediate_actions'].extend([
                'Call crisis hotline (988) if experiencing suicidal thoughts',
                'Contact trusted friend or family member immediately',
                'Remove means of self-harm from environment',
                'Go to emergency room if in immediate danger'
            ])
        
        if (df['stress_level'] >= 9).any():
            recommendations['immediate_actions'].extend([
                'Practice deep breathing exercises',
                'Use grounding techniques (5-4-3-2-1 method)',
                'Call crisis support line for immediate help'
            ])
        
        # Within 24 hours
        severe_indicators = ((df['mood_score'] <= 3).sum() >= 3 or 
                           (df['stress_level'] >= 8).sum() >= 3)
        
        if severe_indicators:
            recommendations['within_24_hours'].extend([
                'Contact mental health professional',
                'Reach out to primary care physician',
                'Arrange for friend/family to stay with you',
                'Create safety plan with trusted person'
            ])
        
        # Within week
        if (df['mood_score'] <= 4).sum() >= 7:
            recommendations['within_week'].extend([
                'Schedule therapy appointment',
                'Consider medication evaluation',
                'Join support group',
                'Implement daily check-ins with support person'
            ])
        
        # Ongoing support
        recommendations['ongoing_support'].extend([
            'Continue mood tracking',
            'Maintain regular sleep schedule',
            'Engage in daily physical activity',
            'Practice stress management techniques',
            'Stay connected with support network'
        ])
        
        return recommendations
    
    def _assess_safety_plan_urgency(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess urgency of safety plan implementation"""
        urgency = {
            'level': 'low',
            'reasons': [],
            'timeline': 'within_month'
        }
        
        # High urgency factors
        high_urgency_factors = []
        
        if (df['mood_score'] <= 2).any():
            high_urgency_factors.append('Severe mood episodes detected')
        
        if (df['stress_level'] >= 9).any():
            high_urgency_factors.append('Extreme stress levels detected')
        
        if (df['sleep_hours'] < 3).any():
            high_urgency_factors.append('Severe sleep deprivation')
        
        # Medium urgency factors
        medium_urgency_factors = []
        
        if (df['mood_score'] <= 4).sum() >= 7:
            medium_urgency_factors.append('Prolonged low mood period')
        
        if (df['social_interactions'] == 0).sum() >= 7:
            medium_urgency_factors.append('Extended social isolation')
        
        if (df['stress_level'] >= 7).sum() >= 7:
            medium_urgency_factors.append('Sustained high stress')
        
        # Determine urgency level
        if len(high_urgency_factors) > 0:
            urgency['level'] = 'immediate'
            urgency['timeline'] = 'within_24_hours'
            urgency['reasons'] = high_urgency_factors
        elif len(medium_urgency_factors) >= 2:
            urgency['level'] = 'high'
            urgency['timeline'] = 'within_week'
            urgency['reasons'] = medium_urgency_factors
        elif len(medium_urgency_factors) >= 1:
            urgency['level'] = 'moderate'
            urgency['timeline'] = 'within_two_weeks'
            urgency['reasons'] = medium_urgency_factors
        
        return urgency
    
    def _prioritize_professional_contacts(self, df: pd.DataFrame, mental_health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Prioritize professional contacts based on crisis indicators"""
        priorities = {
            'immediate_contact': [],
            'urgent_contact': [],
            'scheduled_contact': []
        }
        
        # Immediate contact needed
        if (df['mood_score'] <= 2).any():
            priorities['immediate_contact'].extend([
                'Crisis hotline (988)',
                'Emergency services (911) if in immediate danger',
                'Mobile crisis team'
            ])
        
        # Urgent contact (within 24-48 hours)
        if ((df['mood_score'] <= 3).sum() >= 3 or 
            (df['stress_level'] >= 8).sum() >= 3):
            priorities['urgent_contact'].extend([
                'Primary therapist/counselor',
                'Psychiatrist (if applicable)',
                'Primary care physician',
                'Crisis intervention team'
            ])
        
        # Scheduled contact (within week)
        if ((df['mood_score'] <= 5).sum() >= 7 or 
            (df['stress_level'] >= 6).sum() >= 7):
            priorities['scheduled_contact'].extend([
                'Mental health counselor',
                'Support group facilitator',
                'Case manager',
                'Peer support specialist'
            ])
        
        return priorities
    
    def _calculate_crisis_score(self, crisis_analysis: Dict[str, Any]) -> float:
        """Calculate overall crisis score"""
        score = 0.0
        
        # Immediate risk factors
        immediate_risk = crisis_analysis.get('immediate_risk_assessment', {})
        score += immediate_risk.get('total_risk_score', 0.0)
        
        # Suicidal ideation
        suicidal_ideation = crisis_analysis.get('suicidal_ideation_indicators', {})
        if suicidal_ideation.get('urgency_level') == 'critical':
            score += 0.4
        elif suicidal_ideation.get('urgency_level') == 'high':
            score += 0.3
        elif suicidal_ideation.get('urgency_level') == 'moderate':
            score += 0.2
        
        # Severe depression
        severe_depression = crisis_analysis.get('severe_depression_indicators', {})
        score += severe_depression.get('severity_score', 0.0)
        
        # Acute anxiety
        acute_anxiety = crisis_analysis.get('acute_anxiety_crisis', {})
        if acute_anxiety.get('crisis_level') == 'high':
            score += 0.2
        elif acute_anxiety.get('crisis_level') == 'moderate':
            score += 0.1
        
        # Psychotic symptoms
        psychotic_symptoms = crisis_analysis.get('psychotic_symptoms', {})
        if psychotic_symptoms.get('risk_level') == 'high':
            score += 0.3
        elif psychotic_symptoms.get('risk_level') == 'moderate':
            score += 0.2
        
        # Functional impairment
        functional_impairment = crisis_analysis.get('functional_impairment', {})
        if functional_impairment.get('overall_impairment') == 'severe':
            score += 0.2
        elif functional_impairment.get('overall_impairment') == 'moderate':
            score += 0.1
        
        # Social safety net
        social_safety = crisis_analysis.get('social_safety_net', {})
        if social_safety.get('safety_net_strength') == 'weak':
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _determine_crisis_level(self, crisis_score: float) -> str:
        """Determine crisis level based on score"""
        if crisis_score >= 0.8:
            return 'critical'
        elif crisis_score >= 0.6:
            return 'high'
        elif crisis_score >= 0.3:
            return 'moderate'
        else:
            return 'low'
    
    def _generate_immediate_action_plan(self, crisis_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate immediate action plan based on crisis analysis"""
        action_plan = {
            'priority_level': crisis_analysis.get('crisis_level', 'low'),
            'immediate_steps': [],
            'safety_contacts': [],
            'professional_resources': [],
            'self_care_actions': [],
            'monitoring_plan': []
        }
        
        crisis_level = crisis_analysis.get('crisis_level', 'low')
        
        if crisis_level == 'critical':
            action_plan['immediate_steps'] = [
                '🚨 EMERGENCY: Call 988 (Suicide & Crisis Lifeline) immediately',
                '🚨 Call 911 if in immediate physical danger',
                '🚨 Contact trusted person to stay with you',
                '🚨 Remove means of self-harm from environment',
                '🚨 Go to emergency room if thoughts of self-harm persist'
            ]
            action_plan['safety_contacts'] = [
                'Crisis Hotline: 988',
                'Emergency Services: 911',
                'Crisis Text Line: Text HOME to 741741'
            ]
            
        elif crisis_level == 'high':
            action_plan['immediate_steps'] = [
                '⚠️ Call crisis hotline (988) for support',
                '⚠️ Contact mental health professional within 24 hours',
                '⚠️ Reach out to trusted friend or family member',
                '⚠️ Create safety plan with support person',
                '⚠️ Monitor mood and stress levels closely'
            ]
            action_plan['professional_resources'] = [
                'Contact primary therapist/counselor',
                'Schedule urgent appointment with psychiatrist',
                'Consider mobile crisis team evaluation'
            ]
            
        elif crisis_level == 'moderate':
            action_plan['immediate_steps'] = [
                '📋 Schedule mental health appointment within 1 week',
                '📋 Implement daily check-ins with support person',
                '📋 Increase use of coping strategies',
                '📋 Monitor warning signs daily',
                '📋 Maintain regular sleep and exercise routine'
            ]
            action_plan['professional_resources'] = [
                'Schedule therapy appointment',
                'Consider medication evaluation',
                'Join support group'
            ]
            
        else:  # low
            action_plan['immediate_steps'] = [
                '✅ Continue current mood tracking',
                '✅ Maintain healthy routines',
                '✅ Stay connected with support network',
                '✅ Practice regular self-care',
                '✅ Monitor for changes in mood patterns'
            ]
        
        # Universal self-care actions
        action_plan['self_care_actions'] = [
            'Practice deep breathing exercises',
            'Maintain regular sleep schedule',
            'Engage in gentle physical activity',
            'Connect with supportive people',
            'Use grounding techniques when stressed'
        ]
        
        # Monitoring plan
        action_plan['monitoring_plan'] = [
            'Track mood daily (1-10 scale)',
            'Note stress levels and triggers',
            'Monitor sleep quality and duration',
            'Record social interactions',
            'Weekly review of patterns with support person'
        ]
        
        return action_plan
    
    def get_crisis_report(self) -> Dict[str, Any]:
        """Get comprehensive crisis detection report"""
        if not self.crisis_results:
            return {'error': 'No crisis detection results available. Run detect_crisis() first.'}
        
        return {
            'crisis_level': self.crisis_results.get('crisis_level', 'low'),
            'crisis_score': self.crisis_results.get('overall_crisis_score', 0.0),
            'immediate_action_plan': self.crisis_results.get('immediate_action_plan', {}),
            'key_indicators': {
                'suicidal_ideation': self.crisis_results.get('suicidal_ideation_indicators', {}),
                'severe_depression': self.crisis_results.get('severe_depression_indicators', {}),
                'acute_anxiety': self.crisis_results.get('acute_anxiety_crisis', {}),
                'functional_impairment': self.crisis_results.get('functional_impairment', {})
            },
            'professional_contacts': self.crisis_results.get('professional_contact_priority', {}),
            'safety_plan_urgency': self.crisis_results.get('safety_plan_urgency', {})
        } 