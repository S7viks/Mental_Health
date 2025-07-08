"""
Mental Health Indicators - Identify mental health risk factors and indicators
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class MentalHealthIndicators:
    """
    Identify mental health indicators and risk factors from mood tracking data
    """
    
    def __init__(self):
        self.indicators_results = {}
        self.risk_factors = {}
        self.warning_signs = {}
        
    def analyze_mental_health_indicators(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive analysis of mental health indicators
        
        Args:
            data: List of mood tracking entries
            
        Returns:
            Dictionary with mental health indicators analysis
        """
        logger.info("Starting mental health indicators analysis")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        results = {
            'depression_indicators': self._analyze_depression_indicators(df),
            'anxiety_indicators': self._analyze_anxiety_indicators(df),
            'stress_indicators': self._analyze_stress_indicators(df),
            'sleep_disorder_indicators': self._analyze_sleep_disorder_indicators(df),
            'seasonal_affective_indicators': self._analyze_seasonal_affective_indicators(df),
            'social_isolation_indicators': self._analyze_social_isolation_indicators(df),
            'burnout_indicators': self._analyze_burnout_indicators(df),
            'bipolar_indicators': self._analyze_bipolar_indicators(df),
            'overall_risk_assessment': self._assess_overall_risk(df),
            'professional_help_recommendations': self._generate_professional_help_recommendations(df)
        }
        
        self.indicators_results = results
        logger.info("Mental health indicators analysis completed")
        
        return results
    
    def _analyze_depression_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators consistent with depression"""
        indicators = {}
        
        # Persistent low mood
        low_mood_days = (df['mood_score'] <= 4).sum()
        low_mood_percentage = low_mood_days / len(df) * 100
        
        indicators['persistent_low_mood'] = {
            'days_with_low_mood': low_mood_days,
            'percentage': low_mood_percentage,
            'severity': 'high' if low_mood_percentage > 40 else 'medium' if low_mood_percentage > 20 else 'low'
        }
        
        # Anhedonia indicators (lack of interest/pleasure)
        if 'emotions' in df.columns:
            positive_emotions = ['happy', 'excited', 'joyful', 'content', 'hopeful']
            negative_emotions = ['sad', 'depressed', 'hopeless', 'empty']
            
            positive_emotion_days = 0
            negative_emotion_days = 0
            
            for emotions in df['emotions']:
                if isinstance(emotions, list):
                    if any(emotion in positive_emotions for emotion in emotions):
                        positive_emotion_days += 1
                    if any(emotion in negative_emotions for emotion in emotions):
                        negative_emotion_days += 1
            
            indicators['anhedonia_indicators'] = {
                'positive_emotion_days': positive_emotion_days,
                'negative_emotion_days': negative_emotion_days,
                'positive_emotion_percentage': positive_emotion_days / len(df) * 100,
                'negative_emotion_percentage': negative_emotion_days / len(df) * 100
            }
        
        # Sleep disturbances
        sleep_issues = {
            'insomnia_nights': (df['sleep_hours'] < 5).sum(),
            'hypersomnia_days': (df['sleep_hours'] > 10).sum(),
            'sleep_variability': df['sleep_hours'].std()
        }
        
        indicators['sleep_disturbances'] = sleep_issues
        
        # Social withdrawal
        social_withdrawal = {
            'isolation_days': (df['social_interactions'] == 0).sum(),
            'avg_social_interactions': df['social_interactions'].mean(),
            'social_decline': self._detect_social_decline(df)
        }
        
        indicators['social_withdrawal'] = social_withdrawal
        
        # Energy/fatigue indicators
        energy_indicators = {
            'low_exercise_days': (df['exercise_minutes'] < 15).sum(),
            'exercise_decline': self._detect_exercise_decline(df),
            'avg_exercise': df['exercise_minutes'].mean()
        }
        
        indicators['energy_fatigue'] = energy_indicators
        
        # Calculate overall depression risk score
        depression_score = self._calculate_depression_risk_score(indicators)
        indicators['depression_risk_score'] = depression_score
        
        return indicators
    
    def _analyze_anxiety_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators consistent with anxiety disorders"""
        indicators = {}
        
        # High stress levels
        high_stress_days = (df['stress_level'] >= 7).sum()
        high_stress_percentage = high_stress_days / len(df) * 100
        
        indicators['high_stress_frequency'] = {
            'days_with_high_stress': high_stress_days,
            'percentage': high_stress_percentage,
            'severity': 'high' if high_stress_percentage > 50 else 'medium' if high_stress_percentage > 25 else 'low'
        }
        
        # Mood volatility (anxiety often causes mood swings)
        mood_volatility = df['mood_score'].rolling(window=7).std().mean()
        indicators['mood_volatility'] = {
            'average_volatility': mood_volatility,
            'high_volatility_days': (df['mood_score'].rolling(window=7).std() > mood_volatility * 1.5).sum(),
            'severity': 'high' if mood_volatility > 2 else 'medium' if mood_volatility > 1 else 'low'
        }
        
        # Anxiety-related emotions
        if 'emotions' in df.columns:
            anxiety_emotions = ['anxious', 'worried', 'stressed', 'overwhelmed', 'nervous']
            anxiety_emotion_days = 0
            
            for emotions in df['emotions']:
                if isinstance(emotions, list):
                    if any(emotion in anxiety_emotions for emotion in emotions):
                        anxiety_emotion_days += 1
            
            indicators['anxiety_emotions'] = {
                'days_with_anxiety_emotions': anxiety_emotion_days,
                'percentage': anxiety_emotion_days / len(df) * 100
            }
        
        # Sleep disturbances related to anxiety
        sleep_anxiety_indicators = {
            'insomnia_pattern': (df['sleep_hours'] < 6).sum(),
            'sleep_onset_issues': self._detect_sleep_onset_issues(df),
            'restless_sleep': df['sleep_hours'].std()
        }
        
        indicators['sleep_anxiety'] = sleep_anxiety_indicators
        
        # Physical symptoms proxy (exercise avoidance due to anxiety)
        physical_indicators = {
            'exercise_avoidance': (df['exercise_minutes'] == 0).sum(),
            'exercise_consistency': 1 / (1 + df['exercise_minutes'].std())
        }
        
        indicators['physical_symptoms'] = physical_indicators
        
        # Calculate overall anxiety risk score
        anxiety_score = self._calculate_anxiety_risk_score(indicators)
        indicators['anxiety_risk_score'] = anxiety_score
        
        return indicators
    
    def _analyze_stress_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze chronic stress indicators"""
        indicators = {}
        
        # Chronic stress patterns
        chronic_stress = {
            'avg_stress_level': df['stress_level'].mean(),
            'high_stress_days': (df['stress_level'] >= 7).sum(),
            'chronic_stress_periods': self._detect_chronic_stress_periods(df),
            'stress_trend': self._analyze_stress_trend(df)
        }
        
        indicators['chronic_stress'] = chronic_stress
        
        # Stress impact on mood
        stress_mood_correlation = df['stress_level'].corr(df['mood_score'])
        indicators['stress_mood_impact'] = {
            'correlation': stress_mood_correlation,
            'impact_severity': 'high' if stress_mood_correlation < -0.5 else 'medium' if stress_mood_correlation < -0.3 else 'low'
        }
        
        # Stress-related behavioral changes
        behavioral_changes = {
            'sleep_disruption': self._analyze_stress_sleep_relationship(df),
            'exercise_impact': self._analyze_stress_exercise_relationship(df),
            'social_impact': self._analyze_stress_social_relationship(df)
        }
        
        indicators['behavioral_changes'] = behavioral_changes
        
        # Calculate overall stress risk score
        stress_score = self._calculate_stress_risk_score(indicators)
        indicators['stress_risk_score'] = stress_score
        
        return indicators
    
    def _analyze_sleep_disorder_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators of sleep disorders"""
        indicators = {}
        
        # Sleep duration patterns
        sleep_duration = {
            'avg_sleep_hours': df['sleep_hours'].mean(),
            'sleep_variability': df['sleep_hours'].std(),
            'insomnia_nights': (df['sleep_hours'] < 5).sum(),
            'hypersomnia_days': (df['sleep_hours'] > 10).sum(),
            'optimal_sleep_percentage': ((df['sleep_hours'] >= 7) & (df['sleep_hours'] <= 9)).sum() / len(df) * 100
        }
        
        indicators['sleep_duration'] = sleep_duration
        
        # Sleep consistency
        sleep_consistency = {
            'consistency_score': 1 / (1 + df['sleep_hours'].std()),
            'irregular_sleep_pattern': df['sleep_hours'].std() > 1.5,
            'sleep_debt_days': (df['sleep_hours'] < 7).sum()
        }
        
        indicators['sleep_consistency'] = sleep_consistency
        
        # Sleep-mood relationship
        sleep_mood_correlation = df['sleep_hours'].corr(df['mood_score'])
        indicators['sleep_mood_relationship'] = {
            'correlation': sleep_mood_correlation,
            'relationship_strength': 'strong' if abs(sleep_mood_correlation) > 0.5 else 'moderate' if abs(sleep_mood_correlation) > 0.3 else 'weak'
        }
        
        # Calculate sleep disorder risk score
        sleep_disorder_score = self._calculate_sleep_disorder_risk_score(indicators)
        indicators['sleep_disorder_risk_score'] = sleep_disorder_score
        
        return indicators
    
    def _analyze_seasonal_affective_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators of Seasonal Affective Disorder"""
        indicators = {}
        
        # Monthly mood patterns
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.month
        monthly_mood = df_monthly.groupby('month')['mood_score'].mean()
        
        # Winter months analysis
        winter_months = [11, 12, 1, 2]
        summer_months = [5, 6, 7, 8]
        
        winter_mood = monthly_mood[monthly_mood.index.isin(winter_months)].mean()
        summer_mood = monthly_mood[monthly_mood.index.isin(summer_months)].mean()
        
        seasonal_patterns = {
            'winter_average_mood': winter_mood,
            'summer_average_mood': summer_mood,
            'seasonal_difference': summer_mood - winter_mood,
            'winter_depression_risk': (summer_mood - winter_mood) > 1.5 and winter_mood < 5
        }
        
        indicators['seasonal_patterns'] = seasonal_patterns
        
        # Seasonal behavioral changes
        if len(df_monthly) > 12:  # Need at least a year of data
            seasonal_behaviors = {
                'winter_sleep_changes': self._analyze_winter_sleep_changes(df_monthly),
                'winter_exercise_changes': self._analyze_winter_exercise_changes(df_monthly),
                'winter_social_changes': self._analyze_winter_social_changes(df_monthly)
            }
            
            indicators['seasonal_behaviors'] = seasonal_behaviors
        
        # Calculate SAD risk score
        sad_score = self._calculate_sad_risk_score(indicators)
        indicators['sad_risk_score'] = sad_score
        
        return indicators
    
    def _analyze_social_isolation_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators of social isolation"""
        indicators = {}
        
        # Social interaction patterns
        social_patterns = {
            'avg_daily_interactions': df['social_interactions'].mean(),
            'isolation_days': (df['social_interactions'] == 0).sum(),
            'isolation_percentage': (df['social_interactions'] == 0).sum() / len(df) * 100,
            'social_decline_trend': self._analyze_social_trend(df)
        }
        
        indicators['social_patterns'] = social_patterns
        
        # Extended isolation periods
        isolation_periods = self._detect_isolation_periods(df)
        indicators['isolation_periods'] = isolation_periods
        
        # Social isolation impact on mood
        social_mood_correlation = df['social_interactions'].corr(df['mood_score'])
        indicators['social_mood_impact'] = {
            'correlation': social_mood_correlation,
            'impact_severity': 'high' if social_mood_correlation > 0.5 else 'medium' if social_mood_correlation > 0.3 else 'low'
        }
        
        # Calculate social isolation risk score
        isolation_score = self._calculate_isolation_risk_score(indicators)
        indicators['isolation_risk_score'] = isolation_score
        
        return indicators
    
    def _analyze_burnout_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators of burnout"""
        indicators = {}
        
        # Energy and motivation indicators
        energy_indicators = {
            'avg_exercise_minutes': df['exercise_minutes'].mean(),
            'exercise_decline': self._detect_exercise_decline(df),
            'low_energy_days': (df['exercise_minutes'] < 15).sum(),
            'energy_variability': df['exercise_minutes'].std()
        }
        
        indicators['energy_indicators'] = energy_indicators
        
        # Chronic stress and mood relationship
        stress_mood_pattern = {
            'chronic_high_stress': (df['stress_level'] >= 7).sum() / len(df) * 100,
            'stress_mood_correlation': df['stress_level'].corr(df['mood_score']),
            'emotional_exhaustion': self._detect_emotional_exhaustion(df)
        }
        
        indicators['stress_mood_pattern'] = stress_mood_pattern
        
        # Sleep and recovery indicators
        recovery_indicators = {
            'poor_sleep_recovery': (df['sleep_hours'] < 6).sum(),
            'sleep_stress_relationship': self._analyze_sleep_stress_relationship(df),
            'weekend_recovery': self._analyze_weekend_recovery(df)
        }
        
        indicators['recovery_indicators'] = recovery_indicators
        
        # Calculate burnout risk score
        burnout_score = self._calculate_burnout_risk_score(indicators)
        indicators['burnout_risk_score'] = burnout_score
        
        return indicators
    
    def _analyze_bipolar_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze indicators that might suggest bipolar patterns"""
        indicators = {}
        
        # Mood volatility and extreme swings
        mood_volatility = {
            'overall_volatility': df['mood_score'].std(),
            'extreme_high_days': (df['mood_score'] >= 9).sum(),
            'extreme_low_days': (df['mood_score'] <= 3).sum(),
            'mood_range': df['mood_score'].max() - df['mood_score'].min(),
            'rapid_cycling': self._detect_rapid_cycling(df)
        }
        
        indicators['mood_volatility'] = mood_volatility
        
        # Sleep patterns during mood episodes
        sleep_patterns = {
            'sleep_during_high_mood': self._analyze_sleep_during_high_mood(df),
            'sleep_during_low_mood': self._analyze_sleep_during_low_mood(df),
            'sleep_variability': df['sleep_hours'].std()
        }
        
        indicators['sleep_patterns'] = sleep_patterns
        
        # Energy and activity patterns
        activity_patterns = {
            'exercise_during_high_mood': self._analyze_exercise_during_high_mood(df),
            'exercise_during_low_mood': self._analyze_exercise_during_low_mood(df),
            'activity_extremes': self._detect_activity_extremes(df)
        }
        
        indicators['activity_patterns'] = activity_patterns
        
        # Calculate bipolar risk indicators (note: this is NOT a diagnosis)
        bipolar_indicators_score = self._calculate_bipolar_indicators_score(indicators)
        indicators['bipolar_indicators_score'] = bipolar_indicators_score
        
        return indicators
    
    def _assess_overall_risk(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Assess overall mental health risk"""
        risk_assessment = {}
        
        # Collect all risk scores
        risk_scores = {}
        
        if hasattr(self, 'indicators_results'):
            for category, results in self.indicators_results.items():
                if isinstance(results, dict):
                    for key, value in results.items():
                        if key.endswith('_risk_score') and isinstance(value, dict):
                            risk_scores[key] = value.get('score', 0)
        
        # Calculate composite risk score
        if risk_scores:
            composite_score = np.mean(list(risk_scores.values()))
            max_score = max(risk_scores.values())
            
            risk_assessment['composite_risk_score'] = composite_score
            risk_assessment['highest_risk_area'] = max(risk_scores.items(), key=lambda x: x[1])
            risk_assessment['individual_scores'] = risk_scores
            
            # Determine overall risk level
            if composite_score >= 0.7 or max_score >= 0.8:
                risk_level = 'high'
            elif composite_score >= 0.5 or max_score >= 0.6:
                risk_level = 'moderate'
            else:
                risk_level = 'low'
            
            risk_assessment['overall_risk_level'] = risk_level
        
        # Red flags that require immediate attention
        red_flags = self._identify_red_flags(df)
        risk_assessment['red_flags'] = red_flags
        
        return risk_assessment
    
    def _generate_professional_help_recommendations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate recommendations for professional help"""
        recommendations = {}
        
        # Immediate help recommendations
        immediate_help = []
        
        # Crisis indicators
        if (df['mood_score'] <= 2).any():
            immediate_help.append("Severe mood episodes detected - consider immediate professional support")
        
        if (df['stress_level'] >= 9).any():
            immediate_help.append("Extreme stress levels detected - seek professional help immediately")
        
        # Professional therapy recommendations
        therapy_recommendations = []
        
        # Depression indicators
        if (df['mood_score'] <= 4).sum() / len(df) > 0.3:
            therapy_recommendations.append("Consider evaluation for depression by a mental health professional")
        
        # Anxiety indicators
        if (df['stress_level'] >= 7).sum() / len(df) > 0.4:
            therapy_recommendations.append("Consider anxiety evaluation and treatment")
        
        # Sleep disorder indicators
        if df['sleep_hours'].std() > 2 or (df['sleep_hours'] < 5).sum() / len(df) > 0.2:
            therapy_recommendations.append("Consider sleep disorder evaluation")
        
        # Specialist referrals
        specialist_referrals = []
        
        # Psychiatrist referral
        if len(immediate_help) > 0 or len(therapy_recommendations) > 2:
            specialist_referrals.append("Consider psychiatrist evaluation for medication assessment")
        
        # Sleep specialist
        if (df['sleep_hours'] < 5).sum() / len(df) > 0.2:
            specialist_referrals.append("Consider sleep specialist consultation")
        
        recommendations['immediate_help'] = immediate_help
        recommendations['therapy_recommendations'] = therapy_recommendations
        recommendations['specialist_referrals'] = specialist_referrals
        recommendations['urgency_level'] = self._determine_urgency_level(immediate_help, therapy_recommendations)
        
        return recommendations
    
    # Helper methods for calculations
    def _detect_social_decline(self, df: pd.DataFrame) -> bool:
        """Detect declining social interactions"""
        if len(df) < 14:
            return False
        
        recent_social = df.tail(14)['social_interactions'].mean()
        earlier_social = df.head(14)['social_interactions'].mean()
        
        return recent_social < earlier_social * 0.5
    
    def _detect_exercise_decline(self, df: pd.DataFrame) -> bool:
        """Detect declining exercise patterns"""
        if len(df) < 14:
            return False
        
        recent_exercise = df.tail(14)['exercise_minutes'].mean()
        earlier_exercise = df.head(14)['exercise_minutes'].mean()
        
        return recent_exercise < earlier_exercise * 0.5
    
    def _calculate_depression_risk_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate depression risk score"""
        score = 0
        factors = []
        
        # Low mood frequency
        if indicators['persistent_low_mood']['percentage'] > 40:
            score += 0.3
            factors.append('persistent_low_mood')
        elif indicators['persistent_low_mood']['percentage'] > 20:
            score += 0.2
        
        # Sleep issues
        if indicators['sleep_disturbances']['sleep_variability'] > 2:
            score += 0.2
            factors.append('sleep_disturbances')
        
        # Social withdrawal
        if indicators['social_withdrawal']['isolation_days'] > len(indicators) * 0.3:
            score += 0.2
            factors.append('social_withdrawal')
        
        # Energy/fatigue
        if indicators['energy_fatigue']['low_exercise_days'] > len(indicators) * 0.5:
            score += 0.2
            factors.append('low_energy')
        
        return {
            'score': min(score, 1.0),
            'risk_level': 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low',
            'contributing_factors': factors
        }
    
    def _calculate_anxiety_risk_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate anxiety risk score"""
        score = 0
        factors = []
        
        # High stress frequency
        if indicators['high_stress_frequency']['percentage'] > 50:
            score += 0.3
            factors.append('high_stress_frequency')
        elif indicators['high_stress_frequency']['percentage'] > 25:
            score += 0.2
        
        # Mood volatility
        if indicators['mood_volatility']['average_volatility'] > 2:
            score += 0.2
            factors.append('mood_volatility')
        
        # Sleep anxiety
        if indicators['sleep_anxiety']['insomnia_pattern'] > 10:
            score += 0.2
            factors.append('sleep_anxiety')
        
        return {
            'score': min(score, 1.0),
            'risk_level': 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low',
            'contributing_factors': factors
        }
    
    def _calculate_stress_risk_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate stress risk score"""
        score = 0
        factors = []
        
        # Chronic stress
        if indicators['chronic_stress']['avg_stress_level'] > 7:
            score += 0.3
            factors.append('chronic_high_stress')
        
        # Stress-mood impact
        if indicators['stress_mood_impact']['correlation'] < -0.5:
            score += 0.2
            factors.append('stress_mood_impact')
        
        return {
            'score': min(score, 1.0),
            'risk_level': 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low',
            'contributing_factors': factors
        }
    
    def _calculate_sleep_disorder_risk_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate sleep disorder risk score"""
        score = 0
        factors = []
        
        # Sleep duration issues
        if indicators['sleep_duration']['optimal_sleep_percentage'] < 50:
            score += 0.2
            factors.append('poor_sleep_duration')
        
        # Sleep consistency
        if indicators['sleep_consistency']['sleep_variability'] > 2:
            score += 0.2
            factors.append('inconsistent_sleep')
        
        return {
            'score': min(score, 1.0),
            'risk_level': 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low',
            'contributing_factors': factors
        }
    
    def _calculate_sad_risk_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate Seasonal Affective Disorder risk score"""
        score = 0
        factors = []
        
        if 'seasonal_patterns' in indicators:
            # Winter depression risk
            if indicators['seasonal_patterns']['winter_depression_risk']:
                score += 0.4
                factors.append('winter_depression_pattern')
            
            # Seasonal difference
            if indicators['seasonal_patterns']['seasonal_difference'] > 1.0:
                score += 0.2
                factors.append('significant_seasonal_variation')
        
        return {
            'score': min(score, 1.0),
            'risk_level': 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low',
            'contributing_factors': factors
        }
    
    def _calculate_isolation_risk_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate social isolation risk score"""
        score = 0
        factors = []
        
        # Isolation percentage
        if indicators['social_patterns']['isolation_percentage'] > 50:
            score += 0.3
            factors.append('high_isolation_frequency')
        
        # Social decline
        if indicators['social_patterns']['social_decline_trend']:
            score += 0.2
            factors.append('declining_social_connections')
        
        return {
            'score': min(score, 1.0),
            'risk_level': 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low',
            'contributing_factors': factors
        }
    
    def _calculate_burnout_risk_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate burnout risk score"""
        score = 0
        factors = []
        
        # Chronic stress
        if indicators['stress_mood_pattern']['chronic_high_stress'] > 60:
            score += 0.3
            factors.append('chronic_high_stress')
        
        # Energy decline
        if indicators['energy_indicators']['exercise_decline']:
            score += 0.2
            factors.append('energy_decline')
        
        return {
            'score': min(score, 1.0),
            'risk_level': 'high' if score > 0.7 else 'moderate' if score > 0.4 else 'low',
            'contributing_factors': factors
        }
    
    def _calculate_bipolar_indicators_score(self, indicators: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate bipolar pattern indicators score (NOT a diagnosis)"""
        score = 0
        factors = []
        
        # Mood volatility
        if indicators['mood_volatility']['overall_volatility'] > 2.5:
            score += 0.2
            factors.append('high_mood_volatility')
        
        # Extreme mood episodes
        if indicators['mood_volatility']['extreme_high_days'] > 0 and indicators['mood_volatility']['extreme_low_days'] > 0:
            score += 0.2
            factors.append('extreme_mood_episodes')
        
        return {
            'score': min(score, 1.0),
            'risk_level': 'indicators_present' if score > 0.3 else 'few_indicators',
            'contributing_factors': factors,
            'note': 'This is NOT a diagnosis - consult a mental health professional'
        }
    
    def _identify_red_flags(self, df: pd.DataFrame) -> List[str]:
        """Identify red flags requiring immediate attention"""
        red_flags = []
        
        # Severe mood episodes
        if (df['mood_score'] <= 2).any():
            red_flags.append("Severe mood episodes (score ≤ 2)")
        
        # Extreme stress
        if (df['stress_level'] >= 9).any():
            red_flags.append("Extreme stress levels (≥ 9/10)")
        
        # Severe sleep deprivation
        if (df['sleep_hours'] < 3).any():
            red_flags.append("Severe sleep deprivation (< 3 hours)")
        
        # Prolonged isolation
        if (df['social_interactions'] == 0).sum() > 14:
            red_flags.append("Prolonged social isolation (> 14 days)")
        
        # Concerning emotion patterns
        if 'emotions' in df.columns:
            concerning_emotions = ['suicidal', 'hopeless', 'worthless']
            for emotions in df['emotions']:
                if isinstance(emotions, list):
                    if any(emotion in concerning_emotions for emotion in emotions):
                        red_flags.append("Concerning emotional states detected")
                        break
        
        return red_flags
    
    def _determine_urgency_level(self, immediate_help: List[str], therapy_recommendations: List[str]) -> str:
        """Determine urgency level for seeking help"""
        if len(immediate_help) > 0:
            return 'immediate'
        elif len(therapy_recommendations) > 2:
            return 'high'
        elif len(therapy_recommendations) > 0:
            return 'moderate'
        else:
            return 'low'
    
    # Additional helper methods (simplified implementations)
    def _detect_sleep_onset_issues(self, df: pd.DataFrame) -> int:
        """Detect sleep onset issues"""
        return (df['sleep_hours'] < 6).sum()
    
    def _detect_chronic_stress_periods(self, df: pd.DataFrame) -> int:
        """Detect chronic stress periods"""
        return (df['stress_level'] >= 7).sum()
    
    def _analyze_stress_trend(self, df: pd.DataFrame) -> str:
        """Analyze stress trend"""
        recent_stress = df.tail(14)['stress_level'].mean()
        earlier_stress = df.head(14)['stress_level'].mean()
        
        if recent_stress > earlier_stress + 1:
            return 'increasing'
        elif recent_stress < earlier_stress - 1:
            return 'decreasing'
        else:
            return 'stable'
    
    def _analyze_stress_sleep_relationship(self, df: pd.DataFrame) -> float:
        """Analyze relationship between stress and sleep"""
        return df['stress_level'].corr(df['sleep_hours'])
    
    def _analyze_stress_exercise_relationship(self, df: pd.DataFrame) -> float:
        """Analyze relationship between stress and exercise"""
        return df['stress_level'].corr(df['exercise_minutes'])
    
    def _analyze_stress_social_relationship(self, df: pd.DataFrame) -> float:
        """Analyze relationship between stress and social interactions"""
        return df['stress_level'].corr(df['social_interactions'])
    
    def _analyze_winter_sleep_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze winter sleep changes"""
        winter_months = [11, 12, 1, 2]
        winter_sleep = df[df['month'].isin(winter_months)]['sleep_hours'].mean()
        other_sleep = df[~df['month'].isin(winter_months)]['sleep_hours'].mean()
        
        return {
            'winter_sleep_avg': winter_sleep,
            'other_months_avg': other_sleep,
            'difference': winter_sleep - other_sleep
        }
    
    def _analyze_winter_exercise_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze winter exercise changes"""
        winter_months = [11, 12, 1, 2]
        winter_exercise = df[df['month'].isin(winter_months)]['exercise_minutes'].mean()
        other_exercise = df[~df['month'].isin(winter_months)]['exercise_minutes'].mean()
        
        return {
            'winter_exercise_avg': winter_exercise,
            'other_months_avg': other_exercise,
            'difference': winter_exercise - other_exercise
        }
    
    def _analyze_winter_social_changes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze winter social changes"""
        winter_months = [11, 12, 1, 2]
        winter_social = df[df['month'].isin(winter_months)]['social_interactions'].mean()
        other_social = df[~df['month'].isin(winter_months)]['social_interactions'].mean()
        
        return {
            'winter_social_avg': winter_social,
            'other_months_avg': other_social,
            'difference': winter_social - other_social
        }
    
    def _analyze_social_trend(self, df: pd.DataFrame) -> str:
        """Analyze social interaction trend"""
        recent_social = df.tail(14)['social_interactions'].mean()
        earlier_social = df.head(14)['social_interactions'].mean()
        
        if recent_social < earlier_social * 0.5:
            return 'declining'
        elif recent_social > earlier_social * 1.5:
            return 'increasing'
        else:
            return 'stable'
    
    def _detect_isolation_periods(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Detect extended isolation periods"""
        isolation_periods = []
        current_period_start = None
        current_period_length = 0
        
        for idx, interactions in df['social_interactions'].items():
            if interactions == 0:
                if current_period_start is None:
                    current_period_start = idx
                current_period_length += 1
            else:
                if current_period_start is not None and current_period_length >= 7:
                    isolation_periods.append({
                        'start_date': current_period_start,
                        'end_date': idx,
                        'duration_days': current_period_length
                    })
                current_period_start = None
                current_period_length = 0
        
        return isolation_periods
    
    def _detect_emotional_exhaustion(self, df: pd.DataFrame) -> bool:
        """Detect emotional exhaustion patterns"""
        # High stress with low mood consistently
        exhaustion_days = ((df['stress_level'] >= 7) & (df['mood_score'] <= 4)).sum()
        return exhaustion_days > len(df) * 0.3
    
    def _analyze_sleep_stress_relationship(self, df: pd.DataFrame) -> float:
        """Analyze sleep-stress relationship"""
        return df['sleep_hours'].corr(df['stress_level'])
    
    def _analyze_weekend_recovery(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekend recovery patterns"""
        df_weekend = df.copy()
        df_weekend['is_weekend'] = df_weekend.index.dayofweek >= 5
        
        weekend_mood = df_weekend[df_weekend['is_weekend']]['mood_score'].mean()
        weekday_mood = df_weekend[~df_weekend['is_weekend']]['mood_score'].mean()
        
        return {
            'weekend_mood_avg': weekend_mood,
            'weekday_mood_avg': weekday_mood,
            'recovery_difference': weekend_mood - weekday_mood
        }
    
    def _detect_rapid_cycling(self, df: pd.DataFrame) -> bool:
        """Detect rapid cycling patterns"""
        mood_changes = df['mood_score'].diff().abs()
        rapid_changes = (mood_changes > 3).sum()
        return rapid_changes > len(df) * 0.1
    
    def _analyze_sleep_during_high_mood(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sleep during high mood periods"""
        high_mood_days = df[df['mood_score'] >= 8]
        
        if len(high_mood_days) > 0:
            return {
                'avg_sleep_hours': high_mood_days['sleep_hours'].mean(),
                'sleep_reduction': high_mood_days['sleep_hours'].mean() < df['sleep_hours'].mean() - 1
            }
        else:
            return {'avg_sleep_hours': 0, 'sleep_reduction': False}
    
    def _analyze_sleep_during_low_mood(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sleep during low mood periods"""
        low_mood_days = df[df['mood_score'] <= 4]
        
        if len(low_mood_days) > 0:
            return {
                'avg_sleep_hours': low_mood_days['sleep_hours'].mean(),
                'sleep_disturbance': abs(low_mood_days['sleep_hours'].mean() - df['sleep_hours'].mean()) > 1
            }
        else:
            return {'avg_sleep_hours': 0, 'sleep_disturbance': False}
    
    def _analyze_exercise_during_high_mood(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze exercise during high mood periods"""
        high_mood_days = df[df['mood_score'] >= 8]
        
        if len(high_mood_days) > 0:
            return {
                'avg_exercise_minutes': high_mood_days['exercise_minutes'].mean(),
                'exercise_increase': high_mood_days['exercise_minutes'].mean() > df['exercise_minutes'].mean() * 1.5
            }
        else:
            return {'avg_exercise_minutes': 0, 'exercise_increase': False}
    
    def _analyze_exercise_during_low_mood(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze exercise during low mood periods"""
        low_mood_days = df[df['mood_score'] <= 4]
        
        if len(low_mood_days) > 0:
            return {
                'avg_exercise_minutes': low_mood_days['exercise_minutes'].mean(),
                'exercise_decrease': low_mood_days['exercise_minutes'].mean() < df['exercise_minutes'].mean() * 0.5
            }
        else:
            return {'avg_exercise_minutes': 0, 'exercise_decrease': False}
    
    def _detect_activity_extremes(self, df: pd.DataFrame) -> bool:
        """Detect extreme activity patterns"""
        high_activity_days = (df['exercise_minutes'] > df['exercise_minutes'].mean() + 2 * df['exercise_minutes'].std()).sum()
        low_activity_days = (df['exercise_minutes'] < df['exercise_minutes'].mean() - 2 * df['exercise_minutes'].std()).sum()
        
        return (high_activity_days + low_activity_days) > len(df) * 0.1
    
    def get_mental_health_report(self) -> Dict[str, Any]:
        """Get comprehensive mental health indicators report"""
        if not self.indicators_results:
            return {'error': 'No mental health indicators results available. Run analyze_mental_health_indicators() first.'}
        
        return {
            'overall_risk_assessment': self.indicators_results.get('overall_risk_assessment', {}),
            'professional_help_recommendations': self.indicators_results.get('professional_help_recommendations', {}),
            'key_indicators': {
                'depression_risk': self.indicators_results.get('depression_indicators', {}).get('depression_risk_score', {}),
                'anxiety_risk': self.indicators_results.get('anxiety_indicators', {}).get('anxiety_risk_score', {}),
                'stress_risk': self.indicators_results.get('stress_indicators', {}).get('stress_risk_score', {}),
                'isolation_risk': self.indicators_results.get('social_isolation_indicators', {}).get('isolation_risk_score', {})
            }
        } 