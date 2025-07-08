"""
Anomaly Detector - Detect concerning patterns and anomalies in mood data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class AnomalyDetector:
    """
    Detect concerning patterns and anomalies in mood tracking data
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize anomaly detector
        
        Args:
            contamination: Expected proportion of anomalies in the data
        """
        self.contamination = contamination
        self.scaler = StandardScaler()
        self.models = {}
        self.anomaly_results = {}
        
    def detect_anomalies(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive anomaly detection across multiple dimensions
        
        Args:
            data: List of mood tracking entries
            
        Returns:
            Dictionary with anomaly detection results
        """
        logger.info("Starting comprehensive anomaly detection")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Prepare features for analysis
        features_df = self._prepare_features(df)
        
        results = {
            'statistical_anomalies': self._detect_statistical_anomalies(df),
            'mood_anomalies': self._detect_mood_anomalies(features_df),
            'behavioral_anomalies': self._detect_behavioral_anomalies(features_df),
            'temporal_anomalies': self._detect_temporal_anomalies(df),
            'multivariate_anomalies': self._detect_multivariate_anomalies(features_df),
            'crisis_indicators': self._detect_crisis_indicators(df),
            'concerning_patterns': self._detect_concerning_patterns(df),
            'anomaly_summary': self._summarize_anomalies(df, features_df)
        }
        
        self.anomaly_results = results
        logger.info("Anomaly detection completed")
        
        return results
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for anomaly detection"""
        features = df.copy()
        
        # Rolling statistics
        features['mood_ma_7'] = features['mood_score'].rolling(window=7).mean()
        features['mood_ma_14'] = features['mood_score'].rolling(window=14).mean()
        features['mood_std_7'] = features['mood_score'].rolling(window=7).std()
        features['mood_change'] = features['mood_score'].diff()
        features['mood_change_rate'] = features['mood_change'].rolling(window=3).mean()
        
        # Sleep patterns
        features['sleep_deviation'] = features['sleep_hours'] - features['sleep_hours'].rolling(window=14).mean()
        features['sleep_variability'] = features['sleep_hours'].rolling(window=7).std()
        
        # Exercise patterns
        features['exercise_deviation'] = features['exercise_minutes'] - features['exercise_minutes'].rolling(window=14).mean()
        features['exercise_consistency'] = features['exercise_minutes'].rolling(window=7).std()
        
        # Stress patterns
        features['stress_change'] = features['stress_level'].diff()
        features['stress_trend'] = features['stress_level'].rolling(window=7).mean()
        
        # Social patterns
        features['social_change'] = features['social_interactions'].diff()
        features['social_isolation'] = (features['social_interactions'] < 1).astype(int)
        
        # Time-based features
        features['day_of_week'] = features.index.dayofweek
        features['hour'] = features.index.hour
        features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
        
        return features.fillna(method='ffill').fillna(method='bfill')
    
    def _detect_statistical_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect statistical anomalies using standard deviation and IQR"""
        anomalies = {}
        
        # Z-score based anomalies
        for column in ['mood_score', 'sleep_hours', 'exercise_minutes', 'stress_level']:
            if column in df.columns:
                z_scores = np.abs(stats.zscore(df[column]))
                z_threshold = 3
                
                outliers = df[z_scores > z_threshold]
                anomalies[f'{column}_zscore'] = {
                    'count': len(outliers),
                    'dates': outliers.index.tolist(),
                    'values': outliers[column].tolist(),
                    'severity': 'high' if len(outliers) > len(df) * 0.05 else 'medium'
                }
        
        # IQR based anomalies
        for column in ['mood_score', 'sleep_hours', 'exercise_minutes', 'stress_level']:
            if column in df.columns:
                Q1 = df[column].quantile(0.25)
                Q3 = df[column].quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
                anomalies[f'{column}_iqr'] = {
                    'count': len(outliers),
                    'dates': outliers.index.tolist(),
                    'values': outliers[column].tolist(),
                    'lower_bound': lower_bound,
                    'upper_bound': upper_bound
                }
        
        return anomalies
    
    def _detect_mood_anomalies(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect mood-specific anomalies"""
        anomalies = {}
        
        # Sudden mood drops
        mood_changes = features_df['mood_change'].fillna(0)
        severe_drops = features_df[mood_changes <= -3]
        
        anomalies['sudden_mood_drops'] = {
            'count': len(severe_drops),
            'dates': severe_drops.index.tolist(),
            'changes': mood_changes[mood_changes <= -3].tolist(),
            'severity': 'high' if len(severe_drops) > 0 else 'low'
        }
        
        # Prolonged low mood
        low_mood_threshold = 3
        low_mood_streak = 0
        max_streak = 0
        current_streak_start = None
        prolonged_periods = []
        
        for idx, mood in features_df['mood_score'].items():
            if mood <= low_mood_threshold:
                if low_mood_streak == 0:
                    current_streak_start = idx
                low_mood_streak += 1
                max_streak = max(max_streak, low_mood_streak)
            else:
                if low_mood_streak >= 7:  # 7+ days of low mood
                    prolonged_periods.append({
                        'start_date': current_streak_start,
                        'end_date': idx,
                        'duration': low_mood_streak,
                        'average_mood': features_df.loc[current_streak_start:idx, 'mood_score'].mean()
                    })
                low_mood_streak = 0
        
        anomalies['prolonged_low_mood'] = {
            'count': len(prolonged_periods),
            'periods': prolonged_periods,
            'max_streak': max_streak,
            'severity': 'high' if max_streak >= 14 else 'medium' if max_streak >= 7 else 'low'
        }
        
        # Mood volatility
        mood_volatility = features_df['mood_score'].rolling(window=7).std()
        high_volatility_threshold = mood_volatility.quantile(0.9)
        volatile_periods = features_df[mood_volatility > high_volatility_threshold]
        
        anomalies['high_mood_volatility'] = {
            'count': len(volatile_periods),
            'dates': volatile_periods.index.tolist(),
            'volatility_scores': mood_volatility[mood_volatility > high_volatility_threshold].tolist(),
            'threshold': high_volatility_threshold
        }
        
        return anomalies
    
    def _detect_behavioral_anomalies(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect behavioral pattern anomalies"""
        anomalies = {}
        
        # Sleep disruption
        sleep_disruption = features_df[
            (features_df['sleep_hours'] < 4) | 
            (features_df['sleep_hours'] > 12) |
            (features_df['sleep_variability'] > 2)
        ]
        
        anomalies['sleep_disruption'] = {
            'count': len(sleep_disruption),
            'dates': sleep_disruption.index.tolist(),
            'sleep_hours': sleep_disruption['sleep_hours'].tolist(),
            'severity': 'high' if len(sleep_disruption) > len(features_df) * 0.1 else 'medium'
        }
        
        # Exercise anomalies
        exercise_cessation = features_df[
            (features_df['exercise_minutes'] == 0) &
            (features_df['exercise_minutes'].rolling(window=7).mean() < 30)
        ]
        
        anomalies['exercise_cessation'] = {
            'count': len(exercise_cessation),
            'dates': exercise_cessation.index.tolist(),
            'severity': 'medium' if len(exercise_cessation) > len(features_df) * 0.3 else 'low'
        }
        
        # Social isolation
        social_isolation = features_df[
            (features_df['social_interactions'] == 0) &
            (features_df['social_interactions'].rolling(window=7).mean() < 1)
        ]
        
        anomalies['social_isolation'] = {
            'count': len(social_isolation),
            'dates': social_isolation.index.tolist(),
            'severity': 'high' if len(social_isolation) > len(features_df) * 0.2 else 'medium'
        }
        
        # Stress escalation
        stress_escalation = features_df[
            (features_df['stress_level'] >= 8) &
            (features_df['stress_trend'] > 7)
        ]
        
        anomalies['stress_escalation'] = {
            'count': len(stress_escalation),
            'dates': stress_escalation.index.tolist(),
            'stress_levels': stress_escalation['stress_level'].tolist(),
            'severity': 'high' if len(stress_escalation) > 0 else 'low'
        }
        
        return anomalies
    
    def _detect_temporal_anomalies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect temporal pattern anomalies"""
        anomalies = {}
        
        # Unusual time patterns
        df_temporal = df.copy()
        df_temporal['hour'] = df_temporal.index.hour
        df_temporal['day_of_week'] = df_temporal.index.dayofweek
        
        # Late night entries (potential insomnia indicator)
        late_night_entries = df_temporal[df_temporal['hour'].between(2, 5)]
        
        anomalies['late_night_entries'] = {
            'count': len(late_night_entries),
            'dates': late_night_entries.index.tolist(),
            'hours': late_night_entries['hour'].tolist(),
            'mood_scores': late_night_entries['mood_score'].tolist(),
            'severity': 'medium' if len(late_night_entries) > len(df) * 0.1 else 'low'
        }
        
        # Weekend vs weekday anomalies
        weekday_mood = df_temporal[df_temporal['day_of_week'] < 5]['mood_score'].mean()
        weekend_mood = df_temporal[df_temporal['day_of_week'] >= 5]['mood_score'].mean()
        
        weekend_anomaly = abs(weekend_mood - weekday_mood) > 2
        
        anomalies['weekend_pattern_anomaly'] = {
            'anomaly_detected': weekend_anomaly,
            'weekday_average': weekday_mood,
            'weekend_average': weekend_mood,
            'difference': weekend_mood - weekday_mood,
            'severity': 'medium' if weekend_anomaly else 'low'
        }
        
        return anomalies
    
    def _detect_multivariate_anomalies(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Detect multivariate anomalies using machine learning"""
        anomalies = {}
        
        # Select features for multivariate analysis
        feature_columns = [
            'mood_score', 'sleep_hours', 'exercise_minutes', 'stress_level',
            'social_interactions', 'mood_change', 'sleep_deviation'
        ]
        
        # Filter available columns
        available_columns = [col for col in feature_columns if col in features_df.columns]
        
        if len(available_columns) < 3:
            return {'error': 'Insufficient features for multivariate analysis'}
        
        # Prepare data
        X = features_df[available_columns].dropna()
        
        if len(X) < 10:
            return {'error': 'Insufficient data for multivariate analysis'}
        
        # Normalize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        iso_forest = IsolationForest(contamination=self.contamination, random_state=42)
        iso_predictions = iso_forest.fit_predict(X_scaled)
        iso_anomalies = X[iso_predictions == -1]
        
        # One-Class SVM
        svm_model = OneClassSVM(nu=self.contamination)
        svm_predictions = svm_model.fit_predict(X_scaled)
        svm_anomalies = X[svm_predictions == -1]
        
        # DBSCAN clustering
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        dbscan_anomalies = X[dbscan_labels == -1]
        
        anomalies['isolation_forest'] = {
            'count': len(iso_anomalies),
            'dates': iso_anomalies.index.tolist(),
            'anomaly_scores': iso_forest.decision_function(X_scaled)[iso_predictions == -1].tolist()
        }
        
        anomalies['one_class_svm'] = {
            'count': len(svm_anomalies),
            'dates': svm_anomalies.index.tolist(),
            'anomaly_scores': svm_model.decision_function(X_scaled)[svm_predictions == -1].tolist()
        }
        
        anomalies['dbscan_outliers'] = {
            'count': len(dbscan_anomalies),
            'dates': dbscan_anomalies.index.tolist(),
            'cluster_labels': dbscan_labels[dbscan_labels == -1].tolist()
        }
        
        # Consensus anomalies (detected by multiple methods)
        consensus_dates = set(iso_anomalies.index) & set(svm_anomalies.index)
        anomalies['consensus_anomalies'] = {
            'count': len(consensus_dates),
            'dates': list(consensus_dates),
            'severity': 'high' if len(consensus_dates) > 0 else 'low'
        }
        
        return anomalies
    
    def _detect_crisis_indicators(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect potential crisis indicators"""
        crisis_indicators = {}
        
        # Severe mood episodes
        severe_low_mood = df[df['mood_score'] <= 2]
        crisis_indicators['severe_low_mood'] = {
            'count': len(severe_low_mood),
            'dates': severe_low_mood.index.tolist(),
            'mood_scores': severe_low_mood['mood_score'].tolist(),
            'severity': 'critical' if len(severe_low_mood) > 0 else 'low'
        }
        
        # Extreme stress levels
        extreme_stress = df[df['stress_level'] >= 9]
        crisis_indicators['extreme_stress'] = {
            'count': len(extreme_stress),
            'dates': extreme_stress.index.tolist(),
            'stress_levels': extreme_stress['stress_level'].tolist(),
            'severity': 'high' if len(extreme_stress) > 0 else 'low'
        }
        
        # Sleep deprivation
        sleep_deprivation = df[df['sleep_hours'] < 3]
        crisis_indicators['severe_sleep_deprivation'] = {
            'count': len(sleep_deprivation),
            'dates': sleep_deprivation.index.tolist(),
            'sleep_hours': sleep_deprivation['sleep_hours'].tolist(),
            'severity': 'high' if len(sleep_deprivation) > 0 else 'low'
        }
        
        # Complete social isolation
        complete_isolation = df[
            (df['social_interactions'] == 0) &
            (df['social_interactions'].rolling(window=14).sum() == 0)
        ]
        crisis_indicators['complete_social_isolation'] = {
            'count': len(complete_isolation),
            'dates': complete_isolation.index.tolist(),
            'severity': 'high' if len(complete_isolation) > 0 else 'low'
        }
        
        # Concerning emotion patterns
        if 'emotions' in df.columns:
            crisis_emotions = ['depressed', 'hopeless', 'suicidal', 'overwhelmed']
            crisis_emotion_entries = []
            
            for idx, emotions in df['emotions'].items():
                if isinstance(emotions, list):
                    if any(emotion in crisis_emotions for emotion in emotions):
                        crisis_emotion_entries.append(idx)
            
            crisis_indicators['concerning_emotions'] = {
                'count': len(crisis_emotion_entries),
                'dates': crisis_emotion_entries,
                'severity': 'critical' if len(crisis_emotion_entries) > 0 else 'low'
            }
        
        return crisis_indicators
    
    def _detect_concerning_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect overall concerning patterns"""
        patterns = {}
        
        # Declining trend
        recent_period = df.tail(14)  # Last 2 weeks
        older_period = df.head(14)  # First 2 weeks
        
        if len(recent_period) >= 7 and len(older_period) >= 7:
            recent_avg = recent_period['mood_score'].mean()
            older_avg = older_period['mood_score'].mean()
            
            declining_trend = (older_avg - recent_avg) > 1.5
            
            patterns['declining_mood_trend'] = {
                'trend_detected': declining_trend,
                'recent_average': recent_avg,
                'older_average': older_avg,
                'decline_magnitude': older_avg - recent_avg,
                'severity': 'high' if declining_trend else 'low'
            }
        
        # Increasing volatility
        early_volatility = df.head(len(df)//2)['mood_score'].std()
        recent_volatility = df.tail(len(df)//2)['mood_score'].std()
        
        increasing_volatility = recent_volatility > early_volatility * 1.5
        
        patterns['increasing_volatility'] = {
            'pattern_detected': increasing_volatility,
            'early_volatility': early_volatility,
            'recent_volatility': recent_volatility,
            'volatility_ratio': recent_volatility / early_volatility if early_volatility > 0 else 0,
            'severity': 'medium' if increasing_volatility else 'low'
        }
        
        # Behavioral deterioration
        behavioral_scores = []
        
        # Sleep quality decline
        sleep_trend = df['sleep_hours'].rolling(window=7).mean()
        sleep_decline = sleep_trend.iloc[-7:].mean() < sleep_trend.iloc[:7].mean() - 1
        if sleep_decline:
            behavioral_scores.append('sleep_decline')
        
        # Exercise reduction
        exercise_trend = df['exercise_minutes'].rolling(window=7).mean()
        exercise_decline = exercise_trend.iloc[-7:].mean() < exercise_trend.iloc[:7].mean() * 0.5
        if exercise_decline:
            behavioral_scores.append('exercise_decline')
        
        # Social withdrawal
        social_trend = df['social_interactions'].rolling(window=7).mean()
        social_decline = social_trend.iloc[-7:].mean() < social_trend.iloc[:7].mean() * 0.5
        if social_decline:
            behavioral_scores.append('social_withdrawal')
        
        patterns['behavioral_deterioration'] = {
            'deterioration_indicators': behavioral_scores,
            'count': len(behavioral_scores),
            'severity': 'high' if len(behavioral_scores) >= 2 else 'medium' if len(behavioral_scores) == 1 else 'low'
        }
        
        return patterns
    
    def _summarize_anomalies(self, df: pd.DataFrame, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize all anomaly detection results"""
        summary = {
            'total_entries': len(df),
            'analysis_period': {
                'start': df.index.min(),
                'end': df.index.max(),
                'duration_days': (df.index.max() - df.index.min()).days
            },
            'overall_risk_assessment': 'low',
            'priority_concerns': [],
            'recommendations': []
        }
        
        # Collect all high-severity anomalies
        high_severity_count = 0
        critical_count = 0
        
        if hasattr(self, 'anomaly_results'):
            for category, results in self.anomaly_results.items():
                if isinstance(results, dict):
                    for anomaly_type, details in results.items():
                        if isinstance(details, dict) and 'severity' in details:
                            if details['severity'] == 'critical':
                                critical_count += 1
                                summary['priority_concerns'].append({
                                    'category': category,
                                    'type': anomaly_type,
                                    'severity': 'critical',
                                    'count': details.get('count', 0)
                                })
                            elif details['severity'] == 'high':
                                high_severity_count += 1
                                summary['priority_concerns'].append({
                                    'category': category,
                                    'type': anomaly_type,
                                    'severity': 'high',
                                    'count': details.get('count', 0)
                                })
        
        # Determine overall risk
        if critical_count > 0:
            summary['overall_risk_assessment'] = 'critical'
        elif high_severity_count >= 3:
            summary['overall_risk_assessment'] = 'high'
        elif high_severity_count >= 1:
            summary['overall_risk_assessment'] = 'medium'
        
        # Generate recommendations
        summary['recommendations'] = self._generate_anomaly_recommendations(
            summary['overall_risk_assessment'], 
            summary['priority_concerns']
        )
        
        return summary
    
    def _generate_anomaly_recommendations(self, risk_level: str, concerns: List[Dict]) -> List[str]:
        """Generate recommendations based on anomaly detection results"""
        recommendations = []
        
        if risk_level == 'critical':
            recommendations.append("🚨 URGENT: Consider immediate professional mental health support")
            recommendations.append("Contact a crisis helpline or emergency services if needed")
            recommendations.append("Reach out to trusted friends, family, or support network")
        
        elif risk_level == 'high':
            recommendations.append("⚠️ HIGH PRIORITY: Schedule appointment with mental health professional")
            recommendations.append("Implement daily check-ins with trusted person")
            recommendations.append("Consider professional counseling or therapy")
        
        elif risk_level == 'medium':
            recommendations.append("Consider speaking with healthcare provider about mood patterns")
            recommendations.append("Implement stress management techniques")
            recommendations.append("Focus on sleep hygiene and regular exercise")
        
        # Specific recommendations based on concerns
        concern_types = [c['type'] for c in concerns]
        
        if 'severe_low_mood' in concern_types:
            recommendations.append("Monitor mood closely and seek immediate help if worsening")
        
        if 'sleep_disruption' in concern_types:
            recommendations.append("Prioritize sleep hygiene and consider sleep specialist consultation")
        
        if 'social_isolation' in concern_types:
            recommendations.append("Gradually increase social connections and activities")
        
        if 'extreme_stress' in concern_types:
            recommendations.append("Learn and practice stress reduction techniques")
        
        if 'behavioral_deterioration' in concern_types:
            recommendations.append("Focus on maintaining basic self-care routines")
        
        return recommendations
    
    def get_anomaly_report(self) -> Dict[str, Any]:
        """Get comprehensive anomaly detection report"""
        if not self.anomaly_results:
            return {'error': 'No anomaly detection results available. Run detect_anomalies() first.'}
        
        return {
            'summary': self.anomaly_results.get('anomaly_summary', {}),
            'crisis_indicators': self.anomaly_results.get('crisis_indicators', {}),
            'concerning_patterns': self.anomaly_results.get('concerning_patterns', {}),
            'recommendations': self.anomaly_results.get('anomaly_summary', {}).get('recommendations', [])
        } 