"""
Pattern Recognizer - Identify patterns and trends in mood data using ML techniques
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from scipy import stats
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class PatternRecognizer:
    """
    Identify patterns and trends in mood tracking data using machine learning
    """
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.pattern_results = {}
        self.models = {}
        
    def recognize_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Comprehensive pattern recognition across multiple dimensions
        
        Args:
            data: List of mood tracking entries
            
        Returns:
            Dictionary with pattern recognition results
        """
        logger.info("Starting comprehensive pattern recognition")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Prepare features
        features_df = self._prepare_features(df)
        
        results = {
            'mood_patterns': self._identify_mood_patterns(features_df),
            'behavioral_patterns': self._identify_behavioral_patterns(features_df),
            'temporal_patterns': self._identify_temporal_patterns(df),
            'correlation_patterns': self._identify_correlation_patterns(features_df),
            'clustering_patterns': self._identify_clustering_patterns(features_df),
            'cyclical_patterns': self._identify_cyclical_patterns(df),
            'transition_patterns': self._identify_transition_patterns(df),
            'predictive_patterns': self._identify_predictive_patterns(features_df),
            'pattern_summary': self._summarize_patterns(df, features_df)
        }
        
        self.pattern_results = results
        logger.info("Pattern recognition completed")
        
        return results
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare features for pattern recognition"""
        features = df.copy()
        
        # Rolling statistics
        for window in [3, 7, 14, 30]:
            features[f'mood_ma_{window}'] = features['mood_score'].rolling(window=window).mean()
            features[f'mood_std_{window}'] = features['mood_score'].rolling(window=window).std()
            features[f'stress_ma_{window}'] = features['stress_level'].rolling(window=window).mean()
            features[f'sleep_ma_{window}'] = features['sleep_hours'].rolling(window=window).mean()
        
        # Derivatives and changes
        features['mood_change'] = features['mood_score'].diff()
        features['mood_change_rate'] = features['mood_change'].rolling(window=3).mean()
        features['mood_acceleration'] = features['mood_change'].diff()
        
        # Momentum indicators
        features['mood_momentum'] = features['mood_score'].rolling(window=7).mean() - features['mood_score'].rolling(window=14).mean()
        features['stress_momentum'] = features['stress_level'].rolling(window=7).mean() - features['stress_level'].rolling(window=14).mean()
        
        # Volatility measures
        features['mood_volatility'] = features['mood_score'].rolling(window=7).std()
        features['stress_volatility'] = features['stress_level'].rolling(window=7).std()
        
        # Behavioral ratios
        features['sleep_exercise_ratio'] = features['sleep_hours'] / (features['exercise_minutes'] / 60 + 1)
        features['stress_social_ratio'] = features['stress_level'] / (features['social_interactions'] + 1)
        
        # Time-based features
        features['day_of_week'] = features.index.dayofweek
        features['hour'] = features.index.hour
        features['day_of_month'] = features.index.day
        features['month'] = features.index.month
        features['is_weekend'] = (features.index.dayofweek >= 5).astype(int)
        features['is_monday'] = (features.index.dayofweek == 0).astype(int)
        features['is_friday'] = (features.index.dayofweek == 4).astype(int)
        
        return features.fillna(method='ffill').fillna(method='bfill')
    
    def _identify_mood_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify mood-specific patterns"""
        patterns = {}
        
        # Mood cycles
        mood_cycles = self._detect_mood_cycles(features_df['mood_score'])
        patterns['mood_cycles'] = mood_cycles
        
        # Mood states clustering
        mood_states = self._cluster_mood_states(features_df)
        patterns['mood_states'] = mood_states
        
        # Mood trend patterns
        trend_patterns = self._analyze_mood_trends(features_df['mood_score'])
        patterns['trend_patterns'] = trend_patterns
        
        # Mood range patterns
        range_patterns = self._analyze_mood_ranges(features_df['mood_score'])
        patterns['range_patterns'] = range_patterns
        
        # Mood stability patterns
        stability_patterns = self._analyze_mood_stability(features_df['mood_score'])
        patterns['stability_patterns'] = stability_patterns
        
        return patterns
    
    def _identify_behavioral_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify behavioral patterns"""
        patterns = {}
        
        # Sleep patterns
        sleep_patterns = self._analyze_sleep_patterns(features_df)
        patterns['sleep_patterns'] = sleep_patterns
        
        # Exercise patterns
        exercise_patterns = self._analyze_exercise_patterns(features_df)
        patterns['exercise_patterns'] = exercise_patterns
        
        # Social patterns
        social_patterns = self._analyze_social_patterns(features_df)
        patterns['social_patterns'] = social_patterns
        
        # Stress patterns
        stress_patterns = self._analyze_stress_patterns(features_df)
        patterns['stress_patterns'] = stress_patterns
        
        # Behavioral clusters
        behavioral_clusters = self._cluster_behavioral_patterns(features_df)
        patterns['behavioral_clusters'] = behavioral_clusters
        
        return patterns
    
    def _identify_temporal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify temporal patterns"""
        patterns = {}
        
        # Day of week patterns
        dow_patterns = self._analyze_day_of_week_patterns(df)
        patterns['day_of_week_patterns'] = dow_patterns
        
        # Time of day patterns
        tod_patterns = self._analyze_time_of_day_patterns(df)
        patterns['time_of_day_patterns'] = tod_patterns
        
        # Monthly patterns
        monthly_patterns = self._analyze_monthly_patterns(df)
        patterns['monthly_patterns'] = monthly_patterns
        
        # Seasonal patterns
        seasonal_patterns = self._analyze_seasonal_patterns(df)
        patterns['seasonal_patterns'] = seasonal_patterns
        
        return patterns
    
    def _identify_correlation_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify correlation patterns between variables"""
        patterns = {}
        
        # Core variable correlations
        core_vars = ['mood_score', 'sleep_hours', 'exercise_minutes', 'stress_level', 'social_interactions']
        available_vars = [var for var in core_vars if var in features_df.columns]
        
        if len(available_vars) >= 2:
            correlation_matrix = features_df[available_vars].corr()
            patterns['core_correlations'] = correlation_matrix.to_dict()
            
            # Strong correlations
            strong_correlations = []
            for i in range(len(correlation_matrix.columns)):
                for j in range(i+1, len(correlation_matrix.columns)):
                    corr_value = correlation_matrix.iloc[i, j]
                    if abs(corr_value) > 0.5:
                        strong_correlations.append({
                            'variable1': correlation_matrix.columns[i],
                            'variable2': correlation_matrix.columns[j],
                            'correlation': corr_value,
                            'strength': 'strong' if abs(corr_value) > 0.7 else 'moderate'
                        })
            
            patterns['strong_correlations'] = strong_correlations
        
        # Lagged correlations
        lagged_correlations = self._analyze_lagged_correlations(features_df)
        patterns['lagged_correlations'] = lagged_correlations
        
        return patterns
    
    def _identify_clustering_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify clustering patterns in the data"""
        patterns = {}
        
        # Mood-based clustering
        mood_clusters = self._perform_mood_clustering(features_df)
        patterns['mood_clusters'] = mood_clusters
        
        # Behavioral clustering
        behavioral_clusters = self._perform_behavioral_clustering(features_df)
        patterns['behavioral_clusters'] = behavioral_clusters
        
        # Temporal clustering
        temporal_clusters = self._perform_temporal_clustering(features_df)
        patterns['temporal_clusters'] = temporal_clusters
        
        return patterns
    
    def _identify_cyclical_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify cyclical patterns in mood data"""
        patterns = {}
        
        # Daily cycles
        daily_cycles = self._detect_daily_cycles(df)
        patterns['daily_cycles'] = daily_cycles
        
        # Weekly cycles
        weekly_cycles = self._detect_weekly_cycles(df)
        patterns['weekly_cycles'] = weekly_cycles
        
        # Monthly cycles
        monthly_cycles = self._detect_monthly_cycles(df)
        patterns['monthly_cycles'] = monthly_cycles
        
        return patterns
    
    def _identify_transition_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Identify state transition patterns"""
        patterns = {}
        
        # Mood state transitions
        mood_transitions = self._analyze_mood_transitions(df)
        patterns['mood_transitions'] = mood_transitions
        
        # Behavioral state transitions
        behavioral_transitions = self._analyze_behavioral_transitions(df)
        patterns['behavioral_transitions'] = behavioral_transitions
        
        return patterns
    
    def _identify_predictive_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify patterns that can predict future mood"""
        patterns = {}
        
        # Feature importance for mood prediction
        feature_importance = self._analyze_feature_importance(features_df)
        patterns['feature_importance'] = feature_importance
        
        # Leading indicators
        leading_indicators = self._identify_leading_indicators(features_df)
        patterns['leading_indicators'] = leading_indicators
        
        return patterns
    
    def _detect_mood_cycles(self, mood_series: pd.Series) -> Dict[str, Any]:
        """Detect cyclical patterns in mood data"""
        # Find peaks and troughs
        peaks, _ = find_peaks(mood_series.values, height=mood_series.mean())
        troughs, _ = find_peaks(-mood_series.values, height=-mood_series.mean())
        
        # Calculate cycle lengths
        cycle_lengths = []
        if len(peaks) > 1:
            cycle_lengths.extend(np.diff(peaks))
        if len(troughs) > 1:
            cycle_lengths.extend(np.diff(troughs))
        
        cycle_info = {
            'peaks_count': len(peaks),
            'troughs_count': len(troughs),
            'average_cycle_length': np.mean(cycle_lengths) if cycle_lengths else 0,
            'cycle_regularity': np.std(cycle_lengths) if cycle_lengths else 0,
            'peak_dates': mood_series.index[peaks].tolist(),
            'trough_dates': mood_series.index[troughs].tolist()
        }
        
        return cycle_info
    
    def _cluster_mood_states(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster mood states using K-means"""
        mood_features = ['mood_score', 'mood_ma_7', 'mood_std_7', 'mood_volatility']
        available_features = [f for f in mood_features if f in features_df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient features for mood clustering'}
        
        X = features_df[available_features].dropna()
        
        if len(X) < 10:
            return {'error': 'Insufficient data for clustering'}
        
        # Find optimal number of clusters
        max_clusters = min(8, len(X) // 5)
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = np.argmax(silhouette_scores) + 2
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Analyze clusters
        cluster_analysis = []
        for i in range(optimal_k):
            cluster_data = X[cluster_labels == i]
            cluster_info = {
                'cluster_id': i,
                'size': len(cluster_data),
                'percentage': len(cluster_data) / len(X) * 100,
                'avg_mood': cluster_data['mood_score'].mean(),
                'characteristics': self._describe_cluster(cluster_data)
            }
            cluster_analysis.append(cluster_info)
        
        return {
            'optimal_clusters': optimal_k,
            'cluster_analysis': cluster_analysis,
            'silhouette_score': silhouette_scores[optimal_k - 2]
        }
    
    def _analyze_mood_trends(self, mood_series: pd.Series) -> Dict[str, Any]:
        """Analyze mood trend patterns"""
        # Overall trend
        x = np.arange(len(mood_series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, mood_series.values)
        
        # Trend segments
        segments = []
        window_size = max(7, len(mood_series) // 10)
        
        for i in range(0, len(mood_series) - window_size, window_size):
            segment = mood_series.iloc[i:i+window_size]
            x_seg = np.arange(len(segment))
            slope_seg, _, r_val_seg, p_val_seg, _ = stats.linregress(x_seg, segment.values)
            
            segments.append({
                'start_date': segment.index[0],
                'end_date': segment.index[-1],
                'slope': slope_seg,
                'r_squared': r_val_seg**2,
                'p_value': p_val_seg,
                'trend_type': 'increasing' if slope_seg > 0 else 'decreasing'
            })
        
        return {
            'overall_trend': {
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'direction': 'increasing' if slope > 0 else 'decreasing'
            },
            'trend_segments': segments
        }
    
    def _analyze_mood_ranges(self, mood_series: pd.Series) -> Dict[str, Any]:
        """Analyze mood range patterns"""
        # Rolling ranges
        rolling_ranges = {}
        for window in [7, 14, 30]:
            rolling_max = mood_series.rolling(window=window).max()
            rolling_min = mood_series.rolling(window=window).min()
            rolling_range = rolling_max - rolling_min
            
            rolling_ranges[f'range_{window}d'] = {
                'mean': rolling_range.mean(),
                'std': rolling_range.std(),
                'max': rolling_range.max(),
                'min': rolling_range.min()
            }
        
        # Percentile analysis
        percentiles = [10, 25, 50, 75, 90]
        percentile_values = {f'p{p}': mood_series.quantile(p/100) for p in percentiles}
        
        return {
            'overall_range': mood_series.max() - mood_series.min(),
            'rolling_ranges': rolling_ranges,
            'percentiles': percentile_values,
            'interquartile_range': percentile_values['p75'] - percentile_values['p25']
        }
    
    def _analyze_mood_stability(self, mood_series: pd.Series) -> Dict[str, Any]:
        """Analyze mood stability patterns"""
        # Stability metrics
        stability_metrics = {}
        
        # Coefficient of variation
        stability_metrics['coefficient_of_variation'] = mood_series.std() / mood_series.mean()
        
        # Change frequency
        changes = mood_series.diff().abs()
        stability_metrics['change_frequency'] = (changes > 1).sum() / len(changes)
        
        # Stability periods
        stable_threshold = 0.5
        stable_periods = []
        current_period_start = None
        
        for i, change in enumerate(changes):
            if change <= stable_threshold:
                if current_period_start is None:
                    current_period_start = i
            else:
                if current_period_start is not None:
                    stable_periods.append({
                        'start': mood_series.index[current_period_start],
                        'end': mood_series.index[i-1],
                        'duration': i - current_period_start
                    })
                    current_period_start = None
        
        stability_metrics['stable_periods'] = stable_periods
        stability_metrics['avg_stable_period'] = np.mean([p['duration'] for p in stable_periods]) if stable_periods else 0
        
        return stability_metrics
    
    def _analyze_sleep_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze sleep patterns"""
        sleep_patterns = {}
        
        # Sleep duration patterns
        sleep_patterns['duration_stats'] = {
            'mean': features_df['sleep_hours'].mean(),
            'std': features_df['sleep_hours'].std(),
            'optimal_range': (7, 9),
            'optimal_percentage': ((features_df['sleep_hours'] >= 7) & (features_df['sleep_hours'] <= 9)).sum() / len(features_df) * 100
        }
        
        # Sleep consistency
        sleep_patterns['consistency'] = {
            'variability': features_df['sleep_hours'].std(),
            'consistency_score': 1 / (1 + features_df['sleep_hours'].std())
        }
        
        return sleep_patterns
    
    def _analyze_exercise_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze exercise patterns"""
        exercise_patterns = {}
        
        # Exercise frequency
        exercise_days = (features_df['exercise_minutes'] > 0).sum()
        exercise_patterns['frequency'] = {
            'days_with_exercise': exercise_days,
            'percentage_active_days': exercise_days / len(features_df) * 100,
            'average_minutes': features_df['exercise_minutes'].mean()
        }
        
        # Exercise consistency
        exercise_patterns['consistency'] = {
            'variability': features_df['exercise_minutes'].std(),
            'consistency_score': 1 / (1 + features_df['exercise_minutes'].std())
        }
        
        return exercise_patterns
    
    def _analyze_social_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze social interaction patterns"""
        social_patterns = {}
        
        # Social frequency
        social_patterns['frequency'] = {
            'average_interactions': features_df['social_interactions'].mean(),
            'isolation_days': (features_df['social_interactions'] == 0).sum(),
            'isolation_percentage': (features_df['social_interactions'] == 0).sum() / len(features_df) * 100
        }
        
        return social_patterns
    
    def _analyze_stress_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze stress patterns"""
        stress_patterns = {}
        
        # Stress levels
        stress_patterns['levels'] = {
            'average_stress': features_df['stress_level'].mean(),
            'high_stress_days': (features_df['stress_level'] >= 7).sum(),
            'high_stress_percentage': (features_df['stress_level'] >= 7).sum() / len(features_df) * 100
        }
        
        return stress_patterns
    
    def _cluster_behavioral_patterns(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Cluster behavioral patterns"""
        behavioral_features = ['sleep_hours', 'exercise_minutes', 'social_interactions', 'stress_level']
        available_features = [f for f in behavioral_features if f in features_df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient features for behavioral clustering'}
        
        X = features_df[available_features].dropna()
        
        if len(X) < 10:
            return {'error': 'Insufficient data for clustering'}
        
        # Perform clustering
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        # Analyze clusters
        cluster_analysis = []
        for i in range(3):
            cluster_data = X[cluster_labels == i]
            cluster_info = {
                'cluster_id': i,
                'size': len(cluster_data),
                'characteristics': {col: cluster_data[col].mean() for col in available_features}
            }
            cluster_analysis.append(cluster_info)
        
        return {'cluster_analysis': cluster_analysis}
    
    def _analyze_day_of_week_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze day of week patterns"""
        df_dow = df.copy()
        df_dow['day_of_week'] = df_dow.index.dayofweek
        df_dow['day_name'] = df_dow.index.day_name()
        
        dow_stats = df_dow.groupby(['day_of_week', 'day_name'])['mood_score'].agg(['mean', 'std', 'count']).reset_index()
        
        return {
            'day_of_week_stats': dow_stats.to_dict('records'),
            'best_day': dow_stats.loc[dow_stats['mean'].idxmax(), 'day_name'],
            'worst_day': dow_stats.loc[dow_stats['mean'].idxmin(), 'day_name']
        }
    
    def _analyze_time_of_day_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze time of day patterns"""
        df_tod = df.copy()
        df_tod['hour'] = df_tod.index.hour
        
        hourly_stats = df_tod.groupby('hour')['mood_score'].agg(['mean', 'std', 'count'])
        
        return {
            'hourly_stats': hourly_stats.to_dict(),
            'peak_hour': hourly_stats['mean'].idxmax(),
            'lowest_hour': hourly_stats['mean'].idxmin()
        }
    
    def _analyze_monthly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly patterns"""
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.month
        
        monthly_stats = df_monthly.groupby('month')['mood_score'].agg(['mean', 'std', 'count'])
        
        return {
            'monthly_stats': monthly_stats.to_dict(),
            'best_month': monthly_stats['mean'].idxmax(),
            'worst_month': monthly_stats['mean'].idxmin()
        }
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns"""
        df_seasonal = df.copy()
        df_seasonal['season'] = df_seasonal.index.month.map(lambda x: 
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else
            'Fall' if x in [9, 10, 11] else
            'Winter'
        )
        
        seasonal_stats = df_seasonal.groupby('season')['mood_score'].agg(['mean', 'std', 'count'])
        
        return {
            'seasonal_stats': seasonal_stats.to_dict(),
            'best_season': seasonal_stats['mean'].idxmax(),
            'worst_season': seasonal_stats['mean'].idxmin()
        }
    
    def _analyze_lagged_correlations(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze lagged correlations between variables"""
        lagged_correlations = {}
        
        variables = ['mood_score', 'sleep_hours', 'exercise_minutes', 'stress_level']
        available_vars = [var for var in variables if var in features_df.columns]
        
        for var1 in available_vars:
            for var2 in available_vars:
                if var1 != var2:
                    correlations = []
                    for lag in range(1, 8):  # Check up to 7 days lag
                        if len(features_df) > lag:
                            corr = features_df[var1].corr(features_df[var2].shift(lag))
                            if not np.isnan(corr):
                                correlations.append({
                                    'lag': lag,
                                    'correlation': corr
                                })
                    
                    if correlations:
                        lagged_correlations[f'{var1}_vs_{var2}'] = correlations
        
        return lagged_correlations
    
    def _perform_mood_clustering(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform mood-based clustering"""
        mood_features = ['mood_score', 'mood_ma_7', 'mood_std_7']
        available_features = [f for f in mood_features if f in features_df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient features for mood clustering'}
        
        X = features_df[available_features].dropna()
        
        if len(X) < 10:
            return {'error': 'Insufficient data for clustering'}
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
    
    def _perform_behavioral_clustering(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform behavioral clustering"""
        behavioral_features = ['sleep_hours', 'exercise_minutes', 'social_interactions']
        available_features = [f for f in behavioral_features if f in features_df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient features for behavioral clustering'}
        
        X = features_df[available_features].dropna()
        
        if len(X) < 10:
            return {'error': 'Insufficient data for clustering'}
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
    
    def _perform_temporal_clustering(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Perform temporal clustering"""
        temporal_features = ['day_of_week', 'hour', 'month']
        available_features = [f for f in temporal_features if f in features_df.columns]
        
        if len(available_features) < 2:
            return {'error': 'Insufficient features for temporal clustering'}
        
        X = features_df[available_features].dropna()
        
        if len(X) < 10:
            return {'error': 'Insufficient data for clustering'}
        
        kmeans = KMeans(n_clusters=4, random_state=42)
        cluster_labels = kmeans.fit_predict(X)
        
        return {
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist()
        }
    
    def _detect_daily_cycles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect daily cycles in mood"""
        if 'hour' not in df.columns:
            df = df.copy()
            df['hour'] = df.index.hour
        
        hourly_mood = df.groupby('hour')['mood_score'].mean()
        
        # Find peaks in daily cycle
        peaks, _ = find_peaks(hourly_mood.values)
        
        return {
            'hourly_pattern': hourly_mood.to_dict(),
            'peak_hours': peaks.tolist(),
            'cycle_amplitude': hourly_mood.max() - hourly_mood.min()
        }
    
    def _detect_weekly_cycles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect weekly cycles in mood"""
        df_weekly = df.copy()
        df_weekly['day_of_week'] = df_weekly.index.dayofweek
        
        weekly_mood = df_weekly.groupby('day_of_week')['mood_score'].mean()
        
        return {
            'weekly_pattern': weekly_mood.to_dict(),
            'cycle_amplitude': weekly_mood.max() - weekly_mood.min()
        }
    
    def _detect_monthly_cycles(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect monthly cycles in mood"""
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.month
        
        monthly_mood = df_monthly.groupby('month')['mood_score'].mean()
        
        return {
            'monthly_pattern': monthly_mood.to_dict(),
            'cycle_amplitude': monthly_mood.max() - monthly_mood.min()
        }
    
    def _analyze_mood_transitions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze mood state transitions"""
        # Define mood states
        mood_states = pd.cut(df['mood_score'], bins=[0, 3, 7, 10], labels=['Low', 'Medium', 'High'])
        
        # Count transitions
        transitions = {}
        for i in range(1, len(mood_states)):
            prev_state = mood_states.iloc[i-1]
            curr_state = mood_states.iloc[i]
            
            if pd.notna(prev_state) and pd.notna(curr_state):
                transition_key = f"{prev_state}_to_{curr_state}"
                transitions[transition_key] = transitions.get(transition_key, 0) + 1
        
        return {
            'transition_counts': transitions,
            'total_transitions': sum(transitions.values())
        }
    
    def _analyze_behavioral_transitions(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze behavioral state transitions"""
        # Define behavioral states based on sleep and exercise
        behavioral_states = []
        
        for _, row in df.iterrows():
            sleep_good = row['sleep_hours'] >= 7
            exercise_good = row['exercise_minutes'] >= 30
            
            if sleep_good and exercise_good:
                state = 'Healthy'
            elif sleep_good or exercise_good:
                state = 'Moderate'
            else:
                state = 'Poor'
            
            behavioral_states.append(state)
        
        # Count transitions
        transitions = {}
        for i in range(1, len(behavioral_states)):
            prev_state = behavioral_states[i-1]
            curr_state = behavioral_states[i]
            
            transition_key = f"{prev_state}_to_{curr_state}"
            transitions[transition_key] = transitions.get(transition_key, 0) + 1
        
        return {
            'transition_counts': transitions,
            'total_transitions': sum(transitions.values())
        }
    
    def _analyze_feature_importance(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze feature importance for mood prediction"""
        feature_columns = [
            'sleep_hours', 'exercise_minutes', 'stress_level', 'social_interactions',
            'mood_ma_7', 'mood_std_7', 'day_of_week', 'is_weekend'
        ]
        
        available_features = [f for f in feature_columns if f in features_df.columns]
        
        if len(available_features) < 3:
            return {'error': 'Insufficient features for importance analysis'}
        
        # Prepare data for prediction
        X = features_df[available_features].dropna()
        y = features_df.loc[X.index, 'mood_score']
        
        if len(X) < 20:
            return {'error': 'Insufficient data for importance analysis'}
        
        # Train Random Forest
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        
        # Discretize mood scores for classification
        y_discrete = pd.cut(y, bins=3, labels=['Low', 'Medium', 'High'])
        
        rf.fit(X, y_discrete)
        
        # Get feature importance
        importance_scores = dict(zip(available_features, rf.feature_importances_))
        
        return {
            'feature_importance': importance_scores,
            'top_features': sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _identify_leading_indicators(self, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Identify leading indicators for mood changes"""
        leading_indicators = {}
        
        # Analyze how variables today predict mood tomorrow
        for var in ['sleep_hours', 'exercise_minutes', 'stress_level', 'social_interactions']:
            if var in features_df.columns:
                tomorrow_mood = features_df['mood_score'].shift(-1)
                correlation = features_df[var].corr(tomorrow_mood)
                
                if abs(correlation) > 0.3:
                    leading_indicators[var] = {
                        'correlation': correlation,
                        'predictive_strength': abs(correlation)
                    }
        
        return leading_indicators
    
    def _describe_cluster(self, cluster_data: pd.DataFrame) -> Dict[str, str]:
        """Describe characteristics of a cluster"""
        characteristics = {}
        
        if 'mood_score' in cluster_data.columns:
            avg_mood = cluster_data['mood_score'].mean()
            if avg_mood >= 7:
                characteristics['mood_level'] = 'High'
            elif avg_mood >= 4:
                characteristics['mood_level'] = 'Medium'
            else:
                characteristics['mood_level'] = 'Low'
        
        return characteristics
    
    def _summarize_patterns(self, df: pd.DataFrame, features_df: pd.DataFrame) -> Dict[str, Any]:
        """Summarize all pattern recognition results"""
        summary = {
            'total_patterns_detected': 0,
            'key_insights': [],
            'strongest_patterns': [],
            'recommendations': []
        }
        
        # Count patterns from results
        if hasattr(self, 'pattern_results'):
            for category, results in self.pattern_results.items():
                if isinstance(results, dict) and 'error' not in results:
                    summary['total_patterns_detected'] += len(results)
        
        # Generate key insights
        mood_avg = df['mood_score'].mean()
        mood_std = df['mood_score'].std()
        
        summary['key_insights'] = [
            f"Average mood score: {mood_avg:.1f}/10",
            f"Mood variability: {mood_std:.1f} (lower is more stable)",
            f"Data spans {len(df)} entries over {(df.index.max() - df.index.min()).days} days"
        ]
        
        return summary
    
    def get_pattern_report(self) -> Dict[str, Any]:
        """Get comprehensive pattern recognition report"""
        if not self.pattern_results:
            return {'error': 'No pattern recognition results available. Run recognize_patterns() first.'}
        
        return {
            'summary': self.pattern_results.get('pattern_summary', {}),
            'mood_patterns': self.pattern_results.get('mood_patterns', {}),
            'behavioral_patterns': self.pattern_results.get('behavioral_patterns', {}),
            'temporal_patterns': self.pattern_results.get('temporal_patterns', {})
        } 