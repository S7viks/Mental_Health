"""
Trend Detector - Advanced trend detection and analysis for mood tracking data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class TrendDetector:
    """
    Advanced trend detection for mood tracking data
    """
    
    def __init__(self):
        self.trend_data = None
        self.trend_results = {}
        self.scaler = StandardScaler()
        
    def detect_trends(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect various types of trends in mood data
        
        Args:
            data: DataFrame with mood tracking data
            
        Returns:
            Dictionary with trend detection results
        """
        logger.info("Starting trend detection analysis")
        
        # Prepare data
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Create daily aggregation
        daily_mood = df.groupby(df.index.date)['mood_score'].mean()
        daily_mood.index = pd.to_datetime(daily_mood.index)
        
        results = {
            'overall_trend': self._detect_overall_trend(daily_mood),
            'trend_periods': self._detect_trend_periods(daily_mood),
            'trend_strength': self._calculate_trend_strength(daily_mood),
            'trend_volatility': self._calculate_trend_volatility(daily_mood),
            'trend_acceleration': self._detect_trend_acceleration(daily_mood),
            'cyclical_patterns': self._detect_cyclical_patterns(daily_mood),
            'regime_changes': self._detect_regime_changes(daily_mood),
            'trend_quality': self._assess_trend_quality(daily_mood)
        }
        
        self.trend_results = results
        logger.info("Trend detection analysis completed")
        
        return results
    
    def _detect_overall_trend(self, series: pd.Series) -> Dict[str, Any]:
        """Detect overall trend direction and strength"""
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        
        # Determine trend direction
        if p_value < 0.05:  # Significant trend
            if slope > 0:
                direction = 'increasing'
            else:
                direction = 'decreasing'
        else:
            direction = 'stable'
        
        # Calculate trend strength
        strength = abs(r_value)
        
        return {
            'direction': direction,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value,
            'strength': strength,
            'confidence': 1 - p_value,
            'interpretation': self._interpret_trend(slope, r_value, p_value)
        }
    
    def _detect_trend_periods(self, series: pd.Series, window: int = 14) -> List[Dict[str, Any]]:
        """Detect different trend periods using rolling regression"""
        periods = []
        
        for i in range(window, len(series)):
            window_data = series.iloc[i-window:i]
            x = np.arange(len(window_data))
            
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, window_data.values)
            
            # Determine trend type
            if p_value < 0.05:
                if slope > 0:
                    trend_type = 'increasing'
                else:
                    trend_type = 'decreasing'
            else:
                trend_type = 'stable'
            
            periods.append({
                'start_date': window_data.index[0],
                'end_date': window_data.index[-1],
                'trend_type': trend_type,
                'slope': slope,
                'r_squared': r_value**2,
                'p_value': p_value,
                'strength': abs(r_value)
            })
        
        return periods
    
    def _calculate_trend_strength(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate various measures of trend strength"""
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        
        # Calculate additional strength metrics
        detrended = series.values - (slope * x + intercept)
        trend_to_noise = abs(slope * len(series)) / np.std(detrended)
        
        # Calculate directional strength
        differences = np.diff(series.values)
        positive_changes = np.sum(differences > 0)
        negative_changes = np.sum(differences < 0)
        directional_strength = abs(positive_changes - negative_changes) / len(differences)
        
        return {
            'linear_strength': abs(r_value),
            'trend_to_noise_ratio': trend_to_noise,
            'directional_strength': directional_strength,
            'consistency': self._calculate_trend_consistency(series),
            'persistence': self._calculate_trend_persistence(series)
        }
    
    def _calculate_trend_volatility(self, series: pd.Series) -> Dict[str, Any]:
        """Calculate trend volatility metrics"""
        # Calculate returns
        returns = series.pct_change().dropna()
        
        # Calculate volatility
        volatility = returns.std()
        
        # Calculate rolling volatility
        rolling_vol = returns.rolling(window=7).std()
        
        # Calculate volatility of volatility
        vol_of_vol = rolling_vol.std()
        
        return {
            'overall_volatility': volatility,
            'rolling_volatility': rolling_vol,
            'volatility_of_volatility': vol_of_vol,
            'max_volatility': rolling_vol.max(),
            'min_volatility': rolling_vol.min(),
            'volatility_trend': self._detect_volatility_trend(rolling_vol)
        }
    
    def _detect_trend_acceleration(self, series: pd.Series) -> Dict[str, Any]:
        """Detect trend acceleration or deceleration"""
        # Calculate first and second derivatives
        first_diff = np.diff(series.values)
        second_diff = np.diff(first_diff)
        
        # Calculate acceleration metrics
        acceleration = np.mean(second_diff)
        acceleration_volatility = np.std(second_diff)
        
        # Detect acceleration periods
        acceleration_periods = []
        for i in range(1, len(second_diff)):
            if abs(second_diff[i]) > 2 * acceleration_volatility:
                acceleration_periods.append({
                    'date': series.index[i+1],
                    'acceleration': second_diff[i],
                    'type': 'acceleration' if second_diff[i] > 0 else 'deceleration'
                })
        
        return {
            'overall_acceleration': acceleration,
            'acceleration_volatility': acceleration_volatility,
            'acceleration_periods': acceleration_periods,
            'trend_momentum': self._calculate_trend_momentum(series)
        }
    
    def _detect_cyclical_patterns(self, series: pd.Series) -> Dict[str, Any]:
        """Detect cyclical patterns in the data"""
        # Weekly patterns
        weekly_pattern = self._detect_weekly_pattern(series)
        
        # Monthly patterns
        monthly_pattern = self._detect_monthly_pattern(series)
        
        # Seasonal patterns
        seasonal_pattern = self._detect_seasonal_pattern(series)
        
        return {
            'weekly_pattern': weekly_pattern,
            'monthly_pattern': monthly_pattern,
            'seasonal_pattern': seasonal_pattern,
            'dominant_cycle': self._find_dominant_cycle(series)
        }
    
    def _detect_regime_changes(self, series: pd.Series) -> Dict[str, Any]:
        """Detect regime changes in the mood data"""
        # Use clustering to identify different regimes
        values = series.values.reshape(-1, 1)
        scaled_values = self.scaler.fit_transform(values)
        
        # Find optimal number of clusters
        max_clusters = min(8, len(series) // 10)
        silhouette_scores = []
        
        for k in range(2, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(scaled_values)
            silhouette_avg = silhouette_score(scaled_values, cluster_labels)
            silhouette_scores.append(silhouette_avg)
        
        optimal_k = np.argmax(silhouette_scores) + 2
        
        # Perform final clustering
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        cluster_labels = kmeans.fit_predict(scaled_values)
        
        # Identify regime changes
        regime_changes = []
        for i in range(1, len(cluster_labels)):
            if cluster_labels[i] != cluster_labels[i-1]:
                regime_changes.append({
                    'date': series.index[i],
                    'from_regime': cluster_labels[i-1],
                    'to_regime': cluster_labels[i],
                    'change_magnitude': abs(values[i] - values[i-1])
                })
        
        return {
            'num_regimes': optimal_k,
            'regime_labels': cluster_labels,
            'regime_changes': regime_changes,
            'regime_stability': self._calculate_regime_stability(cluster_labels)
        }
    
    def _assess_trend_quality(self, series: pd.Series) -> Dict[str, Any]:
        """Assess the quality of trend detection"""
        # Calculate various quality metrics
        x = np.arange(len(series))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, series.values)
        
        # Calculate residuals
        predicted = slope * x + intercept
        residuals = series.values - predicted
        
        # Quality metrics
        quality_score = r_value**2 * (1 - p_value)
        signal_to_noise = np.var(predicted) / np.var(residuals) if np.var(residuals) > 0 else float('inf')
        
        return {
            'quality_score': quality_score,
            'signal_to_noise_ratio': signal_to_noise,
            'residual_autocorrelation': self._calculate_residual_autocorrelation(residuals),
            'trend_reliability': self._calculate_trend_reliability(series),
            'confidence_intervals': self._calculate_confidence_intervals(series, slope, std_err)
        }
    
    def _interpret_trend(self, slope: float, r_value: float, p_value: float) -> str:
        """Interpret trend results in human-readable format"""
        if p_value >= 0.05:
            return "No significant trend detected"
        
        strength = abs(r_value)
        if strength > 0.7:
            strength_desc = "strong"
        elif strength > 0.4:
            strength_desc = "moderate"
        else:
            strength_desc = "weak"
        
        direction = "upward" if slope > 0 else "downward"
        
        return f"Significant {strength_desc} {direction} trend (r² = {r_value**2:.3f})"
    
    def _calculate_trend_consistency(self, series: pd.Series) -> float:
        """Calculate how consistent the trend is"""
        differences = np.diff(series.values)
        same_direction = np.sum(differences[:-1] * differences[1:] > 0)
        return same_direction / (len(differences) - 1) if len(differences) > 1 else 0
    
    def _calculate_trend_persistence(self, series: pd.Series) -> float:
        """Calculate trend persistence (Hurst exponent approximation)"""
        n = len(series)
        if n < 10:
            return 0.5
        
        # Calculate variance of differences at different lags
        lags = [1, 2, 4, 8, 16]
        variances = []
        
        for lag in lags:
            if lag < n:
                diffs = np.diff(series.values, n=lag)
                variances.append(np.var(diffs))
        
        if len(variances) < 2:
            return 0.5
        
        # Estimate Hurst exponent
        log_lags = np.log(lags[:len(variances)])
        log_vars = np.log(variances)
        
        slope, _ = np.polyfit(log_lags, log_vars, 1)
        hurst = slope / 2
        
        return min(1.0, max(0.0, hurst))
    
    def _detect_volatility_trend(self, rolling_vol: pd.Series) -> Dict[str, Any]:
        """Detect trend in volatility"""
        x = np.arange(len(rolling_vol))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, rolling_vol.values)
        
        if p_value < 0.05:
            direction = 'increasing' if slope > 0 else 'decreasing'
        else:
            direction = 'stable'
        
        return {
            'direction': direction,
            'slope': slope,
            'r_squared': r_value**2,
            'p_value': p_value
        }
    
    def _calculate_trend_momentum(self, series: pd.Series) -> float:
        """Calculate trend momentum"""
        recent_period = series.iloc[-14:] if len(series) >= 14 else series
        overall_period = series
        
        # Calculate slopes for recent and overall periods
        x_recent = np.arange(len(recent_period))
        x_overall = np.arange(len(overall_period))
        
        slope_recent, *_ = stats.linregress(x_recent, recent_period.values)
        slope_overall, *_ = stats.linregress(x_overall, overall_period.values)
        
        # Momentum is the ratio of recent to overall slope
        momentum = slope_recent / slope_overall if slope_overall != 0 else 0
        
        return momentum
    
    def _detect_weekly_pattern(self, series: pd.Series) -> Dict[str, Any]:
        """Detect weekly patterns in mood data"""
        df = pd.DataFrame({'mood': series.values}, index=series.index)
        df['day_of_week'] = df.index.dayofweek
        
        weekly_stats = df.groupby('day_of_week')['mood'].agg(['mean', 'std', 'count'])
        
        # Test for significant weekly pattern
        f_stat, p_value = stats.f_oneway(*[group['mood'].values for name, group in df.groupby('day_of_week')])
        
        return {
            'weekly_stats': weekly_stats.to_dict(),
            'significant_pattern': p_value < 0.05,
            'f_statistic': f_stat,
            'p_value': p_value,
            'best_day': weekly_stats['mean'].idxmax(),
            'worst_day': weekly_stats['mean'].idxmin()
        }
    
    def _detect_monthly_pattern(self, series: pd.Series) -> Dict[str, Any]:
        """Detect monthly patterns in mood data"""
        df = pd.DataFrame({'mood': series.values}, index=series.index)
        df['month'] = df.index.month
        
        monthly_stats = df.groupby('month')['mood'].agg(['mean', 'std', 'count'])
        
        # Test for significant monthly pattern
        f_stat, p_value = stats.f_oneway(*[group['mood'].values for name, group in df.groupby('month')])
        
        return {
            'monthly_stats': monthly_stats.to_dict(),
            'significant_pattern': p_value < 0.05,
            'f_statistic': f_stat,
            'p_value': p_value,
            'best_month': monthly_stats['mean'].idxmax(),
            'worst_month': monthly_stats['mean'].idxmin()
        }
    
    def _detect_seasonal_pattern(self, series: pd.Series) -> Dict[str, Any]:
        """Detect seasonal patterns"""
        df = pd.DataFrame({'mood': series.values}, index=series.index)
        df['season'] = df.index.month.map(lambda x: 
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else
            'Fall' if x in [9, 10, 11] else
            'Winter'
        )
        
        seasonal_stats = df.groupby('season')['mood'].agg(['mean', 'std', 'count'])
        
        # Test for significant seasonal pattern
        f_stat, p_value = stats.f_oneway(*[group['mood'].values for name, group in df.groupby('season')])
        
        return {
            'seasonal_stats': seasonal_stats.to_dict(),
            'significant_pattern': p_value < 0.05,
            'f_statistic': f_stat,
            'p_value': p_value,
            'best_season': seasonal_stats['mean'].idxmax(),
            'worst_season': seasonal_stats['mean'].idxmin()
        }
    
    def _find_dominant_cycle(self, series: pd.Series) -> Dict[str, Any]:
        """Find the dominant cycle in the data using FFT"""
        # Apply FFT to find dominant frequencies
        fft_values = np.fft.fft(series.values)
        frequencies = np.fft.fftfreq(len(series))
        
        # Find dominant frequency (excluding DC component)
        power_spectrum = np.abs(fft_values[1:len(fft_values)//2])
        dominant_freq_idx = np.argmax(power_spectrum)
        dominant_frequency = frequencies[dominant_freq_idx + 1]
        
        # Convert to period (in days)
        if dominant_frequency != 0:
            dominant_period = 1 / abs(dominant_frequency)
        else:
            dominant_period = len(series)
        
        return {
            'dominant_frequency': dominant_frequency,
            'dominant_period_days': dominant_period,
            'cycle_strength': power_spectrum[dominant_freq_idx] / np.sum(power_spectrum)
        }
    
    def _calculate_regime_stability(self, cluster_labels: np.ndarray) -> float:
        """Calculate how stable the regimes are"""
        changes = np.sum(np.diff(cluster_labels) != 0)
        return 1 - (changes / len(cluster_labels))
    
    def _calculate_residual_autocorrelation(self, residuals: np.ndarray) -> float:
        """Calculate autocorrelation of residuals"""
        if len(residuals) < 2:
            return 0
        
        # Calculate lag-1 autocorrelation
        autocorr = np.corrcoef(residuals[:-1], residuals[1:])[0, 1]
        return autocorr if not np.isnan(autocorr) else 0
    
    def _calculate_trend_reliability(self, series: pd.Series) -> float:
        """Calculate trend reliability based on consistency"""
        # Split series into two halves and compare trends
        mid = len(series) // 2
        first_half = series.iloc[:mid]
        second_half = series.iloc[mid:]
        
        if len(first_half) < 3 or len(second_half) < 3:
            return 0
        
        # Calculate trends for both halves
        x1 = np.arange(len(first_half))
        x2 = np.arange(len(second_half))
        
        slope1, *_ = stats.linregress(x1, first_half.values)
        slope2, *_ = stats.linregress(x2, second_half.values)
        
        # Reliability is based on similarity of slopes
        if slope1 == 0 and slope2 == 0:
            return 1.0
        
        if slope1 == 0 or slope2 == 0:
            return 0.0
        
        # Normalize by magnitude
        reliability = 1 - abs(slope1 - slope2) / (abs(slope1) + abs(slope2))
        return max(0, reliability)
    
    def _calculate_confidence_intervals(self, series: pd.Series, slope: float, std_err: float) -> Dict[str, float]:
        """Calculate confidence intervals for trend"""
        # 95% confidence interval
        t_critical = 1.96  # Approximate for large samples
        
        lower_bound = slope - t_critical * std_err
        upper_bound = slope + t_critical * std_err
        
        return {
            'lower_bound': lower_bound,
            'upper_bound': upper_bound,
            'margin_of_error': t_critical * std_err
        } 