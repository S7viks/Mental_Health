"""
Seasonal Analyzer - Specialized seasonal pattern analysis for mood tracking data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced time series libraries
try:
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.x13 import x13_arima_analysis
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False

logger = logging.getLogger(__name__)

class SeasonalAnalyzer:
    """
    Specialized seasonal pattern analysis for mood tracking data
    """
    
    def __init__(self):
        self.seasonal_data = None
        self.seasonal_results = {}
        self.scaler = StandardScaler()
        
    def analyze_seasonal_patterns(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Comprehensive seasonal pattern analysis
        
        Args:
            data: DataFrame with mood tracking data
            
        Returns:
            Dictionary with seasonal analysis results
        """
        logger.info("Starting seasonal pattern analysis")
        
        # Prepare data
        df = data.copy()
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        
        # Create daily aggregation
        daily_mood = df.groupby(df.index.date)['mood_score'].mean()
        daily_mood.index = pd.to_datetime(daily_mood.index)
        
        results = {
            'seasonal_decomposition': self._perform_seasonal_decomposition(daily_mood),
            'weekly_patterns': self._analyze_weekly_patterns(df),
            'monthly_patterns': self._analyze_monthly_patterns(df),
            'seasonal_patterns': self._analyze_seasonal_patterns(df),
            'holiday_effects': self._analyze_holiday_effects(df),
            'seasonal_affective_patterns': self._detect_seasonal_affective_patterns(df),
            'circadian_patterns': self._analyze_circadian_patterns(df),
            'weather_correlations': self._analyze_weather_correlations(df),
            'seasonal_stability': self._assess_seasonal_stability(daily_mood),
            'seasonal_recommendations': self._generate_seasonal_recommendations(df)
        }
        
        self.seasonal_results = results
        logger.info("Seasonal pattern analysis completed")
        
        return results
    
    def _perform_seasonal_decomposition(self, series: pd.Series) -> Dict[str, Any]:
        """Perform seasonal decomposition of the time series"""
        if len(series) < 30:  # Need sufficient data for decomposition
            return {'error': 'Insufficient data for seasonal decomposition'}
        
        try:
            # Try different periods
            periods = [7, 30, 365]  # Weekly, monthly, yearly
            decomposition_results = {}
            
            for period in periods:
                if len(series) >= 2 * period:
                    if ADVANCED_AVAILABLE:
                        decomp = seasonal_decompose(series, model='additive', period=period)
                        decomposition_results[f'period_{period}'] = {
                            'trend': decomp.trend,
                            'seasonal': decomp.seasonal,
                            'residual': decomp.resid,
                            'strength_of_trend': self._calculate_strength_of_trend(decomp),
                            'strength_of_seasonality': self._calculate_strength_of_seasonality(decomp)
                        }
                    else:
                        # Simple decomposition without statsmodels
                        decomposition_results[f'period_{period}'] = self._simple_decomposition(series, period)
            
            return decomposition_results
            
        except Exception as e:
            logger.error(f"Error in seasonal decomposition: {e}")
            return {'error': str(e)}
    
    def _simple_decomposition(self, series: pd.Series, period: int) -> Dict[str, Any]:
        """Simple seasonal decomposition without statsmodels"""
        # Calculate trend using moving average
        trend = series.rolling(window=period, center=True).mean()
        
        # Calculate seasonal component
        detrended = series - trend
        seasonal_means = detrended.groupby(detrended.index.dayofyear % period).mean()
        seasonal = pd.Series(index=series.index)
        
        for i, idx in enumerate(series.index):
            seasonal.iloc[i] = seasonal_means.iloc[i % period]
        
        # Calculate residual
        residual = series - trend - seasonal
        
        return {
            'trend': trend,
            'seasonal': seasonal,
            'residual': residual,
            'strength_of_trend': trend.std() / series.std() if series.std() > 0 else 0,
            'strength_of_seasonality': seasonal.std() / series.std() if series.std() > 0 else 0
        }
    
    def _analyze_weekly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze weekly patterns in mood data"""
        # Add day of week information
        df_weekly = df.copy()
        df_weekly['day_of_week'] = df_weekly.index.dayofweek
        df_weekly['day_name'] = df_weekly.index.day_name()
        
        # Calculate statistics by day of week
        weekly_stats = df_weekly.groupby(['day_of_week', 'day_name'])['mood_score'].agg([
            'mean', 'std', 'count', 'min', 'max', 'median'
        ]).reset_index()
        
        # Test for significant weekly pattern
        daily_groups = [group['mood_score'].values for name, group in df_weekly.groupby('day_of_week')]
        f_stat, p_value = stats.f_oneway(*daily_groups)
        
        # Calculate effect size (eta squared)
        total_variance = np.var(df_weekly['mood_score'])
        between_group_variance = np.sum([(len(group) * (np.mean(group) - np.mean(df_weekly['mood_score']))**2) 
                                        for group in daily_groups]) / len(df_weekly)
        eta_squared = between_group_variance / total_variance if total_variance > 0 else 0
        
        # Detect patterns
        patterns = self._detect_weekly_patterns(weekly_stats)
        
        return {
            'weekly_statistics': weekly_stats.to_dict('records'),
            'significant_pattern': p_value < 0.05,
            'f_statistic': f_stat,
            'p_value': p_value,
            'effect_size': eta_squared,
            'patterns': patterns,
            'recommendations': self._generate_weekly_recommendations(weekly_stats)
        }
    
    def _analyze_monthly_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze monthly patterns in mood data"""
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.month
        df_monthly['month_name'] = df_monthly.index.strftime('%B')
        
        # Calculate statistics by month
        monthly_stats = df_monthly.groupby(['month', 'month_name'])['mood_score'].agg([
            'mean', 'std', 'count', 'min', 'max', 'median'
        ]).reset_index()
        
        # Test for significant monthly pattern
        monthly_groups = [group['mood_score'].values for name, group in df_monthly.groupby('month')]
        f_stat, p_value = stats.f_oneway(*monthly_groups)
        
        # Calculate effect size
        total_variance = np.var(df_monthly['mood_score'])
        between_group_variance = np.sum([(len(group) * (np.mean(group) - np.mean(df_monthly['mood_score']))**2) 
                                        for group in monthly_groups]) / len(df_monthly)
        eta_squared = between_group_variance / total_variance if total_variance > 0 else 0
        
        # Detect patterns
        patterns = self._detect_monthly_patterns(monthly_stats)
        
        return {
            'monthly_statistics': monthly_stats.to_dict('records'),
            'significant_pattern': p_value < 0.05,
            'f_statistic': f_stat,
            'p_value': p_value,
            'effect_size': eta_squared,
            'patterns': patterns,
            'recommendations': self._generate_monthly_recommendations(monthly_stats)
        }
    
    def _analyze_seasonal_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze seasonal patterns (Spring, Summer, Fall, Winter)"""
        df_seasonal = df.copy()
        df_seasonal['season'] = df_seasonal.index.month.map(lambda x: 
            'Spring' if x in [3, 4, 5] else
            'Summer' if x in [6, 7, 8] else
            'Fall' if x in [9, 10, 11] else
            'Winter'
        )
        
        # Calculate statistics by season
        seasonal_stats = df_seasonal.groupby('season')['mood_score'].agg([
            'mean', 'std', 'count', 'min', 'max', 'median'
        ]).reset_index()
        
        # Test for significant seasonal pattern
        seasonal_groups = [group['mood_score'].values for name, group in df_seasonal.groupby('season')]
        f_stat, p_value = stats.f_oneway(*seasonal_groups)
        
        # Calculate effect size
        total_variance = np.var(df_seasonal['mood_score'])
        between_group_variance = np.sum([(len(group) * (np.mean(group) - np.mean(df_seasonal['mood_score']))**2) 
                                        for group in seasonal_groups]) / len(df_seasonal)
        eta_squared = between_group_variance / total_variance if total_variance > 0 else 0
        
        # Detect seasonal affective patterns
        seasonal_patterns = self._detect_seasonal_affective_indicators(seasonal_stats)
        
        return {
            'seasonal_statistics': seasonal_stats.to_dict('records'),
            'significant_pattern': p_value < 0.05,
            'f_statistic': f_stat,
            'p_value': p_value,
            'effect_size': eta_squared,
            'seasonal_affective_indicators': seasonal_patterns,
            'recommendations': self._generate_seasonal_recommendations(seasonal_stats)
        }
    
    def _analyze_holiday_effects(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze the effect of holidays on mood"""
        # Define major holidays (simplified - US holidays)
        holidays = {
            'New Year': [(1, 1)],
            'Valentine\'s Day': [(2, 14)],
            'Easter': [(3, 31), (4, 1), (4, 2)],  # Approximate
            'Memorial Day': [(5, 25)],  # Approximate
            'Independence Day': [(7, 4)],
            'Labor Day': [(9, 1)],  # Approximate
            'Halloween': [(10, 31)],
            'Thanksgiving': [(11, 25)],  # Approximate
            'Christmas': [(12, 25)]
        }
        
        # Create holiday indicators
        df_holiday = df.copy()
        df_holiday['is_holiday'] = False
        df_holiday['holiday_name'] = ''
        
        for holiday_name, dates in holidays.items():
            for month, day in dates:
                holiday_mask = (df_holiday.index.month == month) & (df_holiday.index.day == day)
                df_holiday.loc[holiday_mask, 'is_holiday'] = True
                df_holiday.loc[holiday_mask, 'holiday_name'] = holiday_name
        
        # Analyze holiday effects
        holiday_stats = df_holiday.groupby('is_holiday')['mood_score'].agg([
            'mean', 'std', 'count'
        ])
        
        # Statistical test
        holiday_moods = df_holiday[df_holiday['is_holiday']]['mood_score']
        non_holiday_moods = df_holiday[~df_holiday['is_holiday']]['mood_score']
        
        if len(holiday_moods) > 0 and len(non_holiday_moods) > 0:
            t_stat, p_value = stats.ttest_ind(holiday_moods, non_holiday_moods)
            effect_size = (holiday_moods.mean() - non_holiday_moods.mean()) / np.sqrt(
                ((len(holiday_moods) - 1) * holiday_moods.var() + (len(non_holiday_moods) - 1) * non_holiday_moods.var()) / 
                (len(holiday_moods) + len(non_holiday_moods) - 2)
            )
        else:
            t_stat, p_value, effect_size = 0, 1, 0
        
        # Analyze individual holidays
        individual_holidays = df_holiday[df_holiday['is_holiday']].groupby('holiday_name')['mood_score'].agg([
            'mean', 'std', 'count'
        ]).reset_index()
        
        return {
            'holiday_statistics': holiday_stats.to_dict(),
            'significant_effect': p_value < 0.05,
            't_statistic': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'individual_holidays': individual_holidays.to_dict('records'),
            'recommendations': self._generate_holiday_recommendations(holiday_stats, individual_holidays)
        }
    
    def _detect_seasonal_affective_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Detect patterns consistent with Seasonal Affective Disorder"""
        # Calculate monthly averages
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.month
        monthly_avg = df_monthly.groupby('month')['mood_score'].mean()
        
        # Define winter months (Nov-Feb in Northern Hemisphere)
        winter_months = [11, 12, 1, 2]
        summer_months = [5, 6, 7, 8]
        
        # Calculate seasonal scores
        winter_score = monthly_avg[monthly_avg.index.isin(winter_months)].mean()
        summer_score = monthly_avg[monthly_avg.index.isin(summer_months)].mean()
        
        # Calculate SAD indicators
        seasonal_difference = summer_score - winter_score
        seasonal_ratio = summer_score / winter_score if winter_score > 0 else 0
        
        # Detect significant winter depression
        winter_depression = seasonal_difference > 1.0 and seasonal_ratio > 1.2
        
        # Analyze light exposure correlation (if available)
        daylight_correlation = self._analyze_daylight_correlation(df)
        
        return {
            'winter_average': winter_score,
            'summer_average': summer_score,
            'seasonal_difference': seasonal_difference,
            'seasonal_ratio': seasonal_ratio,
            'winter_depression_indicated': winter_depression,
            'daylight_correlation': daylight_correlation,
            'sad_risk_factors': self._identify_sad_risk_factors(df),
            'recommendations': self._generate_sad_recommendations(winter_depression, seasonal_difference)
        }
    
    def _analyze_circadian_patterns(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze circadian patterns in mood if timestamp includes time"""
        if 'timestamp' not in df.columns:
            return {'error': 'Timestamp data required for circadian analysis'}
        
        df_circadian = df.copy()
        df_circadian['hour'] = df_circadian.index.hour
        df_circadian['time_of_day'] = df_circadian['hour'].map(lambda x: 
            'Morning' if 6 <= x < 12 else
            'Afternoon' if 12 <= x < 18 else
            'Evening' if 18 <= x < 22 else
            'Night'
        )
        
        # Analyze by hour
        hourly_stats = df_circadian.groupby('hour')['mood_score'].agg([
            'mean', 'std', 'count'
        ])
        
        # Analyze by time of day
        time_stats = df_circadian.groupby('time_of_day')['mood_score'].agg([
            'mean', 'std', 'count'
        ])
        
        # Test for significant circadian pattern
        hourly_groups = [group['mood_score'].values for name, group in df_circadian.groupby('hour')]
        f_stat, p_value = stats.f_oneway(*hourly_groups)
        
        # Find peak and trough times
        peak_hour = hourly_stats['mean'].idxmax()
        trough_hour = hourly_stats['mean'].idxmin()
        
        return {
            'hourly_statistics': hourly_stats.to_dict(),
            'time_of_day_statistics': time_stats.to_dict(),
            'significant_pattern': p_value < 0.05,
            'f_statistic': f_stat,
            'p_value': p_value,
            'peak_hour': peak_hour,
            'trough_hour': trough_hour,
            'circadian_amplitude': hourly_stats['mean'].max() - hourly_stats['mean'].min(),
            'recommendations': self._generate_circadian_recommendations(hourly_stats, time_stats)
        }
    
    def _analyze_weather_correlations(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlations between weather and mood"""
        if 'weather' not in df.columns:
            return {'error': 'Weather data not available'}
        
        # Weather-mood correlation
        weather_stats = df.groupby('weather')['mood_score'].agg([
            'mean', 'std', 'count'
        ])
        
        # Test for significant weather effect
        weather_groups = [group['mood_score'].values for name, group in df.groupby('weather')]
        f_stat, p_value = stats.f_oneway(*weather_groups)
        
        # Calculate effect size
        total_variance = np.var(df['mood_score'])
        between_group_variance = np.sum([(len(group) * (np.mean(group) - np.mean(df['mood_score']))**2) 
                                        for group in weather_groups]) / len(df)
        eta_squared = between_group_variance / total_variance if total_variance > 0 else 0
        
        # Identify best and worst weather for mood
        best_weather = weather_stats['mean'].idxmax()
        worst_weather = weather_stats['mean'].idxmin()
        
        return {
            'weather_statistics': weather_stats.to_dict(),
            'significant_effect': p_value < 0.05,
            'f_statistic': f_stat,
            'p_value': p_value,
            'effect_size': eta_squared,
            'best_weather': best_weather,
            'worst_weather': worst_weather,
            'recommendations': self._generate_weather_recommendations(weather_stats)
        }
    
    def _assess_seasonal_stability(self, daily_series: pd.Series) -> Dict[str, Any]:
        """Assess how stable seasonal patterns are over time"""
        if len(daily_series) < 365:
            return {'error': 'Need at least one year of data for stability assessment'}
        
        # Split into years
        years = daily_series.groupby(daily_series.index.year)
        
        if len(years) < 2:
            return {'error': 'Need at least two years of data for stability assessment'}
        
        # Calculate seasonal patterns for each year
        yearly_patterns = {}
        for year, year_data in years:
            if len(year_data) >= 90:  # Need at least 3 months
                month_pattern = year_data.groupby(year_data.index.month).mean()
                yearly_patterns[year] = month_pattern
        
        # Calculate stability metrics
        if len(yearly_patterns) >= 2:
            # Calculate correlation between years
            correlations = []
            years_list = list(yearly_patterns.keys())
            
            for i in range(len(years_list)):
                for j in range(i+1, len(years_list)):
                    year1, year2 = years_list[i], years_list[j]
                    # Find common months
                    common_months = set(yearly_patterns[year1].index) & set(yearly_patterns[year2].index)
                    if len(common_months) >= 6:
                        pattern1 = yearly_patterns[year1][list(common_months)]
                        pattern2 = yearly_patterns[year2][list(common_months)]
                        corr = np.corrcoef(pattern1, pattern2)[0, 1]
                        if not np.isnan(corr):
                            correlations.append(corr)
            
            avg_correlation = np.mean(correlations) if correlations else 0
            stability_score = avg_correlation
        else:
            stability_score = 0
        
        return {
            'yearly_patterns': {year: pattern.to_dict() for year, pattern in yearly_patterns.items()},
            'stability_score': stability_score,
            'pattern_consistency': 'High' if stability_score > 0.7 else 'Medium' if stability_score > 0.4 else 'Low',
            'recommendations': self._generate_stability_recommendations(stability_score)
        }
    
    def _generate_seasonal_recommendations(self, seasonal_stats: pd.DataFrame) -> List[str]:
        """Generate recommendations based on seasonal patterns"""
        recommendations = []
        
        # Find worst season
        worst_season = seasonal_stats.loc[seasonal_stats['mean'].idxmin(), 'season']
        worst_score = seasonal_stats['mean'].min()
        
        if worst_score < 5:
            recommendations.append(f"Consider extra self-care during {worst_season} when mood tends to be lower")
            
            if worst_season == 'Winter':
                recommendations.append("Light therapy may help with winter mood challenges")
                recommendations.append("Maintain regular exercise routine during winter months")
                recommendations.append("Consider vitamin D supplementation (consult healthcare provider)")
            
            elif worst_season == 'Summer':
                recommendations.append("Stay hydrated and avoid excessive heat during summer")
                recommendations.append("Maintain regular sleep schedule despite longer daylight hours")
        
        # Find best season
        best_season = seasonal_stats.loc[seasonal_stats['mean'].idxmax(), 'season']
        recommendations.append(f"Take advantage of naturally higher mood during {best_season}")
        
        return recommendations
    
    def _generate_weekly_recommendations(self, weekly_stats: pd.DataFrame) -> List[str]:
        """Generate recommendations based on weekly patterns"""
        recommendations = []
        
        # Find worst day
        worst_day = weekly_stats.loc[weekly_stats['mean'].idxmin(), 'day_name']
        worst_score = weekly_stats['mean'].min()
        
        if worst_score < 5:
            recommendations.append(f"Plan extra self-care activities on {worst_day}s")
            
            if worst_day in ['Monday', 'Tuesday']:
                recommendations.append("Consider Sunday evening routine to prepare for the week")
            elif worst_day in ['Friday']:
                recommendations.append("End-of-week stress management may be beneficial")
        
        # Find best day
        best_day = weekly_stats.loc[weekly_stats['mean'].idxmax(), 'day_name']
        recommendations.append(f"Leverage naturally higher mood on {best_day}s for challenging tasks")
        
        return recommendations
    
    def _generate_monthly_recommendations(self, monthly_stats: pd.DataFrame) -> List[str]:
        """Generate recommendations based on monthly patterns"""
        recommendations = []
        
        # Find worst month
        worst_month = monthly_stats.loc[monthly_stats['mean'].idxmin(), 'month_name']
        worst_score = monthly_stats['mean'].min()
        
        if worst_score < 5:
            recommendations.append(f"Prepare for potential mood challenges in {worst_month}")
            
            if worst_month in ['November', 'December', 'January', 'February']:
                recommendations.append("Consider seasonal affective disorder evaluation")
                recommendations.append("Maintain social connections during darker months")
        
        return recommendations
    
    def _generate_holiday_recommendations(self, holiday_stats: pd.DataFrame, 
                                        individual_holidays: pd.DataFrame) -> List[str]:
        """Generate recommendations based on holiday patterns"""
        recommendations = []
        
        if len(holiday_stats) >= 2:
            holiday_mean = holiday_stats.loc[True, 'mean'] if True in holiday_stats.index else 0
            non_holiday_mean = holiday_stats.loc[False, 'mean'] if False in holiday_stats.index else 0
            
            if holiday_mean < non_holiday_mean:
                recommendations.append("Holidays may be challenging - plan additional support")
                recommendations.append("Consider maintaining routine during holiday periods")
            else:
                recommendations.append("Holidays appear to have positive effect on mood")
        
        # Individual holiday recommendations
        if len(individual_holidays) > 0:
            worst_holiday = individual_holidays.loc[individual_holidays['mean'].idxmin(), 'holiday_name']
            recommendations.append(f"Pay special attention to self-care around {worst_holiday}")
        
        return recommendations
    
    def _generate_sad_recommendations(self, winter_depression: bool, seasonal_difference: float) -> List[str]:
        """Generate recommendations for seasonal affective patterns"""
        recommendations = []
        
        if winter_depression:
            recommendations.append("Consider evaluation for Seasonal Affective Disorder")
            recommendations.append("Light therapy may be beneficial (consult healthcare provider)")
            recommendations.append("Maintain regular sleep schedule and exercise routine")
            recommendations.append("Consider vitamin D supplementation")
            recommendations.append("Plan social activities during winter months")
        
        if seasonal_difference > 0.5:
            recommendations.append("Significant seasonal mood variation detected")
            recommendations.append("Consider preventive measures before difficult seasons")
        
        return recommendations
    
    def _generate_circadian_recommendations(self, hourly_stats: pd.DataFrame, 
                                          time_stats: pd.DataFrame) -> List[str]:
        """Generate recommendations based on circadian patterns"""
        recommendations = []
        
        # Find peak and trough times
        peak_hour = hourly_stats['mean'].idxmax()
        trough_hour = hourly_stats['mean'].idxmin()
        
        recommendations.append(f"Your mood peaks around {peak_hour}:00 - schedule important activities then")
        recommendations.append(f"Your mood is lowest around {trough_hour}:00 - practice extra self-care")
        
        # Time of day recommendations
        if 'Morning' in time_stats.index and time_stats.loc['Morning', 'mean'] > 6:
            recommendations.append("Morning appears to be a good time for you - leverage this")
        
        if 'Night' in time_stats.index and time_stats.loc['Night', 'mean'] < 5:
            recommendations.append("Consider evening routine to improve night-time mood")
        
        return recommendations
    
    def _generate_weather_recommendations(self, weather_stats: pd.DataFrame) -> List[str]:
        """Generate recommendations based on weather patterns"""
        recommendations = []
        
        best_weather = weather_stats['mean'].idxmax()
        worst_weather = weather_stats['mean'].idxmin()
        
        recommendations.append(f"Your mood is best during {best_weather} weather")
        recommendations.append(f"Take extra care during {worst_weather} weather")
        
        if worst_weather == 'rainy':
            recommendations.append("Consider indoor activities and light therapy during rainy days")
        elif worst_weather == 'cloudy':
            recommendations.append("Bright indoor lighting may help during cloudy days")
        
        return recommendations
    
    def _generate_stability_recommendations(self, stability_score: float) -> List[str]:
        """Generate recommendations based on pattern stability"""
        recommendations = []
        
        if stability_score > 0.7:
            recommendations.append("Your seasonal patterns are stable - continue current strategies")
        elif stability_score > 0.4:
            recommendations.append("Some seasonal variation - monitor patterns annually")
        else:
            recommendations.append("Seasonal patterns vary significantly - consider tracking triggers")
            recommendations.append("Consult healthcare provider about mood stability")
        
        return recommendations
    
    # Helper methods
    def _calculate_strength_of_trend(self, decomp) -> float:
        """Calculate strength of trend component"""
        if decomp.trend is None:
            return 0
        return decomp.trend.var() / decomp.observed.var()
    
    def _calculate_strength_of_seasonality(self, decomp) -> float:
        """Calculate strength of seasonal component"""
        if decomp.seasonal is None:
            return 0
        return decomp.seasonal.var() / decomp.observed.var()
    
    def _detect_weekly_patterns(self, weekly_stats: pd.DataFrame) -> Dict[str, Any]:
        """Detect specific weekly patterns"""
        patterns = {}
        
        # Weekend effect
        weekend_days = weekly_stats[weekly_stats['day_of_week'].isin([5, 6])]  # Sat, Sun
        weekday_days = weekly_stats[~weekly_stats['day_of_week'].isin([5, 6])]
        
        if len(weekend_days) > 0 and len(weekday_days) > 0:
            weekend_mean = weekend_days['mean'].mean()
            weekday_mean = weekday_days['mean'].mean()
            patterns['weekend_effect'] = weekend_mean - weekday_mean
        
        # Monday blues
        monday_stats = weekly_stats[weekly_stats['day_of_week'] == 0]
        if len(monday_stats) > 0:
            monday_mean = monday_stats['mean'].iloc[0]
            overall_mean = weekly_stats['mean'].mean()
            patterns['monday_blues'] = monday_mean < overall_mean - 0.5
        
        return patterns
    
    def _detect_monthly_patterns(self, monthly_stats: pd.DataFrame) -> Dict[str, Any]:
        """Detect specific monthly patterns"""
        patterns = {}
        
        # Holiday months effect
        holiday_months = [11, 12, 1]  # Nov, Dec, Jan
        holiday_data = monthly_stats[monthly_stats['month'].isin(holiday_months)]
        non_holiday_data = monthly_stats[~monthly_stats['month'].isin(holiday_months)]
        
        if len(holiday_data) > 0 and len(non_holiday_data) > 0:
            holiday_mean = holiday_data['mean'].mean()
            non_holiday_mean = non_holiday_data['mean'].mean()
            patterns['holiday_season_effect'] = holiday_mean - non_holiday_mean
        
        return patterns
    
    def _detect_seasonal_affective_indicators(self, seasonal_stats: pd.DataFrame) -> Dict[str, Any]:
        """Detect indicators of seasonal affective patterns"""
        indicators = {}
        
        # Winter depression indicator
        if 'Winter' in seasonal_stats['season'].values:
            winter_mean = seasonal_stats[seasonal_stats['season'] == 'Winter']['mean'].iloc[0]
            overall_mean = seasonal_stats['mean'].mean()
            indicators['winter_depression'] = winter_mean < overall_mean - 1.0
        
        # Seasonal amplitude
        seasonal_range = seasonal_stats['mean'].max() - seasonal_stats['mean'].min()
        indicators['seasonal_amplitude'] = seasonal_range
        indicators['high_seasonal_variation'] = seasonal_range > 2.0
        
        return indicators
    
    def _analyze_daylight_correlation(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze correlation with daylight hours (simplified)"""
        # Simplified daylight calculation based on month
        daylight_hours = {
            1: 9.5, 2: 10.5, 3: 11.5, 4: 12.5, 5: 13.5, 6: 14.5,
            7: 14.5, 8: 13.5, 9: 12.5, 10: 11.5, 11: 10.5, 12: 9.5
        }
        
        df_daylight = df.copy()
        df_daylight['month'] = df_daylight.index.month
        df_daylight['daylight_hours'] = df_daylight['month'].map(daylight_hours)
        
        # Calculate correlation
        correlation = df_daylight[['mood_score', 'daylight_hours']].corr().iloc[0, 1]
        
        return {
            'correlation': correlation,
            'significant': abs(correlation) > 0.3,
            'interpretation': 'Positive correlation with daylight' if correlation > 0.3 else 
                           'Negative correlation with daylight' if correlation < -0.3 else 
                           'No significant correlation with daylight'
        }
    
    def _identify_sad_risk_factors(self, df: pd.DataFrame) -> List[str]:
        """Identify risk factors for seasonal affective disorder"""
        risk_factors = []
        
        # Winter mood drop
        df_monthly = df.copy()
        df_monthly['month'] = df_monthly.index.month
        monthly_avg = df_monthly.groupby('month')['mood_score'].mean()
        
        winter_months = [11, 12, 1, 2]
        summer_months = [5, 6, 7, 8]
        
        winter_avg = monthly_avg[monthly_avg.index.isin(winter_months)].mean()
        summer_avg = monthly_avg[monthly_avg.index.isin(summer_months)].mean()
        
        if summer_avg - winter_avg > 1.5:
            risk_factors.append("Significant winter mood decrease")
        
        # Low winter scores
        if winter_avg < 4:
            risk_factors.append("Low winter mood scores")
        
        # High seasonal variation
        seasonal_range = monthly_avg.max() - monthly_avg.min()
        if seasonal_range > 2.5:
            risk_factors.append("High seasonal mood variation")
        
        return risk_factors 