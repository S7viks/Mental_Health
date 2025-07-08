"""
Time Series Analyzer - Advanced time series analysis for mood tracking data
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Try to import advanced time series libraries
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.seasonal import seasonal_decompose
    from statsmodels.tsa.stattools import adfuller
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    ADVANCED_AVAILABLE = True
except ImportError:
    ADVANCED_AVAILABLE = False
    print("Advanced time series libraries not available. Install statsmodels for full functionality.")

logger = logging.getLogger(__name__)

class TimeSeriesAnalyzer:
    """
    Advanced time series analysis for mood tracking data
    """
    
    def __init__(self):
        self.data = None
        self.time_series = None
        self.models = {}
        self.forecasts = {}
        self.analysis_results = {}
        
    def load_data(self, data: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        Load and prepare mood data for time series analysis
        
        Args:
            data: List of mood entries
            
        Returns:
            DataFrame ready for time series analysis
        """
        logger.info("Loading data for time series analysis")
        
        # Convert to DataFrame
        df = pd.DataFrame(data)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Set timestamp as index
        df.set_index('timestamp', inplace=True)
        
        # Sort by timestamp
        df.sort_index(inplace=True)
        
        # Store original data
        self.data = df
        
        logger.info(f"Loaded {len(df)} entries spanning {df.index.min()} to {df.index.max()}")
        
        return df
    
    def create_daily_aggregation(self, target_column: str = 'mood_score', 
                               agg_method: str = 'mean') -> pd.Series:
        """
        Create daily aggregated time series
        
        Args:
            target_column: Column to analyze
            agg_method: Aggregation method ('mean', 'median', 'min', 'max')
            
        Returns:
            Daily aggregated time series
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_data() first.")
        
        # Group by date and aggregate
        if agg_method == 'mean':
            daily_series = self.data.groupby(self.data.index.date)[target_column].mean()
        elif agg_method == 'median':
            daily_series = self.data.groupby(self.data.index.date)[target_column].median()
        elif agg_method == 'min':
            daily_series = self.data.groupby(self.data.index.date)[target_column].min()
        elif agg_method == 'max':
            daily_series = self.data.groupby(self.data.index.date)[target_column].max()
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")
        
        # Convert index back to datetime
        daily_series.index = pd.to_datetime(daily_series.index)
        
        # Fill missing dates with interpolation
        full_date_range = pd.date_range(start=daily_series.index.min(), 
                                       end=daily_series.index.max(), 
                                       freq='D')
        daily_series = daily_series.reindex(full_date_range)
        daily_series = daily_series.interpolate(method='linear')
        
        self.time_series = daily_series
        
        logger.info(f"Created daily {agg_method} time series with {len(daily_series)} points")
        
        return daily_series
    
    def check_stationarity(self, series: pd.Series = None) -> Dict[str, Any]:
        """
        Check if time series is stationary using Augmented Dickey-Fuller test
        
        Args:
            series: Time series to test (uses self.time_series if None)
            
        Returns:
            Dictionary with stationarity test results
        """
        if series is None:
            series = self.time_series
        
        if series is None:
            raise ValueError("No time series available. Create one first.")
        
        if not ADVANCED_AVAILABLE:
            logger.warning("Advanced statistical tests not available")
            return {'stationary': False, 'method': 'basic_check'}
        
        # Perform Augmented Dickey-Fuller test
        result = adfuller(series.dropna())
        
        stationarity_result = {
            'adf_statistic': result[0],
            'p_value': result[1],
            'critical_values': result[4],
            'stationary': result[1] < 0.05,
            'interpretation': 'Stationary' if result[1] < 0.05 else 'Non-stationary'
        }
        
        logger.info(f"Stationarity test: {stationarity_result['interpretation']} (p-value: {result[1]:.4f})")
        
        return stationarity_result
    
    def make_stationary(self, series: pd.Series = None, method: str = 'diff') -> pd.Series:
        """
        Make time series stationary
        
        Args:
            series: Time series to make stationary
            method: Method to use ('diff', 'log_diff', 'detrend')
            
        Returns:
            Stationary time series
        """
        if series is None:
            series = self.time_series
        
        if series is None:
            raise ValueError("No time series available")
        
        if method == 'diff':
            stationary_series = series.diff().dropna()
        elif method == 'log_diff':
            # Add small constant to avoid log(0)
            log_series = np.log(series + 1e-8)
            stationary_series = log_series.diff().dropna()
        elif method == 'detrend':
            # Remove linear trend
            trend = np.polyfit(range(len(series)), series, 1)
            trend_line = np.polyval(trend, range(len(series)))
            stationary_series = series - trend_line
        else:
            raise ValueError(f"Unknown method: {method}")
        
        logger.info(f"Made series stationary using {method} method")
        
        return stationary_series
    
    def seasonal_decomposition(self, series: pd.Series = None, 
                             model: str = 'additive', 
                             period: int = 7) -> Dict[str, pd.Series]:
        """
        Perform seasonal decomposition
        
        Args:
            series: Time series to decompose
            model: 'additive' or 'multiplicative'
            period: Seasonal period (7 for weekly pattern)
            
        Returns:
            Dictionary with trend, seasonal, and residual components
        """
        if series is None:
            series = self.time_series
        
        if series is None:
            raise ValueError("No time series available")
        
        if not ADVANCED_AVAILABLE:
            logger.warning("Advanced decomposition not available. Using basic trend calculation.")
            # Basic trend calculation using rolling mean
            trend = series.rolling(window=period, center=True).mean()
            seasonal = series - trend
            residual = series - trend - seasonal
            
            return {
                'trend': trend,
                'seasonal': seasonal,
                'residual': residual,
                'original': series
            }
        
        # Ensure we have enough data points
        if len(series) < 2 * period:
            logger.warning(f"Not enough data for seasonal decomposition. Need at least {2 * period} points, have {len(series)}")
            # Return basic components
            trend = series.rolling(window=min(period, len(series)//2), center=True).mean()
            return {
                'trend': trend,
                'seasonal': pd.Series(index=series.index, data=0),
                'residual': series - trend,
                'original': series
            }
        
        try:
            decomposition = seasonal_decompose(series, model=model, period=period)
            
            result = {
                'trend': decomposition.trend,
                'seasonal': decomposition.seasonal,
                'residual': decomposition.resid,
                'original': series
            }
            
            logger.info(f"Performed {model} seasonal decomposition with period {period}")
            
            return result
        
        except Exception as e:
            logger.warning(f"Seasonal decomposition failed: {e}. Using basic trend calculation.")
            # Fallback to basic trend
            trend = series.rolling(window=period, center=True).mean()
            return {
                'trend': trend,
                'seasonal': pd.Series(index=series.index, data=0),
                'residual': series - trend,
                'original': series
            }
    
    def fit_arima_model(self, series: pd.Series = None, 
                       order: Tuple[int, int, int] = None,
                       auto_order: bool = True) -> Dict[str, Any]:
        """
        Fit ARIMA model to time series
        
        Args:
            series: Time series to model
            order: ARIMA order (p, d, q)
            auto_order: Automatically determine best order
            
        Returns:
            Dictionary with model results
        """
        if series is None:
            series = self.time_series
        
        if series is None:
            raise ValueError("No time series available")
        
        series = series.dropna()
        
        if not ADVANCED_AVAILABLE:
            logger.warning("ARIMA modeling not available. Using simple linear trend.")
            # Simple linear trend as fallback
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, 1)
            trend_line = np.polyval(coeffs, x)
            
            return {
                'model_type': 'linear_trend',
                'slope': coeffs[0],
                'intercept': coeffs[1],
                'fitted_values': pd.Series(trend_line, index=series.index),
                'residuals': series - trend_line,
                'aic': None,
                'forecast_function': lambda steps: coeffs[1] + coeffs[0] * (len(series) + np.arange(1, steps + 1))
            }
        
        try:
            if auto_order:
                # Simple order selection based on series length
                if len(series) < 20:
                    order = (1, 1, 1)
                elif len(series) < 50:
                    order = (2, 1, 2)
                else:
                    order = (3, 1, 3)
            
            if order is None:
                order = (1, 1, 1)  # Default order
            
            # Fit ARIMA model
            model = ARIMA(series, order=order)
            fitted_model = model.fit()
            
            result = {
                'model': fitted_model,
                'order': order,
                'aic': fitted_model.aic,
                'bic': fitted_model.bic,
                'fitted_values': fitted_model.fittedvalues,
                'residuals': fitted_model.resid,
                'summary': str(fitted_model.summary()),
                'forecast_function': lambda steps: fitted_model.forecast(steps=steps)
            }
            
            # Store model
            self.models['arima'] = result
            
            logger.info(f"Fitted ARIMA{order} model with AIC: {fitted_model.aic:.2f}")
            
            return result
        
        except Exception as e:
            logger.error(f"ARIMA fitting failed: {e}")
            # Fallback to linear trend
            x = np.arange(len(series))
            coeffs = np.polyfit(x, series.values, 1)
            trend_line = np.polyval(coeffs, x)
            
            return {
                'model_type': 'linear_trend_fallback',
                'error': str(e),
                'slope': coeffs[0],
                'intercept': coeffs[1],
                'fitted_values': pd.Series(trend_line, index=series.index),
                'residuals': series - trend_line,
                'aic': None,
                'forecast_function': lambda steps: coeffs[1] + coeffs[0] * (len(series) + np.arange(1, steps + 1))
            }
    
    def calculate_moving_averages(self, series: pd.Series = None, 
                                windows: List[int] = [7, 14, 30]) -> Dict[str, pd.Series]:
        """
        Calculate moving averages for different windows
        
        Args:
            series: Time series
            windows: List of window sizes
            
        Returns:
            Dictionary of moving averages
        """
        if series is None:
            series = self.time_series
        
        if series is None:
            raise ValueError("No time series available")
        
        moving_averages = {}
        
        for window in windows:
            if len(series) >= window:
                ma = series.rolling(window=window, center=True).mean()
                moving_averages[f'MA_{window}'] = ma
                logger.info(f"Calculated {window}-period moving average")
            else:
                logger.warning(f"Not enough data for {window}-period moving average")
        
        return moving_averages
    
    def detect_trends(self, series: pd.Series = None, 
                     window: int = 7) -> Dict[str, Any]:
        """
        Detect trends in time series
        
        Args:
            series: Time series
            window: Window for trend calculation
            
        Returns:
            Dictionary with trend information
        """
        if series is None:
            series = self.time_series
        
        if series is None:
            raise ValueError("No time series available")
        
        # Calculate trend using linear regression on rolling windows
        trends = []
        trend_strengths = []
        
        for i in range(window, len(series)):
            # Get window data
            window_data = series.iloc[i-window:i]
            x = np.arange(len(window_data))
            
            # Fit linear trend
            slope, intercept = np.polyfit(x, window_data.values, 1)
            
            # Calculate R-squared as trend strength
            fitted = slope * x + intercept
            ss_res = np.sum((window_data.values - fitted) ** 2)
            ss_tot = np.sum((window_data.values - np.mean(window_data.values)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            
            trends.append(slope)
            trend_strengths.append(r_squared)
        
        # Create series with trends
        trend_index = series.index[window:]
        trend_series = pd.Series(trends, index=trend_index)
        strength_series = pd.Series(trend_strengths, index=trend_index)
        
        # Classify trends
        trend_classifications = []
        for slope in trends:
            if slope > 0.1:
                trend_classifications.append('Strong Upward')
            elif slope > 0.05:
                trend_classifications.append('Moderate Upward')
            elif slope > 0:
                trend_classifications.append('Weak Upward')
            elif slope > -0.05:
                trend_classifications.append('Stable')
            elif slope > -0.1:
                trend_classifications.append('Weak Downward')
            elif slope > -0.2:
                trend_classifications.append('Moderate Downward')
            else:
                trend_classifications.append('Strong Downward')
        
        result = {
            'trend_slopes': trend_series,
            'trend_strengths': strength_series,
            'trend_classifications': trend_classifications,
            'current_trend': trend_classifications[-1] if trend_classifications else 'Unknown',
            'average_slope': np.mean(trends),
            'trend_volatility': np.std(trends)
        }
        
        logger.info(f"Detected trends: Current trend is {result['current_trend']}")
        
        return result
    
    def forecast(self, steps: int = 7, model_type: str = 'arima') -> Dict[str, Any]:
        """
        Generate forecasts
        
        Args:
            steps: Number of steps to forecast
            model_type: Type of model to use for forecasting
            
        Returns:
            Dictionary with forecast results
        """
        if model_type == 'arima' and 'arima' in self.models:
            model_result = self.models['arima']
            
            if 'forecast_function' in model_result:
                try:
                    forecast_values = model_result['forecast_function'](steps)
                    
                    # Create forecast index
                    last_date = self.time_series.index[-1]
                    forecast_index = pd.date_range(start=last_date + timedelta(days=1), 
                                                 periods=steps, freq='D')
                    
                    forecast_series = pd.Series(forecast_values, index=forecast_index)
                    
                    result = {
                        'forecast': forecast_series,
                        'method': 'ARIMA',
                        'steps': steps,
                        'confidence_level': 0.95
                    }
                    
                    logger.info(f"Generated {steps}-step ARIMA forecast")
                    
                    return result
                
                except Exception as e:
                    logger.error(f"ARIMA forecasting failed: {e}")
        
        # Fallback to simple trend extrapolation
        if self.time_series is None:
            raise ValueError("No time series available for forecasting")
        
        # Simple linear extrapolation
        recent_data = self.time_series.tail(14)  # Use last 14 days
        x = np.arange(len(recent_data))
        slope, intercept = np.polyfit(x, recent_data.values, 1)
        
        # Generate forecast
        forecast_values = []
        for i in range(1, steps + 1):
            forecast_value = slope * (len(recent_data) + i - 1) + intercept
            forecast_values.append(forecast_value)
        
        # Create forecast index
        last_date = self.time_series.index[-1]
        forecast_index = pd.date_range(start=last_date + timedelta(days=1), 
                                     periods=steps, freq='D')
        
        forecast_series = pd.Series(forecast_values, index=forecast_index)
        
        result = {
            'forecast': forecast_series,
            'method': 'Linear Trend',
            'steps': steps,
            'slope': slope,
            'confidence_level': None
        }
        
        logger.info(f"Generated {steps}-step linear trend forecast")
        
        return result
    
    def analyze(self, data: List[Dict[str, Any]], 
               target_column: str = 'mood_score') -> Dict[str, Any]:
        """
        Perform comprehensive time series analysis
        
        Args:
            data: Mood tracking data
            target_column: Column to analyze
            
        Returns:
            Complete analysis results
        """
        logger.info("Starting comprehensive time series analysis")
        
        # Load data
        df = self.load_data(data)
        
        # Create daily time series
        daily_series = self.create_daily_aggregation(target_column)
        
        # Basic statistics
        basic_stats = {
            'mean': daily_series.mean(),
            'std': daily_series.std(),
            'min': daily_series.min(),
            'max': daily_series.max(),
            'count': len(daily_series),
            'missing_days': daily_series.isna().sum()
        }
        
        # Check stationarity
        stationarity = self.check_stationarity(daily_series)
        
        # Seasonal decomposition
        decomposition = self.seasonal_decomposition(daily_series)
        
        # Fit ARIMA model
        arima_result = self.fit_arima_model(daily_series)
        
        # Calculate moving averages
        moving_averages = self.calculate_moving_averages(daily_series)
        
        # Detect trends
        trend_analysis = self.detect_trends(daily_series)
        
        # Generate forecast
        forecast_result = self.forecast(steps=7)
        
        # Compile results
        analysis_results = {
            'basic_statistics': basic_stats,
            'stationarity': stationarity,
            'seasonal_decomposition': decomposition,
            'arima_model': arima_result,
            'moving_averages': moving_averages,
            'trend_analysis': trend_analysis,
            'forecast': forecast_result,
            'time_series': daily_series,
            'analysis_timestamp': datetime.now().isoformat()
        }
        
        # Store results
        self.analysis_results = analysis_results
        
        logger.info("Time series analysis complete")
        
        return analysis_results 