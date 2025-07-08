"""
Trend Visualizer - Create visualizations for time series analysis
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import logging
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

logger = logging.getLogger(__name__)

class TrendVisualizer:
    """
    Create visualizations for time series analysis results
    """
    
    def __init__(self, save_path: str = "visualizations/trend_analysis"):
        self.save_path = Path(save_path)
        self.save_path.mkdir(parents=True, exist_ok=True)
        
        # Set up figure parameters
        self.figsize = (12, 8)
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72',
            'accent': '#F18F01',
            'success': '#C73E1D',
            'warning': '#F18F01',
            'trend_up': '#2E8B57',
            'trend_down': '#CD5C5C',
            'neutral': '#708090'
        }
        
    def plot_time_series(self, series: pd.Series, 
                        title: str = "Time Series Analysis",
                        save_filename: str = "time_series.png") -> str:
        """
        Plot basic time series with trend line
        
        Args:
            series: Time series data
            title: Plot title
            save_filename: Filename to save plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot time series
        ax.plot(series.index, series.values, 
                color=self.colors['primary'], linewidth=2, label='Mood Score')
        
        # Add trend line
        x_numeric = np.arange(len(series))
        slope, intercept = np.polyfit(x_numeric, series.values, 1)
        trend_line = slope * x_numeric + intercept
        
        ax.plot(series.index, trend_line, 
                color=self.colors['secondary'], linestyle='--', 
                linewidth=2, label=f'Trend (slope: {slope:.3f})')
        
        # Add moving average
        if len(series) >= 7:
            ma_7 = series.rolling(window=7, center=True).mean()
            ax.plot(ma_7.index, ma_7.values, 
                    color=self.colors['accent'], linewidth=2, 
                    alpha=0.7, label='7-day Moving Average')
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Mood Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_path / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Time series plot saved to {save_path}")
        return str(save_path)
    
    def plot_seasonal_decomposition(self, decomposition: Dict[str, pd.Series],
                                  save_filename: str = "seasonal_decomposition.png") -> str:
        """
        Plot seasonal decomposition components
        
        Args:
            decomposition: Dictionary with trend, seasonal, residual components
            save_filename: Filename to save plot
            
        Returns:
            Path to saved plot
        """
        fig, axes = plt.subplots(4, 1, figsize=(12, 12))
        
        # Original
        axes[0].plot(decomposition['original'].index, decomposition['original'].values, 
                     color=self.colors['primary'], linewidth=2)
        axes[0].set_title('Original Time Series', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Trend
        if 'trend' in decomposition and decomposition['trend'] is not None:
            axes[1].plot(decomposition['trend'].index, decomposition['trend'].values, 
                         color=self.colors['secondary'], linewidth=2)
        axes[1].set_title('Trend Component', fontsize=12, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        # Seasonal
        if 'seasonal' in decomposition and decomposition['seasonal'] is not None:
            axes[2].plot(decomposition['seasonal'].index, decomposition['seasonal'].values, 
                         color=self.colors['accent'], linewidth=2)
        axes[2].set_title('Seasonal Component', fontsize=12, fontweight='bold')
        axes[2].grid(True, alpha=0.3)
        
        # Residual
        if 'residual' in decomposition and decomposition['residual'] is not None:
            axes[3].plot(decomposition['residual'].index, decomposition['residual'].values, 
                         color=self.colors['neutral'], linewidth=2)
        axes[3].set_title('Residual Component', fontsize=12, fontweight='bold')
        axes[3].grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in axes:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_path / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Seasonal decomposition plot saved to {save_path}")
        return str(save_path)
    
    def plot_trend_analysis(self, trend_data: Dict[str, Any],
                           save_filename: str = "trend_analysis.png") -> str:
        """
        Plot trend analysis results
        
        Args:
            trend_data: Dictionary with trend analysis results
            save_filename: Filename to save plot
            
        Returns:
            Path to saved plot
        """
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot trend slopes
        if 'trend_slopes' in trend_data:
            slopes = trend_data['trend_slopes']
            ax1.plot(slopes.index, slopes.values, 
                     color=self.colors['primary'], linewidth=2)
            ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax1.fill_between(slopes.index, slopes.values, 0, 
                           where=(slopes.values > 0), 
                           color=self.colors['trend_up'], alpha=0.3, 
                           label='Upward Trend')
            ax1.fill_between(slopes.index, slopes.values, 0, 
                           where=(slopes.values < 0), 
                           color=self.colors['trend_down'], alpha=0.3, 
                           label='Downward Trend')
            
            ax1.set_title('Trend Slopes Over Time', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Trend Slope', fontsize=10)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
        
        # Plot trend strengths
        if 'trend_strengths' in trend_data:
            strengths = trend_data['trend_strengths']
            ax2.plot(strengths.index, strengths.values, 
                     color=self.colors['accent'], linewidth=2)
            ax2.fill_between(strengths.index, strengths.values, 0, 
                           color=self.colors['accent'], alpha=0.3)
            
            ax2.set_title('Trend Strength (R²)', fontsize=12, fontweight='bold')
            ax2.set_ylabel('R² Value', fontsize=10)
            ax2.set_xlabel('Date', fontsize=10)
            ax2.grid(True, alpha=0.3)
        
        # Format x-axis
        for ax in [ax1, ax2]:
            ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_path / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Trend analysis plot saved to {save_path}")
        return str(save_path)
    
    def plot_forecast(self, historical_data: pd.Series, 
                     forecast_data: pd.Series,
                     title: str = "Mood Score Forecast",
                     save_filename: str = "forecast.png") -> str:
        """
        Plot historical data with forecast
        
        Args:
            historical_data: Historical time series
            forecast_data: Forecast results
            title: Plot title
            save_filename: Filename to save plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot historical data
        ax.plot(historical_data.index, historical_data.values, 
                color=self.colors['primary'], linewidth=2, 
                label='Historical Data')
        
        # Plot forecast
        ax.plot(forecast_data.index, forecast_data.values, 
                color=self.colors['secondary'], linewidth=2, 
                linestyle='--', label='Forecast')
        
        # Add vertical line at forecast start
        forecast_start = forecast_data.index[0]
        ax.axvline(x=forecast_start, color='red', linestyle=':', 
                  alpha=0.7, label='Forecast Start')
        
        # Shade forecast area
        ax.fill_between(forecast_data.index, forecast_data.values, 
                       alpha=0.2, color=self.colors['secondary'])
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Mood Score', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_path / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Forecast plot saved to {save_path}")
        return str(save_path)
    
    def plot_moving_averages(self, series: pd.Series, 
                           moving_averages: Dict[str, pd.Series],
                           save_filename: str = "moving_averages.png") -> str:
        """
        Plot moving averages comparison
        
        Args:
            series: Original time series
            moving_averages: Dictionary of moving averages
            save_filename: Filename to save plot
            
        Returns:
            Path to saved plot
        """
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Plot original series
        ax.plot(series.index, series.values, 
                color=self.colors['neutral'], linewidth=1, 
                alpha=0.7, label='Original')
        
        # Plot moving averages
        colors = [self.colors['primary'], self.colors['secondary'], 
                 self.colors['accent'], self.colors['success']]
        
        for i, (ma_name, ma_series) in enumerate(moving_averages.items()):
            color = colors[i % len(colors)]
            ax.plot(ma_series.index, ma_series.values, 
                    color=color, linewidth=2, label=ma_name)
        
        # Formatting
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Mood Score', fontsize=12)
        ax.set_title('Moving Averages Comparison', fontsize=14, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Rotate x-axis labels
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_path / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Moving averages plot saved to {save_path}")
        return str(save_path)
    
    def plot_mood_distribution(self, data: pd.DataFrame,
                             save_filename: str = "mood_distribution.png") -> str:
        """
        Plot mood score distribution analysis
        
        Args:
            data: DataFrame with mood data
            save_filename: Filename to save plot
            
        Returns:
            Path to saved plot
        """
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Histogram
        ax1.hist(data['mood_score'], bins=20, color=self.colors['primary'], 
                alpha=0.7, edgecolor='black')
        ax1.set_title('Mood Score Distribution', fontweight='bold')
        ax1.set_xlabel('Mood Score')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(data['mood_score'], patch_artist=True,
                   boxprops=dict(facecolor=self.colors['secondary'], alpha=0.7))
        ax2.set_title('Mood Score Box Plot', fontweight='bold')
        ax2.set_ylabel('Mood Score')
        ax2.grid(True, alpha=0.3)
        
        # Time series scatter
        if 'timestamp' in data.columns:
            ax3.scatter(data['timestamp'], data['mood_score'], 
                       color=self.colors['accent'], alpha=0.6)
            ax3.set_title('Mood Score Over Time', fontweight='bold')
            ax3.set_xlabel('Date')
            ax3.set_ylabel('Mood Score')
            ax3.grid(True, alpha=0.3)
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # Weekly pattern
        if 'timestamp' in data.columns:
            data['day_of_week'] = pd.to_datetime(data['timestamp']).dt.day_name()
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                        'Friday', 'Saturday', 'Sunday']
            
            weekly_mood = data.groupby('day_of_week')['mood_score'].mean()
            weekly_mood = weekly_mood.reindex(day_order)
            
            bars = ax4.bar(weekly_mood.index, weekly_mood.values, 
                          color=self.colors['success'], alpha=0.7)
            ax4.set_title('Average Mood by Day of Week', fontweight='bold')
            ax4.set_xlabel('Day of Week')
            ax4.set_ylabel('Average Mood Score')
            ax4.grid(True, alpha=0.3)
            plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        save_path = self.save_path / save_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Mood distribution plot saved to {save_path}")
        return str(save_path)
    
    def create_comprehensive_report(self, analysis_results: Dict[str, Any],
                                  report_filename: str = "comprehensive_trend_report.png") -> str:
        """
        Create a comprehensive trend analysis report
        
        Args:
            analysis_results: Complete analysis results
            report_filename: Filename for the report
            
        Returns:
            Path to saved report
        """
        fig = plt.figure(figsize=(16, 12))
        
        # Create subplots
        gs = fig.add_gridspec(3, 3, height_ratios=[1, 1, 1], width_ratios=[2, 1, 1])
        
        # Main time series (top row, spanning 2 columns)
        ax1 = fig.add_subplot(gs[0, :2])
        series = analysis_results['time_series']
        ax1.plot(series.index, series.values, 
                color=self.colors['primary'], linewidth=2, label='Mood Score')
        
        # Add forecast if available
        if 'forecast' in analysis_results and analysis_results['forecast']:
            forecast = analysis_results['forecast']['forecast']
            ax1.plot(forecast.index, forecast.values, 
                    color=self.colors['secondary'], linewidth=2, 
                    linestyle='--', label='Forecast')
        
        ax1.set_title('Mood Score Time Series & Forecast', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Statistics summary (top right)
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.axis('off')
        
        stats = analysis_results['basic_statistics']
        stats_text = f"""
        Statistics Summary:
        
        Mean: {stats['mean']:.2f}
        Std Dev: {stats['std']:.2f}
        Min: {stats['min']:.2f}
        Max: {stats['max']:.2f}
        
        Data Points: {stats['count']}
        
        Current Trend:
        {analysis_results['trend_analysis']['current_trend']}
        """
        
        ax2.text(0.1, 0.9, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue', alpha=0.7))
        
        # Trend analysis (middle left)
        ax3 = fig.add_subplot(gs[1, :2])
        if 'trend_slopes' in analysis_results['trend_analysis']:
            slopes = analysis_results['trend_analysis']['trend_slopes']
            ax3.plot(slopes.index, slopes.values, 
                    color=self.colors['accent'], linewidth=2)
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax3.fill_between(slopes.index, slopes.values, 0, 
                           where=(slopes.values > 0), 
                           color=self.colors['trend_up'], alpha=0.3)
            ax3.fill_between(slopes.index, slopes.values, 0, 
                           where=(slopes.values < 0), 
                           color=self.colors['trend_down'], alpha=0.3)
        
        ax3.set_title('Trend Analysis', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Trend Slope')
        ax3.grid(True, alpha=0.3)
        
        # Moving averages (middle right)
        ax4 = fig.add_subplot(gs[1, 2])
        if 'moving_averages' in analysis_results:
            for ma_name, ma_series in analysis_results['moving_averages'].items():
                if len(ma_series.dropna()) > 0:
                    ax4.plot(ma_series.index, ma_series.values, 
                            linewidth=2, label=ma_name)
        
        ax4.set_title('Moving Averages', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Seasonal decomposition (bottom row)
        if 'seasonal_decomposition' in analysis_results:
            decomp = analysis_results['seasonal_decomposition']
            
            # Trend component
            ax5 = fig.add_subplot(gs[2, 0])
            if 'trend' in decomp and decomp['trend'] is not None:
                ax5.plot(decomp['trend'].index, decomp['trend'].values, 
                        color=self.colors['secondary'], linewidth=2)
            ax5.set_title('Trend Component', fontsize=10, fontweight='bold')
            ax5.grid(True, alpha=0.3)
            
            # Seasonal component
            ax6 = fig.add_subplot(gs[2, 1])
            if 'seasonal' in decomp and decomp['seasonal'] is not None:
                ax6.plot(decomp['seasonal'].index, decomp['seasonal'].values, 
                        color=self.colors['accent'], linewidth=2)
            ax6.set_title('Seasonal Component', fontsize=10, fontweight='bold')
            ax6.grid(True, alpha=0.3)
            
            # Residual component
            ax7 = fig.add_subplot(gs[2, 2])
            if 'residual' in decomp and decomp['residual'] is not None:
                ax7.plot(decomp['residual'].index, decomp['residual'].values, 
                        color=self.colors['neutral'], linewidth=2)
            ax7.set_title('Residual Component', fontsize=10, fontweight='bold')
            ax7.grid(True, alpha=0.3)
        
        # Format x-axis labels
        for ax in [ax1, ax3, ax4, ax5, ax6, ax7]:
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        # Save report
        save_path = self.save_path / report_filename
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Comprehensive trend report saved to {save_path}")
        return str(save_path) 