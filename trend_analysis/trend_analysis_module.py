"""
Trend Analysis Module - Main interface for time series analysis and trend detection
"""

import json
import logging
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np
from colorama import init, Fore, Style

from .time_series_analyzer import TimeSeriesAnalyzer
from .trend_visualizer import TrendVisualizer

# Initialize colorama
init()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TrendAnalysisModule:
    """
    Main module for trend analysis and mood pattern detection
    """
    
    def __init__(self, output_dir: str = "trend_analysis_results"):
        """
        Initialize the trend analysis module
        
        Args:
            output_dir: Directory to save results
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.analyzer = TimeSeriesAnalyzer()
        self.visualizer = TrendVisualizer(save_path=str(self.output_dir / "visualizations"))
        
        # Results storage
        self.analysis_results = {}
        self.trends = {}
        self.patterns = {}
        
        logger.info("Trend Analysis Module initialized")
    
    def load_data(self, data_source: str = "data/json/combined_comprehensive_dataset.json") -> List[Dict[str, Any]]:
        """
        Load mood data from file
        
        Args:
            data_source: Path to data file
            
        Returns:
            List of mood entries
        """
        try:
            with open(data_source, 'r') as f:
                data = json.load(f)
            
            print(f"{Fore.GREEN}✅ Loaded {len(data)} mood entries from {data_source}{Style.RESET_ALL}")
            logger.info(f"Loaded {len(data)} entries from {data_source}")
            
            return data
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error loading data: {e}{Style.RESET_ALL}")
            logger.error(f"Error loading data: {e}")
            return []
    
    def analyze_trends(self, data: List[Dict[str, Any]], 
                      target_columns: List[str] = ['mood_score', 'stress_level', 'sleep_hours']) -> Dict[str, Any]:
        """
        Perform comprehensive trend analysis
        
        Args:
            data: Mood tracking data
            target_columns: Columns to analyze
            
        Returns:
            Dictionary with analysis results for all columns
        """
        print(f"\n{Fore.CYAN}🔍 Starting Trend Analysis...{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        
        all_results = {}
        
        for column in target_columns:
            if not data:
                continue
                
            # Check if column exists in data
            if column not in data[0]:
                print(f"{Fore.YELLOW}⚠️  Column '{column}' not found in data{Style.RESET_ALL}")
                continue
            
            print(f"\n{Fore.BLUE}📊 Analyzing {column}...{Style.RESET_ALL}")
            
            try:
                # Perform analysis
                results = self.analyzer.analyze(data, target_column=column)
                
                # Store results
                all_results[column] = results
                
                # Generate visualizations
                self._create_visualizations(results, column)
                
                # Print summary
                self._print_analysis_summary(results, column)
                
            except Exception as e:
                print(f"{Fore.RED}❌ Error analyzing {column}: {e}{Style.RESET_ALL}")
                logger.error(f"Error analyzing {column}: {e}")
        
        # Store all results
        self.analysis_results = all_results
        
        print(f"\n{Fore.GREEN}✅ Trend analysis complete!{Style.RESET_ALL}")
        
        return all_results
    
    def _create_visualizations(self, results: Dict[str, Any], column: str) -> None:
        """
        Create visualizations for analysis results
        
        Args:
            results: Analysis results
            column: Column name for labeling
        """
        try:
            # Basic time series plot
            self.visualizer.plot_time_series(
                results['time_series'], 
                title=f"{column.replace('_', ' ').title()} Time Series Analysis",
                save_filename=f"{column}_time_series.png"
            )
            
            # Seasonal decomposition
            if 'seasonal_decomposition' in results:
                self.visualizer.plot_seasonal_decomposition(
                    results['seasonal_decomposition'],
                    save_filename=f"{column}_seasonal_decomposition.png"
                )
            
            # Trend analysis
            if 'trend_analysis' in results:
                self.visualizer.plot_trend_analysis(
                    results['trend_analysis'],
                    save_filename=f"{column}_trend_analysis.png"
                )
            
            # Moving averages
            if 'moving_averages' in results:
                self.visualizer.plot_moving_averages(
                    results['time_series'],
                    results['moving_averages'],
                    save_filename=f"{column}_moving_averages.png"
                )
            
            # Forecast
            if 'forecast' in results:
                self.visualizer.plot_forecast(
                    results['time_series'],
                    results['forecast']['forecast'],
                    title=f"{column.replace('_', ' ').title()} Forecast",
                    save_filename=f"{column}_forecast.png"
                )
            
            # Comprehensive report
            self.visualizer.create_comprehensive_report(
                results,
                report_filename=f"{column}_comprehensive_report.png"
            )
            
        except Exception as e:
            print(f"{Fore.YELLOW}⚠️  Visualization error for {column}: {e}{Style.RESET_ALL}")
            logger.warning(f"Visualization error for {column}: {e}")
    
    def _print_analysis_summary(self, results: Dict[str, Any], column: str) -> None:
        """
        Print summary of analysis results
        
        Args:
            results: Analysis results
            column: Column name
        """
        print(f"\n{Fore.CYAN}📈 {column.replace('_', ' ').title()} Analysis Summary:{Style.RESET_ALL}")
        
        # Basic statistics
        if 'basic_statistics' in results:
            stats = results['basic_statistics']
            print(f"{Fore.WHITE}  • Mean: {stats['mean']:.2f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}  • Standard Deviation: {stats['std']:.2f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}  • Range: {stats['min']:.2f} - {stats['max']:.2f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}  • Data Points: {stats['count']}{Style.RESET_ALL}")
        
        # Trend information
        if 'trend_analysis' in results:
            trend = results['trend_analysis']
            print(f"{Fore.WHITE}  • Current Trend: {trend['current_trend']}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}  • Average Slope: {trend['average_slope']:.4f}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}  • Trend Volatility: {trend['trend_volatility']:.4f}{Style.RESET_ALL}")
        
        # Stationarity
        if 'stationarity' in results:
            stationarity = results['stationarity']
            status = "✅ Stationary" if stationarity.get('stationary', False) else "❌ Non-stationary"
            print(f"{Fore.WHITE}  • Stationarity: {status}{Style.RESET_ALL}")
        
        # ARIMA model
        if 'arima_model' in results:
            arima = results['arima_model']
            if 'aic' in arima and arima['aic'] is not None:
                print(f"{Fore.WHITE}  • ARIMA Model AIC: {arima['aic']:.2f}{Style.RESET_ALL}")
            if 'order' in arima:
                print(f"{Fore.WHITE}  • ARIMA Order: {arima['order']}{Style.RESET_ALL}")
        
        # Forecast
        if 'forecast' in results:
            forecast = results['forecast']
            print(f"{Fore.WHITE}  • Forecast Method: {forecast['method']}{Style.RESET_ALL}")
            print(f"{Fore.WHITE}  • Forecast Steps: {forecast['steps']}{Style.RESET_ALL}")
    
    def detect_patterns(self, data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect patterns in mood data
        
        Args:
            data: Mood tracking data
            
        Returns:
            Dictionary with detected patterns
        """
        print(f"\n{Fore.CYAN}🔍 Detecting Patterns...{Style.RESET_ALL}")
        
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        patterns = {}
        
        # Weekly patterns
        df['day_of_week'] = df['timestamp'].dt.day_name()
        weekly_mood = df.groupby('day_of_week')['mood_score'].mean()
        patterns['weekly'] = {
            'best_day': weekly_mood.idxmax(),
            'worst_day': weekly_mood.idxmin(),
            'best_score': weekly_mood.max(),
            'worst_score': weekly_mood.min(),
            'pattern': weekly_mood.to_dict()
        }
        
        # Monthly patterns
        df['month'] = df['timestamp'].dt.month_name()
        monthly_mood = df.groupby('month')['mood_score'].mean()
        patterns['monthly'] = {
            'best_month': monthly_mood.idxmax(),
            'worst_month': monthly_mood.idxmin(),
            'pattern': monthly_mood.to_dict()
        }
        
        # Correlation patterns
        numeric_cols = ['mood_score', 'stress_level', 'sleep_hours', 'exercise_minutes']
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            correlation_matrix = df[available_cols].corr()
            patterns['correlations'] = correlation_matrix.to_dict()
        
        # Mood streaks
        mood_streaks = self._detect_mood_streaks(df)
        patterns['streaks'] = mood_streaks
        
        # Critical episodes
        critical_episodes = self._detect_critical_episodes(df)
        patterns['critical_episodes'] = critical_episodes
        
        self.patterns = patterns
        
        print(f"{Fore.GREEN}✅ Pattern detection complete!{Style.RESET_ALL}")
        
        return patterns
    
    def _detect_mood_streaks(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect mood streaks (consecutive good/bad days)
        
        Args:
            df: DataFrame with mood data
            
        Returns:
            Dictionary with streak information
        """
        # Define thresholds
        good_threshold = 7
        bad_threshold = 4
        
        # Create binary series
        df['good_mood'] = df['mood_score'] >= good_threshold
        df['bad_mood'] = df['mood_score'] <= bad_threshold
        
        # Find streaks
        def find_streaks(series):
            streaks = []
            current_streak = 0
            
            for value in series:
                if value:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        streaks.append(current_streak)
                    current_streak = 0
            
            if current_streak > 0:
                streaks.append(current_streak)
            
            return streaks
        
        good_streaks = find_streaks(df['good_mood'])
        bad_streaks = find_streaks(df['bad_mood'])
        
        return {
            'good_streaks': {
                'count': len(good_streaks),
                'max_length': max(good_streaks) if good_streaks else 0,
                'average_length': np.mean(good_streaks) if good_streaks else 0
            },
            'bad_streaks': {
                'count': len(bad_streaks),
                'max_length': max(bad_streaks) if bad_streaks else 0,
                'average_length': np.mean(bad_streaks) if bad_streaks else 0
            }
        }
    
    def _detect_critical_episodes(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Detect critical mental health episodes
        
        Args:
            df: DataFrame with mood data
            
        Returns:
            Dictionary with critical episodes
        """
        critical_episodes = {
            'severe_low_mood': [],
            'high_stress_periods': [],
            'sleep_deprivation': [],
            'combined_crisis': []
        }
        
        # Severe low mood episodes (mood <= 3)
        severe_low = df[df['mood_score'] <= 3]
        if not severe_low.empty:
            critical_episodes['severe_low_mood'] = {
                'count': len(severe_low),
                'dates': severe_low['timestamp'].dt.date.tolist(),
                'percentage': len(severe_low) / len(df) * 100
            }
        
        # High stress periods (stress >= 8)
        if 'stress_level' in df.columns:
            high_stress = df[df['stress_level'] >= 8]
            if not high_stress.empty:
                critical_episodes['high_stress_periods'] = {
                    'count': len(high_stress),
                    'dates': high_stress['timestamp'].dt.date.tolist(),
                    'percentage': len(high_stress) / len(df) * 100
                }
        
        # Sleep deprivation (sleep <= 4 hours)
        if 'sleep_hours' in df.columns:
            sleep_deprived = df[df['sleep_hours'] <= 4]
            if not sleep_deprived.empty:
                critical_episodes['sleep_deprivation'] = {
                    'count': len(sleep_deprived),
                    'dates': sleep_deprived['timestamp'].dt.date.tolist(),
                    'percentage': len(sleep_deprived) / len(df) * 100
                }
        
        # Combined crisis (low mood + high stress + poor sleep)
        combined_crisis = df[
            (df['mood_score'] <= 4) & 
            (df.get('stress_level', 0) >= 7) & 
            (df.get('sleep_hours', 8) <= 5)
        ]
        if not combined_crisis.empty:
            critical_episodes['combined_crisis'] = {
                'count': len(combined_crisis),
                'dates': combined_crisis['timestamp'].dt.date.tolist(),
                'percentage': len(combined_crisis) / len(df) * 100
            }
        
        return critical_episodes
    
    def generate_insights(self, analysis_results: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate actionable insights from analysis
        
        Args:
            analysis_results: Analysis results (uses stored results if None)
            
        Returns:
            Dictionary with insights and recommendations
        """
        if analysis_results is None:
            analysis_results = self.analysis_results
        
        if not analysis_results:
            return {"error": "No analysis results available"}
        
        print(f"\n{Fore.CYAN}💡 Generating Insights...{Style.RESET_ALL}")
        
        insights = {
            'overall_trends': {},
            'risk_factors': {},
            'positive_patterns': {},
            'recommendations': []
        }
        
        # Analyze mood trends
        if 'mood_score' in analysis_results:
            mood_results = analysis_results['mood_score']
            
            # Overall trend direction
            if 'trend_analysis' in mood_results:
                trend = mood_results['trend_analysis']
                current_trend = trend['current_trend']
                
                if 'Downward' in current_trend:
                    insights['overall_trends']['mood'] = "⚠️ Declining mood trend detected"
                    insights['recommendations'].append("Consider implementing mood-boosting activities")
                elif 'Upward' in current_trend:
                    insights['overall_trends']['mood'] = "✅ Improving mood trend detected"
                    insights['recommendations'].append("Continue current positive practices")
                else:
                    insights['overall_trends']['mood'] = "📊 Stable mood pattern"
            
            # Risk assessment
            if 'basic_statistics' in mood_results:
                stats = mood_results['basic_statistics']
                if stats['mean'] < 5:
                    insights['risk_factors']['low_mood'] = "Average mood below healthy threshold"
                if stats['std'] > 2:
                    insights['risk_factors']['mood_volatility'] = "High mood volatility detected"
        
        # Analyze stress patterns
        if 'stress_level' in analysis_results:
            stress_results = analysis_results['stress_level']
            
            if 'basic_statistics' in stress_results:
                stats = stress_results['basic_statistics']
                if stats['mean'] > 6:
                    insights['risk_factors']['high_stress'] = "Average stress level is elevated"
                    insights['recommendations'].append("Implement stress management techniques")
        
        # Analyze sleep patterns
        if 'sleep_hours' in analysis_results:
            sleep_results = analysis_results['sleep_hours']
            
            if 'basic_statistics' in sleep_results:
                stats = sleep_results['basic_statistics']
                if stats['mean'] < 7:
                    insights['risk_factors']['poor_sleep'] = "Average sleep duration below recommended"
                    insights['recommendations'].append("Focus on improving sleep hygiene")
                elif stats['mean'] > 8:
                    insights['positive_patterns']['good_sleep'] = "Maintaining healthy sleep duration"
        
        # Pattern-based insights
        if self.patterns:
            if 'weekly' in self.patterns:
                weekly = self.patterns['weekly']
                best_day = weekly['best_day']
                worst_day = weekly['worst_day']
                
                insights['positive_patterns']['weekly'] = f"Best mood typically on {best_day}"
                insights['risk_factors']['weekly'] = f"Lowest mood typically on {worst_day}"
                
                insights['recommendations'].append(f"Plan self-care activities for {worst_day}")
            
            if 'streaks' in self.patterns:
                streaks = self.patterns['streaks']
                if streaks['bad_streaks']['max_length'] > 3:
                    insights['risk_factors']['mood_streaks'] = "Extended periods of low mood detected"
                    insights['recommendations'].append("Consider professional support for mood management")
        
        # Generate priority recommendations
        self._prioritize_recommendations(insights)
        
        print(f"{Fore.GREEN}✅ Insights generated!{Style.RESET_ALL}")
        
        return insights
    
    def _prioritize_recommendations(self, insights: Dict[str, Any]) -> None:
        """
        Prioritize recommendations based on risk factors
        
        Args:
            insights: Insights dictionary to modify
        """
        risk_count = len(insights['risk_factors'])
        
        if risk_count >= 3:
            insights['priority_level'] = "HIGH"
            insights['recommendations'].insert(0, "🚨 Multiple risk factors detected - consider professional consultation")
        elif risk_count >= 2:
            insights['priority_level'] = "MEDIUM"
            insights['recommendations'].insert(0, "⚠️ Several concerns identified - implement targeted interventions")
        else:
            insights['priority_level'] = "LOW"
            insights['recommendations'].insert(0, "✅ Overall patterns appear healthy - maintain current practices")
    
    def save_results(self, filename: str = "trend_analysis_results.json") -> str:
        """
        Save all analysis results to file
        
        Args:
            filename: Output filename
            
        Returns:
            Path to saved file
        """
        output_path = self.output_dir / filename
        
        # Prepare results for JSON serialization
        serializable_results = self._prepare_for_json(self.analysis_results)
        
        # Add patterns and insights
        serializable_results['patterns'] = self.patterns
        serializable_results['insights'] = self.generate_insights()
        serializable_results['analysis_timestamp'] = datetime.now().isoformat()
        
        try:
            with open(output_path, 'w') as f:
                json.dump(serializable_results, f, indent=2, default=str)
            
            print(f"{Fore.GREEN}✅ Results saved to {output_path}{Style.RESET_ALL}")
            return str(output_path)
        
        except Exception as e:
            print(f"{Fore.RED}❌ Error saving results: {e}{Style.RESET_ALL}")
            return ""
    
    def _prepare_for_json(self, data: Any) -> Any:
        """
        Prepare data for JSON serialization
        
        Args:
            data: Data to serialize
            
        Returns:
            JSON-serializable data
        """
        if isinstance(data, dict):
            return {k: self._prepare_for_json(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._prepare_for_json(item) for item in data]
        elif isinstance(data, pd.Series):
            return data.to_dict()
        elif isinstance(data, pd.DataFrame):
            return data.to_dict()
        elif isinstance(data, (np.integer, np.floating)):
            return float(data)
        elif isinstance(data, np.ndarray):
            return data.tolist()
        else:
            return data
    
    def run_complete_analysis(self, data_source: str = "data/json/combined_comprehensive_dataset.json") -> Dict[str, Any]:
        """
        Run complete trend analysis pipeline
        
        Args:
            data_source: Path to data file
            
        Returns:
            Complete analysis results
        """
        print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🧠 Mental Health Trend Analysis Module{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
        
        # Load data
        data = self.load_data(data_source)
        if not data:
            return {"error": "No data loaded"}
        
        # Analyze trends
        analysis_results = self.analyze_trends(data)
        
        # Detect patterns
        patterns = self.detect_patterns(data)
        
        # Generate insights
        insights = self.generate_insights(analysis_results)
        
        # Save results
        results_path = self.save_results()
        
        # Final summary
        print(f"\n{Fore.GREEN}🎉 Complete Analysis Finished!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Analysis Results: {len(analysis_results)} metrics analyzed{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🔍 Patterns Detected: {len(patterns)} pattern types{Style.RESET_ALL}")
        print(f"{Fore.CYAN}💡 Insights Generated: {len(insights.get('recommendations', []))} recommendations{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📁 Results saved to: {results_path}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📋 Visualizations saved to: {self.output_dir}/visualizations{Style.RESET_ALL}")
        
        return {
            'analysis_results': analysis_results,
            'patterns': patterns,
            'insights': insights,
            'results_path': results_path
        } 