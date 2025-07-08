"""
Mental Health Mood Tracker - Main Application

This is the main entry point that demonstrates the complete pipeline:
Mood Entries → Trend Analysis → Pattern Recognition → Actionable Insights
"""

import sys
import os
import logging
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('mood_tracker.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def print_banner():
    """Print the application banner"""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧠 Mental Health Mood Tracker with Insights{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Pipeline: Mood Entries → Trend Analysis → Pattern Recognition → Insights{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Status: System Ready{Style.RESET_ALL}\n")

def main():
    """Main application entry point"""
    print_banner()
    
    try:
        # Import modules (will be implemented in steps)
        from data_collection import MoodCollector, DataValidator, StorageManager
        from trend_analysis import TimeSeriesAnalyzer, TrendDetector, SeasonalAnalyzer
        from pattern_detection import AnomalyDetector, PatternRecognizer, MentalHealthIndicators
        from recommendations import InsightGenerator, CareRecommender, CrisisDetector
        
        logger.info("All modules imported successfully")
        
        # Initialize components
        mood_collector = MoodCollector()
        data_validator = DataValidator()
        storage_manager = StorageManager()
        
        time_series_analyzer = TimeSeriesAnalyzer()
        trend_detector = TrendDetector()
        seasonal_analyzer = SeasonalAnalyzer()
        
        anomaly_detector = AnomalyDetector()
        pattern_recognizer = PatternRecognizer()
        mental_health_indicators = MentalHealthIndicators()
        
        insight_generator = InsightGenerator()
        care_recommender = CareRecommender()
        crisis_detector = CrisisDetector()
        
        logger.info("All components initialized successfully")
        
        # Run the complete pipeline
        print(f"{Fore.BLUE}🔄 Starting Mental Health Mood Analysis Pipeline...{Style.RESET_ALL}")
        
        # Step 1: Data Collection
        print(f"{Fore.YELLOW}Step 1: Data Collection{Style.RESET_ALL}")
        mood_data = mood_collector.collect_sample_data()
        validated_data = data_validator.validate(mood_data)
        storage_manager.save_data(validated_data)
        
        # Step 2: Trend Analysis
        print(f"{Fore.YELLOW}Step 2: Trend Analysis{Style.RESET_ALL}")
        trends = time_series_analyzer.analyze(validated_data)
        trend_patterns = trend_detector.detect_trends(trends)
        seasonal_patterns = seasonal_analyzer.analyze_seasonal_patterns(validated_data)
        
        # Step 3: Pattern Detection
        print(f"{Fore.YELLOW}Step 3: Pattern Detection{Style.RESET_ALL}")
        anomalies = anomaly_detector.detect_anomalies(validated_data)
        patterns = pattern_recognizer.identify_patterns(validated_data)
        health_indicators = mental_health_indicators.assess_indicators(validated_data)
        
        # Step 4: Recommendations
        print(f"{Fore.YELLOW}Step 4: Generating Insights & Recommendations{Style.RESET_ALL}")
        insights = insight_generator.generate_insights(
            trends, patterns, anomalies, health_indicators
        )
        care_recommendations = care_recommender.recommend_care(health_indicators)
        crisis_alerts = crisis_detector.detect_crisis(anomalies, health_indicators)
        
        # Display results
        print(f"\n{Fore.GREEN}✅ Pipeline Complete! Results:{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📊 Trends Identified: {len(trend_patterns)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🔍 Patterns Found: {len(patterns)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}⚠️  Anomalies Detected: {len(anomalies)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}💡 Insights Generated: {len(insights)}{Style.RESET_ALL}")
        print(f"{Fore.CYAN}🏥 Care Recommendations: {len(care_recommendations)}{Style.RESET_ALL}")
        
        if crisis_alerts:
            print(f"{Fore.RED}🚨 Crisis Alerts: {len(crisis_alerts)}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}System demonstration complete!{Style.RESET_ALL}")
        
    except ImportError as e:
        logger.error(f"Module import failed: {e}")
        print(f"{Fore.RED}❌ Some modules not yet implemented. Building in progress...{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}ℹ️  This is expected during step-by-step implementation.{Style.RESET_ALL}")
    except Exception as e:
        logger.error(f"Application error: {e}")
        print(f"{Fore.RED}❌ Error: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 