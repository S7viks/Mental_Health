"""
Run Trend Analysis - Test the complete trend analysis pipeline
"""

import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from trend_analysis.trend_analysis_module import TrendAnalysisModule
from colorama import init, Fore, Style

# Initialize colorama
init()

def main():
    """Run the complete trend analysis pipeline"""
    
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧠 Mental Health Mood Tracker - Trend Analysis Module{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*70}{Style.RESET_ALL}")
    
    # Initialize trend analysis module
    trend_module = TrendAnalysisModule()
    
    # Check if data exists
    data_path = "data/json/combined_comprehensive_dataset.json"
    if not Path(data_path).exists():
        print(f"{Fore.RED}❌ Data file not found: {data_path}{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}💡 Please run combine_datasets.py first to create the dataset{Style.RESET_ALL}")
        return
    
    try:
        # Run complete analysis
        results = trend_module.run_complete_analysis(data_path)
        
        if "error" in results:
            print(f"{Fore.RED}❌ Analysis failed: {results['error']}{Style.RESET_ALL}")
            return
        
        # Print summary of results
        print(f"\n{Fore.GREEN}📊 Analysis Summary:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}{'='*40}{Style.RESET_ALL}")
        
        # Print insights
        if 'insights' in results:
            insights = results['insights']
            
            print(f"\n{Fore.CYAN}💡 Key Insights:{Style.RESET_ALL}")
            
            # Overall trends
            if 'overall_trends' in insights:
                for trend_type, trend_info in insights['overall_trends'].items():
                    print(f"{Fore.WHITE}  • {trend_type}: {trend_info}{Style.RESET_ALL}")
            
            # Risk factors
            if 'risk_factors' in insights:
                print(f"\n{Fore.YELLOW}⚠️  Risk Factors:{Style.RESET_ALL}")
                for risk_type, risk_info in insights['risk_factors'].items():
                    print(f"{Fore.WHITE}  • {risk_type}: {risk_info}{Style.RESET_ALL}")
            
            # Positive patterns
            if 'positive_patterns' in insights:
                print(f"\n{Fore.GREEN}✅ Positive Patterns:{Style.RESET_ALL}")
                for pattern_type, pattern_info in insights['positive_patterns'].items():
                    print(f"{Fore.WHITE}  • {pattern_type}: {pattern_info}{Style.RESET_ALL}")
            
            # Recommendations
            if 'recommendations' in insights:
                print(f"\n{Fore.CYAN}📋 Recommendations:{Style.RESET_ALL}")
                for i, recommendation in enumerate(insights['recommendations'], 1):
                    print(f"{Fore.WHITE}  {i}. {recommendation}{Style.RESET_ALL}")
        
        # Print patterns
        if 'patterns' in results:
            patterns = results['patterns']
            
            print(f"\n{Fore.CYAN}🔍 Detected Patterns:{Style.RESET_ALL}")
            
            # Weekly patterns
            if 'weekly' in patterns:
                weekly = patterns['weekly']
                print(f"{Fore.WHITE}  • Best day: {weekly['best_day']} (avg: {weekly['best_score']:.2f}){Style.RESET_ALL}")
                print(f"{Fore.WHITE}  • Worst day: {weekly['worst_day']} (avg: {weekly['worst_score']:.2f}){Style.RESET_ALL}")
            
            # Mood streaks
            if 'streaks' in patterns:
                streaks = patterns['streaks']
                print(f"{Fore.WHITE}  • Max good streak: {streaks['good_streaks']['max_length']} days{Style.RESET_ALL}")
                print(f"{Fore.WHITE}  • Max bad streak: {streaks['bad_streaks']['max_length']} days{Style.RESET_ALL}")
            
            # Critical episodes
            if 'critical_episodes' in patterns:
                critical = patterns['critical_episodes']
                for episode_type, episode_data in critical.items():
                    if episode_data and 'count' in episode_data:
                        print(f"{Fore.WHITE}  • {episode_type}: {episode_data['count']} episodes{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}🎉 Trend Analysis Complete!{Style.RESET_ALL}")
        print(f"{Fore.CYAN}📁 Check the 'trend_analysis_results' folder for detailed results and visualizations{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error during analysis: {e}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 