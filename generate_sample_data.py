"""
Generate Sample Data Script
Creates comprehensive mood tracking sample data and exports to Excel
"""

import sys
import os
from datetime import datetime
from colorama import init, Fore, Style

# Initialize colorama for colored output
init()

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection.mood_collector import MoodCollector
from data_collection.data_validator import DataValidator
from data_collection.storage_manager import StorageManager

def main():
    """Generate sample mood data and export to Excel"""
    
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧠 Mental Health Mood Tracker - Sample Data Generator{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    try:
        # Initialize components
        mood_collector = MoodCollector()
        data_validator = DataValidator()
        storage_manager = StorageManager()
        
        print(f"{Fore.YELLOW}📊 Generating realistic mood data...{Style.RESET_ALL}")
        
        # Generate sample data for 90 days
        sample_data = mood_collector.collect_sample_data(days=90)
        
        print(f"{Fore.GREEN}✅ Generated {len(sample_data)} mood entries{Style.RESET_ALL}")
        
        # Validate the data
        print(f"{Fore.YELLOW}🔍 Validating data quality...{Style.RESET_ALL}")
        validated_data = data_validator.validate(sample_data)
        
        print(f"{Fore.GREEN}✅ Validated {len(validated_data)} entries{Style.RESET_ALL}")
        
        # Get data quality report
        quality_report = data_validator.get_data_quality_report(validated_data)
        print(f"{Fore.CYAN}📋 Data Quality: {quality_report['validation_rate']}% valid entries{Style.RESET_ALL}")
        
        # Save to JSON
        print(f"{Fore.YELLOW}💾 Saving data to JSON...{Style.RESET_ALL}")
        json_path = storage_manager.save_data(validated_data, "sample_mood_data.json")
        print(f"{Fore.GREEN}✅ JSON saved to: {json_path}{Style.RESET_ALL}")
        
        # Export to Excel
        print(f"{Fore.YELLOW}📤 Exporting to Excel...{Style.RESET_ALL}")
        excel_path = storage_manager.export_to_excel(validated_data, "Mental_Health_Mood_Tracker_Sample_Data.xlsx")
        print(f"{Fore.GREEN}✅ Excel file created: {excel_path}{Style.RESET_ALL}")
        
        # Display summary statistics
        print(f"\n{Fore.CYAN}📊 Sample Data Summary:{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Total Entries: {quality_report['total_entries']}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Date Range: {quality_report['date_range_days']} days{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Average Mood: {quality_report['mood_statistics']['mean']:.2f}/10{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Data Completeness: {quality_report['data_completeness']:.2f}%{Style.RESET_ALL}")
        
        # Show most common emotions and triggers
        print(f"\n{Fore.CYAN}🎭 Most Common Emotions:{Style.RESET_ALL}")
        for emotion, count in quality_report['most_common_emotions'][:5]:
            print(f"{Fore.WHITE}  • {emotion}: {count} times{Style.RESET_ALL}")
        
        print(f"\n{Fore.CYAN}⚡ Most Common Triggers:{Style.RESET_ALL}")
        for trigger, count in quality_report['most_common_triggers'][:5]:
            print(f"{Fore.WHITE}  • {trigger}: {count} times{Style.RESET_ALL}")
        
        # Create backup
        print(f"\n{Fore.YELLOW}🔄 Creating backup...{Style.RESET_ALL}")
        backup_path = storage_manager.backup_data(validated_data)
        print(f"{Fore.GREEN}✅ Backup created: {backup_path}{Style.RESET_ALL}")
        
        # Show file structure
        print(f"\n{Fore.CYAN}📁 Generated Files:{Style.RESET_ALL}")
        files = storage_manager.list_files()
        for file_type, file_list in files.items():
            if file_list:
                print(f"{Fore.WHITE}  {file_type}:{Style.RESET_ALL}")
                for file in file_list:
                    print(f"{Fore.WHITE}    • {file}{Style.RESET_ALL}")
        
        print(f"\n{Fore.GREEN}🎉 Sample data generation complete!{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📋 The Excel file contains multiple sheets with comprehensive analysis{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}🔍 Use this data to test and demonstrate the mood tracking system{Style.RESET_ALL}")
        
    except Exception as e:
        print(f"{Fore.RED}❌ Error generating sample data: {e}{Style.RESET_ALL}")
        sys.exit(1)

if __name__ == "__main__":
    main() 