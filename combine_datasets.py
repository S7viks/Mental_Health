"""
Combine Datasets Script
Merges real public mental health data with synthetic data for comprehensive dataset
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
from colorama import init, Fore, Style
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_collection import DataValidator, StorageManager

# Initialize colorama
init()

def load_public_data():
    """Load the processed public datasets"""
    public_data_path = Path("data/public_datasets/processed/combined_public_datasets.json")
    
    if not public_data_path.exists():
        print(f"{Fore.RED}❌ Public datasets not found. Run process_public_datasets.py first{Style.RESET_ALL}")
        return []
    
    try:
        with open(public_data_path, 'r') as f:
            data = json.load(f)
        print(f"{Fore.GREEN}✅ Loaded {len(data)} entries from public datasets{Style.RESET_ALL}")
        return data
    except Exception as e:
        print(f"{Fore.RED}❌ Error loading public data: {e}{Style.RESET_ALL}")
        return []

def load_synthetic_data():
    """Load the synthetic mood data"""
    synthetic_data_path = Path("data/json/sample_mood_data.json")
    
    if not synthetic_data_path.exists():
        print(f"{Fore.RED}❌ Synthetic data not found. Run generate_sample_data.py first{Style.RESET_ALL}")
        return []
    
    try:
        with open(synthetic_data_path, 'r') as f:
            data = json.load(f)
        print(f"{Fore.GREEN}✅ Loaded {len(data)} entries from synthetic data{Style.RESET_ALL}")
        return data
    except Exception as e:
        print(f"{Fore.RED}❌ Error loading synthetic data: {e}{Style.RESET_ALL}")
        return []

def standardize_entry_ids(data, prefix):
    """Standardize entry IDs to avoid conflicts"""
    for i, entry in enumerate(data):
        # Update entry ID with prefix to avoid conflicts
        timestamp = entry.get('timestamp', datetime.now().isoformat())
        date_part = timestamp.split('T')[0].replace('-', '')
        entry['entry_id'] = f"{prefix}_{i+1}_{date_part}"
        
        # Add source information
        entry['data_source'] = prefix
    
    return data

def merge_datasets(public_data, synthetic_data):
    """Merge public and synthetic datasets"""
    print(f"\n{Fore.BLUE}🔄 Merging datasets...{Style.RESET_ALL}")
    
    # Standardize entry IDs
    public_data = standardize_entry_ids(public_data, "public")
    synthetic_data = standardize_entry_ids(synthetic_data, "synthetic")
    
    # Combine datasets
    combined_data = public_data + synthetic_data
    
    # Sort by timestamp
    combined_data.sort(key=lambda x: x['timestamp'])
    
    print(f"{Fore.GREEN}✅ Combined {len(public_data)} public + {len(synthetic_data)} synthetic = {len(combined_data)} total entries{Style.RESET_ALL}")
    
    return combined_data

def analyze_combined_dataset(data):
    """Analyze the combined dataset and generate insights"""
    print(f"\n{Fore.CYAN}📊 Combined Dataset Analysis{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
    
    df = pd.DataFrame(data)
    
    # Basic statistics
    print(f"{Fore.WHITE}Total Entries: {len(data)}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Date Range: {len(df['timestamp'].apply(lambda x: x.split('T')[0]).unique())} unique days{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Average Mood Score: {df['mood_score'].mean():.2f}/10{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Mood Score Range: {df['mood_score'].min()}-{df['mood_score'].max()}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Average Stress Level: {df['stress_level'].mean():.2f}/10{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Average Sleep Hours: {df['sleep_hours'].mean():.2f}{Style.RESET_ALL}")
    print(f"{Fore.WHITE}Average Exercise Minutes: {df['exercise_minutes'].mean():.2f}{Style.RESET_ALL}")
    
    # Data source breakdown
    source_counts = df['data_source'].value_counts()
    print(f"\n{Fore.CYAN}📁 Data Source Breakdown:{Style.RESET_ALL}")
    for source, count in source_counts.items():
        percentage = (count / len(data)) * 100
        print(f"{Fore.WHITE}  • {source}: {count} entries ({percentage:.1f}%){Style.RESET_ALL}")
    
    # Mood distribution
    mood_ranges = {
        'Very Low (1-2)': len(df[df['mood_score'] <= 2]),
        'Low (3-4)': len(df[(df['mood_score'] >= 3) & (df['mood_score'] <= 4)]),
        'Moderate (5-6)': len(df[(df['mood_score'] >= 5) & (df['mood_score'] <= 6)]),
        'Good (7-8)': len(df[(df['mood_score'] >= 7) & (df['mood_score'] <= 8)]),
        'Excellent (9-10)': len(df[df['mood_score'] >= 9])
    }
    
    print(f"\n{Fore.CYAN}🎭 Mood Distribution:{Style.RESET_ALL}")
    for mood_range, count in mood_ranges.items():
        percentage = (count / len(data)) * 100
        print(f"{Fore.WHITE}  • {mood_range}: {count} entries ({percentage:.1f}%){Style.RESET_ALL}")
    
    # Most common emotions
    all_emotions = []
    for entry in data:
        all_emotions.extend(entry.get('emotions', []))
    
    emotion_counts = pd.Series(all_emotions).value_counts()
    print(f"\n{Fore.CYAN}😊 Most Common Emotions:{Style.RESET_ALL}")
    for emotion, count in emotion_counts.head(8).items():
        percentage = (count / len(all_emotions)) * 100
        print(f"{Fore.WHITE}  • {emotion}: {count} times ({percentage:.1f}%){Style.RESET_ALL}")
    
    # Most common triggers
    all_triggers = []
    for entry in data:
        all_triggers.extend(entry.get('triggers', []))
    
    if all_triggers:
        trigger_counts = pd.Series(all_triggers).value_counts()
        print(f"\n{Fore.CYAN}⚡ Most Common Triggers:{Style.RESET_ALL}")
        for trigger, count in trigger_counts.head(8).items():
            percentage = (count / len(all_triggers)) * 100
            print(f"{Fore.WHITE}  • {trigger}: {count} times ({percentage:.1f}%){Style.RESET_ALL}")
    
    # Mental health indicators
    low_mood_entries = len(df[df['mood_score'] <= 4])
    high_stress_entries = len(df[df['stress_level'] >= 7])
    low_sleep_entries = len(df[df['sleep_hours'] <= 5])
    
    print(f"\n{Fore.CYAN}🏥 Mental Health Indicators:{Style.RESET_ALL}")
    print(f"{Fore.WHITE}  • Low mood episodes (≤4): {low_mood_entries} ({(low_mood_entries/len(data)*100):.1f}%){Style.RESET_ALL}")
    print(f"{Fore.WHITE}  • High stress episodes (≥7): {high_stress_entries} ({(high_stress_entries/len(data)*100):.1f}%){Style.RESET_ALL}")
    print(f"{Fore.WHITE}  • Poor sleep episodes (≤5h): {low_sleep_entries} ({(low_sleep_entries/len(data)*100):.1f}%){Style.RESET_ALL}")
    
    return {
        'total_entries': len(data),
        'mood_average': df['mood_score'].mean(),
        'stress_average': df['stress_level'].mean(),
        'low_mood_percentage': (low_mood_entries/len(data)*100),
        'high_stress_percentage': (high_stress_entries/len(data)*100),
        'data_sources': source_counts.to_dict()
    }

def save_combined_dataset(data, filename="combined_comprehensive_dataset.json"):
    """Save the combined dataset"""
    print(f"\n{Fore.BLUE}💾 Saving combined dataset...{Style.RESET_ALL}")
    
    # Save to main data directory
    main_data_path = Path("data/json") / filename
    main_data_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(main_data_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"{Fore.GREEN}✅ Combined dataset saved to: {main_data_path}{Style.RESET_ALL}")
        
        # Also save to public datasets processed folder
        public_data_path = Path("data/public_datasets/processed") / filename
        with open(public_data_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"{Fore.GREEN}✅ Backup saved to: {public_data_path}{Style.RESET_ALL}")
        
        return str(main_data_path)
    
    except Exception as e:
        print(f"{Fore.RED}❌ Error saving combined dataset: {e}{Style.RESET_ALL}")
        return None

def export_combined_to_excel(data):
    """Export combined dataset to Excel with analysis"""
    print(f"\n{Fore.BLUE}📤 Exporting to Excel...{Style.RESET_ALL}")
    
    try:
        storage_manager = StorageManager()
        excel_path = storage_manager.export_to_excel(data, "Combined_Mental_Health_Dataset_Analysis.xlsx")
        print(f"{Fore.GREEN}✅ Excel analysis exported to: {excel_path}{Style.RESET_ALL}")
        return excel_path
    except Exception as e:
        print(f"{Fore.RED}❌ Error exporting to Excel: {e}{Style.RESET_ALL}")
        return None

def main():
    """Main function to combine datasets"""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧠 Mental Health Mood Tracker - Dataset Combination{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    # Load datasets
    print(f"{Fore.YELLOW}📂 Loading datasets...{Style.RESET_ALL}")
    public_data = load_public_data()
    synthetic_data = load_synthetic_data()
    
    if not public_data and not synthetic_data:
        print(f"{Fore.RED}❌ No datasets found to combine{Style.RESET_ALL}")
        return
    
    # Merge datasets
    combined_data = merge_datasets(public_data, synthetic_data)
    
    # Validate combined data
    print(f"\n{Fore.YELLOW}🔍 Validating combined dataset...{Style.RESET_ALL}")
    validator = DataValidator()
    validated_data = validator.validate(combined_data)
    
    validation_rate = (len(validated_data) / len(combined_data)) * 100
    print(f"{Fore.GREEN}✅ Validation complete: {len(validated_data)}/{len(combined_data)} entries valid ({validation_rate:.1f}%){Style.RESET_ALL}")
    
    # Analyze combined dataset
    analysis_results = analyze_combined_dataset(validated_data)
    
    # Save combined dataset
    saved_path = save_combined_dataset(validated_data)
    
    # Export to Excel
    excel_path = export_combined_to_excel(validated_data)
    
    # Create backup
    if saved_path:
        storage_manager = StorageManager()
        backup_path = storage_manager.backup_data(validated_data)
        print(f"{Fore.GREEN}✅ Backup created: {backup_path}{Style.RESET_ALL}")
    
    # Final summary
    print(f"\n{Fore.GREEN}🎉 Dataset combination complete!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📊 Final Dataset: {len(validated_data)} entries from real + synthetic sources{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📁 Main file: data/json/combined_comprehensive_dataset.json{Style.RESET_ALL}")
    print(f"{Fore.CYAN}📋 Excel analysis: {excel_path if excel_path else 'Export failed'}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}🚀 Ready for Trend Analysis Module implementation!{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 