"""
Process Public Mental Health Datasets
Converts various public datasets into our standardized mood tracking format
"""

import pandas as pd
import os
import sys
import json
from datetime import datetime, timedelta
import numpy as np
from pathlib import Path
from colorama import init, Fore, Style

# Initialize colorama
init()

class PublicDatasetProcessor:
    """
    Processes public mental health datasets into our standardized format
    """
    
    def __init__(self):
        self.raw_data_dir = Path("data/public_datasets/raw")
        self.processed_data_dir = Path("data/public_datasets/processed")
        self.output_dir = Path("data/public_datasets/processed")
        
        # Create directories if they don't exist
        self.raw_data_dir.mkdir(parents=True, exist_ok=True)
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard emotion mapping
        self.emotion_mapping = {
            'happy': ['happy', 'joy', 'positive', 'good', 'satisfied', 'content'],
            'sad': ['sad', 'unhappy', 'down', 'low', 'depressed', 'blue'],
            'anxious': ['anxious', 'worried', 'nervous', 'stressed', 'tense'],
            'angry': ['angry', 'mad', 'frustrated', 'irritated', 'annoyed'],
            'calm': ['calm', 'peaceful', 'relaxed', 'serene', 'tranquil'],
            'excited': ['excited', 'enthusiastic', 'energetic', 'thrilled'],
            'lonely': ['lonely', 'isolated', 'alone', 'disconnected'],
            'hopeful': ['hopeful', 'optimistic', 'positive', 'confident']
        }
        
        # Standard trigger mapping
        self.trigger_mapping = {
            'work_stress': ['work', 'job', 'career', 'workplace', 'office'],
            'family_issues': ['family', 'parent', 'spouse', 'child', 'relative'],
            'health_concerns': ['health', 'illness', 'medical', 'physical'],
            'financial_worry': ['money', 'financial', 'debt', 'income', 'budget'],
            'relationship_problems': ['relationship', 'partner', 'dating', 'marriage'],
            'social_isolation': ['social', 'friends', 'isolation', 'loneliness'],
            'sleep_deprivation': ['sleep', 'insomnia', 'tired', 'exhausted']
        }
    
    def check_available_datasets(self):
        """Check which datasets are available in the raw data directory"""
        print(f"\n{Fore.CYAN}📂 Checking for available datasets...{Style.RESET_ALL}")
        
        expected_files = [
            'mental_health_tech_survey.csv',
            'depression_anxiety_dataset.csv',
            'mood_tracking_dataset.csv',
            'sleep_health_dataset.csv',
            'student_mental_health.csv'
        ]
        
        available_files = []
        missing_files = []
        
        for file in expected_files:
            filepath = self.raw_data_dir / file
            if filepath.exists():
                available_files.append(file)
                print(f"{Fore.GREEN}✅ Found: {file}{Style.RESET_ALL}")
            else:
                missing_files.append(file)
                print(f"{Fore.RED}❌ Missing: {file}{Style.RESET_ALL}")
        
        if missing_files:
            print(f"\n{Fore.YELLOW}📋 To download missing datasets:{Style.RESET_ALL}")
            dataset_urls = {
                'mental_health_tech_survey.csv': 'https://www.kaggle.com/datasets/osmi/mental-health-in-tech-survey',
                'depression_anxiety_dataset.csv': 'https://www.kaggle.com/datasets/diegobabio/depression-and-anxiety-dataset',
                'mood_tracking_dataset.csv': 'https://www.kaggle.com/datasets/arashnic/mood-tracking-dataset',
                'sleep_health_dataset.csv': 'https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset',
                'student_mental_health.csv': 'https://www.kaggle.com/datasets/shariful07/student-mental-health'
            }
            
            for file in missing_files:
                if file in dataset_urls:
                    print(f"{Fore.WHITE}• {file}: {dataset_urls[file]}{Style.RESET_ALL}")
        
        return available_files
    
    def process_mental_health_tech_survey(self):
        """Process the Mental Health in Tech Survey dataset"""
        file_path = self.raw_data_dir / 'mental_health_tech_survey.csv'
        if not file_path.exists():
            print(f"{Fore.YELLOW}⚠️  Mental Health Tech Survey not found{Style.RESET_ALL}")
            return None
        
        print(f"{Fore.BLUE}🔄 Processing Mental Health in Tech Survey...{Style.RESET_ALL}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"{Fore.GREEN}✅ Loaded {len(df)} records{Style.RESET_ALL}")
            
            # Convert to our format
            processed_data = []
            
            for idx, row in df.iterrows():
                # Generate timestamp (simulate daily entries over past 3 months)
                days_ago = np.random.randint(0, 90)
                timestamp = datetime.now() - timedelta(days=days_ago)
                
                # Extract mood score (1-10 scale)
                mood_score = self._extract_mood_score(row)
                
                # Extract emotions
                emotions = self._extract_emotions(row)
                
                # Extract contextual data
                sleep_hours = np.random.uniform(4, 10)  # Simulate sleep data
                exercise_minutes = np.random.randint(0, 120)
                social_interactions = np.random.randint(0, 10)
                stress_level = self._extract_stress_level(row)
                
                # Extract triggers
                triggers = self._extract_triggers(row)
                
                entry = {
                    'timestamp': timestamp.isoformat(),
                    'mood_score': mood_score,
                    'emotions': emotions,
                    'sleep_hours': sleep_hours,
                    'exercise_minutes': exercise_minutes,
                    'social_interactions': social_interactions,
                    'stress_level': stress_level,
                    'weather': np.random.choice(['sunny', 'cloudy', 'rainy', 'snowy']),
                    'notes': f"Tech survey entry {idx+1}",
                    'triggers': triggers,
                    'entry_id': f"tech_survey_{idx+1}_{timestamp.strftime('%Y%m%d')}"
                }
                
                processed_data.append(entry)
            
            # Save processed data
            output_file = self.output_dir / 'mental_health_tech_processed.json'
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"{Fore.GREEN}✅ Processed {len(processed_data)} entries from Tech Survey{Style.RESET_ALL}")
            return processed_data
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing Tech Survey: {e}{Style.RESET_ALL}")
            return None
    
    def process_student_mental_health(self):
        """Process the Student Mental Health dataset"""
        file_path = self.raw_data_dir / 'student_mental_health.csv'
        if not file_path.exists():
            print(f"{Fore.YELLOW}⚠️  Student Mental Health dataset not found{Style.RESET_ALL}")
            return None
        
        print(f"{Fore.BLUE}🔄 Processing Student Mental Health dataset...{Style.RESET_ALL}")
        
        try:
            df = pd.read_csv(file_path)
            print(f"{Fore.GREEN}✅ Loaded {len(df)} records{Style.RESET_ALL}")
            
            processed_data = []
            
            for idx, row in df.iterrows():
                # Generate timestamp
                days_ago = np.random.randint(0, 90)
                timestamp = datetime.now() - timedelta(days=days_ago)
                
                # Extract mood score based on depression/anxiety indicators
                mood_score = self._calculate_mood_from_mental_health_indicators(row)
                
                # Extract emotions based on mental health status
                emotions = self._extract_emotions_from_mental_health_status(row)
                
                # Student-specific contextual data
                sleep_hours = np.random.uniform(4, 12)  # Students have varied sleep patterns
                exercise_minutes = np.random.randint(0, 90)
                social_interactions = np.random.randint(1, 15)  # Students typically more social
                stress_level = self._extract_student_stress_level(row)
                
                # Student-specific triggers
                triggers = self._extract_student_triggers(row)
                
                entry = {
                    'timestamp': timestamp.isoformat(),
                    'mood_score': mood_score,
                    'emotions': emotions,
                    'sleep_hours': sleep_hours,
                    'exercise_minutes': exercise_minutes,
                    'social_interactions': social_interactions,
                    'stress_level': stress_level,
                    'weather': np.random.choice(['sunny', 'cloudy', 'rainy', 'snowy']),
                    'notes': f"Student mental health entry {idx+1}",
                    'triggers': triggers,
                    'entry_id': f"student_mh_{idx+1}_{timestamp.strftime('%Y%m%d')}"
                }
                
                processed_data.append(entry)
            
            # Save processed data
            output_file = self.output_dir / 'student_mental_health_processed.json'
            with open(output_file, 'w') as f:
                json.dump(processed_data, f, indent=2)
            
            print(f"{Fore.GREEN}✅ Processed {len(processed_data)} entries from Student Mental Health{Style.RESET_ALL}")
            return processed_data
            
        except Exception as e:
            print(f"{Fore.RED}❌ Error processing Student Mental Health: {e}{Style.RESET_ALL}")
            return None
    
    def _extract_mood_score(self, row):
        """Extract mood score from various possible columns"""
        # Common mood-related column names
        mood_columns = ['mood', 'mental_health_score', 'wellbeing', 'happiness', 'satisfaction']
        
        for col in mood_columns:
            if col in row.index and pd.notna(row[col]):
                # Normalize to 1-10 scale
                value = row[col]
                if isinstance(value, str):
                    # Handle categorical responses
                    if value.lower() in ['very poor', 'poor', 'bad']:
                        return np.random.randint(1, 4)
                    elif value.lower() in ['fair', 'okay', 'average']:
                        return np.random.randint(4, 7)
                    elif value.lower() in ['good', 'well', 'fine']:
                        return np.random.randint(6, 9)
                    elif value.lower() in ['very good', 'excellent', 'great']:
                        return np.random.randint(8, 11)
                elif isinstance(value, (int, float)):
                    # Normalize numeric values
                    if value <= 1:
                        return int(value * 10)
                    elif value <= 5:
                        return int(value * 2)
                    else:
                        return min(int(value), 10)
        
        # Default random mood if no specific mood column found
        return np.random.randint(3, 8)
    
    def _extract_emotions(self, row):
        """Extract emotions from row data"""
        emotions = []
        
        # Check for depression/anxiety indicators
        if any(col for col in row.index if 'depression' in col.lower() or 'depressed' in col.lower()):
            emotions.append('sad')
        if any(col for col in row.index if 'anxiety' in col.lower() or 'anxious' in col.lower()):
            emotions.append('anxious')
        if any(col for col in row.index if 'stress' in col.lower() or 'stressed' in col.lower()):
            emotions.append('stressed')
        
        # Ensure at least one emotion
        if not emotions:
            emotions = [np.random.choice(['content', 'calm', 'worried', 'hopeful'])]
        
        return emotions[:3]  # Limit to 3 emotions
    
    def _extract_stress_level(self, row):
        """Extract stress level from row data"""
        stress_columns = ['stress', 'work_stress', 'pressure', 'tension']
        
        for col in stress_columns:
            if col in row.index and pd.notna(row[col]):
                value = row[col]
                if isinstance(value, str):
                    if 'high' in value.lower() or 'severe' in value.lower():
                        return np.random.randint(7, 11)
                    elif 'moderate' in value.lower() or 'medium' in value.lower():
                        return np.random.randint(4, 8)
                    elif 'low' in value.lower() or 'mild' in value.lower():
                        return np.random.randint(1, 5)
                elif isinstance(value, (int, float)):
                    return min(int(value * 2), 10)
        
        return np.random.randint(3, 8)
    
    def _extract_triggers(self, row):
        """Extract triggers from row data"""
        triggers = []
        
        # Check for work-related issues
        if any(col for col in row.index if 'work' in col.lower() or 'job' in col.lower()):
            triggers.append('work_stress')
        
        # Check for family issues
        if any(col for col in row.index if 'family' in col.lower()):
            triggers.append('family_issues')
        
        # Check for health concerns
        if any(col for col in row.index if 'health' in col.lower() or 'medical' in col.lower()):
            triggers.append('health_concerns')
        
        # Default triggers if none found
        if not triggers:
            triggers = [np.random.choice(['work_stress', 'sleep_deprivation', 'social_isolation'])]
        
        return triggers[:3]  # Limit to 3 triggers
    
    def _calculate_mood_from_mental_health_indicators(self, row):
        """Calculate mood score based on mental health indicators"""
        base_mood = 6  # Start with neutral mood
        
        # Check for depression indicators
        depression_cols = [col for col in row.index if 'depression' in col.lower() or 'depressed' in col.lower()]
        if depression_cols:
            base_mood -= 2
        
        # Check for anxiety indicators
        anxiety_cols = [col for col in row.index if 'anxiety' in col.lower() or 'anxious' in col.lower()]
        if anxiety_cols:
            base_mood -= 1
        
        # Check for treatment indicators (positive effect)
        treatment_cols = [col for col in row.index if 'treatment' in col.lower() or 'therapy' in col.lower()]
        if treatment_cols:
            base_mood += 1
        
        # Ensure mood stays in valid range
        return max(1, min(10, base_mood + np.random.randint(-1, 2)))
    
    def _extract_emotions_from_mental_health_status(self, row):
        """Extract emotions based on mental health status"""
        emotions = []
        
        # Check columns for mental health indicators
        for col in row.index:
            col_lower = col.lower()
            if 'depression' in col_lower or 'depressed' in col_lower:
                emotions.extend(['sad', 'lonely'])
            elif 'anxiety' in col_lower or 'anxious' in col_lower:
                emotions.extend(['anxious', 'worried'])
            elif 'panic' in col_lower:
                emotions.extend(['anxious', 'overwhelmed'])
            elif 'stress' in col_lower:
                emotions.extend(['stressed', 'frustrated'])
        
        # Remove duplicates and limit
        emotions = list(set(emotions))[:3]
        
        # Ensure at least one emotion
        if not emotions:
            emotions = ['content']
        
        return emotions
    
    def _extract_student_stress_level(self, row):
        """Extract stress level specific to student data"""
        # Students typically have higher stress levels
        base_stress = np.random.randint(4, 8)
        
        # Check for academic stress indicators
        academic_cols = [col for col in row.index if any(term in col.lower() for term in ['academic', 'study', 'exam', 'grade'])]
        if academic_cols:
            base_stress += 1
        
        return min(10, base_stress)
    
    def _extract_student_triggers(self, row):
        """Extract triggers specific to student data"""
        triggers = []
        
        # Common student triggers
        if any(col for col in row.index if any(term in col.lower() for term in ['academic', 'study', 'exam'])):
            triggers.append('work_stress')  # Academic stress maps to work stress
        
        if any(col for col in row.index if 'financial' in col.lower()):
            triggers.append('financial_worry')
        
        if any(col for col in row.index if any(term in col.lower() for term in ['social', 'peer', 'friend'])):
            triggers.append('social_isolation')
        
        # Default student triggers
        if not triggers:
            triggers = ['work_stress', 'sleep_deprivation']
        
        return triggers[:3]
    
    def combine_all_processed_data(self):
        """Combine all processed datasets into one comprehensive dataset"""
        print(f"\n{Fore.BLUE}🔄 Combining all processed datasets...{Style.RESET_ALL}")
        
        all_data = []
        
        # Load all processed files
        processed_files = list(self.output_dir.glob('*_processed.json'))
        
        for file in processed_files:
            try:
                with open(file, 'r') as f:
                    data = json.load(f)
                    all_data.extend(data)
                    print(f"{Fore.GREEN}✅ Added {len(data)} entries from {file.name}{Style.RESET_ALL}")
            except Exception as e:
                print(f"{Fore.RED}❌ Error reading {file.name}: {e}{Style.RESET_ALL}")
        
        if all_data:
            # Sort by timestamp
            all_data.sort(key=lambda x: x['timestamp'])
            
            # Save combined dataset
            combined_file = self.output_dir / 'combined_public_datasets.json'
            with open(combined_file, 'w') as f:
                json.dump(all_data, f, indent=2)
            
            print(f"\n{Fore.GREEN}🎉 Combined dataset created with {len(all_data)} total entries{Style.RESET_ALL}")
            print(f"{Fore.GREEN}📁 Saved to: {combined_file}{Style.RESET_ALL}")
            
            return all_data
        else:
            print(f"{Fore.RED}❌ No data to combine{Style.RESET_ALL}")
            return None
    
    def generate_summary_report(self, data):
        """Generate a summary report of the processed data"""
        if not data:
            print(f"{Fore.RED}❌ No data to analyze{Style.RESET_ALL}")
            return
        
        print(f"\n{Fore.CYAN}📊 Dataset Summary Report{Style.RESET_ALL}")
        print(f"{Fore.CYAN}{'='*50}{Style.RESET_ALL}")
        
        # Basic statistics
        df = pd.DataFrame(data)
        
        print(f"{Fore.WHITE}Total Entries: {len(data)}{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Date Range: {len(df['timestamp'].apply(lambda x: x.split('T')[0]).unique())} unique days{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Average Mood Score: {df['mood_score'].mean():.2f}/10{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Average Stress Level: {df['stress_level'].mean():.2f}/10{Style.RESET_ALL}")
        print(f"{Fore.WHITE}Average Sleep Hours: {df['sleep_hours'].mean():.2f}{Style.RESET_ALL}")
        
        # Most common emotions
        all_emotions = []
        for entry in data:
            all_emotions.extend(entry['emotions'])
        
        emotion_counts = pd.Series(all_emotions).value_counts()
        print(f"\n{Fore.CYAN}🎭 Most Common Emotions:{Style.RESET_ALL}")
        for emotion, count in emotion_counts.head(5).items():
            print(f"{Fore.WHITE}  • {emotion}: {count} times{Style.RESET_ALL}")
        
        # Most common triggers
        all_triggers = []
        for entry in data:
            all_triggers.extend(entry['triggers'])
        
        trigger_counts = pd.Series(all_triggers).value_counts()
        print(f"\n{Fore.CYAN}⚡ Most Common Triggers:{Style.RESET_ALL}")
        for trigger, count in trigger_counts.head(5).items():
            print(f"{Fore.WHITE}  • {trigger}: {count} times{Style.RESET_ALL}")

def main():
    """Main function to process public datasets"""
    print(f"\n{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}🧠 Mental Health Mood Tracker - Public Dataset Processor{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    
    processor = PublicDatasetProcessor()
    
    # Check available datasets
    available_datasets = processor.check_available_datasets()
    
    if not available_datasets:
        print(f"\n{Fore.RED}❌ No datasets found in the raw data directory{Style.RESET_ALL}")
        print(f"{Fore.YELLOW}📋 Please download datasets from the provided URLs first{Style.RESET_ALL}")
        return
    
    # Process available datasets
    all_processed_data = []
    
    if 'mental_health_tech_survey.csv' in available_datasets:
        tech_data = processor.process_mental_health_tech_survey()
        if tech_data:
            all_processed_data.extend(tech_data)
    
    if 'student_mental_health.csv' in available_datasets:
        student_data = processor.process_student_mental_health()
        if student_data:
            all_processed_data.extend(student_data)
    
    # Combine all processed data
    if all_processed_data:
        combined_data = processor.combine_all_processed_data()
        if combined_data:
            processor.generate_summary_report(combined_data)
    
    print(f"\n{Fore.GREEN}🎉 Public dataset processing complete!{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}📁 Processed data saved in: data/public_datasets/processed/{Style.RESET_ALL}")

if __name__ == "__main__":
    main() 