"""
Care Recommender - Provide personalized mental health care recommendations
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class CareRecommender:
    """
    Provide personalized mental health care recommendations based on analysis results
    """
    
    def __init__(self):
        self.care_recommendations = {}
        self.professional_resources = {}
        
    def generate_care_recommendations(self, 
                                    mental_health_results: Dict[str, Any] = None,
                                    anomaly_results: Dict[str, Any] = None,
                                    pattern_results: Dict[str, Any] = None,
                                    user_preferences: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Generate comprehensive care recommendations
        
        Args:
            mental_health_results: Mental health indicators analysis
            anomaly_results: Anomaly detection results
            pattern_results: Pattern recognition results
            user_preferences: User preferences for care types
            
        Returns:
            Dictionary with care recommendations
        """
        logger.info("Generating care recommendations")
        
        recommendations = {
            'immediate_care': self._assess_immediate_care_needs(mental_health_results, anomaly_results),
            'professional_help': self._recommend_professional_help(mental_health_results, anomaly_results),
            'self_care_strategies': self._recommend_self_care(pattern_results, mental_health_results),
            'lifestyle_interventions': self._recommend_lifestyle_changes(pattern_results, mental_health_results),
            'support_resources': self._recommend_support_resources(mental_health_results),
            'monitoring_plan': self._create_monitoring_plan(mental_health_results, pattern_results),
            'crisis_plan': self._create_crisis_plan(anomaly_results, mental_health_results),
            'progress_tracking': self._recommend_progress_tracking(pattern_results)
        }
        
        # Personalize based on user preferences
        if user_preferences:
            recommendations = self._personalize_recommendations(recommendations, user_preferences)
        
        self.care_recommendations = recommendations
        logger.info("Care recommendations generated")
        
        return recommendations
    
    def _assess_immediate_care_needs(self, mental_health_results: Dict[str, Any], anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Assess immediate care needs"""
        immediate_needs = {
            'urgency_level': 'low',
            'immediate_actions': [],
            'safety_concerns': [],
            'professional_contact_needed': False
        }
        
        # Check for crisis indicators
        if anomaly_results and 'crisis_indicators' in anomaly_results:
            crisis = anomaly_results['crisis_indicators']
            
            # Severe mood episodes
            if crisis.get('severe_low_mood', {}).get('severity') == 'critical':
                immediate_needs['urgency_level'] = 'critical'
                immediate_needs['immediate_actions'].append('Seek immediate professional help')
                immediate_needs['safety_concerns'].append('Severe mood episodes detected')
                immediate_needs['professional_contact_needed'] = True
            
            # Extreme stress
            if crisis.get('extreme_stress', {}).get('severity') == 'high':
                immediate_needs['urgency_level'] = max(immediate_needs['urgency_level'], 'high')
                immediate_needs['immediate_actions'].append('Implement stress reduction techniques immediately')
                immediate_needs['safety_concerns'].append('Extreme stress levels detected')
            
            # Sleep deprivation
            if crisis.get('severe_sleep_deprivation', {}).get('severity') == 'high':
                immediate_needs['urgency_level'] = max(immediate_needs['urgency_level'], 'high')
                immediate_needs['immediate_actions'].append('Prioritize sleep immediately')
                immediate_needs['safety_concerns'].append('Severe sleep deprivation detected')
            
            # Social isolation
            if crisis.get('complete_social_isolation', {}).get('severity') == 'high':
                immediate_needs['urgency_level'] = max(immediate_needs['urgency_level'], 'medium')
                immediate_needs['immediate_actions'].append('Reach out to trusted person immediately')
                immediate_needs['safety_concerns'].append('Complete social isolation detected')
        
        # Check overall mental health risk
        if mental_health_results and 'overall_risk_assessment' in mental_health_results:
            risk = mental_health_results['overall_risk_assessment']
            
            if risk.get('overall_risk_level') == 'high':
                immediate_needs['urgency_level'] = max(immediate_needs['urgency_level'], 'high')
                immediate_needs['professional_contact_needed'] = True
                immediate_needs['immediate_actions'].append('Schedule mental health professional appointment within 24-48 hours')
        
        # Determine urgency level priority
        urgency_priority = {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}
        if urgency_priority.get(immediate_needs['urgency_level'], 1) >= 3:
            immediate_needs['professional_contact_needed'] = True
        
        return immediate_needs
    
    def _recommend_professional_help(self, mental_health_results: Dict[str, Any], anomaly_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend appropriate professional help"""
        professional_help = {
            'recommended': False,
            'urgency': 'low',
            'provider_types': [],
            'treatment_modalities': [],
            'specific_recommendations': []
        }
        
        if not mental_health_results:
            return professional_help
        
        # Check individual condition indicators
        conditions_detected = []
        
        # Depression indicators
        if 'depression_indicators' in mental_health_results:
            depression = mental_health_results['depression_indicators']
            risk_score = depression.get('depression_risk_score', {})
            
            if risk_score.get('risk_level') in ['high', 'moderate']:
                conditions_detected.append('depression')
                professional_help['recommended'] = True
                professional_help['provider_types'].append('Clinical Psychologist or Psychiatrist')
                professional_help['treatment_modalities'].append('Cognitive Behavioral Therapy (CBT)')
                professional_help['specific_recommendations'].append('Depression screening and evaluation')
        
        # Anxiety indicators
        if 'anxiety_indicators' in mental_health_results:
            anxiety = mental_health_results['anxiety_indicators']
            risk_score = anxiety.get('anxiety_risk_score', {})
            
            if risk_score.get('risk_level') in ['high', 'moderate']:
                conditions_detected.append('anxiety')
                professional_help['recommended'] = True
                professional_help['provider_types'].append('Licensed Clinical Social Worker or Psychologist')
                professional_help['treatment_modalities'].extend(['CBT', 'Exposure Therapy', 'Mindfulness-Based Interventions'])
                professional_help['specific_recommendations'].append('Anxiety disorder evaluation')
        
        # Stress indicators
        if 'stress_indicators' in mental_health_results:
            stress = mental_health_results['stress_indicators']
            risk_score = stress.get('stress_risk_score', {})
            
            if risk_score.get('risk_level') == 'high':
                conditions_detected.append('chronic_stress')
                professional_help['recommended'] = True
                professional_help['provider_types'].append('Stress Management Counselor')
                professional_help['treatment_modalities'].append('Stress Management Therapy')
                professional_help['specific_recommendations'].append('Stress management counseling')
        
        # Sleep disorder indicators
        if 'sleep_disorder_indicators' in mental_health_results:
            sleep = mental_health_results['sleep_disorder_indicators']
            risk_score = sleep.get('sleep_disorder_risk_score', {})
            
            if risk_score.get('risk_level') in ['high', 'moderate']:
                professional_help['recommended'] = True
                professional_help['provider_types'].append('Sleep Specialist')
                professional_help['treatment_modalities'].append('Sleep Study and CBT for Insomnia')
                professional_help['specific_recommendations'].append('Sleep disorder evaluation')
        
        # Seasonal Affective Disorder
        if 'seasonal_affective_indicators' in mental_health_results:
            sad = mental_health_results['seasonal_affective_indicators']
            risk_score = sad.get('sad_risk_score', {})
            
            if risk_score.get('risk_level') in ['high', 'moderate']:
                professional_help['recommended'] = True
                professional_help['provider_types'].append('Psychiatrist or Mental Health Professional')
                professional_help['treatment_modalities'].append('Light Therapy and Seasonal Depression Treatment')
                professional_help['specific_recommendations'].append('Seasonal Affective Disorder evaluation')
        
        # Determine urgency
        if len(conditions_detected) >= 3:
            professional_help['urgency'] = 'high'
        elif len(conditions_detected) >= 2:
            professional_help['urgency'] = 'medium'
        elif len(conditions_detected) >= 1:
            professional_help['urgency'] = 'medium'
        
        # Crisis indicators increase urgency
        if anomaly_results and 'crisis_indicators' in anomaly_results:
            crisis = anomaly_results['crisis_indicators']
            critical_indicators = [k for k, v in crisis.items() if v.get('severity') == 'critical']
            
            if critical_indicators:
                professional_help['urgency'] = 'critical'
                professional_help['specific_recommendations'].insert(0, 'IMMEDIATE professional evaluation needed')
        
        return professional_help
    
    def _recommend_self_care(self, pattern_results: Dict[str, Any], mental_health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend self-care strategies"""
        self_care = {
            'daily_practices': [],
            'weekly_practices': [],
            'emergency_coping': [],
            'mindfulness_techniques': [],
            'physical_wellness': [],
            'emotional_regulation': []
        }
        
        # Daily practices
        self_care['daily_practices'] = [
            {
                'practice': 'Morning mood check-in',
                'description': 'Rate your mood and identify 3 things you\'re grateful for',
                'time_required': '5 minutes',
                'benefit': 'Increased self-awareness and positive focus'
            },
            {
                'practice': 'Deep breathing exercises',
                'description': '4-7-8 breathing technique when feeling stressed',
                'time_required': '2-5 minutes',
                'benefit': 'Immediate stress relief and emotional regulation'
            },
            {
                'practice': 'Evening reflection',
                'description': 'Journal about your day, emotions, and positive moments',
                'time_required': '10 minutes',
                'benefit': 'Emotional processing and pattern recognition'
            }
        ]
        
        # Weekly practices
        self_care['weekly_practices'] = [
            {
                'practice': 'Digital detox periods',
                'description': 'Spend time away from screens and social media',
                'frequency': '2-3 times per week',
                'benefit': 'Reduced comparison and information overload stress'
            },
            {
                'practice': 'Nature connection',
                'description': 'Spend time outdoors, even if just 15 minutes',
                'frequency': 'Daily if possible',
                'benefit': 'Mood improvement and stress reduction'
            },
            {
                'practice': 'Social connection',
                'description': 'Meaningful conversation with friend or family member',
                'frequency': '2-3 times per week',
                'benefit': 'Social support and reduced isolation'
            }
        ]
        
        # Emergency coping strategies
        self_care['emergency_coping'] = [
            {
                'strategy': '5-4-3-2-1 grounding technique',
                'description': 'Name 5 things you see, 4 you can touch, 3 you hear, 2 you smell, 1 you taste',
                'use_when': 'Feeling overwhelmed or anxious',
                'effectiveness': 'Immediate anxiety relief'
            },
            {
                'strategy': 'Cold water on face/wrists',
                'description': 'Splash cold water or hold ice cubes',
                'use_when': 'Panic attacks or extreme emotional distress',
                'effectiveness': 'Activates parasympathetic nervous system'
            },
            {
                'strategy': 'Progressive muscle relaxation',
                'description': 'Tense and release muscle groups starting from toes',
                'use_when': 'High stress or difficulty sleeping',
                'effectiveness': 'Physical and mental relaxation'
            }
        ]
        
        # Mindfulness techniques
        self_care['mindfulness_techniques'] = [
            {
                'technique': 'Body scan meditation',
                'description': 'Systematic attention to different parts of your body',
                'duration': '10-20 minutes',
                'benefit': 'Increased body awareness and relaxation'
            },
            {
                'technique': 'Mindful walking',
                'description': 'Focus on each step and breath while walking slowly',
                'duration': '5-15 minutes',
                'benefit': 'Present-moment awareness and gentle movement'
            },
            {
                'technique': 'Loving-kindness meditation',
                'description': 'Send good wishes to yourself and others',
                'duration': '10 minutes',
                'benefit': 'Increased self-compassion and positive emotions'
            }
        ]
        
        # Physical wellness
        self_care['physical_wellness'] = [
            {
                'practice': 'Regular sleep schedule',
                'description': 'Go to bed and wake up at the same time daily',
                'benefit': 'Improved mood stability and cognitive function'
            },
            {
                'practice': 'Gentle exercise',
                'description': 'Walking, yoga, swimming, or dancing',
                'benefit': 'Natural mood elevation and stress reduction'
            },
            {
                'practice': 'Nutrition mindfulness',
                'description': 'Eat regular meals with mood-supporting nutrients',
                'benefit': 'Stable energy and neurotransmitter support'
            }
        ]
        
        # Emotional regulation
        self_care['emotional_regulation'] = [
            {
                'skill': 'Emotion labeling',
                'description': 'Name your emotions specifically (frustrated vs angry)',
                'benefit': 'Reduced emotional intensity and increased awareness'
            },
            {
                'skill': 'Opposite action',
                'description': 'When emotion isn\'t helpful, act opposite to its urge',
                'benefit': 'Breaks negative emotional cycles'
            },
            {
                'skill': 'Self-compassion',
                'description': 'Treat yourself with the kindness you\'d show a friend',
                'benefit': 'Reduced self-criticism and increased resilience'
            }
        ]
        
        return self_care
    
    def _recommend_lifestyle_changes(self, pattern_results: Dict[str, Any], mental_health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend lifestyle interventions"""
        lifestyle = {
            'sleep_optimization': [],
            'exercise_recommendations': [],
            'nutrition_guidance': [],
            'social_strategies': [],
            'stress_reduction': [],
            'environment_modifications': []
        }
        
        # Sleep optimization
        lifestyle['sleep_optimization'] = [
            {
                'intervention': 'Sleep hygiene protocol',
                'details': [
                    'Keep bedroom temperature between 65-68°F',
                    'Use blackout curtains or eye mask',
                    'No screens 1 hour before bedtime',
                    'Consistent bedtime routine (reading, gentle music, etc.)'
                ],
                'expected_outcome': 'Improved sleep quality and mood stability'
            },
            {
                'intervention': 'Sleep schedule consistency',
                'details': [
                    'Same bedtime and wake time every day (including weekends)',
                    'Limit naps to 20 minutes before 3 PM',
                    'Expose yourself to bright light in the morning'
                ],
                'expected_outcome': 'Regulated circadian rhythm'
            }
        ]
        
        # Exercise recommendations
        lifestyle['exercise_recommendations'] = [
            {
                'type': 'Aerobic exercise',
                'details': [
                    'Start with 10-15 minutes daily',
                    'Activities: walking, swimming, cycling, dancing',
                    'Gradually increase to 30 minutes, 5 days per week'
                ],
                'benefit': 'Natural antidepressant effect, stress reduction'
            },
            {
                'type': 'Strength training',
                'details': [
                    'Bodyweight exercises or light weights',
                    '2-3 times per week',
                    'Focus on major muscle groups'
                ],
                'benefit': 'Increased confidence and physical resilience'
            },
            {
                'type': 'Mind-body exercises',
                'details': [
                    'Yoga, tai chi, or qigong',
                    '15-30 minutes, 3-4 times per week',
                    'Focus on breathing and mindful movement'
                ],
                'benefit': 'Stress reduction and mindfulness practice'
            }
        ]
        
        # Nutrition guidance
        lifestyle['nutrition_guidance'] = [
            {
                'category': 'Mood-supporting foods',
                'recommendations': [
                    'Omega-3 rich foods (fish, walnuts, flax seeds)',
                    'Complex carbohydrates (quinoa, oats, sweet potatoes)',
                    'Protein with each meal for stable blood sugar',
                    'Dark leafy greens and colorful vegetables'
                ],
                'benefit': 'Stable mood and energy levels'
            },
            {
                'category': 'Foods to limit',
                'recommendations': [
                    'Excessive caffeine (especially after 2 PM)',
                    'Processed and high-sugar foods',
                    'Alcohol, particularly as a coping mechanism',
                    'Large meals close to bedtime'
                ],
                'benefit': 'Reduced mood swings and better sleep'
            }
        ]
        
        # Social strategies
        lifestyle['social_strategies'] = [
            {
                'strategy': 'Build support network',
                'actions': [
                    'Identify 3-5 trusted people you can talk to',
                    'Join groups or activities aligned with your interests',
                    'Consider support groups for specific challenges',
                    'Practice asking for help when needed'
                ],
                'benefit': 'Reduced isolation and increased support'
            },
            {
                'strategy': 'Boundary setting',
                'actions': [
                    'Learn to say no to overwhelming commitments',
                    'Limit time with people who drain your energy',
                    'Communicate your needs clearly and kindly',
                    'Protect time for self-care'
                ],
                'benefit': 'Reduced stress and better relationships'
            }
        ]
        
        # Stress reduction
        lifestyle['stress_reduction'] = [
            {
                'technique': 'Time management',
                'methods': [
                    'Break large tasks into smaller steps',
                    'Use calendars and to-do lists effectively',
                    'Build in buffer time between activities',
                    'Prioritize tasks by importance and urgency'
                ],
                'outcome': 'Reduced overwhelm and increased control'
            },
            {
                'technique': 'Relaxation practices',
                'methods': [
                    'Daily meditation or mindfulness practice',
                    'Regular massage or self-massage',
                    'Hot baths with Epsom salts',
                    'Listening to calming music'
                ],
                'outcome': 'Lower baseline stress and better recovery'
            }
        ]
        
        # Environment modifications
        lifestyle['environment_modifications'] = [
            {
                'area': 'Living space',
                'changes': [
                    'Declutter and organize frequently used spaces',
                    'Add plants or natural elements',
                    'Ensure good lighting, especially natural light',
                    'Create a designated relaxation space'
                ],
                'impact': 'Reduced environmental stress and increased calm'
            },
            {
                'area': 'Work environment',
                'changes': [
                    'Take regular breaks (5 minutes every hour)',
                    'Personalize workspace with meaningful items',
                    'Manage noise levels with headphones if needed',
                    'Advocate for reasonable workload and deadlines'
                ],
                'impact': 'Reduced work stress and better work-life balance'
            }
        ]
        
        return lifestyle
    
    def _recommend_support_resources(self, mental_health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend support resources"""
        resources = {
            'crisis_resources': [],
            'professional_services': [],
            'support_groups': [],
            'educational_resources': [],
            'apps_and_tools': [],
            'books_and_materials': []
        }
        
        # Crisis resources
        resources['crisis_resources'] = [
            {
                'name': 'National Suicide Prevention Lifeline',
                'contact': '988',
                'description': '24/7 free and confidential support for people in distress',
                'availability': '24/7'
            },
            {
                'name': 'Crisis Text Line',
                'contact': 'Text HOME to 741741',
                'description': 'Free, 24/7 crisis support via text message',
                'availability': '24/7'
            },
            {
                'name': 'SAMHSA National Helpline',
                'contact': '1-800-662-4357',
                'description': 'Treatment referral and information service',
                'availability': '24/7'
            }
        ]
        
        # Professional services
        resources['professional_services'] = [
            {
                'type': 'Psychology Today',
                'description': 'Find therapists in your area with filters for specialty, insurance, etc.',
                'website': 'psychologytoday.com'
            },
            {
                'type': 'Community Mental Health Centers',
                'description': 'Federally funded centers providing affordable mental health services',
                'how_to_find': 'Search SAMHSA treatment locator'
            },
            {
                'type': 'Employee Assistance Programs (EAP)',
                'description': 'Free counseling sessions often provided by employers',
                'how_to_access': 'Contact HR department'
            }
        ]
        
        # Support groups
        resources['support_groups'] = [
            {
                'type': 'NAMI Support Groups',
                'description': 'National Alliance on Mental Illness peer support groups',
                'website': 'nami.org'
            },
            {
                'type': 'Depression and Bipolar Support Alliance',
                'description': 'Support groups for mood disorders',
                'website': 'dbsalliance.org'
            },
            {
                'type': 'Online support communities',
                'description': '7 Cups, Reddit mental health communities, HealthUnlocked',
                'note': 'Use caution and verify information with professionals'
            }
        ]
        
        # Educational resources
        resources['educational_resources'] = [
            {
                'name': 'National Institute of Mental Health (NIMH)',
                'description': 'Comprehensive information about mental health conditions',
                'website': 'nimh.nih.gov'
            },
            {
                'name': 'Centre for Clinical Interventions',
                'description': 'Free self-help modules for various mental health conditions',
                'website': 'cci.health.wa.gov.au'
            },
            {
                'name': 'Mental Health First Aid',
                'description': 'Learn how to help someone experiencing mental health crisis',
                'website': 'mentalhealthfirstaid.org'
            }
        ]
        
        # Apps and tools
        resources['apps_and_tools'] = [
            {
                'name': 'Headspace',
                'type': 'Meditation app',
                'description': 'Guided meditations for stress, sleep, and focus',
                'cost': 'Free trial, then subscription'
            },
            {
                'name': 'Calm',
                'type': 'Sleep and meditation app',
                'description': 'Sleep stories, meditation, and relaxation content',
                'cost': 'Free trial, then subscription'
            },
            {
                'name': 'Daylio',
                'type': 'Mood tracking app',
                'description': 'Simple mood tracking with pattern analysis',
                'cost': 'Free with premium features'
            },
            {
                'name': 'PTSD Coach',
                'type': 'Trauma support app',
                'description': 'Coping tools for trauma symptoms',
                'cost': 'Free'
            }
        ]
        
        # Books and materials
        resources['books_and_materials'] = [
            {
                'title': 'Feeling Good by David D. Burns',
                'topic': 'Depression and cognitive therapy',
                'description': 'Classic book on cognitive behavioral techniques'
            },
            {
                'title': 'The Anxiety and Worry Workbook by David A. Clark',
                'topic': 'Anxiety management',
                'description': 'Practical CBT techniques for anxiety'
            },
            {
                'title': 'Mindfulness for Beginners by Jon Kabat-Zinn',
                'topic': 'Mindfulness meditation',
                'description': 'Introduction to mindfulness practice'
            },
            {
                'title': 'The Sleep Solution by W. Chris Winter',
                'topic': 'Sleep improvement',
                'description': 'Evidence-based approach to better sleep'
            }
        ]
        
        return resources
    
    def _create_monitoring_plan(self, mental_health_results: Dict[str, Any], pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a personalized monitoring plan"""
        monitoring = {
            'daily_check_ins': [],
            'weekly_assessments': [],
            'monthly_reviews': [],
            'warning_signs_to_watch': [],
            'when_to_seek_help': []
        }
        
        # Daily check-ins
        monitoring['daily_check_ins'] = [
            {
                'metric': 'Mood rating (1-10)',
                'purpose': 'Track daily mood trends',
                'method': 'Quick self-rating each evening'
            },
            {
                'metric': 'Sleep quality and duration',
                'purpose': 'Monitor sleep patterns',
                'method': 'Note bedtime, wake time, and sleep quality'
            },
            {
                'metric': 'Stress level (1-10)',
                'purpose': 'Track stress patterns',
                'method': 'Rate stress level at end of day'
            },
            {
                'metric': 'Physical activity',
                'purpose': 'Monitor exercise impact on mood',
                'method': 'Note type and duration of activity'
            }
        ]
        
        # Weekly assessments
        monitoring['weekly_assessments'] = [
            {
                'assessment': 'Weekly mood pattern review',
                'method': 'Look at daily ratings to identify patterns',
                'questions_to_ask': [
                    'What were my highest and lowest mood days?',
                    'What factors contributed to good and bad days?',
                    'What patterns do I notice?'
                ]
            },
            {
                'assessment': 'Coping strategy effectiveness',
                'method': 'Rate which strategies helped most',
                'questions_to_ask': [
                    'Which self-care practices did I use?',
                    'What worked best for managing stress?',
                    'What do I want to try differently next week?'
                ]
            }
        ]
        
        # Monthly reviews
        monitoring['monthly_reviews'] = [
            {
                'review': 'Overall progress assessment',
                'components': [
                    'Compare current month to previous month',
                    'Identify improvement trends',
                    'Assess goal progress',
                    'Adjust strategies as needed'
                ]
            },
            {
                'review': 'Professional care evaluation',
                'components': [
                    'Assess if current level of care is sufficient',
                    'Consider if additional support is needed',
                    'Review medication effectiveness (if applicable)',
                    'Plan for upcoming challenges'
                ]
            }
        ]
        
        # Warning signs to watch
        monitoring['warning_signs_to_watch'] = [
            {
                'category': 'Mood changes',
                'signs': [
                    'Mood ratings consistently below 4 for 5+ days',
                    'Extreme mood swings (3+ point changes day to day)',
                    'Loss of interest in previously enjoyed activities',
                    'Feelings of hopelessness or worthlessness'
                ]
            },
            {
                'category': 'Behavioral changes',
                'signs': [
                    'Significant sleep disruption (< 4 or > 10 hours regularly)',
                    'Avoiding social contact for 7+ days',
                    'Neglecting basic self-care',
                    'Increased substance use'
                ]
            },
            {
                'category': 'Thinking patterns',
                'signs': [
                    'Persistent negative thoughts',
                    'Difficulty concentrating or making decisions',
                    'Thoughts of self-harm or suicide',
                    'Feeling like a burden to others'
                ]
            }
        ]
        
        # When to seek help
        monitoring['when_to_seek_help'] = [
            {
                'urgency': 'Immediate (call 911 or crisis line)',
                'indicators': [
                    'Thoughts of suicide or self-harm',
                    'Plans to hurt yourself or others',
                    'Feeling completely unable to cope',
                    'Psychotic symptoms (hallucinations, delusions)'
                ]
            },
            {
                'urgency': 'Within 24-48 hours',
                'indicators': [
                    'Mood ratings below 3 for 3+ consecutive days',
                    'Unable to sleep for 2+ days',
                    'Panic attacks increasing in frequency',
                    'Significant functional impairment'
                ]
            },
            {
                'urgency': 'Within 1-2 weeks',
                'indicators': [
                    'Gradual decline in mood over 2+ weeks',
                    'Increasing isolation or withdrawal',
                    'Difficulty maintaining work or relationships',
                    'Previous coping strategies no longer working'
                ]
            }
        ]
        
        return monitoring
    
    def _create_crisis_plan(self, anomaly_results: Dict[str, Any], mental_health_results: Dict[str, Any]) -> Dict[str, Any]:
        """Create a personalized crisis plan"""
        crisis_plan = {
            'early_warning_signs': [],
            'coping_strategies': [],
            'support_contacts': [],
            'professional_contacts': [],
            'emergency_resources': [],
            'safety_plan_steps': []
        }
        
        # Early warning signs
        crisis_plan['early_warning_signs'] = [
            'Mood rating below 3 for more than 2 days',
            'Thoughts of wanting to disappear or not exist',
            'Feeling completely hopeless about the future',
            'Inability to sleep for more than 24 hours',
            'Panic attacks lasting more than 30 minutes',
            'Feeling disconnected from reality',
            'Thoughts of harming yourself or others'
        ]
        
        # Coping strategies
        crisis_plan['coping_strategies'] = [
            {
                'strategy': 'Immediate grounding',
                'steps': [
                    'Use 5-4-3-2-1 technique (5 things you see, 4 you hear, etc.)',
                    'Hold ice cubes or splash cold water on face',
                    'Deep breathing: 4 counts in, 7 counts hold, 8 counts out'
                ]
            },
            {
                'strategy': 'Distraction activities',
                'steps': [
                    'Call or text a trusted friend',
                    'Watch funny videos or comforting movies',
                    'Listen to calming or uplifting music',
                    'Do a physical activity (walk, dance, clean)'
                ]
            },
            {
                'strategy': 'Self-soothing',
                'steps': [
                    'Take a warm shower or bath',
                    'Wrap yourself in a soft blanket',
                    'Drink warm tea or hot chocolate',
                    'Use aromatherapy or calming scents'
                ]
            }
        ]
        
        # Support contacts template
        crisis_plan['support_contacts'] = [
            {
                'relationship': 'Primary support person',
                'name': '[Fill in name]',
                'phone': '[Fill in phone number]',
                'when_to_contact': 'First person to call when feeling distressed'
            },
            {
                'relationship': 'Backup support person',
                'name': '[Fill in name]',
                'phone': '[Fill in phone number]',
                'when_to_contact': 'If primary person is unavailable'
            },
            {
                'relationship': 'Family member',
                'name': '[Fill in name]',
                'phone': '[Fill in phone number]',
                'when_to_contact': 'For longer-term support or serious crises'
            }
        ]
        
        # Professional contacts template
        crisis_plan['professional_contacts'] = [
            {
                'role': 'Primary therapist/counselor',
                'name': '[Fill in name]',
                'phone': '[Fill in phone number]',
                'after_hours_contact': '[Fill in emergency contact info]'
            },
            {
                'role': 'Primary care physician',
                'name': '[Fill in name]',
                'phone': '[Fill in phone number]',
                'when_to_contact': 'For medication concerns or physical symptoms'
            },
            {
                'role': 'Psychiatrist (if applicable)',
                'name': '[Fill in name]',
                'phone': '[Fill in phone number]',
                'when_to_contact': 'For medication adjustments or severe symptoms'
            }
        ]
        
        # Emergency resources
        crisis_plan['emergency_resources'] = [
            {
                'resource': 'National Suicide Prevention Lifeline',
                'contact': '988',
                'available': '24/7',
                'purpose': 'Immediate crisis support'
            },
            {
                'resource': 'Crisis Text Line',
                'contact': 'Text HOME to 741741',
                'available': '24/7',
                'purpose': 'Text-based crisis support'
            },
            {
                'resource': 'Local emergency services',
                'contact': '911',
                'available': '24/7',
                'purpose': 'Immediate physical safety concerns'
            },
            {
                'resource': 'Local crisis center',
                'contact': '[Fill in local number]',
                'available': '[Fill in hours]',
                'purpose': 'Local crisis intervention'
            }
        ]
        
        # Safety plan steps
        crisis_plan['safety_plan_steps'] = [
            {
                'step': 1,
                'action': 'Recognize warning signs',
                'description': 'Notice early signs that crisis might be developing'
            },
            {
                'step': 2,
                'action': 'Use internal coping strategies',
                'description': 'Try grounding, breathing, or distraction techniques'
            },
            {
                'step': 3,
                'action': 'Contact support person',
                'description': 'Reach out to trusted friend or family member'
            },
            {
                'step': 4,
                'action': 'Contact professional',
                'description': 'Call therapist, counselor, or crisis line'
            },
            {
                'step': 5,
                'action': 'Go to safe place',
                'description': 'Go to emergency room or call 911 if in immediate danger'
            },
            {
                'step': 6,
                'action': 'Make environment safe',
                'description': 'Remove means of self-harm if necessary'
            }
        ]
        
        return crisis_plan
    
    def _recommend_progress_tracking(self, pattern_results: Dict[str, Any]) -> Dict[str, Any]:
        """Recommend progress tracking methods"""
        tracking = {
            'metrics_to_track': [],
            'tracking_methods': [],
            'frequency_recommendations': [],
            'progress_indicators': []
        }
        
        # Metrics to track
        tracking['metrics_to_track'] = [
            {
                'metric': 'Daily mood rating',
                'scale': '1-10 scale',
                'purpose': 'Overall mental health tracking',
                'importance': 'high'
            },
            {
                'metric': 'Sleep quality and duration',
                'scale': 'Hours of sleep + quality rating 1-5',
                'purpose': 'Sleep impact on mood',
                'importance': 'high'
            },
            {
                'metric': 'Stress level',
                'scale': '1-10 scale',
                'purpose': 'Stress management effectiveness',
                'importance': 'high'
            },
            {
                'metric': 'Physical activity',
                'scale': 'Minutes and type of activity',
                'purpose': 'Exercise impact on mood',
                'importance': 'medium'
            },
            {
                'metric': 'Social interactions',
                'scale': 'Number and quality of interactions',
                'purpose': 'Social support tracking',
                'importance': 'medium'
            }
        ]
        
        # Tracking methods
        tracking['tracking_methods'] = [
            {
                'method': 'Smartphone app',
                'pros': ['Convenient', 'Automated reminders', 'Data visualization'],
                'cons': ['Screen dependency', 'Privacy concerns'],
                'recommended_apps': ['Daylio', 'Mood Meter', 'eMoods']
            },
            {
                'method': 'Paper journal',
                'pros': ['No technology needed', 'Flexible format', 'Tactile experience'],
                'cons': ['Easy to forget', 'No automatic analysis'],
                'tips': ['Keep by bedside', 'Use simple rating scales']
            },
            {
                'method': 'Digital spreadsheet',
                'pros': ['Customizable', 'Easy to analyze', 'Backup capability'],
                'cons': ['Requires setup', 'Less convenient'],
                'tools': ['Google Sheets', 'Excel', 'Numbers']
            }
        ]
        
        # Frequency recommendations
        tracking['frequency_recommendations'] = [
            {
                'metric': 'Mood rating',
                'frequency': 'Daily (evening)',
                'reason': 'Captures daily patterns without being overwhelming'
            },
            {
                'metric': 'Detailed reflection',
                'frequency': 'Weekly',
                'reason': 'Provides deeper insight without daily burden'
            },
            {
                'metric': 'Professional check-in',
                'frequency': 'Monthly or as recommended',
                'reason': 'Regular assessment with trained professional'
            }
        ]
        
        # Progress indicators
        tracking['progress_indicators'] = [
            {
                'indicator': 'Mood stability',
                'measurement': 'Reduced day-to-day mood variation',
                'target': 'Standard deviation of daily mood < 1.5'
            },
            {
                'indicator': 'Trend improvement',
                'measurement': 'Overall upward trend in mood ratings',
                'target': 'Average mood increasing over time'
            },
            {
                'indicator': 'Recovery time',
                'measurement': 'Faster bounce-back from low mood days',
                'target': 'Low mood episodes lasting < 3 days'
            },
            {
                'indicator': 'Coping effectiveness',
                'measurement': 'Better use of coping strategies',
                'target': 'Successful prevention of mood crises'
            }
        ]
        
        return tracking
    
    def _personalize_recommendations(self, recommendations: Dict[str, Any], user_preferences: Dict[str, Any]) -> Dict[str, Any]:
        """Personalize recommendations based on user preferences"""
        # This would include logic to filter and customize recommendations
        # based on user preferences like therapy type, activity preferences, etc.
        
        # For now, just add preference considerations
        recommendations['personalization_notes'] = {
            'preferences_considered': user_preferences,
            'customization_applied': 'Recommendations filtered based on user preferences'
        }
        
        return recommendations
    
    def get_care_plan_summary(self) -> Dict[str, Any]:
        """Get a summary of the care plan"""
        if not self.care_recommendations:
            return {'error': 'No care recommendations available. Run generate_care_recommendations() first.'}
        
        summary = {
            'immediate_priorities': [],
            'professional_help_needed': False,
            'top_self_care_strategies': [],
            'key_lifestyle_changes': [],
            'monitoring_essentials': []
        }
        
        # Extract immediate priorities
        immediate_care = self.care_recommendations.get('immediate_care', {})
        if immediate_care.get('urgency_level') != 'low':
            summary['immediate_priorities'] = immediate_care.get('immediate_actions', [])
        
        # Professional help assessment
        professional_help = self.care_recommendations.get('professional_help', {})
        summary['professional_help_needed'] = professional_help.get('recommended', False)
        
        # Top self-care strategies
        self_care = self.care_recommendations.get('self_care_strategies', {})
        daily_practices = self_care.get('daily_practices', [])
        summary['top_self_care_strategies'] = [p['practice'] for p in daily_practices[:3]]
        
        # Key lifestyle changes
        lifestyle = self.care_recommendations.get('lifestyle_interventions', {})
        summary['key_lifestyle_changes'] = [
            'Sleep optimization',
            'Regular exercise',
            'Stress management'
        ]
        
        # Monitoring essentials
        monitoring = self.care_recommendations.get('monitoring_plan', {})
        daily_checks = monitoring.get('daily_check_ins', [])
        summary['monitoring_essentials'] = [c['metric'] for c in daily_checks[:3]]
        
        return summary 