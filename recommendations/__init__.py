"""
Recommendations Module for Mental Health Mood Tracker

This module handles:
- Actionable insights generation
- Professional help recommendations
- Crisis intervention protocols
- Personalized care suggestions
"""

from .insight_generator import InsightGenerator
from .care_recommender import CareRecommender
from .crisis_detector import CrisisDetector

__all__ = ['InsightGenerator', 'CareRecommender', 'CrisisDetector'] 