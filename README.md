# Mental Health Mood Tracker with Insights

## Overview
A comprehensive system that analyzes daily mood entries and identifies patterns or concerning trends using AI/ML techniques.

## Skills Demonstrated
- **AI/ML**: Time series analysis, pattern recognition, anomaly detection for mood patterns
- **Critical Thinking**: Mental health indicators, professional help recommendations
- **Problem Solving**: Subjective data handling, missing entries, seasonal patterns, privacy concerns
- **Modular Structure**: Separate data collection, trend analysis, pattern detection, and recommendation modules
- **Clear Architecture**: Pipeline from mood entries → trend analysis → pattern recognition → actionable insights

## System Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│  Data Collection│───▶│  Trend Analysis │───▶│Pattern Detection│───▶│ Recommendation  │
│     Module      │    │     Module      │    │     Module      │    │     Module      │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Modules

### 1. Data Collection Module (`data_collection/`)
- Handles daily mood entries
- Manages contextual data (sleep, exercise, stress)
- Validates and stores mood data
- Handles missing entries

### 2. Trend Analysis Module (`trend_analysis/`)
- Time series analysis using ARIMA models
- Moving averages and trend detection
- Seasonal pattern identification
- Statistical analysis of mood patterns

### 3. Pattern Detection Module (`pattern_detection/`)
- Anomaly detection algorithms
- Machine learning pattern recognition
- Mental health indicator identification
- Crisis pattern detection

### 4. Recommendation Module (`recommendations/`)
- Generates actionable insights
- Professional help recommendations
- Crisis intervention protocols
- Personalized care suggestions

## Features
- ✅ Daily mood tracking with contextual factors
- ✅ Time series analysis for trend identification
- ✅ AI/ML pattern recognition
- ✅ Anomaly detection for concerning trends
- ✅ Mental health indicator monitoring
- ✅ Professional care recommendations
- ✅ Privacy-focused data handling
- ✅ Seasonal pattern analysis
- ✅ Missing data handling

## Installation
```bash
pip install -r requirements.txt
```

## Usage
```bash
python main.py
```

## Data Format
The system accepts mood entries in JSON format with fields for:
- Mood score (1-10)
- Emotions (multiple selection)
- Contextual factors (sleep, exercise, stress)
- Timestamps
- Optional notes

## Output
- Trend analysis reports
- Pattern detection results
- Anomaly alerts
- Professional care recommendations
- Visual insights and charts 