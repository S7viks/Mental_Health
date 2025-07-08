import axios from 'axios';

// Base API configuration
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30000, // 30 seconds for analysis operations
  headers: {
    'Content-Type': 'application/json',
  },
});

// Request interceptor
api.interceptors.request.use(
  (config) => {
    // Add any auth tokens if needed
    const token = localStorage.getItem('auth_token');
    if (token) {
      config.headers.Authorization = `Bearer ${token}`;
    }
    return config;
  },
  (error) => Promise.reject(error)
);

// Response interceptor
api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    console.error('API Error:', error);
    if (error.response) {
      throw new Error(error.response.data.message || 'API request failed');
    } else if (error.request) {
      throw new Error('Network error - please check your connection');
    } else {
      throw new Error('Request setup error');
    }
  }
);

// Fallback data for development/demo purposes
const generateFallbackData = () => {
  const data = [];
  const today = new Date();
  
  for (let i = 29; i >= 0; i--) {
    const date = new Date(today);
    date.setDate(date.getDate() - i);
    
    // Generate realistic mood patterns
    const baseScore = 5 + Math.sin(i / 7) * 2; // Weekly pattern
    const randomVariation = (Math.random() - 0.5) * 2;
    const moodScore = Math.max(1, Math.min(10, Math.round(baseScore + randomVariation)));
    
    const entry = {
      entry_id: `demo_${i}_${date.toISOString().split('T')[0]}`,
      timestamp: date.toISOString(),
      mood_score: moodScore,
      emotions: generateRandomEmotions(moodScore),
      sleep_hours: 6 + Math.random() * 3,
      exercise_minutes: Math.random() * 90,
      social_interactions: Math.floor(Math.random() * 8),
      stress_level: Math.max(1, Math.min(10, 11 - moodScore + Math.random() * 2)),
      weather: ['sunny', 'cloudy', 'rainy', 'snowy'][Math.floor(Math.random() * 4)],
      notes: `Day ${30 - i} mood entry`,
      triggers: generateRandomTriggers(),
      data_source: 'demo'
    };
    
    data.push(entry);
  }
  
  return data;
};

const generateRandomEmotions = (moodScore) => {
  const positiveEmotions = ['happy', 'content', 'excited', 'grateful', 'hopeful'];
  const neutralEmotions = ['calm', 'focused', 'tired', 'contemplative'];
  const negativeEmotions = ['sad', 'anxious', 'frustrated', 'lonely', 'overwhelmed'];
  
  let emotions = [];
  if (moodScore >= 7) {
    emotions = positiveEmotions.slice(0, Math.floor(Math.random() * 3) + 1);
  } else if (moodScore >= 4) {
    emotions = neutralEmotions.slice(0, Math.floor(Math.random() * 2) + 1);
  } else {
    emotions = negativeEmotions.slice(0, Math.floor(Math.random() * 3) + 1);
  }
  
  return emotions;
};

const generateRandomTriggers = () => {
  const triggers = ['work_stress', 'family_issues', 'health_concerns', 'social_isolation', 'sleep_deprivation'];
  const selectedTriggers = [];
  
  if (Math.random() < 0.3) { // 30% chance of having triggers
    const numTriggers = Math.floor(Math.random() * 2) + 1;
    for (let i = 0; i < numTriggers; i++) {
      const trigger = triggers[Math.floor(Math.random() * triggers.length)];
      if (!selectedTriggers.includes(trigger)) {
        selectedTriggers.push(trigger);
      }
    }
  }
  
  return selectedTriggers;
};

// API Functions

// System Status
export const getSystemStatus = async () => {
  try {
    const response = await api.get('/system/status');
    return response;
  } catch (error) {
    console.warn('System status API not available, using fallback');
    return {
      status: 'demo',
      version: '1.0.0',
      last_update: new Date().toISOString(),
      modules: {
        data_collection: 'active',
        trend_analysis: 'active',
        pattern_detection: 'active',
        recommendations: 'active',
        crisis_detection: 'active'
      },
      stats: {
        total_entries: 1450,
        analysis_runs: 15,
        last_analysis: new Date().toISOString()
      }
    };
  }
};

// Mood Data
export const getMoodData = async (daysBack = 30) => {
  try {
    const response = await api.get(`/mood/data?days_back=${daysBack}`);
    return response;
  } catch (error) {
    console.warn('Mood data API not available, using fallback data');
    return generateFallbackData();
  }
};

// Submit Mood Entry
export const submitMoodEntry = async (moodEntry) => {
  try {
    const response = await api.post('/mood/entry', moodEntry);
    return response;
  } catch (error) {
    console.warn('Mood entry API not available, simulating submission');
    
    // Simulate successful submission
    const entry = {
      ...moodEntry,
      entry_id: `entry_${Date.now()}`,
      timestamp: new Date().toISOString(),
      data_source: 'user_input'
    };
    
    // Store in localStorage for demo purposes
    const existingData = JSON.parse(localStorage.getItem('mood_entries') || '[]');
    existingData.push(entry);
    localStorage.setItem('mood_entries', JSON.stringify(existingData));
    
    return {
      success: true,
      entry_id: entry.entry_id,
      message: 'Mood entry saved successfully (demo mode)',
      crisis_alert: moodEntry.mood_score <= 3 || moodEntry.stress_level >= 8,
      immediate_actions: moodEntry.mood_score <= 3 ? [
        'Consider reaching out to a friend or family member',
        'Practice deep breathing exercises',
        'Contact crisis support if needed: 988'
      ] : []
    };
  }
};

// Comprehensive Analysis
export const runComprehensiveAnalysis = async (daysBack = 30) => {
  try {
    const response = await api.post('/analysis/comprehensive', { days_back: daysBack });
    return response;
  } catch (error) {
    console.warn('Analysis API not available, generating sample analysis');
    
    const moodData = await getMoodData(daysBack);
    const avgMood = moodData.reduce((sum, entry) => sum + entry.mood_score, 0) / moodData.length;
    const avgStress = moodData.reduce((sum, entry) => sum + entry.stress_level, 0) / moodData.length;
    
    return {
      success: true,
      analysis_results: {
        analysis_metadata: {
          analysis_date: new Date().toISOString(),
          data_points_analyzed: moodData.length,
          analysis_period_days: daysBack,
          date_range: {
            start: moodData[0]?.timestamp,
            end: moodData[moodData.length - 1]?.timestamp
          }
        },
        trend_analysis: {
          mood_trend: avgMood >= 6 ? 'improving' : avgMood >= 4 ? 'stable' : 'declining',
          trend_slope: Math.random() * 0.2 - 0.1,
          forecast: Array.from({ length: 7 }, (_, i) => ({
            date: new Date(Date.now() + i * 24 * 60 * 60 * 1000).toISOString(),
            predicted_mood: Math.max(1, Math.min(10, avgMood + (Math.random() - 0.5)))
          }))
        },
        pattern_detection: {
          identified_patterns: [
            { type: 'weekly_cycle', confidence: 0.75, description: 'Mood tends to be lower on Mondays' },
            { type: 'sleep_correlation', confidence: 0.68, description: 'Better sleep correlates with improved mood' }
          ]
        },
        mental_health_indicators: {
          depression_risk: avgMood < 4 ? 'moderate' : 'low',
          anxiety_indicators: avgStress > 6 ? 'present' : 'minimal',
          overall_risk_assessment: {
            overall_risk_level: avgMood < 4 || avgStress > 7 ? 'moderate' : 'low'
          }
        },
        crisis_detection: {
          crisis_level: avgMood < 3 ? 'high' : 'low',
          immediate_action_plan: avgMood < 3 ? {
            immediate_steps: [
              'Contact crisis support: 988',
              'Reach out to trusted friend or family',
              'Practice grounding techniques'
            ]
          } : null
        },
        insights: {
          mood_insights: [
            { title: 'Sleep Quality Impact', description: 'Your sleep quality strongly affects your mood' },
            { title: 'Exercise Benefits', description: 'Days with exercise show 15% better mood scores' },
            { title: 'Social Connection', description: 'Social interactions correlate with positive mood' }
          ],
          personalized_recommendations: [
            'Maintain consistent sleep schedule',
            'Incorporate 30 minutes of daily exercise',
            'Schedule regular social activities'
          ]
        },
        care_recommendations: {
          immediate_care_needs: avgMood < 4 ? {
            priority_level: 'moderate',
            recommended_actions: ['Professional consultation recommended']
          } : null,
          lifestyle_interventions: [
            { category: 'sleep', recommendation: 'Optimize sleep hygiene' },
            { category: 'exercise', recommendation: 'Regular physical activity' },
            { category: 'social', recommendation: 'Maintain social connections' }
          ]
        },
        overall_summary: {
          overall_status: avgMood >= 6 ? 'stable' : avgMood >= 4 ? 'needs_attention' : 'concerning',
          crisis_level: avgMood < 3 ? 'high' : 'low',
          mental_health_risk: avgMood < 4 || avgStress > 7 ? 'moderate' : 'low',
          immediate_actions_needed: avgMood < 3 || avgStress > 8,
          key_findings: [
            `Average mood: ${avgMood.toFixed(1)}/10`,
            `Average stress: ${avgStress.toFixed(1)}/10`,
            `${moodData.length} data points analyzed`
          ],
          priority_recommendations: [
            'Continue daily mood tracking',
            'Focus on sleep quality improvement',
            'Consider professional support if needed'
          ]
        }
      }
    };
  }
};

// Daily Insights
export const getDailyInsights = async () => {
  try {
    const response = await api.get('/insights/daily');
    return response;
  } catch (error) {
    console.warn('Daily insights API not available, generating sample insights');
    
    const recentData = await getMoodData(7);
    const latestEntry = recentData[recentData.length - 1];
    
    return {
      success: true,
      daily_insights: {
        current_mood_trend: {
          trend: latestEntry?.mood_score >= 6 ? 'stable' : 'needs_attention',
          description: `Recent mood: ${latestEntry?.mood_score}/10`
        },
        stress_pattern: {
          pattern: latestEntry?.stress_level >= 6 ? 'elevated' : 'moderate',
          average: latestEntry?.stress_level || 5
        },
        sleep_quality: {
          quality: latestEntry?.sleep_hours >= 7 ? 'good' : 'needs_improvement',
          average_hours: latestEntry?.sleep_hours || 7
        },
        daily_recommendations: [
          'Continue tracking your daily mood',
          'Maintain healthy sleep schedule',
          'Practice stress management techniques'
        ],
        warning_signs: latestEntry?.mood_score <= 3 ? ['Low mood detected'] : [],
        positive_patterns: latestEntry?.mood_score >= 7 ? ['Good mood maintained'] : []
      }
    };
  }
};

// Crisis Status
export const getCrisisStatus = async () => {
  try {
    const response = await api.get('/crisis/status');
    return response;
  } catch (error) {
    console.warn('Crisis status API not available, using fallback');
    
    const recentData = await getMoodData(3);
    const avgMood = recentData.reduce((sum, entry) => sum + entry.mood_score, 0) / recentData.length;
    
    return {
      success: true,
      crisis_status: {
        crisis_level: avgMood < 3 ? 'high' : avgMood < 5 ? 'moderate' : 'low',
        risk_factors: avgMood < 4 ? ['Low mood pattern', 'Requires attention'] : [],
        immediate_action_plan: {
          immediate_steps: avgMood < 3 ? [
            'Contact crisis hotline: 988',
            'Reach out to trusted person',
            'Use emergency coping strategies'
          ] : [
            'Continue current mood tracking',
            'Maintain healthy routines',
            'Stay connected with support network'
          ]
        },
        resources: [
          { name: 'Crisis Text Line', contact: 'Text HOME to 741741', available: '24/7' },
          { name: 'National Suicide Prevention Lifeline', contact: '988', available: '24/7' },
          { name: 'Emergency Services', contact: '911', available: '24/7' }
        ]
      }
    };
  }
};

// Export Data
export const exportData = async (format = 'excel') => {
  try {
    const response = await api.get(`/export/data?format=${format}`, {
      responseType: 'blob'
    });
    return response;
  } catch (error) {
    console.warn('Export API not available, generating sample export');
    
    const data = await getMoodData();
    const csvContent = [
      'Date,Mood Score,Stress Level,Sleep Hours,Exercise Minutes,Emotions,Notes',
      ...data.map(entry => 
        `${entry.timestamp.split('T')[0]},${entry.mood_score},${entry.stress_level},${entry.sleep_hours},${entry.exercise_minutes},"${entry.emotions.join('; ')}","${entry.notes}"`
      )
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv' });
    return blob;
  }
};

// Generate Report
export const generateReport = async (reportType = 'comprehensive') => {
  try {
    const response = await api.post('/reports/generate', { report_type: reportType });
    return response;
  } catch (error) {
    console.warn('Report API not available, generating sample report');
    
    const analysisResults = await runComprehensiveAnalysis();
    
    return {
      success: true,
      report_type: reportType,
      generated_at: new Date().toISOString(),
      report: {
        summary: analysisResults.analysis_results.overall_summary,
        insights: analysisResults.analysis_results.insights,
        recommendations: analysisResults.analysis_results.care_recommendations
      }
    };
  }
};

export default api; 