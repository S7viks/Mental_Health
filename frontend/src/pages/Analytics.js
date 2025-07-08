import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  BarChart3, 
  PieChart, 
  Calendar,
  RefreshCw,
  Download,
  AlertCircle
} from 'lucide-react';
import { format, parseISO, subDays } from 'date-fns';
import { 
  LineChart, 
  Line, 
  AreaChart, 
  Area, 
  BarChart, 
  Bar,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart as RechartsPieChart,
  Cell
} from 'recharts';

import { runComprehensiveAnalysis } from '../utils/api';
import LoadingSpinner from '../components/LoadingSpinner';

const Analytics = ({ moodData, systemStatus }) => {
  const [analysis, setAnalysis] = useState(null);
  const [loading, setLoading] = useState(true);
  const [timeRange, setTimeRange] = useState(30);

  useEffect(() => {
    loadAnalysis();
  }, [timeRange]);

  const loadAnalysis = async () => {
    try {
      setLoading(true);
      const result = await runComprehensiveAnalysis(timeRange);
      setAnalysis(result.analysis_results);
    } catch (error) {
      console.error('Failed to load analysis:', error);
    } finally {
      setLoading(false);
    }
  };

  // Prepare chart data
  const chartData = moodData
    ?.slice(-timeRange)
    ?.map(entry => ({
      date: format(parseISO(entry.timestamp), 'MMM dd'),
      fullDate: entry.timestamp,
      mood: entry.mood_score,
      stress: entry.stress_level,
      sleep: entry.sleep_hours,
      exercise: entry.exercise_minutes,
      social: entry.social_interactions
    })) || [];

  // Mood distribution data
  const moodDistribution = [
    { name: 'Excellent (8-10)', value: 0, color: '#10b981' },
    { name: 'Good (6-7)', value: 0, color: '#84cc16' },
    { name: 'Moderate (4-5)', value: 0, color: '#f59e0b' },
    { name: 'Low (2-3)', value: 0, color: '#f97316' },
    { name: 'Very Low (1)', value: 0, color: '#ef4444' }
  ];

  // Calculate mood distribution
  moodData?.forEach(entry => {
    const score = entry.mood_score;
    if (score >= 8) moodDistribution[0].value++;
    else if (score >= 6) moodDistribution[1].value++;
    else if (score >= 4) moodDistribution[2].value++;
    else if (score >= 2) moodDistribution[3].value++;
    else moodDistribution[4].value++;
  });

  // Weekly patterns
  const weeklyPatterns = {};
  moodData?.forEach(entry => {
    const dayOfWeek = format(parseISO(entry.timestamp), 'EEEE');
    if (!weeklyPatterns[dayOfWeek]) {
      weeklyPatterns[dayOfWeek] = { total: 0, count: 0 };
    }
    weeklyPatterns[dayOfWeek].total += entry.mood_score;
    weeklyPatterns[dayOfWeek].count++;
  });

  const weeklyData = Object.keys(weeklyPatterns).map(day => ({
    day,
    averageMood: weeklyPatterns[day].count > 0 
      ? (weeklyPatterns[day].total / weeklyPatterns[day].count).toFixed(1)
      : 0
  }));

  // Sort by day of week
  const dayOrder = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'];
  weeklyData.sort((a, b) => dayOrder.indexOf(a.day) - dayOrder.indexOf(b.day));

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <LoadingSpinner size="large" message="Running comprehensive analysis..." />
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Analytics</h1>
          <p className="mt-1 text-gray-500">
            Comprehensive analysis of your mental health patterns
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          <select
            value={timeRange}
            onChange={(e) => setTimeRange(parseInt(e.target.value))}
            className="form-select"
          >
            <option value={7}>Last 7 Days</option>
            <option value={14}>Last 14 Days</option>
            <option value={30}>Last 30 Days</option>
            <option value={90}>Last 90 Days</option>
          </select>
          <button 
            onClick={loadAnalysis}
            className="btn btn-outline"
          >
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
        </div>
      </div>

      {/* Analysis Summary */}
      {analysis && (
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card"
          >
            <div className="card-body">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Overall Status</p>
                  <p className="text-2xl font-bold text-gray-900 capitalize">
                    {analysis.overall_summary?.overall_status || 'Analyzing...'}
                  </p>
                </div>
                <div className="p-3 bg-blue-50 rounded-lg">
                  <BarChart3 className="w-6 h-6 text-blue-600" />
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.1 }}
            className="card"
          >
            <div className="card-body">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Mental Health Risk</p>
                  <p className="text-2xl font-bold text-gray-900 capitalize">
                    {analysis.overall_summary?.mental_health_risk || 'Low'}
                  </p>
                </div>
                <div className="p-3 bg-green-50 rounded-lg">
                  <TrendingUp className="w-6 h-6 text-green-600" />
                </div>
              </div>
            </div>
          </motion.div>

          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 0.2 }}
            className="card"
          >
            <div className="card-body">
              <div className="flex items-center justify-between">
                <div>
                  <p className="text-sm text-gray-500">Data Points</p>
                  <p className="text-2xl font-bold text-gray-900">
                    {analysis.analysis_metadata?.data_points_analyzed || 0}
                  </p>
                </div>
                <div className="p-3 bg-purple-50 rounded-lg">
                  <Calendar className="w-6 h-6 text-purple-600" />
                </div>
              </div>
            </div>
          </motion.div>
        </div>
      )}

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Mood Trend */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Mood Trend</h3>
            <p className="text-sm text-gray-500">Daily mood scores over time</p>
          </div>
          <div className="card-body">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                  <YAxis domain={[1, 10]} stroke="#6b7280" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px'
                    }}
                  />
                  <Area 
                    type="monotone" 
                    dataKey="mood" 
                    stroke="#3b82f6" 
                    fill="#93c5fd" 
                    strokeWidth={2}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>

        {/* Mood Distribution */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Mood Distribution</h3>
            <p className="text-sm text-gray-500">How often you experience different mood levels</p>
          </div>
          <div className="card-body">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <RechartsPieChart>
                  <Tooltip />
                  <RechartsPieChart
                    data={moodDistribution}
                    dataKey="value"
                    nameKey="name"
                    cx="50%"
                    cy="50%"
                    outerRadius={80}
                    fill="#8884d8"
                  >
                    {moodDistribution.map((entry, index) => (
                      <Cell key={`cell-${index}`} fill={entry.color} />
                    ))}
                  </RechartsPieChart>
                </RechartsPieChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>

        {/* Weekly Patterns */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Weekly Patterns</h3>
            <p className="text-sm text-gray-500">Average mood by day of the week</p>
          </div>
          <div className="card-body">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={weeklyData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="day" stroke="#6b7280" fontSize={12} />
                  <YAxis domain={[1, 10]} stroke="#6b7280" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px'
                    }}
                  />
                  <Bar dataKey="averageMood" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>

        {/* Correlations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Mood vs Factors</h3>
            <p className="text-sm text-gray-500">How different factors relate to your mood</p>
          </div>
          <div className="card-body">
            <div className="chart-container">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#f0f0f0" />
                  <XAxis dataKey="date" stroke="#6b7280" fontSize={12} />
                  <YAxis stroke="#6b7280" fontSize={12} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: '#fff', 
                      border: '1px solid #e5e7eb',
                      borderRadius: '8px'
                    }}
                  />
                  <Line 
                    type="monotone" 
                    dataKey="mood" 
                    stroke="#3b82f6" 
                    strokeWidth={2}
                    name="Mood"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="sleep" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Sleep (hours)"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="stress" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    name="Stress"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Insights */}
      {analysis?.insights && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Key Insights</h3>
            <p className="text-sm text-gray-500">AI-powered analysis of your patterns</p>
          </div>
          <div className="card-body">
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Mood Insights</h4>
                <div className="space-y-3">
                  {analysis.insights.mood_insights?.map((insight, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-2 h-2 bg-blue-500 rounded-full mt-2"></div>
                      <div>
                        <p className="font-medium text-gray-900">{insight.title}</p>
                        <p className="text-sm text-gray-600">{insight.description}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              
              <div>
                <h4 className="font-medium text-gray-900 mb-3">Recommendations</h4>
                <div className="space-y-3">
                  {analysis.insights.personalized_recommendations?.map((rec, index) => (
                    <div key={index} className="flex items-start space-x-3">
                      <div className="flex-shrink-0 w-2 h-2 bg-green-500 rounded-full mt-2"></div>
                      <p className="text-sm text-gray-600">{rec}</p>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </motion.div>
      )}

      {/* Crisis Detection */}
      {analysis?.crisis_detection?.crisis_level === 'high' && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="card border-l-4 border-red-500"
        >
          <div className="card-body">
            <div className="flex items-start space-x-3">
              <AlertCircle className="w-6 h-6 text-red-500 flex-shrink-0 mt-1" />
              <div>
                <h3 className="text-lg font-semibold text-red-900">Crisis Alert</h3>
                <p className="text-red-700 mt-1">
                  Our analysis indicates you may need immediate support. Please consider reaching out for professional help.
                </p>
                <div className="mt-4 space-y-2">
                  {analysis.crisis_detection.immediate_action_plan?.immediate_steps?.map((step, index) => (
                    <div key={index} className="flex items-start space-x-2">
                      <span className="text-red-600 font-medium">•</span>
                      <span className="text-red-800">{step}</span>
                    </div>
                  ))}
                </div>
                <button className="mt-4 btn btn-danger">
                  View Crisis Resources
                </button>
              </div>
            </div>
          </div>
        </motion.div>
      )}
    </div>
  );
};

export default Analytics; 