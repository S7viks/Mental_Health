import React, { useState, useEffect } from 'react';
import { motion } from 'framer-motion';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity, 
  Heart, 
  Brain,
  AlertTriangle,
  Plus,
  Eye,
  Calendar,
  Clock,
  BarChart3,
  Smile,
  Frown,
  Meh
} from 'lucide-react';
import { format, subDays, parseISO } from 'date-fns';
import { LineChart, Line, AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, BarChart, Bar } from 'recharts';

import { getDailyInsights, getCrisisStatus } from '../utils/api';

const Dashboard = ({ moodData, systemStatus, onRefresh }) => {
  const [insights, setInsights] = useState(null);
  const [crisisStatus, setCrisisStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadDashboardData();
  }, []);

  const loadDashboardData = async () => {
    try {
      setLoading(true);
      const [insightsData, crisisData] = await Promise.all([
        getDailyInsights(),
        getCrisisStatus()
      ]);
      
      setInsights(insightsData.daily_insights);
      setCrisisStatus(crisisData.crisis_status);
    } catch (error) {
      console.error('Failed to load dashboard data:', error);
    } finally {
      setLoading(false);
    }
  };

  // Calculate dashboard stats
  const calculateStats = () => {
    if (!moodData || moodData.length === 0) {
      return {
        averageMood: 0,
        moodTrend: 0,
        totalEntries: 0,
        streakDays: 0,
        lastEntry: null
      };
    }

    const sortedData = [...moodData].sort((a, b) => new Date(a.timestamp) - new Date(b.timestamp));
    const recentData = sortedData.slice(-7); // Last 7 days
    
    const averageMood = moodData.reduce((sum, entry) => sum + entry.mood_score, 0) / moodData.length;
    
    // Calculate trend (comparing last 3 days to previous 4 days)
    const lastThree = recentData.slice(-3);
    const previousFour = recentData.slice(-7, -3);
    const lastThreeAvg = lastThree.reduce((sum, entry) => sum + entry.mood_score, 0) / lastThree.length;
    const previousFourAvg = previousFour.reduce((sum, entry) => sum + entry.mood_score, 0) / previousFour.length;
    const moodTrend = lastThreeAvg - previousFourAvg;

    // Calculate streak
    let streakDays = 0;
    const today = new Date();
    for (let i = 0; i < 30; i++) {
      const checkDate = format(subDays(today, i), 'yyyy-MM-dd');
      const hasEntry = moodData.some(entry => 
        format(parseISO(entry.timestamp), 'yyyy-MM-dd') === checkDate
      );
      if (hasEntry) {
        streakDays++;
      } else {
        break;
      }
    }

    return {
      averageMood: averageMood.toFixed(1),
      moodTrend: moodTrend.toFixed(1),
      totalEntries: moodData.length,
      streakDays,
      lastEntry: sortedData[sortedData.length - 1]
    };
  };

  const stats = calculateStats();

  // Prepare chart data
  const chartData = moodData
    ?.slice(-14) // Last 14 days
    ?.map(entry => ({
      date: format(parseISO(entry.timestamp), 'MMM dd'),
      mood: entry.mood_score,
      stress: entry.stress_level,
      sleep: entry.sleep_hours
    })) || [];

  const getMoodIcon = (score) => {
    if (score >= 7) return <Smile className="w-5 h-5 text-green-500" />;
    if (score >= 4) return <Meh className="w-5 h-5 text-yellow-500" />;
    return <Frown className="w-5 h-5 text-red-500" />;
  };

  const getMoodColor = (score) => {
    if (score >= 7) return 'text-green-600 bg-green-50';
    if (score >= 4) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  const getTrendIcon = (trend) => {
    if (trend > 0.5) return <TrendingUp className="w-4 h-4 text-green-500" />;
    if (trend < -0.5) return <TrendingDown className="w-4 h-4 text-red-500" />;
    return <Activity className="w-4 h-4 text-yellow-500" />;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center min-h-96">
        <div className="text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600 mx-auto"></div>
          <p className="mt-4 text-gray-600">Loading dashboard...</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex flex-col sm:flex-row sm:items-center sm:justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="mt-1 text-gray-500">
            Welcome back! Here's your mental health overview.
          </p>
        </div>
        <div className="mt-4 sm:mt-0 flex space-x-3">
          <button 
            onClick={() => window.location.href = '/mood-entry'}
            className="btn btn-primary"
          >
            <Plus className="w-4 h-4 mr-2" />
            Add Entry
          </button>
          <button 
            onClick={() => window.location.href = '/analytics'}
            className="btn btn-outline"
          >
            <Eye className="w-4 h-4 mr-2" />
            View Analysis
          </button>
        </div>
      </motion.div>

      {/* Crisis Alert */}
      {crisisStatus?.crisis_level === 'high' && (
        <motion.div
          initial={{ opacity: 0, scale: 0.9 }}
          animate={{ opacity: 1, scale: 1 }}
          className="bg-red-50 border-l-4 border-red-400 p-4 rounded-lg"
        >
          <div className="flex items-center">
            <AlertTriangle className="w-5 h-5 text-red-400 mr-3" />
            <div>
              <h3 className="text-sm font-medium text-red-800">
                Immediate Attention Needed
              </h3>
              <p className="text-sm text-red-700 mt-1">
                Our analysis suggests you may need professional support. Please consider reaching out for help.
              </p>
              <button className="mt-2 text-sm font-medium text-red-800 underline hover:text-red-900">
                View Crisis Resources →
              </button>
            </div>
          </div>
        </motion.div>
      )}

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        {/* Average Mood */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.1 }}
          className="stat-card"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="stat-label">Average Mood</p>
              <p className={`stat-value ${getMoodColor(parseFloat(stats.averageMood))}`}>
                {stats.averageMood}/10
              </p>
              <div className="stat-trend">
                {getTrendIcon(parseFloat(stats.moodTrend))}
                <span className={`ml-1 ${parseFloat(stats.moodTrend) > 0 ? 'stat-trend-positive' : parseFloat(stats.moodTrend) < 0 ? 'stat-trend-negative' : 'stat-trend-neutral'}`}>
                  {parseFloat(stats.moodTrend) > 0 ? '+' : ''}{stats.moodTrend} trend
                </span>
              </div>
            </div>
            <div className="p-3 bg-blue-50 rounded-lg">
              <Heart className="w-6 h-6 text-blue-600" />
            </div>
          </div>
        </motion.div>

        {/* Streak Days */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.2 }}
          className="stat-card"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="stat-label">Current Streak</p>
              <p className="stat-value">{stats.streakDays}</p>
              <p className="text-sm text-gray-500 mt-1">consecutive days</p>
            </div>
            <div className="p-3 bg-green-50 rounded-lg">
              <Calendar className="w-6 h-6 text-green-600" />
            </div>
          </div>
        </motion.div>

        {/* Total Entries */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.3 }}
          className="stat-card"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="stat-label">Total Entries</p>
              <p className="stat-value">{stats.totalEntries}</p>
              <p className="text-sm text-gray-500 mt-1">mood logs</p>
            </div>
            <div className="p-3 bg-purple-50 rounded-lg">
              <BarChart3 className="w-6 h-6 text-purple-600" />
            </div>
          </div>
        </motion.div>

        {/* Last Entry */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.4 }}
          className="stat-card"
        >
          <div className="flex items-center justify-between">
            <div>
              <p className="stat-label">Last Entry</p>
              <div className="flex items-center space-x-2">
                {stats.lastEntry && getMoodIcon(stats.lastEntry.mood_score)}
                <p className="stat-value">
                  {stats.lastEntry ? stats.lastEntry.mood_score : 'N/A'}/10
                </p>
              </div>
              <p className="text-sm text-gray-500 mt-1">
                {stats.lastEntry ? format(parseISO(stats.lastEntry.timestamp), 'MMM dd, HH:mm') : 'No entries yet'}
              </p>
            </div>
            <div className="p-3 bg-orange-50 rounded-lg">
              <Clock className="w-6 h-6 text-orange-600" />
            </div>
          </div>
        </motion.div>
      </div>

      {/* Charts Grid */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Mood Trend Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.5 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Mood Trend (14 Days)</h3>
            <p className="text-sm text-gray-500">Track your mood patterns over time</p>
          </div>
          <div className="card-body">
            <div className="chart-container h-64">
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

        {/* Stress vs Sleep Chart */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.6 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Stress vs Sleep</h3>
            <p className="text-sm text-gray-500">Compare stress levels with sleep quality</p>
          </div>
          <div className="card-body">
            <div className="chart-container h-64">
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
                    dataKey="stress" 
                    stroke="#ef4444" 
                    strokeWidth={2}
                    name="Stress Level"
                  />
                  <Line 
                    type="monotone" 
                    dataKey="sleep" 
                    stroke="#10b981" 
                    strokeWidth={2}
                    name="Sleep Hours"
                  />
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        </motion.div>
      </div>

      {/* Insights and Recommendations */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Daily Insights */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.7 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Today's Insights</h3>
            <p className="text-sm text-gray-500">AI-powered analysis of your patterns</p>
          </div>
          <div className="card-body space-y-4">
            {insights?.current_mood_trend && (
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 bg-blue-100 rounded-lg flex items-center justify-center">
                  <Brain className="w-4 h-4 text-blue-600" />
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Mood Trend</h4>
                  <p className="text-sm text-gray-600">
                    {insights.current_mood_trend.description}
                  </p>
                </div>
              </div>
            )}

            {insights?.sleep_quality && (
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <Heart className="w-4 h-4 text-green-600" />
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Sleep Quality</h4>
                  <p className="text-sm text-gray-600">
                    Average {insights.sleep_quality.average_hours} hours - {insights.sleep_quality.quality}
                  </p>
                </div>
              </div>
            )}

            {insights?.positive_patterns?.length > 0 && (
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 bg-green-100 rounded-lg flex items-center justify-center">
                  <TrendingUp className="w-4 h-4 text-green-600" />
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Positive Patterns</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {insights.positive_patterns.map((pattern, index) => (
                      <li key={index}>• {pattern}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}

            {insights?.warning_signs?.length > 0 && (
              <div className="flex items-start space-x-3">
                <div className="flex-shrink-0 w-8 h-8 bg-yellow-100 rounded-lg flex items-center justify-center">
                  <AlertTriangle className="w-4 h-4 text-yellow-600" />
                </div>
                <div>
                  <h4 className="font-medium text-gray-900">Areas to Watch</h4>
                  <ul className="text-sm text-gray-600 space-y-1">
                    {insights.warning_signs.map((warning, index) => (
                      <li key={index}>• {warning}</li>
                    ))}
                  </ul>
                </div>
              </div>
            )}
          </div>
        </motion.div>

        {/* Recommendations */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ delay: 0.8 }}
          className="card"
        >
          <div className="card-header">
            <h3 className="text-lg font-semibold text-gray-900">Recommendations</h3>
            <p className="text-sm text-gray-500">Personalized suggestions for your wellbeing</p>
          </div>
          <div className="card-body">
            <div className="space-y-3">
              {insights?.daily_recommendations?.map((recommendation, index) => (
                <div key={index} className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center">
                    <span className="text-xs font-medium text-blue-600">{index + 1}</span>
                  </div>
                  <p className="text-sm text-gray-700">{recommendation}</p>
                </div>
              )) || [
                <div key="default" className="flex items-start space-x-3 p-3 bg-gray-50 rounded-lg">
                  <div className="flex-shrink-0 w-6 h-6 bg-blue-100 rounded-full flex items-center justify-center">
                    <span className="text-xs font-medium text-blue-600">1</span>
                  </div>
                  <p className="text-sm text-gray-700">Continue tracking your daily mood for personalized insights</p>
                </div>
              ]}
            </div>
          </div>
        </motion.div>
      </div>
    </div>
  );
};

export default Dashboard; 