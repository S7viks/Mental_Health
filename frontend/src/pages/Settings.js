import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  Settings, 
  Bell, 
  Palette, 
  Database, 
  Shield, 
  HelpCircle,
  Save,
  RefreshCw
} from 'lucide-react';
import toast from 'react-hot-toast';

const SettingsPage = ({ systemStatus, onRefresh }) => {
  const [settings, setSettings] = useState({
    notifications: {
      dailyReminder: true,
      weeklyReport: true,
      crisisAlert: true,
      reminderTime: '20:00'
    },
    privacy: {
      anonymizeData: false,
      shareAnalytics: true,
      dataRetention: '365'
    },
    theme: {
      colorScheme: 'system',
      compactView: false
    },
    analysis: {
      autoAnalysis: true,
      analysisFrequency: 'weekly',
      includeWeather: true,
      includeSocial: true
    }
  });

  const handleSettingChange = (category, setting, value) => {
    setSettings(prev => ({
      ...prev,
      [category]: {
        ...prev[category],
        [setting]: value
      }
    }));
  };

  const handleSaveSettings = () => {
    // Save settings to localStorage or send to API
    localStorage.setItem('mood_tracker_settings', JSON.stringify(settings));
    toast.success('Settings saved successfully!');
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="flex items-center justify-between"
      >
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Settings</h1>
          <p className="mt-1 text-gray-500">
            Customize your Mental Health Tracker experience
          </p>
        </div>
        <div className="flex space-x-3">
          <button onClick={onRefresh} className="btn btn-outline">
            <RefreshCw className="w-4 h-4 mr-2" />
            Refresh
          </button>
          <button onClick={handleSaveSettings} className="btn btn-primary">
            <Save className="w-4 h-4 mr-2" />
            Save Settings
          </button>
        </div>
      </motion.div>

      {/* Notifications */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <div className="card-header">
          <div className="flex items-center space-x-2">
            <Bell className="w-5 h-5 text-blue-600" />
            <h2 className="text-xl font-semibold text-gray-900">Notifications</h2>
          </div>
        </div>
        <div className="card-body space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Daily Mood Reminder</h3>
              <p className="text-sm text-gray-500">Get reminded to log your daily mood</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={settings.notifications.dailyReminder}
                onChange={(e) => handleSettingChange('notifications', 'dailyReminder', e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Weekly Reports</h3>
              <p className="text-sm text-gray-500">Receive weekly mood analysis reports</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={settings.notifications.weeklyReport}
                onChange={(e) => handleSettingChange('notifications', 'weeklyReport', e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Crisis Alerts</h3>
              <p className="text-sm text-gray-500">Alert when patterns suggest crisis support is needed</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={settings.notifications.crisisAlert}
                onChange={(e) => handleSettingChange('notifications', 'crisisAlert', e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Reminder Time</h3>
              <p className="text-sm text-gray-500">When to send daily mood reminders</p>
            </div>
            <input
              type="time"
              value={settings.notifications.reminderTime}
              onChange={(e) => handleSettingChange('notifications', 'reminderTime', e.target.value)}
              className="form-input w-32"
            />
          </div>
        </div>
      </motion.div>

      {/* Privacy */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="card"
      >
        <div className="card-header">
          <div className="flex items-center space-x-2">
            <Shield className="w-5 h-5 text-green-600" />
            <h2 className="text-xl font-semibold text-gray-900">Privacy & Data</h2>
          </div>
        </div>
        <div className="card-body space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Anonymize Data</h3>
              <p className="text-sm text-gray-500">Remove identifying information from stored data</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={settings.privacy.anonymizeData}
                onChange={(e) => handleSettingChange('privacy', 'anonymizeData', e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Share Analytics</h3>
              <p className="text-sm text-gray-500">Help improve the app by sharing anonymous usage data</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={settings.privacy.shareAnalytics}
                onChange={(e) => handleSettingChange('privacy', 'shareAnalytics', e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Data Retention</h3>
              <p className="text-sm text-gray-500">How long to keep your mood data (days)</p>
            </div>
            <select
              value={settings.privacy.dataRetention}
              onChange={(e) => handleSettingChange('privacy', 'dataRetention', e.target.value)}
              className="form-select w-32"
            >
              <option value="90">90 days</option>
              <option value="180">180 days</option>
              <option value="365">1 year</option>
              <option value="730">2 years</option>
              <option value="forever">Forever</option>
            </select>
          </div>
        </div>
      </motion.div>

      {/* Analysis */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card"
      >
        <div className="card-header">
          <div className="flex items-center space-x-2">
            <Database className="w-5 h-5 text-purple-600" />
            <h2 className="text-xl font-semibold text-gray-900">Analysis Settings</h2>
          </div>
        </div>
        <div className="card-body space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Auto Analysis</h3>
              <p className="text-sm text-gray-500">Automatically run analysis on new mood entries</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={settings.analysis.autoAnalysis}
                onChange={(e) => handleSettingChange('analysis', 'autoAnalysis', e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Analysis Frequency</h3>
              <p className="text-sm text-gray-500">How often to run comprehensive analysis</p>
            </div>
            <select
              value={settings.analysis.analysisFrequency}
              onChange={(e) => handleSettingChange('analysis', 'analysisFrequency', e.target.value)}
              className="form-select w-32"
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
              <option value="monthly">Monthly</option>
            </select>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Include Weather</h3>
              <p className="text-sm text-gray-500">Include weather data in analysis</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={settings.analysis.includeWeather}
                onChange={(e) => handleSettingChange('analysis', 'includeWeather', e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>

          <div className="flex items-center justify-between">
            <div>
              <h3 className="font-medium text-gray-900">Include Social Data</h3>
              <p className="text-sm text-gray-500">Include social interaction data in analysis</p>
            </div>
            <label className="relative inline-flex items-center cursor-pointer">
              <input
                type="checkbox"
                className="sr-only peer"
                checked={settings.analysis.includeSocial}
                onChange={(e) => handleSettingChange('analysis', 'includeSocial', e.target.checked)}
              />
              <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-blue-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-blue-600"></div>
            </label>
          </div>
        </div>
      </motion.div>

      {/* System Status */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.4 }}
        className="card"
      >
        <div className="card-header">
          <div className="flex items-center space-x-2">
            <HelpCircle className="w-5 h-5 text-orange-600" />
            <h2 className="text-xl font-semibold text-gray-900">System Information</h2>
          </div>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            <div>
              <h3 className="font-medium text-gray-900">System Status</h3>
              <p className="text-sm text-gray-600 capitalize">
                {systemStatus?.status || 'Unknown'}
              </p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Version</h3>
              <p className="text-sm text-gray-600">
                {systemStatus?.version || '1.0.0'}
              </p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Total Entries</h3>
              <p className="text-sm text-gray-600">
                {systemStatus?.stats?.total_entries || 0}
              </p>
            </div>
            <div>
              <h3 className="font-medium text-gray-900">Last Analysis</h3>
              <p className="text-sm text-gray-600">
                {systemStatus?.stats?.last_analysis 
                  ? new Date(systemStatus.stats.last_analysis).toLocaleString()
                  : 'Never'
                }
              </p>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default SettingsPage; 