import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import { 
  Smile, 
  Frown, 
  Meh, 
  Heart, 
  Moon, 
  Dumbbell, 
  Users, 
  Cloud,
  Save,
  AlertTriangle,
  CheckCircle,
  X
} from 'lucide-react';
import toast from 'react-hot-toast';

import { submitMoodEntry } from '../utils/api';

const MoodEntry = ({ onEntrySubmit }) => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(false);
  const [showCrisisAlert, setShowCrisisAlert] = useState(false);
  
  const [formData, setFormData] = useState({
    mood_score: 5,
    stress_level: 5,
    sleep_hours: 8,
    exercise_minutes: 0,
    social_interactions: 0,
    emotions: [],
    notes: '',
    weather: 'sunny',
    triggers: []
  });

  const emotionOptions = [
    { value: 'happy', label: 'Happy', color: 'bg-green-100 text-green-800' },
    { value: 'sad', label: 'Sad', color: 'bg-blue-100 text-blue-800' },
    { value: 'anxious', label: 'Anxious', color: 'bg-yellow-100 text-yellow-800' },
    { value: 'excited', label: 'Excited', color: 'bg-purple-100 text-purple-800' },
    { value: 'frustrated', label: 'Frustrated', color: 'bg-red-100 text-red-800' },
    { value: 'content', label: 'Content', color: 'bg-green-100 text-green-800' },
    { value: 'overwhelmed', label: 'Overwhelmed', color: 'bg-orange-100 text-orange-800' },
    { value: 'grateful', label: 'Grateful', color: 'bg-pink-100 text-pink-800' },
    { value: 'lonely', label: 'Lonely', color: 'bg-gray-100 text-gray-800' },
    { value: 'hopeful', label: 'Hopeful', color: 'bg-indigo-100 text-indigo-800' },
    { value: 'tired', label: 'Tired', color: 'bg-gray-100 text-gray-800' },
    { value: 'energetic', label: 'Energetic', color: 'bg-yellow-100 text-yellow-800' }
  ];

  const triggerOptions = [
    { value: 'work_stress', label: 'Work Stress' },
    { value: 'family_issues', label: 'Family Issues' },
    { value: 'health_concerns', label: 'Health Concerns' },
    { value: 'financial_stress', label: 'Financial Stress' },
    { value: 'relationship_issues', label: 'Relationship Issues' },
    { value: 'social_isolation', label: 'Social Isolation' },
    { value: 'sleep_deprivation', label: 'Sleep Deprivation' },
    { value: 'weather', label: 'Weather' },
    { value: 'hormonal_changes', label: 'Hormonal Changes' },
    { value: 'medication_effects', label: 'Medication Effects' }
  ];

  const weatherOptions = [
    { value: 'sunny', label: 'Sunny ☀️', icon: '☀️' },
    { value: 'cloudy', label: 'Cloudy ☁️', icon: '☁️' },
    { value: 'rainy', label: 'Rainy 🌧️', icon: '🌧️' },
    { value: 'snowy', label: 'Snowy ❄️', icon: '❄️' },
    { value: 'stormy', label: 'Stormy ⛈️', icon: '⛈️' }
  ];

  const handleInputChange = (field, value) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
  };

  const handleEmotionToggle = (emotion) => {
    setFormData(prev => ({
      ...prev,
      emotions: prev.emotions.includes(emotion)
        ? prev.emotions.filter(e => e !== emotion)
        : [...prev.emotions, emotion]
    }));
  };

  const handleTriggerToggle = (trigger) => {
    setFormData(prev => ({
      ...prev,
      triggers: prev.triggers.includes(trigger)
        ? prev.triggers.filter(t => t !== trigger)
        : [...prev.triggers, trigger]
    }));
  };

  const getMoodIcon = (score) => {
    if (score >= 7) return <Smile className="w-6 h-6 text-green-500" />;
    if (score >= 4) return <Meh className="w-6 h-6 text-yellow-500" />;
    return <Frown className="w-6 h-6 text-red-500" />;
  };

  const getMoodColor = (score) => {
    if (score >= 7) return 'bg-green-500';
    if (score >= 4) return 'bg-yellow-500';
    return 'bg-red-500';
  };

  const getMoodLabel = (score) => {
    if (score >= 9) return 'Excellent';
    if (score >= 7) return 'Good';
    if (score >= 5) return 'Okay';
    if (score >= 3) return 'Low';
    return 'Very Low';
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);

    try {
      const result = await submitMoodEntry(formData);
      
      if (result.success) {
        toast.success('Mood entry saved successfully!');
        
        // Check for crisis alert
        if (result.crisis_alert) {
          setShowCrisisAlert(true);
          return;
        }
        
        // Refresh data and navigate
        if (onEntrySubmit) {
          await onEntrySubmit();
        }
        
        navigate('/dashboard');
      } else {
        toast.error(result.message || 'Failed to save mood entry');
      }
    } catch (error) {
      console.error('Submit error:', error);
      toast.error('Failed to save mood entry. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const CrisisAlert = () => (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
      <motion.div
        initial={{ scale: 0.9, opacity: 0 }}
        animate={{ scale: 1, opacity: 1 }}
        className="bg-white rounded-lg p-6 max-w-md w-full"
      >
        <div className="flex items-center space-x-3 mb-4">
          <AlertTriangle className="w-8 h-8 text-red-500" />
          <h3 className="text-lg font-semibold text-gray-900">
            We're Here to Help
          </h3>
        </div>
        
        <p className="text-gray-700 mb-4">
          Your mood entry suggests you might be going through a difficult time. 
          Please know that you're not alone and help is available.
        </p>
        
        <div className="space-y-3 mb-6">
          <div className="p-3 bg-red-50 rounded-lg">
            <h4 className="font-medium text-red-800">Crisis Resources</h4>
            <p className="text-sm text-red-700">
              • National Suicide Prevention Lifeline: 988<br/>
              • Crisis Text Line: Text HOME to 741741<br/>
              • Emergency Services: 911
            </p>
          </div>
        </div>
        
        <div className="flex space-x-3">
          <button
            onClick={() => navigate('/crisis-support')}
            className="flex-1 bg-red-600 text-white px-4 py-2 rounded-lg hover:bg-red-700 transition-colors"
          >
            Get Support Now
          </button>
          <button
            onClick={() => {
              setShowCrisisAlert(false);
              if (onEntrySubmit) onEntrySubmit();
              navigate('/dashboard');
            }}
            className="flex-1 bg-gray-200 text-gray-800 px-4 py-2 rounded-lg hover:bg-gray-300 transition-colors"
          >
            Continue
          </button>
        </div>
      </motion.div>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto">
      {showCrisisAlert && <CrisisAlert />}
      
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="space-y-6"
      >
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-900">How Are You Feeling?</h1>
          <p className="mt-2 text-gray-600">
            Take a moment to check in with yourself and log your mood
          </p>
        </div>

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-8">
          {/* Mood Score */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.1 }}
            className="card"
          >
            <div className="card-header">
              <div className="flex items-center space-x-3">
                {getMoodIcon(formData.mood_score)}
                <h3 className="text-lg font-semibold text-gray-900">
                  Overall Mood: {getMoodLabel(formData.mood_score)}
                </h3>
              </div>
            </div>
            <div className="card-body">
              <div className="space-y-4">
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-500 w-8">1</span>
                  <div className="flex-1">
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.mood_score}
                      onChange={(e) => handleInputChange('mood_score', parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      style={{
                        background: `linear-gradient(to right, #ef4444 0%, #f59e0b 50%, #10b981 100%)`
                      }}
                    />
                  </div>
                  <span className="text-sm text-gray-500 w-8">10</span>
                </div>
                <div className="text-center">
                  <div className={`inline-block px-4 py-2 rounded-full text-white font-medium ${getMoodColor(formData.mood_score)}`}>
                    {formData.mood_score}/10
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Emotions */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.2 }}
            className="card"
          >
            <div className="card-header">
              <h3 className="text-lg font-semibold text-gray-900">Emotions</h3>
              <p className="text-sm text-gray-500">Select all that apply</p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 gap-3">
                {emotionOptions.map((emotion) => (
                  <button
                    key={emotion.value}
                    type="button"
                    onClick={() => handleEmotionToggle(emotion.value)}
                    className={`
                      p-3 rounded-lg border transition-all duration-200
                      ${formData.emotions.includes(emotion.value)
                        ? `${emotion.color} border-current`
                        : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                      }
                    `}
                  >
                    <span className="text-sm font-medium">{emotion.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Stress Level */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.3 }}
            className="card"
          >
            <div className="card-header">
              <h3 className="text-lg font-semibold text-gray-900">Stress Level</h3>
            </div>
            <div className="card-body">
              <div className="space-y-4">
                <div className="flex items-center space-x-4">
                  <span className="text-sm text-gray-500 w-8">1</span>
                  <div className="flex-1">
                    <input
                      type="range"
                      min="1"
                      max="10"
                      value={formData.stress_level}
                      onChange={(e) => handleInputChange('stress_level', parseInt(e.target.value))}
                      className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                    />
                  </div>
                  <span className="text-sm text-gray-500 w-8">10</span>
                </div>
                <div className="text-center">
                  <div className="inline-block px-4 py-2 rounded-full bg-orange-100 text-orange-800 font-medium">
                    {formData.stress_level}/10
                  </div>
                </div>
              </div>
            </div>
          </motion.div>

          {/* Sleep, Exercise, Social */}
          <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
            {/* Sleep */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.4 }}
              className="card"
            >
              <div className="card-header">
                <div className="flex items-center space-x-2">
                  <Moon className="w-5 h-5 text-indigo-500" />
                  <h3 className="text-lg font-semibold text-gray-900">Sleep</h3>
                </div>
              </div>
              <div className="card-body">
                <div className="space-y-2">
                  <label className="form-label">Hours of sleep</label>
                  <input
                    type="number"
                    min="0"
                    max="24"
                    step="0.5"
                    value={formData.sleep_hours}
                    onChange={(e) => handleInputChange('sleep_hours', parseFloat(e.target.value))}
                    className="form-input"
                  />
                </div>
              </div>
            </motion.div>

            {/* Exercise */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.5 }}
              className="card"
            >
              <div className="card-header">
                <div className="flex items-center space-x-2">
                  <Dumbbell className="w-5 h-5 text-green-500" />
                  <h3 className="text-lg font-semibold text-gray-900">Exercise</h3>
                </div>
              </div>
              <div className="card-body">
                <div className="space-y-2">
                  <label className="form-label">Minutes of exercise</label>
                  <input
                    type="number"
                    min="0"
                    max="600"
                    value={formData.exercise_minutes}
                    onChange={(e) => handleInputChange('exercise_minutes', parseInt(e.target.value))}
                    className="form-input"
                  />
                </div>
              </div>
            </motion.div>

            {/* Social */}
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ delay: 0.6 }}
              className="card"
            >
              <div className="card-header">
                <div className="flex items-center space-x-2">
                  <Users className="w-5 h-5 text-purple-500" />
                  <h3 className="text-lg font-semibold text-gray-900">Social</h3>
                </div>
              </div>
              <div className="card-body">
                <div className="space-y-2">
                  <label className="form-label">Social interactions</label>
                  <input
                    type="number"
                    min="0"
                    max="50"
                    value={formData.social_interactions}
                    onChange={(e) => handleInputChange('social_interactions', parseInt(e.target.value))}
                    className="form-input"
                  />
                </div>
              </div>
            </motion.div>
          </div>

          {/* Weather */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.7 }}
            className="card"
          >
            <div className="card-header">
              <div className="flex items-center space-x-2">
                <Cloud className="w-5 h-5 text-blue-500" />
                <h3 className="text-lg font-semibold text-gray-900">Weather</h3>
              </div>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
                {weatherOptions.map((weather) => (
                  <button
                    key={weather.value}
                    type="button"
                    onClick={() => handleInputChange('weather', weather.value)}
                    className={`
                      p-3 rounded-lg border transition-all duration-200
                      ${formData.weather === weather.value
                        ? 'bg-blue-50 border-blue-500 text-blue-700'
                        : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                      }
                    `}
                  >
                    <div className="text-center">
                      <div className="text-2xl mb-1">{weather.icon}</div>
                      <div className="text-sm font-medium">{weather.label.split(' ')[0]}</div>
                    </div>
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Triggers */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.8 }}
            className="card"
          >
            <div className="card-header">
              <h3 className="text-lg font-semibold text-gray-900">Triggers</h3>
              <p className="text-sm text-gray-500">What might have affected your mood?</p>
            </div>
            <div className="card-body">
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                {triggerOptions.map((trigger) => (
                  <button
                    key={trigger.value}
                    type="button"
                    onClick={() => handleTriggerToggle(trigger.value)}
                    className={`
                      p-3 rounded-lg border transition-all duration-200 text-left
                      ${formData.triggers.includes(trigger.value)
                        ? 'bg-yellow-50 border-yellow-500 text-yellow-700'
                        : 'bg-gray-50 border-gray-200 hover:bg-gray-100'
                      }
                    `}
                  >
                    <span className="text-sm font-medium">{trigger.label}</span>
                  </button>
                ))}
              </div>
            </div>
          </motion.div>

          {/* Notes */}
          <motion.div
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: 0.9 }}
            className="card"
          >
            <div className="card-header">
              <h3 className="text-lg font-semibold text-gray-900">Notes</h3>
              <p className="text-sm text-gray-500">Any additional thoughts or observations?</p>
            </div>
            <div className="card-body">
              <textarea
                value={formData.notes}
                onChange={(e) => handleInputChange('notes', e.target.value)}
                rows="4"
                className="form-textarea"
                placeholder="Share any thoughts about your day, what went well, or what was challenging..."
              />
            </div>
          </motion.div>

          {/* Submit Button */}
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: 1.0 }}
            className="flex justify-center space-x-4"
          >
            <button
              type="button"
              onClick={() => navigate('/dashboard')}
              className="btn btn-outline"
            >
              Cancel
            </button>
            <button
              type="submit"
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Saving...
                </>
              ) : (
                <>
                  <Save className="w-4 h-4 mr-2" />
                  Save Entry
                </>
              )}
            </button>
          </motion.div>
        </form>
      </motion.div>
    </div>
  );
};

export default MoodEntry; 