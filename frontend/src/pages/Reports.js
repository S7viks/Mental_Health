import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { 
  FileText, 
  Download, 
  Calendar, 
  BarChart3,
  TrendingUp,
  Eye,
  Share
} from 'lucide-react';
import { format } from 'date-fns';
import toast from 'react-hot-toast';

import { generateReport, exportData } from '../utils/api';

const Reports = ({ moodData, systemStatus }) => {
  const [loading, setLoading] = useState(false);
  const [selectedReport, setSelectedReport] = useState('comprehensive');

  const reportTypes = [
    {
      value: 'comprehensive',
      name: 'Comprehensive Report',
      description: 'Complete mental health analysis with insights and recommendations',
      icon: FileText,
      color: 'blue'
    },
    {
      value: 'weekly',
      name: 'Weekly Summary',
      description: 'Summary of the past week\'s mood patterns',
      icon: Calendar,
      color: 'green'
    },
    {
      value: 'monthly',
      name: 'Monthly Analysis',
      description: 'Detailed monthly mood and pattern analysis',
      icon: BarChart3,
      color: 'purple'
    },
    {
      value: 'professional',
      name: 'Professional Report',
      description: 'Detailed report suitable for healthcare providers',
      icon: TrendingUp,
      color: 'orange'
    }
  ];

  const exportFormats = [
    { value: 'excel', name: 'Excel (.xlsx)', icon: '📊' },
    { value: 'csv', name: 'CSV (.csv)', icon: '📄' },
    { value: 'json', name: 'JSON (.json)', icon: '🔧' }
  ];

  const handleGenerateReport = async () => {
    setLoading(true);
    try {
      const result = await generateReport(selectedReport);
      if (result.success) {
        toast.success('Report generated successfully!');
        // You could display the report or download it
      } else {
        toast.error('Failed to generate report');
      }
    } catch (error) {
      toast.error('Error generating report');
    } finally {
      setLoading(false);
    }
  };

  const handleExportData = async (format) => {
    setLoading(true);
    try {
      const result = await exportData(format);
      
      // Create download link
      const url = window.URL.createObjectURL(result);
      const a = document.createElement('a');
      a.href = url;
      a.download = `mood-data-${format}-${format(new Date(), 'yyyy-MM-dd')}.${format}`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      window.URL.revokeObjectURL(url);
      
      toast.success(`Data exported as ${format.toUpperCase()}`);
    } catch (error) {
      toast.error('Error exporting data');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <h1 className="text-3xl font-bold text-gray-900">Reports</h1>
        <p className="mt-2 text-gray-600">
          Generate comprehensive reports and export your mental health data
        </p>
      </motion.div>

      {/* Report Generation */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="text-xl font-semibold text-gray-900">Generate Report</h2>
          <p className="text-sm text-gray-500">Create detailed analysis reports</p>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
            {reportTypes.map((report, index) => {
              const Icon = report.icon;
              return (
                <motion.div
                  key={report.value}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 + index * 0.1 }}
                  onClick={() => setSelectedReport(report.value)}
                  className={`
                    p-4 rounded-lg border cursor-pointer transition-all duration-200
                    ${selectedReport === report.value
                      ? `border-${report.color}-500 bg-${report.color}-50`
                      : 'border-gray-200 hover:border-gray-300 hover:bg-gray-50'
                    }
                  `}
                >
                  <div className="flex items-center space-x-3 mb-2">
                    <Icon className={`w-6 h-6 text-${report.color}-600`} />
                    <h3 className="font-semibold text-gray-900">{report.name}</h3>
                  </div>
                  <p className="text-sm text-gray-600">{report.description}</p>
                </motion.div>
              );
            })}
          </div>
          
          <div className="flex justify-center">
            <button
              onClick={handleGenerateReport}
              disabled={loading}
              className="btn btn-primary"
            >
              {loading ? (
                <>
                  <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white mr-2"></div>
                  Generating...
                </>
              ) : (
                <>
                  <FileText className="w-4 h-4 mr-2" />
                  Generate Report
                </>
              )}
            </button>
          </div>
        </div>
      </motion.div>

      {/* Data Export */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="text-xl font-semibold text-gray-900">Export Data</h2>
          <p className="text-sm text-gray-500">Download your mood data in various formats</p>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {exportFormats.map((format, index) => (
              <motion.div
                key={format.value}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.4 + index * 0.1 }}
                className="p-4 bg-gray-50 rounded-lg text-center"
              >
                <div className="text-3xl mb-2">{format.icon}</div>
                <h3 className="font-semibold text-gray-900 mb-2">{format.name}</h3>
                <button
                  onClick={() => handleExportData(format.value)}
                  disabled={loading}
                  className="btn btn-outline w-full"
                >
                  <Download className="w-4 h-4 mr-2" />
                  Export
                </button>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Quick Stats */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="text-xl font-semibold text-gray-900">Data Overview</h2>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
            <div className="text-center">
              <div className="text-2xl font-bold text-blue-600">
                {moodData?.length || 0}
              </div>
              <div className="text-sm text-gray-500">Total Entries</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-green-600">
                {moodData?.length > 0 
                  ? (moodData.reduce((sum, entry) => sum + entry.mood_score, 0) / moodData.length).toFixed(1)
                  : '0.0'
                }
              </div>
              <div className="text-sm text-gray-500">Average Mood</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-purple-600">
                {moodData?.length > 0 
                  ? Math.ceil((Date.now() - new Date(moodData[0]?.timestamp).getTime()) / (1000 * 60 * 60 * 24))
                  : 0
                }
              </div>
              <div className="text-sm text-gray-500">Days Tracked</div>
            </div>
            <div className="text-center">
              <div className="text-2xl font-bold text-orange-600">
                {moodData?.filter(entry => entry.mood_score <= 3).length || 0}
              </div>
              <div className="text-sm text-gray-500">Low Mood Days</div>
            </div>
          </div>
        </div>
      </motion.div>

      {/* Report Preview */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="text-xl font-semibold text-gray-900">Sample Report Preview</h2>
        </div>
        <div className="card-body">
          <div className="bg-gray-50 rounded-lg p-6">
            <h3 className="text-lg font-semibold mb-4">Mental Health Analysis Summary</h3>
            <div className="space-y-4">
              <div>
                <h4 className="font-medium text-gray-900">Analysis Period</h4>
                <p className="text-sm text-gray-600">
                  {format(new Date(), 'MMMM dd, yyyy')} - Last 30 days
                </p>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">Key Findings</h4>
                <ul className="text-sm text-gray-600 list-disc ml-5 space-y-1">
                  <li>Average mood score: 6.8/10 (Good)</li>
                  <li>Mood trend: Stable with slight improvement</li>
                  <li>Sleep quality correlation: Strong positive</li>
                  <li>Stress levels: Moderate, manageable</li>
                </ul>
              </div>
              <div>
                <h4 className="font-medium text-gray-900">Recommendations</h4>
                <ul className="text-sm text-gray-600 list-disc ml-5 space-y-1">
                  <li>Continue current self-care routines</li>
                  <li>Maintain consistent sleep schedule</li>
                  <li>Consider stress management techniques</li>
                </ul>
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Reports; 