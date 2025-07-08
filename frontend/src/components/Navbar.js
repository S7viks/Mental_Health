import React from 'react';
import { motion } from 'framer-motion';
import { 
  Brain, 
  Menu, 
  RefreshCw, 
  Bell,
  Activity,
  Clock,
  AlertCircle
} from 'lucide-react';
import { format } from 'date-fns';

const Navbar = ({ 
  sidebarOpen, 
  setSidebarOpen, 
  systemStatus, 
  lastUpdate, 
  onRefresh 
}) => {
  const getStatusColor = (status) => {
    switch (status?.status) {
      case 'active':
      case 'demo':
        return 'text-green-500';
      case 'warning':
        return 'text-yellow-500';
      case 'error':
        return 'text-red-500';
      default:
        return 'text-gray-500';
    }
  };

  const getStatusText = (status) => {
    switch (status?.status) {
      case 'active':
        return 'All Systems Active';
      case 'demo':
        return 'Demo Mode';
      case 'warning':
        return 'System Warning';
      case 'error':
        return 'System Error';
      default:
        return 'System Status Unknown';
    }
  };

  return (
    <motion.nav
      initial={{ y: -100 }}
      animate={{ y: 0 }}
      transition={{ duration: 0.5 }}
      className="fixed top-0 left-0 right-0 z-50 bg-white border-b border-gray-200 shadow-sm"
    >
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left Section */}
          <div className="flex items-center space-x-4">
            {/* Menu Button */}
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
              aria-label="Toggle sidebar"
            >
              <Menu className="w-6 h-6" />
            </button>

            {/* Logo and Title */}
            <div className="flex items-center space-x-3">
              <div className="flex items-center justify-center w-10 h-10 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
                <Brain className="w-6 h-6 text-white" />
              </div>
              <div className="hidden sm:block">
                <h1 className="text-xl font-bold text-gray-900">
                  Mental Health Tracker
                </h1>
                <p className="text-sm text-gray-500">
                  AI-Powered Mood Analysis
                </p>
              </div>
            </div>
          </div>

          {/* Right Section */}
          <div className="flex items-center space-x-4">
            {/* System Status */}
            <div className="hidden md:flex items-center space-x-2 px-3 py-1 bg-gray-50 rounded-lg">
              <Activity className={`w-4 h-4 ${getStatusColor(systemStatus)}`} />
              <span className="text-sm font-medium text-gray-700">
                {getStatusText(systemStatus)}
              </span>
            </div>

            {/* Last Update */}
            {lastUpdate && (
              <div className="hidden lg:flex items-center space-x-2 text-sm text-gray-500">
                <Clock className="w-4 h-4" />
                <span>
                  Updated {format(lastUpdate, 'MMM dd, HH:mm')}
                </span>
              </div>
            )}

            {/* Refresh Button */}
            <button
              onClick={onRefresh}
              className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors"
              aria-label="Refresh data"
            >
              <RefreshCw className="w-5 h-5" />
            </button>

            {/* Crisis Alert Indicator */}
            <div className="relative">
              <button className="p-2 rounded-lg text-gray-500 hover:text-gray-700 hover:bg-gray-100 focus:outline-none focus:ring-2 focus:ring-blue-500 transition-colors">
                <Bell className="w-5 h-5" />
              </button>
              {/* Crisis indicator dot */}
              <motion.div
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                className="absolute -top-1 -right-1 w-3 h-3 bg-red-500 rounded-full border-2 border-white"
                style={{ display: 'none' }} // Will be shown when crisis detected
              />
            </div>

            {/* Settings/Profile */}
            <div className="flex items-center space-x-2">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center">
                <span className="text-white text-sm font-medium">U</span>
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Mobile Status Bar */}
      <div className="md:hidden px-4 py-2 bg-gray-50 border-t border-gray-200">
        <div className="flex items-center justify-between text-sm">
          <div className="flex items-center space-x-2">
            <Activity className={`w-4 h-4 ${getStatusColor(systemStatus)}`} />
            <span className="font-medium text-gray-700">
              {getStatusText(systemStatus)}
            </span>
          </div>
          {lastUpdate && (
            <div className="flex items-center space-x-2 text-gray-500">
              <Clock className="w-4 h-4" />
              <span>
                {format(lastUpdate, 'HH:mm')}
              </span>
            </div>
          )}
        </div>
      </div>
    </motion.nav>
  );
};

export default Navbar; 