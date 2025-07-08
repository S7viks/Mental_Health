import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import toast from 'react-hot-toast';

// Components
import Navbar from './components/Navbar';
import Sidebar from './components/Sidebar';
import LoadingSpinner from './components/LoadingSpinner';

// Pages
import Dashboard from './pages/Dashboard';
import MoodEntry from './pages/MoodEntry';
import Analytics from './pages/Analytics';
import Reports from './pages/Reports';
import Settings from './pages/Settings';
import CrisisSupport from './pages/CrisisSupport';

// Utils
import { getMoodData, getSystemStatus } from './utils/api';

// Styles
import './styles/App.css';

function App() {
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [loading, setLoading] = useState(true);
  const [systemStatus, setSystemStatus] = useState(null);
  const [moodData, setMoodData] = useState([]);
  const [lastUpdate, setLastUpdate] = useState(null);

  useEffect(() => {
    initializeApp();
  }, []);

  const initializeApp = async () => {
    try {
      setLoading(true);
      
      // Check system status
      const status = await getSystemStatus();
      setSystemStatus(status);
      
      // Load mood data
      const data = await getMoodData();
      setMoodData(data);
      setLastUpdate(new Date());
      
      toast.success('🧠 Mental Health Tracker loaded successfully!');
    } catch (error) {
      console.error('Failed to initialize app:', error);
      toast.error('Failed to load application. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const refreshData = async () => {
    try {
      const data = await getMoodData();
      setMoodData(data);
      setLastUpdate(new Date());
      toast.success('Data refreshed successfully!');
    } catch (error) {
      console.error('Failed to refresh data:', error);
      toast.error('Failed to refresh data. Please try again.');
    }
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-gradient-to-br from-blue-50 to-purple-50 flex items-center justify-center">
        <LoadingSpinner />
      </div>
    );
  }

  return (
    <Router>
      <div className="min-h-screen bg-gray-50">
        {/* Navigation */}
        <Navbar 
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          systemStatus={systemStatus}
          lastUpdate={lastUpdate}
          onRefresh={refreshData}
        />

        {/* Sidebar */}
        <Sidebar 
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
        />

        {/* Main Content */}
        <main className={`transition-all duration-300 ${sidebarOpen ? 'lg:ml-64' : ''}`}>
          <div className="pt-16">
            <motion.div
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.5 }}
              className="container mx-auto px-4 py-6"
            >
              <Routes>
                <Route path="/" element={<Navigate to="/dashboard" replace />} />
                <Route 
                  path="/dashboard" 
                  element={
                    <Dashboard 
                      moodData={moodData}
                      systemStatus={systemStatus}
                      onRefresh={refreshData}
                    />
                  } 
                />
                <Route 
                  path="/mood-entry" 
                  element={
                    <MoodEntry 
                      onEntrySubmit={refreshData}
                    />
                  } 
                />
                <Route 
                  path="/analytics" 
                  element={
                    <Analytics 
                      moodData={moodData}
                      systemStatus={systemStatus}
                    />
                  } 
                />
                <Route 
                  path="/reports" 
                  element={
                    <Reports 
                      moodData={moodData}
                      systemStatus={systemStatus}
                    />
                  } 
                />
                <Route 
                  path="/crisis-support" 
                  element={<CrisisSupport />} 
                />
                <Route 
                  path="/settings" 
                  element={
                    <Settings 
                      systemStatus={systemStatus}
                      onRefresh={refreshData}
                    />
                  } 
                />
                <Route path="*" element={<Navigate to="/dashboard" replace />} />
              </Routes>
            </motion.div>
          </div>
        </main>
      </div>
    </Router>
  );
}

export default App; 