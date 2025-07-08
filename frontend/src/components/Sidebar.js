import React from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  LayoutDashboard,
  PenTool,
  TrendingUp,
  FileText,
  Heart,
  Settings,
  X,
  AlertTriangle
} from 'lucide-react';

const Sidebar = ({ sidebarOpen, setSidebarOpen }) => {
  const location = useLocation();

  const navigationItems = [
    {
      name: 'Dashboard',
      href: '/dashboard',
      icon: LayoutDashboard,
      description: 'Overview and quick insights'
    },
    {
      name: 'Mood Entry',
      href: '/mood-entry',
      icon: PenTool,
      description: 'Log your daily mood'
    },
    {
      name: 'Analytics',
      href: '/analytics',
      icon: TrendingUp,
      description: 'Detailed analysis and trends'
    },
    {
      name: 'Reports',
      href: '/reports',
      icon: FileText,
      description: 'Generate comprehensive reports'
    },
    {
      name: 'Crisis Support',
      href: '/crisis-support',
      icon: AlertTriangle,
      description: 'Emergency resources and support',
      warning: true
    },
    {
      name: 'Settings',
      href: '/settings',
      icon: Settings,
      description: 'Preferences and configuration'
    }
  ];

  const sidebarVariants = {
    open: {
      x: 0,
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30
      }
    },
    closed: {
      x: "-100%",
      transition: {
        type: "spring",
        stiffness: 300,
        damping: 30
      }
    }
  };

  const itemVariants = {
    hidden: { opacity: 0, x: -20 },
    visible: (index) => ({
      opacity: 1,
      x: 0,
      transition: {
        delay: index * 0.1,
        duration: 0.3
      }
    })
  };

  return (
    <>
      {/* Overlay for mobile */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={() => setSidebarOpen(false)}
            className="fixed inset-0 z-40 bg-black bg-opacity-50 lg:hidden"
          />
        )}
      </AnimatePresence>

      {/* Sidebar */}
      <motion.aside
        variants={sidebarVariants}
        animate={sidebarOpen ? "open" : "closed"}
        className="fixed top-0 left-0 z-50 w-64 h-full bg-white border-r border-gray-200 shadow-lg lg:translate-x-0"
      >
        {/* Header */}
        <div className="flex items-center justify-between p-6 border-b border-gray-200">
          <div className="flex items-center space-x-3">
            <div className="flex items-center justify-center w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg">
              <Heart className="w-5 h-5 text-white" />
            </div>
            <div>
              <h2 className="text-lg font-semibold text-gray-900">
                MoodTracker
              </h2>
              <p className="text-xs text-gray-500">v1.0.0</p>
            </div>
          </div>
          
          <button
            onClick={() => setSidebarOpen(false)}
            className="p-1 rounded-lg text-gray-400 hover:text-gray-600 hover:bg-gray-100 lg:hidden"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-4 py-6 space-y-2">
          {navigationItems.map((item, index) => {
            const isActive = location.pathname === item.href;
            const Icon = item.icon;
            
            return (
              <motion.div
                key={item.name}
                custom={index}
                variants={itemVariants}
                initial="hidden"
                animate="visible"
              >
                <NavLink
                  to={item.href}
                  onClick={() => setSidebarOpen(false)}
                  className={`
                    group flex items-center px-3 py-3 rounded-lg text-sm font-medium transition-all duration-200
                    ${isActive 
                      ? 'bg-blue-50 text-blue-700 border-r-2 border-blue-500' 
                      : 'text-gray-700 hover:bg-gray-50 hover:text-gray-900'
                    }
                    ${item.warning ? 'hover:bg-red-50 hover:text-red-700' : ''}
                  `}
                >
                  <Icon 
                    className={`
                      w-5 h-5 mr-3 flex-shrink-0
                      ${isActive ? 'text-blue-500' : 'text-gray-400 group-hover:text-gray-500'}
                      ${item.warning ? 'text-red-500' : ''}
                    `} 
                  />
                  
                  <div className="flex-1">
                    <div className={`
                      font-medium
                      ${item.warning ? 'text-red-700' : ''}
                    `}>
                      {item.name}
                    </div>
                    <div className="text-xs text-gray-500 mt-0.5">
                      {item.description}
                    </div>
                  </div>

                  {/* Active indicator */}
                  {isActive && (
                    <motion.div
                      layoutId="activeIndicator"
                      className="w-2 h-2 bg-blue-500 rounded-full"
                    />
                  )}

                  {/* Warning indicator */}
                  {item.warning && (
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ repeat: Infinity, duration: 2 }}
                      className="w-2 h-2 bg-red-500 rounded-full"
                    />
                  )}
                </NavLink>
              </motion.div>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200">
          <div className="flex items-center space-x-3 p-3 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg">
            <div className="flex-shrink-0">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-400 to-purple-500 rounded-full flex items-center justify-center">
                <Heart className="w-4 h-4 text-white" />
              </div>
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-gray-900">
                Your Wellbeing
              </p>
              <p className="text-xs text-gray-500">
                Take care of yourself today
              </p>
            </div>
          </div>

          {/* Quick Stats */}
          <div className="mt-4 grid grid-cols-2 gap-2">
            <div className="text-center p-2 bg-gray-50 rounded-lg">
              <div className="text-lg font-bold text-gray-900">7</div>
              <div className="text-xs text-gray-500">Day Streak</div>
            </div>
            <div className="text-center p-2 bg-gray-50 rounded-lg">
              <div className="text-lg font-bold text-blue-600">6.8</div>
              <div className="text-xs text-gray-500">Avg Mood</div>
            </div>
          </div>
        </div>
      </motion.aside>
    </>
  );
};

export default Sidebar; 