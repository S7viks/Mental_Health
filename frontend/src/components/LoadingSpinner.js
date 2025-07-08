import React from 'react';
import { motion } from 'framer-motion';

const LoadingSpinner = ({ size = 'medium', message = 'Loading...' }) => {
  const sizeClasses = {
    small: 'w-6 h-6',
    medium: 'w-12 h-12',
    large: 'w-20 h-20'
  };

  return (
    <div className="flex flex-col items-center justify-center p-8">
      <motion.div
        animate={{ rotate: 360 }}
        transition={{
          duration: 1,
          repeat: Infinity,
          ease: "linear"
        }}
        className={`${sizeClasses[size]} border-4 border-blue-200 border-t-blue-600 rounded-full`}
      />
      
      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="mt-4"
      >
        <div className="flex items-center space-x-2">
          <span className="text-gray-600 font-medium">{message}</span>
          <motion.div
            animate={{ opacity: [1, 0.3, 1] }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: "easeInOut"
            }}
            className="flex space-x-1"
          >
            <div className="w-2 h-2 bg-blue-600 rounded-full" />
            <div className="w-2 h-2 bg-blue-600 rounded-full" />
            <div className="w-2 h-2 bg-blue-600 rounded-full" />
          </motion.div>
        </div>
      </motion.div>
    </div>
  );
};

export default LoadingSpinner; 