import React from 'react';
import { motion } from 'framer-motion';
import { 
  Phone, 
  MessageCircle, 
  Heart, 
  AlertTriangle,
  ExternalLink,
  Clock,
  MapPin,
  User
} from 'lucide-react';

const CrisisSupport = () => {
  const emergencyResources = [
    {
      name: 'National Suicide Prevention Lifeline',
      phone: '988',
      description: 'Free, confidential crisis support 24/7',
      available: '24/7',
      icon: Phone,
      color: 'red'
    },
    {
      name: 'Crisis Text Line',
      phone: '741741',
      description: 'Text HOME for immediate crisis support',
      available: '24/7',
      icon: MessageCircle,
      color: 'blue'
    },
    {
      name: 'Emergency Services',
      phone: '911',
      description: 'For immediate medical emergencies',
      available: '24/7',
      icon: AlertTriangle,
      color: 'red'
    }
  ];

  const supportResources = [
    {
      title: 'SAMHSA National Helpline',
      phone: '1-800-662-4357',
      description: 'Treatment referral and information service',
      website: 'https://www.samhsa.gov/find-help/national-helpline'
    },
    {
      title: 'National Alliance on Mental Illness',
      phone: '1-800-950-6264',
      description: 'Support, education, and advocacy',
      website: 'https://www.nami.org/help'
    },
    {
      title: 'Mental Health America',
      phone: '1-800-969-6642',
      description: 'Mental health resources and screening tools',
      website: 'https://www.mhanational.org/get-help'
    }
  ];

  const selfCareStrategies = [
    {
      title: 'Grounding Techniques',
      description: '5-4-3-2-1 technique: Name 5 things you see, 4 you hear, 3 you feel, 2 you smell, 1 you taste',
      icon: '🌱'
    },
    {
      title: 'Deep Breathing',
      description: 'Breathe in for 4 counts, hold for 4, exhale for 4. Repeat 5 times.',
      icon: '🫁'
    },
    {
      title: 'Safe Space',
      description: 'Go to a place where you feel safe and comfortable',
      icon: '🏠'
    },
    {
      title: 'Trusted Person',
      description: 'Call or text someone you trust and feel comfortable talking to',
      icon: '👥'
    }
  ];

  const warningSigns = [
    'Thoughts of self-harm or suicide',
    'Feeling hopeless or worthless',
    'Extreme mood changes',
    'Withdrawing from friends and family',
    'Substance abuse',
    'Reckless behavior',
    'Giving away possessions',
    'Talking about death or dying'
  ];

  return (
    <div className="space-y-6">
      {/* Header */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="text-center"
      >
        <div className="flex justify-center mb-4">
          <div className="w-16 h-16 bg-red-100 rounded-full flex items-center justify-center">
            <Heart className="w-8 h-8 text-red-600" />
          </div>
        </div>
        <h1 className="text-3xl font-bold text-gray-900">Crisis Support</h1>
        <p className="mt-2 text-gray-600 max-w-2xl mx-auto">
          You are not alone. Help is available 24/7. If you're in immediate danger, please call 911.
        </p>
      </motion.div>

      {/* Emergency Resources */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="card border-l-4 border-red-500"
      >
        <div className="card-header">
          <h2 className="text-xl font-semibold text-red-900">Emergency Resources</h2>
          <p className="text-sm text-red-700">Available 24/7 for immediate support</p>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {emergencyResources.map((resource, index) => {
              const Icon = resource.icon;
              return (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, scale: 0.9 }}
                  animate={{ opacity: 1, scale: 1 }}
                  transition={{ delay: 0.2 + index * 0.1 }}
                  className="p-4 bg-red-50 rounded-lg border border-red-200"
                >
                  <div className="flex items-center space-x-3 mb-3">
                    <Icon className={`w-6 h-6 text-${resource.color}-600`} />
                    <h3 className="font-semibold text-gray-900">{resource.name}</h3>
                  </div>
                  <p className="text-sm text-gray-600 mb-3">{resource.description}</p>
                  <div className="flex items-center justify-between">
                    <a
                      href={`tel:${resource.phone}`}
                      className="text-lg font-bold text-red-700 hover:text-red-800"
                    >
                      {resource.phone}
                    </a>
                    <div className="flex items-center text-sm text-gray-500">
                      <Clock className="w-4 h-4 mr-1" />
                      {resource.available}
                    </div>
                  </div>
                </motion.div>
              );
            })}
          </div>
        </div>
      </motion.div>

      {/* Immediate Actions */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="text-xl font-semibold text-gray-900">Immediate Self-Care Strategies</h2>
          <p className="text-sm text-gray-500">Things you can do right now to feel safer</p>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {selfCareStrategies.map((strategy, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.4 + index * 0.1 }}
                className="p-4 bg-blue-50 rounded-lg"
              >
                <div className="flex items-center space-x-3 mb-2">
                  <span className="text-2xl">{strategy.icon}</span>
                  <h3 className="font-semibold text-gray-900">{strategy.title}</h3>
                </div>
                <p className="text-sm text-gray-600">{strategy.description}</p>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Additional Support */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.5 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="text-xl font-semibold text-gray-900">Additional Support Resources</h2>
          <p className="text-sm text-gray-500">Organizations that can provide ongoing support</p>
        </div>
        <div className="card-body">
          <div className="space-y-4">
            {supportResources.map((resource, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 10 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.6 + index * 0.1 }}
                className="p-4 bg-gray-50 rounded-lg"
              >
                <div className="flex items-center justify-between">
                  <div>
                    <h3 className="font-semibold text-gray-900">{resource.title}</h3>
                    <p className="text-sm text-gray-600 mt-1">{resource.description}</p>
                    <div className="flex items-center space-x-4 mt-2">
                      <a
                        href={`tel:${resource.phone}`}
                        className="text-blue-600 hover:text-blue-800 font-medium"
                      >
                        {resource.phone}
                      </a>
                      <a
                        href={resource.website}
                        target="_blank"
                        rel="noopener noreferrer"
                        className="text-blue-600 hover:text-blue-800 flex items-center"
                      >
                        Website
                        <ExternalLink className="w-4 h-4 ml-1" />
                      </a>
                    </div>
                  </div>
                </div>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Warning Signs */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.7 }}
        className="card"
      >
        <div className="card-header">
          <h2 className="text-xl font-semibold text-gray-900">Warning Signs</h2>
          <p className="text-sm text-gray-500">When to seek immediate help</p>
        </div>
        <div className="card-body">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {warningSigns.map((sign, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -10 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: 0.8 + index * 0.05 }}
                className="flex items-start space-x-3"
              >
                <AlertTriangle className="w-5 h-5 text-yellow-500 flex-shrink-0 mt-0.5" />
                <span className="text-sm text-gray-700">{sign}</span>
              </motion.div>
            ))}
          </div>
        </div>
      </motion.div>

      {/* Call to Action */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.9 }}
        className="text-center p-6 bg-gradient-to-r from-blue-50 to-purple-50 rounded-lg"
      >
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          Your life matters. You matter.
        </h3>
        <p className="text-gray-600 mb-4">
          Crisis support is available 24/7. Don't hesitate to reach out for help.
        </p>
        <div className="flex justify-center space-x-4">
          <a
            href="tel:988"
            className="btn btn-danger"
          >
            <Phone className="w-4 h-4 mr-2" />
            Call 988 Now
          </a>
          <a
            href="sms:741741"
            className="btn btn-primary"
          >
            <MessageCircle className="w-4 h-4 mr-2" />
            Text HOME to 741741
          </a>
        </div>
      </motion.div>
    </div>
  );
};

export default CrisisSupport; 