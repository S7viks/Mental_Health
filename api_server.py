"""
Flask API Server for Mental Health Mood Tracker
Bridges React frontend with Python backend system
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Any

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our mental health tracker system
from main import MentalHealthMoodTracker

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize Mental Health Tracker
tracker = MentalHealthMoodTracker()

@app.route('/api/system/status', methods=['GET'])
def get_system_status():
    """Get system status and health check"""
    try:
        return jsonify({
            'status': 'active',
            'version': '1.0.0',
            'last_update': datetime.now().isoformat(),
            'modules': {
                'data_collection': 'active',
                'trend_analysis': 'active',
                'pattern_detection': 'active',
                'recommendations': 'active',
                'crisis_detection': 'active'
            },
            'stats': {
                'total_entries': len(tracker.current_data) if tracker.current_data else 0,
                'last_analysis': tracker.last_analysis_date.isoformat() if tracker.last_analysis_date else None
            }
        })
    except Exception as e:
        logger.error(f"System status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mood/data', methods=['GET'])
def get_mood_data():
    """Get mood tracking data"""
    try:
        days_back = request.args.get('days_back', 30, type=int)
        
        # Load current data
        if not tracker.current_data:
            tracker.current_data = tracker.storage_manager.load_data()
        
        # Filter data for requested period
        if tracker.current_data:
            cutoff_date = datetime.now() - timedelta(days=days_back)
            filtered_data = [
                entry for entry in tracker.current_data 
                if datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')).replace(tzinfo=None) >= cutoff_date
            ]
            return jsonify(filtered_data)
        else:
            return jsonify([])
            
    except Exception as e:
        logger.error(f"Get mood data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/mood/entry', methods=['POST'])
def submit_mood_entry():
    """Submit a new mood entry"""
    try:
        data = request.get_json()
        
        # Extract mood entry data
        mood_score = data.get('mood_score')
        stress_level = data.get('stress_level')
        sleep_hours = data.get('sleep_hours')
        exercise_minutes = data.get('exercise_minutes', 0)
        social_interactions = data.get('social_interactions', 0)
        emotions = data.get('emotions', [])
        notes = data.get('notes', '')
        
        # Validate required fields
        if mood_score is None or stress_level is None or sleep_hours is None:
            return jsonify({'error': 'Missing required fields'}), 400
        
        # Submit entry through tracker
        result = tracker.collect_mood_entry(
            mood_score=int(mood_score),
            stress_level=int(stress_level),
            sleep_hours=float(sleep_hours),
            exercise_minutes=int(exercise_minutes),
            social_interactions=int(social_interactions),
            emotions=emotions,
            notes=notes
        )
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Submit mood entry error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/analysis/comprehensive', methods=['POST'])
def run_comprehensive_analysis():
    """Run comprehensive mood analysis"""
    try:
        data = request.get_json() or {}
        days_back = data.get('days_back', 30)
        
        # Run analysis through tracker
        result = tracker.run_comprehensive_analysis(days_back=days_back)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Comprehensive analysis error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/insights/daily', methods=['GET'])
def get_daily_insights():
    """Get daily insights and recommendations"""
    try:
        result = tracker.get_daily_insights()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Daily insights error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/crisis/status', methods=['GET'])
def get_crisis_status():
    """Get current crisis status"""
    try:
        result = tracker.get_crisis_status()
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Crisis status error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/export/data', methods=['GET'])
def export_data():
    """Export mood data"""
    try:
        format_type = request.args.get('format', 'excel')
        filename = request.args.get('filename')
        
        result = tracker.export_data(format=format_type, filename=filename)
        
        if result.get('success'):
            return send_file(
                result['filename'],
                as_attachment=True,
                download_name=os.path.basename(result['filename'])
            )
        else:
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Export data error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/reports/generate', methods=['POST'])
def generate_report():
    """Generate analysis report"""
    try:
        data = request.get_json() or {}
        report_type = data.get('report_type', 'comprehensive')
        
        result = tracker.generate_report(report_type=report_type)
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Generate report error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """API health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.now().isoformat(),
        'version': '1.0.0'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    # Ensure data directories exist
    os.makedirs('data', exist_ok=True)
    os.makedirs('data/json', exist_ok=True)
    os.makedirs('data/excel', exist_ok=True)
    
    logger.info("🚀 Mental Health Mood Tracker API Server starting...")
    logger.info("🧠 AI/ML backend system initialized")
    logger.info("🌐 CORS enabled for React frontend")
    logger.info("📊 All endpoints ready for connections")
    
    # Run Flask development server
    app.run(
        host='127.0.0.1',
        port=8000,
        debug=True,
        threaded=True
    ) 