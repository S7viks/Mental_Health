#!/usr/bin/env python3
"""
Mental Health Mood Tracker - UI Start Script
Starts both the Flask API server and React development server
"""

import os
import sys
import subprocess
import threading
import time
import webbrowser
import signal
from pathlib import Path

def print_banner():
    """Print startup banner"""
    print("\n" + "="*60)
    print("🧠 MENTAL HEALTH MOOD TRACKER - UI STARTUP")
    print("="*60)
    print("🎯 Starting comprehensive mental health tracking interface")
    print("🚀 AI-powered mood analysis with React dashboard")
    print("="*60 + "\n")

def check_dependencies():
    """Check if required dependencies are installed"""
    print("🔍 Checking dependencies...")
    
    # Check Python dependencies
    try:
        import flask
        import flask_cors
        print("✅ Python Flask dependencies installed")
    except ImportError as e:
        print(f"❌ Missing Python dependency: {e}")
        print("💡 Run: pip install -r requirements.txt")
        return False
    
    # Check if Node.js is available
    try:
        result = subprocess.run(['node', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ Node.js found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ Node.js not found")
        print("💡 Please install Node.js from https://nodejs.org/")
        return False
    
    # Check if npm is available
    try:
        result = subprocess.run(['npm', '--version'], capture_output=True, text=True, check=True)
        print(f"✅ npm found: {result.stdout.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("❌ npm not found")
        return False
    
    return True

def install_frontend_dependencies():
    """Install frontend dependencies if needed"""
    frontend_dir = Path("frontend")
    
    if not frontend_dir.exists():
        print("❌ Frontend directory not found")
        return False
    
    node_modules = frontend_dir / "node_modules"
    package_json = frontend_dir / "package.json"
    
    if not package_json.exists():
        print("❌ package.json not found in frontend directory")
        return False
    
    if not node_modules.exists():
        print("📦 Installing frontend dependencies...")
        try:
            result = subprocess.run(
                ['npm', 'install'],
                cwd=frontend_dir,
                check=True,
                capture_output=True,
                text=True
            )
            print("✅ Frontend dependencies installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install frontend dependencies: {e}")
            print(f"Error output: {e.stderr}")
            return False
    else:
        print("✅ Frontend dependencies already installed")
    
    return True

def start_api_server():
    """Start the Flask API server"""
    print("🔧 Starting Flask API server...")
    try:
        # Import and start the API server
        import api_server
        # The Flask app will run on port 8000
    except Exception as e:
        print(f"❌ Failed to start API server: {e}")
        sys.exit(1)

def start_frontend_server():
    """Start the React development server"""
    print("⚛️  Starting React development server...")
    frontend_dir = Path("frontend")
    
    try:
        # Set environment variables
        env = os.environ.copy()
        env['REACT_APP_API_URL'] = 'http://localhost:8000/api'
        env['BROWSER'] = 'none'  # Prevent React from opening browser automatically
        
        # Start React development server
        process = subprocess.Popen(
            ['npm', 'start'],
            cwd=frontend_dir,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Wait for React server to start
        print("⏳ Waiting for React server to start...")
        time.sleep(10)  # Give React time to compile
        
        return process
        
    except Exception as e:
        print(f"❌ Failed to start frontend server: {e}")
        return None

def open_browser():
    """Open the application in the default browser"""
    print("🌐 Opening Mental Health Tracker in your browser...")
    time.sleep(3)  # Wait a bit more for servers to be ready
    webbrowser.open('http://localhost:3000')

def signal_handler(signum, frame):
    """Handle shutdown signals"""
    print("\n🛑 Shutting down Mental Health Tracker...")
    print("👋 Thank you for using the Mental Health Mood Tracker!")
    sys.exit(0)

def main():
    """Main function to start the application"""
    # Set up signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Install frontend dependencies
    if not install_frontend_dependencies():
        sys.exit(1)
    
    print("\n🚀 Starting Mental Health Tracker servers...")
    print("📊 API Server: http://localhost:8000")
    print("🎨 Frontend: http://localhost:3000")
    print("\n" + "="*60)
    
    try:
        # Start API server in a separate thread
        api_thread = threading.Thread(target=start_api_server, daemon=True)
        api_thread.start()
        
        # Give API server time to start
        time.sleep(3)
        
        # Start frontend server
        frontend_process = start_frontend_server()
        
        if frontend_process is None:
            print("❌ Failed to start frontend server")
            sys.exit(1)
        
        # Open browser
        browser_thread = threading.Thread(target=open_browser, daemon=True)
        browser_thread.start()
        
        print("\n🎉 Mental Health Tracker is now running!")
        print("📱 Access your dashboard at: http://localhost:3000")
        print("🔧 API available at: http://localhost:8000/api")
        print("\n💡 Tips:")
        print("   • Start by logging your first mood entry")
        print("   • View analytics after a few entries")
        print("   • Check crisis support resources anytime")
        print("   • Export your data from the Reports section")
        print("\n⚠️  Press Ctrl+C to stop all servers")
        print("="*60)
        
        # Keep the main thread alive and monitor the frontend process
        try:
            frontend_process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping servers...")
            frontend_process.terminate()
            frontend_process.wait()
            
    except Exception as e:
        print(f"❌ Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 