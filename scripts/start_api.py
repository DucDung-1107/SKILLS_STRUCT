#!/usr/bin/env python3
"""
🚀 API Launcher Script
Khởi động tất cả APIs cần thiết cho Skill Graph Platform
"""

import subprocess
import time
import sys
import os
from pathlib import Path

# API configurations
APIs = [
    {
        "name": "OCR & Clustering API",
        "file": "services/ocr+clusteringAPI.py",
        "port": 8000,
        "module": "services.ocr+clusteringAPI"
    },
    {
        "name": "JSON Generation API", 
        "file": "services/genjsongraphAPI.py",
        "port": 8001,
        "module": "services.genjsongraphAPI"
    },
    {
        "name": "Graph Management API",
        "file": "services/GraphAPI.py", 
        "port": 8002,
        "module": "services.GraphAPI"
    },
    {
        "name": "Recommendation API",
        "file": "services/recommendapi.py",
        "port": 8003,
        "module": "services.recommendapi"
    },
    {
        "name": "RAG Search API",
        "file": "rag/services/rag_api.py",
        "port": 8004,
        "module": "rag.services.rag_api"
    }
]

def check_file_exists(file_path):
    """Kiểm tra file có tồn tại không"""
    return os.path.exists(file_path)

def start_api(api_config):
    """Start một API server"""
    try:
        print(f"🚀 Starting {api_config['name']} on port {api_config['port']}...")
        
        # Check if file exists
        if not check_file_exists(api_config['file']):
            print(f"❌ File not found: {api_config['file']}")
            return None
        
        # Special handling for files with special characters
        if 'ocr+clusteringAPI' in api_config['file']:
            # For OCR API, start from its directory
            cmd = [
                sys.executable, "-c",
                "import os; os.chdir('services'); import sys; sys.path.insert(0, '.'); "
                "import uvicorn; uvicorn.run('ocr+clusteringAPI:app', host='127.0.0.1', port=8000, reload=True)"
            ]
        else:
            # Start the API using uvicorn with corrected module handling
            module_name = api_config['module']
            
            cmd = [
                sys.executable, "-m", "uvicorn",
                f"{module_name.replace('/', '.').replace('.py', '')}:app",
                "--host", "127.0.0.1",
                "--port", str(api_config['port']),
                "--reload"
            ]
        
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        print(f" {api_config['name']} started on http://127.0.0.1:{api_config['port']}")
        return process
        
    except Exception as e:
        print(f" Failed to start {api_config['name']}: {e}")
        return None

def main():
    """Main launcher"""
    print("=" * 60)
    print("🚀 SKILL GRAPH PLATFORM - API LAUNCHER")
    print("=" * 60)
    
    processes = []
    
    # Check all API files first
    print("\n📋 Checking API files...")
    for api in APIs:
        if check_file_exists(api['file']):
            print(f"✅ {api['name']}: {api['file']}")
        else:
            print(f"❌ {api['name']}: {api['file']} (NOT FOUND)")
    
    print("\n🚀 Starting APIs...")
    
    # Start each API
    for api in APIs:
        if check_file_exists(api['file']):
            process = start_api(api)
            if process:
                processes.append((api, process))
            time.sleep(2)  # Wait between starts
        else:
            print(f"⚠️ Skipping {api['name']} - file not found")
    
    if processes:
        print(f"\n✅ Started {len(processes)} APIs successfully!")
        print("\n📚 API Documentation Links:")
        for api, _ in processes:
            print(f"   🔗 {api['name']}: http://127.0.0.1:{api['port']}/docs")
        
        print("\n🎯 Streamlit App:")
        print("   🔗 Main App: streamlit run streamlit_app.py")
        
        print("\n💡 Press Ctrl+C to stop all APIs")
        
        try:
            # Keep processes running
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n\n🛑 Stopping all APIs...")
            for api, process in processes:
                print(f"🔴 Stopping {api['name']}...")
                process.terminate()
            print("✅ All APIs stopped.")
    else:
        print("\n❌ No APIs were started successfully.")
        print("💡 Please check API files and try again.")

if __name__ == "__main__":
    main()
