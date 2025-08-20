"""
Simple test script for OCR + Clustering API
Port: 8000
"""
import requests
import json
import os

# API endpoint
BASE_URL = "http://localhost:8000"

def test_health_check():
    """Test health check endpoint"""
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check - Status: {response.status_code}")
        if response.status_code == 200:
            print(f"   Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health Check failed: {e}")
        return False

def test_simple():
    """Simple test"""
    try:
        print("Testing OCR + Clustering API (Port 8000)")
        print("=" * 50)
        
        # Test health check first
        if test_health_check():
            print("SUCCESS: API is running")
        else:
            print("FAILED: API is not running. Please start the OCR + Clustering API first.")
            print("   Run: python services/ocr+clusteringAPI.py")
        
        print("=" * 50)
        print("OCR + Clustering API tests completed")
            
    except Exception as e:
        print(f"Test failed: {e}")

if __name__ == "__main__":
    test_simple()
