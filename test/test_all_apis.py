"""
Comprehensive test runner for all SkillStruct APIs
Tests all APIs except RAG API - No emoji version
"""
import requests
import json
import time

# API configurations
APIS = {
    "OCR + Clustering": {
        "port": 8000,
        "endpoints": [
            {"method": "GET", "path": "/health", "name": "Health Check"}
        ]
    },
    "JSON Generation": {
        "port": 8001,
        "endpoints": [
            {"method": "GET", "path": "/health", "name": "Health Check"}
        ]
    },
    "Graph Management": {
        "port": 8002,
        "endpoints": [
            {"method": "GET", "path": "/health", "name": "Health Check"}
        ]
    },
    "Recommendation": {
        "port": 8003,
        "endpoints": [
            {"method": "GET", "path": "/health", "name": "Health Check"}
        ]
    }
}

def test_api_endpoint(base_url, endpoint):
    """Test a single API endpoint"""
    try:
        url = f"{base_url}{endpoint['path']}"
        
        if endpoint['method'] == 'GET':
            response = requests.get(url, timeout=10)
        elif endpoint['method'] == 'POST':
            response = requests.post(url, json=endpoint.get('data', {}), timeout=10)
        
        success = 200 <= response.status_code < 300
        return {
            'success': success,
            'status_code': response.status_code,
            'response': response.json() if success else response.text
        }
    except Exception as e:
        return {
            'success': False,
            'status_code': 0,
            'response': str(e)
        }

def test_single_api(api_name, config):
    """Test all endpoints of a single API"""
    print(f"\nTesting {api_name} API (Port {config['port']})")
    print("-" * 50)
    
    base_url = f"http://localhost:{config['port']}"
    results = []
    
    for endpoint in config['endpoints']:
        print(f"  Testing {endpoint['name']}...", end="")
        result = test_api_endpoint(base_url, endpoint)
        
        if result['success']:
            print(" PASSED")
            print(f"    Status: {result['status_code']}")
        else:
            print(" FAILED")
            print(f"    Status: {result['status_code']}")
            print(f"    Error: {result['response']}")
        
        results.append({
            'endpoint': endpoint['name'],
            'success': result['success']
        })
    
    api_success = all(r['success'] for r in results)
    return api_success, results

def run_all_tests():
    """Run tests for all APIs"""
    print("SKILLSTRUCT API TEST SUITE")
    print("=" * 60)
    print("Running tests for all APIs (except RAG API)...")
    print("=" * 60)
    
    all_results = {}
    
    for api_name, config in APIS.items():
        api_success, endpoint_results = test_single_api(api_name, config)
        all_results[api_name] = {
            'success': api_success,
            'endpoints': endpoint_results
        }
        time.sleep(1)  # Brief pause between API tests
    
    # Print summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    total_passed = 0
    total_failed = 0
    
    for api_name, result in all_results.items():
        status = "PASSED" if result['success'] else "FAILED"
        print(f"  {api_name:<20} : {status}")
        
        if result['success']:
            total_passed += 1
        else:
            total_failed += 1
    
    print("-" * 60)
    print(f"  Total APIs: {len(all_results)}")
    print(f"  Passed: {total_passed}")
    print(f"  Failed: {total_failed}")
    print(f"  Success Rate: {(total_passed/len(all_results)*100):.1f}%")
    
    if total_failed > 0:
        print("\nTROUBLESHOOTING:")
        print("  1. Make sure all APIs are running")
        print("  2. Use 'python scripts/start_api.py' to start APIs")
        print("  3. Check individual API logs for errors")
    
    print("=" * 60)

if __name__ == "__main__":
    run_all_tests()
