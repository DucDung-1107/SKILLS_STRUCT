"""
Main test runner for all SkillStruct APIs
Runs tests for all APIs except RAG API
"""
import subprocess
import sys
import time
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed

# API configurations
APIS = {
    "OCR + Clustering": {
        "port": 8000,
        "script": "test_ocr_clustering_api.py",
        "service": "services/ocr+clusteringAPI.py"
    },
    "JSON Generation": {
        "port": 8001,
        "script": "test_json_generation_api.py", 
        "service": "services/genjsongraphAPI.py"
    },
    "Graph Management": {
        "port": 8002,
        "script": "test_graph_api.py",
        "service": "services/GraphAPI.py"
    },
    "Recommendation": {
        "port": 8003,
        "script": "test_recommendation_api.py",
        "service": "services/recommendapi.py"
    }
}

def check_api_status(name, port):
    """Check if an API is running"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return response.status_code == 200
    except:
        return False

def run_single_test(api_name, config):
    """Run test for a single API"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING {api_name.upper()} API (Port {config['port']})")
    print(f"{'='*60}")
    
    # Check if API is running
    if not check_api_status(api_name, config['port']):
        print(f"❌ {api_name} API is not running on port {config['port']}")
        print(f"   Please start it with: python {config['service']}")
        return False
    
    # Run the test
    try:
        result = subprocess.run([
            sys.executable, f"test/{config['script']}"
        ], capture_output=True, text=True, timeout=60)
        
        print(result.stdout)
        if result.stderr:
            print(f"⚠️ Warnings/Errors:\n{result.stderr}")
        
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"❌ Test for {api_name} API timed out")
        return False
    except Exception as e:
        print(f"❌ Failed to run test for {api_name} API: {e}")
        return False

def run_all_tests_sequential():
    """Run all tests sequentially"""
    print("🚀 SKILLSTRUCT API TEST SUITE")
    print("=" * 80)
    print("Running tests for all APIs sequentially...")
    print("Note: RAG API is excluded from testing")
    print("=" * 80)
    
    results = {}
    
    for api_name, config in APIS.items():
        success = run_single_test(api_name, config)
        results[api_name] = success
        time.sleep(2)  # Brief pause between tests
    
    return results

def run_all_tests_parallel():
    """Run all tests in parallel"""
    print("🚀 SKILLSTRUCT API TEST SUITE (PARALLEL)")
    print("=" * 80)
    print("Running tests for all APIs in parallel...")
    print("Note: RAG API is excluded from testing")
    print("=" * 80)
    
    results = {}
    
    with ThreadPoolExecutor(max_workers=4) as executor:
        # Submit all test jobs
        future_to_api = {
            executor.submit(run_single_test, api_name, config): api_name 
            for api_name, config in APIS.items()
        }
        
        # Collect results
        for future in as_completed(future_to_api):
            api_name = future_to_api[future]
            try:
                results[api_name] = future.result()
            except Exception as e:
                print(f"❌ Test for {api_name} generated an exception: {e}")
                results[api_name] = False
    
    return results

def print_summary(results):
    """Print test summary"""
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)
    
    passed = 0
    failed = 0
    
    for api_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {api_name:<20} : {status}")
        if success:
            passed += 1
        else:
            failed += 1
    
    print("-" * 80)
    print(f"   Total APIs tested: {len(results)}")
    print(f"   Passed: {passed}")
    print(f"   Failed: {failed}")
    print(f"   Success Rate: {(passed/len(results)*100):.1f}%" if results else "0%")
    
    if failed > 0:
        print("\n🛠️ TROUBLESHOOTING:")
        print("   1. Make sure all APIs are running before testing")
        print("   2. Use 'python scripts/start_api.py' to start all APIs")
        print("   3. Check individual API logs for errors")
        print("   4. Verify required dependencies are installed")
    
    print("=" * 80)

def main():
    """Main test runner"""
    import argparse
    
    parser = argparse.ArgumentParser(description="SkillStruct API Test Suite")
    parser.add_argument(
        "--parallel", 
        action="store_true", 
        help="Run tests in parallel (faster but less readable output)"
    )
    parser.add_argument(
        "--api", 
        choices=list(APIS.keys()), 
        help="Test only a specific API"
    )
    
    args = parser.parse_args()
    
    if args.api:
        # Test single API
        config = APIS[args.api]
        success = run_single_test(args.api, config)
        print_summary({args.api: success})
    elif args.parallel:
        # Run all tests in parallel
        results = run_all_tests_parallel()
        print_summary(results)
    else:
        # Run all tests sequentially
        results = run_all_tests_sequential()
        print_summary(results)

if __name__ == "__main__":
    main()
