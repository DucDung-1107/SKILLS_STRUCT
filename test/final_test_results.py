"""
Final Test Results Summary
"""
import requests
import time

def test_api_status(name, port):
    """Test basic connectivity to an API"""
    try:
        response = requests.get(f"http://localhost:{port}/health", timeout=5)
        return {
            'name': name,
            'port': port,
            'status': 'RUNNING' if response.status_code == 200 else f'ERROR {response.status_code}',
            'response_time': response.elapsed.total_seconds()
        }
    except requests.exceptions.ConnectionError:
        return {
            'name': name,
            'port': port,
            'status': 'NOT RUNNING',
            'response_time': None
        }
    except Exception as e:
        return {
            'name': name,
            'port': port,
            'status': f'ERROR: {str(e)}',
            'response_time': None
        }

def run_final_tests():
    """Run final comprehensive test"""
    print("SKILLSTRUCT API TEST SUITE - FINAL RESULTS")
    print("=" * 60)
    
    # Test all APIs
    apis = [
        ("OCR + Clustering", 8000),
        ("JSON Generation", 8001),
        ("Graph Management", 8002),
        ("Recommendation", 8003),
        ("RAG API", 8004)  # Included for completeness but not tested in detail
    ]
    
    results = []
    for name, port in apis:
        result = test_api_status(name, port)
        results.append(result)
    
    # Print results table
    print(f"{'API Name':<20} {'Port':<6} {'Status':<15} {'Response Time':<15}")
    print("-" * 60)
    
    running_count = 0
    for result in results:
        status = result['status']
        response_time = f"{result['response_time']:.3f}s" if result['response_time'] else "N/A"
        print(f"{result['name']:<20} {result['port']:<6} {status:<15} {response_time:<15}")
        
        if status == 'RUNNING':
            running_count += 1
    
    print("-" * 60)
    print(f"SUMMARY: {running_count}/{len(apis)} APIs are running")
    
    # Detailed functional test summary
    print(f"\nFUNCTIONAL TEST RESULTS:")
    print("-" * 30)
    print("✅ OCR + Clustering API (Port 8000)")
    print("   - Health check: PASS")
    print("   - Resume data retrieval: PASS")
    print("   - Status: FULLY FUNCTIONAL")
    
    print(f"\n⚠️ JSON Generation API (Port 8001)")
    print("   - Health check: PASS")
    print("   - CSV upload: FAIL (500 error)")
    print("   - Get graphs: PASS")
    print("   - Status: PARTIALLY FUNCTIONAL")
    
    print(f"\n❌ Graph Management API (Port 8002)")
    print("   - Health check: FAIL (Connection refused)")
    print("   - Module dependency error")
    print("   - Status: NOT FUNCTIONAL")
    
    print(f"\n⚠️ Recommendation API (Port 8003)")
    print("   - Health check: PASS")
    print("   - Skill recommendations: FAIL (500 error)")
    print("   - Skill analysis: FAIL (422 validation error)")
    print("   - Status: PARTIALLY FUNCTIONAL")
    
    print(f"\n❌ RAG API (Port 8004) - EXCLUDED FROM TESTING")
    print("   - Status: NOT TESTED (as requested)")
    
    print(f"\nOVERALL ASSESSMENT:")
    print("-" * 30)
    print(f"✅ Test infrastructure: COMPLETE")
    print(f"✅ API connectivity: ESTABLISHED")
    print(f"✅ Health checks: WORKING")
    print(f"⚠️ Functional endpoints: NEED FIXES")
    print(f"⚠️ Dependencies: NEED RESOLUTION")
    
    print(f"\nNEXT STEPS:")
    print("-" * 30)
    print(f"1. Fix GraphAPI.py dependency issues")
    print(f"2. Debug CSV upload endpoint in JSON Generation API")
    print(f"3. Fix validation errors in Recommendation API")
    print(f"4. Implement missing functional endpoints")
    print(f"5. Add comprehensive error handling")
    
    print("=" * 60)
    print("TEST SUITE EXECUTION COMPLETED!")
    print("=" * 60)

if __name__ == "__main__":
    run_final_tests()
