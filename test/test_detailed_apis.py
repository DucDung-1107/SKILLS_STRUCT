"""
Detailed API functionality tests
Tests actual API endpoints with real data
"""
import requests
import json
import os

def test_ocr_api_detailed():
    """Test OCR API with actual functionality"""
    print("\nDetailed OCR + Clustering API Test")
    print("-" * 40)
    
    base_url = "http://localhost:8000"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
    except Exception as e:
        print(f"Health Check: FAIL - {e}")
        return
    
    # Test 2: Get resume data (actual endpoint)
    try:
        response = requests.get(f"{base_url}/resume-data/")
        print(f"Resume Data: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Found {len(result)} resume records")
    except Exception as e:
        print(f"Resume Data: FAIL - {e}")

def test_json_api_detailed():
    """Test JSON Generation API with actual functionality"""
    print("\nDetailed JSON Generation API Test")
    print("-" * 40)
    
    base_url = "http://localhost:8001"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
    except Exception as e:
        print(f"Health Check: FAIL - {e}")
        return
    
    # Test 2: Upload CSV data (actual endpoint)
    try:
        # Test with sample CSV data
        csv_content = "name,skills,experience\nJohn Doe,Python;Java,5\nJane Smith,React;Node.js,3"
        files = {'file': ('test_data.csv', csv_content, 'text/csv')}
        response = requests.post(f"{base_url}/upload-csv", files=files)
        print(f"CSV Upload: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
        if response.status_code == 200:
            result = response.json()
            print(f"  CSV processed successfully")
    except Exception as e:
        print(f"CSV Upload: FAIL - {e}")
        
    # Test 3: Get all graphs
    try:
        response = requests.get(f"{base_url}/graphs")
        print(f"Get Graphs: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Found {len(result)} graphs")
    except Exception as e:
        print(f"Get Graphs: FAIL - {e}")

def test_graph_api_detailed():
    """Test Graph Management API with actual functionality"""
    print("\nDetailed Graph Management API Test")
    print("-" * 40)
    
    base_url = "http://localhost:8002"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
    except Exception as e:
        print(f"Health Check: FAIL - {e}")
        return
    
    # Test 2: Get all nodes (actual endpoint)
    try:
        response = requests.get(f"{base_url}/nodes")
        print(f"Get Nodes: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Found {len(result)} nodes")
    except Exception as e:
        print(f"Get Nodes: FAIL - {e}")
        
    # Test 3: Get taxonomy
    try:
        response = requests.get(f"{base_url}/taxonomy")
        print(f"Get Taxonomy: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Taxonomy loaded successfully")
    except Exception as e:
        print(f"Get Taxonomy: FAIL - {e}")

def test_recommendation_api_detailed():
    """Test Recommendation API with actual functionality"""
    print("\nDetailed Recommendation API Test")
    print("-" * 40)
    
    base_url = "http://localhost:8003"
    
    # Test 1: Health check
    try:
        response = requests.get(f"{base_url}/health")
        print(f"Health Check: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
    except Exception as e:
        print(f"Health Check: FAIL - {e}")
        return
    
    # Test 2: Get skill recommendations (actual endpoint)
    try:
        # Cần file path thực tế cho recommendation API
        skill_file_path = os.path.join(os.path.dirname(__file__), "..", "data", "skill_taxonomy_tree", "output.json")
        if not os.path.exists(skill_file_path):
            # Tạo một file test đơn giản
            test_skill_data = {
                "nodes": [
                    {"id": "1", "name": "Python", "type": "skill"},
                    {"id": "2", "name": "SQL", "type": "skill"}
                ],
                "edges": []
            }
            os.makedirs(os.path.dirname(skill_file_path), exist_ok=True)
            with open(skill_file_path, 'w') as f:
                json.dump(test_skill_data, f)
        
        test_data = {
            "file_path": skill_file_path,
            "max_recommendations": 3
        }
        response = requests.post(f"{base_url}/recommend", json=test_data)
        print(f"Skill Recommendations: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Recommendations generated: {result.get('recommendations_count', 0)}")
    except Exception as e:
        print(f"Skill Recommendations: FAIL - {e}")
        
    # Test 3: Analyze skills
    try:
        # Tương tự cho analyze endpoint
        test_data = {
            "file_path": skill_file_path
        }
        response = requests.post(f"{base_url}/analyze", json=test_data)
        print(f"Skill Analysis: {response.status_code} - {'PASS' if response.status_code == 200 else 'FAIL'}")
        if response.status_code == 200:
            result = response.json()
            print(f"  Analysis completed: {result.get('total_nodes', 0)} nodes")
    except Exception as e:
        print(f"Skill Analysis: FAIL - {e}")

def run_detailed_tests():
    """Run all detailed API tests"""
    print("DETAILED API FUNCTIONALITY TESTS")
    print("=" * 50)
    
    test_ocr_api_detailed()
    test_json_api_detailed()
    test_graph_api_detailed()
    test_recommendation_api_detailed()
    
    print("\n" + "=" * 50)
    print("DETAILED TESTS COMPLETED")
    print("=" * 50)

if __name__ == "__main__":
    run_detailed_tests()
