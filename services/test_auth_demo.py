#!/usr/bin/env python3
"""
🔐 GraphAPI Authentication Demo
Demo script để test hệ thống authentication và authorization mới
"""

import requests
import json
from datetime import datetime

# API Configuration
API_BASE = "http://localhost:8000"

def test_login(username: str, password: str):
    """Test login functionality"""
    print(f"\n🔑 Testing login for user: {username}")
    
    response = requests.post(f"{API_BASE}/auth/login", json={
        "username": username,
        "password": password
    })
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Login successful!")
        print(f"   Token: {data['token'][:20]}...")
        print(f"   Role: {data['user']['role']}")
        print(f"   Permissions: {list(data['user']['permissions'].keys())}")
        return data['token']
    else:
        print(f"❌ Login failed: {response.text}")
        return None

def test_permission_check(token: str, action: str, resource: str = "taxonomy"):
    """Test permission checking"""
    print(f"\n🛡️ Testing permission: {action} on {resource}")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_BASE}/auth/check-permission", 
                           json={
                               "action": action,
                               "resource": resource,
                               "confirmation": False
                           },
                           headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("requires_confirmation"):
            print(f"⚠️ Permission granted but requires confirmation")
            print(f"   Warning: {data.get('warning')}")
        else:
            print(f"✅ Permission granted")
    else:
        print(f"❌ Permission denied: {response.text}")

def test_protected_endpoint(token: str, endpoint: str = "/taxonomy"):
    """Test accessing protected endpoint"""
    print(f"\n🔒 Testing protected endpoint: {endpoint}")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}{endpoint}", headers=headers)
    
    if response.status_code == 200:
        print(f"✅ Access granted to {endpoint}")
        if endpoint == "/taxonomy":
            data = response.json()
            print(f"   Nodes: {len(data.get('nodes', []))}")
            print(f"   Edges: {len(data.get('edges', []))}")
    else:
        print(f"❌ Access denied to {endpoint}: {response.text}")

def test_create_node(token: str):
    """Test creating a new node"""
    print(f"\n➕ Testing node creation")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.post(f"{API_BASE}/nodes", 
                           json={
                               "name": f"Test Skill {datetime.now().strftime('%H%M%S')}",
                               "type": "skill",
                               "parent_id": "root",
                               "color": "#ff6b6b"
                           },
                           headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Node created successfully!")
        print(f"   ID: {data['id']}")
        print(f"   Name: {data['name']}")
        return data['id']
    else:
        print(f"❌ Node creation failed: {response.text}")
        return None

def test_delete_node(token: str, node_id: str):
    """Test deleting a node (with confirmation)"""
    print(f"\n🗑️ Testing node deletion: {node_id}")
    
    headers = {"Authorization": f"Bearer {token}"}
    
    # First attempt without confirmation
    response = requests.delete(f"{API_BASE}/nodes/{node_id}", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        if data.get("requires_confirmation"):
            print(f"⚠️ Deletion requires confirmation")
            print(f"   Confirmation URL: {data.get('confirmation_url')}")
            
            # Simulate confirmation
            response = requests.delete(f"{API_BASE}/nodes/{node_id}?confirmed=true", 
                                     headers=headers)
            if response.status_code == 200:
                print(f"✅ Node deleted after confirmation")
            else:
                print(f"❌ Deletion failed even with confirmation: {response.text}")
        else:
            print(f"✅ Node deleted successfully")
    else:
        print(f"❌ Deletion failed: {response.text}")

def test_security_logs(token: str):
    """Test accessing security logs (Admin only)"""
    print(f"\n📋 Testing security logs access")
    
    headers = {"Authorization": f"Bearer {token}"}
    response = requests.get(f"{API_BASE}/auth/security-logs?limit=5", headers=headers)
    
    if response.status_code == 200:
        data = response.json()
        print(f"✅ Security logs accessed")
        print(f"   Total logs: {data.get('total_logs', 0)}")
        print(f"   Recent logs: {len(data.get('logs', []))}")
        
        for log in data.get('logs', [])[-3:]:  # Show last 3 logs
            print(f"   - {log.get('timestamp')}: {log.get('action')} by {log.get('user_id')} - {'✅' if log.get('success') else '❌'}")
    else:
        print(f"❌ Security logs access denied: {response.text}")

def main():
    """Main demo function"""
    print("🔐 GraphAPI Authentication & Authorization Demo")
    print("=" * 50)
    
    # Test different user roles
    users = [
        ("admin", "admin123"),
        ("hr_architect", "hr123"),
        ("manager", "manager123"),
        ("viewer", "viewer123")
    ]
    
    tokens = {}
    
    # Test login for all users
    for username, password in users:
        token = test_login(username, password)
        if token:
            tokens[username] = token
    
    print(f"\n{'='*50}")
    print("🧪 Testing Permissions & Operations")
    print("=" * 50)
    
    # Test admin operations
    if "admin" in tokens:
        print(f"\n👑 ADMIN USER TESTS")
        admin_token = tokens["admin"]
        
        test_permission_check(admin_token, "read")
        test_permission_check(admin_token, "create")
        test_permission_check(admin_token, "delete")
        test_protected_endpoint(admin_token, "/taxonomy")
        
        # Test creating and deleting node
        node_id = test_create_node(admin_token)
        if node_id:
            test_delete_node(admin_token, node_id)
        
        test_security_logs(admin_token)
    
    # Test viewer operations (limited permissions)
    if "viewer" in tokens:
        print(f"\n👁️ VIEWER USER TESTS")
        viewer_token = tokens["viewer"]
        
        test_permission_check(viewer_token, "read")
        test_permission_check(viewer_token, "create")  # Should fail
        test_permission_check(viewer_token, "delete")  # Should fail
        test_protected_endpoint(viewer_token, "/taxonomy")
        test_security_logs(viewer_token)  # Should fail
    
    # Test manager operations (read only)
    if "manager" in tokens:
        print(f"\n👔 MANAGER USER TESTS")
        manager_token = tokens["manager"]
        
        test_permission_check(manager_token, "read")
        test_permission_check(manager_token, "create")  # Should fail
        test_protected_endpoint(manager_token, "/taxonomy")
    
    print(f"\n{'='*50}")
    print("✅ Demo completed! Check the API documentation at http://localhost:8000/docs")
    print("🔍 View security logs through admin account")
    print("🌐 Try the permission popup at http://localhost:8000/auth/permissions-popup")

if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("❌ Error: Cannot connect to API. Make sure GraphAPI.py is running on port 8000")
        print("   Run: python GraphAPI.py")
    except Exception as e:
        print(f"❌ Error during demo: {e}")
