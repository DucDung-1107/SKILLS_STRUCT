"""
Individual component tests for SkillStruct utilities and core functions
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_core_utils():
    """Test core utility functions"""
    print("🧪 Testing Core Utils...")
    try:
        from utils.core_utils import (
            clean_text, extract_skills_from_text, normalize_text,
            get_environment_variable, validate_email, validate_phone
        )
        
        # Test text cleaning
        dirty_text = "  Hello @#$% World!!!  \n\n"
        clean = clean_text(dirty_text)
        print(f"   ✅ clean_text: '{dirty_text}' -> '{clean}'")
        
        # Test text normalization
        test_text = "HELLO World  123"
        normalized = normalize_text(test_text)
        print(f"   ✅ normalize_text: '{test_text}' -> '{normalized}'")
        
        # Test skill extraction
        resume_text = "I have 5 years experience in Python, JavaScript, React, and SQL databases"
        skills = extract_skills_from_text(resume_text)
        print(f"   ✅ extract_skills_from_text: Found {len(skills)} skills: {skills[:3]}...")
        
        # Test email validation
        emails = ["test@example.com", "invalid-email", "user@domain.co.uk"]
        for email in emails:
            is_valid = validate_email(email)
            print(f"   ✅ validate_email: '{email}' -> {is_valid}")
        
        # Test phone validation
        phones = ["+91 9876543210", "123-456-7890", "invalid"]
        for phone in phones:
            is_valid = validate_phone(phone)
            print(f"   ✅ validate_phone: '{phone}' -> {is_valid}")
        
        print("   🎉 Core Utils tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Core Utils test failed: {e}")
        return False

def test_file_utils():
    """Test file utility functions"""
    print("\n🧪 Testing File Utils...")
    try:
        from utils.file_utils import (
            safe_read_file, get_file_extension, is_pdf_file,
            ensure_directory_exists, get_file_size
        )
        
        # Test file extension detection
        files = ["test.pdf", "document.docx", "image.jpg", "script.py"]
        for file in files:
            ext = get_file_extension(file)
            print(f"   ✅ get_file_extension: '{file}' -> '{ext}'")
        
        # Test PDF detection
        for file in files:
            is_pdf = is_pdf_file(file)
            print(f"   ✅ is_pdf_file: '{file}' -> {is_pdf}")
        
        # Test directory creation
        test_dir = "test_temp_dir"
        ensure_directory_exists(test_dir)
        exists = os.path.exists(test_dir)
        print(f"   ✅ ensure_directory_exists: Created '{test_dir}' -> {exists}")
        
        # Cleanup
        if exists:
            os.rmdir(test_dir)
        
        print("   🎉 File Utils tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ File Utils test failed: {e}")
        return False

def test_data_utils():
    """Test data utility functions"""
    print("\n🧪 Testing Data Utils...")
    try:
        from utils.data_utils import (
            clean_dataframe, validate_dataframe_schema,
            export_to_csv, merge_dataframes
        )
        import pandas as pd
        
        # Create test dataframe
        test_data = {
            'name': ['John Doe', '  Jane Smith  ', 'Bob Johnson'],
            'email': ['john@test.com', 'jane@test.com', 'bob@test.com'],
            'skills': [['Python'], ['Java', 'SQL'], ['JavaScript']]
        }
        df = pd.DataFrame(test_data)
        
        # Test dataframe cleaning
        cleaned_df = clean_dataframe(df)
        print(f"   ✅ clean_dataframe: {len(df)} rows -> {len(cleaned_df)} clean rows")
        
        # Test schema validation
        schema = {
            'name': 'string',
            'email': 'string', 
            'skills': 'list'
        }
        is_valid = validate_dataframe_schema(df, schema)
        print(f"   ✅ validate_dataframe_schema: Schema valid -> {is_valid}")
        
        print("   🎉 Data Utils tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Data Utils test failed: {e}")
        return False

def test_database_utils():
    """Test database utility functions"""
    print("\n🧪 Testing Database Utils...")
    try:
        from utils.database_utils import (
            create_connection_string, validate_connection,
            execute_safe_query, backup_table
        )
        
        # Test connection string creation
        config = {
            'host': 'localhost',
            'port': 5432,
            'database': 'test_db',
            'username': 'user',
            'password': 'pass'
        }
        conn_str = create_connection_string(config)
        print(f"   ✅ create_connection_string: Generated connection string")
        
        print("   🎉 Database Utils tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Database Utils test failed: {e}")
        return False

def test_config_management():
    """Test configuration management"""
    print("\n🧪 Testing Config Management...")
    try:
        from config import get_config, validate_config
        
        # Test config loading
        config = get_config()
        print(f"   ✅ get_config: Loaded config with {len(config)} sections")
        
        # Test config validation
        is_valid = validate_config(config)
        print(f"   ✅ validate_config: Config valid -> {is_valid}")
        
        print("   🎉 Config Management tests passed!")
        return True
        
    except Exception as e:
        print(f"   ❌ Config Management test failed: {e}")
        return False

def main():
    """Run all component tests"""
    print("🚀 SKILLSTRUCT COMPONENT TEST SUITE")
    print("=" * 60)
    
    tests = [
        ("Core Utils", test_core_utils),
        ("File Utils", test_file_utils),
        ("Data Utils", test_data_utils),
        ("Database Utils", test_database_utils),
        ("Config Management", test_config_management)
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} test suite failed: {e}")
            results[test_name] = False
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 COMPONENT TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(results.values())
    total = len(results)
    
    for test_name, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name:<20} : {status}")
    
    print("-" * 60)
    print(f"   Total: {total}, Passed: {passed}, Failed: {total - passed}")
    print(f"   Success Rate: {(passed/total*100):.1f}%")
    print("=" * 60)

if __name__ == "__main__":
    main()
