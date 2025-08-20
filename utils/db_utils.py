#!/usr/bin/env python3
"""
🗄️ Database Utilities
Các hàm tiện ích cho database operations
"""

import sqlite3
import json
import os
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime
import logging
import pandas as pd

logger = logging.getLogger(__name__)

class DatabaseManager:
    """
    Database manager class for SQLite operations
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """
        Initialize database and create tables
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Resume features table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS resume_features (
                        id TEXT PRIMARY KEY,
                        filename TEXT NOT NULL,
                        name TEXT,
                        email TEXT,
                        phone TEXT,
                        address TEXT,
                        linkedin TEXT,
                        skills TEXT,  -- JSON array
                        experience_years INTEGER,
                        education TEXT,
                        university TEXT,
                        certifications TEXT,  -- JSON array
                        languages TEXT,  -- JSON array
                        job_titles TEXT,  -- JSON array
                        companies TEXT,  -- JSON array
                        summary TEXT,
                        processed_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        raw_text TEXT
                    )
                """)
                
                # Skill taxonomy table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS skill_taxonomy (
                        id TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        type TEXT NOT NULL,
                        level INTEGER NOT NULL,
                        color TEXT,
                        parent_id TEXT,
                        employee_count INTEGER DEFAULT 0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Employee skills table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS employee_skills (
                        id TEXT PRIMARY KEY,
                        employee_id TEXT NOT NULL,
                        employee_name TEXT NOT NULL,
                        department TEXT,
                        team TEXT,
                        skill_name TEXT NOT NULL,
                        proficiency_level TEXT NOT NULL,
                        proficiency_score INTEGER,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # User sessions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS user_sessions (
                        token TEXT PRIMARY KEY,
                        user_id TEXT NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP NOT NULL,
                        is_active BOOLEAN DEFAULT 1,
                        ip_address TEXT,
                        user_agent TEXT
                    )
                """)
                
                # Security logs table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS security_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        resource TEXT NOT NULL,
                        ip_address TEXT,
                        user_agent TEXT,
                        success BOOLEAN NOT NULL,
                        details TEXT
                    )
                """)
                
                # API keys table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS api_keys (
                        key_id TEXT PRIMARY KEY,
                        api_key TEXT UNIQUE NOT NULL,
                        name TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        permissions TEXT,  -- JSON array
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        expires_at TIMESTAMP,
                        is_active BOOLEAN DEFAULT 1,
                        last_used TIMESTAMP
                    )
                """)
                
                # Create indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_resume_email ON resume_features(email)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_skill_name ON employee_skills(skill_name)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_employee_id ON employee_skills(employee_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_sessions_user ON user_sessions(user_id)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_logs_user ON security_logs(user_id)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise

def execute_query(db_path: str, query: str, params: tuple = None, 
                 fetch: str = "all") -> Union[List[Dict], Dict, None]:
    """
    Execute SQL query and return results
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()
            
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch == "one":
                row = cursor.fetchone()
                return dict(row) if row else None
            elif fetch == "all":
                rows = cursor.fetchall()
                return [dict(row) for row in rows]
            else:  # for INSERT, UPDATE, DELETE
                conn.commit()
                return cursor.rowcount
                
    except Exception as e:
        logger.error(f"Error executing query: {e}")
        return None

def bulk_insert(db_path: str, table: str, data: List[Dict[str, Any]]) -> bool:
    """
    Bulk insert data into table
    """
    if not data:
        return True
    
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Get column names from first record
            columns = list(data[0].keys())
            placeholders = ', '.join(['?' for _ in columns])
            column_names = ', '.join(columns)
            
            query = f"INSERT INTO {table} ({column_names}) VALUES ({placeholders})"
            
            # Prepare data tuples
            data_tuples = []
            for record in data:
                values = []
                for col in columns:
                    value = record.get(col)
                    # Convert lists/dicts to JSON strings
                    if isinstance(value, (list, dict)):
                        value = json.dumps(value)
                    values.append(value)
                data_tuples.append(tuple(values))
            
            cursor.executemany(query, data_tuples)
            conn.commit()
            
            logger.info(f"Bulk inserted {len(data)} records into {table}")
            return True
            
    except Exception as e:
        logger.error(f"Error in bulk insert: {e}")
        return False

def backup_database(db_path: str, backup_path: str = None) -> str:
    """
    Create database backup
    """
    if backup_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = f"{db_path}.backup_{timestamp}"
    
    try:
        import shutil
        shutil.copy2(db_path, backup_path)
        logger.info(f"Database backed up to: {backup_path}")
        return backup_path
    except Exception as e:
        logger.error(f"Error backing up database: {e}")
        return None

def migrate_schema(db_path: str, migrations: List[str]) -> bool:
    """
    Apply database schema migrations
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Create migrations table if not exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS schema_migrations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    migration TEXT NOT NULL,
                    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Get applied migrations
            cursor.execute("SELECT migration FROM schema_migrations")
            applied = {row[0] for row in cursor.fetchall()}
            
            # Apply new migrations
            for migration in migrations:
                if migration not in applied:
                    cursor.execute(migration)
                    cursor.execute("INSERT INTO schema_migrations (migration) VALUES (?)", (migration,))
                    logger.info(f"Applied migration: {migration[:50]}...")
            
            conn.commit()
            return True
            
    except Exception as e:
        logger.error(f"Error applying migrations: {e}")
        return False

def save_resume_features(db_path: str, features: Dict[str, Any], 
                        processing_id: str, filename: str) -> bool:
    """
    Save extracted resume features to database
    """
    try:
        # Convert lists to JSON strings
        skills_json = json.dumps(features.get('skills', []))
        certifications_json = json.dumps(features.get('certifications', []))
        languages_json = json.dumps(features.get('languages', []))
        job_titles_json = json.dumps(features.get('job_titles', []))
        companies_json = json.dumps(features.get('companies', []))
        
        query = """
            INSERT INTO resume_features (
                id, filename, name, email, phone, address, linkedin,
                skills, experience_years, education, university,
                certifications, languages, job_titles, companies, summary
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        
        params = (
            processing_id, filename,
            features.get('name'), features.get('email'),
            features.get('phone'), features.get('address'),
            features.get('linkedin'), skills_json,
            features.get('experience_years'), features.get('education'),
            features.get('university'), certifications_json,
            languages_json, job_titles_json, companies_json,
            features.get('summary')
        )
        
        result = execute_query(db_path, query, params, fetch="none")
        return result is not None
        
    except Exception as e:
        logger.error(f"Error saving resume features: {e}")
        return False

def get_all_resumes(db_path: str) -> List[Dict[str, Any]]:
    """
    Get all processed resume data
    """
    query = "SELECT * FROM resume_features ORDER BY processed_at DESC"
    
    results = execute_query(db_path, query)
    if not results:
        return []
    
    # Parse JSON fields
    for result in results:
        for field in ['skills', 'certifications', 'languages', 'job_titles', 'companies']:
            if result.get(field):
                try:
                    result[field] = json.loads(result[field])
                except json.JSONDecodeError:
                    result[field] = []
    
    return results

def save_employee_skills(db_path: str, skills_data: List[Dict[str, Any]]) -> bool:
    """
    Save employee skills data
    """
    return bulk_insert(db_path, "employee_skills", skills_data)

def get_skills_by_employee(db_path: str, employee_id: str) -> List[Dict[str, Any]]:
    """
    Get all skills for an employee
    """
    query = """
        SELECT * FROM employee_skills 
        WHERE employee_id = ? 
        ORDER BY skill_name
    """
    
    return execute_query(db_path, query, (employee_id,)) or []

def get_employees_by_skill(db_path: str, skill_name: str) -> List[Dict[str, Any]]:
    """
    Get all employees with a specific skill
    """
    query = """
        SELECT * FROM employee_skills 
        WHERE skill_name = ? 
        ORDER BY proficiency_level DESC, employee_name
    """
    
    return execute_query(db_path, query, (skill_name,)) or []

def save_security_log(db_path: str, user_id: str, action: str, resource: str,
                     ip_address: str, user_agent: str, success: bool, 
                     details: str = None) -> bool:
    """
    Save security log entry
    """
    query = """
        INSERT INTO security_logs 
        (user_id, action, resource, ip_address, user_agent, success, details)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """
    
    params = (user_id, action, resource, ip_address, user_agent, success, details)
    result = execute_query(db_path, query, params, fetch="none")
    return result is not None

def get_security_logs(db_path: str, user_id: str = None, 
                     limit: int = 100) -> List[Dict[str, Any]]:
    """
    Get security logs
    """
    if user_id:
        query = """
            SELECT * FROM security_logs 
            WHERE user_id = ? 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        params = (user_id, limit)
    else:
        query = """
            SELECT * FROM security_logs 
            ORDER BY timestamp DESC 
            LIMIT ?
        """
        params = (limit,)
    
    return execute_query(db_path, query, params) or []

def save_user_session(db_path: str, token: str, user_id: str, 
                     expires_at: datetime, ip_address: str = None,
                     user_agent: str = None) -> bool:
    """
    Save user session
    """
    query = """
        INSERT INTO user_sessions 
        (token, user_id, expires_at, ip_address, user_agent)
        VALUES (?, ?, ?, ?, ?)
    """
    
    params = (token, user_id, expires_at.isoformat(), ip_address, user_agent)
    result = execute_query(db_path, query, params, fetch="none")
    return result is not None

def get_user_session(db_path: str, token: str) -> Optional[Dict[str, Any]]:
    """
    Get user session by token
    """
    query = """
        SELECT * FROM user_sessions 
        WHERE token = ? AND is_active = 1 AND expires_at > CURRENT_TIMESTAMP
    """
    
    return execute_query(db_path, query, (token,), fetch="one")

def cleanup_expired_sessions(db_path: str) -> int:
    """
    Remove expired sessions
    """
    query = """
        DELETE FROM user_sessions 
        WHERE expires_at <= CURRENT_TIMESTAMP OR is_active = 0
    """
    
    result = execute_query(db_path, query, fetch="none")
    return result or 0

def get_database_stats(db_path: str) -> Dict[str, Any]:
    """
    Get database statistics
    """
    stats = {}
    
    tables = [
        'resume_features', 'skill_taxonomy', 'employee_skills',
        'user_sessions', 'security_logs', 'api_keys'
    ]
    
    for table in tables:
        query = f"SELECT COUNT(*) as count FROM {table}"
        result = execute_query(db_path, query, fetch="one")
        stats[f"{table}_count"] = result['count'] if result else 0
    
    # Database file size
    if os.path.exists(db_path):
        stats['db_size_mb'] = round(os.path.getsize(db_path) / (1024 * 1024), 2)
    else:
        stats['db_size_mb'] = 0
    
    return stats

def export_to_csv(db_path: str, table: str, output_path: str) -> bool:
    """
    Export table data to CSV
    """
    try:
        query = f"SELECT * FROM {table}"
        results = execute_query(db_path, query)
        
        if results:
            df = pd.DataFrame(results)
            df.to_csv(output_path, index=False)
            logger.info(f"Exported {len(results)} records to {output_path}")
            return True
        else:
            logger.warning(f"No data found in table {table}")
            return False
            
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        return False

def import_from_csv(db_path: str, table: str, csv_path: str) -> bool:
    """
    Import CSV data to table
    """
    try:
        df = pd.read_csv(csv_path)
        data = df.to_dict('records')
        
        return bulk_insert(db_path, table, data)
        
    except Exception as e:
        logger.error(f"Error importing from CSV: {e}")
        return False

def vacuum_database(db_path: str) -> bool:
    """
    Optimize database by running VACUUM
    """
    try:
        with sqlite3.connect(db_path) as conn:
            conn.execute("VACUUM")
            logger.info("Database vacuumed successfully")
            return True
    except Exception as e:
        logger.error(f"Error vacuuming database: {e}")
        return False

def check_database_integrity(db_path: str) -> Dict[str, Any]:
    """
    Check database integrity
    """
    try:
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()
            
            # Run integrity check
            cursor.execute("PRAGMA integrity_check")
            integrity_result = cursor.fetchone()[0]
            
            # Check foreign key constraints
            cursor.execute("PRAGMA foreign_key_check")
            fk_violations = cursor.fetchall()
            
            return {
                "integrity_ok": integrity_result == "ok",
                "integrity_result": integrity_result,
                "foreign_key_violations": len(fk_violations),
                "violations": fk_violations
            }
            
    except Exception as e:
        logger.error(f"Error checking database integrity: {e}")
        return {
            "integrity_ok": False,
            "error": str(e)
        }
