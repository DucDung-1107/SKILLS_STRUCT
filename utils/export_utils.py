#!/usr/bin/env python3
"""
📤 Export Utilities
Các hàm tiện ích để export data ra các format khác nhau
"""

import json
import csv
import xml.etree.ElementTree as ET
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from io import StringIO, BytesIO
import zipfile
import base64
import logging

logger = logging.getLogger(__name__)

def export_to_json(data: Any, pretty: bool = True, ensure_ascii: bool = False) -> str:
    """
    Export data to JSON format
    """
    try:
        if pretty:
            return json.dumps(data, indent=2, ensure_ascii=ensure_ascii, default=str)
        else:
            return json.dumps(data, ensure_ascii=ensure_ascii, default=str)
    except Exception as e:
        logger.error(f"Error exporting to JSON: {e}")
        raise ValueError(f"Cannot export to JSON: {e}")

def export_to_csv(data: List[Dict[str, Any]], filename: str = None) -> str:
    """
    Export list of dictionaries to CSV format
    """
    if not data:
        return ""
    
    try:
        output = StringIO()
        fieldnames = list(data[0].keys())
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        
        writer.writeheader()
        for row in data:
            # Convert complex objects to strings
            clean_row = {}
            for key, value in row.items():
                if isinstance(value, (dict, list)):
                    clean_row[key] = json.dumps(value)
                else:
                    clean_row[key] = str(value) if value is not None else ""
            writer.writerow(clean_row)
        
        return output.getvalue()
    except Exception as e:
        logger.error(f"Error exporting to CSV: {e}")
        raise ValueError(f"Cannot export to CSV: {e}")

def export_to_xml(data: Any, root_name: str = "data") -> str:
    """
    Export data to XML format
    """
    try:
        root = ET.Element(root_name)
        _dict_to_xml(data, root)
        return ET.tostring(root, encoding='unicode')
    except Exception as e:
        logger.error(f"Error exporting to XML: {e}")
        raise ValueError(f"Cannot export to XML: {e}")

def _dict_to_xml(data: Any, parent: ET.Element) -> None:
    """
    Helper function to convert dictionary to XML
    """
    if isinstance(data, dict):
        for key, value in data.items():
            # Clean key name for XML
            clean_key = str(key).replace(' ', '_').replace('-', '_')
            child = ET.SubElement(parent, clean_key)
            _dict_to_xml(value, child)
    elif isinstance(data, list):
        for i, item in enumerate(data):
            child = ET.SubElement(parent, f"item_{i}")
            _dict_to_xml(item, child)
    else:
        parent.text = str(data) if data is not None else ""

def export_resumes_to_csv(resumes: List[Dict[str, Any]]) -> str:
    """
    Export resume data to CSV with flattened structure
    """
    if not resumes:
        return ""
    
    flattened_resumes = []
    
    for resume in resumes:
        flat_resume = {
            "id": resume.get("id"),
            "file_name": resume.get("file_name"),
            "upload_date": resume.get("upload_date"),
            "processed_date": resume.get("processed_date"),
            "status": resume.get("status"),
            "candidate_name": resume.get("extracted_data", {}).get("personal_info", {}).get("name"),
            "candidate_email": resume.get("extracted_data", {}).get("personal_info", {}).get("email"),
            "candidate_phone": resume.get("extracted_data", {}).get("personal_info", {}).get("phone"),
            "total_experience": resume.get("extracted_data", {}).get("experience_years"),
            "education_level": resume.get("extracted_data", {}).get("highest_education"),
            "skills": ", ".join(resume.get("extracted_data", {}).get("skills", [])),
            "technologies": ", ".join(resume.get("extracted_data", {}).get("technologies", [])),
            "certifications": ", ".join(resume.get("extracted_data", {}).get("certifications", [])),
            "languages": ", ".join(resume.get("extracted_data", {}).get("languages", [])),
            "current_position": resume.get("extracted_data", {}).get("current_position"),
            "location": resume.get("extracted_data", {}).get("location"),
            "ai_score": resume.get("ai_analysis", {}).get("overall_score"),
            "skill_match_score": resume.get("ai_analysis", {}).get("skill_matching_score"),
            "experience_score": resume.get("ai_analysis", {}).get("experience_score"),
            "education_score": resume.get("ai_analysis", {}).get("education_score")
        }
        flattened_resumes.append(flat_resume)
    
    return export_to_csv(flattened_resumes)

def export_skills_to_csv(skills: List[Dict[str, Any]]) -> str:
    """
    Export skill taxonomy to CSV
    """
    if not skills:
        return ""
    
    flattened_skills = []
    
    for skill in skills:
        flat_skill = {
            "id": skill.get("id"),
            "name": skill.get("name"),
            "category": skill.get("category"),
            "level": skill.get("level"),
            "parent_id": skill.get("parent_id"),
            "description": skill.get("description"),
            "synonyms": ", ".join(skill.get("synonyms", [])),
            "related_skills": ", ".join(skill.get("related_skills", [])),
            "industry": skill.get("industry"),
            "popularity_score": skill.get("popularity_score"),
            "demand_score": skill.get("demand_score"),
            "created_date": skill.get("created_date"),
            "updated_date": skill.get("updated_date")
        }
        flattened_skills.append(flat_skill)
    
    return export_to_csv(flattened_skills)

def export_analytics_to_csv(analytics: Dict[str, Any]) -> str:
    """
    Export analytics data to CSV format
    """
    datasets = []
    
    # Skill distribution
    if "skill_distribution" in analytics:
        for skill, count in analytics["skill_distribution"].items():
            datasets.append({
                "metric_type": "skill_distribution",
                "name": skill,
                "value": count,
                "percentage": analytics.get("skill_percentages", {}).get(skill, 0)
            })
    
    # Experience distribution
    if "experience_distribution" in analytics:
        for exp_range, count in analytics["experience_distribution"].items():
            datasets.append({
                "metric_type": "experience_distribution",
                "name": exp_range,
                "value": count,
                "percentage": 0  # Calculate if needed
            })
    
    # Education distribution
    if "education_distribution" in analytics:
        for edu_level, count in analytics["education_distribution"].items():
            datasets.append({
                "metric_type": "education_distribution",
                "name": edu_level,
                "value": count,
                "percentage": 0  # Calculate if needed
            })
    
    return export_to_csv(datasets)

def create_excel_export(data: Dict[str, List[Dict[str, Any]]]) -> bytes:
    """
    Create Excel file with multiple sheets
    """
    try:
        import pandas as pd
        
        output = BytesIO()
        
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            for sheet_name, sheet_data in data.items():
                if sheet_data:
                    df = pd.DataFrame(sheet_data)
                    df.to_excel(writer, sheet_name=sheet_name, index=False)
        
        output.seek(0)
        return output.getvalue()
    
    except ImportError:
        # Fallback without pandas
        logger.warning("pandas not available, creating simple Excel-like CSV")
        csv_content = ""
        for sheet_name, sheet_data in data.items():
            csv_content += f"Sheet: {sheet_name}\n"
            csv_content += export_to_csv(sheet_data)
            csv_content += "\n\n"
        
        return csv_content.encode('utf-8')
    
    except Exception as e:
        logger.error(f"Error creating Excel export: {e}")
        raise ValueError(f"Cannot create Excel export: {e}")

def create_zip_export(files: Dict[str, Union[str, bytes]], 
                     zip_name: str = "export.zip") -> bytes:
    """
    Create ZIP archive with multiple files
    """
    try:
        zip_buffer = BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for filename, content in files.items():
                if isinstance(content, str):
                    zip_file.writestr(filename, content.encode('utf-8'))
                else:
                    zip_file.writestr(filename, content)
        
        zip_buffer.seek(0)
        return zip_buffer.getvalue()
    
    except Exception as e:
        logger.error(f"Error creating ZIP export: {e}")
        raise ValueError(f"Cannot create ZIP export: {e}")

def export_full_database(db_data: Dict[str, Any]) -> Dict[str, str]:
    """
    Export entire database to multiple formats
    """
    exports = {}
    
    try:
        # JSON export
        exports["database_full.json"] = export_to_json(db_data, pretty=True)
        
        # Individual CSV exports
        if "resumes" in db_data:
            exports["resumes.csv"] = export_resumes_to_csv(db_data["resumes"])
        
        if "skills" in db_data:
            exports["skills.csv"] = export_skills_to_csv(db_data["skills"])
        
        if "users" in db_data:
            exports["users.csv"] = export_to_csv(db_data["users"])
        
        if "analytics" in db_data:
            exports["analytics.csv"] = export_analytics_to_csv(db_data["analytics"])
        
        # XML export
        exports["database_full.xml"] = export_to_xml(db_data, "skillstruct_database")
        
        return exports
    
    except Exception as e:
        logger.error(f"Error in full database export: {e}")
        raise ValueError(f"Cannot export database: {e}")

def create_backup_export(db_data: Dict[str, Any], 
                        include_metadata: bool = True) -> bytes:
    """
    Create comprehensive backup export
    """
    backup_data = db_data.copy()
    
    if include_metadata:
        backup_data["backup_metadata"] = {
            "created_at": datetime.now().isoformat(),
            "version": "1.0.0",
            "total_resumes": len(db_data.get("resumes", [])),
            "total_skills": len(db_data.get("skills", [])),
            "total_users": len(db_data.get("users", [])),
            "backup_type": "full"
        }
    
    files = export_full_database(backup_data)
    return create_zip_export(files, "skillstruct_backup.zip")

def export_filtered_data(data: List[Dict[str, Any]], 
                        filters: Dict[str, Any],
                        format: str = "json") -> str:
    """
    Export filtered data in specified format
    """
    # Apply filters
    filtered_data = data
    
    for field, value in filters.items():
        if field == "date_range":
            start_date = value.get("start")
            end_date = value.get("end")
            filtered_data = [
                item for item in filtered_data
                if (not start_date or item.get("created_date", "") >= start_date) and
                   (not end_date or item.get("created_date", "") <= end_date)
            ]
        elif field == "skills":
            filtered_data = [
                item for item in filtered_data
                if any(skill in item.get("skills", []) for skill in value)
            ]
        elif field == "experience_range":
            min_exp = value.get("min", 0)
            max_exp = value.get("max", 100)
            filtered_data = [
                item for item in filtered_data
                if min_exp <= item.get("experience_years", 0) <= max_exp
            ]
    
    # Export in requested format
    if format.lower() == "json":
        return export_to_json(filtered_data)
    elif format.lower() == "csv":
        return export_to_csv(filtered_data)
    elif format.lower() == "xml":
        return export_to_xml(filtered_data, "filtered_data")
    else:
        raise ValueError(f"Unsupported export format: {format}")

def create_report_export(report_data: Dict[str, Any], 
                        report_type: str = "summary") -> Dict[str, str]:
    """
    Create specialized report exports
    """
    exports = {}
    
    if report_type == "summary":
        # Summary report
        summary = {
            "report_generated": datetime.now().isoformat(),
            "total_resumes": report_data.get("total_resumes", 0),
            "total_skills": report_data.get("total_skills", 0),
            "avg_experience": report_data.get("avg_experience", 0),
            "top_skills": report_data.get("top_skills", []),
            "skill_gaps": report_data.get("skill_gaps", []),
            "recommendations": report_data.get("recommendations", [])
        }
        exports["summary_report.json"] = export_to_json(summary)
        
    elif report_type == "detailed":
        # Detailed analysis report
        exports["detailed_report.json"] = export_to_json(report_data)
        
        if "candidate_analysis" in report_data:
            exports["candidates.csv"] = export_to_csv(report_data["candidate_analysis"])
        
        if "skill_analysis" in report_data:
            exports["skill_analysis.csv"] = export_to_csv(report_data["skill_analysis"])
    
    elif report_type == "skills_matrix":
        # Skills matrix report
        if "skills_matrix" in report_data:
            exports["skills_matrix.csv"] = export_to_csv(report_data["skills_matrix"])
            exports["skills_matrix.json"] = export_to_json(report_data["skills_matrix"])
    
    return exports

def encode_file_for_download(content: Union[str, bytes], 
                           filename: str) -> Dict[str, str]:
    """
    Encode file content for download
    """
    if isinstance(content, str):
        content = content.encode('utf-8')
    
    encoded_content = base64.b64encode(content).decode('utf-8')
    
    return {
        "filename": filename,
        "content": encoded_content,
        "content_type": _get_content_type(filename),
        "size": len(content)
    }

def _get_content_type(filename: str) -> str:
    """
    Get content type based on file extension
    """
    ext = filename.lower().split('.')[-1]
    
    content_types = {
        "json": "application/json",
        "csv": "text/csv",
        "xml": "application/xml",
        "xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "zip": "application/zip",
        "pdf": "application/pdf",
        "txt": "text/plain"
    }
    
    return content_types.get(ext, "application/octet-stream")

def create_data_export_summary(exports: Dict[str, str]) -> Dict[str, Any]:
    """
    Create summary of exported data
    """
    summary = {
        "export_date": datetime.now().isoformat(),
        "total_files": len(exports),
        "files": []
    }
    
    total_size = 0
    
    for filename, content in exports.items():
        file_size = len(content.encode('utf-8')) if isinstance(content, str) else len(content)
        total_size += file_size
        
        summary["files"].append({
            "filename": filename,
            "size_bytes": file_size,
            "size_readable": _format_file_size(file_size),
            "type": _get_content_type(filename)
        })
    
    summary["total_size_bytes"] = total_size
    summary["total_size_readable"] = _format_file_size(total_size)
    
    return summary

def _format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    """
    if size_bytes < 1024:
        return f"{size_bytes} B"
    elif size_bytes < 1024**2:
        return f"{size_bytes/1024:.1f} KB"
    elif size_bytes < 1024**3:
        return f"{size_bytes/(1024**2):.1f} MB"
    else:
        return f"{size_bytes/(1024**3):.1f} GB"

def validate_export_request(export_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate export request configuration
    """
    errors = []
    
    # Check required fields
    if "format" not in export_config:
        errors.append("Export format is required")
    elif export_config["format"] not in ["json", "csv", "xml", "excel", "zip"]:
        errors.append("Invalid export format")
    
    if "data_type" not in export_config:
        errors.append("Data type is required")
    elif export_config["data_type"] not in ["resumes", "skills", "users", "analytics", "all"]:
        errors.append("Invalid data type")
    
    # Validate date ranges if provided
    if "date_range" in export_config:
        date_range = export_config["date_range"]
        if "start" in date_range and "end" in date_range:
            try:
                start = datetime.fromisoformat(date_range["start"])
                end = datetime.fromisoformat(date_range["end"])
                if start > end:
                    errors.append("Start date must be before end date")
            except ValueError:
                errors.append("Invalid date format")
    
    # Validate limit
    if "limit" in export_config:
        limit = export_config["limit"]
        if not isinstance(limit, int) or limit <= 0:
            errors.append("Limit must be a positive integer")
        elif limit > 10000:
            errors.append("Limit cannot exceed 10,000 records")
    
    return {
        "valid": len(errors) == 0,
        "errors": errors
    }
