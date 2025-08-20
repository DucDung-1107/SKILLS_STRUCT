from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from typing import List, Optional, Dict, Any, Union
import os
import tempfile
import shutil
import json
import pandas as pd
from datetime import datetime
import uuid
import logging
import zipfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
import mimetypes
import uvicorn
import sqlite3
import math

# Import các module xử lý từ code gốc
import fitz  # PyMuPDF
from docx import Document
import docx2txt
from pdf2image import convert_from_path
from PIL import Image
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Google API
api_key = os.getenv("GOOGLE_API_KEY", "AIzaSyDL7j6v9YK0U0l_ooD-WaXxWaGMOdRQvnA")
genai.configure(api_key=api_key)

# Database configuration - sử dụng existing SQL file
DB_PATH = r"C:\Users\Admin\Downloads\SkillStruct\resume_features.db"

# Initialize FastAPI app
app = FastAPI(
    title="Resume Processing API",
    description="API for processing resumes and extracting structured information",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên chỉ định cụ thể
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ResumeFeatures(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    address: Optional[str] = None
    linkedin: Optional[str] = None
    skills: Optional[List[str]] = []
    experience_years: Optional[Union[int, float]] = None
    education: Optional[str] = None
    university: Optional[str] = None
    certifications: Optional[List[str]] = []
    languages: Optional[List[str]] = []
    job_titles: Optional[List[str]] = []
    companies: Optional[List[str]] = []
    summary: Optional[str] = None
    
    @field_validator('experience_years')
    @classmethod
    def validate_experience_years(cls, v):
        """Validate and convert experience_years to int"""
        if v is None:
            return None
        if isinstance(v, (int, float)):
            # Round to nearest integer for storage
            return int(round(v))
        try:
            return int(float(v))
        except (ValueError, TypeError):
            return 0

class ProcessResponse(BaseModel):
    success: bool
    message: str
    data: Optional[ResumeFeatures] = None
    processing_id: str
    timestamp: str

class BatchProcessResponse(BaseModel):
    success: bool
    message: str
    total_files: int
    processed_successfully: int
    failed_files: List[str]
    results: List[ProcessResponse]
    batch_id: str
    timestamp: str

class FileStatus(BaseModel):
    filename: str
    status: str  # "processing", "completed", "failed"
    processing_id: Optional[str] = None
    error_message: Optional[str] = None

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database and create tables if they don't exist"""
        try:
            # Create database directory if it doesn't exist
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create resume_features table to match existing SQL structure
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS resume_features (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT,
                    email TEXT,
                    phone TEXT,
                    address TEXT,
                    linkedin TEXT,
                    experience_years INTEGER,
                    education TEXT,
                    university TEXT,
                    skills TEXT,
                    certifications TEXT,
                    languages TEXT,
                    job_titles TEXT,
                    companies TEXT,
                    summary TEXT,
                    processing_id TEXT,
                    batch_id TEXT,
                    timestamp TEXT,
                    source_file TEXT
                )
            ''')
            
            conn.commit()
            conn.close()
            logger.info(f"Database initialized at {self.db_path}")
            
        except Exception as e:
            logger.error(f"Error initializing database: {str(e)}")
    
    def save_resume(self, features: Dict[str, Any], processing_id: str, filename: str, batch_id: str = None):
        """Save resume data to database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert lists to JSON strings
            features_copy = features.copy()
            for key, value in features_copy.items():
                if isinstance(value, list):
                    features_copy[key] = json.dumps(value)
            
            cursor.execute('''
                INSERT INTO resume_features (
                    name, email, phone, address, linkedin, experience_years, 
                    education, university, skills, certifications, languages, 
                    job_titles, companies, summary, processing_id, batch_id, 
                    timestamp, source_file
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                features_copy.get('name'),
                features_copy.get('email'),
                features_copy.get('phone'),
                features_copy.get('address'),
                features_copy.get('linkedin'),
                features_copy.get('experience_years'),
                features_copy.get('education'),
                features_copy.get('university'),
                features_copy.get('skills'),
                features_copy.get('certifications'),
                features_copy.get('languages'),
                features_copy.get('job_titles'),
                features_copy.get('companies'),
                features_copy.get('summary'),
                processing_id,
                batch_id,
                datetime.now().isoformat(),
                filename
            ))
            
            conn.commit()
            conn.close()
            logger.info(f"Resume data saved to database for {filename}")
            
        except Exception as e:
            logger.error(f"Error saving to database: {str(e)}")
    
    def get_all_resumes(self):
        """Get all resumes from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM resume_features ORDER BY timestamp DESC')
            rows = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                row_dict = dict(zip(column_names, row))
                
                # Convert JSON strings back to lists
                for key in ['skills', 'certifications', 'languages', 'job_titles', 'companies']:
                    if row_dict.get(key):
                        try:
                            row_dict[key] = json.loads(row_dict[key])
                        except:
                            row_dict[key] = []
                    else:
                        row_dict[key] = []
                
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting all resumes: {str(e)}")
            return []
    
    def get_resume_by_id(self, processing_id: str):
        """Get resume by processing ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM resume_features WHERE processing_id = ?', (processing_id,))
            row = cursor.fetchone()
            
            if not row:
                return None
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            row_dict = dict(zip(column_names, row))
            
            # Convert JSON strings back to lists
            for key in ['skills', 'certifications', 'languages', 'job_titles', 'companies']:
                if row_dict.get(key):
                    try:
                        row_dict[key] = json.loads(row_dict[key])
                    except:
                        row_dict[key] = []
                else:
                    row_dict[key] = []
            
            conn.close()
            return row_dict
            
        except Exception as e:
            logger.error(f"Error getting resume by ID: {str(e)}")
            return None
    
    def get_batch_resumes(self, batch_id: str):
        """Get all resumes from a specific batch"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('SELECT * FROM resume_features WHERE batch_id = ?', (batch_id,))
            rows = cursor.fetchall()
            
            # Get column names
            column_names = [description[0] for description in cursor.description]
            
            conn.close()
            
            # Convert to list of dictionaries
            results = []
            for row in rows:
                row_dict = dict(zip(column_names, row))
                
                # Convert JSON strings back to lists
                for key in ['skills', 'certifications', 'languages', 'job_titles', 'companies']:
                    if row_dict.get(key):
                        try:
                            row_dict[key] = json.loads(row_dict[key])
                        except:
                            row_dict[key] = []
                    else:
                        row_dict[key] = []
                
                results.append(row_dict)
            
            return results
            
        except Exception as e:
            logger.error(f"Error getting batch resumes: {str(e)}")
            return []
    
    def delete_resume(self, processing_id: str):
        """Delete a resume by processing ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('DELETE FROM resume_features WHERE processing_id = ?', (processing_id,))
            deleted_count = cursor.rowcount
            
            conn.commit()
            conn.close()
            
            return deleted_count > 0
            
        except Exception as e:
            logger.error(f"Error deleting resume: {str(e)}")
            return False

class ResumeProcessor:
    def __init__(self, db_manager: DatabaseManager):
        self.temp_dir = tempfile.mkdtemp()
        self.db_manager = db_manager
        self.executor = ThreadPoolExecutor(max_workers=4)  # For concurrent processing
    
    def is_valid_resume_file(self, filename: str) -> bool:
        """Check if file is a valid resume file"""
        allowed_extensions = ['.pdf', '.doc', '.docx']
        file_extension = os.path.splitext(filename)[1].lower()
        return file_extension in allowed_extensions
    
    def extract_files_from_zip(self, zip_path: str) -> List[str]:
        """Extract all resume files from zip archive"""
        extracted_files = []
        extract_dir = os.path.join(self.temp_dir, f"extracted_{uuid.uuid4()}")
        os.makedirs(extract_dir, exist_ok=True)
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for file_info in zip_ref.filelist:
                    if not file_info.is_dir() and self.is_valid_resume_file(file_info.filename):
                        # Extract file
                        zip_ref.extract(file_info, extract_dir)
                        extracted_file_path = os.path.join(extract_dir, file_info.filename)
                        extracted_files.append(extracted_file_path)
        
        except Exception as e:
            logger.error(f"Error extracting zip file: {str(e)}")
        
        return extracted_files
    
    def extract_text_from_pdf(self, pdf_path: str) -> Optional[str]:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF {pdf_path}: {str(e)}")
            return None

    def extract_text_from_word(self, word_path: str) -> Optional[str]:
        """Extract text from Word document"""
        try:
            if word_path.lower().endswith('.docx'):
                doc = Document(word_path)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                return text.strip()
            else:
                text = docx2txt.process(word_path)
                return text.strip() if text else None
        except Exception as e:
            logger.error(f"Error extracting text from Word document {word_path}: {str(e)}")
            return None

    def convert_to_images_for_ocr(self, file_path: str) -> List[tuple]:
        """Convert PDF or Word document to images for OCR processing"""
        images = []
        
        try:
            if file_path.lower().endswith('.pdf'):
                pdf_images = convert_from_path(file_path, dpi=300)
                for i, image in enumerate(pdf_images):
                    images.append((f"page_{i+1}", image))
            
            elif file_path.lower().endswith(('.doc', '.docx')):
                # Simplified version - chỉ xử lý .docx
                if file_path.lower().endswith('.docx'):
                    # For production, you might want to implement Word to PDF conversion
                    # For now, we'll rely on direct text extraction
                    pass
                    
        except Exception as e:
            logger.error(f"Error converting {file_path} to images: {str(e)}")
        
        return images

    def extract_features_from_text(self, text_content: str) -> Optional[Dict[str, Any]]:
        """Use Gemini API to extract structured features from resume text"""
        try:
            generation_config = {
                "temperature": 0.1,
                "top_p": 0.95,
                "top_k": 40,
                "max_output_tokens": 8192,
                "response_mime_type": "application/json",
            }
            model = genai.GenerativeModel(
                model_name="gemini-1.5-flash",
                generation_config=generation_config,
            )
            prompt = (
                "Extract the following information from this resume text and return as a JSON object. "
                "If any information is not found, use null or empty array. "
                "Return only valid JSON, no explanation. "
                "Fields: name, email, phone, address, linkedin, skills (array), experience_years, education, university, certifications (array), languages (array), job_titles (array), companies (array), summary.\n"
                f"Resume text:\n{text_content}\n"
            )
            chat_session = model.start_chat(history=[])
            response = chat_session.send_message(prompt)
            logger.info(f"[Gemini] Raw response: {response.text}")
            # Parse the JSON response
            try:
                features = json.loads(response.text)
                return features
            except json.JSONDecodeError:
                logger.error(f"Error parsing JSON response. Raw response: {response.text}")
                # Try to extract JSON from the response
                import re
                json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                if json_match:
                    features = json.loads(json_match.group())
                    return features
                else:
                    logger.error(f"[Gemini] Không extract được JSON. Text đầu vào: {text_content[:1000]}")
                    # Trả về text thô để frontend hiển thị
                    return {"raw_response": response.text}
        except Exception as e:
            logger.error(f"Error extracting features: {str(e)}. Text đầu vào: {text_content[:1000]}")
            return {"error": str(e)}

    def process_resume_file(self, file_path: str, filename: str) -> Optional[Dict[str, Any]]:
        """Process a single resume file: luôn kết hợp text extraction và OCR"""
        try:
            extracted_text = ""
            ocr_text = ""
            # 1. Trích xuất text trực tiếp
            if file_path.lower().endswith('.pdf'):
                extracted_text = self.extract_text_from_pdf(file_path) or ""
            elif file_path.lower().endswith(('.doc', '.docx')):
                extracted_text = self.extract_text_from_word(file_path) or ""
            else:
                raise ValueError("Unsupported file format")

            # 2. Luôn thử OCR nếu là PDF hoặc file word
            images = self.convert_to_images_for_ocr(file_path)
            if images:
                generation_config = {
                    "temperature": 0.1,
                    "top_p": 0.95,
                    "top_k": 40,
                    "max_output_tokens": 8192,
                    "response_mime_type": "text/plain",
                }
                try:
                    model = genai.GenerativeModel(
                        model_name="gemini-1.5-flash",
                        generation_config=generation_config,
                    )
                except Exception as e:
                    logger.error(f"[Gemini] Lỗi khởi tạo model: {e}")
                    return None
                for page_name, image in images:
                    try:
                        # Resize image if too large
                        max_size = (2048, 2048)
                        if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
                            image.thumbnail(max_size, Image.Resampling.LANCZOS)
                        chat_session = model.start_chat(history=[])
                        response = chat_session.send_message([
                            image,
                            f"Extract all text from this resume document page with perfect accuracy. "
                            f"Include all names, contact information, skills, experience, education, and other details. "
                            f"Maintain the structure and formatting as much as possible."
                        ])
                        logger.info(f"[Gemini] Đã nhận response cho {page_name}, độ dài: {len(response.text) if hasattr(response, 'text') else 'N/A'}")
                        ocr_text += f"\n--- {page_name} ---\n"
                        ocr_text += response.text if hasattr(response, 'text') else str(response)
                    except Exception as e:
                        logger.error(f"[Gemini] Lỗi khi gửi ảnh {page_name}: {e}")
                        continue

            # 3. Kết hợp text từ cả hai nguồn (ưu tiên text trực tiếp, nối thêm OCR nếu có)
            combined_text = (extracted_text.strip() + "\n" + ocr_text.strip()).strip()
            if not combined_text or len(combined_text) < 30:
                logger.error(f"Không thể trích xuất nội dung từ {filename} (text + OCR đều rỗng)")
                return None
            logger.info(f"Đã trích xuất text (direct + OCR) cho: {filename}, độ dài: {len(combined_text)} ký tự")
            return self.extract_features_from_text(combined_text)
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return None

    def process_single_file(self, file_path: str, filename: str, batch_id: str = None) -> Dict[str, Any]:
        """Process a single resume file and return result"""
        processing_id = str(uuid.uuid4())
        
        try:
            # Process the file
            features = self.process_resume_file(file_path, filename)
            
            if features:
                # Save to database
                self.db_manager.save_resume(features, processing_id, filename, batch_id)
                
                return {
                    "success": True,
                    "message": "Resume processed successfully",
                    "data": features,
                    "processing_id": processing_id,
                    "timestamp": datetime.now().isoformat(),
                    "filename": filename
                }
            else:
                return {
                    "success": False,
                    "message": "Failed to extract features from resume",
                    "processing_id": processing_id,
                    "timestamp": datetime.now().isoformat(),
                    "filename": filename
                }
        
        except Exception as e:
            logger.error(f"Error processing {filename}: {str(e)}")
            return {
                "success": False,
                "message": f"Error processing file: {str(e)}",
                "processing_id": processing_id,
                "timestamp": datetime.now().isoformat(),
                "filename": filename
            }

# Initialize database manager and processor
db_manager = DatabaseManager(DB_PATH)
processor = ResumeProcessor(db_manager)

# API Routes
@app.post("/process-resume/", response_model=ProcessResponse)
async def process_resume(file: UploadFile = File(...)):
    """
    Process an uploaded resume file and extract structured information
    """
    processing_id = str(uuid.uuid4())
    
    # Validate file type
    allowed_extensions = ['.pdf', '.doc', '.docx']
    file_extension = os.path.splitext(file.filename)[1].lower()
    
    if file_extension not in allowed_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type. Allowed types: {', '.join(allowed_extensions)}"
        )
    
    # Save uploaded file temporarily
    temp_file_path = os.path.join(processor.temp_dir, f"{processing_id}_{file.filename}")
    
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Process the file
        features = processor.process_resume_file(temp_file_path, file.filename)
        
        if features:
            # Save to database
            db_manager.save_resume(features, processing_id, file.filename)
            
            # Convert to Pydantic model
            resume_features = ResumeFeatures(**features)
            
            return ProcessResponse(
                success=True,
                message="Resume processed successfully",
                data=resume_features,
                processing_id=processing_id,
                timestamp=datetime.now().isoformat()
            )
        else:
            return ProcessResponse(
                success=False,
                message="Failed to extract features from resume",
                processing_id=processing_id,
                timestamp=datetime.now().isoformat()
            )
    
    except Exception as e:
        logger.error(f"Error processing resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up temporary file
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)

@app.post("/process-multiple-resumes/", response_model=BatchProcessResponse)
async def process_multiple_resumes(files: List[UploadFile] = File(...)):
    """
    Process multiple uploaded resume files
    """
    batch_id = str(uuid.uuid4())
    results = []
    failed_files = []
    
    logger.info(f"Starting batch processing with {len(files)} files")
    
    for file in files:
        # Validate file type
        if not processor.is_valid_resume_file(file.filename):
            failed_files.append(file.filename)
            continue
        
        # Save uploaded file temporarily
        processing_id = str(uuid.uuid4())
        temp_file_path = os.path.join(processor.temp_dir, f"{processing_id}_{file.filename}")
        
        try:
            with open(temp_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
            
            # Process the file
            result = processor.process_single_file(temp_file_path, file.filename, batch_id)
            
            if result["success"]:
                resume_features = ResumeFeatures(**result["data"])
                results.append(ProcessResponse(
                    success=True,
                    message="Resume processed successfully",
                    data=resume_features,
                    processing_id=result["processing_id"],
                    timestamp=result["timestamp"]
                ))
            else:
                failed_files.append(file.filename)
                results.append(ProcessResponse(
                    success=False,
                    message=result["message"],
                    processing_id=result["processing_id"],
                    timestamp=result["timestamp"]
                ))
        
        except Exception as e:
            logger.error(f"Error processing {file.filename}: {str(e)}")
            failed_files.append(file.filename)
        
        finally:
            # Clean up temporary file
            if os.path.exists(temp_file_path):
                os.remove(temp_file_path)
    
    successful_count = len([r for r in results if r.success])
    
    return BatchProcessResponse(
        success=len(failed_files) == 0,
        message=f"Batch processing completed. {successful_count}/{len(files)} files processed successfully",
        total_files=len(files),
        processed_successfully=successful_count,
        failed_files=failed_files,
        results=results,
        batch_id=batch_id,
        timestamp=datetime.now().isoformat()
    )

@app.post("/process-resume-folder/", response_model=BatchProcessResponse)
async def process_resume_folder(file: UploadFile = File(...)):
    """
    Process a ZIP folder containing multiple resume files
    """
    batch_id = str(uuid.uuid4())
    
    # Validate that uploaded file is a ZIP
    if not file.filename.lower().endswith('.zip'):
        raise HTTPException(
            status_code=400,
            detail="Please upload a ZIP file containing resume documents"
        )
    
    # Save uploaded ZIP file temporarily
    temp_zip_path = os.path.join(processor.temp_dir, f"{batch_id}_{file.filename}")
    
    try:
        with open(temp_zip_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Extract files from ZIP
        extracted_files = processor.extract_files_from_zip(temp_zip_path)
        
        if not extracted_files:
            raise HTTPException(
                status_code=400,
                detail="No valid resume files found in the ZIP archive"
            )
        
        logger.info(f"Found {len(extracted_files)} resume files in ZIP")
        
        # Process files
        results = []
        failed_files = []
        
        # Process each file
        for file_path in extracted_files:
            filename = os.path.basename(file_path)
            
            try:
                result = processor.process_single_file(file_path, filename, batch_id)
                
                if result["success"]:
                    resume_features = ResumeFeatures(**result["data"])
                    results.append(ProcessResponse(
                        success=True,
                        message="Resume processed successfully",
                        data=resume_features,
                        processing_id=result["processing_id"],
                        timestamp=result["timestamp"]
                    ))
                else:
                    failed_files.append(filename)
                    results.append(ProcessResponse(
                        success=False,
                        message=result["message"],
                        processing_id=result["processing_id"],
                        timestamp=result["timestamp"]
                    ))
            
            except Exception as e:
                logger.error(f"Error processing {filename}: {str(e)}")
                failed_files.append(filename)
        
        successful_count = len([r for r in results if r.success])
        
        return BatchProcessResponse(
            success=len(failed_files) == 0,
            message=f"Folder processing completed. {successful_count}/{len(extracted_files)} files processed successfully",
            total_files=len(extracted_files),
            processed_successfully=successful_count,
            failed_files=failed_files,
            results=results,
            batch_id=batch_id,
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error processing folder: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
    
    finally:
        # Clean up temporary ZIP file
        if os.path.exists(temp_zip_path):
            os.remove(temp_zip_path)

@app.get("/batch-status/{batch_id}")
async def get_batch_status(batch_id: str):
    """
    Get status of a batch processing job
    """
    try:
        batch_data = db_manager.get_batch_resumes(batch_id)
        
        if not batch_data:
            raise HTTPException(status_code=404, detail="Batch not found")
        
        return {
            "batch_id": batch_id,
            "total_files": len(batch_data),
            "results": batch_data
        }
    
    except Exception as e:
        logger.error(f"Error getting batch status: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/resume-data/")
async def get_resume_data():
    """
    Get all processed resume data from database
    """
    try:
        data = db_manager.get_all_resumes()
        
        return {
            "total_resumes": len(data),
            "data": data
        }
    
    except Exception as e:
        logger.error(f"Error getting resume data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/resume-data/{processing_id}")
async def get_resume_by_id(processing_id: str):
    """
    Get specific resume data by processing ID
    """
    try:
        resume_data = db_manager.get_resume_by_id(processing_id)
        
        if not resume_data:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        return resume_data
    
    except Exception as e:
        logger.error(f"Error getting resume by ID: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.delete("/resume-data/{processing_id}")
async def delete_resume(processing_id: str):
    """
    Delete a specific resume record
    """
    try:
        deleted = db_manager.delete_resume(processing_id)
        
        if not deleted:
            raise HTTPException(status_code=404, detail="Resume not found")
        
        return {"message": f"Resume {processing_id} deleted successfully"}
    
    except Exception as e:
        logger.error(f"Error deleting resume: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """
    Health check endpoint
    """
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)