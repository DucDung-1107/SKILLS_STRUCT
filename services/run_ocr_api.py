#!/usr/bin/env python3
"""
Runner script for OCR API to handle the + character in filename
"""
import sys
import os

# Add current directory to Python path
sys.path.insert(0, '/app')

# Import and run the OCR API
if __name__ == "__main__":
    # Import the module directly by filename
    import importlib.util
    
    spec = importlib.util.spec_from_file_location(
        "ocr_clustering_api", 
        "/app/services/ocr+clusteringAPI.py"
    )
    ocr_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ocr_module)
    
    # Get the app from the module
    app = ocr_module.app
    
    # Run with uvicorn
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
