@echo off
REM Quick build script for SkillStruct APIs

echo 🚀 Building SkillStruct API Services...

REM Build services individually for better error tracking
echo 📦 Building OCR API...
docker build -f services/Dockerfile.ocr-api -t skillstruct-ocr-api .
if errorlevel 1 goto :error

echo 📦 Building JSON API...
docker build -f services/Dockerfile.json-api -t skillstruct-json-api .
if errorlevel 1 goto :error

echo 📦 Building Graph API...
docker build -f services/Dockerfile.graph-api -t skillstruct-graph-api .
if errorlevel 1 goto :error

echo 📦 Building Recommend API...
docker build -f services/Dockerfile.recommend-api -t skillstruct-recommend-api .
if errorlevel 1 goto :error

echo ✅ All services built successfully!

echo 🔗 Starting services...
cd services
docker-compose up -d
if errorlevel 1 goto :error

echo 🎯 Services running at:
echo   - OCR API: http://localhost:8000/docs
echo   - JSON API: http://localhost:8001/docs  
echo   - Graph API: http://localhost:8002/docs
echo   - Recommend API: http://localhost:8003/docs
echo   - API Gateway: http://localhost:80/status

goto :end

:error
echo ❌ Build failed! Check the error messages above.
exit /b 1

:end
echo 🎉 SkillStruct APIs are ready!
