@echo off
REM Quick start script for SkillStruct APIs

echo 🚀 Starting SkillStruct API Services...

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker first.
    exit /b 1
)

REM Start only the database first
echo 📦 Starting PostgreSQL...
docker-compose -f docker-compose.apis.yml up -d postgres

REM Wait for database
echo ⏳ Waiting for database to be ready...
timeout /t 15 /nobreak >nul

REM Start the APIs that are already built
echo 🔗 Starting API services...

REM Check if images exist and start them
docker image inspect skillstruct-json-api >nul 2>&1
if not errorlevel 1 (
    echo ✅ Starting JSON API...
    docker-compose -f docker-compose.apis.yml up -d json-generation-api
)

docker image inspect skillstruct-recommend-api >nul 2>&1
if not errorlevel 1 (
    echo ✅ Starting Recommendation API...
    docker-compose -f docker-compose.apis.yml up -d recommendation-api
)

docker image inspect skillstruct-graph-api >nul 2>&1
if not errorlevel 1 (
    echo ✅ Starting Graph API...
    docker-compose -f docker-compose.apis.yml up -d graph-management-api
)

docker image inspect skillstruct-ocr-api >nul 2>&1
if not errorlevel 1 (
    echo ✅ Starting OCR API...
    docker-compose -f docker-compose.apis.yml up -d ocr-clustering-api
)

echo 🎯 Available APIs:
echo   - PostgreSQL: localhost:5432
docker-compose -f docker-compose.apis.yml ps

echo.
echo 🔍 Testing API health...
timeout /t 10 /nobreak >nul

REM Test APIs
curl -s http://localhost:8001/health >nul 2>&1
if not errorlevel 1 (
    echo ✅ JSON API: HEALTHY
) else (
    echo ❌ JSON API: DOWN
)

curl -s http://localhost:8003/health >nul 2>&1
if not errorlevel 1 (
    echo ✅ Recommend API: HEALTHY
) else (
    echo ❌ Recommend API: DOWN
)

curl -s http://localhost:8002/health >nul 2>&1
if not errorlevel 1 (
    echo ✅ Graph API: HEALTHY
) else (
    echo ❌ Graph API: DOWN
)

curl -s http://localhost:8000/health >nul 2>&1
if not errorlevel 1 (
    echo ✅ OCR API: HEALTHY
) else (
    echo ❌ OCR API: DOWN
)

echo.
echo 🎉 SkillStruct APIs setup complete!
echo 📚 API Documentation:
echo   - JSON API: http://localhost:8001/docs
echo   - Recommend API: http://localhost:8003/docs
echo   - Graph API: http://localhost:8002/docs
echo   - OCR API: http://localhost:8000/docs
