@echo off
REM SkillStruct Services Docker Management Script

setlocal enabledelayedexpansion

set PROJECT_NAME=skillstruct-services

:print_status
echo [INFO] %~1
goto :eof

:print_success
echo [SUCCESS] %~1
goto :eof

:print_error
echo [ERROR] %~1
goto :eof

:check_docker
docker info >nul 2>&1
if errorlevel 1 (
    call :print_error "Docker is not running. Please start Docker first."
    exit /b 1
)
call :print_success "Docker is running"
goto :eof

:build_services
call :print_status "Building all API services..."
cd /d "%~dp0"
docker-compose build --no-cache
if errorlevel 1 (
    call :print_error "Failed to build services"
    exit /b 1
)
call :print_success "All services built successfully"
goto :eof

:start_services
call :print_status "Starting SkillStruct API services..."
cd /d "%~dp0"
docker-compose up -d
if errorlevel 1 (
    call :print_error "Failed to start services"
    exit /b 1
)

call :print_status "Waiting for services to be ready..."
timeout /t 30 /nobreak >nul

call :print_success "All API services started"
call :print_status "Services available at:"
echo   - OCR API: http://localhost:8000/docs
echo   - JSON API: http://localhost:8001/docs  
echo   - Graph API: http://localhost:8002/docs
echo   - Recommend API: http://localhost:8003/docs
echo   - API Gateway: http://localhost:80/status
goto :eof

:stop_services
call :print_status "Stopping all API services..."
cd /d "%~dp0"
docker-compose down
call :print_success "All services stopped"
goto :eof

:restart_services
call :stop_services
call :start_services
goto :eof

:view_logs
cd /d "%~dp0"
if "%~2"=="" (
    docker-compose logs -f
) else (
    docker-compose logs -f %2
)
goto :eof

:show_status
call :print_status "API Services Status:"
cd /d "%~dp0"
docker-compose ps
goto :eof

:clean_services
call :print_status "Cleaning up services..."
cd /d "%~dp0"
docker-compose down -v --remove-orphans
docker system prune -f
call :print_success "Cleanup completed"
goto :eof

:test_apis
call :print_status "Testing API endpoints..."

REM Test OCR API
curl -s http://localhost:8000/health >nul 2>&1
if errorlevel 1 (
    echo [FAIL] OCR API not responding
) else (
    echo [PASS] OCR API is healthy
)

REM Test JSON API
curl -s http://localhost:8001/health >nul 2>&1
if errorlevel 1 (
    echo [FAIL] JSON API not responding
) else (
    echo [PASS] JSON API is healthy
)

REM Test Graph API
curl -s http://localhost:8002/health >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Graph API not responding
) else (
    echo [PASS] Graph API is healthy
)

REM Test Recommend API
curl -s http://localhost:8003/health >nul 2>&1
if errorlevel 1 (
    echo [FAIL] Recommend API not responding
) else (
    echo [PASS] Recommend API is healthy
)

REM Test Gateway
curl -s http://localhost:80/health >nul 2>&1
if errorlevel 1 (
    echo [FAIL] API Gateway not responding
) else (
    echo [PASS] API Gateway is healthy
)
goto :eof

:show_help
echo SkillStruct API Services Docker Management
echo.
echo Usage: %0 [COMMAND]
echo.
echo Commands:
echo   build     Build all API service images
echo   start     Start all API services
echo   stop      Stop all API services
echo   restart   Restart all API services
echo   status    Show service status
echo   logs      Show logs (add service name for specific service)
echo   test      Test all API endpoints
echo   clean     Clean up Docker resources
echo   help      Show this help message
echo.
echo Examples:
echo   %0 build
echo   %0 start
echo   %0 logs ocr-clustering-api
echo   %0 test
goto :eof

REM Main script logic
if "%1"=="build" (
    call :check_docker
    call :build_services
) else if "%1"=="start" (
    call :check_docker
    call :start_services
) else if "%1"=="stop" (
    call :check_docker
    call :stop_services
) else if "%1"=="restart" (
    call :check_docker
    call :restart_services
) else if "%1"=="status" (
    call :check_docker
    call :show_status
) else if "%1"=="logs" (
    call :check_docker
    call :view_logs %*
) else if "%1"=="test" (
    call :test_apis
) else if "%1"=="clean" (
    call :check_docker
    call :clean_services
) else if "%1"=="help" (
    call :show_help
) else (
    call :print_error "Unknown command: %1"
    echo.
    call :show_help
    exit /b 1
)

endlocal
