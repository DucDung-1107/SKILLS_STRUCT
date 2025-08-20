#!/bin/bash
# Quick start script for SkillStruct APIs

echo "🚀 Starting SkillStruct API Services..."

# Check if Docker is running
if ! docker info >/dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker first."
    exit 1
fi

# Start only the database first
echo "📦 Starting PostgreSQL..."
docker-compose -f docker-compose.apis.yml up -d postgres

# Wait for database
echo "⏳ Waiting for database to be ready..."
sleep 15

# Start the APIs that are already built
echo "🔗 Starting API services..."

# Check if images exist and start them
if docker image inspect skillstruct-json-api >/dev/null 2>&1; then
    echo "✅ Starting JSON API..."
    docker-compose -f docker-compose.apis.yml up -d json-generation-api
fi

if docker image inspect skillstruct-recommend-api >/dev/null 2>&1; then
    echo "✅ Starting Recommendation API..."
    docker-compose -f docker-compose.apis.yml up -d recommendation-api
fi

if docker image inspect skillstruct-graph-api >/dev/null 2>&1; then
    echo "✅ Starting Graph API..."
    docker-compose -f docker-compose.apis.yml up -d graph-management-api
fi

if docker image inspect skillstruct-ocr-api >/dev/null 2>&1; then
    echo "✅ Starting OCR API..."
    docker-compose -f docker-compose.apis.yml up -d ocr-clustering-api
fi

echo "🎯 Available APIs:"
echo "  - PostgreSQL: localhost:5432"
docker-compose -f docker-compose.apis.yml ps

echo ""
echo "🔍 Testing API health..."
sleep 10

# Test APIs
curl -s http://localhost:8001/health && echo "✅ JSON API: HEALTHY" || echo "❌ JSON API: DOWN"
curl -s http://localhost:8003/health && echo "✅ Recommend API: HEALTHY" || echo "❌ Recommend API: DOWN"
curl -s http://localhost:8002/health && echo "✅ Graph API: HEALTHY" || echo "❌ Graph API: DOWN"
curl -s http://localhost:8000/health && echo "✅ OCR API: HEALTHY" || echo "❌ OCR API: DOWN"

echo ""
echo "🎉 SkillStruct APIs setup complete!"
echo "📚 API Documentation:"
echo "  - JSON API: http://localhost:8001/docs"
echo "  - Recommend API: http://localhost:8003/docs"
echo "  - Graph API: http://localhost:8002/docs"
echo "  - OCR API: http://localhost:8000/docs"
