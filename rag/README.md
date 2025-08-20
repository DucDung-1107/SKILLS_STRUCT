# 🧠 SkillStruct RAG System

Advanced Retrieval-Augmented Generation system cho SkillStruct platform với Milvus vector database và Google AI embeddings.

## ✨ Features

- **🔍 Advanced Search**: Semantic search, skill-aware retrieval, và hybrid search
- **📊 Vector Database**: Milvus vector store với high-performance indexing
- **🤖 AI Embeddings**: Google Generative AI embeddings cho skill và resume data
- **🎯 Smart Retrieval**: Multi-stage retrieval với reranking và filtering
- **📈 Analytics**: Comprehensive metrics và monitoring
- **🔄 Real-time Updates**: Live indexing và document management
- **💾 Backup & Restore**: Automated backup với retention policies
- **🚀 High Performance**: Async operations với connection pooling

## 🏗️ Architecture

```
rag/
├── core/                   # Core RAG system
│   └── rag_system.py      # Main RAG orchestrator
├── embeddings/            # Embedding components
│   └── skill_embedder.py  # Skill-specific embedder
├── vector_stores/         # Vector database integrations
│   └── milvus_store.py    # Milvus vector store
├── retrievers/           # Retrieval components
│   └── advanced_retriever.py  # Advanced retrieval system
├── services/             # API services
│   └── rag_api.py        # FastAPI service
├── schemas.py            # Pydantic models
├── config.py             # Configuration management
├── requirements.txt      # Dependencies
└── README.md            # Documentation
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd SkillSruct/rag
pip install -r requirements.txt
```

### 2. Setup Environment

```bash
# Copy environment template
cp .env.rag.template .env

# Edit configuration
nano .env
```

### 3. Configure Milvus

```bash
# Start Milvus với Docker
docker run -d --name milvus-standalone \
  -p 19530:19530 \
  -v milvus_data:/var/lib/milvus \
  milvusdb/milvus:latest standalone
```

### 4. Start RAG API

```bash
# Development mode
python -m rag.services.rag_api

# Production mode với Gunicorn
gunicorn rag.services.rag_api:app \
  --workers 4 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8004
```

## 📖 API Usage

### Health Check

```bash
curl -X GET "http://localhost:8004/health"
```

### Index Skill Taxonomy

```python
import requests

# Skill taxonomy data
taxonomy_data = {
    "nodes": [
        {
            "id": "python",
            "name": "Python",
            "type": "programming_language",
            "level": 3,
            "employees": ["john.doe", "jane.smith"]
        }
    ],
    "edges": [
        {
            "source": "programming",
            "target": "python",
            "relationship": "contains"
        }
    ]
}

response = requests.post(
    "http://localhost:8004/index/skill-taxonomy",
    json={"taxonomy_data": taxonomy_data}
)
```

### Search Skills

```python
# Find similar skills
response = requests.post(
    "http://localhost:8004/search/similar-skills",
    json={
        "skill_name": "Python",
        "top_k": 5
    }
)

# Search candidates with skills
response = requests.post(
    "http://localhost:8004/search/candidates",
    json={
        "required_skills": ["Python", "FastAPI", "PostgreSQL"],
        "top_k": 10
    }
)

# Get skill recommendations
response = requests.post(
    "http://localhost:8004/recommendations/skills",
    json={
        "current_skills": ["Python", "Django"],
        "top_k": 5
    }
)
```

### Advanced Search

```python
from rag.schemas import QueryRequest, QueryType, SearchFilter, FilterType

query_request = QueryRequest(
    query="Find Python developers with machine learning experience",
    query_type=QueryType.CANDIDATE_SEARCH,
    top_k=10,
    score_threshold=0.7,
    filters=[
        SearchFilter(
            filter_type=FilterType.EXPERIENCE_YEARS,
            values=[3, 10]  # 3-10 years experience
        ),
        SearchFilter(
            filter_type=FilterType.TAGS,
            values=["machine_learning", "data_science"]
        )
    ],
    include_facets=True
)

response = requests.post(
    "http://localhost:8004/search",
    json=query_request.model_dump()
)
```

## ⚙️ Configuration

### Environment Variables

```bash
# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=skillstruct_rag

# Google AI API
GOOGLE_API_KEY=your_api_key_here
GOOGLE_MODEL_NAME=models/embedding-001

# API Settings
RAG_API_PORT=8004
DEFAULT_TOP_K=10
DEFAULT_SCORE_THRESHOLD=0.7

# Performance Tuning
EMBEDDING_BATCH_SIZE=32
INDEX_BATCH_SIZE=100
MAX_CONCURRENT_REQUESTS=100

# Reranking Weights
SKILL_WEIGHT=0.3
FRESHNESS_WEIGHT=0.1
QUALITY_WEIGHT=0.2
DIVERSITY_WEIGHT=0.1
```

### Custom Configuration

```python
from rag.config import RAGSystemConfig, MilvusIndexConfig, VectorIndexType

config = RAGSystemConfig(
    milvus_host="your-milvus-host",
    milvus_port=19530,
    google_api_key="your-api-key",
    embedding_dimension=768,
    index_config=MilvusIndexConfig(
        index_type=VectorIndexType.HNSW,
        index_params={"M": 16, "efConstruction": 200}
    )
)

# Validate configuration
errors = config.validate()
if errors:
    print("Configuration errors:", errors)
```

## 🔧 Integration với SkillStruct

### Với Existing APIs

```python
# Import RAG system trong existing services
from rag.core.rag_system import create_skillstruct_rag

async def initialize_rag():
    global rag_system
    rag_system = await create_skillstruct_rag(
        milvus_host="localhost",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        collection_name="skillstruct_rag"
    )

# Sử dụng trong recommendation API
async def enhanced_skill_recommendations(user_skills: List[str]):
    query_request = QueryRequest(
        query=f"Recommend skills for: {', '.join(user_skills)}",
        query_type=QueryType.RECOMMENDATION,
        top_k=5
    )
    
    response = await rag_system.search(query_request)
    return response.documents
```

### Với Streamlit Frontend

```python
import streamlit as st
from rag.core.rag_system import create_skillstruct_rag

@st.cache_resource
def get_rag_system():
    return asyncio.run(create_skillstruct_rag())

def rag_search_page():
    st.title("🔍 Advanced Skill Search")
    
    query = st.text_input("Search query:")
    search_type = st.selectbox("Search type:", [
        "Semantic Search",
        "Skill Search", 
        "Candidate Search",
        "Recommendations"
    ])
    
    if st.button("Search"):
        rag = get_rag_system()
        # Perform search...
```

## 📊 Monitoring & Analytics

### Health Monitoring

```python
# Get system health
response = requests.get("http://localhost:8004/health")
health = response.json()

print(f"Status: {health['status']}")
print(f"Vector Store: {health['details']['components']['vector_store']['status']}")
```

### Analytics Dashboard

```python
# Get system analytics
response = requests.get("http://localhost:8004/analytics")
analytics = response.json()

print(f"Total Documents: {analytics['data']['collection_stats']['entity_count']}")
print(f"Indexing Success Rate: {analytics['data']['indexing_stats']['successful_indexing']}")
```

### Performance Metrics

```python
# Prometheus metrics available at /metrics endpoint
# Custom metrics:
# - rag_search_duration_seconds
# - rag_index_operations_total
# - rag_vector_store_connections
# - rag_embedding_batch_size
```

## 🛠️ Advanced Features

### Custom Skill Context

```python
from rag.embeddings.skill_embedder import SkillStructEmbedder

embedder = SkillStructEmbedder(embedding_config)

# Create rich context for skill
skill_context = embedder.create_skill_context(
    skill_node=skill_data,
    taxonomy_data=full_taxonomy,
    include_relationships=True
)
```

### Hybrid Retrieval

```python
from rag.schemas import RetrievalMethod

query_request = QueryRequest(
    query="Python machine learning experts",
    retrieval_method=RetrievalMethod.HYBRID,  # Combines semantic + skill-based
    top_k=20
)
```

### Custom Reranking

```python
from rag.schemas import RerankingConfig

reranking_config = RerankingConfig(
    base_weight=0.4,
    skill_weight=0.3,
    freshness_weight=0.1,
    quality_weight=0.2,
    diversity_weight=0.1
)

response = rag.retriever.retrieve(query_request, reranking_config)
```

## 🔧 Development & Testing

### Run Tests

```bash
# Install test dependencies
pip install pytest pytest-asyncio

# Run tests
pytest rag/tests/ -v

# Run với coverage
pytest rag/tests/ --cov=rag --cov-report=html
```

### Development Mode

```bash
# Start với auto-reload
uvicorn rag.services.rag_api:app --reload --port 8004

# Debug mode
export LOG_LEVEL=DEBUG
python -m rag.services.rag_api
```

### Performance Testing

```bash
# Load testing với locust
pip install locust

# Create locustfile.py
locust -f tests/load_test.py --host=http://localhost:8004
```

## 🚀 Production Deployment

### Docker Deployment

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY rag/ ./rag/
COPY requirements.txt .

RUN pip install -r requirements.txt

EXPOSE 8004
CMD ["gunicorn", "rag.services.rag_api:app", "--workers", "4", "--worker-class", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8004"]
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: skillstruct-rag
spec:
  replicas: 3
  selector:
    matchLabels:
      app: skillstruct-rag
  template:
    metadata:
      labels:
        app: skillstruct-rag
    spec:
      containers:
      - name: rag-api
        image: skillstruct/rag:latest
        ports:
        - containerPort: 8004
        env:
        - name: MILVUS_HOST
          value: "milvus-service"
        - name: GOOGLE_API_KEY
          valueFrom:
            secretKeyRef:
              name: rag-secrets
              key: google-api-key
```

### Scaling Considerations

- **Milvus Cluster**: Deploy Milvus cluster cho high availability
- **Load Balancing**: Sử dụng nginx hoặc ALB cho API load balancing
- **Caching**: Implement Redis caching cho frequent queries
- **Monitoring**: Setup Prometheus + Grafana cho metrics
- **Logging**: Centralized logging với ELK stack

## 🔍 Troubleshooting

### Common Issues

1. **Milvus Connection Failed**
   ```bash
   # Check Milvus status
   docker ps | grep milvus
   
   # Check logs
   docker logs milvus-standalone
   ```

2. **Google AI API Errors**
   ```bash
   # Verify API key
   curl -H "Authorization: Bearer $GOOGLE_API_KEY" \
     "https://generativelanguage.googleapis.com/v1/models"
   ```

3. **Memory Issues**
   ```bash
   # Reduce batch sizes
   export EMBEDDING_BATCH_SIZE=16
   export INDEX_BATCH_SIZE=50
   ```

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Enable detailed Milvus logging
from pymilvus import connections
connections.connect(
    alias="default",
    host="localhost",
    port="19530",
    log_level="DEBUG"
)
```

## 📚 References

- [Milvus Documentation](https://milvus.io/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [Google AI Documentation](https://ai.google.dev/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests cho new features
4. Run tests và linting
5. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.
