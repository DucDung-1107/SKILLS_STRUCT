#  Skill Graph Platform

AI-powered skill taxonomy management and candidate matching platform.

##  Quick Start

### Local Deployment

1. Install requirements:
```bash
pip install -r streamlit_requirements.txt
```

2. Run the app:
```bash
streamlit run streamlit_app.py
```

### Streamlit Cloud Deployment

1. Fork this repository
2. Connect to Streamlit Cloud
3. Set environment variables in secrets
4. Deploy!

## API Services

The platform requires these API services to be running:

- **OCR & Clustering API** (Port 8000): Process resume files
- **JSON Generation API** (Port 8001): Generate skill taxonomy
- **Graph Management API** (Port 8002): Manage skill nodes
- **Recommendation API** (Port 8003): AI skill recommendations

##Configuration

### Environment Variables

Set these in `.streamlit/secrets.toml`:

```toml
[api]
google_api_key = "your_google_api_key"

[database]
connection_string = "your_database_connection"
```

### API Configuration

Update API endpoints in `streamlit_app.py` if deploying to different hosts.

## Features

- **Resume Processing**: Upload PDF resumes or provide URLs
- **Skill Taxonomy**: Visual skill tree management
-  **Permission System**: Role-based access control
-  **AI Recommendations**: Smart skill suggestions
-  **Analytics**: Skill gap analysis and reporting

## Architecture

```
streamlit_app.py (Frontend)
    ↓
API Layer:
├── OCR & Clustering API (Resume processing)
├── JSON Generation API (Taxonomy creation)
├── Graph Management API (CRUD operations)
└── Recommendation API (AI suggestions)
    ↓
Data Layer:
├── Skill Taxonomy JSON
├── Resume Database
└── User Permissions
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch
3. Make changes
4. Submit pull request

## 📄 License

MIT License - see LICENSE file for details.