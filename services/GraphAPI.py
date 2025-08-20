from fastapi import FastAPI, HTTPException, Depends, status, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from enum import Enum
import json
import uuid
from datetime import datetime, timedelta
import os
import hashlib
import secrets
from functools import wraps

app = FastAPI(title="Skill Taxonomy Management API", version="1.0.0")
security = HTTPBearer()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục data nếu chưa tồn tại
DATA_DIR = "data"
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

# Enums for roles and node types
class Role(str, Enum):
    ADMIN = "admin"
    HR_ARCHITECT = "hr_architect"
    MANAGER = "manager"
    VIEWER = "viewer"

class NodeType(str, Enum):
    ROOT = "root"
    SKILL_GROUP = "skill_group"
    SKILL = "skill"
    SUB_SKILL = "sub_skill"

class EdgeType(str, Enum):
    CONTAINS = "contains"
    INCLUDES = "includes"
    SPECIALIZES = "specializes"

# Pydantic models
class User(BaseModel):
    id: str
    username: str
    role: Role
    email: Optional[str] = None
    last_login: Optional[datetime] = None
    permissions: Dict[str, bool] = {}
    session_token: Optional[str] = None
    
class AuthSession(BaseModel):
    user_id: str
    token: str
    created_at: datetime
    expires_at: datetime
    is_active: bool = True

class AuthRequest(BaseModel):
    username: str
    password: str

class PermissionRequest(BaseModel):
    action: str
    resource: str
    confirmation: bool = False

class SecurityLog(BaseModel):
    timestamp: datetime
    user_id: str
    action: str
    resource: str
    ip_address: str
    user_agent: str
    success: bool
    details: Optional[str] = None
    
class Node(BaseModel):
    id: str
    name: str
    type: NodeType
    level: int
    color: str
    employees: List[str] = []
    employee_count: int = 0
    proficiency_stats: Dict[str, Any] = {}

class Edge(BaseModel):
    id: str
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0

class SkillTaxonomy(BaseModel):
    metadata: Dict[str, Any]
    nodes: List[Node]
    edges: List[Edge]
    skill_owners: Dict[str, Any] = {}
    color_scheme: Dict[str, Any]
    mermaid_export: Dict[str, Any]

class CreateNodeRequest(BaseModel):
    name: str
    type: NodeType
    parent_id: Optional[str] = None
    color: Optional[str] = None

class UpdateNodeRequest(BaseModel):
    name: Optional[str] = None
    color: Optional[str] = None

class RenameNodeRequest(BaseModel):
    new_name: str

# Mock database - In production, use a real database
TAXONOMY_FILE = os.path.join(DATA_DIR, "taxonomy_data.json")
SECURITY_LOG_FILE = os.path.join(DATA_DIR, "security_log.json")
SESSIONS_FILE = os.path.join(DATA_DIR, "active_sessions.json")

# Enhanced user database with hashed passwords
USERS_DB = {
    "admin": {
        "id": "1",
        "username": "admin",
        "password_hash": hashlib.sha256("admin123".encode()).hexdigest(),
        "role": Role.ADMIN,
        "email": "admin@company.com",
        "permissions": {
            "read": True, "create": True, "update": True, "delete": True,
            "import_export": True, "versioning": True, "user_management": True
        }
    },
    "hr_architect": {
        "id": "2", 
        "username": "hr_architect",
        "password_hash": hashlib.sha256("hr123".encode()).hexdigest(),
        "role": Role.HR_ARCHITECT,
        "email": "hr@company.com",
        "permissions": {
            "read": True, "create": True, "update": True, "delete": True,
            "import_export": False, "versioning": False, "user_management": False
        }
    },
    "manager": {
        "id": "3",
        "username": "manager", 
        "password_hash": hashlib.sha256("manager123".encode()).hexdigest(),
        "role": Role.MANAGER,
        "email": "manager@company.com",
        "permissions": {
            "read": True, "create": False, "update": False, "delete": False,
            "import_export": False, "versioning": False, "user_management": False
        }
    },
    "viewer": {
        "id": "4",
        "username": "viewer",
        "password_hash": hashlib.sha256("viewer123".encode()).hexdigest(), 
        "role": Role.VIEWER,
        "email": "viewer@company.com",
        "permissions": {
            "read": True, "create": False, "update": False, "delete": False,
            "import_export": False, "versioning": False, "user_management": False
        }
    }
}

# Active sessions storage
active_sessions: Dict[str, AuthSession] = {}

# Role permissions
PERMISSIONS = {
    Role.ADMIN: {
        "read": True,
        "create": True,
        "update": True,
        "delete": True,
        "import_export": True,
        "versioning": True
    },
    Role.HR_ARCHITECT: {
        "read": True,
        "create": True,
        "update": True,
        "delete": True,
        "import_export": False,
        "versioning": False
    },
    Role.MANAGER: {
        "read": True,
        "create": False,
        "update": False,
        "delete": False,
        "import_export": False,
        "versioning": False
    },
    Role.VIEWER: {
        "read": True,
        "create": False,
        "update": False,
        "delete": False,
        "import_export": False,
        "versioning": False
    }
}

# Authentication and authorization
def hash_password(password: str) -> str:
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def generate_session_token() -> str:
    """Generate secure session token"""
    return secrets.token_urlsafe(32)

def log_security_event(user_id: str, action: str, resource: str, 
                      ip_address: str, user_agent: str, success: bool, details: str = None):
    """Log security events"""
    try:
        log_entry = SecurityLog(
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details
        )
        
        # Load existing logs
        logs = []
        if os.path.exists(SECURITY_LOG_FILE):
            try:
                with open(SECURITY_LOG_FILE, 'r', encoding='utf-8') as f:
                    logs = json.load(f)
            except:
                logs = []
        
        # Add new log (keep only last 1000 entries)
        logs.append(log_entry.dict(default=str))
        logs = logs[-1000:]  
        
        # Save logs
        with open(SECURITY_LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(logs, f, ensure_ascii=False, indent=2, default=str)
            
    except Exception as e:
        print(f"Failed to log security event: {e}")

async def get_current_user(request: Request, credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user"""
    token = credentials.credentials
    ip_address = request.client.host
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    # Check if token exists in active sessions
    if token not in active_sessions:
        log_security_event("unknown", "authentication", "token_validation", 
                         ip_address, user_agent, False, "Invalid token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication token",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    session = active_sessions[token]
    
    # Check if session is expired
    if datetime.now() > session.expires_at or not session.is_active:
        del active_sessions[token]
        log_security_event(session.user_id, "authentication", "session_expired", 
                         ip_address, user_agent, False, "Session expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Find user data
    user_data = None
    for username, data in USERS_DB.items():
        if data["id"] == session.user_id:
            user_data = data
            break
    
    if not user_data:
        log_security_event(session.user_id, "authentication", "user_not_found", 
                         ip_address, user_agent, False, "User not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create User object
    user = User(
        id=user_data["id"],
        username=user_data["username"],
        role=user_data["role"],
        email=user_data.get("email"),
        permissions=user_data["permissions"],
        session_token=token
    )
    
    # Log successful authentication
    log_security_event(user.id, "authentication", "success", 
                     ip_address, user_agent, True, "User authenticated")
    
    return user

def check_permission(user: User, action: str, resource: str = "taxonomy", 
                   require_confirmation: bool = False):
    """Check user permissions with optional confirmation requirement"""
    if not user.permissions.get(action, False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Insufficient permissions",
                "required_permission": action,
                "user_role": user.role,
                "user_permissions": user.permissions,
                "message": f"User '{user.username}' with role '{user.role}' does not have '{action}' permission"
            }
        )
    
    # For sensitive operations, require additional confirmation
    sensitive_actions = ["delete", "import_export", "user_management"]
    if action in sensitive_actions and require_confirmation:
        # This would be handled by frontend popup
        # For now, we'll include a flag in the response
        return {
            "permission_granted": True,
            "requires_confirmation": True,
            "action": action,
            "resource": resource,
            "warning": f"This action '{action}' on '{resource}' requires confirmation. Are you sure you want to proceed?"
        }
    
    return {"permission_granted": True, "requires_confirmation": False}

# Helper functions
def load_taxonomy() -> SkillTaxonomy:
    """Load taxonomy from file"""
    if os.path.exists(TAXONOMY_FILE):
        try:
            with open(TAXONOMY_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                return SkillTaxonomy(**data)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"Error loading taxonomy file: {e}")
            return create_default_taxonomy()
    else:
        return create_default_taxonomy()

def create_default_taxonomy() -> SkillTaxonomy:
    """Create default taxonomy structure"""
    return SkillTaxonomy(
        metadata={
            "title": "Employee Skill Taxonomy",
            "description": "Hierarchical skill structure",
            "created_date": datetime.now().isoformat(),
            "total_employees": 0,
            "total_skills": 0
        },
        nodes=[
            Node(
                id="root",
                name="Company",
                type=NodeType.ROOT,
                level=0,
                color="#808080"
            )
        ],
        edges=[],
        skill_owners={},
        color_scheme={
            "root": "#808080",
            "skill_group": "#ff7f0e",
            "skill": "#2ca02c",
            "sub_skill": "#d62728"
        },
        mermaid_export={
            "syntax": "graph TD",
            "example": "root[Company] --> skill_group_programming[Programming]"
        }
    )

def save_taxonomy(taxonomy: SkillTaxonomy):
    """Save taxonomy to file"""
    try:
        with open(TAXONOMY_FILE, 'w', encoding='utf-8') as f:
            json.dump(taxonomy.dict(), f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"Error saving taxonomy: {e}")
        raise HTTPException(status_code=500, detail="Failed to save taxonomy")

def find_node_by_id(taxonomy: SkillTaxonomy, node_id: str) -> Optional[Node]:
    """Find node by ID"""
    for node in taxonomy.nodes:
        if node.id == node_id:
            return node
    return None

def get_node_level(taxonomy: SkillTaxonomy, parent_id: str) -> int:
    """Get level for new node based on parent"""
    if parent_id:
        parent = find_node_by_id(taxonomy, parent_id)
        return parent.level + 1 if parent else 1
    return 1

def get_node_color(node_type: NodeType) -> str:
    """Get default color for node type"""
    color_map = {
        NodeType.ROOT: "#808080",
        NodeType.SKILL_GROUP: "#ff7f0e",
        NodeType.SKILL: "#2ca02c",
        NodeType.SUB_SKILL: "#d62728"
    }
    return color_map.get(node_type, "#808080")

# API Endpoints

# Authentication endpoints
@app.post("/auth/login", summary="User Login")
async def login(request: Request, auth_request: AuthRequest):
    """Authenticate user and create session"""
    ip_address = request.client.host
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    # Find user
    user_data = USERS_DB.get(auth_request.username)
    if not user_data:
        log_security_event("unknown", "login", "failed", ip_address, user_agent, 
                         False, f"Username '{auth_request.username}' not found")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Verify password
    password_hash = hash_password(auth_request.password)
    if password_hash != user_data["password_hash"]:
        log_security_event(user_data["id"], "login", "failed", ip_address, user_agent,
                         False, "Invalid password")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )
    
    # Create session
    session_token = generate_session_token()
    session = AuthSession(
        user_id=user_data["id"],
        token=session_token,
        created_at=datetime.now(),
        expires_at=datetime.now() + timedelta(hours=8),  # 8 hour session
        is_active=True
    )
    
    active_sessions[session_token] = session
    
    # Log successful login
    log_security_event(user_data["id"], "login", "success", ip_address, user_agent,
                     True, "User logged in successfully")
    
    return {
        "message": "Login successful",
        "token": session_token,
        "user": {
            "id": user_data["id"],
            "username": user_data["username"],
            "role": user_data["role"],
            "email": user_data.get("email"),
            "permissions": user_data["permissions"]
        },
        "expires_at": session.expires_at.isoformat()
    }

@app.post("/auth/logout", summary="User Logout")
async def logout(request: Request, current_user: User = Depends(get_current_user)):
    """Logout user and invalidate session"""
    ip_address = request.client.host
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    # Invalidate session
    if current_user.session_token in active_sessions:
        del active_sessions[current_user.session_token]
    
    log_security_event(current_user.id, "logout", "success", ip_address, user_agent,
                     True, "User logged out")
    
    return {"message": "Logout successful"}

@app.get("/auth/me", summary="Get Current User Info")
async def get_me(current_user: User = Depends(get_current_user)):
    """Get current user information"""
    return {
        "user": {
            "id": current_user.id,
            "username": current_user.username,
            "role": current_user.role,
            "email": current_user.email,
            "permissions": current_user.permissions
        },
        "session_info": {
            "token": current_user.session_token,
            "expires_at": active_sessions.get(current_user.session_token).expires_at.isoformat() if current_user.session_token in active_sessions else None
        }
    }

@app.post("/auth/check-permission", summary="Check Permission")
async def check_permission_endpoint(
    request: Request,
    permission_request: PermissionRequest,
    current_user: User = Depends(get_current_user)
):
    """Check if user has specific permission"""
    ip_address = request.client.host
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    try:
        result = check_permission(
            current_user, 
            permission_request.action, 
            permission_request.resource,
            require_confirmation=not permission_request.confirmation
        )
        
        log_security_event(current_user.id, "permission_check", permission_request.action,
                         ip_address, user_agent, True, 
                         f"Permission check for '{permission_request.action}' on '{permission_request.resource}'")
        
        return result
        
    except HTTPException as e:
        log_security_event(current_user.id, "permission_check", permission_request.action,
                         ip_address, user_agent, False, 
                         f"Permission denied for '{permission_request.action}' on '{permission_request.resource}'")
        raise e

@app.get("/auth/security-logs", summary="Get Security Logs (Admin Only)")
async def get_security_logs(
    limit: int = 100,
    current_user: User = Depends(get_current_user)
):
    """Get security audit logs (Admin only)"""
    if current_user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can access security logs"
        )
    
    try:
        if os.path.exists(SECURITY_LOG_FILE):
            with open(SECURITY_LOG_FILE, 'r', encoding='utf-8') as f:
                logs = json.load(f)
            return {"logs": logs[-limit:], "total_logs": len(logs)}
        else:
            return {"logs": [], "total_logs": 0}
    except Exception as e:
        return {"error": f"Failed to load security logs: {e}", "logs": [], "total_logs": 0}

@app.get("/auth/active-sessions", summary="Get Active Sessions (Admin Only)")
async def get_active_sessions(current_user: User = Depends(get_current_user)):
    """Get all active sessions (Admin only)"""
    if current_user.role != Role.ADMIN:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Only admin users can view active sessions"
        )
    
    sessions_info = []
    for token, session in active_sessions.items():
        user_data = None
        for username, data in USERS_DB.items():
            if data["id"] == session.user_id:
                user_data = data
                break
        
        sessions_info.append({
            "user_id": session.user_id,
            "username": user_data["username"] if user_data else "Unknown",
            "role": user_data["role"] if user_data else "Unknown",
            "created_at": session.created_at.isoformat(),
            "expires_at": session.expires_at.isoformat(),
            "is_active": session.is_active,
            "token_preview": token[:8] + "..."
        })
    
    return {"active_sessions": sessions_info, "total_sessions": len(sessions_info)}

@app.get("/auth/permissions-popup", response_class=HTMLResponse, summary="Permission Confirmation Popup")
async def permissions_popup():
    """HTML popup for permission confirmation"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Permission Confirmation</title>
        <meta charset="UTF-8">
        <style>
            body { font-family: Arial, sans-serif; padding: 20px; background-color: #f5f5f5; }
            .popup { background: white; border-radius: 8px; padding: 30px; max-width: 500px; margin: 0 auto; box-shadow: 0 4px 6px rgba(0,0,0,0.1); }
            .warning { color: #e74c3c; font-weight: bold; margin-bottom: 20px; }
            .action { background-color: #3498db; color: white; padding: 10px; border-radius: 4px; margin: 10px 0; }
            .buttons { text-align: center; margin-top: 30px; }
            button { padding: 12px 24px; margin: 0 10px; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }
            .confirm { background-color: #27ae60; color: white; }
            .cancel { background-color: #e74c3c; color: white; }
            .details { background-color: #ecf0f1; padding: 15px; border-radius: 4px; margin: 15px 0; }
        </style>
    </head>
    <body>
        <div class="popup">
            <h2>🔐 Permission Confirmation Required</h2>
            <div class="warning">⚠️ This action requires additional confirmation</div>
            
            <div class="details">
                <p><strong>Action:</strong> <span id="action">{{action}}</span></p>
                <p><strong>Resource:</strong> <span id="resource">{{resource}}</span></p>
                <p><strong>User:</strong> <span id="user">{{username}} ({{role}})</span></p>
            </div>
            
            <p>Are you sure you want to proceed with this action? This operation may have significant impact on the system.</p>
            
            <div class="buttons">
                <button class="confirm" onclick="confirmAction()">✅ Yes, I'm Sure</button>
                <button class="cancel" onclick="cancelAction()">❌ Cancel</button>
            </div>
        </div>
        
        <script>
            function confirmAction() {
                // Return confirmation to parent window
                if (window.opener) {
                    window.opener.postMessage({type: 'permission_confirmed', confirmed: true}, '*');
                }
                window.close();
            }
            
            function cancelAction() {
                // Return cancellation to parent window
                if (window.opener) {
                    window.opener.postMessage({type: 'permission_confirmed', confirmed: false}, '*');
                }
                window.close();
            }
            
            // Get URL parameters to populate form
            const urlParams = new URLSearchParams(window.location.search);
            document.getElementById('action').textContent = urlParams.get('action') || 'Unknown';
            document.getElementById('resource').textContent = urlParams.get('resource') || 'Unknown';
            document.getElementById('user').textContent = (urlParams.get('username') || 'Unknown') + ' (' + (urlParams.get('role') || 'Unknown') + ')';
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/health", summary="Health Check")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Skill Taxonomy Management API",
        "version": "1.0.0",
        "timestamp": datetime.now().isoformat()
    }

@app.get("/", summary="API Information")
async def root():
    return {
        "message": "Skill Taxonomy Management API",
        "version": "1.0.0",
        "status": "running",
        "server_url": "http://localhost:8000",
        "documentation": "http://localhost:8000/docs",
        "redoc": "http://localhost:8000/redoc",
        "endpoints": {
            "authentication": "/auth/login",
            "taxonomy": "/taxonomy",
            "nodes": "/nodes",
            "search": "/search",
            "stats": "/stats",
            "export": "/export",
            "import": "/import",
            "permission_popup": "/auth/permissions-popup"
        },
        "authentication": {
            "type": "Session-based with Bearer Token",
            "login_endpoint": "/auth/login",
            "demo_credentials": {
                "admin": {"username": "admin", "password": "admin123"},
                "hr_architect": {"username": "hr_architect", "password": "hr123"},
                "manager": {"username": "manager", "password": "manager123"},
                "viewer": {"username": "viewer", "password": "viewer123"}
            },
            "security_features": [
                "Password hashing",
                "Session management",
                "Security audit logging",
                "Permission confirmation popup",
                "Role-based access control"
            ]
        }
    }

@app.get("/taxonomy", response_model=SkillTaxonomy, summary="Get Complete Taxonomy")
async def get_taxonomy(request: Request, current_user: User = Depends(get_current_user)):
    """Get the complete skill taxonomy structure"""
    check_permission(current_user, "read")
    
    # Log access
    ip_address = request.client.host
    user_agent = request.headers.get("User-Agent", "Unknown")
    log_security_event(current_user.id, "read", "taxonomy", ip_address, user_agent, 
                     True, "Accessed complete taxonomy")
    
    return load_taxonomy()

@app.get("/nodes", response_model=List[Node], summary="Get All Nodes")
async def get_nodes(current_user: User = Depends(get_current_user)):
    """Get all nodes in the taxonomy"""
    check_permission(current_user, "read")
    taxonomy = load_taxonomy()
    return taxonomy.nodes

@app.get("/nodes/{node_id}", response_model=Node, summary="Get Node by ID")
async def get_node(node_id: str, current_user: User = Depends(get_current_user)):
    """Get a specific node by ID"""
    check_permission(current_user, "read")
    taxonomy = load_taxonomy()
    node = find_node_by_id(taxonomy, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    return node

@app.post("/nodes", response_model=Node, summary="Create New Node")
async def create_node(
    request: Request,
    node_request: CreateNodeRequest,
    current_user: User = Depends(get_current_user)
):
    """Create a new node in the taxonomy"""
    ip_address = request.client.host
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    # Check permission with confirmation for sensitive operations
    permission_result = check_permission(current_user, "create", "node")
    
    taxonomy = load_taxonomy()
    
    # Validate parent exists if specified
    if node_request.parent_id:
        parent = find_node_by_id(taxonomy, node_request.parent_id)
        if not parent:
            log_security_event(current_user.id, "create", "node", ip_address, user_agent,
                             False, f"Parent node '{node_request.parent_id}' not found")
            raise HTTPException(status_code=404, detail="Parent node not found")
    
    # Create new node
    node_id = f"skill_{str(uuid.uuid4()).replace('-', '_')}"
    level = get_node_level(taxonomy, node_request.parent_id)
    color = node_request.color or get_node_color(node_request.type)
    
    new_node = Node(
        id=node_id,
        name=node_request.name,
        type=node_request.type,
        level=level,
        color=color
    )
    
    taxonomy.nodes.append(new_node)
    
    # Create edge to parent if specified
    if node_request.parent_id:
        edge_id = f"edge_{len(taxonomy.edges)}"
        edge_type = EdgeType.CONTAINS if node_request.type == NodeType.SKILL_GROUP else EdgeType.INCLUDES
        
        new_edge = Edge(
            id=edge_id,
            source=node_request.parent_id,
            target=node_id,
            type=edge_type
        )
        taxonomy.edges.append(new_edge)
    
    # Update metadata
    taxonomy.metadata["total_skills"] = len([n for n in taxonomy.nodes if n.type in [NodeType.SKILL, NodeType.SUB_SKILL]])
    
    save_taxonomy(taxonomy)
    
    # Log successful creation
    log_security_event(current_user.id, "create", "node", ip_address, user_agent,
                     True, f"Created node '{new_node.name}' (ID: {node_id})")
    
    return new_node

@app.put("/nodes/{node_id}", response_model=Node, summary="Update Node")
async def update_node(
    node_id: str,
    request: UpdateNodeRequest,
    current_user: User = Depends(get_current_user)
):
    """Update an existing node"""
    check_permission(current_user, "update")
    
    taxonomy = load_taxonomy()
    node = find_node_by_id(taxonomy, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    # Update node properties
    if request.name is not None:
        node.name = request.name
    if request.color is not None:
        node.color = request.color
    
    save_taxonomy(taxonomy)
    return node

@app.put("/nodes/{node_id}/rename", response_model=Node, summary="Rename Node")
async def rename_node(
    node_id: str,
    request: RenameNodeRequest,
    current_user: User = Depends(get_current_user)
):
    """Rename a node"""
    check_permission(current_user, "update")
    
    taxonomy = load_taxonomy()
    node = find_node_by_id(taxonomy, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    node.name = request.new_name
    save_taxonomy(taxonomy)
    return node

@app.delete("/nodes/{node_id}", summary="Delete Node")
async def delete_node(
    node_id: str, 
    request: Request,
    confirmed: bool = False,
    current_user: User = Depends(get_current_user)
):
    """Delete a node and all its connections"""
    ip_address = request.client.host
    user_agent = request.headers.get("User-Agent", "Unknown")
    
    # Check permission with confirmation requirement for delete operations
    permission_result = check_permission(current_user, "delete", "node", require_confirmation=not confirmed)
    
    if permission_result.get("requires_confirmation") and not confirmed:
        return {
            "requires_confirmation": True,
            "confirmation_url": f"/auth/permissions-popup?action=delete&resource=node_{node_id}&username={current_user.username}&role={current_user.role}",
            "message": f"Deleting node '{node_id}' requires confirmation. Please confirm this action.",
            "retry_with_confirmation": f"/nodes/{node_id}?confirmed=true"
        }
    
    taxonomy = load_taxonomy()
    node = find_node_by_id(taxonomy, node_id)
    if not node:
        log_security_event(current_user.id, "delete", "node", ip_address, user_agent,
                         False, f"Node '{node_id}' not found")
        raise HTTPException(status_code=404, detail="Node not found")
    
    if node.type == NodeType.ROOT:
        log_security_event(current_user.id, "delete", "node", ip_address, user_agent,
                         False, f"Attempted to delete root node '{node_id}'")
        raise HTTPException(status_code=400, detail="Cannot delete root node")
    
    # Remove node
    taxonomy.nodes = [n for n in taxonomy.nodes if n.id != node_id]
    
    # Remove related edges
    taxonomy.edges = [e for e in taxonomy.edges if e.source != node_id and e.target != node_id]
    
    # Update metadata
    taxonomy.metadata["total_skills"] = len([n for n in taxonomy.nodes if n.type in [NodeType.SKILL, NodeType.SUB_SKILL]])
    
    save_taxonomy(taxonomy)
    
    # Log successful deletion
    log_security_event(current_user.id, "delete", "node", ip_address, user_agent,
                     True, f"Deleted node '{node.name}' (ID: {node_id})")
    
    return {"message": f"Node '{node_id}' deleted successfully", "deleted_node": node.name}

def find_path_to_root(taxonomy: SkillTaxonomy, node_id: str) -> List[str]:
    """Find path from root to specific node"""
    if node_id == "root":
        return ["root"]
    
    # Build parent mapping from edges
    parent_map = {}
    for edge in taxonomy.edges:
        parent_map[edge.target] = edge.source
    
    # Trace path back to root
    path = []
    current = node_id
    
    while current and current not in path:  # Prevent infinite loops
        path.append(current)
        current = parent_map.get(current)
        if current == "root":
            path.append("root")
            break
    
    # Reverse to get root-to-node path
    return list(reversed(path))

def get_path_visualization_data(taxonomy: SkillTaxonomy, path: List[str]) -> Dict[str, Any]:
    """Get visualization data for path highlighting"""
    path_nodes = []
    path_edges = []
    
    # Get nodes in path
    for node_id in path:
        node = find_node_by_id(taxonomy, node_id)
        if node:
            path_nodes.append({
                "id": node.id,
                "name": node.name,
                "type": node.type,
                "level": node.level,
                "color": node.color,
                "highlight": True
            })
    
    # Get edges in path
    for i in range(len(path) - 1):
        source_id = path[i]
        target_id = path[i + 1]
        
        # Find the edge
        for edge in taxonomy.edges:
            if edge.source == source_id and edge.target == target_id:
                path_edges.append({
                    "id": edge.id,
                    "source": edge.source,
                    "target": edge.target,
                    "type": edge.type,
                    "highlight": True
                })
                break
    
    return {
        "path": path,
        "path_nodes": path_nodes,
        "path_edges": path_edges,
        "path_length": len(path),
        "levels_traversed": len(set(node.get("level", 0) for node in path_nodes))
    }

@app.get("/search", response_model=List[Node], summary="Search Nodes")
async def search_nodes(
    query: str,
    current_user: User = Depends(get_current_user)
):
    """Search nodes by name"""
    check_permission(current_user, "read")
    
    taxonomy = load_taxonomy()
    matching_nodes = [
        node for node in taxonomy.nodes 
        if query.lower() in node.name.lower()
    ]
    return matching_nodes

@app.get("/search/with-path", summary="Search Nodes with Path to Root")
async def search_nodes_with_path(
    query: str,
    current_user: User = Depends(get_current_user)
):
    """Search nodes by name and return path from root to each found node"""
    check_permission(current_user, "read")
    
    taxonomy = load_taxonomy()
    matching_nodes = [
        node for node in taxonomy.nodes 
        if query.lower() in node.name.lower()
    ]
    
    results = []
    for node in matching_nodes:
        path = find_path_to_root(taxonomy, node.id)
        path_data = get_path_visualization_data(taxonomy, path)
        
        results.append({
            "node": node,
            "path_to_root": path,
            "path_data": path_data,
            "distance_from_root": len(path) - 1
        })
    
    # Sort by distance from root (closer nodes first)
    results.sort(key=lambda x: x["distance_from_root"])
    
    return {
        "query": query,
        "total_matches": len(results),
        "results": results
    }

@app.get("/nodes/{node_id}/path", summary="Get Path from Root to Node")
async def get_node_path(
    node_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get path from root to specific node with visualization data"""
    check_permission(current_user, "read")
    
    taxonomy = load_taxonomy()
    node = find_node_by_id(taxonomy, node_id)
    if not node:
        raise HTTPException(status_code=404, detail="Node not found")
    
    path = find_path_to_root(taxonomy, node_id)
    path_data = get_path_visualization_data(taxonomy, path)
    
    return {
        "node": node,
        "path_to_root": path,
        "path_data": path_data,
        "distance_from_root": len(path) - 1
    }

@app.get("/nodes/{node_id}/children", response_model=List[Node], summary="Get Child Nodes")
async def get_child_nodes(
    node_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get all child nodes of a specific node"""
    check_permission(current_user, "read")
    
    taxonomy = load_taxonomy()
    
    # Find child node IDs
    child_ids = [
        edge.target for edge in taxonomy.edges 
        if edge.source == node_id
    ]
    
    # Get child nodes
    child_nodes = [
        node for node in taxonomy.nodes 
        if node.id in child_ids
    ]
    
    return child_nodes

@app.get("/nodes/{node_id}/parent", response_model=Optional[Node], summary="Get Parent Node")
async def get_parent_node(
    node_id: str,
    current_user: User = Depends(get_current_user)
):
    """Get parent node of a specific node"""
    check_permission(current_user, "read")
    
    taxonomy = load_taxonomy()
    
    # Find parent node ID
    parent_edge = next((edge for edge in taxonomy.edges if edge.target == node_id), None)
    if not parent_edge:
        return None
    
    parent_node = find_node_by_id(taxonomy, parent_edge.source)
    return parent_node

@app.get("/export", summary="Export Taxonomy")
async def export_taxonomy(current_user: User = Depends(get_current_user)):
    """Export taxonomy data"""
    check_permission(current_user, "import_export")
    
    taxonomy = load_taxonomy()
    return taxonomy.dict()

@app.post("/import", summary="Import Taxonomy")
async def import_taxonomy(
    taxonomy_data: SkillTaxonomy,
    current_user: User = Depends(get_current_user)
):
    """Import taxonomy data"""
    check_permission(current_user, "import_export")
    
    save_taxonomy(taxonomy_data)
    return {"message": "Taxonomy imported successfully"}

@app.get("/stats", summary="Get Taxonomy Statistics")
async def get_stats(current_user: User = Depends(get_current_user)):
    """Get taxonomy statistics"""
    check_permission(current_user, "read")
    
    taxonomy = load_taxonomy()
    
    stats = {
        "total_nodes": len(taxonomy.nodes),
        "total_edges": len(taxonomy.edges),
        "node_types": {},
        "levels": {}
    }
    
    for node in taxonomy.nodes:
        stats["node_types"][node.type] = stats["node_types"].get(node.type, 0) + 1
        stats["levels"][f"level_{node.level}"] = stats["levels"].get(f"level_{node.level}", 0) + 1
    
    return stats

# Error handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return {
        "error": exc.detail,
        "status_code": exc.status_code
    }

if __name__ == "__main__":
    import uvicorn

    
    uvicorn.run(app, host="0.0.0.0", port=8002)