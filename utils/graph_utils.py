#!/usr/bin/env python3
"""
🌳 Graph Processing Utilities
Các hàm tiện ích cho xử lý skill taxonomy graph
"""

import json
import uuid
from typing import Dict, List, Any, Optional, Tuple, Set
from collections import defaultdict, deque
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

def find_node_by_id(taxonomy: Dict[str, Any], node_id: str) -> Optional[Dict[str, Any]]:
    """
    Tìm node theo ID trong taxonomy
    """
    if not taxonomy or 'nodes' not in taxonomy:
        return None
    
    for node in taxonomy['nodes']:
        if node.get('id') == node_id:
            return node
    
    return None

def find_nodes_by_name(taxonomy: Dict[str, Any], name: str, exact_match: bool = False) -> List[Dict[str, Any]]:
    """
    Tìm nodes theo tên
    """
    if not taxonomy or 'nodes' not in taxonomy:
        return []
    
    results = []
    search_name = name.lower().strip()
    
    for node in taxonomy['nodes']:
        node_name = node.get('name', '').lower().strip()
        
        if exact_match:
            if node_name == search_name:
                results.append(node)
        else:
            if search_name in node_name:
                results.append(node)
    
    return results

def get_node_level(taxonomy: Dict[str, Any], parent_id: str = None) -> int:
    """
    Tính level của node dựa trên parent
    """
    if not parent_id:
        return 0  # Root level
    
    parent_node = find_node_by_id(taxonomy, parent_id)
    if parent_node:
        return parent_node.get('level', 0) + 1
    
    return 1  # Default level if parent not found

def get_node_color(node_type: str, color_scheme: Dict[str, str] = None) -> str:
    """
    Lấy màu cho node dựa trên type
    """
    default_colors = {
        "root": "#808080",
        "skill_group": "#ff7f0e", 
        "skill": "#2ca02c",
        "sub_skill": "#d62728"
    }
    
    if color_scheme:
        return color_scheme.get(node_type, default_colors.get(node_type, "#cccccc"))
    
    return default_colors.get(node_type, "#cccccc")

def get_children(taxonomy: Dict[str, Any], parent_id: str) -> List[Dict[str, Any]]:
    """
    Lấy tất cả children của một node
    """
    if not taxonomy or 'edges' not in taxonomy:
        return []
    
    child_ids = []
    
    # Find child node IDs from edges
    for edge in taxonomy['edges']:
        if edge.get('source') == parent_id:
            child_ids.append(edge.get('target'))
    
    # Get child nodes
    children = []
    for node in taxonomy.get('nodes', []):
        if node.get('id') in child_ids:
            children.append(node)
    
    return children

def get_parent(taxonomy: Dict[str, Any], child_id: str) -> Optional[Dict[str, Any]]:
    """
    Lấy parent của một node
    """
    if not taxonomy or 'edges' not in taxonomy:
        return None
    
    # Find parent ID from edges
    for edge in taxonomy['edges']:
        if edge.get('target') == child_id:
            parent_id = edge.get('source')
            return find_node_by_id(taxonomy, parent_id)
    
    return None

def get_node_path(taxonomy: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
    """
    Lấy đường dẫn từ root đến node
    """
    path = []
    current_node = find_node_by_id(taxonomy, node_id)
    
    if not current_node:
        return path
    
    # Build path from node to root
    path.append(current_node)
    
    while True:
        parent = get_parent(taxonomy, current_node['id'])
        if not parent:
            break
        path.insert(0, parent)  # Insert at beginning
        current_node = parent
    
    return path

def get_descendants(taxonomy: Dict[str, Any], node_id: str) -> List[Dict[str, Any]]:
    """
    Lấy tất cả descendants (con cháu) của một node
    """
    descendants = []
    queue = deque([node_id])
    visited = set()
    
    while queue:
        current_id = queue.popleft()
        if current_id in visited:
            continue
        
        visited.add(current_id)
        children = get_children(taxonomy, current_id)
        
        for child in children:
            child_id = child['id']
            descendants.append(child)
            queue.append(child_id)
    
    return descendants

def calculate_node_depth(taxonomy: Dict[str, Any], node_id: str) -> int:
    """
    Tính độ sâu của node từ root
    """
    path = get_node_path(taxonomy, node_id)
    return len(path) - 1 if path else 0

def calculate_subtree_size(taxonomy: Dict[str, Any], node_id: str) -> int:
    """
    Tính số lượng nodes trong subtree
    """
    descendants = get_descendants(taxonomy, node_id)
    return len(descendants) + 1  # +1 for the node itself

def validate_graph_structure(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate cấu trúc graph và trả về report
    """
    issues = []
    warnings = []
    
    if not isinstance(taxonomy, dict):
        issues.append("Taxonomy must be a dictionary")
        return {"valid": False, "issues": issues, "warnings": warnings}
    
    # Check required fields
    required_fields = ['nodes', 'edges', 'metadata']
    for field in required_fields:
        if field not in taxonomy:
            issues.append(f"Missing required field: {field}")
    
    if issues:
        return {"valid": False, "issues": issues, "warnings": warnings}
    
    nodes = taxonomy['nodes']
    edges = taxonomy['edges']
    
    # Validate nodes
    node_ids = set()
    for i, node in enumerate(nodes):
        if not isinstance(node, dict):
            issues.append(f"Node {i} is not a dictionary")
            continue
        
        node_id = node.get('id')
        if not node_id:
            issues.append(f"Node {i} missing ID")
            continue
        
        if node_id in node_ids:
            issues.append(f"Duplicate node ID: {node_id}")
        node_ids.add(node_id)
        
        # Check required node fields
        required_node_fields = ['name', 'type', 'level']
        for field in required_node_fields:
            if field not in node:
                warnings.append(f"Node {node_id} missing field: {field}")
    
    # Validate edges
    for i, edge in enumerate(edges):
        if not isinstance(edge, dict):
            issues.append(f"Edge {i} is not a dictionary")
            continue
        
        source = edge.get('source')
        target = edge.get('target')
        
        if not source or not target:
            issues.append(f"Edge {i} missing source or target")
            continue
        
        if source not in node_ids:
            issues.append(f"Edge {i} references non-existent source node: {source}")
        
        if target not in node_ids:
            issues.append(f"Edge {i} references non-existent target node: {target}")
        
        if source == target:
            issues.append(f"Edge {i} has self-reference: {source}")
    
    # Check for cycles
    if has_cycles(taxonomy):
        issues.append("Graph contains cycles")
    
    # Check connectivity
    if not is_connected(taxonomy):
        warnings.append("Graph is not fully connected")
    
    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "node_count": len(nodes),
        "edge_count": len(edges)
    }

def has_cycles(taxonomy: Dict[str, Any]) -> bool:
    """
    Kiểm tra xem graph có cycle không
    """
    if not taxonomy or 'nodes' not in taxonomy or 'edges' not in taxonomy:
        return False
    
    # Build adjacency list
    adj_list = defaultdict(list)
    for edge in taxonomy['edges']:
        source = edge.get('source')
        target = edge.get('target')
        if source and target:
            adj_list[source].append(target)
    
    visited = set()
    rec_stack = set()
    
    def dfs(node):
        visited.add(node)
        rec_stack.add(node)
        
        for neighbor in adj_list[node]:
            if neighbor not in visited:
                if dfs(neighbor):
                    return True
            elif neighbor in rec_stack:
                return True
        
        rec_stack.remove(node)
        return False
    
    # Check all nodes
    for node in taxonomy['nodes']:
        node_id = node.get('id')
        if node_id not in visited:
            if dfs(node_id):
                return True
    
    return False

def is_connected(taxonomy: Dict[str, Any]) -> bool:
    """
    Kiểm tra xem graph có connected không
    """
    if not taxonomy or 'nodes' not in taxonomy or 'edges' not in taxonomy:
        return False
    
    nodes = taxonomy['nodes']
    edges = taxonomy['edges']
    
    if not nodes:
        return True
    
    # Build undirected adjacency list
    adj_list = defaultdict(set)
    for edge in edges:
        source = edge.get('source')
        target = edge.get('target')
        if source and target:
            adj_list[source].add(target)
            adj_list[target].add(source)
    
    # BFS from first node
    start_node = nodes[0]['id']
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        current = queue.popleft()
        for neighbor in adj_list[current]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append(neighbor)
    
    # Check if all nodes were visited
    all_node_ids = {node['id'] for node in nodes}
    return len(visited) == len(all_node_ids)

def calculate_graph_metrics(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Tính các metrics của graph
    """
    if not taxonomy or 'nodes' not in taxonomy or 'edges' not in taxonomy:
        return {}
    
    nodes = taxonomy['nodes']
    edges = taxonomy['edges']
    
    metrics = {
        'node_count': len(nodes),
        'edge_count': len(edges),
        'density': 0,
        'max_depth': 0,
        'avg_degree': 0,
        'node_types': defaultdict(int),
        'levels': defaultdict(int)
    }
    
    if not nodes:
        return metrics
    
    # Calculate density
    n = len(nodes)
    if n > 1:
        max_edges = n * (n - 1) / 2  # For undirected graph
        metrics['density'] = len(edges) / max_edges
    
    # Build degree map
    degree_map = defaultdict(int)
    for edge in edges:
        degree_map[edge.get('source', '')] += 1
        degree_map[edge.get('target', '')] += 1
    
    # Calculate average degree
    if degree_map:
        metrics['avg_degree'] = sum(degree_map.values()) / len(degree_map)
    
    # Analyze nodes
    for node in nodes:
        node_type = node.get('type', 'unknown')
        level = node.get('level', 0)
        
        metrics['node_types'][node_type] += 1
        metrics['levels'][level] += 1
        
        if level > metrics['max_depth']:
            metrics['max_depth'] = level
    
    return metrics

def export_to_mermaid(taxonomy: Dict[str, Any]) -> str:
    """
    Export graph sang Mermaid syntax
    """
    if not taxonomy or 'nodes' not in taxonomy or 'edges' not in taxonomy:
        return ""
    
    mermaid_lines = ["graph TD"]
    
    # Add nodes with styling
    for node in taxonomy['nodes']:
        node_id = node.get('id', '')
        node_name = node.get('name', '')
        node_type = node.get('type', '')
        
        # Clean node name for Mermaid
        clean_name = node_name.replace('"', "'").replace('[', '(').replace(']', ')')
        
        # Different shapes for different node types
        if node_type == 'root':
            line = f'  {node_id}["{clean_name}"]'
        elif node_type == 'skill_group':
            line = f'  {node_id}("{clean_name}")'
        elif node_type == 'skill':
            line = f'  {node_id}["{clean_name}"]'
        else:  # sub_skill
            line = f'  {node_id}("{clean_name}")'
        
        mermaid_lines.append(line)
    
    # Add edges
    for edge in taxonomy['edges']:
        source = edge.get('source', '')
        target = edge.get('target', '')
        if source and target:
            mermaid_lines.append(f'  {source} --> {target}')
    
    # Add styling
    color_scheme = taxonomy.get('color_scheme', {})
    for node_type, color in color_scheme.items():
        if color.startswith('#'):
            color = color[1:]  # Remove # for Mermaid
        mermaid_lines.append(f'  classDef {node_type} fill:#{color}')
    
    return '\n'.join(mermaid_lines)

def export_to_cytoscape(taxonomy: Dict[str, Any]) -> Dict[str, Any]:
    """
    Export graph cho Cytoscape.js format
    """
    if not taxonomy or 'nodes' not in taxonomy or 'edges' not in taxonomy:
        return {"elements": {"nodes": [], "edges": []}}
    
    cyto_nodes = []
    cyto_edges = []
    
    # Convert nodes
    for node in taxonomy['nodes']:
        cyto_node = {
            "data": {
                "id": node.get('id', ''),
                "label": node.get('name', ''),
                "type": node.get('type', ''),
                "level": node.get('level', 0),
                "employee_count": node.get('employee_count', 0)
            },
            "style": {
                "background-color": node.get('color', '#cccccc')
            }
        }
        cyto_nodes.append(cyto_node)
    
    # Convert edges
    for edge in taxonomy['edges']:
        cyto_edge = {
            "data": {
                "id": edge.get('id', ''),
                "source": edge.get('source', ''),
                "target": edge.get('target', ''),
                "type": edge.get('type', 'includes')
            }
        }
        cyto_edges.append(cyto_edge)
    
    return {
        "elements": {
            "nodes": cyto_nodes,
            "edges": cyto_edges
        }
    }

def generate_node_id(prefix: str = "skill") -> str:
    """
    Generate unique node ID
    """
    return f"{prefix}_{str(uuid.uuid4()).replace('-', '_')}"

def generate_edge_id(source: str, target: str) -> str:
    """
    Generate edge ID
    """
    return f"edge_{source}_{target}"

def merge_taxonomies(taxonomy1: Dict[str, Any], taxonomy2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge hai taxonomies
    """
    if not taxonomy1:
        return taxonomy2
    if not taxonomy2:
        return taxonomy1
    
    merged = {
        "metadata": {
            **taxonomy1.get('metadata', {}),
            **taxonomy2.get('metadata', {}),
            "merged_at": str(datetime.now())
        },
        "nodes": [],
        "edges": [],
        "skill_owners": {
            **taxonomy1.get('skill_owners', {}),
            **taxonomy2.get('skill_owners', {})
        },
        "color_scheme": {
            **taxonomy1.get('color_scheme', {}),
            **taxonomy2.get('color_scheme', {})
        },
        "mermaid_export": taxonomy1.get('mermaid_export', {})
    }
    
    # Merge nodes (avoid duplicates by ID)
    node_ids = set()
    for taxonomy in [taxonomy1, taxonomy2]:
        for node in taxonomy.get('nodes', []):
            node_id = node.get('id')
            if node_id not in node_ids:
                merged['nodes'].append(node)
                node_ids.add(node_id)
    
    # Merge edges (avoid duplicates)
    edge_pairs = set()
    for taxonomy in [taxonomy1, taxonomy2]:
        for edge in taxonomy.get('edges', []):
            source = edge.get('source')
            target = edge.get('target')
            edge_pair = (source, target)
            if edge_pair not in edge_pairs:
                merged['edges'].append(edge)
                edge_pairs.add(edge_pair)
    
    return merged

def create_node(name: str, node_type: str, parent_id: str = None, 
               level: int = None, color: str = None) -> Dict[str, Any]:
    """
    Tạo node mới
    """
    node_id = generate_node_id()
    
    if level is None:
        level = 1 if parent_id else 0
    
    if color is None:
        color = get_node_color(node_type)
    
    return {
        "id": node_id,
        "name": name,
        "type": node_type,
        "level": level,
        "color": color,
        "employees": [],
        "employee_count": 0,
        "proficiency_stats": {}
    }

def create_edge(source: str, target: str, edge_type: str = "includes") -> Dict[str, Any]:
    """
    Tạo edge mới
    """
    edge_id = generate_edge_id(source, target)
    
    return {
        "id": edge_id,
        "source": source,
        "target": target,
        "type": edge_type,
        "weight": 1.0
    }
