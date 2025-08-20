#!/usr/bin/env python3
"""
🚀 SkillStruct Platform Manager
Advanced startup script với RAG system integration
"""

import os
import sys
import subprocess
import time
import signal
import logging
from pathlib import Path
from typing import List, Dict, Any
import requests
import threading
import argparse

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config import SERVICES, STREAMLIT_PORT, STREAMLIT_FILE, RAG_SETTINGS

logger = logging.getLogger(__name__)

class ServiceManager:
    """
    Advanced service manager cho SkillStruct platform
    """
    
    def __init__(self):
        self.processes: Dict[str, subprocess.Popen] = {}
        self.service_status: Dict[str, str] = {}
        self.running = True
        
    def start_service(self, service_name: str, service_config: Dict[str, Any]) -> bool:
        """
        Start a service
        
        Args:
            service_name: Name of the service
            service_config: Service configuration
            
        Returns:
            Success status
        """
        try:
            port = service_config["port"]
            file_path = service_config["file"]
            
            # Check if port is already in use
            if self._is_port_in_use(port):
                logger.warning(f"Port {port} already in use for {service_name}")
                return False
            
            # Start service
            cmd = [sys.executable, file_path]
            
            logger.info(f"🚀 Starting {service_name} on port {port}...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=project_root,
                env=os.environ.copy()
            )
            
            # Wait a moment to see if process starts successfully
            time.sleep(2)
            
            if process.poll() is None:
                self.processes[service_name] = process
                self.service_status[service_name] = "running"
                logger.info(f"✅ {service_name} started successfully on port {port}")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error(f"❌ Failed to start {service_name}")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting {service_name}: {e}")
            return False
    
    def start_rag_service(self) -> bool:
        """
        Start RAG service with special handling
        
        Returns:
            Success status
        """
        try:
            if not RAG_SETTINGS.get("enabled", False):
                logger.info("📝 RAG system is disabled in configuration")
                return True
            
            # Check requirements
            if not RAG_SETTINGS.get("google_api_key"):
                logger.warning("⚠️ Google API key not configured for RAG")
                return False
            
            rag_config = SERVICES.get("rag_api")
            if not rag_config:
                logger.error("❌ RAG API configuration not found")
                return False
            
            # Set environment variables for RAG
            os.environ.update({
                "MILVUS_HOST": RAG_SETTINGS["milvus_host"],
                "MILVUS_PORT": str(RAG_SETTINGS["milvus_port"]),
                "GOOGLE_API_KEY": RAG_SETTINGS["google_api_key"],
                "RAG_COLLECTION_NAME": RAG_SETTINGS["collection_name"]
            })
            
            return self.start_service("rag_api", rag_config)
            
        except Exception as e:
            logger.error(f"❌ Error starting RAG service: {e}")
            return False
    
    def start_streamlit(self) -> bool:
        """
        Start Streamlit application
        
        Returns:
            Success status
        """
        try:
            if self._is_port_in_use(STREAMLIT_PORT):
                logger.warning(f"Port {STREAMLIT_PORT} already in use for Streamlit")
                return False
            
            cmd = [
                sys.executable, "-m", "streamlit", "run",
                STREAMLIT_FILE,
                "--server.port", str(STREAMLIT_PORT),
                "--server.address", "0.0.0.0",
                "--server.headless", "true"
            ]
            
            logger.info(f"🚀 Starting Streamlit on port {STREAMLIT_PORT}...")
            
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=project_root,
                env=os.environ.copy()
            )
            
            time.sleep(3)
            
            if process.poll() is None:
                self.processes["streamlit"] = process
                self.service_status["streamlit"] = "running"
                logger.info(f"✅ Streamlit started successfully on port {STREAMLIT_PORT}")
                return True
            else:
                stdout, stderr = process.communicate()
                logger.error("❌ Failed to start Streamlit")
                logger.error(f"STDOUT: {stdout.decode()}")
                logger.error(f"STDERR: {stderr.decode()}")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error starting Streamlit: {e}")
            return False
    
    def _is_port_in_use(self, port: int) -> bool:
        """Check if port is in use"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def wait_for_service(self, service_name: str, port: int, timeout: int = 30) -> bool:
        """
        Wait for service to become ready
        
        Args:
            service_name: Name of the service
            port: Port number
            timeout: Timeout in seconds
            
        Returns:
            Success status
        """
        logger.info(f"⏳ Waiting for {service_name} to be ready...")
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                # Try health check endpoint first
                response = requests.get(f"http://localhost:{port}/health", timeout=2)
                if response.status_code == 200:
                    logger.info(f"✅ {service_name} is ready!")
                    return True
            except:
                # Try basic connection
                try:
                    response = requests.get(f"http://localhost:{port}", timeout=2)
                    if response.status_code in [200, 404]:  # 404 is OK for root endpoint
                        logger.info(f"✅ {service_name} is ready!")
                        return True
                except:
                    pass
            
            time.sleep(1)
        
        logger.warning(f"⚠️ {service_name} not ready after {timeout}s")
        return False
    
    def check_service_health(self, service_name: str, port: int) -> bool:
        """Check service health"""
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def stop_service(self, service_name: str) -> bool:
        """Stop a service"""
        if service_name in self.processes:
            try:
                process = self.processes[service_name]
                process.terminate()
                
                # Wait for graceful shutdown
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                
                del self.processes[service_name]
                self.service_status[service_name] = "stopped"
                logger.info(f"🛑 {service_name} stopped")
                return True
                
            except Exception as e:
                logger.error(f"❌ Error stopping {service_name}: {e}")
                return False
        return True
    
    def stop_all_services(self):
        """Stop all services"""
        logger.info("🛑 Stopping all services...")
        
        for service_name in list(self.processes.keys()):
            self.stop_service(service_name)
        
        self.running = False
        logger.info("✅ All services stopped")
    
    def get_service_status(self) -> Dict[str, Any]:
        """Get status of all services"""
        status = {}
        
        # Check API services
        for service_name, config in SERVICES.items():
            port = config["port"]
            is_healthy = self.check_service_health(service_name, port)
            
            status[service_name] = {
                "name": config["name"],
                "port": port,
                "status": "healthy" if is_healthy else "unhealthy",
                "url": f"http://localhost:{port}"
            }
        
        # Check Streamlit
        streamlit_healthy = self.check_service_health("streamlit", STREAMLIT_PORT)
        status["streamlit"] = {
            "name": "Streamlit Frontend",
            "port": STREAMLIT_PORT,
            "status": "healthy" if streamlit_healthy else "unhealthy",
            "url": f"http://localhost:{STREAMLIT_PORT}"
        }
        
        return status
    
    def print_service_status(self):
        """Print service status table"""
        status = self.get_service_status()
        
        print("\n" + "="*80)
        print("🚀 SKILLSTRUCT PLATFORM STATUS")
        print("="*80)
        print(f"{'Service':<20} {'Status':<10} {'Port':<6} {'URL'}")
        print("-"*80)
        
        for service_name, info in status.items():
            status_emoji = "✅" if info["status"] == "healthy" else "❌"
            print(f"{info['name']:<20} {status_emoji} {info['status']:<6} {info['port']:<6} {info['url']}")
        
        print("-"*80)
        print(f"Total Services: {len(status)}")
        healthy_count = sum(1 for s in status.values() if s["status"] == "healthy")
        print(f"Healthy: {healthy_count}/{len(status)}")
        print("="*80)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="SkillStruct Platform Manager")
    parser.add_argument("--services", nargs="+", help="Specific services to start")
    parser.add_argument("--no-streamlit", action="store_true", help="Don't start Streamlit")
    parser.add_argument("--no-rag", action="store_true", help="Don't start RAG service")
    parser.add_argument("--status", action="store_true", help="Show service status only")
    parser.add_argument("--stop", action="store_true", help="Stop all services")
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    manager = ServiceManager()
    
    if args.status:
        manager.print_service_status()
        return
    
    if args.stop:
        # Find and stop existing services
        for service_name, config in SERVICES.items():
            port = config["port"]
            if manager._is_port_in_use(port):
                logger.info(f"🛑 Stopping service on port {port}")
        
        if manager._is_port_in_use(STREAMLIT_PORT):
            logger.info(f"🛑 Stopping Streamlit on port {STREAMLIT_PORT}")
        
        return
    
    def signal_handler(signum, frame):
        """Handle shutdown signals"""
        logger.info("\n📡 Shutdown signal received")
        manager.stop_all_services()
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        print("\n🚀 STARTING SKILLSTRUCT PLATFORM")
        print("="*50)
        
        success_count = 0
        total_services = 0
        
        # Determine which services to start
        services_to_start = args.services if args.services else list(SERVICES.keys())
        
        # Start API services
        for service_name in services_to_start:
            if service_name in SERVICES:
                config = SERVICES[service_name]
                total_services += 1
                
                if service_name == "rag_api" and args.no_rag:
                    logger.info(f"⏭️ Skipping RAG service (--no-rag)")
                    continue
                
                if service_name == "rag_api":
                    if manager.start_rag_service():
                        success_count += 1
                        # Wait for RAG service to be ready
                        manager.wait_for_service(service_name, config["port"])
                else:
                    if manager.start_service(service_name, config):
                        success_count += 1
                        # Wait for service to be ready
                        manager.wait_for_service(service_name, config["port"])
        
        # Start Streamlit
        if not args.no_streamlit:
            total_services += 1
            if manager.start_streamlit():
                success_count += 1
                manager.wait_for_service("streamlit", STREAMLIT_PORT)
        
        print(f"\n📊 STARTUP SUMMARY")
        print(f"✅ {success_count}/{total_services} services started successfully")
        
        # Print service status
        time.sleep(2)
        manager.print_service_status()
        
        if success_count > 0:
            print(f"\n🎉 Platform ready! Access points:")
            
            # Show healthy services
            status = manager.get_service_status()
            for service_name, info in status.items():
                if info["status"] == "healthy":
                    print(f"  • {info['name']}: {info['url']}")
            
            print(f"\n🔍 Monitoring services... Press Ctrl+C to stop")
            
            # Keep main thread alive
            while manager.running:
                time.sleep(1)
        else:
            logger.error("❌ No services started successfully")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("\n📡 Keyboard interrupt received")
        manager.stop_all_services()
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        manager.stop_all_services()
        sys.exit(1)

if __name__ == "__main__":
    main()
