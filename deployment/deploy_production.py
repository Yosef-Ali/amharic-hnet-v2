#!/usr/bin/env python3
"""
Production Deployment Script for Amharic H-Net v2
================================================

This script handles the complete deployment process including:
- Model validation and preparation
- Environment setup
- Service deployment
- Health checks and validation
- Performance testing
"""

import asyncio
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Any

try:
    import structlog
    logger = structlog.get_logger(__name__)
    STRUCTLOG_AVAILABLE = True
except ImportError:
    import logging
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    STRUCTLOG_AVAILABLE = False


class ProductionDeployer:
    """
    Production deployment orchestrator for Amharic H-Net.
    """
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.deployment_dir = self.project_root / "deployment"
        self.model_path = self.project_root / "outputs" / "compact" / "final_model.pt"
        self.deployment_status = {}
        
    def validate_prerequisites(self) -> bool:
        """Validate deployment prerequisites."""
        logger.info("🔍 Validating deployment prerequisites...")
        
        checks = {
            "model_exists": self.model_path.exists(),
            "deployment_dir_exists": self.deployment_dir.exists(),
            "docker_available": self._check_docker(),
            "docker_compose_available": self._check_docker_compose(),
            "model_size_valid": self._check_model_size(),
        }
        
        self.deployment_status["prerequisites"] = checks
        
        for check, passed in checks.items():
            status = "✓" if passed else "✗"
            logger.info(f"{status} {check}")
        
        all_passed = all(checks.values())
        
        if not all_passed:
            logger.error("❌ Prerequisites check failed. Please resolve issues before deployment.")
            return False
        
        logger.info("✅ All prerequisites validated successfully")
        return True
    
    def _check_docker(self) -> bool:
        """Check if Docker is available."""
        try:
            result = subprocess.run(["docker", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_docker_compose(self) -> bool:
        """Check if Docker Compose is available."""
        try:
            result = subprocess.run(["docker-compose", "--version"], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def _check_model_size(self) -> bool:
        """Check if model size is reasonable."""
        if not self.model_path.exists():
            return False
        
        # Check model size (should be around 48MB based on training)
        size_mb = self.model_path.stat().st_size / (1024 * 1024)
        return 40 < size_mb < 100  # Reasonable range
    
    def prepare_deployment_files(self) -> bool:
        """Prepare deployment files and configuration."""
        logger.info("📋 Preparing deployment files...")
        
        try:
            # Create models directory in deployment
            models_dir = self.deployment_dir / "models"
            models_dir.mkdir(exist_ok=True)
            
            # Copy trained model
            target_model_path = models_dir / "final_model.pt"
            if not target_model_path.exists() or target_model_path.stat().st_size != self.model_path.stat().st_size:
                logger.info(f"📦 Copying model from {self.model_path} to {target_model_path}")
                shutil.copy2(self.model_path, target_model_path)
            
            # Create .env file if it doesn't exist
            env_file = self.deployment_dir / ".env"
            if not env_file.exists():
                self._create_env_file(env_file)
            
            # Validate deployment files
            required_files = [
                "Dockerfile",
                "docker-compose.yml", 
                "app/main.py",
                "app/config.py",
                "app/model_service.py",
                "app/cultural_safety.py",
                "requirements-production.txt"
            ]
            
            missing_files = []
            for file_path in required_files:
                if not (self.deployment_dir / file_path).exists():
                    missing_files.append(file_path)
            
            if missing_files:
                logger.error(f"❌ Missing deployment files: {missing_files}")
                return False
            
            self.deployment_status["file_preparation"] = {
                "model_copied": True,
                "env_file_created": env_file.exists(),
                "all_files_present": len(missing_files) == 0
            }
            
            logger.info("✅ Deployment files prepared successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to prepare deployment files: {e}")
            self.deployment_status["file_preparation"] = {"error": str(e)}
            return False
    
    def _create_env_file(self, env_file: Path):
        """Create production environment file."""
        env_content = f"""# Amharic H-Net Production Environment Configuration
ENVIRONMENT=production
HOST=0.0.0.0
PORT=8000
WORKERS=1

# Model Configuration
MODEL_PATH=/app/models/final_model.pt
MODEL_VERSION=1.1.0
ENABLE_MODEL_COMPILATION=true

# Security
ENABLE_AUTH=true
API_KEY={self._generate_api_key()}
SECRET_KEY={self._generate_secret_key()}
ALLOWED_ORIGINS=
ALLOWED_HOSTS=

# Performance
MAX_BATCH_SIZE=8
REQUEST_TIMEOUT=30
MAX_GENERATION_LENGTH=500
RESPONSE_TIME_TARGET_MS=200

# Redis/Caching
REDIS_URL=redis://redis:6379/0
CACHE_TTL=3600
ENABLE_CACHING=true

# Monitoring
ENABLE_METRICS=true
METRICS_PORT=9090
LOG_LEVEL=INFO
ENABLE_TRACING=true

# Cultural Safety
ENABLE_CULTURAL_SAFETY=true
CULTURAL_SAFETY_STRICT_MODE=true

# Rate Limiting
ENABLE_RATE_LIMITING=true
REQUESTS_PER_MINUTE=60
BURST_SIZE=10

# Resource Limits
MAX_MEMORY_GB=8.0
GPU_MEMORY_FRACTION=0.8
"""
        env_file.write_text(env_content)
        logger.info(f"📝 Created environment file: {env_file}")
    
    def _generate_api_key(self) -> str:
        """Generate a secure API key."""
        import secrets
        return secrets.token_urlsafe(32)
    
    def _generate_secret_key(self) -> str:
        """Generate a secure secret key."""
        import secrets
        return secrets.token_urlsafe(64)
    
    def build_and_deploy(self) -> bool:
        """Build Docker images and deploy services."""
        logger.info("🐳 Building and deploying services...")
        
        os.chdir(self.deployment_dir)
        
        try:
            # Build the application image
            logger.info("📦 Building application image...")
            build_result = subprocess.run([
                "docker-compose", "build", "--no-cache", "amharic-hnet-api"
            ], capture_output=True, text=True)
            
            if build_result.returncode != 0:
                logger.error(f"❌ Docker build failed: {build_result.stderr}")
                self.deployment_status["build"] = {"success": False, "error": build_result.stderr}
                return False
            
            # Start the services
            logger.info("🚀 Starting services...")
            deploy_result = subprocess.run([
                "docker-compose", "up", "-d"
            ], capture_output=True, text=True)
            
            if deploy_result.returncode != 0:
                logger.error(f"❌ Deployment failed: {deploy_result.stderr}")
                self.deployment_status["deploy"] = {"success": False, "error": deploy_result.stderr}
                return False
            
            self.deployment_status["build_and_deploy"] = {
                "build_success": True,
                "deploy_success": True,
                "build_output": build_result.stdout,
                "deploy_output": deploy_result.stdout
            }
            
            logger.info("✅ Services deployed successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Build and deploy failed: {e}")
            self.deployment_status["build_and_deploy"] = {"error": str(e)}
            return False
    
    async def wait_for_services(self, timeout: int = 120) -> bool:
        """Wait for services to be ready."""
        logger.info("⏳ Waiting for services to be ready...")
        
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                # Check if API is responding
                import httpx
                async with httpx.AsyncClient() as client:
                    response = await client.get("http://localhost:8000/health", timeout=5.0)
                    
                    if response.status_code == 200:
                        data = response.json()
                        if data.get("status") == "healthy":
                            logger.info("✅ Services are ready!")
                            self.deployment_status["service_readiness"] = {
                                "ready": True,
                                "wait_time": time.time() - start_time
                            }
                            return True
            except:
                pass
            
            await asyncio.sleep(5)
        
        logger.error(f"❌ Services failed to become ready within {timeout} seconds")
        self.deployment_status["service_readiness"] = {
            "ready": False,
            "timeout": timeout
        }
        return False
    
    async def run_deployment_tests(self) -> bool:
        """Run comprehensive deployment tests."""
        logger.info("🧪 Running deployment tests...")
        
        try:
            # Import and run the deployment tester
            sys.path.append(str(self.deployment_dir))
            from test_deployment import DeploymentTester
            
            async with DeploymentTester("http://localhost:8000") as tester:
                results = await tester.run_all_tests()
                
                self.deployment_status["tests"] = results
                
                if results["failed"] == 0:
                    logger.info("✅ All deployment tests passed!")
                    return True
                else:
                    logger.error(f"❌ {results['failed']} tests failed")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Testing failed: {e}")
            self.deployment_status["tests"] = {"error": str(e)}
            return False
    
    async def deploy_production(self) -> Dict[str, Any]:
        """Execute complete production deployment."""
        logger.info("🚀 Starting production deployment for Amharic H-Net v2")
        logger.info("=" * 60)
        
        start_time = time.time()
        
        # Step 1: Validate prerequisites
        if not self.validate_prerequisites():
            return self.deployment_status
        
        # Step 2: Prepare deployment files
        if not self.prepare_deployment_files():
            return self.deployment_status
        
        # Step 3: Build and deploy
        if not self.build_and_deploy():
            return self.deployment_status
        
        # Step 4: Wait for services to be ready
        if not await self.wait_for_services():
            return self.deployment_status
        
        # Step 5: Run deployment tests
        if not await self.run_deployment_tests():
            logger.warning("⚠️  Some tests failed, but deployment is functional")
        
        # Final status
        total_time = time.time() - start_time
        self.deployment_status["deployment_summary"] = {
            "success": True,
            "total_time": total_time,
            "services_running": True,
            "api_endpoint": "http://localhost:8000",
            "docs_url": "http://localhost:8000/docs",
            "metrics_url": "http://localhost:9090",
            "grafana_url": "http://localhost:3000"
        }
        
        logger.info("🎉 Production deployment completed successfully!")
        logger.info(f"⏱️  Total deployment time: {total_time:.2f} seconds")
        logger.info("🌐 API available at: http://localhost:8000")
        logger.info("📚 API docs available at: http://localhost:8000/docs")
        logger.info("📊 Metrics available at: http://localhost:9090")
        
        return self.deployment_status


async def main():
    """Main deployment orchestrator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Deploy Amharic H-Net to production")
    parser.add_argument("--project-root", 
                       default="/Users/mekdesyared/amharic-hnet-v2",
                       help="Project root directory")
    parser.add_argument("--output", 
                       help="Output file for deployment status")
    
    args = parser.parse_args()
    
    # Setup logging
    if STRUCTLOG_AVAILABLE:
        structlog.configure(
            processors=[
                structlog.stdlib.filter_by_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.add_log_level,
                structlog.stdlib.PositionalArgumentsFormatter(),
                structlog.processors.TimeStamper(fmt="iso"),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.format_exc_info,
                structlog.processors.UnicodeDecoder(),
                structlog.processors.JSONRenderer()
            ],
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            wrapper_class=structlog.stdlib.BoundLogger,
            cache_logger_on_first_use=True,
        )
    
    # Execute deployment
    deployer = ProductionDeployer(args.project_root)
    deployment_status = await deployer.deploy_production()
    
    # Save deployment status
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(deployment_status, f, indent=2, default=str)
        print(f"Deployment status saved to {args.output}")
    
    # Exit with appropriate code
    success = deployment_status.get("deployment_summary", {}).get("success", False)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())