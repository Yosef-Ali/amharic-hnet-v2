#!/usr/bin/env python3
"""
Deployment Monitor for Amharic H-Net Production Pipeline
Monitors training progress and automates submission creation
"""

import os
import sys
import time
import subprocess
import json
from pathlib import Path
from typing import Dict, Any, Optional
import psutil

class DeploymentMonitor:
    """
    Monitors the entire deployment pipeline from training to submission.
    """
    
    def __init__(self):
        self.start_time = time.time()
        self.training_process = None
        self.status = {
            'training_started': False,
            'training_completed': False,
            'model_saved': False,
            'submission_created': False,
            'kaggle_submitted': False
        }
        
        print("🔍 Deployment Monitor initialized")
        print(f"📅 Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    def check_training_status(self) -> Dict[str, Any]:
        """Check if training is running and its progress."""
        training_info = {
            'is_running': False,
            'progress': 0,
            'current_epoch': 0,
            'total_epochs': 20,
            'latest_loss': None,
            'estimated_completion': None
        }
        
        # Check for training process
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                if 'claude_gpu_training.py' in ' '.join(proc.info['cmdline'] or []):
                    training_info['is_running'] = True
                    self.status['training_started'] = True
                    break
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        
        # Check for training logs/outputs
        if os.path.exists('training.log'):
            training_info.update(self._parse_training_log())
        
        # Check for saved model
        model_files = [
            'production_model_optimized.pt',
            'model_checkpoint_latest.pt',
            'trained_model.pt'
        ]
        
        for model_file in model_files:
            if os.path.exists(model_file):
                self.status['model_saved'] = True
                if not training_info['is_running']:
                    self.status['training_completed'] = True
                break
        
        return training_info
    
    def _parse_training_log(self) -> Dict[str, Any]:
        """Parse training log for progress information."""
        info = {}
        
        try:
            with open('training.log', 'r') as f:
                lines = f.readlines()
            
            # Look for epoch and loss information
            for line in reversed(lines[-50:]):  # Check last 50 lines
                if 'Epoch' in line and 'Loss=' in line:
                    # Extract epoch and loss
                    parts = line.split()
                    for i, part in enumerate(parts):
                        if part.startswith('Epoch'):
                            try:
                                epoch_info = parts[i+1].split('/')
                                info['current_epoch'] = int(epoch_info[0].rstrip(','))
                                info['total_epochs'] = int(epoch_info[1].rstrip(','))
                                info['progress'] = (info['current_epoch'] / info['total_epochs']) * 100
                            except:
                                pass
                        elif part.startswith('Loss='):
                            try:
                                info['latest_loss'] = float(part.split('=')[1])
                            except:
                                pass
                    break
        
        except Exception as e:
            print(f"⚠️  Log parsing error: {e}")
        
        return info
    
    def check_system_resources(self) -> Dict[str, Any]:
        """Monitor system resources during training."""
        return {
            'cpu_percent': psutil.cpu_percent(interval=1),
            'memory_percent': psutil.virtual_memory().percent,
            'disk_usage': psutil.disk_usage('.').percent,
            'gpu_available': os.system('nvidia-smi > /dev/null 2>&1') == 0
        }
    
    def auto_create_submission(self) -> bool:
        """Automatically create Kaggle submission when training completes."""
        if not self.status['training_completed'] or self.status['submission_created']:
            return False
        
        print("🚀 Training completed! Creating Kaggle submission...")
        
        try:
            # Run submission creation script
            result = subprocess.run(
                [sys.executable, 'create_kaggle_submission.py'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                print("✅ Submission created successfully!")
                print(result.stdout)
                self.status['submission_created'] = True
                return True
            else:
                print(f"❌ Submission creation failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("⏰ Submission creation timed out")
            return False
        except Exception as e:
            print(f"❌ Submission creation error: {e}")
            return False
    
    def generate_deployment_report(self) -> Dict[str, Any]:
        """Generate comprehensive deployment report."""
        elapsed_time = time.time() - self.start_time
        
        report = {
            'deployment_status': self.status,
            'elapsed_time_minutes': elapsed_time / 60,
            'training_info': self.check_training_status(),
            'system_resources': self.check_system_resources(),
            'files_created': self._list_created_files(),
            'next_steps': self._get_next_steps()
        }
        
        return report
    
    def _list_created_files(self) -> Dict[str, bool]:
        """List all expected deployment files and their status."""
        expected_files = {
            'production_config.yaml': False,
            'claude_gpu_training.py': False,
            'production_inference.py': False,
            'requirements_production.txt': False,
            'create_kaggle_submission.py': False,
            'kaggle_credentials.json': False,
            'setup_kaggle_api.py': False,
            'production_model_optimized.pt': False,
            'submission.csv': False
        }
        
        for filename in expected_files:
            expected_files[filename] = os.path.exists(filename)
        
        return expected_files
    
    def _get_next_steps(self) -> list:
        """Determine next steps based on current status."""
        steps = []
        
        if not self.status['training_started']:
            steps.append("Start training with: python claude_gpu_training.py")
        elif not self.status['training_completed']:
            steps.append("Wait for training to complete")
        elif not self.status['submission_created']:
            steps.append("Create submission with: python create_kaggle_submission.py")
        elif not self.status['kaggle_submitted']:
            steps.append("Upload submission.csv to Kaggle competition")
        else:
            steps.append("Monitor Kaggle leaderboard for results")
        
        return steps
    
    def run_monitoring_loop(self, check_interval: int = 30, max_duration: int = 7200):
        """Run continuous monitoring loop."""
        print(f"🔄 Starting monitoring loop (check every {check_interval}s, max {max_duration/60:.0f} minutes)")
        
        loop_start = time.time()
        
        while time.time() - loop_start < max_duration:
            print(f"\n⏰ {time.strftime('%H:%M:%S')} - Checking deployment status...")
            
            # Check training status
            training_info = self.check_training_status()
            
            if training_info['is_running']:
                print(f"🏃 Training in progress: Epoch {training_info.get('current_epoch', '?')}/{training_info.get('total_epochs', '?')}")
                if training_info.get('latest_loss'):
                    print(f"📉 Latest loss: {training_info['latest_loss']:.4f}")
                print(f"📊 Progress: {training_info.get('progress', 0):.1f}%")
            elif self.status['training_completed']:
                print("✅ Training completed!")
                
                # Auto-create submission
                if self.auto_create_submission():
                    print("🎉 Deployment pipeline completed successfully!")
                    break
            else:
                print("⏳ Waiting for training to start...")
            
            # Check system resources
            resources = self.check_system_resources()
            print(f"💻 CPU: {resources['cpu_percent']:.1f}%, RAM: {resources['memory_percent']:.1f}%")
            
            # Wait for next check
            time.sleep(check_interval)
        
        # Final report
        print("\n📋 Final Deployment Report:")
        print("=" * 40)
        report = self.generate_deployment_report()
        
        for key, value in report['deployment_status'].items():
            status_icon = "✅" if value else "❌"
            print(f"{status_icon} {key.replace('_', ' ').title()}: {value}")
        
        print(f"\n⏱️  Total time: {report['elapsed_time_minutes']:.1f} minutes")
        
        if report['next_steps']:
            print("\n📝 Next steps:")
            for i, step in enumerate(report['next_steps'], 1):
                print(f"{i}. {step}")
        
        return report

def main():
    """Main monitoring function."""
    print("🔍 Amharic H-Net Deployment Monitor")
    print("=" * 40)
    
    monitor = DeploymentMonitor()
    
    # Run monitoring loop
    try:
        report = monitor.run_monitoring_loop(
            check_interval=30,  # Check every 30 seconds
            max_duration=7200   # Run for max 2 hours
        )
        
        # Save report
        with open('deployment_report.json', 'w') as f:
            json.dump(report, f, indent=2)
        
        print("\n📄 Deployment report saved to: deployment_report.json")
        
    except KeyboardInterrupt:
        print("\n⏹️  Monitoring stopped by user")
    except Exception as e:
        print(f"\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    main()